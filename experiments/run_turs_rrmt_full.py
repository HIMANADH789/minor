"""TURS-RRMT full experiment pipeline.

Stages: audit -> per-dataset (train A0..A5 canonical protocol -> evaluate ->
routing extraction -> D1-D9 diagnostics on frozen A4 -> A6/A7 interventions)
-> ablation significance (A4 vs A3 paired) -> tables -> figures -> report.

Usage:
  python experiments/run_turs_rrmt_full.py --all
  python experiments/run_turs_rrmt_full.py --dataset ECG5000_UNBAL
  python experiments/run_turs_rrmt_full.py --ablation A4 --dataset ECG5000_BAL
  ... --train-only | --diagnostics-only | --report-only | --plots-only | --force
Default: --all --resume.
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from src.diagnostics import statistics as S
from experiments.turs_rrmt.data import DATASETS, SEED, load_split, manifest, ROOT as PROOT
from experiments.turs_rrmt import train_eval as TE
from experiments.turs_rrmt import diagnostics as DG

RESULTS = os.path.join(ROOT, "results", "turs_rrmt")
CKPT_DIR = os.path.join(RESULTS, "checkpoints")
TABLE_DIR = os.path.join(RESULTS, "tables")
FIG_DIR = os.path.join(RESULTS, "figures")
AUDIT_DIR = os.path.join(RESULTS, "audit")
REPORT = os.path.join(RESULTS, "TURS_RRMT_REPORT.md")
PIPELINE_LOG = os.path.join(ROOT, "logs", "turs_rrmt_full.log")
PID_FILE = os.path.join(ROOT, "turs_rrmt.pid")

VARIANTS = ["A1", "A2", "A3", "A4", "A5"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _sha(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def phase0_audit():
    os.makedirs(AUDIT_DIR, exist_ok=True)
    model_src = os.path.join(ROOT, "models", "turs_rrmt", "model.py")
    S.dump_json(dict(
        model="models/turs_rrmt/model.py::TURSRRMT",
        sha256=_sha(model_src),
        pattern_bank="fixed MiniROCKET-inspired, M=128, lengths {7,11,15,23,31}, seed 42, 0 trainable params",
        local_ppv="avg_pool window round(0.05*T), min 3",
        router="Linear(256->32)->ReLU->Linear(32->4)->softmax",
        flavors=["standard W1 coarse", "tail-weighted W1", "fine-grid W1", "multilag drift"],
        reference="TRAIN-only quantile/lag template (buffers, saved in ckpt)",
        ablations=dict(A0="Ridge on fixed-bank global features",
                       A1="pattern only (transport zero-padded)",
                       A2="single transport flavor",
                       A3="uniform multi-transport (CONTROL)",
                       A4="routed multi-transport + top-2 (THE MODEL)",
                       A5="routed without top-K",
                       A6="shuffled routing on frozen A4",
                       A7="fixed one-hot routing on frozen A4"),
        shared="TransportBuilder views from models/tursnet.py (raw/sorted/drift)",
        protocol=dict(seed=42, lr=3e-4, wd=1e-2, batch=64, max_epochs=30,
                      patience=8, optimizer="AdamW", scheduler="OneCycleLR",
                      monitor="val macro-F1", normalization="per-sample z-norm"),
    ), os.path.join(AUDIT_DIR, "implementation_audit.json"))
    log("  audit written")


def extract_routing(model, ds, device, batch=256):
    """Routing outputs + per-sample predictions for all splits."""
    outs = {}
    for split, X, y in [("train", ds["Xtr"], ds["y_train"]),
                        ("val", ds["Xva"], ds["y_val"]),
                        ("test", ds["Xte"], ds["y_test"])]:
        ps, Ws, Ams, TEs_ = [], [], [], []
        with torch.no_grad():
            for i in range(0, len(X), batch):
                xb = torch.from_numpy(X[i:i + batch]).float().to(device)
                logits, o = model(xb, return_aux=True)
                ps.append(torch.softmax(logits, 1).cpu().numpy())
                w = o["w"]
                Ws.append(w.cpu().numpy())
                # pattern activity cache (bank is cheap to recompute)
                R = model.bank(xb)
                Aa, Sa = model.activity(R)
                Ams.append(Aa.mean(1).cpu().numpy())
                TEs_.append(o["F_T"].mean(-1).cpu().numpy())
        probs = np.concatenate(ps)
        W = np.concatenate(Ws, 0)                      # [N, J, T]
        pred = probs.argmax(1)
        r_ent = -(W * np.log(W + 1e-12)).sum(1).mean(1)
        outs[split] = dict(
            probs=probs, routing_weights=W,
            A_mean=np.concatenate(Ams, 0),
            F_T_pool=np.concatenate(TEs_, 0),
            pred=pred, y=np.asarray(y),
            correct=(pred == np.asarray(y)).astype(np.int8),
            confidence=probs.max(1),
            routing_entropy=r_ent,
            top1_rate=float((W.argmax(1) == 0).mean()),
            flavor_freq=np.bincount(W.argmax(1).ravel(), minlength=model.J)
            / W.shape[0] / W.shape[2],
            route_switch=float((W.argmax(1)[:, 1:] != W.argmax(1)[:, :-1]).mean()),
            X=X)
    return outs


def evaluate_variant(variant, model_or_ridge, ds, device):
    if variant == "A0":
        # RidgeClassifier: coef_ [n_classes, D]; predict via argmax of scores
        feats = model_or_ridge["feats"]

        def predict(F_):
            return (np.asarray(F_) @ model_or_ridge["coef"].T
                    + model_or_ridge["intercept"]).argmax(1)
        out = {}
        for split, (F_, y_) in [("val", feats["val"]), ("test", feats["test"])]:
            pred = predict(F_)
            out[split] = TE.full_metrics(np.asarray(y_), pred, ds["n_cls"])
        out["params_trainable"] = int(model_or_ridge["coef"].size)
        return out
    probs = TE.predict(model_or_ridge, ds["Xte"], device)
    pred = probs.argmax(1)
    out = dict(test=TE.full_metrics(ds["y_test"], pred, ds["n_cls"]))
    probs_va = TE.predict(model_or_ridge, ds["Xva"], device)
    out["val"] = TE.full_metrics(ds["y_val"], probs_va.argmax(1), ds["n_cls"])
    out["test_nll"] = float(-np.log(probs[np.arange(len(probs)), ds["y_test"]]
                                    + 1e-12).mean())
    from src.diagnostics.calibration import ece, brier
    out["test_ece"] = ece(probs, ds["y_test"])
    out["test_brier"] = brier(probs, ds["y_test"])
    return out


def run_dataset(tag, device, force=False, train_only=False, only_ablation=None,
                diag_only=False):
    log(f"=== DATASET {tag} ===")
    ds = load_split(tag)
    os.makedirs(os.path.join(RESULTS, tag), exist_ok=True)
    S.dump_json(manifest(tag, ds), os.path.join(RESULTS, tag, "dataset_manifest.json"))

    variants = [only_ablation] if only_ablation else VARIANTS + ["A0"]
    metas, models = {}, {}
    for v in variants:
        log(f"  [{tag}] train/eval {v}")
        path, meta = TE.train_variant(v, ds, device, log=log, force=force,
                                      ckpt_dir=CKPT_DIR)
        metas[v] = meta
        if v == "A0":
            z = np.load(path, allow_pickle=True)
            models[v] = dict(coef=z["coef"], intercept=z["intercept"],
                             feats=z["feats"].item())
        else:
            models[v] = TE.load_variant(v, ds, device, CKPT_DIR)
    if train_only:
        return None

    results = {"dataset": tag, "variants": {}, "meta": {}}
    for v in variants:
        results["variants"][v] = evaluate_variant(
            v, models[v], ds, device)
        m = {k: val for k, val in metas[v].items() if k != "history"}
        results["meta"][v] = m

    # paired A4 vs A3 significance (THE architectural test)
    if "A3" in models and "A4" in models:
        p3 = TE.predict(models["A3"], ds["Xte"], device)
        p4 = TE.predict(models["A4"], ds["Xte"], device)
        y = ds["y_test"]
        c3 = (p3.argmax(1) == y).astype(float)
        c4 = (p4.argmax(1) == y).astype(float)
        d, lo, hi = S.bootstrap_diff_ci(c4, c3, paired=True)
        chi2, pm = S.mcnemar(c4, c3)
        obs, pp = S.paired_permutation_test(
            -np.log(p4[np.arange(len(y)), y] + 1e-12),
            -np.log(p3[np.arange(len(y)), y] + 1e-12))
        results["A4_vs_A3"] = dict(
            mf1_delta=results["variants"]["A4"]["test"]["macro_f1"]
            - results["variants"]["A3"]["test"]["macro_f1"],
            acc_diff=d, acc_diff_ci95=[lo, hi],
            mcnemar_p=pm, nll_paired_perm_p=pp,
            nll_mean_delta=float(obs),
            cohens_d=S.cohens_d_paired(c4, c3))
        log(f"  [{tag}] A4-vs-A3: dMF1={results['A4_vs_A3']['mf1_delta']:+.4f} "
            f"McNemar p={pm:.4f}")

    if diag_only:
        # load A4 if not trained this session
        if "A4" not in models:
            models["A4"] = TE.load_variant("A4", ds, device, CKPT_DIR)
        model = models["A4"]
        probs = TE.predict(model, ds["Xte"], device)
        ext = extract_routing(model, ds, device)
    else:
        model = models["A4"]
        probs = TE.predict(model, ds["Xte"], device)
        ext = extract_routing(model, ds, device)

    # routing outputs saved per split
    for split in ["train", "val", "test"]:
        e = ext[split]
        np.savez_compressed(os.path.join(RESULTS, tag, f"routing_outputs_{split}.npz"),
                            routing_weights=e["routing_weights"], A_mean=e["A_mean"],
                            F_T_pool=e["F_T_pool"], probs=e["probs"],
                            pred=e["pred"], correct=e["correct"])
    # A6/A7 interventions
    log(f"  [{tag}] A6 shuffled-routing / A7 fixed-routing interventions")
    a6 = TE.routing_intervention(model, ds, device, "shuffled")
    a7 = TE.routing_intervention(model, ds, device, "fixed_0")
    y = ds["y_test"]
    results["variants"]["A6"] = dict(test=TE.full_metrics(y, a6.argmax(1), ds["n_cls"]),
                                     intervention="shuffled routing (frozen A4)")
    results["variants"]["A7"] = dict(test=TE.full_metrics(y, a7.argmax(1), ds["n_cls"]),
                                     intervention="fixed flavor-0 routing (frozen A4)")
    results["intervention_flips"] = dict(
        A6=float((a6.argmax(1) != probs.argmax(1)).mean()),
        A7=float((a7.argmax(1) != probs.argmax(1)).mean()))

    # diagnostics D1-D9 on frozen A4
    if not train_only:
        log(f"  [{tag}] diagnostics D1-D9")
        diag = DG.run_all_diagnostics(model, ds, device, probs, ext["test"], log=log)
        results["diagnostics"] = {k: v for k, v in diag.items() if k != "_hyp_rows"}
        np.save(os.path.join(RESULTS, tag, "hyp_rows.npy"),
                np.array([{**r} for r in diag["_hyp_rows"]], dtype=object),
                allow_pickle=True)
        with open(os.path.join(RESULTS, tag, "full_results.json"), "w") as f:
            json.dump(S.jsonable(results), f, indent=2)
    return results, diag if not train_only else None


def ablation_table(all_results):
    rows = []
    for tag, R in all_results.items():
        row = dict(Dataset=tag)
        for v in ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7"]:
            row[v] = round(R["variants"][v]["test"]["macro_f1"], 4) \
                if v in R["variants"] else None
        rows.append(row)
    S.dump_csv(rows, os.path.join(TABLE_DIR, "model_comparison.csv"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", default=True)
    ap.add_argument("--dataset", type=str, default=None)
    ap.add_argument("--ablation", type=str, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--diagnostics-only", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--plots-only", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    tags = [args.dataset] if args.dataset else DATASETS
    for d in [RESULTS, CKPT_DIR, TABLE_DIR, FIG_DIR, AUDIT_DIR]:
        os.makedirs(d, exist_ok=True)

    if not args.report_only and not args.plots_only:
        phase0_audit()

    all_results, diag_rows = {}, []
    for tag in tags:
        fp = os.path.join(RESULTS, tag, "full_results.json")
        if args.report_only or args.plots_only:
            all_results[tag] = json.load(open(fp))
            continue
        if os.path.exists(fp) and not args.force:
            log(f"  {tag}: completed, skipping (--force to redo)")
            all_results[tag] = json.load(open(fp))
            continue
        out = run_dataset(tag, torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"),
            force=args.force, train_only=args.train_only,
            only_ablation=args.ablation, diag_only=args.diagnostics_only)
        if out is None:
            continue
        res, diag = out
        all_results[tag] = res
        if diag:
            diag_rows.extend([dict(dataset=tag, **r) for r in diag["_hyp_rows"]])
    if args.train_only:
        log("TRAIN-ONLY complete")
        return

    ablation_table(all_results)

    # significance registry with FDR
    if diag_rows:
        fams = sorted(set(r["family"] for r in diag_rows))
        for fam in fams:
            idx = [i for i, r in enumerate(diag_rows) if r["family"] == fam]
            ps = [diag_rows[i]["p_value"] if diag_rows[i]["p_value"] is not None
                  else 1.0 for i in idx]
            qs = S.benjamini_hochberg(ps)
            for i, q in zip(idx, qs):
                diag_rows[i]["q_value"] = float(q)
                diag_rows[i]["significant"] = bool(
                    q < 0.05 and diag_rows[i]["p_value"] is not None)
        S.dump_csv([{k: r.get(k) for k in ["dataset", "family", "test", "comparison",
                                           "estimate", "p_value", "q_value",
                                           "significant"]} for r in diag_rows],
                   os.path.join(TABLE_DIR, "significance.csv"))
        S.dump_csv([{k: r.get(k) for k in ["dataset", "family", "comparison",
                                           "estimate", "extra"]}
                    for r in diag_rows],
                   os.path.join(TABLE_DIR, "effect_sizes.csv"))
        S.dump_json(diag_rows, os.path.join(RESULTS, "hypothesis_registry.json"))

    # routing statistics table
    rrows = []
    for tag in tags:
        fp = os.path.join(RESULTS, tag, "routing_outputs_test.npz")
        if os.path.exists(fp):
            z = np.load(fp)
            W = z["routing_weights"]
            rrows.append(dict(
                dataset=tag, n=int(W.shape[0]),
                mean_w=round(float(W.mean()), 4),
                std_w=round(float(W.std()), 4),
                routing_entropy=round(float(-(W * np.log(W + 1e-12)).sum(1).mean()), 4),
                top1_flavor=int(W.mean(0).argmax()),
                flavor_freq=[round(float(f), 4) for f in
                             np.bincount(W.argmax(1).ravel(), minlength=W.shape[1])
                             / W.shape[0] / W.shape[2]],
                route_switch_rate=round(float(
                    (W.argmax(1)[:, 1:] != W.argmax(1)[:, :-1]).mean()), 4)))
    S.dump_csv(rrows, os.path.join(TABLE_DIR, "routing_statistics.csv"))

    # diagnostic summary
    drows = []
    for tag, R in all_results.items():
        d = R.get("diagnostics", {})
        for k in ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"]:
            if k in d:
                drows.append(dict(dataset=tag, experiment=k,
                                  summary=json.dumps(S.jsonable(d[k]))[:400]))
    S.dump_csv(drows, os.path.join(TABLE_DIR, "diagnostic_results.csv"))

    # figures
    if not args.report_only:
        log("figures")
        try:
            from experiments.turs_rrmt.plotting import make_figures
            make_figures(all_results, FIG_DIR)
        except Exception as e:
            log(f"  [plots] WARNING {type(e).__name__}: {e}")

    # metadata
    S.dump_json(dict(
        date=time.strftime("%Y-%m-%d %H:%M:%S"),
        elapsed_s=round(time.time() - t0, 1),
        python=platform.python_version(), torch=torch.__version__,
        device="cuda" if torch.cuda.is_available() else "cpu",
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        seeds=dict(model=SEED, bootstrap=7200, permutation=7300,
                   interventions=SEED),
        checkpoints={f"{t}_{v}": _sha(os.path.join(CKPT_DIR, f"{t}_{v}.pt"))
                     for t in tags for v in VARIANTS
                     if os.path.exists(os.path.join(CKPT_DIR, f"{t}_{v}.pt"))},
    ), os.path.join(RESULTS, "run_metadata.json"))

    # report
    from experiments.turs_rrmt.reporting import write_report
    write_report(all_results, diag_rows, REPORT,
                 elapsed_s=time.time() - t0)
    log(f"COMPLETE in {(time.time()-t0)/60:.1f} min -> {REPORT}")


if __name__ == "__main__":
    main()
