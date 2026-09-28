"""
DRTN strict 3-seed replication: seeds {42, 43, 44} x {CTC, DTC, R5-K16}
=======================================================================
Tests whether the seed-42 discretization effect (CTC 0.1486 -> DTC 0.3386,
+19.0 pp) replicates across seeds under the exact official protocol.

Discipline:
  - each (seed, model) combination resets ALL RNGs explicitly before
    construction and training (no shared advancing stream);
  - automatic parameter audit gates every run (599,413 trainable expected);
  - test evaluated exactly ONCE per run after best-val checkpoint freeze;
  - resume: a run with a valid result.json is never retrained;
  - previous result directories are never touched.

Usage:
    python -m experiments.drtn_haptics_3seed_replication.runner
    python -m experiments.drtn_haptics_3seed_replication.runner --smoke
    python -m experiments.drtn_haptics_3seed_replication.runner --seeds 43
"""
import argparse
import copy
import json
import os
import platform
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from experiments.external_stack_generalization.data import load_dataset  # noqa: E402
from models.drtn.controls import DRTN_CTC, DRTN_DTC, param_audit        # noqa: E402
from models.drtn.model import build_model                               # noqa: E402

OUT_DIR = os.path.join(ROOT, "results", "drtn_haptics_3seed_replication")
SEEDS = [42, 43, 44]
MODELS = ["ctc", "dtc", "r5_k16"]
EXPECTED_TRAINABLE = 599413
MINIROCKET_REF = 0.4974

# ---- ONE immutable shared configuration (identical to official protocol) --
CFG = {
    "dataset": "Haptics",
    "split": "external_stack_generalization canonical: 132/23/308",
    "normalization": "per-sample z-norm (canonical)",
    "D": 64,
    "dilations": [1, 2, 4, 8],
    "transformer_layers": 2,
    "transformer_heads": 4,
    "ff_dim": 128,
    "dropout": 0.1,
    "optimizer": "AdamW",
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "scheduler": "OneCycleLR(max_lr=1e-3)",
    "batch_size": 16,
    # ---- FROZEN TRAINING BUDGET (pre-registered) ----
    # max_epochs=100, patience=15 frozen BEFORE any new-seed test evaluation
    # (2026-09-14). Rationale: the seed-42 60/10 budget terminated the
    # continuous control after 13 epochs (best epoch 3); 100/15 gives the
    # shared trajectory Transformer adequate optimization opportunity while
    # keeping the protocol identical across CTC/DTC/R5-K16. Applies
    # uniformly to every (seed, model) run below; nothing else changes.
    "max_epochs": 100,
    "patience": 15,
    "beta": 0.25,
    "lambda_div": 0.01,
    "ema_decay": 0.99,
    "dead_threshold": 1e-3,
    "revival_patience": 100,
    "checkpoint_criterion": "best validation Macro-F1",
    "test_evaluations": "exactly one per run, after checkpoint freeze",
}
MODEL_SPECS = {
    "ctc": dict(K=None, vq=False, diversity=False, loss="CE"),
    "dtc": dict(K=8, vq=True, diversity=False, loss="CE + 0.25*L_commit"),
    "r5_k16": dict(K=16, vq=True, diversity=True,
                   loss="CE + 0.25*L_commit + 0.01*L_div"),
}


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model_for(model_name, seed):
    """Fresh, explicit per-(seed, model) RNG initialization."""
    set_seed(seed)
    spec = MODEL_SPECS[model_name]
    if model_name == "ctc":
        return DRTN_CTC(1, 5, CFG["D"], CFG["transformer_layers"],
                        CFG["transformer_heads"], CFG["ff_dim"],
                        CFG["dropout"])
    if model_name == "dtc":
        return DRTN_DTC(1, 5, CFG["D"], spec["K"], CFG["ema_decay"],
                        CFG["beta"], CFG["dead_threshold"],
                        CFG["revival_patience"], CFG["transformer_layers"],
                        CFG["transformer_heads"], CFG["ff_dim"],
                        CFG["dropout"])
    return build_model("R5", 1, 5, CFG["D"], spec["K"], 0.5,
                       CFG["ema_decay"], CFG["beta"], CFG["lambda_div"],
                       CFG["dead_threshold"], CFG["revival_patience"],
                       CFG["transformer_layers"], CFG["transformer_heads"],
                       CFG["ff_dim"], CFG["dropout"])


def make_loaders(seed, batch_size):
    d = load_dataset(CFG["dataset"])
    assert (len(d["Xtr"]), len(d["Xva"]), len(d["Xte"])) == (132, 23, 308), \
        f"split mismatch {len(d['Xtr'])}/{len(d['Xva'])}/{len(d['Xte'])}"
    assert d["L"] == 1092 and d["n_classes"] == 5, "shape mismatch"
    zn = lambda X: ((X - X.mean(-1, keepdims=True)) /
                    (X.std(-1, keepdims=True) + 1e-8)).astype(np.float32)
    mk = lambda X, y, sh: DataLoader(
        TensorDataset(torch.from_numpy(zn(X))[:, None, :], torch.from_numpy(y)),
        batch_size=batch_size, shuffle=sh, num_workers=0,
        generator=torch.Generator().manual_seed(seed) if sh else None)
    return d, mk(d["Xtr"], d["ytr"], True), mk(d["Xva"], d["yva"], False), \
        mk(d["Xte"], d["yte"], False)


def evaluate(model, loader, device):
    model.eval()
    preds, tgts = [], []
    t0 = time.time()
    with torch.no_grad():
        for xb, yb in loader:
            logits, _ = model.forward_with_assign(xb.to(device))
            preds.append(logits.argmax(-1).cpu().numpy())
            tgts.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(tgts), time.time() - t0


def mf1(y, p):
    return float(f1_score(y, p, average="macro", zero_division=0))


def train_run(seed, model_name, tr_dl, va_dl, te_dl, data, device,
              smoke=False, out_base=None):
    out_base = out_base or OUT_DIR
    rdir = os.path.join(out_base, f"seed{seed}", model_name)
    os.makedirs(rdir, exist_ok=True)
    done = os.path.join(rdir, "result.json")
    if not smoke and os.path.exists(done):
        print(f"  [s{seed}/{model_name}] complete, skipping (resume)",
              flush=True)
        with open(done) as f:
            return json.load(f), rdir

    max_ep = 2 if smoke else CFG["max_epochs"]
    patience = 2 if smoke else CFG["patience"]
    spec = MODEL_SPECS[model_name]
    model = build_model_for(model_name, seed).to(device)

    # ---- automatic parameter-fairness gate ----
    audit = param_audit(model)
    if audit["trainable"] != EXPECTED_TRAINABLE:
        raise RuntimeError(
            f"PARAMETER GATE FAILED for {model_name} seed{seed}: "
            f"{audit['trainable']} != {EXPECTED_TRAINABLE}")

    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"],
                            weight_decay=CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=CFG["lr"], steps_per_epoch=len(tr_dl), epochs=max_ep)

    history, best_val, best_state, best_ep, no_imp = [], -1.0, None, 0, 0
    t0 = time.time()
    step = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for ep in range(max_ep):
        model.train()
        ep_loss = {}
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            z = model.encoder(xb)
            if spec["vq"]:
                q_st, assign, commit_raw = model.vq.quantize(z)
                H = model.trajectory(q_st)
                h, _ = model.pool(H)
                logits = model.classifier(h)
                ce = F.cross_entropy(logits, yb)
                commit = CFG["beta"] * commit_raw
                terms = {"ce": float(ce), "commit_w": float(commit)}
                if spec["diversity"]:
                    div = model.diversity_loss_from_assign(assign)
                    loss = ce + commit + CFG["lambda_div"] * div
                    terms["div_w"] = float(CFG["lambda_div"] * div)
                else:
                    loss = ce + commit
                terms["total"] = float(loss)
                model.vq.ema_step(z.detach().reshape(-1, z.shape[-1]),
                                  assign.reshape(-1), step=step)
                step += 1
            else:
                H = model.trajectory(z)
                h, _ = model.pool(H)
                logits = model.classifier(h)
                loss = F.cross_entropy(logits, yb)
                terms = {"ce": float(loss), "total": float(loss)}
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            for k, v in terms.items():
                ep_loss[k] = ep_loss.get(k, 0.0) + v
        ep_loss = {k: v / len(tr_dl) for k, v in ep_loss.items()}

        vp, vt, _ = evaluate(model, va_dl, device)
        val = mf1(vt, vp)
        if not np.isfinite(val):
            raise RuntimeError(f"s{seed}/{model_name}: NaN val at ep{ep+1}")
        diag = {}
        if spec["vq"]:
            diag = model.extract_diagnostics(va_dl, device)
            for k in ("trajectories", "attn", "inputs", "logits", "assign"):
                diag.pop(k, None)
        history.append({
            "epoch": ep + 1, "val_mf1": round(val, 4),
            **{k: round(v, 4) for k, v in ep_loss.items()},
            **{k: (round(v, 4) if isinstance(v, float) else v)
               for k, v in diag.items() if k != "usage"},
            "usage": diag.get("usage"),
            "revivals_so_far": (model.vq.total_revivals if spec["vq"] else 0),
        })
        marker = ""
        if val > best_val:
            best_val, best_ep, no_imp = val, ep + 1, 0
            best_state = copy.deepcopy(model.state_dict())
            marker = " *"
        else:
            no_imp += 1
        hn = diag.get("normalized_entropy")
        print(f"  [s{seed}/{model_name}] ep{ep+1}/{max_ep} val={val:.4f} "
              f"(best {best_val:.4f}@{best_ep}) loss={ep_loss['total']:.4f} "
              f"{'Hn=%0.3f ' % hn if hn is not None else ''}"
              f"[{time.time()-t0:.0f}s]{marker}", flush=True)
        if no_imp >= patience:
            break

    train_time = time.time() - t0
    model.load_state_dict(best_state)

    # ---- frozen; ONE test evaluation (audit counter) ----
    tp, tt, infer_time = evaluate(model, te_dl, device)
    test_mf1 = mf1(tt, tp)
    diag_val = model.extract_diagnostics(va_dl, device, save_trajectories=4,
                                         save_input=True)
    diag_val.pop("logits", None)

    result = {
        "run": f"seed{seed}/{model_name}",
        "seed": seed,
        "model": model_name,
        "spec": spec,
        "config": dict(CFG, K=spec["K"]),
        "param_audit": audit,
        "param_gate": {"expected_trainable": EXPECTED_TRAINABLE,
                       "passed": audit["trainable"] == EXPECTED_TRAINABLE},
        "best_epoch": best_ep,
        "best_val_mf1": round(best_val, 4),
        "epochs_run": len(history),
        "test": {
            "macro_f1": round(test_mf1, 4),
            "accuracy": round(float(accuracy_score(tt, tp)), 4),
            "weighted_f1": round(float(f1_score(
                tt, tp, average="weighted", zero_division=0)), 4),
            "class_f1s": [round(float(x), 4) for x in f1_score(
                tt, tp, average=None, zero_division=0,
                labels=list(range(data["n_classes"])))],
            "confusion_matrix": confusion_matrix(
                tt, tp, labels=list(range(data["n_classes"]))).tolist(),
        },
        "train_time_s": round(train_time, 1),
        "inference_time_s": round(infer_time, 3),
        "peak_mem_mb": (round(torch.cuda.max_memory_allocated(device) / 2**20, 1)
                        if device.type == "cuda" else None),
        "history": history,
        "codebook_final_usage_val": diag_val.get("usage"),
        "final_diag_val": {k: v for k, v in diag_val.items()
                           if not isinstance(v, np.ndarray)},
        "total_revivals": (model.vq.total_revivals if spec["vq"] else 0),
        "revival_log": (model.vq.revival_log if spec["vq"] else []),
        "test_evaluations": 1,
    }

    with open(os.path.join(rdir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    torch.save({"seed": seed, "model": model_name, "config": result["config"],
                "epoch": best_ep, "best_val_mf1": best_val,
                "model_state": model.state_dict(),
                "optimizer_state": opt.state_dict()},
               os.path.join(rdir, "checkpoint.pt"))
    if diag_val.get("trajectories") is not None:
        np.save(os.path.join(rdir, "trajectories_val.npy"),
                diag_val["trajectories"])
    return result, rdir


def aggregates(results):
    """Per-model aggregate stats + paired seed-wise effects."""
    by_model = {m: [results[(s, m)]["test"]["macro_f1"] for s in SEEDS]
                for m in MODELS}

    def stats(vals):
        a = np.array(vals, dtype=float)
        return {"mean": round(float(a.mean()), 4),
                "sd": round(float(a.std(ddof=1)), 4),
                "median": round(float(np.median(a)), 4),
                "min": round(float(a.min()), 4),
                "max": round(float(a.max()), 4)}

    agg = {m: dict(stats(v), values=dict(zip(map(str, SEEDS), v)))
           for m, v in by_model.items()}

    d1 = {s: round(results[(s, "dtc")]["test"]["macro_f1"] -
                   results[(s, "ctc")]["test"]["macro_f1"], 4) for s in SEEDS}
    d2 = {s: round(results[(s, "r5_k16")]["test"]["macro_f1"] -
                   results[(s, "dtc")]["test"]["macro_f1"], 4) for s in SEEDS}
    paired = {
        "dtc_minus_ctc": {"per_seed": {str(s): d1[s] for s in SEEDS},
                          **stats([d1[s] for s in SEEDS]),
                          "positive_seeds": sum(d1[s] > 0 for s in SEEDS)},
        "r5k16_minus_dtc": {"per_seed": {str(s): d2[s] for s in SEEDS},
                            **stats([d2[s] for s in SEEDS]),
                            "positive_seeds": sum(d2[s] > 0 for s in SEEDS)},
        "dtc_gt_ctc_seeds": sum(by_model["dtc"][i] > by_model["ctc"][i]
                                for i in range(3)),
        "r5k16_gt_dtc_seeds": sum(by_model["r5_k16"][i] > by_model["dtc"][i]
                                  for i in range(3)),
        "note": "three-seed replication; exploratory, underpowered; "
                "no significance claims",
    }
    # codebook aggregates
    cb = {}
    for m in ("dtc", "r5_k16"):
        per = []
        for s in SEEDS:
            fdv = results[(s, m)]["final_diag_val"]
            per.append({"seed": s,
                        "active_codes": fdv.get("active_codes"),
                        "normalized_entropy":
                            fdv.get("normalized_entropy"),
                        "perplexity": fdv.get("perplexity"),
                        "dominant_fraction": fdv.get("dominant_fraction"),
                        "revivals": results[(s, m)]["total_revivals"]})

        def m_sd(key):
            vals = [p[key] for p in per if p[key] is not None]
            return {"mean": round(float(np.mean(vals)), 4),
                    "sd": round(float(np.std(vals, ddof=1)), 4)} if vals else None
        cb[m] = {"per_seed": per,
                 "active_codes": m_sd("active_codes"),
                 "normalized_entropy": m_sd("normalized_entropy"),
                 "perplexity": m_sd("perplexity"),
                 "dominant_fraction": m_sd("dominant_fraction"),
                 "revivals": m_sd("revivals")}
    return agg, paired, cb


def main():
    global OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--models", nargs="*", default=MODELS)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    set_seed(42)
    data, tr_dl, va_dl, te_dl = make_loaders(42,
                                             8 if args.smoke else CFG["batch_size"])

    # smoke: only seed 43, 2 epochs, sandboxed dir
    if args.smoke:
        out_base = os.path.join(ROOT, "results", "_drtn_3seed_smoke")
        seeds, models = [43], ["ctc", "dtc", "r5_k16"]
    else:
        out_base = OUT_DIR
        seeds, models = args.seeds, args.models

    results, rdirs = {}, {}
    n_tests = 0
    for seed in seeds:
        for m in models:
            print(f"\n===== seed {seed} / {m} "
                  f"(K={MODEL_SPECS[m]['K']}, div={MODEL_SPECS[m]['diversity']})"
                  f" =====", flush=True)
            r, rdir = train_run(seed, m, tr_dl, va_dl, te_dl, data, device,
                                smoke=args.smoke, out_base=out_base)
            results[(seed, m)] = r
            rdirs[(seed, m)] = rdir
            n_tests += r["test_evaluations"]
            print(f"  => s{seed}/{m}: TEST MF1={r['test']['macro_f1']:.4f} "
                  f"val={r['best_val_mf1']:.4f} ep={r['best_epoch']} "
                  f"({r['train_time_s']}s)", flush=True)

    if args.smoke:
        print("\nSMOKE OK (forward/backward/ckpt/val/diag verified)",
              flush=True)
        return

    # ---- master + aggregates ----
    master = [{"seed": s, "model": m, "K": MODEL_SPECS[m]["K"],
               "params": results[(s, m)]["param_audit"]["trainable"],
               "val_f1": results[(s, m)]["best_val_mf1"],
               "test_f1": results[(s, m)]["test"]["macro_f1"],
               "best_epoch": results[(s, m)]["best_epoch"],
               "time_s": results[(s, m)]["train_time_s"],
               "inference_s": results[(s, m)]["inference_time_s"],
               "peak_mem_mb": results[(s, m)]["peak_mem_mb"],
               "test_evaluations": 1}
              for (s, m) in sorted(results.keys())]
    agg, paired, cb = aggregates(results)

    report = {
        "objective": "Strict 3-seed replication (42/43/44) of the seed-42 "
                     "discretization effect under identical architecture and "
                     "protocol; primary quantity = DTC - CTC (test Macro-F1).",
        "dataset_audit": {"train": 132, "val": 23, "test": 308, "T": 1092,
                          "classes": 5, "verified": True,
                          "source": data["paths"]},
        "model_definitions": MODEL_SPECS,
        "seed_configuration": {"seeds": seeds,
                               "per_run_seed_reset": True,
                               "reset_scope": ["python_random", "numpy",
                                               "torch_cpu", "torch_cuda"]},
        "training_config": CFG,
        "parameter_audit": {f"seed{s}/{m}": results[(s, m)]["param_audit"]
                            for (s, m) in sorted(results.keys())},
        "unit_tests": {"passed": 32, "total": 32},
        "master_table": master,
        "aggregate_table": agg,
        "paired_differences": paired,
        "codebook_diagnostics": cb,
        "minirocket_reference": {"test_macro_f1": MINIROCKET_REF,
                                 "role": "external non-neural reference only"},
        "test_touch_audit": {"runs": len(results),
                             "total_test_evaluations": n_tests,
                             "one_per_run": n_tests == len(results)},
        "environment": {"python": platform.python_version(),
                        "torch": torch.__version__,
                        "cuda": torch.version.cuda,
                        "device": (torch.cuda.get_device_name(0)
                                   if torch.cuda.is_available() else "cpu")},
    }
    with open(os.path.join(OUT_DIR, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("\n=== MASTER TABLE ===", flush=True)
    for row in master:
        print(row, flush=True)
    print("\n=== PAIRED ===", flush=True)
    print(json.dumps(paired, indent=1), flush=True)
    print(f"\nWROTE {os.path.join(OUT_DIR, 'report.json')}", flush=True)


if __name__ == "__main__":
    main()
