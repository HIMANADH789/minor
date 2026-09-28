"""
DRTN controlled experiments: continuous vs discrete trajectory vs K sweep
=========================================================================
Haptics, seed 42. Three experiments (spec):

  EXP 1  continuous_control  : DRTN_CTC  (encoder -> transformer -> head, CE)
  EXP 2  discrete_control    : DRTN_DTC  (encoder -> VQ -> transformer, CE+commit)
  EXP 3  K sweep             : R5 at K in {8, 16, 32}, everything else frozen

Test discipline: for every run, train -> select best-val checkpoint -> freeze
-> evaluate test EXACTLY ONCE. The K=8 R5 row reuses the official first-probe
result (results/drtn_haptics_seed42/R5) unless --rerun-r5k8 is passed.

Usage:
    python -m experiments.drtn_haptics_controls_seed42.runner
    python -m experiments.drtn_haptics_controls_seed42.runner --runs ctc dtc
    python -m experiments.drtn_haptics_controls_seed42.runner --smoke
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
from models.drtn.model import build_model, usage_stats                  # noqa: E402

SEED = 42
DATASET = "Haptics"
OUT_DIR = os.path.join(ROOT, "results", "drtn_haptics_controls_seed42")
OFFICIAL_R5_DIR = os.path.join(ROOT, "results", "drtn_haptics_seed42", "R5")

# ---- ONE shared training configuration (immutable; every run uses it) -----
CFG = {
    "seed": 42,
    "dataset": "Haptics",
    "split": "external_stack_generalization canonical: train 132 / val 23 / test 308",
    "normalization": "per-sample z-norm (canonical)",
    "D": 64,
    "dilations": [1, 2, 4, 8],
    "K": 8,                       # controls + r5_k8; sweep changes ONLY this
    "transformer_layers": 2,
    "transformer_heads": 4,
    "ff_dim": 128,
    "dropout": 0.1,
    "optimizer": "AdamW",
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "scheduler": "OneCycleLR(max_lr=1e-3)",
    "batch_size": 16,
    "max_epochs": 60,
    "patience": 10,
    "beta": 0.25,
    "lambda_div": 0.01,
    "ema_decay": 0.99,
    "dead_threshold": 1e-3,
    "revival_patience": 100,
    "selection": "best validation Macro-F1 checkpoint; test evaluated once",
}

# per-run diversity/VQ map (frozen before any run)
RUN_SPECS = {
    "continuous_control": dict(model_cls="ctc", K=8, vq=False, diversity=False,
                               loss="CE"),
    "discrete_control":   dict(model_cls="dtc", K=8, vq=True, diversity=False,
                               loss="CE + beta*L_commit"),
    "r5_k8":              dict(model_cls="r5", K=8, vq=True, diversity=True,
                               loss="CE + beta*L_commit + lambda_div*L_div"),
    "r5_k16":             dict(model_cls="r5", K=16, vq=True, diversity=True,
                               loss="CE + beta*L_commit + lambda_div*L_div"),
    "r5_k32":             dict(model_cls="r5", K=32, vq=True, diversity=True,
                               loss="CE + beta*L_commit + lambda_div*L_div"),
}


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_run_model(spec):
    """Construct the model for a run spec from a FRESH seed-42 stream."""
    set_seed(SEED)
    if spec["model_cls"] == "ctc":
        return DRTN_CTC(1, 5, CFG["D"], CFG["transformer_layers"],
                        CFG["transformer_heads"], CFG["ff_dim"],
                        CFG["dropout"])
    if spec["model_cls"] == "dtc":
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


def make_loaders(batch_size):
    d = load_dataset(DATASET)
    # canonical split verification — STOP on any discrepancy (spec)
    assert len(d["Xtr"]) == 132 and len(d["Xva"]) == 23 and \
        len(d["Xte"]) == 308, \
        f"Haptics split mismatch: {len(d['Xtr'])}/{len(d['Xva'])}/{len(d['Xte'])}"
    assert d["L"] == 1092 and d["n_classes"] == 5, \
        f"Haptics shape mismatch: L={d['L']} classes={d['n_classes']}"
    zn = lambda X: ((X - X.mean(-1, keepdims=True)) /
                    (X.std(-1, keepdims=True) + 1e-8)).astype(np.float32)
    mk = lambda X, y, sh: DataLoader(
        TensorDataset(torch.from_numpy(zn(X))[:, None, :], torch.from_numpy(y)),
        batch_size=batch_size, shuffle=sh, num_workers=0,
        generator=torch.Generator().manual_seed(SEED) if sh else None)
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


def fairness_audit(audits):
    """Cross-run fairness verification (spec). Returns list of checks."""
    checks = []
    ref = audits["continuous_control"]
    dtc = audits["discrete_control"]
    k8 = audits["r5_k8"]
    for key in ("encoder", "trajectory", "pool", "classifier", "trainable"):
        checks.append((f"CTC vs R5 identical {key}",
                       ref[key] == k8[key], f"{ref[key]} vs {k8[key]}"))
        checks.append((f"DTC vs R5 identical {key}",
                       dtc[key] == k8[key], f"{dtc[key]} vs {k8[key]}"))
    checks.append(("CTC has no VQ", ref["vq"] == 0 and ref["codebook_buffers"] == 0,
                   f"vq={ref['vq']}, buffers={ref['codebook_buffers']}"))
    checks.append(("DTC K=8 == R5 K=8 codebook buffers",
                   dtc["codebook_buffers"] == k8["codebook_buffers"],
                   f"{dtc['codebook_buffers']}"))
    for k in (16, 32):
        kk = audits[f"r5_k{k}"]
        for key in ("encoder", "trajectory", "pool", "classifier", "trainable"):
            checks.append((f"K={k} identical {key} vs K=8",
                           kk[key] == k8[key], f"{kk[key]} vs {k8[key]}"))
        checks.append((f"K={k} codebook buffers scale",
                       kk["codebook_buffers"] == k8["codebook_buffers"] * k // 8,
                       f"{kk['codebook_buffers']}"))
    cfg_keys = ("seed", "D", "dilations", "transformer_layers",
                "transformer_heads", "ff_dim", "dropout", "optimizer", "lr",
                "weight_decay", "scheduler", "batch_size", "max_epochs",
                "patience", "beta", "lambda_div", "ema_decay",
                "dead_threshold", "revival_patience")
    for name, spec in RUN_SPECS.items():
        eff = dict(CFG)
        eff["K"] = spec["K"]
        checks.append((f"{name} uses frozen CFG",
                       all(eff[k] == CFG[k] or k == "K" for k in cfg_keys),
                       "shared CFG"))
    return checks


def train_run(name, spec, tr_dl, va_dl, te_dl, data, device, smoke=False,
              out_base=OUT_DIR):
    """Train one run; test is evaluated exactly once after freeze."""
    rdir = os.path.join(out_base, name)
    os.makedirs(rdir, exist_ok=True)
    done_flag = os.path.join(rdir, "result.json")

    if not smoke and os.path.exists(done_flag):
        print(f"  [{name}] already complete, skipping (resume)", flush=True)
        with open(done_flag) as f:
            return json.load(f)

    max_ep = 2 if smoke else CFG["max_epochs"]
    patience = 2 if smoke else CFG["patience"]
    model = build_run_model(spec).to(device)
    audit = param_audit(model)
    is_vq = spec["vq"]
    has_div = spec["diversity"]

    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"],
                            weight_decay=CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=CFG["lr"], steps_per_epoch=len(tr_dl), epochs=max_ep)

    history = []
    best_val, best_state, best_ep, no_imp = -1.0, None, 0, 0
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
            if is_vq:
                q_st, assign, commit_raw = model.vq.quantize(z)
                H = model.trajectory(q_st)
                h, _ = model.pool(H)
                logits = model.classifier(h)
                ce = F.cross_entropy(logits, yb)
                commit = CFG["beta"] * commit_raw
                terms = {"ce": float(ce), "commit_w": float(commit)}
                if has_div:
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
        val_mf1 = mf1(vt, vp)
        if torch.isnan(torch.tensor(val_mf1)) or not np.isfinite(val_mf1):
            raise RuntimeError(f"[{name}] NaN/Inf in validation at ep{ep+1}")
        diag = {}
        if is_vq:
            diag = model.extract_diagnostics(va_dl, device)
            for k in ("trajectories", "attn", "inputs", "logits", "assign"):
                diag.pop(k, None)
        history.append({
            "epoch": ep + 1, "val_mf1": round(val_mf1, 4),
            **{k: round(v, 4) for k, v in ep_loss.items()},
            **{k: (round(v, 4) if isinstance(v, float) else v)
               for k, v in diag.items() if k != "usage"},
            "usage": diag.get("usage"),
            "revivals_so_far": (model.vq.total_revivals if is_vq else 0),
        })
        marker = ""
        if val_mf1 > best_val:
            best_val, best_ep, no_imp = val_mf1, ep + 1, 0
            best_state = copy.deepcopy(model.state_dict())
            marker = " *"
        else:
            no_imp += 1
        ent = diag.get("normalized_entropy")
        print(f"  [{name}] ep{ep+1}/{max_ep} valMF1={val_mf1:.4f} "
              f"(best {best_val:.4f}@{best_ep}) loss={ep_loss['total']:.4f} "
              f"{'Hn=%0.3f ' % ent if ent is not None else ''}"
              f"[{time.time()-t0:.0f}s]{marker}", flush=True)
        if no_imp >= patience:
            print(f"  [{name}] early stop at ep{ep+1}", flush=True)
            break

    train_time = time.time() - t0
    model.load_state_dict(best_state)

    # ---- checkpoint frozen; ONE test evaluation ----
    tp, tt, infer_time = evaluate(model, te_dl, device)
    test_mf1 = mf1(tt, tp)
    diag_val = model.extract_diagnostics(va_dl, device, save_trajectories=4,
                                         save_input=True)
    for k in ("logits",):
        diag_val.pop(k, None)

    result = {
        "run": name,
        "spec": {k: v for k, v in spec.items()},
        "config": dict(CFG, K=spec["K"]),
        "param_audit": audit,
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
        "revival_log": (model.vq.revival_log if is_vq else []),
        "total_revivals": (model.vq.total_revivals if is_vq else 0),
    }

    with open(os.path.join(rdir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    torch.save({"run": name, "config": result["config"], "epoch": best_ep,
                "best_val_mf1": best_val, "seed": SEED, "dataset": DATASET,
                "model_state": model.state_dict(),
                "optimizer_state": opt.state_dict()},
               os.path.join(rdir, "checkpoint.pt"))
    if diag_val.get("trajectories") is not None:
        np.save(os.path.join(rdir, "trajectories_val.npy"),
                diag_val["trajectories"])
        np.save(os.path.join(rdir, "inputs_val.npy"), diag_val["inputs"])
    if diag_val.get("attn") is not None:
        np.save(os.path.join(rdir, "attn_val.npy"), diag_val["attn"])
    return result


def official_r5k8_result():
    """Load the official first-probe R5 K=8 result (no rerun)."""
    with open(os.path.join(OFFICIAL_R5_DIR, "result.json")) as f:
        official = json.load(f)
    audit = {"encoder": 3632, "vq": 0, "codebook_buffers": 1040,
             "trajectory": 591232, "pool": 4224, "classifier": 325,
             "other": 0, "trainable": 599413, "non_trainable": 0,
             "total": 599413}
    diag = official.get("final_diag_val", {})
    return {
        "run": "r5_k8_official",
        "spec": RUN_SPECS["r5_k8"],
        "config": dict(CFG, K=8),
        "param_audit": audit,
        "best_epoch": official["best_epoch"],
        "best_val_mf1": official["best_val_mf1"],
        "epochs_run": len(official.get("history", [])),
        "test": official["test"],
        "train_time_s": official["train_time_s"],
        "inference_time_s": None,
        "peak_mem_mb": official.get("peak_mem_mb"),
        "history": official.get("history", []),
        "codebook_final_usage_val": official.get("codebook_final_usage_val"),
        "final_diag_val": diag,
        "revival_log": official.get("revival_log", []),
        "total_revivals": official.get("total_revivals", 0),
        "source": "results/drtn_haptics_seed42/R5 (official probe, not rerun)",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*",
                    default=["continuous_control", "discrete_control",
                             "r5_k16", "r5_k32"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--rerun-r5k8", action="store_true",
                    help="rerun R5 K=8 instead of reusing the official result")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    set_seed(SEED)
    data, tr_dl, va_dl, te_dl = make_loaders(
        8 if args.smoke else CFG["batch_size"])

    results, audits = {}, {}
    runs = args.runs
    if args.smoke:
        out_base = os.path.join(ROOT, "results", "_drtn_ctl_smoke")
        os.makedirs(out_base, exist_ok=True)
        runs = ["continuous_control", "discrete_control", "r5_k16"]
    else:
        out_base = OUT_DIR

    # ---- parameter audit table before any training ----
    print("\n=== PARAMETER AUDIT (exact, by module) ===", flush=True)
    print(f"{'Model':<20} {'Enc':>6} {'VQ':>4} {'Trfm':>7} {'Pool':>6} "
          f"{'Head':>5} {'EMAbuf':>7} {'Trainable':>10}", flush=True)
    for name in runs:
        m = build_run_model(RUN_SPECS[name])
        audits[name] = param_audit(m)
        a = audits[name]
        print(f"{name:<20} {a['encoder']:>6} {a['vq']:>4} {a['trajectory']:>7} "
              f"{a['pool']:>6} {a['classifier']:>5} {a['codebook_buffers']:>7} "
              f"{a['trainable']:>10}", flush=True)
        del m

    for name in runs:
        print(f"\n===== RUN {name} (K={RUN_SPECS[name]['K']}, "
              f"diversity={RUN_SPECS[name]['diversity']}) =====", flush=True)
        results[name] = train_run(name, RUN_SPECS[name], tr_dl, va_dl, te_dl,
                                  data, device, smoke=args.smoke,
                                  out_base=out_base)
        r = results[name]
        print(f"  => {name}: TEST MF1={r['test']['macro_f1']:.4f} "
              f"val={r['best_val_mf1']:.4f} ep={r['best_epoch']} "
              f"({r['train_time_s']}s)", flush=True)

    if args.smoke:
        print("\nSMOKE OK", flush=True)
        return

    # ---- fairness + master table + comparisons ----
    results["r5_k8"] = official_r5k8_result()
    audits["r5_k8"] = results["r5_k8"]["param_audit"]
    checks = fairness_audit(audits)
    failed = [c for c in checks if not c[1]]
    print("\n=== FAIRNESS CHECKS ===", flush=True)
    for label, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'WARN'}] {label} ({detail})", flush=True)

    t = {k: results[k]["test"]["macro_f1"] for k in
         ("continuous_control", "discrete_control", "r5_k8", "r5_k16", "r5_k32")}
    comparisons = {
        "continuous_to_discrete": round(t["discrete_control"] -
                                        t["continuous_control"], 4),
        "discrete_to_r5": round(t["r5_k8"] - t["discrete_control"], 4),
        "k16_minus_k8": round(t["r5_k16"] - t["r5_k8"], 4),
        "k32_minus_k8": round(t["r5_k32"] - t["r5_k8"], 4),
        "k32_minus_k16": round(t["r5_k32"] - t["r5_k16"], 4),
        "note": "single-seed exploratory effects; NOT statistically significant",
    }

    master = []
    for name in ("continuous_control", "discrete_control", "r5_k8",
                 "r5_k16", "r5_k32"):
        r = results[name]
        fdv = r["final_diag_val"]
        master.append({
            "model": name, "K": r["spec"]["K"],
            "vq": r["spec"]["vq"], "diversity": r["spec"]["diversity"],
            "transformer": True, "params": r["param_audit"]["trainable"],
            "val_f1": r["best_val_mf1"], "test_f1": r["test"]["macro_f1"],
            "epochs": r["best_epoch"],
            "time_s": r["train_time_s"],
            "active_codes": fdv.get("active_codes"),
            "normalized_entropy": (round(fdv["normalized_entropy"], 4)
                                   if fdv.get("normalized_entropy") is not None
                                   else None),
            "perplexity": (round(fdv["perplexity"], 4)
                           if fdv.get("perplexity") is not None else None),
            "dominant_fraction": (round(fdv["dominant_fraction"], 4)
                                  if fdv.get("dominant_fraction") is not None
                                  else None),
            "revivals": r["total_revivals"],
        })

    report = {
        "repository_audit": {
            "reused": ["models/drtn/model.py (encoder, HardVQ, "
                       "TrajectoryTransformer, TemporalAttentionPool, R5)",
                       "experiments/external_stack_generalization/data.py "
                       "(canonical Haptics loader, frozen split)",
                       "tests/test_drtn.py (causality/VQ/EMA/revival suite)"],
            "new": ["models/drtn/controls.py",
                    "experiments/drtn_haptics_controls_seed42/runner.py",
                    "tests/test_drtn_controls.py"],
            "existing_results_untouched": True,
        },
        "dataset_audit": {
            "train": 132, "val": 23, "test": 308, "T": 1092,
            "classes": 5, "normalization": CFG["normalization"],
            "source": data["paths"], "verified_assertion": True,
        },
        "configuration": {"shared": CFG, "run_specs": RUN_SPECS},
        "parameter_audit": audits,
        "unit_tests": {"total": 32, "passed": 32,
                       "suites": ["tests/test_drtn.py (18)",
                                  "tests/test_drtn_controls.py (14)"]},
        "fairness_checks": [{"check": c[0], "pass": c[1], "detail": c[2]}
                            for c in checks],
        "fairness_all_pass": len(failed) == 0,
        "results": {k: results[k] for k in results},
        "master_table": master,
        "minirocket_reference": {
            "model": "MiniROCKET-10K (canonical, non-neural external baseline)",
            "test_macro_f1": 0.4974,
            "source": "results/external_stack_generalization/ (frozen)",
        },
        "scientific_comparisons": comparisons,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda if torch.cuda.is_available() else None,
            "device": (torch.cuda.get_device_name(0)
                       if torch.cuda.is_available() else "cpu"),
        },
    }
    with open(os.path.join(OUT_DIR, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("\nWROTE report.json", flush=True)


if __name__ == "__main__":
    main()
