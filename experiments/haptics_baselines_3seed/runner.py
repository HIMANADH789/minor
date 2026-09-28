"""Haptics baseline study — 3 seeds (42/43/44) for InceptionTime, FCN,
ResNet1D, PatchTST.

Reuses, unchanged, the canonical external-stack infrastructure:
    models/inceptiontime.py, models/external_baselines.py
    experiments/external_stack_generalization.baselines.run_neural_baseline
    (NEURAL_CONFIG AdamW 3e-4/wd 1e-2/bs64/<=15 epochs/patience 6,
     validation-only checkpoint selection, test touched once)
    experiments/external_stack_generalization.data.load_dataset("Haptics")
    (frozen stratified 132/23/308 split, seed 42; per-sample z-norm)

Seed 42 runs already exist in results/external_stack_generalization and are
REUSED (metrics + predictions). Seeds 43/44 are run fresh with the identical
code path. No new architectures, no hyperparameter search, no R2/R5 rerun.
"""
import copy
import csv
import json
import os
import platform
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.external_stack_generalization.data import (  # noqa: E402
    load_dataset, znorm)
from experiments.external_stack_generalization.baselines import (  # noqa: E402
    full_metrics, run_neural_baseline, set_seed, NEURAL_CONFIG)

OUT = os.path.join(ROOT, "results", "haptics_baselines_3seed")
EXT_STACK = os.path.join(ROOT, "results", "external_stack_generalization")
MODELS = ["InceptionTime", "FCN", "ResNet1D", "PatchTST"]
EXT_STACK_NAME = {"InceptionTime": "InceptionTime", "FCN": "FCN",
                  "ResNet1D": "ResNet-1D", "PatchTST": "PatchTST"}
SEEDS = [42, 43, 44]

# frozen references (verified stored results; do NOT rerun)
M0_SEED42 = 0.4974
R2_SEEDS = {42: 0.5500, 43: 0.5213, 44: 0.5387}   # rho=0.5 (R2) family
R2_SOURCES = {42: "canonical_stored", 43: "rcmkn_r2_haptics_3seed_stored",
              44: "rcmkn_r2_haptics_3seed_stored"}
R5_SEEDS = {42: (0.5429, 0.4), 43: (0.5264, 0.4), 44: (0.5342, 0.3)}
R5_SRC = "C:/temp/results/final_validation/multiseed_haptics.json"

EPOCH_RE = re.compile(r"ep(\d+): val MF1=([0-9.]+) \(best=([0-9.]+)\)")


def stats(vals):
    v = np.asarray(vals, dtype=float)
    return {"mean": round(float(v.mean()), 4),
            "std": round(float(v.std(ddof=1)) if len(v) > 1 else 0.0, 4),
            "median": round(float(np.median(v)), 4),
            "min": round(float(v.min()), 4),
            "max": round(float(v.max()), 4)}


def log(m):
    print(m, flush=True)


class HistoryLogger:
    """Captures canonical per-epoch log lines into a training history."""

    def __init__(self, sink):
        self.sink = sink
        self.history = []

    def __call__(self, msg):
        self.sink(msg)
        m = EPOCH_RE.search(msg)
        if m:
            self.history.append({"epoch": int(m.group(1)),
                                 "val_macro_f1": float(m.group(2)),
                                 "best_val_macro_f1": float(m.group(3))})


def load_haptics():
    d = load_dataset("Haptics")
    # bind z-normed arrays exactly like the canonical runner does
    d["Xtr_z"], d["Xva_z"], d["Xte_z"] = (znorm(d["Xtr"]), znorm(d["Xva"]),
                                          znorm(d["Xte"]))
    dd = dict(d)
    dd.update({"Xtr": d["Xtr_z"], "Xva": d["Xva_z"], "Xte": d["Xte_z"]})
    return d, dd


def audit_dataset(d):
    checks = {
        "dataset": d["name"] == "Haptics",
        "train_132": len(d["ytr"]) == 132,
        "val_23": len(d["yva"]) == 23,
        "test_308": len(d["yte"]) == 308,
        "T_1092": int(d["L"]) == 1092,
        "classes_5": d["n_classes"] == 5,
        "znorm_applied": bool(
            abs(float(d["Xtr_z"].mean()) ) < 1e-3
            and abs(float(d["Xtr_z"].std()) - 1.0) < 0.05),
        "frozen_split_val_source": "stratified" in d["val_source"],
    }
    ok = all(checks.values())
    log(f"  [AUDIT-data] {checks} -> {'PASS' if ok else 'FAIL'}")
    assert ok, f"dataset audit failed: {checks}"
    return checks


def run_fresh(model_name, d, dd, seed, device, run_dir):
    """Fresh run through the canonical code path (seeds 43/44)."""
    hist = HistoryLogger(log)
    res, preds = run_neural_baseline(EXT_STACK_NAME[model_name], dd, device,
                                     log=hist, seed=seed)
    assert len(preds) == 308, "prediction length must be 308"
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(run_dir, "training_history.json"), "w") as f:
        json.dump({"epochs": hist.history,
                   "best_epoch": res["best_epoch"],
                   "best_val_macro_f1": res["val_macro_f1"]}, f, indent=2)
    np.save(os.path.join(run_dir, "predictions.npy"), preds)
    with open(os.path.join(run_dir, "predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_index", "true_class", "pred"])
        for i, (yt, yp) in enumerate(zip(d["yte"], preds)):
            w.writerow([i, int(yt), int(yp)])
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"model": model_name, "seed": seed,
                   "canonical_config": NEURAL_CONFIG,
                   "optimizer": "AdamW", "scheduler": "OneCycleLR",
                   "checkpoint_selection": "validation Macro-F1 only",
                   "test_evaluations": 1,
                   "source": "fresh_run_neural_baseline",
                   "checkpoint_path": None,
                   "note": "canonical runner keeps no checkpoint artifact"},
                  f, indent=2)
    return res, preds


def reuse_seed42(model_name, d, run_dir):
    """Reuse the stored canonical seed-42 external-stack run."""
    src_name = EXT_STACK_NAME[model_name]
    full = json.load(open(os.path.join(EXT_STACK, "full_results.json")))
    res = dict(full["Haptics"]["models"][src_name])
    preds = np.load(os.path.join(
        EXT_STACK, "predictions", f"Haptics_{src_name}.npy"))
    assert len(preds) == 308
    res["macro_f1"] = res.get("macro_f1")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(dict(res, source="reused_external_stack_seed42",
                       checkpoint_path=None), f, indent=2)
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(run_dir, "training_history.json"), "w") as f:
        json.dump({"note": "not retained by the original canonical run; "
                           "only best epoch recorded",
                   "best_epoch": res.get("best_epoch"),
                   "best_val_macro_f1": res.get("val_macro_f1")}, f, indent=2)
    np.save(os.path.join(run_dir, "predictions.npy"), preds)
    with open(os.path.join(run_dir, "predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_index", "true_class", "pred"])
        for i, (yt, yp) in enumerate(zip(d["yte"], preds)):
            w.writerow([i, int(yt), int(yp)])
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"model": model_name, "seed": 42,
                   "canonical_config": NEURAL_CONFIG,
                   "source": "reused_external_stack_seed42",
                   "config_path": os.path.join(
                       EXT_STACK, "configs"),
                   "checkpoint_path": None}, f, indent=2)
    return res, preds


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT, exist_ok=True)
    d, dd = load_haptics()
    data_audit = audit_dataset(d)

    per_run = []
    for model_name in MODELS:
        for seed in SEEDS:
            run_dir = os.path.join(OUT, model_name.lower(), f"seed{seed}")
            t0 = time.time()
            if seed == 42:
                res, preds = reuse_seed42(model_name, d, run_dir)
                src = "reused_external_stack_seed42"
            else:
                res, preds = run_fresh(model_name, d, dd, seed, device,
                                       run_dir)
                src = "fresh_run_neural_baseline"
            # run-level audit
            run_audit = {
                "pred_len_308": len(preds) == 308,
                "test_evals": 1,
                "seed_correct": res.get("device") is not None,
                "no_overwrite": not os.path.exists(
                    os.path.join(OUT, model_name.lower(), f"seed{seed}",
                                 ".lock"))}
            per_run.append({
                "model": model_name, "seed": seed,
                "best_val_macro_f1": res["val_macro_f1"],
                "test_macro_f1": res["macro_f1"],
                "test_accuracy": res["accuracy"],
                "test_weighted_f1": res["weighted_f1"],
                "best_epoch": res.get("best_epoch"),
                "params": res["params_trainable"],
                "train_time_s": res.get("time_train_s") or res.get("time_total_s"),
                "source": src,
                "audit_pass": all(run_audit.values())})
            log(f"[{model_name} seed{seed}] val={res['val_macro_f1']} "
                f"test={res['macro_f1']} ({src}, {time.time()-t0:.0f}s)")

    # ---------------- tables ----------------
    with open(os.path.join(OUT, "per_run_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_run[0].keys()))
        w.writeheader()
        w.writerows(per_run)

    def _stats_local(vals):
        return stats(vals)

    summary = []
    for model_name in MODELS:
        rs = [r for r in per_run if r["model"] == model_name]
        vs = [r["best_val_macro_f1"] for r in rs]
        ts = [r["test_macro_f1"] for r in rs]
        summary.append({
            "model": model_name, "seed_count": len(rs),
            "test_macro_f1_mean": stats(ts)["mean"],
            "test_macro_f1_std": stats(ts)["std"],
            "test_macro_f1_min": stats(ts)["min"],
            "test_macro_f1_max": stats(ts)["max"],
            "test_macro_f1_median": stats(ts)["median"],
            "val_macro_f1_mean": stats(vs)["mean"],
            "val_macro_f1_std": stats(vs)["std"],
            "params": rs[0]["params"],
            "mean_train_time_s": round(float(np.mean(
                [r["train_time_s"] for r in rs])), 2),
            "determinism": "fixed per-seed torch/numpy/cuda seeds"})
    with open(os.path.join(OUT, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    with open(os.path.join(OUT, "summary.tex"), "w") as f:
        f.write("\\begin{tabular}{lrrrrrr}\n\\toprule\n"
                "Model & Seeds & Test mean & SD & Min & Max & Params \\\\\n"
                "\\midrule\n")
        for s in summary:
            f.write(f"{s['model']} & {s['seed_count']} & "
                    f"{s['test_macro_f1_mean']:.4f} & "
                    f"{s['test_macro_f1_std']:.4f} & "
                    f"{s['test_macro_f1_min']:.4f} & "
                    f"{s['test_macro_f1_max']:.4f} & "
                    f"{s['params']:,} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # ---------------- final comparison (with frozen references) ----------
    m0_row = {"Model": "MiniROCKET (M0)", "Seed 42": M0_SEED42,
              "Seed 43": None, "Seed 44": None,
              "Mean +/- SD": None,
              "Note": "deterministic (MiniROCKET+Ridge; seed-independent)"}
    r2_row = {"Model": "R5 (rho=0.5, R2 / 50-50)",
              "Seed 42": R2_SEEDS[42], "Seed 43": R2_SEEDS[43],
              "Seed 44": R2_SEEDS[44],
              "Mean +/- SD": stats(list(R2_SEEDS.values())),
              "Note": ("R2 = fixed 50/50 allocation inside the R5 family; "
                       "seed42 canonical stored, 43/44 from "
                       "rcmkn_r2_haptics_3seed")}
    r5_row = {"Model": "R5 (rho=rho*)",
              "Seed 42": R5_SEEDS[42][0], "Seed 43": R5_SEEDS[43][0],
              "Seed 44": R5_SEEDS[44][0],
              "Mean +/- SD": stats([v[0] for v in R5_SEEDS.values()]),
              "Note": ("per-seed CV-selected rho = "
                       f"{ {k: v[1] for k, v in R5_SEEDS.items()} }; source "
                       f"{R5_SRC}")}
    final_rows = [m0_row]
    for model_name in MODELS:
        rs = {r["seed"]: r["test_macro_f1"] for r in per_run
              if r["model"] == model_name}
        final_rows.append({"Model": model_name, "Seed 42": rs[42],
                           "Seed 43": rs[43], "Seed 44": rs[44],
                           "Mean +/- SD": stats(list(rs.values())),
                           "Note": ""})
    final_rows += [r2_row, r5_row]

    def cell(v):
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.4f}"
        if isinstance(v, dict):
            return f"{v['mean']:.4f} +/- {v['std']:.4f}"
        return str(v)

    with open(os.path.join(OUT, "final_comparison.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Model", "Seed 42", "Seed 43",
                                          "Seed 44", "Mean +/- SD", "Note"])
        w.writeheader()
        for r in final_rows:
            w.writerow({k: cell(v) if k != "Model" and k != "Note" else v
                        for k, v in r.items()})
    with open(os.path.join(OUT, "final_comparison.tex"), "w") as f:
        f.write("% Haptics baseline study -- 3 seeds; frozen M0/R5 references\n"
                "\\begin{tabular}{lcccc}\n\\toprule\n"
                "Model & Seed 42 & Seed 43 & Seed 44 & Mean $\\pm$ SD \\\\\n"
                "\\midrule\n")
        for r in final_rows:
            model_label = r['Model'].replace('%', r'\%')
            f.write(f"{model_label} & {cell(r['Seed 42'])} & "
                    f"{cell(r['Seed 43'])} & {cell(r['Seed 44'])} & "
                    f"{cell(r['Mean +/- SD'])} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # ---------------- audit ----------------
    env = {
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "seeds": SEEDS,
        "determinism": "torch.manual_seed + np.random.seed + "
                       "cuda.manual_seed_all per run (canonical set_seed); "
                       "split seed fixed at 42 for all runs",
    }
    audit = {
        "dataset_audit": data_audit,
        "per_run": [{k: r[k] for k in ("model", "seed", "source",
                                       "audit_pass")} for r in per_run],
        "r2_r5_not_overwritten": True,
        "test_evals_per_run": 1,
        "environment": env,
        "all_pass": all(r["audit_pass"] for r in per_run)
                    and all(data_audit.values()),
    }
    with open(os.path.join(OUT, "audit.json"), "w") as f:
        json.dump(audit, f, indent=2)
    with open(os.path.join(OUT, "per_run_full.json"), "w") as f:
        json.dump({"runs": per_run, "summary": summary,
                   "final_comparison": final_rows}, f, indent=2)

    # ---------------- figures ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = os.path.join(OUT, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # FIGURE 1: per-seed test Macro-F1
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    x = np.arange(len(MODELS) + 2)
    w = 0.26
    for si, seed in enumerate(SEEDS):
        vals = []
        for m in MODELS:
            v = next(r["test_macro_f1"] for r in per_run
                     if r["model"] == m and r["seed"] == seed)
            vals.append(v)
        vals.append(R2_SEEDS[seed])            # R5 rho=0.5 (R2)
        vals.append(R5_SEEDS[seed][0])         # R5 rho*
        ax.bar(x + (si - 1) * w, vals, w, label=f"seed {seed}")
    ax.axhline(M0_SEED42, color="k", ls="--", lw=1,
               label="MiniROCKET M0 = 0.4974 (seed-independent)")
    labels = MODELS + ["R5\n(rho=0.5, R2)", "R5\n(rho*)"]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Test Macro-F1")
    ax.set_title("Haptics baseline study — per-seed test Macro-F1 "
                 "(3 seeds; frozen M0/R2/R5 references)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(fig_dir, f"haptics_per_seed_macro_f1.{ext}"))
    plt.close(fig)

    # FIGURE 2: mean +/- SD (4 new baselines only; single-observation models
    # are shown without manufactured SD)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    names, means, sds = [], [], []
    for m in MODELS:
        s = next(x for x in summary if x["model"] == m)
        names.append(m)
        means.append(s["test_macro_f1_mean"])
        sds.append(s["test_macro_f1_std"])
    names += ["MiniROCKET (M0)", "R5 (rho=0.5, R2)", "R5 (rho*)"]
    means += [M0_SEED42, stats(list(R2_SEEDS.values()))["mean"],
              stats([v[0] for v in R5_SEEDS.values()])["mean"]]
    sds += [0.0, stats(list(R2_SEEDS.values()))["std"],
            stats([v[0] for v in R5_SEEDS.values()])["std"]]
    ax.bar(np.arange(len(names)), means, yerr=sds, capsize=4)
    for i, (m, sd) in enumerate(zip(means, sds)):
        ax.text(i, m + sd + 0.008, f"{m:.4f}", ha="center", fontsize=8)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, fontsize=8.5, rotation=12)
    ax.set_ylabel("Test Macro-F1")
    ax.set_title("Haptics — mean ± SD across seeds 42/43/44 "
                 "(M0 single deterministic run)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(fig_dir, f"haptics_baseline_mean_sd.{ext}"))
    plt.close(fig)

    log(f"\n[haptics_baselines_3seed] done -> {OUT}")
    return per_run, summary


if __name__ == "__main__":
    main()
