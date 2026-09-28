"""HAPTICS FINAL 5-SEED -- Phase 6 aggregation.

Reads all audited artifacts and produces the final tables under
results/haptics_final_5seed/:
    PER_SEED_RESULTS.csv, AGGREGATED_RESULTS.csv, FINAL_RESULTS.csv,
    BASELINE_AUDIT.csv, SANITY_CHECKS.csv, CONFIG_AUDIT.csv,
    status_matrix.csv, FINAL_TABLE.md, FINAL_REPORT.md

Sources (read-only):
    MR        : results/haptics_final_5seed/phase2_mr.json (+ gate)
    HERAMBA   : results/rcmkn_r2_haptics_3seed/seed{42,43,44}/result.json
                + results/haptics_final_5seed/heramba_seed{45,46}/result.json
    Baselines : results/baseline_audit/retrained_haptics/
                (selection/, seed{43..46}/, all_runs.json)
    Original collapsed baselines (audit trail):
                results/haptics_baselines_3seed/*/
Metric: test Macro-F1 (primary); val Macro-F1 reported separately.
Paired stats: per-seed HERAMBA - MR differences (descriptive only, n=5).
"""
import csv
import glob
import json
import os
from collections import Counter

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", ".."))
OUT = os.path.join(ROOT, "results", "haptics_final_5seed")
RETRAIN = os.path.join(ROOT, "results", "baseline_audit", "retrained_haptics")
OLD_BASE = os.path.join(ROOT, "results", "haptics_baselines_3seed")
SEEDS = [42, 43, 44, 45, 46]
BASELINES = ["InceptionTime", "FCN", "ResNet1D", "PatchTST"]


def log(m):
    print(m, flush=True)


def jload(p):
    with open(p) as f:
        return json.load(f)


def collect_mr():
    d = jload(os.path.join(OUT, "phase2_mr.json"))
    gate = jload(os.path.join(OUT, "mr_seed42_gate.json"))
    rows = []
    for r in d["rows"]:
        rows.append({"method": "MiniRocket", "seed": r["seed"],
                     "val_macro_f1": r["val_macro_f1"],
                     "test_macro_f1": r["test_macro_f1"],
                     "status": "REUSED_DETERMINISTIC",
                     "source": "canonical M0 convention; gated rerun "
                               f"(gate_pass={gate['pass']})",
                     "config": "MiniROCKET rs=42 (train-fit), "
                               "RidgeCV logspace(-4,4,20) on train+val"})
    return rows, gate


def collect_heramba():
    rows = []
    for seed in (42, 43, 44):
        r = jload(os.path.join(ROOT, "results", "rcmkn_r2_haptics_3seed",
                               f"seed{seed}", "result.json"))["results"]
        rows.append({"method": "HERAMBA", "seed": seed,
                     "val_macro_f1": r["val_macro_f1"],
                     "test_macro_f1": r["test_macro_f1"],
                     "status": "REUSED",
                     "source": f"results/rcmkn_r2_haptics_3seed/seed{seed}",
                     "config": "R2 [G||H] fixed rho=0.5, audited 3-seed core"})
    for seed in (45, 46):
        r = jload(os.path.join(OUT, f"heramba_seed{seed}",
                               "result.json"))["results"]
        rows.append({"method": "HERAMBA", "seed": seed,
                     "val_macro_f1": r["val_macro_f1"],
                     "test_macro_f1": r["test_macro_f1"],
                     "status": "NEW_RUN_AUDITED_PATH",
                     "source": f"results/haptics_final_5seed/"
                               f"heramba_seed{seed}",
                     "config": "R2 [G||H] fixed rho=0.5, audited 3-seed core"})
    return rows


def collect_baselines():
    """Retrained baselines: seed 42 = selected config's selection run;
    seeds 43-46 = frozen-config runs."""
    all_runs = jload(os.path.join(RETRAIN, "all_runs.json"))
    selected = jload(os.path.join(RETRAIN, "selected_configs.json"))
    # selection runs: name -> per-config runs
    sel_runs = {}
    for r in all_runs:
        if "selection" in str(r.get("_run_dir", "")):
            sel_runs.setdefault(r["model"], []).append(r)
    # fallback: scan run dirs
    for name in BASELINES:
        for cfg_dir in sorted(glob.glob(os.path.join(
                RETRAIN, name, "selection", "*"))):
            mp = os.path.join(cfg_dir, "metrics.json")
            if os.path.exists(mp):
                sel_runs.setdefault(name, []).append(jload(mp))
    rows, audits = [], []
    for name in BASELINES:
        cfg = selected[name]
        got42 = None
        for r in sel_runs.get(name, []):
            if r.get("config") == cfg or (
                    r.get("lr") == cfg["lr"] and r.get("wd") == cfg["wd"]):
                got42 = r if got42 is None or (
                    r.get("val_macro_f1", -1) > got42.get("val_macro_f1", -1)
                ) else got42
        rows.append({"method": name, "seed": 42,
                     "val_macro_f1": got42.get("val_macro_f1"),
                     "test_macro_f1": got42.get("macro_f1"),
                     "status": "RETRAIN_SELECTION_RUN",
                     "source": os.path.join(
                         "results/baseline_audit/retrained_haptics", name,
                         "selection"),
                     "config": f"lr={cfg['lr']}, wd={cfg['wd']} (val-only "
                               "grid seed 42, frozen)"})
        for seed in (43, 44, 45, 46):
            mp = os.path.join(RETRAIN, name, f"seed{seed}", "metrics.json")
            if not os.path.exists(mp):
                rows.append({"method": name, "seed": seed,
                             "val_macro_f1": None, "test_macro_f1": None,
                             "status": "MISSING", "source": mp,
                             "config": f"lr={cfg['lr']}, wd={cfg['wd']}"})
                continue
            r = jload(mp)
            rows.append({"method": name, "seed": seed,
                         "val_macro_f1": r.get("val_macro_f1"),
                         "test_macro_f1": r.get("macro_f1"),
                         "status": "RETRAIN_FROZEN",
                         "source": os.path.join(
                             "results/baseline_audit/retrained_haptics",
                             name, f"seed{seed}"),
                         "config": f"lr={cfg['lr']}, wd={cfg['wd']} (frozen)"})
    return rows


def old_baseline_sanity():
    """Audit trail of the original collapsed baseline runs."""
    rows = []
    for name in BASELINES:
        for seed in (42, 43, 44):
            mp = os.path.join(OLD_BASE, name.lower(), f"seed{seed}",
                              "metrics.json")
            pp = os.path.join(OLD_BASE, name.lower(), f"seed{seed}",
                              "predictions.npy")
            if not os.path.exists(mp):
                continue
            m = jload(mp)
            n_classes_pred, dist = 0, {}
            if os.path.exists(pp):
                p = np.load(pp)
                c = Counter(p.tolist())
                n_classes_pred = len(c)
                dist = {str(k): int(v) for k, v in sorted(c.items())}
            max_cls_frac = (max(dist.values()) / sum(dist.values())
                            if dist else None)
            rows.append({
                "baseline": name, "seed": seed,
                "old_test_macro_f1": m.get("macro_f1"),
                "best_epoch": m.get("best_epoch"),
                "n_pred_classes": n_classes_pred,
                "max_class_fraction": round(max_cls_frac, 4) if
                max_cls_frac is not None else None,
                "single_class_collapse": n_classes_pred == 1,
                "near_collapse": bool(max_cls_frac and max_cls_frac >= 0.95),
                "verdict": "INVALID_FOR_BENCHMARK (documented collapse)",
                "pred_dist": json.dumps(dist)})
    return rows


def write_csv(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    log(f"  wrote {os.path.relpath(path, ROOT)} ({len(rows)} rows)")


def stats(vals):
    v = np.asarray([x for x in vals if x is not None], dtype=float)
    if len(v) == 0:
        return None, None
    return round(float(v.mean()), 4), round(float(v.std(ddof=1)), 4) \
        if len(v) > 1 else 0.0


def main():
    log("PHASE 6 -- aggregation")
    mr_rows, mr_gate = collect_mr()
    her_rows = collect_heramba()
    base_rows = collect_baselines()
    all_rows = mr_rows + her_rows + base_rows

    # ---------------- PER_SEED_RESULTS.csv ----------------
    write_csv(os.path.join(OUT, "PER_SEED_RESULTS.csv"), all_rows,
              ["method", "seed", "val_macro_f1", "test_macro_f1", "status",
               "source", "config"])

    # ---------------- sanity checks on predictions ----------------
    sanity = []
    her_pred_gates = {}
    for seed in (45, 46):
        p = np.load(os.path.join(OUT, f"heramba_seed{seed}",
                                 "test_predictions.npy"))
        her_pred_gates[seed] = int(len(set(p.tolist())))
    for r in base_rows:
        if r["status"].startswith("RETRAIN"):
            mp = None
            if r["seed"] == 42:
                cands = glob.glob(os.path.join(RETRAIN, r["method"],
                                               "selection", "*",
                                               "predictions.npy"))
                cands = [c for c in cands
                         if jload(os.path.join(os.path.dirname(c),
                                               "metrics.json")).get(
                             "config") == jload(os.path.join(
                                 RETRAIN, f"{r['method']}_selected_config.json"
                             ))["selected_config"]]
                mp = cands[0] if cands else None
            else:
                cand = os.path.join(RETRAIN, r["method"], f"seed{r['seed']}",
                                    "predictions.npy")
                mp = cand if os.path.exists(cand) else None
            if mp and r["test_macro_f1"] is not None:
                p = np.load(mp)
                c = Counter(p.tolist())
                sanity.append({
                    "method": r["method"], "seed": r["seed"],
                    "n_test": int(len(p)), "n_pred_classes": len(c),
                    "max_class_fraction": round(max(c.values()) / len(p), 4),
                    "single_class": len(c) == 1,
                    "explains_collapse": True})
    for seed in (45, 46):
        p = np.load(os.path.join(OUT, f"heramba_seed{seed}",
                                 "test_predictions.npy"))
        c = Counter(p.tolist())
        sanity.append({"method": "HERAMBA", "seed": seed, "n_test": len(p),
                       "n_pred_classes": len(c),
                       "max_class_fraction": round(max(c.values()) / len(p),
                                                   4),
                       "single_class": len(c) == 1, "explains_collapse": True})
    write_csv(os.path.join(OUT, "SANITY_CHECKS.csv"), sanity,
              ["method", "seed", "n_test", "n_pred_classes",
               "max_class_fraction", "single_class", "explains_collapse"])

    # ---------------- baseline audit (old runs) ----------------
    old_rows = old_baseline_sanity()
    write_csv(os.path.join(OUT, "BASELINE_AUDIT.csv"), old_rows,
              ["baseline", "seed", "old_test_macro_f1", "best_epoch",
               "n_pred_classes", "max_class_fraction", "single_class_collapse",
               "near_collapse", "verdict", "pred_dist"])

    # ---------------- config audit ----------------
    cfg_rows = [
        {"method": "MiniRocket",
         "config": "aeon MiniROCKET random_state=42 fit train-only; "
                   "RidgeClassifierCV alphas=logspace(-4,4,20) fit train+val; "
                   "deterministic -> one run reused across seeds 42-46",
         "protocol": "per-sample z-norm before transform; val MF1 reported; "
                     "single test evaluation"},
        {"method": "HERAMBA",
         "config": "R2 family [G||H]: G=4998 MiniROCKET global (rs=42 fixed), "
                   "H=4998 occupancy-weighted regime heterogeneity, learned "
                   "context (SSL+VQ K=8) retrained per outer seed",
         "protocol": "per-seed context; RidgeCV on train+val; single test "
                     "evaluation; audits 1-17 embedded"},
        {"method": "Baselines",
         "config": "architectures verbatim (InceptionTime/FCN/ResNet1D/"
                   "PatchTST); BS=16, max 100 epochs, patience 25, OneCycleLR, "
                   "grad clip 1.0, CE; LR/WD grid {3e-4,1e-3}x{1e-2,1e-4} "
                   "selected on VALIDATION only at seed 42, frozen for 43-46",
         "protocol": "per-sample z-norm; checkpoint = best val MF1; single "
                     "test evaluation; no test-based tuning"},
    ]
    write_csv(os.path.join(OUT, "CONFIG_AUDIT.csv"), cfg_rows,
              ["method", "config", "protocol"])

    # ---------------- aggregation ----------------
    methods = ["MiniRocket", "HERAMBA"] + BASELINES
    agg, tbl_rows = [], []
    for m in methods:
        per = {s: None for s in SEEDS}
        vals, status = [], {}
        for r in all_rows:
            if r["method"] == m:
                per[r["seed"]] = r["test_macro_f1"]
                vals.append(r["test_macro_f1"])
                status[r["seed"]] = r["status"]
        mu, sd = stats(vals)
        n_valid = sum(1 for v in vals if v is not None)
        agg.append({"method": m, "n_valid_seeds": n_valid,
                    "mean_test_macro_f1": mu, "std_test_macro_f1": sd,
                    "per_seed": per})
        tbl_rows.append({"method": m, **{f"seed{s}": per[s] for s in SEEDS},
                         "mean": mu, "std": sd,
                         "all_valid": n_valid == 5})
    write_csv(os.path.join(OUT, "AGGREGATED_RESULTS.csv"),
              [{"method": a["method"],
                **{f"test_f1_seed{s}": a["per_seed"][s] for s in SEEDS},
                "mean": a["mean_test_macro_f1"],
                "std": a["std_test_macro_f1"],
                "n_valid": a["n_valid_seeds"]} for a in agg],
              ["method"] + [f"test_f1_seed{s}" for s in SEEDS]
              + ["mean", "std", "n_valid"])

    # ---------------- paired MR vs HERAMBA ----------------
    mr_by = {r["seed"]: r["test_macro_f1"] for r in mr_rows}
    her_by = {r["seed"]: r["test_macro_f1"] for r in her_rows}
    diffs = [round(her_by[s] - mr_by[s], 4) for s in SEEDS]
    paired = {"seeds": SEEDS, "heramaba_minus_mr": diffs,
              "mean_diff": round(float(np.mean(diffs)), 4),
              "median_diff": round(float(np.median(diffs)), 4),
              "std_diff": round(float(np.std(diffs, ddof=1)), 4),
              "note": "descriptive only; n=5 seeds, no significance claim"}
    with open(os.path.join(OUT, "PAIRED_MR_VS_HERAMBA.json"), "w") as f:
        json.dump(paired, f, indent=2)

    # ---------------- status matrix ----------------
    st = {a["method"]: a for a in agg}
    with open(os.path.join(OUT, "status_matrix.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Method"] + [str(s) for s in SEEDS] + ["Final"])
        for m in methods:
            cells = []
            for s in SEEDS:
                v = st[m]["per_seed"][s]
                cells.append("VALID" if v is not None else "MISSING")
            w.writerow([m] + cells +
                       ["COMPLETE" if all(c == "VALID" for c in cells)
                        else "INCOMPLETE"])

    # ---------------- FINAL_RESULTS.csv / FINAL_TABLE.md ----------------
    write_csv(os.path.join(OUT, "FINAL_RESULTS.csv"), all_rows,
              ["method", "seed", "val_macro_f1", "test_macro_f1", "status",
               "source", "config"])
    lines = ["# Haptics final 5-seed benchmark (test Macro-F1)", "",
             "| Method | " + " | ".join(f"Seed {s}" for s in SEEDS)
             + " | Mean +- Std |", "|" + "---|" * 8]
    for m in methods:
        a = next(x for x in agg if x["method"] == m)
        cells = ["-" if a["per_seed"][s] is None
                 else f"{a['per_seed'][s]:.4f}" for s in SEEDS]
        lines.append(f"| {m} | " + " | ".join(cells)
                     + f" | {a['mean_test_macro_f1']:.4f} +- "
                       f"{a['std_test_macro_f1']:.4f} |")
    lines += ["", "# Validation Macro-F1", "",
              "| Method | " + " | ".join(f"Seed {s}" for s in SEEDS) + " |",
              "|" + "---|" * 7]
    for m in methods:
        vv = {r["seed"]: r["val_macro_f1"] for r in all_rows
              if r["method"] == m}
        lines.append(f"| {m} | " + " | ".join(
            "-" if vv[s] is None else f"{vv[s]:.4f}" for s in SEEDS) + " |")
    lines += ["", f"MR gate: observed {mr_gate['observed']} vs canonical "
              f"{mr_gate['reference']} (tol {mr_gate['tolerance']}) -> "
              f"{'PASS' if mr_gate['pass'] else 'FAIL'}",
              "", "Paired HERAMBA - MR per seed: " + str(paired[
                  "heramaba_minus_mr"]),
              f"mean {paired['mean_diff']:+.4f}, median "
              f"{paired['median_diff']:+.4f}, std {paired['std_diff']:.4f} "
              "(descriptive, n=5)"]
    with open(os.path.join(OUT, "FINAL_TABLE.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    log("  wrote FINAL_TABLE.md")
    return agg, paired


if __name__ == "__main__":
    main()
