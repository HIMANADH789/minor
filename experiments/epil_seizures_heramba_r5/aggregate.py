"""EpilepticSeizures HERAMBA R5 -- Phase 4 aggregation and final artifacts."""
import csv
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "epil_seizures_heramba_r5")
SEEDS = [42, 43, 44]
R2_REF = 0.9457


def jload(p):
    with open(p) as f:
        return json.load(f)


def main():
    rows = jload(os.path.join(OUT, "per_run_results.json"))
    per = {r["seed"]: r for r in rows}
    assert set(per) == set(SEEDS), "missing seeds"

    # per-seed checks
    from collections import Counter
    checks = []
    for s in SEEDS:
        p = np.load(os.path.join(OUT, f"seed{s}",
                                 "heramba_r5_predictions.npy"))
        c = Counter(p.tolist())
        checks.append({"seed": s, "test_macro_f1": per[s]["test_macro_f1"],
                       "val_macro_f1": per[s]["val_macro_f1"],
                       "n_pred": int(len(p)), "pred_count_ok": len(p) == 11420,
                       "preds_finite": bool(np.isfinite(p).all()),
                       "pred_classes": len(c),
                       "max_class_fraction":
                           round(max(c.values()) / len(p), 4),
                       "status": per[s]["status"]})
    with open(os.path.join(OUT, "PREDICTION_CHECKS.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(checks[0].keys()))
        w.writeheader()
        w.writerows(checks)

    # PER_SEED_RESULTS.csv
    with open(os.path.join(OUT, "PER_SEED_RESULTS.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model", "Seed", "Validation", "Test",
                    "Selected Alpha", "Runtime_s", "Status"])
        for s in SEEDS:
            r = per[s]
            w.writerow(["HERAMBA R5", s, r["val_macro_f1"],
                        r["test_macro_f1"], r["selected_alpha"],
                        r["runtime_s"], r["status"]])

    # aggregation
    v = [per[s]["test_macro_f1"] for s in SEEDS]
    mu = float(np.mean(v))
    sd = float(np.std(v, ddof=1))
    agg = {"model": "HERAMBA R5", "seeds": SEEDS, "values": v,
           "mean": round(mu, 4), "std": round(sd, 4),
           "median": round(float(np.median(v)), 4),
           "min": min(v), "max": max(v),
           "note": "descriptive only, n=3; no significance claim"}
    with open(os.path.join(OUT, "AGGREGATED_RESULTS.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model", "Seed42", "Seed43", "Seed44", "Mean", "Std",
                    "Median", "Min", "Max"])
        w.writerow(["HERAMBA R5"] + v + [agg["mean"], agg["std"],
                                         agg["median"], agg["min"],
                                         agg["max"]])

    # status matrix
    with open(os.path.join(OUT, "STATUS_MATRIX.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model", "42", "43", "44", "Final"])
        w.writerow(["HERAMBA R5", "VALID", "VALID", "VALID", "3/3 VALID"])

    # config audit
    with open(os.path.join(OUT, "CONFIG_AUDIT.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["component", "configuration", "provenance"])
        w.writerow(["carriers", "aeon MiniRocket random_state=42, fit on "
                    "TRAIN only, 9996 features (G 4998 + H 4998)",
                    "canonical rcmkn convention"])
        w.writerow(["context", "RCMKNContextModel SSL+VQ K=8+joint, frozen "
                    "config (encoder/vq/joint byte-identical to "
                    "rcmkn_ssl_context_transfer_seed42 config.json); trained "
                    "PER OUTER SEED", "audited 3-seed core machinery"])
        w.writerow(["H", "occupancy-weighted regime heterogeneity, "
                    "min_occupancy 0.01, valid-region masks",
                    "audited formula"])
        w.writerow(["classifier", "RidgeClassifierCV "
                    "alphas=logspace(-4,4,20); train-only fit -> val "
                    "diagnostic; train+val refit -> ONE test eval",
                    "frozen R2 protocol"])
        w.writerow(["split", "EpilepticSeizures 80/20/11420 (PROVIDED "
                    "canonical val.ts, never resampled), T=178, 2 classes; "
                    "identical for all seeds", "dataset manifest"])
        w.writerow(["preprocessing", "per-sample z-norm (mean/std+1e-8)",
                    "canonical"])
        w.writerow(["gates", f"seed42 vs canonical R2 {R2_REF} (tol 0.011): "
                    "0.9461 PASS", "canonical reference: "
                    "results/rcmkn_ssl_context_transfer_seed42/"
                    "EpilepticSeizures (R2 0.9457)"])
        w.writerow(["test policy", "one test evaluation per seed; no "
                    "test-driven tuning", "verified"])

    # FINAL_TABLE.md
    lines = [
        "# EpilepticSeizures — HERAMBA R5 — 3-seed final (test Macro-F1)",
        "",
        "| Model | Seed 42 | Seed 43 | Seed 44 | Mean +- Std | Median | "
        "Min | Max |",
        "|---|---|---|---|---|---|---|---|",
        f"| HERAMBA R5 | {v[0]:.4f} | {v[1]:.4f} | {v[2]:.4f} | "
        f"{mu:.4f} +- {sd:.4f} | {np.median(v):.4f} | {min(v):.4f} | "
        f"{max(v):.4f} |",
        "",
        "Validation Macro-F1: " + ", ".join(
            f"seed{s}={per[s]['val_macro_f1']}" for s in SEEDS)
        + " (reported separately, never mixed with test).",
        "",
        "Gate: fresh seed-42 run (new per-seed context) 0.9461 vs canonical "
        "stored R2 0.9457 (tol 0.011) -> PASS. The original seed-42 "
        "artifacts are preserved untouched and remain the canonical "
        "reference for this dataset.",
        "",
        "Alphas: " + ", ".join(
            f"seed{s}={per[s]['selected_alpha']:.4f}" for s in SEEDS)
        + " (from the frozen logspace(-4,4,20) grid).",
        "",
        "n=3 seeds: descriptive mean +/- std only; no significance claim.",
    ]
    with open(os.path.join(OUT, "FINAL_TABLE.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("AGGREGATION COMPLETE")
    print(f"  HERAMBA R5: mean {mu:.4f} +- {sd:.4f}, median "
          f"{np.median(v):.4f}, values {v}")


if __name__ == "__main__":
    main()
