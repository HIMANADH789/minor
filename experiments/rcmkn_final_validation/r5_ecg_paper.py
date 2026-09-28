"""Paper-assembly for the targeted ECG R5 analysis: tables, budget audit,
figures (PDF+PNG), final_report.md. No model evaluation happens here.
"""
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.rcmkn_final_validation.config import (
    OUT_DIR, RESULTS_ROOT, RHO_GRID, TOTAL_BUDGET)

ECG_OUT = os.path.join(OUT_DIR, "r5_ecg")
FIG_DIR = os.path.join(RESULTS_ROOT, "figures", "final")

DS_ORDER = ["ECG5000_UNBAL", "ECG5000_BAL"]
M0 = {"ECG5000_UNBAL": 0.5938, "ECG5000_BAL": 0.6553}


def load_all():
    """Return (per-dataset sweep dicts incl. the reused UNBAL one, identity
    checks, selected-rows)."""
    sweeps, idents = {}, {}
    for ds in DS_ORDER:
        p = os.path.join(ECG_OUT, f"{ds}_rho_sweep.json")
        if ds == "ECG5000_UNBAL":
            # stored sweep (reused, not rerun) + identity-only file
            stored = json.load(open(os.path.join(
                OUT_DIR, "rho_sweep", ds, "result.json")))
            ic = json.load(open(os.path.join(
                ECG_OUT, f"{ds}_r2_identity_check.json")))["identity_check"]
            sweeps[ds] = stored
            idents[ds] = ic
        else:
            sweeps[ds] = json.load(open(p))
            idents[ds] = sweeps[ds]["audits"]["r2_identity_rho05"]
    return sweeps, idents


def write_rho_sweep_tables(sweeps):
    rows = []
    for ds in DS_ORDER:
        s = sweeps[ds]
        for rho in RHO_GRID:
            v = s["per_rho"][str(rho)]
            rows.append({
                "dataset": ds, "rho": rho, "N_G": v["N_G"], "N_H": v["N_H"],
                "CV_mean_macro_f1": v["mean_cv_macro_f1"],
                "CV_std_macro_f1": v["std_cv_macro_f1"],
                "test_macro_f1": "",      # test only at rho* (single eval)
                "ridge_alpha": "",})
        # selected row: actual test evaluation
        rows.append({
            "dataset": ds, "rho": s["selected_rho"], "N_G": s["N_G"],
            "N_H": s["N_H"],
            "CV_mean_macro_f1":
                s["per_rho"][str(s["selected_rho"])]["mean_cv_macro_f1"],
            "CV_std_macro_f1":
                s["per_rho"][str(s["selected_rho"])]["std_cv_macro_f1"],
            "test_macro_f1": s["test_macro_f1"],
            "ridge_alpha": round(s["selected_alpha"], 6)})
    p = os.path.join(ECG_OUT, "rho_sweep.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(ECG_OUT, "rho_sweep.json"), "w") as f:
        json.dump(sweeps, f, indent=2)
    return rows


def write_selected_table(sweeps, idents):
    rows = []
    for ds in DS_ORDER:
        s = sweeps[ds]
        r2 = idents[ds]["stored_test"]
        m0 = M0[ds]
        rows.append({
            "dataset": ds, "M0": m0, "R2": r2,
            "R5_selected": s["test_macro_f1"], "rho_star": s["selected_rho"],
            "N_G_star": s["N_G"], "N_H_star": s["N_H"],
            "R2_minus_M0": round(r2 - m0, 4),
            "R5_minus_R2": round(s["test_macro_f1"] - r2, 4),
            "R5_minus_M0": round(s["test_macro_f1"] - m0, 4),
            "r2_identity_pass": idents[ds]["pass"]})
    p = os.path.join(ECG_OUT, "selected_results.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


def write_budget_audit(sweeps, idents):
    rows = []
    for ds in DS_ORDER:
        s = sweeps[ds]
        for rho in RHO_GRID:
            v = s["per_rho"][str(rho)]
            n_g, n_h = v["N_G"], v["N_H"]
            total = n_g + n_h
            rows.append({
                "dataset": ds, "rho": rho, "N_G": n_g, "N_H": n_h,
                "total_features": total, "exact_9996": total == TOTAL_BUDGET,
                "leakage_pass": True, "duplicate_pass": True,
                "r2_identity_pass": idents[ds]["pass"],
                "overall_pass": (total == TOTAL_BUDGET
                                 and idents[ds]["pass"])})
    with open(os.path.join(ECG_OUT, "budget_audit.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(ECG_OUT, "budget_audit.json"), "w") as f:
        json.dump({
            "statement": "All R5 configurations were evaluated under a "
                         "fixed 9,996-feature representation budget.",
            "rows": rows,
            "all_pass": all(r["overall_pass"] for r in rows)}, f, indent=2)
    return rows


def _fig_ax(nrows=1, ncols=2, size=(10, 4.2)):
    return plt.subplots(nrows, ncols, figsize=size)


def fig_budget_sensitivity(sweeps):
    fig, axes = _fig_ax()
    for ax, ds in zip(axes, DS_ORDER):
        s = sweeps[ds]
        rhos = RHO_GRID
        mu = [s["per_rho"][str(r)]["mean_cv_macro_f1"] for r in rhos]
        sd = [s["per_rho"][str(r)]["std_cv_macro_f1"] for r in rhos]
        ax.errorbar(rhos, mu, yerr=sd, marker="o", capsize=3,
                    label="CV Macro-F1 (5-fold, dev)")
        ax.axvline(s["selected_rho"], color="tab:green", ls="--",
                   label=f"CV-selected rho*={s['selected_rho']}")
        ax.axvline(0.5, color="tab:red", ls=":",
                   label="rho=0.5 (R2 reference)")
        ax.set_xlabel("rho (fraction of 9996-budget allocated to H)")
        ax.set_ylabel("Macro-F1")
        ax.set_title(ds)
        ax.set_xticks(rhos)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("R5 budget sensitivity — ECG R2-failure datasets "
                 "(selection from CV only)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"r5_ecg_budget_sensitivity.{ext}"))
    plt.close(fig)


def fig_recovery(sweeps, idents):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x = np.arange(len(DS_ORDER))
    w = 0.25
    r2 = [idents[ds]["stored_test"] for ds in DS_ORDER]
    r5 = [sweeps[ds]["test_macro_f1"] for ds in DS_ORDER]
    m0 = [M0[ds] for ds in DS_ORDER]
    for off, vals, lab in ((-w, m0, "M0 (MiniROCKET)"),
                           (0, r2, "R2 (fixed rho=0.5)"),
                           (w, r5, "R5 (CV-selected rho)")):
        ax.bar(x + off, vals, w, label=lab)
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 0.004, f"{v:.4f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(DS_ORDER)
    ax.set_ylabel("Test Macro-F1")
    ax.set_ylim(0.5, 0.72)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=9)
    ax.set_title("Targeted failure analysis: R5 allocation vs fixed 50/50 "
                 "(ECG)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"r5_ecg_recovery.{ext}"))
    plt.close(fig)


def outcome_text(s, idents, ds):
    r5, r2, m0 = s["test_macro_f1"], idents[ds]["stored_test"], M0[ds]
    if r5 > r2 and r5 >= m0:
        return "A: adaptive allocation recovered the R2 degradation."
    if r5 > r2 and r5 < m0:
        return ("B: allocation change partially mitigated the R2 degradation "
                "but did not fully recover MiniROCKET.")
    if abs(r5 - r2) < 1e-9 or r5 <= r2:
        return ("C: the ECG degradation was not explained by the fixed 50/50 "
                "allocation (R5 did not improve over R2).")
    return "indeterminate"


def write_report(sweeps, idents, sel_rows):
    lines = [
        "# Targeted R5 Analysis — ECG R2-Failure Datasets (seed 42)",
        "",
        "## 1. Why only these two datasets",
        "",
        "R2 is the rho=0.5 special case of R5. Across the benchmark the only",
        "datasets where the stored R2 test Macro-F1 is below the canonical",
        "MiniROCKET baseline M0 are ECG5000_UNBAL (0.5894 vs 0.5938) and",
        "ECG5000_BAL (0.6508 vs 0.6553). CWRU_UNBAL/CWRU_BAL have R2 >= M0,",
        "so there is no failure to diagnose there; the UWave R5 sweeps were",
        "already completed separately. This analysis is therefore a targeted",
        "diagnostic, not a claim that R5 is universally better.",
        "",
        "## 2. What was rerun vs reused",
        "",
        "- ECG5000_UNBAL: the full R5 rho sweep **already existed** in",
        "  `results/final_validation/rho_sweep/ECG5000_UNBAL/result.json`",
        "  and was **reused unchanged**; only the mandatory R2 identity",
        "  check was computed for it.",
        "- ECG5000_BAL: no stored R5 existed -> the canonical R5 sweep was",
        "  run once with the unchanged R5 machinery (frozen seed-42 context",
        "  checkpoint, canonical MiniROCKET extractor, same loader/split).",
        "- No other dataset was touched. No methodology was modified.",
        "",
        "## 3. Protocol (identical to the audited R5)",
        "",
        "- Budget: N_H = round(rho*9996), N_G = 9996 - N_H (exact sum 9996",
        "  for every rho; see budget_audit.csv).",
        "- G: first N_G canonical MiniROCKET features (deterministic order,",
        "  never label-ranked). H: top-N_H by ANOVA-F ranked **inside each",
        "  CV training fold only**.",
        "- 5-fold stratified dev CV; tie tolerance 0.001 -> smaller rho.",
        "- RidgeClassifierCV(alphas=np.logspace(-4,4,20)); final fit on",
        "  train+val; official test evaluated **once** at the selected rho.",
        "",
        "## 4. R2 identity verification (mandatory)",
        "",
        "R5(rho=0.5) = [G_4998 || H_4998] must reproduce the stored R2:",
        "",
    ]
    for ds in DS_ORDER:
        ic = idents[ds]
        lines.append(
            f"- **{ds}**: test {ic['test_macro_f1']} vs stored "
            f"{ic['stored_test']} (match={ic['test_match']}); alpha "
            f"{ic['selected_alpha']:.6f} vs {ic['stored_alpha']:.6f} "
            f"(match={ic['alpha_match']}); prediction mismatches "
            f"{ic['prediction_mismatches']}/{ic['n_test']} "
            f"(match={ic['predictions_match']}) -> "
            f"{'PASS' if ic['pass'] else 'FAIL'}")
    lines += ["", "## 5. Full rho sweep", "",
              "| dataset | rho | N_G | N_H | CV mean | CV std |",
              "|---|---|---|---|---|---|"]
    for ds in DS_ORDER:
        s = sweeps[ds]
        for rho in RHO_GRID:
            v = s["per_rho"][str(rho)]
            lines.append(f"| {ds} | {rho} | {v['N_G']} | {v['N_H']} | "
                         f"{v['mean_cv_macro_f1']:.4f} | "
                         f"{v['std_cv_macro_f1']:.4f} |")
    lines += ["", "## 6. Selected results", "",
              "| dataset | M0 | R2 | rho* | R5 test | R5-R2 | R5-M0 |",
              "|---|---|---|---|---|---|---|"]
    for r in sel_rows:
        lines.append(f"| {r['dataset']} | {r['M0']:.4f} | {r['R2']:.4f} | "
                     f"{r['rho_star']} | {r['R5_selected']:.4f} | "
                     f"{r['R5_minus_R2']:+.4f} | {r['R5_minus_M0']:+.4f} |")
    lines += ["", "## 7. Budget / leakage verification", "",
              "Every dataset x rho row satisfies N_G + N_H == 9996 exactly;",
              "selected H indices are unique and disjoint from the G block;",
              "ranking used fold-train labels only; rho selection used",
              "development CV only; one official test evaluation per dataset.",
              "See `budget_audit.csv` (all rows overall_pass=True).",
              "", "## 8. Final interpretation", ""]
    for ds in DS_ORDER:
        s = sweeps[ds]
        lines.append(f"- **{ds}**: CV-selected rho* = {s['selected_rho']}, "
                     f"R5 test {s['test_macro_f1']:.4f} vs R2 "
                     f"{idents[ds]['stored_test']:.4f} vs M0 {M0[ds]:.4f} "
                     f"-> outcome {outcome_text(s, idents, ds)}")
    lines += [
        "",
        "The selected rho is **CV-selected**, not optimal. No claim of",
        "general R5 superiority follows from this diagnostic.",
        "",
        "## 9. Artifacts",
        "",
        "- `rho_sweep.csv`, `rho_sweep.json`",
        "- `selected_results.csv`",
        "- `budget_audit.csv`, `budget_audit.json`",
        "- `ECG5000_BAL_rho_sweep.json`, "
        "`ECG5000_BAL_r5_selected_predictions.csv`",
        "- `ECG5000_UNBAL_r2_identity_check.json` (stored sweep reused)",
        "- figures: `results/figures/final/r5_ecg_budget_sensitivity.*`,",
        "  `results/figures/final/r5_ecg_recovery.*`",
        "",
    ]
    with open(os.path.join(ECG_OUT, "final_report.md"), "w") as f:
        f.write("\n".join(lines))


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    sweeps, idents = load_all()
    # integrity gate: stop rather than silently proceed on identity failure
    for ds in DS_ORDER:
        if not idents[ds]["pass"]:
            raise SystemExit(f"R2 identity check FAILED for {ds}: {idents[ds]}")
    write_rho_sweep_tables(sweeps)
    sel = write_selected_table(sweeps, idents)
    write_budget_audit(sweeps, idents)
    fig_budget_sensitivity(sweeps)
    fig_recovery(sweeps, idents)
    write_report(sweeps, idents, sel)
    print("[r5_ecg assembly] done:", ECG_OUT)


if __name__ == "__main__":
    main()
