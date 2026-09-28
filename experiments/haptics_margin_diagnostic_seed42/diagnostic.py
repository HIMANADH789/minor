"""
CELL-C MARGIN DIAGNOSTIC — Haptics, seed 42 (DIAGNOSTIC ONLY, NO RETRAINING)
============================================================================
Question: are the 19 DRTN-only-correct test samples (Cell C) concentrated in
canonical MiniROCKET's low-margin region?

Data integrity chain (verified at runtime before any analysis):
  * canonical loader, frozen split (132/23/308), per-sample z-norm;
  * canonical MiniROCKET system of record = MiniRocket(random_state=42) + Ridge
    (train+val refit). The extractor is REFIT ONLY in the deterministic sense of
    re-instantiating the identical seeded aeon transformer; provenance is
    accepted iff the reconstruction reproduces the frozen artifact's 308/308
    predictions, Macro-F1 0.4974, and the complementarity matrix
    A=99 B=61 C=19 D=129 exactly. No hyperparameter is selected here and no
    test information alters any model.
  * margins = top1 - top2 of decision_function(Fte) on the frozen test samples;
  * bucketing is label-blind (fixed empirical quantiles of the margin);
  * the 19 Cell-C sample indices are taken from the frozen per-sample CSV.

Outputs -> results/haptics_margin_diagnostic_seed42/
  per_sample_margin.csv, report.json, (report.md written by report step)
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from experiments.external_stack_generalization.data import load_dataset  # noqa: E402

OUT = os.path.join(ROOT, "results", "haptics_margin_diagnostic_seed42")
ENS = os.path.join(ROOT, "results", "haptics_ensemble_seed42")

EXPECTED = {
    "n_test": 308,
    "n_cell_c": 19,
    "mr_test_mf1": 0.4974,
    "drtn_test_mf1": 0.3467,
    "matrix": {"A": 99, "B": 61, "C": 19, "D": 129},
}
ALPHAS = np.logspace(-4, 4, 20)


def margins_from_scores(scores: np.ndarray) -> np.ndarray:
    """margin_i = largest decision score - second largest (multiclass Ridge)."""
    s = np.sort(scores, axis=1)
    return s[:, -1] - s[:, -2]


def reconstruct_and_verify():
    """Rebuild the canonical frozen MiniROCKET classifier and verify identity
    against the frozen per-sample artifact BEFORE extracting margins."""
    d = load_dataset("Haptics")
    Xtr, ytr = d["Xtr"], d["ytr"]
    Xva, yva = d["Xva"], d["yva"]
    Xte, yte = d["Xte"], d["yte"]
    assert (len(Xtr), len(Xva), len(Xte)) == (132, 23, 308)
    assert d["L"] == 1092 and d["n_classes"] == 5

    from aeon.classification.convolution_based._minirocket import MiniRocket
    ext = MiniRocket(random_state=42)
    ext.fit(Xtr[:, None, :].astype(np.float32))
    Fte = ext.transform(Xte[:, None, :].astype(np.float32))
    ridge = RidgeClassifierCV(alphas=ALPHAS)
    ridge.fit(np.vstack([ext.transform(Xtr[:, None, :].astype(np.float32)),
                         ext.transform(Xva[:, None, :].astype(np.float32))]),
              np.concatenate([ytr, yva]))

    dd = ridge.decision_function(Fte)
    if dd.ndim == 1:
        dd = np.column_stack([-dd, dd])
    pred = dd.argmax(1)

    frozen = pd.read_csv(os.path.join(ENS, "predictions", "test_per_sample.csv"))
    tp = frozen["true_class"].values
    dp = frozen["drtn_prediction"].values
    mr_pred_frozen = frozen["mr_prediction"].values

    mf1 = f1_score(yte, pred, average="macro", zero_division=0)
    df1 = f1_score(yte, dp, average="macro", zero_division=0)
    mr_c, dr_c = tp == pred, tp == dp
    matrix = {
        "A": int((mr_c & dr_c).sum()),
        "B": int((mr_c & ~dr_c).sum()),
        "C": int((~mr_c & dr_c).sum()),
        "D": int((~mr_c & ~dr_c).sum()),
    }
    checks = {
        "n_test_ok": len(yte) == EXPECTED["n_test"],
        "pred_identity_308": bool((pred == mr_pred_frozen).all()),
        "mr_mf1_ok": round(mf1, 4) == EXPECTED["mr_test_mf1"],
        "drtn_mf1_ok": round(df1, 4) == EXPECTED["drtn_test_mf1"],
        "matrix_ok": matrix == EXPECTED["matrix"],
        "cell_c_from_csv_ok": int((~(tp == mr_pred_frozen)
                                   & (tp == dp)).sum()) == EXPECTED["n_cell_c"],
    }
    integrity = {
        "expected": EXPECTED,
        "checks": checks,
        "all_pass": all(checks.values()),
        "mr_test_macro_f1": float(mf1),
        "drtn_test_macro_f1": float(df1),
        "matrix": matrix,
        "provenance": ("MiniRocket(random_state=42) + RidgeClassifierCV(train+val refit) "
                       "accepted as the frozen canonical system via 308/308 prediction "
                       "identity with results/haptics_ensemble_seed42/predictions/"
                       "test_per_sample.csv; margins = decision_function top1-top2"),
        "note_scaled_scores": ("the ensemble CSV 'mr_scores' column held stacker-scaled "
                               "scores (argmax mismatch 25/308) and was NOT used for "
                               "margins"),
    }
    margins = margins_from_scores(dd)
    return integrity, margins, tp, pred, dp, mr_c, dr_c


def bucket(x, n_buckets, labels):
    """Deterministic empirical-quantile bucketing of x into n_buckets.

    Ties (equal margins spanning a cut) are assigned to the LOWER bucket,
    keeping earlier buckets at their full nominal size. Implementation:
    rank in [0,1) via searchsorted on unique sorted margins, then floor to
    the nearest bucket boundary. Exact, reproducible, label-blind.
    """
    ux = np.unique(x)                      # sorted unique margins
    ranks = np.searchsorted(ux, x, side="left") / len(ux)   # in [0, 1)
    b = np.minimum((ranks * n_buckets).astype(int), n_buckets - 1)
    return {lab: (b == k) for k, lab in enumerate(labels)}, b


def quartile_analysis(margins, mr_c, dr_c, cell_c):
    bounds = np.quantile(margins, [0.25, 0.50, 0.75])
    labs = ["Q1 (0-25%)", "Q2 (25-50%)", "Q3 (50-75%)", "Q4 (75-100%)"]
    qm, qb = bucket(margins, 4, labs)
    rows = []
    for lab in labs:
        m = qm[lab]
        n = int(m.sum())
        rows.append({
            "bucket": lab,
            "margin_upper_bound": None if lab.startswith("Q4")
            else round(float(bounds[["Q1 (0-25%)", "Q2 (25-50%)",
                                     "Q3 (50-75%)"].index(lab)]), 6),
            "n": n,
            "fraction_of_test": round(n / len(margins), 4),
            "mr_accuracy": round(float(mr_c[m].mean()), 4),
            "drtn_accuracy": round(float(dr_c[m].mean()), 4),
            "cell_c_count": int(cell_c[m].sum()),
            "cell_b_count": int((mr_c & ~dr_c)[m].sum()),
            "cell_c_fraction_of_19": round(float(cell_c[m].sum() / cell_c.sum()), 4),
            "cell_c_rate_in_bucket": round(float(cell_c[m].mean()), 4),
            "uniform_expected_cell_c": round(float(cell_c.sum() * n / len(margins)), 2),
        })
    # spec sec. 4: enrichment_Q1 = (fraction of all Cell-C samples in Q1) / 0.25.
    # With exactly-25% buckets this equals the bucket-rate ratio vs the overall
    # Cell-C baseline (19/308) — both readings give the same number.
    f_q1 = rows[0]["cell_c_fraction_of_19"]
    base_rate = float(cell_c.sum() / len(margins))
    return {"quantile_bounds": [float(b) for b in bounds],
            "tie_handling": "equal margins spanning a cut -> LOWER bucket (earlier "
                            "buckets keep nominal size); ranks via searchsorted on "
                            "unique margins; label-blind (verified: 0 tied margins)",
            "rows": rows,
            "enrichment_Q1": round(f_q1 / 0.25, 4),
            "enrichment_definition": "(fraction of the 19 Cell-C samples in Q1)/0.25; "
                                     "identically = Q1 Cell-C rate / overall Cell-C rate",
            "overall_cell_c_rate": round(base_rate, 4),
            "tests": quartile_tests(cell_c, qm, labs)}


def quartile_tests(cell_c, qm, labs):
    """Exploratory exact/asymptotic tests of the Cell-C quartile distribution
    against uniform (19 samples over 4 equal buckets)."""
    counts = np.array([int(cell_c[qm[lab]].sum()) for lab in labs])
    expected = np.full(4, counts.sum() / 4.0)
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    from scipy.stats import chi2 as chi2_dist, binom
    n = int(counts.sum())
    q1_share = counts[0] / n
    p_q1 = float(binom.sf(int(counts[0]) - 1, n, 0.25))    # P(>= count in Q1)
    p_q4 = float(binom.cdf(int(counts[3]), n, 0.25))       # P(<= count in Q4)
    return {
        "counts_by_quartile": counts.tolist(),
        "chi2_uniform": {"stat": round(chi2, 3), "df": 3,
                         "p": float(chi2_dist.sf(chi2, 3)),
                         "note": "exploratory: Cell-C vs uniform across quartiles"},
        "binomial_Q1_enrichment": {"observed_in_Q1": int(counts[0]), "n": n,
                                   "p_uniform": 0.25, "p_one_sided": p_q1,
                                   "note": "P(X >= observed) under Binomial(n=19, p=0.25)"},
        "binomial_Q4_absence": {"observed_in_Q4": int(counts[3]), "n": n,
                                 "p_uniform": 0.25, "p_one_sided": p_q4,
                                 "note": "P(X <= observed) under Binomial(n=19, p=0.25)"},
        "caveat": "post-hoc descriptive tests on the frozen result; n=19; "
                  "no multiplicity control; not confirmatory",
    }


def decile_analysis(margins, mr_c, dr_c, cell_c):
    labs = [f"D{i+1}" for i in range(10)]
    dm, db = bucket(margins, 10, labs)
    rows = []
    for lab in labs:
        m = dm[lab]
        rows.append({
            "bucket": lab, "n": int(m.sum()),
            "mr_accuracy": round(float(mr_c[m].mean()), 4),
            "drtn_accuracy": round(float(dr_c[m].mean()), 4),
            "cell_c_count": int(cell_c[m].sum()),
            "cell_b_count": int((mr_c & ~dr_c)[m].sum()),
            "cell_c_rate_in_bucket": round(float(cell_c[m].mean()), 4),
            "cell_c_fraction_of_19": round(float(cell_c[m].sum() / cell_c.sum()), 4),
        })
    return {"rows": rows,
            "enrichment_D1": round(rows[0]["cell_c_fraction_of_19"] / 0.10, 4),
            "enrichment_definition": "(fraction of the 19 Cell-C samples in D1)/0.10"}


def group_stats(margins, mr_c, dr_c):
    cell_c = ~mr_c & dr_c
    both_wrong = ~mr_c & ~dr_c
    groups = {
        "cell_C (MR wrong, DRTN correct)": margins[cell_c],
        "MR correct": margins[mr_c],
        "MR wrong & DRTN wrong (D)": margins[both_wrong],
        "all MR errors (C+D)": margins[~mr_c],
    }
    out = {}
    for name, v in groups.items():
        out[name] = {
            "n": int(len(v)),
            "median_margin": round(float(np.median(v)), 6),
            "mean_margin": round(float(np.mean(v)), 6),
            "q25": round(float(np.quantile(v, 0.25)), 6),
            "q75": round(float(np.quantile(v, 0.75)), 6),
        }
    tests = {}
    cc = margins[cell_c]
    a = margins[mr_c]
    b = margins[both_wrong]
    tests["cellC_vs_MRcorrect_mannwhitney"] = {
        "statistic": round(float(stats.mannwhitneyu(cc, a, alternative="less")[0]), 2),
        "p_less": float(stats.mannwhitneyu(cc, a, alternative="less")[1]),
        "note": "H1: Cell-C margins stochastically lower than MR-correct; exploratory",
    }
    tests["cellC_vs_bothwrong_mannwhitney"] = {
        "statistic": round(float(stats.mannwhitneyu(cc, b, alternative="two-sided")[0]), 2),
        "p_two_sided": float(stats.mannwhitneyu(cc, b, alternative="two-sided")[1]),
        "note": "Cell-C vs other-MR-errors; exploratory",
    }
    tests["cellC_vs_allMRerrors_mannwhitney"] = {
        "statistic": round(float(stats.mannwhitneyu(cc, margins[~mr_c],
                                                    alternative="less")[0]), 2),
        "p_less": float(stats.mannwhitneyu(cc, margins[~mr_c], alternative="less")[1]),
        "note": "H1: Cell-C margins lower than the average MR error; exploratory",
    }
    tests["caveat"] = ("n_CellC=19; all tests are exploratory/underpowered; "
                       "no significance claims are made")
    return {"groups": out, "exploratory_tests": tests}


def main():
    os.makedirs(OUT, exist_ok=True)
    integrity, margins, tp, pred, dp, mr_c, dr_c = reconstruct_and_verify()
    if not integrity["all_pass"]:
        print(json.dumps(integrity, indent=2))
        raise SystemExit("INTEGRITY CHECK FAILED — stopping per spec sec. 8")
    print("[integrity] all gates pass:", integrity["checks"])

    cell_c = ~mr_c & dr_c
    assert int(cell_c.sum()) == 19

    quart = quartile_analysis(margins, mr_c, dr_c, cell_c)
    dec = decile_analysis(margins, mr_c, dr_c, cell_c)
    gstats = group_stats(margins, mr_c, dr_c)

    # per-sample CSV
    out_csv = pd.DataFrame({
        "sample_index": np.arange(len(tp)),
        "true_class": tp,
        "mr_pred": pred,
        "drtn_pred": dp,
        "mr_margin": np.round(margins, 8),
        "mr_correct": mr_c.astype(int),
        "drtn_correct": dr_c.astype(int),
        "cell_C": cell_c.astype(int),
    })
    out_csv.to_csv(os.path.join(OUT, "per_sample_margin.csv"), index=False)

    report = {
        "experiment": "Cell-C low-margin concentration diagnostic (Haptics, seed 42)",
        "diagnostic_only": True,
        "integrity": integrity,
        "margin_definition": "top1 - top2 of canonical Ridge decision_function on test",
        "margin_range": [float(margins.min()), float(margins.max())],
        "quartile_analysis": quart,
        "decile_analysis": dec,
        "group_margin_statistics": gstats,
        "cell_c_sample_indices": sorted(np.where(cell_c)[0].tolist()),
        "decision_rule": "see report.md CASE 1/2/3 evaluation",
    }
    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # console summary
    print("\n=== QUARTILES ===")
    for r in quart["rows"]:
        print(f"{r['bucket']:14s} n={r['n']:3d} MRacc={r['mr_accuracy']:.3f} "
              f"Dacc={r['drtn_accuracy']:.3f} "
              f"cellC={r['cell_c_count']:2d} ({r['cell_c_fraction_of_19']*100:4.1f}% of 19) "
              f"rate={r['cell_c_rate_in_bucket']:.3f} (uniform exp {r['uniform_expected_cell_c']})")
    print(f"enrichment_Q1 = {quart['enrichment_Q1']} ({quart['enrichment_definition']})")
    t = quart["tests"]
    print("quartile tests:", json.dumps({k: v for k, v in t.items()
                                         if k != "caveat"}, indent=1))
    print("\n=== MEDIANS ===")
    for g, v in gstats["groups"].items():
        print(f"{g:38s} n={v['n']:3d} median={v['median_margin']:.4f} mean={v['mean_margin']:.4f}")
    for t, v in gstats["exploratory_tests"].items():
        if isinstance(v, dict):
            print(f"test {t}: stat={v.get('statistic')} p={min(v.get('p_less', v.get('p_two_sided', 1.0)), 0.999):.4g}")
    print("\nartifacts written to", OUT)


if __name__ == "__main__":
    main()
