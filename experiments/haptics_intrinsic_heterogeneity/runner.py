"""LABEL-FREE INTRINSIC TEMPORAL HETEROGENEITY AUDIT — Haptics, seed 42.

Question: does Haptics exhibit a nontrivial, stable, label-free measure
of temporal structural heterogeneity convertible into a deterministic
H-capacity signal BEFORE validation?

Pipeline:
  1. raw train signals only -> HI_spec (W=8, predeclared) + HI_var
  2. dataset index = median(HI_spec over train)     [predeclared]
  3. B_H = clip(HI_dataset / log 2, 0, 1)           [predeclared]
  4. stability across W=7/8/9 (diagnostic only)
  5. leakage audit (poisoned val/test signals)
  6. downstream demonstration ONLY: A) G 0.5037, B) G+H 0.5500,
     C) intrinsic-budget [G || H_budgeted] (label-free top-s quantile
     of train-only H-carrier strength; secondary, not optimized)

The index is frozen before any classifier is fitted; labels are touched
only for post-hoc figures.  Deterministic: no RNG in the index path.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from analysis.intrinsic_heterogeneity import (  # noqa: E402
    LOG2, W_PRIMARY, compute_dataset_index, compute_sample_spectral_heterogeneity,
    compute_sample_variance_heterogeneity, derive_intrinsic_h_budget,
    run_leakage_audit, stationarity_diagnostics, window_stability)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
CACHE = r"C:/temp/results"
OUT = os.path.join(ROOT, "results", "haptics_intrinsic_heterogeneity", "seed42")

SEED = 42
GATE_A, GATE_B = 0.5037, 0.55


def log(m):
    print(m, flush=True)


# ----------------------------------------------------------------------
def load_raw():
    from experiments.external_stack_generalization.data import load_dataset
    d = load_dataset("Haptics")
    Xtr = np.asarray(d["Xtr"], dtype=np.float64)
    Xva = np.asarray(d["Xva"], dtype=np.float64)
    Xte = np.asarray(d["Xte"], dtype=np.float64)
    assert Xtr.shape == (132, 1092) and Xva.shape == (23, 1092) \
        and Xte.shape == (308, 1092)
    assert np.isfinite(Xtr).all()
    return d, Xtr, Xva, Xte


# ----------------------------------------------------------------------
# label-free H-budget group selection (train-only, deterministic)
# ----------------------------------------------------------------------
def h_carrier_strength(H_train: np.ndarray, W: int = W_PRIMARY,
                       T: int = 1092) -> np.ndarray:
    """Per-H-feature intrinsic strength: the train-median normalized
    heterogeneity statistic itself (predeclared).  H (n_train, 4998)."""
    return np.median(H_train, axis=0)


def select_h_budget_groups(H_train: np.ndarray, B_H: float) -> dict:
    """Label-free capacity mapping: keep the strongest carriers ranked by
    TRAIN-median normalized heterogeneity; deterministic tie-break by
    ascending feature index.  No labels, no validation, no test."""
    s = h_carrier_strength(H_train)
    order = np.lexsort((np.arange(s.size), -s))     # strength desc, index asc
    total_groups = int(s.size)
    keep_q = int(np.ceil(B_H * total_groups))
    keep_hier = int(np.ceil(B_H * 8.0) * 8)          # 8 regime-slices/kernels
    keep_top = int(np.ceil(B_H * 64.0) * 64)         # 64 kernels/quantile-block
    mask_q = np.zeros(total_groups, dtype=bool)
    mask_q[order[:keep_q]] = True
    mask_hier = mask_q.copy()
    mask_hier[order[keep_hier:]] = False
    mask_top = mask_q.copy()
    mask_top[order[keep_top:]] = False
    return {
        "total_groups": total_groups,
        "ranking_statistic": "train-median normalized heterogeneity statistic",
        "tie_break": "ascending feature index (deterministic)",
        "kept_quantile": keep_q, "kept_hierarchy": keep_hier,
        "kept_block": keep_top,
        "mask_quantile": mask_q, "mask_hierarchy": mask_hier,
        "mask_block": mask_top,
    }


# ----------------------------------------------------------------------
def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "predictions"), exist_ok=True)
    log("=== INTRINSIC TEMPORAL HETEROGENEITY AUDIT — Haptics seed 42 ===")

    d, Xtr, Xva, Xte = load_raw()
    ytr, yva, yte = d["ytr"], d["yva"], d["yte"]

    # ---- 1-2: primary + secondary indices on TRAIN ---------------------
    log("computing HI_spec / HI_var on train (W=8)...")
    hi_tr = compute_sample_spectral_heterogeneity(Xtr, W_PRIMARY)
    hiv_tr = compute_sample_variance_heterogeneity(Xtr, W_PRIMARY)
    hi_dataset = compute_dataset_index(hi_tr)
    log(f"HI_dataset (median) = {hi_dataset:.6f}")

    # ---- 3: predeclared budget rule -------------------------------------
    budget = derive_intrinsic_h_budget(hi_dataset)
    B_H = budget["B_H"]
    log(f"B_H = {B_H:.6f}  band = {budget['band']}")

    # ---- 4: stability across W (diagnostic) ------------------------------
    log("window stability W=7/8/9 (diagnostic)...")
    stab = window_stability(Xtr, (7, 8, 9))

    # ---- 5: leakage audit -------------------------------------------------
    log("leakage audit (poisoned val/test)...")
    leak = run_leakage_audit(compute_sample_spectral_heterogeneity,
                             Xtr, Xva, Xte, W_PRIMARY)
    assert leak["val_poison_invariant"] and leak["test_poison_invariant"]
    log(f"leakage: val-invariant={leak['val_poison_invariant']} "
        f"test-invariant={leak['test_poison_invariant']}")

    # ---- supporting stationarity diagnostics (never used for budget) ----
    log("stationarity diagnostics (supporting only)...")
    stat_diag = stationarity_diagnostics(Xtr, max_n=None)

    # ---- post-hoc val/test indices (diagnostic only) ----------------------
    hi_va = compute_sample_spectral_heterogeneity(Xva, W_PRIMARY)
    hi_te = compute_sample_spectral_heterogeneity(Xte, W_PRIMARY)
    hiv_va = compute_sample_variance_heterogeneity(Xva, W_PRIMARY)
    hiv_te = compute_sample_variance_heterogeneity(Xte, W_PRIMARY)

    # ---- sample CSV --------------------------------------------------------
    with open(os.path.join(OUT, "sample_heterogeneity.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "sample_id", "HI_spec", "HI_var", "label"])
        for i, (a, b) in enumerate(zip(hi_tr, hiv_tr)):
            w.writerow(["train", i, round(float(a), 6), round(float(b), 4),
                        int(ytr[i])])
        for i, (a, b) in enumerate(zip(hi_va, hiv_va)):
            w.writerow(["val", i, round(float(a), 6), round(float(b), 4),
                        int(yva[i])])
        for i, (a, b) in enumerate(zip(hi_te, hiv_te)):
            w.writerow(["test", i, round(float(a), 6), round(float(b), 4),
                        int(yte[i])])

    # ---- summary CSV -------------------------------------------------------
    from analysis.intrinsic_heterogeneity import _summary
    rows = []
    for name, v in (("HI_spec_train", hi_tr), ("HI_var_train", hiv_tr),
                    ("HI_spec_val", hi_va), ("HI_var_val", hiv_va),
                    ("HI_spec_test", hi_te), ("HI_var_test", hiv_te)):
        s = _summary(v)
        rows.append([name] + [round(s[k], 6) if isinstance(s[k], float)
                              else s[k] for k in
                              ("n", "mean", "median", "std", "iqr", "min",
                               "max", "p10", "p25", "p50", "p75", "p90",
                               "n_unique", "cv")])
    with open(os.path.join(OUT, "heterogeneity_summary.csv"), "w",
              newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["statistic", "n", "mean", "median", "std", "iqr",
                       "min", "max", "p10", "p25", "p50", "p75", "p90",
                       "n_unique", "cv"])
        wcsv.writerows(rows)

    # ---- dataset JSON + budget JSON ----------------------------------------
    dataset_json = {
        "seed": SEED, "dataset": "Haptics",
        "T": 1092, "univariate": True,
        "train_val_test": [132, 23, 308],
        "primary_index": "HI_spec (time-varying spectral JS heterogeneity)",
        "W": W_PRIMARY, "log_convention": "natural (base e)",
        "HI_reference": LOG2,
        "HI_dataset_train_median": hi_dataset,
        "train_summary_HI_spec": _summary(hi_tr),
        "train_summary_HI_var": _summary(hiv_tr),
        "val_summary_HI_spec": _summary(hi_va),
        "test_summary_HI_spec": _summary(hi_te),
        "stationarity_diagnostics": stat_diag,
        "non_degeneracy": {
            "variance_HI_spec_train": float(hi_tr.var()),
            "n_unique_train": int(np.unique(hi_tr).size),
            "n_train": int(hi_tr.size),
            "verdict": "non-degenerate" if hi_tr.var() > 0 and
            np.unique(hi_tr).size > 1 else "degenerate",
        },
    }
    with open(os.path.join(OUT, "dataset_heterogeneity.json"), "w") as f:
        json.dump(dataset_json, f, indent=1)
    with open(os.path.join(OUT, "h_budget.json"), "w") as f:
        json.dump(budget, f, indent=1)

    # ---- downstream demonstration (secondary) ------------------------------
    log("downstream demonstration: gates + intrinsic-budget model...")
    G_trva = np.load(os.path.join(CACHE, "haptics_inference_banks_G_trva.npy")
                     ).astype(np.float64)
    H_trva = np.load(os.path.join(CACHE, "haptics_inference_banks_H_trva.npy")
                     ).astype(np.float64)
    G_te = np.load(os.path.join(CACHE, "haptics_inference_banks_G_te.npy")
                   ).astype(np.float64)
    H_te = np.load(os.path.join(CACHE, "haptics_inference_banks_H_te.npy")
                   ).astype(np.float64)
    n_tr = len(ytr)
    y_dev = np.concatenate([ytr, yva])

    from experiments.heramba_cca_ranked_haptics_seed42.runner import ridge_eval
    m0_res, m0_pred = ridge_eval(G_trva, y_dev, G_te, yte)
    assert abs(m0_res["macro_f1"] - GATE_A) < 5e-5, m0_res
    log(f"[gate A] MiniROCKET = {m0_res['macro_f1']} OK")
    r2_res, r2_pred = ridge_eval(np.hstack([G_trva, H_trva]), y_dev,
                                 np.hstack([G_te, H_te]), yte)
    assert r2_res["macro_f1"] == GATE_B, r2_res
    log(f"[gate B] Raw G+H   = {r2_res['macro_f1']} OK")

    sel = select_h_budget_groups(H_trva[:n_tr], B_H)
    strength = h_carrier_strength(H_trva[:n_tr])
    demos = {}
    preds = {"minirocket": m0_pred, "raw_gh": r2_pred}
    from sklearn.metrics import accuracy_score, f1_score
    for key, mkey in (("quantile", "mask_quantile"),
                      ("hierarchy", "mask_hierarchy"),
                      ("block", "mask_block")):
        mask = sel[mkey]
        Xd = np.hstack([G_trva, H_trva[:, mask]])
        Xt = np.hstack([G_te, H_te[:, mask]])
        res, pred = ridge_eval(Xd, y_dev, Xt, yte)
        demos[key] = {"n_H_features": int(mask.sum()),
                      "macro_f1": res["macro_f1"],
                      "accuracy": res["accuracy"],
                      "selected_alpha": res["selected_alpha"]}
        preds[f"intrinsic_budget_{key}"] = pred
        log(f"[intrinsic-budget {key}] kept {int(mask.sum())} H features "
            f"-> test MF1 {res['macro_f1']}")

    # ---- figures -------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(hi_tr, bins=24, color="tab:blue", alpha=.8)
    axes[0].axvline(hi_dataset, color="tab:red", ls="--",
                    label=f"median = {hi_dataset:.4f}")
    axes[0].set_xlabel("HI_spec (train samples, W=8)")
    axes[0].set_ylabel("count")
    axes[0].legend()
    axes[0].set_title("Sample-level spectral heterogeneity (train)")
    # post-hoc label overlay (index already frozen)
    for c in np.unique(ytr):
        axes[1].hist(hi_tr[ytr == c], bins=16, alpha=.45,
                     label=f"class {c}")
    axes[1].set_xlabel("HI_spec")
    axes[1].set_title("POST-HOC: HI_spec by class (index frozen)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "figures",
                                 f"heterogeneity_distribution.{ext}"))
    plt.close(fig)

    # spectral heterogeneity examples: highest/lowest/median train samples
    order = np.argsort(hi_tr)
    picks = {"lowest": order[0], "median": order[len(order) // 2],
             "highest": order[-1]}
    from analysis.intrinsic_heterogeneity import compute_window_spectrum
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (nm, idx) in zip(axes, picks.items()):
        P = compute_window_spectrum(Xtr[idx], W_PRIMARY)
        for w in range(P.shape[0]):
            ax.plot(P[w], lw=.7, alpha=.75)
        ax.set_title(f"{nm} (HI_spec={hi_tr[idx]:.4f}, "
                     f"train idx {idx}, class {int(ytr[idx])})")
        ax.set_xlabel("frequency bin")
        ax.set_ylabel("normalized power")
    fig.suptitle("Per-window normalized spectra (POST-HOC example picks)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "figures",
                                 f"spectral_heterogeneity_examples.{ext}"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for W in (7, 8, 9):
        s = compute_sample_spectral_heterogeneity(Xtr, W)
        ax.hist(s, bins=20, alpha=.5, label=f"W={W} (median {np.median(s):.4f})")
    ax.set_xlabel("HI_spec")
    ax.set_title("Window-count sensitivity (diagnostic; W=8 primary)")
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "figures", f"window_sensitivity.{ext}"))
    plt.close(fig)

    # ---- artifacts -------------------------------------------------------------
    final_rows = [
        ["MiniROCKET", 4998, m0_res["macro_f1"], "identity gate A"],
        ["Raw G+H", 9996, r2_res["macro_f1"], "identity gate B"],
    ]
    for key in ("quantile", "hierarchy", "block"):
        final_rows.append([f"Intrinsic-budget demonstration ({key})",
                           4998 + demos[key]["n_H_features"],
                           demos[key]["macro_f1"],
                           "intrinsic-budget demonstration (NOT optimized)"])
    with open(os.path.join(OUT, "final_comparison.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["Method", "Total dims", "Test Macro-F1", "Note"])
        w.writerows(final_rows)

    for nm, pr in preds.items():
        np.save(os.path.join(OUT, "predictions", f"{nm}.npy"), pr)

    leakage_json = {
        "poison_value": 1e6,
        "train_index_baseline": leak["train_index_baseline"],
        "train_index_recomputed_with_poisoned_heldout":
            leak["train_index_recomputed_with_poisoned_heldout"],
        "recomputation_invariant": leak["recomputation_invariant"],
        "val_split_own_index": leak["val_split_own_index"],
        "test_split_own_index": leak["test_split_own_index"],
        "heldout_signals_enter_budget": leak["heldout_signals_enter_budget"],
        "labels_enter_index": leak["labels_enter_budget"],
        "note": ("index computed from train raw signals only; val/test "
                 "signals and all labels excluded from the budget"),
    }
    with open(os.path.join(OUT, "leakage_audit.json"), "w") as f:
        json.dump(leakage_json, f, indent=1)

    with open(os.path.join(OUT, "stability_analysis.json"), "w") as f:
        json.dump({**stab, "primary_W": W_PRIMARY,
                   "note": "W=7/9 are diagnostics only; W=8 is primary"},
                  f, indent=1)

    with open(os.path.join(OUT, "config.json"), "w") as f:
        json.dump({
            "seed": SEED, "W_primary": W_PRIMARY,
            "spectral_preprocessing": {
                "detrend": "per-window linear (endpoint)",
                "window_fn": "hann", "n_fft": "next pow2 of window length",
                "normalization": "per-window power sum to 1 (eps 1e-12)",
                "js": "scipy jensenshannon base=e, squared",
                "aggregation": "mean over window pairs, then channels",
            },
            "dataset_index_rule": "median over train samples",
            "budget_rule": "B_H = clip(HI_dataset / log 2, 0, 1)",
            "bands": budget["bands"],
            "group_selection": {
                "ranking": "train-median normalized heterogeneity (label-free)",
                "tie_break": "ascending feature index",
                "variants": ["quantile", "hierarchy(8-slice)",
                             "block(64-kernel)"],
            },
            "frozen_before": "any classifier fitting; labels only post-hoc",
        }, f, indent=1)

    results = {
        "seed": SEED, "dataset": "Haptics",
        "HI_dataset": hi_dataset, "B_H": B_H, "band": budget["band"],
        "gates": {"A_minirocket": m0_res["macro_f1"],
                  "B_raw_gh": r2_res["macro_f1"]},
        "intrinsic_budget_demos": demos,
        "delta_H_known_pp": 4.63,
        "runtime_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=1)

    log(f"complete in {time.time()-t0:.0f}s -> {OUT}")
    return results


if __name__ == "__main__":
    main()
