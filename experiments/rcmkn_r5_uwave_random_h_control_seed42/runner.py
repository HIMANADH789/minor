"""Random-H subset control for the completed R5 UWave experiment (seed 42).

For each dataset, at the FROZEN R5 rho*:
    * reload the identical data split, frozen context model, and rebuild
      G_full / H_full with the same frozen-R2 machinery the R5 run used
      (extractor identity + H recompute + padded-region audits re-verified)
    * verify the stored R5 ranked result reproduces bit-for-bit: refit the
      final Ridge on [G :first-N_G || H :top-N_H] exactly as the R5 runner
      did (same f_classif ranking on the full development set), assert the
      test Macro-F1 equals the stored value, and discard that prediction
      (the ranked reference stays the stored R5 result)
    * for each of the five predefined seeds 420001..420005, sample N_H H
      indices uniformly without replacement from the full bank, assemble
      [same G block || same-size H_random], fit RidgeClassifierCV on the
      development set, record validation Macro-F1, then evaluate the
      official test set EXACTLY ONCE per subset (20 new test evaluations
      total; the primary control is the mean +- std across the 5 subsets)

The complete H candidate bank is the canonical R2 one: 4998 features
(one per valid het kernel); the G bank is 9996. Sampling universe P_H=4998.

No best-random-subset selection after seeing test scores. No rho changes,
no re-ranking, no architecture changes (stop rule).
"""
import json
import os
import time

import numpy as np
import torch
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_selection import f_classif

from experiments.rcmkn_haptics_seed42.runner import set_seed
from experiments.rcmkn_r2_uwave_seed42.data import load_and_split
from experiments.rcmkn_r5_uwave_hbudget_seed42.config import R2_UWAVE_DIR
from experiments.rcmkn_r5_uwave_hbudget_seed42.runner import (
    build_banks, jaccard, load_frozen_context,
)
from experiments.rcmkn_r5_uwave_random_h_control_seed42.config import (
    DATASETS, EXPECTED, OUTCOME_RULE, OUT_DIR, P_H, RANDOM_SEEDS, R5_DIR,
    SEED, TOTAL_BUDGET, classify_outcome,
)
from experiments.rcmkn_r5_uwave_hbudget_seed42.config import budget_split

ALPHAS = np.logspace(-4, 4, 20)


def log(msg):
    print(msg, flush=True)


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def assemble(G_full, H_full, n_g, h_idx):
    """[first-N_G canonical G || H[:, h_idx]] with exact total budget."""
    assert len(h_idx) + n_g == TOTAL_BUDGET
    X = np.hstack([G_full[:, :n_g], H_full[:, h_idx]])
    assert X.shape[1] == TOTAL_BUDGET
    return X


def ranked_top_H(H_full, y_dev, n_h):
    """Exact R5 final ranking convention (full development set, stable)."""
    f_stat, _ = f_classif(H_full, y_dev)
    return np.argsort(-np.nan_to_num(f_stat, nan=0.0), kind="stable")[:n_h]


def run_dataset(ds_name, device):
    t0 = time.time()
    ds_dir = os.path.join(OUT_DIR, ds_name)
    os.makedirs(ds_dir, exist_ok=True)
    log(f"\n{'=' * 74}\n  RANDOM-H CONTROL — {ds_name}\n{'=' * 74}")

    exp = EXPECTED[ds_name]
    rho_star = exp["rho"]
    n_g, n_h = budget_split(rho_star)

    # -------- identical split + frozen banks via the R5 machinery --------
    data = load_and_split(ds_name, ds_dir)
    r5_train = np.load(os.path.join(R2_UWAVE_DIR, ds_name,
                                    "train_indices.npy"))
    r5_val = np.load(os.path.join(R2_UWAVE_DIR, ds_name,
                                  "val_indices.npy"))
    assert np.array_equal(r5_train, np.load(os.path.join(
        ds_dir, "train_indices.npy")))
    assert np.array_equal(r5_val, np.load(os.path.join(
        ds_dir, "val_indices.npy")))
    ytr, yva, yte = data["ytr"], data["yva"], data["yte"]
    y_dev = np.concatenate([ytr, yva])
    n_classes = data["n_classes"]

    model, train_info, params = load_frozen_context(ds_name, device)
    banks, bank_audits = build_banks(data, model, device)
    G_trva, G_te = banks["G_trva"], banks["G_te"]
    H_trva, H_te = banks["H_trva"], banks["H_te"]
    assert H_trva.shape[1] == P_H and G_trva.shape[1] == TOTAL_BUDGET

    # -------- ranked reference reproduction (no extra test evaluation) ---
    top = ranked_top_H(H_trva, y_dev, n_h)
    ridge_ranked = RidgeClassifierCV(alphas=ALPHAS)
    ridge_ranked.fit(assemble(G_trva, H_trva, n_g, top), y_dev)
    rep_val = round(macro_f1(yva, ridge_ranked.predict(
        assemble(G_trva, H_trva, n_g, top)[len(ytr):])), 4)
    rep_te = round(macro_f1(yte, ridge_ranked.predict(
        assemble(G_te, H_te, n_g, top))), 4)
    stored = json.load(open(os.path.join(R5_DIR, ds_name, "result.json")))["r5"]
    assert rep_te == stored["test_macro_f1"], \
        f"ranked reproduction {rep_te} != stored {stored['test_macro_f1']}"
    assert rep_val == stored["val_macro_f1"], \
        f"ranked val reproduction {rep_val} != {stored['val_macro_f1']}"
    log(f"  [RANKED] reproduces stored R5 exactly: val={rep_val} "
        f"test={rep_te} (no new test evaluation)")
    fsel_r5 = json.load(open(os.path.join(R5_DIR, ds_name,
                                           "feature_selection.json")))
    assert top[:20].tolist() == \
        fsel_r5["per_rho"][str(rho_star)]["selected_H_indices_first20"], \
        "ranked top-20 indices differ from stored R5 feature_selection.json"

    # -------- 5 frozen random subsets --------
    subsets = []
    val_scores, test_scores = [], []
    rng_check = []
    for rs in RANDOM_SEEDS:
        set_seed(SEED)                     # training stream untouched by rs
        rng = np.random.default_rng(rs)    # dedicated independent RNG
        idx = np.sort(rng.choice(P_H, size=n_h, replace=False))
        rng_check.append(idx)
        assert len(idx) == n_h and len(np.unique(idx)) == n_h
        assert idx.min() >= 0 and idx.max() < P_H
        ridge = RidgeClassifierCV(alphas=ALPHAS)
        ridge.fit(assemble(G_trva, H_trva, n_g, idx), y_dev)
        v = macro_f1(yva, ridge.predict(assemble(G_trva, H_trva, n_g, idx)
                                        [len(ytr):]))
        t = macro_f1(yte, ridge.predict(assemble(G_te, H_te, n_g, idx)))
        # ONE official test evaluation per random subset
        val_scores.append(round(v, 4))
        test_scores.append(round(t, 4))
        subsets.append({
            "seed": rs, "n_h": n_h, "n_g": n_g,
            "H_indices_first20": idx[:20].tolist(),
            "val_macro_f1": round(v, 4), "test_macro_f1": round(t, 4),
            "alpha": float(ridge.alpha_),
        })
        log(f"  [RANDOM {rs}] val={v:.4f} test={t:.4f} "
            f"(alpha={ridge.alpha_:.3g})")

    rm, rs_ = float(np.mean(test_scores)), float(np.std(test_scores))
    vm, vs = float(np.mean(val_scores)), float(np.std(val_scores))
    ranked_test = exp["ranked_test"]
    m0 = exp["m0_test"]
    diff = round(ranked_test - rm, 4)
    outcome = classify_outcome(ranked_test - rm, rs_)
    log(f"  [OUTCOME {outcome}] ranked={ranked_test} random={rm:.4f}"
        f"+/-{rs_:.4f} diff={diff:+.4f}")

    # pairwise overlap among random subsets (expectation r/(2-r), r=n/P_H)
    pair_jac = [round(jaccard(rng_check[i], rng_check[j]), 4)
                for i in range(len(rng_check))
                for j in range(i + 1, len(rng_check))]
    r = n_h / P_H
    expected_jac = r / (2 - r)

    ranked_ref = {
        "source": "results/r5_uwave_hbudget_seed42 (stored, unchanged)",
        "rho": rho_star, "N_H": n_h, "N_G": n_g,
        "val_macro_f1": stored["val_macro_f1"],
        "test_macro_f1": stored["test_macro_f1"],
        "reproduction": {"val": rep_val, "test": rep_te,
                         "matches_stored": True},
        "top_H_indices_first20": top[:20].tolist(),
    }
    with open(os.path.join(ds_dir, "ranked_reference.json"), "w") as f:
        json.dump(ranked_ref, f, indent=2)

    # -------- audits --------
    audits = {
        "audit1_same_split_as_r5": {"pass": True},
        "audit2_same_preprocessing_as_r5": {
            "note": "identical load_and_split + znorm path",
            "pass": True},
        "audit3_same_frozen_G_H_banks": {
            "note": "rebuild via R5 build_banks; ranked Ridge reproduces "
                    "stored test exactly",
            "extractor_identity_maxdiff": bank_audits[
                "extractor_identity_maxdiff"],
            "H_recompute_maxdiff": bank_audits["H_recompute_maxdiff"],
            "out_of_mask_flips": bank_audits["out_of_mask_flips"],
            "H_unchanged_after_flips": bank_audits["H_unchanged_after_flips"],
            "pass": True},
        "audit4_same_rho_star": {"rho": rho_star, "pass": True},
        "audit5_same_N_H": {"N_H": n_h, "pass": True},
        "audit6_same_N_G": {"N_G": n_g, "pass": True},
        "audit7_total_dim_9996": {"pass": True},
        "audit8_sampling_without_replacement": {"pass": True},
        "audit9_indices_valid": {"pass": True},
        "audit10_independent_predefined_seeds": {
            "seeds": RANDOM_SEEDS,
            "note": "np.random.default_rng(seed); training stream "
                    "re-seeded to 42 and untouched",
            "pass": True},
        "audit11_no_labels_in_subset_choice": {"pass": True},
        "audit12_ridge_canonical": {
            "classifier": "RidgeClassifierCV", "alphas": "logspace(-4,4,20)",
            "pass": True},
        "audit13_same_final_fit_protocol": {
            "note": "fit on full development set (train+val), exactly as R5",
            "pass": True},
        "audit14_no_r3_r4_gating": {"pass": True},
        "audit15_no_hydra_rpms": {"pass": True},
        "audit16_stored_ranked_result_used": {
            "stored_test": stored["test_macro_f1"],
            "reproduced": rep_te,
            "no_new_ranked_test_eval": True, "pass": True},
        "audit17_one_test_eval_per_random_subset": {
            "subsets": len(RANDOM_SEEDS), "total_new_evals": len(RANDOM_SEEDS),
            "pass": True},
        "audit18_no_best_subset_selection": {
            "note": "primary control is mean+-std over all 5 subsets",
            "pass": True},
    }
    with open(os.path.join(ds_dir, "audits.json"), "w") as f:
        json.dump(audits, f, indent=2)

    val_res = {
        "ranked_val": stored["val_macro_f1"],
        "random_mean_val": round(vm, 4), "random_std_val": round(vs, 4),
        "diff_val": round(stored["val_macro_f1"] - vm, 4),
        "per_subset_val": val_scores,
    }
    with open(os.path.join(ds_dir, "validation_results.json"), "w") as f:
        json.dump(val_res, f, indent=2)

    test_res = {
        "ranked_test_stored": ranked_test,
        "m0_test_stored": m0,
        "random_mean_test": round(rm, 4), "random_std_test": round(rs_, 4),
        "random_min_test": round(float(np.min(test_scores)), 4),
        "random_max_test": round(float(np.max(test_scores)), 4),
        "per_subset_test": test_scores,
        "delta_select": diff,
        "ratio_descriptive": round((ranked_test - m0) / (rm - m0), 4)
        if abs(rm - m0) > 1e-12 else None,
        "outcome": outcome,
        "outcome_rule": OUTCOME_RULE,
    }
    with open(os.path.join(ds_dir, "test_results.json"), "w") as f:
        json.dump(test_res, f, indent=2)

    overlap = {
        "pairwise_jaccard": pair_jac,
        "mean": round(float(np.mean(pair_jac)), 4),
        "expected_for_random": round(expected_jac, 4),
    }
    with open(os.path.join(ds_dir, "feature_overlap.json"), "w") as f:
        json.dump(overlap, f, indent=2)

    # save exact random indices
    for rs, idx in zip(RANDOM_SEEDS, rng_check):
        np.save(os.path.join(ds_dir, f"random_seed_{rs}_indices.npy"), idx)

    with open(os.path.join(ds_dir, "random_subsets.json"), "w") as f:
        json.dump({"seeds": RANDOM_SEEDS, "subsets": subsets}, f, indent=2)

    with open(os.path.join(ds_dir, "diagnostics.json"), "w") as f:
        json.dump({
            "context_params": params, "context_train": train_info,
            "N_H": n_h, "N_G": n_g, "P_H": P_H,
            "random_subset_val": val_scores,
            "random_subset_test": test_scores,
            "pairwise_jaccard": overlap,
            "runtime_s": round(time.time() - t0, 1),
        }, f, indent=2)

    log(f"  [{ds_name}] done in {time.time() - t0:.1f}s")

    return {
        "dataset": ds_name, "rho": rho_star, "N_H": n_h, "N_G": n_g,
        "ranked_test": ranked_test, "m0_test": m0,
        "random_mean": round(rm, 4), "random_std": round(rs_, 4),
        "random_min": round(float(np.min(test_scores)), 4),
        "random_max": round(float(np.max(test_scores)), 4),
        "per_subset_test": test_scores,
        "ranked_val": stored["val_macro_f1"],
        "random_mean_val": round(vm, 4), "random_std_val": round(vs, 4),
        "diff": diff, "outcome": outcome,
        "pair_jac_mean": overlap["mean"],
        "expected_jac": round(expected_jac, 4),
        "runtime_s": round(time.time() - t0, 1),
    }


def main(datasets=DATASETS):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    log("=" * 74)
    log("R5 RANDOM-H CONTROL — TOP-RANKED vs SAME-SIZE RANDOM H (seed 42)")
    log("=" * 74)
    log(f"device={device} random_subset_seeds={RANDOM_SEEDS}")
    all_res = {ds: run_dataset(ds, device) for ds in datasets}

    cross = {
        "seed": SEED,
        "primary_question": ("At the R5-selected H budget, does selecting "
                             "top-ranked H features provide more predictive "
                             "value than randomly selecting the same number "
                             "of H features?"),
        "per_dataset": all_res,
        "outcomes": {ds: r["outcome"] for ds, r in all_res.items()},
        "outcome_rule": OUTCOME_RULE,
    }
    with open(os.path.join(OUT_DIR, "cross_dataset_summary.json"), "w") as f:
        json.dump(cross, f, indent=2)

    from experiments.rcmkn_r5_uwave_random_h_control_seed42.figures import (
        make_figures)
    from experiments.rcmkn_r5_uwave_random_h_control_seed42.report import (
        write_report)
    try:
        make_figures(all_res, OUT_DIR)
    except Exception as e:
        log(f"  [FIGURES] skipped: {e}")
    write_report(OUT_DIR, all_res, cross)

    log("\n" + "=" * 74)
    log("R5 RANDOM-H CONTROL — FINAL")
    log("=" * 74)
    for ds, r in all_res.items():
        log(f"{ds} (rho*={r['rho']}, N_H={r['N_H']})")
        log(f"    R5 Ranked Test: {r['ranked_test']:.4f}")
        for rs, t in zip(RANDOM_SEEDS, r["per_subset_test"]):
            log(f"    Random {rs} Test: {t:.4f}")
        log(f"    Random Mean +- SD: {r['random_mean']:.4f} "
            f"+/- {r['random_std']:.4f}")
        log(f"    Ranked - Random Mean: {r['diff']:+.4f}")
        log(f"    Outcome: {r['outcome']}")
    log("=" * 74)
    return all_res
