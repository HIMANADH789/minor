"""
TURS-KTM-AV: Alignment-Validated Kernel Timing Moments
========================================================
The FINAL controlled experiment in the KTM/timing line.

Hypothesis:
  "Does the original window-relative KTM-W contain statistically significant
   predictive information beyond canonical MiniROCKET, as opposed to merely
   responding to arbitrary temporal arrangement?"

Protocol:
  1. Establish canonical MiniROCKET baseline (10K, random_state=42)
  2. Select top-K kernels by absolute MiniROCKET coefficient magnitude
  3. Compute KTM-W on real validation windows
  4. Build circular-shift null distribution (N_NULL_SHIFTS=20)
  5. One-sided empirical p-value: does D_real exceed null distribution?
  6. KEEP or DROP decision (frozen before test access)
  7. Refit final model on train+val, evaluate ONCE on held-out test

DO NOT modify MiniROCKET.
DO NOT add CPT, TURS Local, G4, routing, gating, attention, or ensembles.
DO NOT design another architecture.

This is the last timing experiment.
"""
import csv
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.features.ktm import (
    ktm_transform, block_stats, feature_owners, concatenate_blocks,
    standardize_blocks, activation_count_summary,
)

# ============================================================================
# Configuration
# ============================================================================
SEED = 42
K_TOP = 1024  # fixed across all datasets, selected BEFORE null evaluation
N_NULL_SHIFTS = 20  # per validation sample
ALPHA_SIGNIFICANCE = 0.05

DATASETS = [
    ("ECG5000_UNBAL", "data/ecg5000_resplit.npz", 5),
    ("ECG5000_BAL",   "data/ecg5000_fair_balanced.npz", 5),
    ("CWRU_UNBAL",    "data/cwru_unbalanced.npz", 4),
    ("CWRU_BAL",      "data/cwru_balanced.npz", 4),
]

REF_VALUES = {
    "ECG5000_UNBAL": 0.5938,
    "ECG5000_BAL":   0.6553,
    "CWRU_UNBAL":    0.9917,
    "CWRU_BAL":      0.9947,
}

OUT_DIR = os.path.join(ROOT, "results", "turs_ktm_av")


# ============================================================================
# Helpers
# ============================================================================
def znorm(X):
    mu = X.mean(axis=-1, keepdims=True)
    sig = X.std(axis=-1, keepdims=True) + 1e-8
    return ((X - mu) / sig).astype(np.float32)


def load_split(ds_file):
    data = np.load(os.path.join(ROOT, ds_file))
    if "X_train" in data:
        Xa, ya = data["X_train"], data["y_train"].astype(int)
        Xte, yte = data["X_test"], data["y_test"].astype(int)
        Xtr, Xva, ytr, yva = train_test_split(
            Xa, ya, test_size=0.15, stratify=ya, random_state=SEED)
    else:
        Xa, ya = data["X"], data["y"].astype(int)
        Xtr, Xte, ytr, yte = train_test_split(
            Xa, ya, test_size=0.15, stratify=ya, random_state=SEED)
        Xtr, Xva, ytr, yva = train_test_split(
            Xtr, ytr, test_size=0.15, stratify=ytr, random_state=SEED)
    return Xtr, Xva, Xte, ytr, yva, yte


def run_ridge(X_trva, y_trva, X_te, y_te):
    clf = RidgeClassifierCV(alphas=np.logspace(-4, 4, 20))
    clf.fit(X_trva, y_trva)
    preds = clf.predict(X_te)
    acc = float(accuracy_score(y_te, preds))
    mf1 = float(f1_score(y_te, preds, average="macro", zero_division=0))
    wf1 = float(f1_score(y_te, preds, average="weighted", zero_division=0))
    per_class = f1_score(y_te, preds, average=None, zero_division=0,
                         labels=list(range(int(y_te.max()) + 1)))
    cm = confusion_matrix(y_te, preds).tolist()
    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(mf1, 4),
        "weighted_f1": round(wf1, 4),
        "class_f1s": [round(float(f), 4) for f in per_class],
        "confusion_matrix": cm,
        "alpha": round(float(clf.alpha_), 6),
        "n_features": int(X_trva.shape[1]),
    }


def compute_ktm_w(X, transformer):
    ppv, mu, W, counts = ktm_transform(X, transformer, return_counts=True)
    return W, counts


def circular_shift_signal(x, shift, rng):
    T = len(x)
    shift = int(shift) % T
    return np.roll(x, -shift)


def select_top_k_kernels_by_mr_coefficients(Z_tr, ytr, mr, K):
    clf = RidgeClassifierCV(alphas=np.logspace(-4, 4, 20))
    clf.fit(Z_tr, ytr)
    coefs = clf.coef_
    abs_coefs = np.abs(coefs).mean(axis=0)
    dil_vals, dil_idx, kern_idx = feature_owners(mr)
    n_kernels = len(np.unique(kern_idx))
    kernel_scores = np.zeros(n_kernels)
    for k in range(n_kernels):
        mask = kern_idx == k
        kernel_scores[k] = abs_coefs[mask].sum()
    K_actual = min(K, n_kernels)
    top_k_indices = np.argsort(kernel_scores)[::-1][:K_actual]
    top_k_feature_mask = np.zeros(Z_tr.shape[1], dtype=bool)
    for k_idx in top_k_indices:
        top_k_feature_mask |= kern_idx == k_idx
    return top_k_indices, top_k_feature_mask, abs_coefs, kernel_scores


def standardize_block(train_block, val_block, test_block):
    trva = np.concatenate([train_block, val_block])
    m = trva.mean(axis=0, keepdims=True)
    s = trva.std(axis=0, keepdims=True)
    s = np.where(s < 1e-8, 1.0, s)
    return (train_block - m) / s, (val_block - m) / s, (test_block - m) / s


def compute_block_diagnostics(block, name):
    b = np.asarray(block, dtype=np.float64)
    return {
        "name": name,
        "n_features": int(b.shape[1]),
        "n_samples": int(b.shape[0]),
        "mean": float(b.mean()),
        "std": float(b.std()),
        "min": float(b.min()),
        "max": float(b.max()),
        "near_zero_variance_count": int((b.std(axis=0) < 1e-8).sum()),
    }


def empirical_p_value(delta_real, delta_null):
    N = len(delta_null)
    count_ge = np.sum(np.asarray(delta_null) >= delta_real)
    return (1 + count_ge) / (1 + N)


def save_all_results(log, full_results, all_decisions, all_validation_real,
                     all_validation_null, all_final_test, all_top_k,
                     all_runtime, all_diagnostics):
    log("  SAVING RESULTS")

    with open(os.path.join(OUT_DIR, "full_results.json"), "w") as f:
        json.dump(full_results, f, indent=2)

    with open(os.path.join(OUT_DIR, "decisions.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset", "delta_real", "null_mean", "null_std",
            "null_median", "null_95th", "p_value", "decision",
            "final_model", "final_test_mf1", "delta_test",
        ])
        writer.writeheader()
        for d in all_decisions:
            writer.writerow({k: d[k] for k in writer.fieldnames})

    with open(os.path.join(OUT_DIR, "validation_real.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset", "mr_baseline_val_mf1", "real_ktm_val_mf1",
            "delta_real", "p_value", "decision",
        ])
        writer.writeheader()
        for v in all_validation_real:
            writer.writerow(v)

    with open(os.path.join(OUT_DIR, "validation_null.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset", "null_mean", "null_std", "null_median",
            "null_95th", "null_deltas",
        ])
        writer.writeheader()
        for v in all_validation_real:
            ds = v["dataset"]
            if ds in full_results:
                null_data = full_results[ds]["validation"]
                writer.writerow({
                    "dataset": ds,
                    "null_mean": null_data["null_mean"],
                    "null_std": null_data["null_std"],
                    "null_median": null_data["null_median"],
                    "null_95th": null_data["null_95th"],
                    "null_deltas": json.dumps(null_data["null_deltas"]),
                })

    with open(os.path.join(OUT_DIR, "final_test_results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset", "mr_baseline_test_mf1", "final_model",
            "final_test_mf1", "final_test_acc", "final_test_wf1",
            "delta_test", "ref_value",
        ])
        writer.writeheader()
        for t in all_final_test:
            writer.writerow(t)

    with open(os.path.join(OUT_DIR, "top_k_kernels.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset", "rank", "kernel_id", "kernel_score",
        ])
        writer.writeheader()
        for k in all_top_k:
            writer.writerow(k)

    with open(os.path.join(OUT_DIR, "runtime.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset", "mr_fit_s", "top_k_selection_s", "ktm_extract_s",
            "null_build_s", "total_s",
        ])
        writer.writeheader()
        for r in all_runtime:
            writer.writerow(r)

    with open(os.path.join(OUT_DIR, "diagnostics.json"), "w") as f:
        json.dump(all_diagnostics, f, indent=2)

    config = {
        "experiment": "TURS-KTM-AV",
        "version": "1.0",
        "seed": SEED,
        "K_TOP": K_TOP,
        "N_NULL_SHIFTS": N_NULL_SHIFTS,
        "ALPHA_SIGNIFICANCE": ALPHA_SIGNIFICANCE,
        "n_kernels": 10000,
        "datasets": [d[0] for d in DATASETS],
        "ref_values": REF_VALUES,
        "ktm_formula": "W = mean_i |x_i - i/(k+1)|, x_i = tau_i / T",
        "selection_criterion": "absolute MiniROCKET coefficient magnitude",
        "null_method": "circular shift per validation sample",
        "decision_rule": "p < 0.05 AND delta_real > 0",
    }
    with open(os.path.join(OUT_DIR, "configs", "experiment_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    with open(os.path.join(OUT_DIR, "configs", "null_shifts.json"), "w") as f:
        json.dump(all_validation_null, f, indent=2)


# ============================================================================
# Main experiment
# ============================================================================
def main():
    os.makedirs(os.path.join(OUT_DIR, "configs"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "logs"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "figures"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "reports"), exist_ok=True)

    log_path = os.path.join(OUT_DIR, "logs", "turs_ktm_av.log")
    log_fh = open(log_path, "w")

    def log(msg):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        log_fh.write(line + "\n")
        log_fh.flush()

    log("TURS-KTM-AV: Alignment-Validated Kernel Timing Moments")
    log(f"Configuration: K_TOP={K_TOP}, N_NULL_SHIFTS={N_NULL_SHIFTS}, "
        f"ALPHA={ALPHA_SIGNIFICANCE}, SEED={SEED}")

    grand_t0 = time.time()

    all_decisions = []
    all_validation_real = []
    all_validation_null = []
    all_final_test = []
    all_top_k = []
    all_runtime = []
    all_diagnostics = {}
    full_results = {}

    for ds_idx, (ds_name, ds_file, n_cls) in enumerate(DATASETS):
        log(f"\n{'='*70}")
        log(f"  DATASET {ds_idx+1}/4: {ds_name} ({n_cls} classes)")
        log(f"{'='*70}")

        t0 = time.time()

        Xtr, Xva, Xte, ytr, yva, yte = load_split(ds_file)
        Xn_tr = znorm(Xtr)
        Xn_va = znorm(Xva)
        Xn_te = znorm(Xte)
        y_trva = np.concatenate([ytr, yva])

        log(f"  Data: train={len(Xtr)}, val={len(Xva)}, test={len(Xte)}, T={Xn_tr.shape[1]}")

        from aeon.transformations.collection.convolution_based import MiniRocket

        t_mr = time.time()
        mr = MiniRocket(n_kernels=10000, random_state=SEED, n_jobs=-1)
        mr.fit_transform(Xn_tr[:, None, :])
        t_mr_fit = time.time() - t_mr

        Z_tr = mr.transform(Xn_tr[:, None, :]).astype(np.float64)
        Z_va = mr.transform(Xn_va[:, None, :]).astype(np.float64)
        Z_te = mr.transform(Xn_te[:, None, :]).astype(np.float64)
        log(f"  MiniROCKET: {Z_tr.shape[1]} features ({t_mr_fit:.1f}s)")

        r_baseline_test = run_ridge(np.concatenate([Z_tr, Z_va]), y_trva, Z_te, yte)
        log(f"  MR baseline (test): MF1={r_baseline_test['macro_f1']:.4f}")

        ref = REF_VALUES.get(ds_name)
        if ref is not None:
            delta_ref = abs(r_baseline_test['macro_f1'] - ref)
            if delta_ref > 0.02:
                log(f"  WARNING: deviates from ref {ref:.4f} by {delta_ref:.4f}")
            else:
                log(f"  Baseline check PASS: {r_baseline_test['macro_f1']:.4f} vs {ref:.4f}")

        r_baseline_val = run_ridge(Z_tr, ytr, Z_va, yva)
        log(f"  MR baseline (val): MF1={r_baseline_val['macro_f1']:.4f}")

        t_sel = time.time()
        top_k_idx, feat_mask, mr_coefs, kernel_scores = \
            select_top_k_kernels_by_mr_coefficients(Z_tr, ytr, mr, K_TOP)
        Z_tr_k = Z_tr[:, feat_mask]
        Z_va_k = Z_va[:, feat_mask]
        Z_te_k = Z_te[:, feat_mask]
        t_sel = time.time() - t_sel
        log(f"  Top-K: K={len(top_k_idx)}, features={feat_mask.sum()} ({t_sel:.1f}s)")

        t_ktm = time.time()
        W_tr, ct_tr = compute_ktm_w(Xn_tr, mr)
        W_va, ct_va = compute_ktm_w(Xn_va, mr)
        W_te, ct_te = compute_ktm_w(Xn_te, mr)
        t_ktm = time.time() - t_ktm

        W_tr_k = W_tr[:, feat_mask]
        W_va_k = W_va[:, feat_mask]
        W_te_k = W_te[:, feat_mask]
        ct_tr_k = ct_tr[:, feat_mask]

        zero_frac = (ct_tr_k == 0).mean()
        mean_count = ct_tr_k.mean()
        log(f"  KTM-W: zero-frac={zero_frac:.3f}, mean-count={mean_count:.1f} ({t_ktm:.1f}s)")

        W_tr_s, W_va_s, W_te_s = standardize_block(W_tr_k, W_va_k, W_te_k)

        F_tr_real = np.concatenate([Z_tr_k, W_tr_s], axis=1)
        F_va_real = np.concatenate([Z_va_k, W_va_s], axis=1)

        r_real_val = run_ridge(F_tr_real, ytr, F_va_real, yva)
        delta_real = r_real_val['macro_f1'] - r_baseline_val['macro_f1']
        log(f"  MR+KTM-W real (val): MF1={r_real_val['macro_f1']:.4f}, D={delta_real:+.4f}")

        # Null distribution
        log(f"  Building null ({N_NULL_SHIFTS} shifts)...")
        t_null = time.time()
        rng_null = np.random.default_rng(SEED + ds_idx)
        null_mf1s = []
        null_deltas = []
        null_shifts_saved = []

        all_shifts = rng_null.integers(0, Xn_va.shape[1],
                                        size=(N_NULL_SHIFTS, len(Xn_va)))

        for j in range(N_NULL_SHIFTS):
            Xn_va_shifted = np.zeros_like(Xn_va)
            for i in range(len(Xn_va)):
                Xn_va_shifted[i] = circular_shift_signal(
                    Xn_va[i], all_shifts[j, i], rng_null)

            _, _, W_s_all, _ = ktm_transform(Xn_va_shifted, mr, return_counts=True)
            W_va_null_j = W_s_all[:, feat_mask]
            null_shifts_saved.append([int(s) for s in all_shifts[j]])

            W_va_null_s = (W_va_null_j - W_tr_k.mean(axis=0, keepdims=True)) / \
                np.where(W_tr_k.std(axis=0, keepdims=True) < 1e-8, 1.0,
                         W_tr_k.std(axis=0, keepdims=True))

            F_va_null = np.concatenate([Z_va_k, W_va_null_s], axis=1)
            r_null_val = run_ridge(F_tr_real, ytr, F_va_null, yva)

            delta_null_j = r_null_val['macro_f1'] - r_baseline_val['macro_f1']
            null_mf1s.append(r_null_val['macro_f1'])
            null_deltas.append(delta_null_j)

            if (j + 1) % 5 == 0 or j == 0:
                log(f"    null[{j+1}/{N_NULL_SHIFTS}]: MF1={r_null_val['macro_f1']:.4f}, D={delta_null_j:+.4f}")

        t_null = time.time() - t_null
        log(f"  Null built ({t_null:.1f}s)")

        null_deltas_arr = np.array(null_deltas)
        null_mf1s_arr = np.array(null_mf1s)

        p_value = empirical_p_value(delta_real, null_deltas_arr)
        null_mean = float(null_deltas_arr.mean())
        null_std = float(null_deltas_arr.std())
        null_median = float(np.median(null_deltas_arr))
        null_95th = float(np.percentile(null_deltas_arr, 95))

        log(f"  D_real={delta_real:+.4f}, null mean={null_mean:+.4f}, "
            f"null 95th={null_95th:+.4f}, p={p_value:.4f}")

        keep_ktm = (p_value < ALPHA_SIGNIFICANCE) and (delta_real > 0)
        decision = "KEEP_KTM_W" if keep_ktm else "DROP_KTM_W"
        log(f"  DECISION: {decision}")

        if keep_ktm:
            F_tr_final = np.concatenate([Z_tr_k, W_tr_s], axis=1)
            F_va_final = np.concatenate([Z_va_k, W_va_s], axis=1)
            F_te_final = np.concatenate([Z_te_k, W_te_s], axis=1)
            r_final = run_ridge(np.concatenate([F_tr_final, F_va_final]), y_trva, F_te_final, yte)
            final_model = "MR + KTM-W"
        else:
            r_final = run_ridge(np.concatenate([Z_tr_k, Z_va_k]), y_trva, Z_te_k, yte)
            final_model = "MR"

        delta_test = r_final['macro_f1'] - r_baseline_test['macro_f1']
        log(f"  FINAL TEST: MF1={r_final['macro_f1']:.4f} (D={delta_test:+.4f})")

        t_total = time.time() - t0
        log(f"  Dataset done ({t_total:.1f}s)")

        all_decisions.append({
            "dataset": ds_name, "n_classes": n_cls,
            "n_train": len(Xtr), "n_val": len(Xva), "n_test": len(Xte),
            "T": int(Xn_tr.shape[1]),
            "K_top": int(len(top_k_idx)),
            "n_features_mr": int(Z_tr.shape[1]),
            "n_features_mr_k": int(Z_tr_k.shape[1]),
            "n_features_w_k": int(W_tr_k.shape[1]),
            "n_features_combined": int(F_tr_real.shape[1]),
            "mr_val_mf1": r_baseline_val['macro_f1'],
            "mr_test_mf1": r_baseline_test['macro_f1'],
            "real_val_mf1": r_real_val['macro_f1'],
            "delta_real": round(delta_real, 4),
            "null_mean": round(null_mean, 4), "null_std": round(null_std, 4),
            "null_median": round(null_median, 4), "null_95th": round(null_95th, 4),
            "null_mf1s": [round(float(m), 4) for m in null_mf1s],
            "null_deltas": [round(float(d), 4) for d in null_deltas],
            "p_value": round(p_value, 4), "decision": decision,
            "final_model": final_model,
            "final_test_mf1": r_final['macro_f1'],
            "final_test_acc": r_final['accuracy'],
            "final_test_wf1": r_final['weighted_f1'],
            "delta_test": round(delta_test, 4),
        })

        all_validation_null.append({
            "dataset": ds_name,
            "null_shifts": null_shifts_saved,
        })

        all_validation_real.append({
            "dataset": ds_name,
            "mr_baseline_val_mf1": r_baseline_val['macro_f1'],
            "real_ktm_val_mf1": r_real_val['macro_f1'],
            "delta_real": round(delta_real, 4),
            "p_value": round(p_value, 4),
            "decision": decision,
        })

        all_final_test.append({
            "dataset": ds_name,
            "mr_baseline_test_mf1": r_baseline_test['macro_f1'],
            "final_model": final_model,
            "final_test_mf1": r_final['macro_f1'],
            "final_test_acc": r_final['accuracy'],
            "final_test_wf1": r_final['weighted_f1'],
            "delta_test": round(delta_test, 4),
            "ref_value": ref,
        })

        for k_order, k_idx in enumerate(top_k_idx):
            all_top_k.append({
                "dataset": ds_name, "rank": k_order + 1,
                "kernel_id": int(k_idx),
                "kernel_score": round(float(kernel_scores[k_idx]), 6),
            })

        all_runtime.append({
            "dataset": ds_name,
            "mr_fit_s": round(t_mr_fit, 1),
            "top_k_selection_s": round(t_sel, 1),
            "ktm_extract_s": round(t_ktm, 1),
            "null_build_s": round(t_null, 1),
            "total_s": round(t_total, 1),
        })

        all_diagnostics[ds_name] = {
            "mr_block": compute_block_diagnostics(Z_tr, "MR (full)"),
            "mr_k_block": compute_block_diagnostics(Z_tr_k, "MR (top-K)"),
            "ktm_w_k_block": compute_block_diagnostics(W_tr_k, "KTM-W (top-K)"),
            "ktm_w_k_block_std": compute_block_diagnostics(W_tr_s, "KTM-W (top-K, std)"),
            "combined_block": compute_block_diagnostics(F_tr_real, "MR+KTM-W"),
            "counts": {"zero_frac": round(float(zero_frac), 4),
                       "mean_count": round(float(mean_count), 2)},
            "coefficient_audit": {
                "top_k_kernel_ids": [int(k) for k in top_k_idx],
                "top_k_scores": [round(float(kernel_scores[k]), 6) for k in top_k_idx[:10]],
            },
        }

        full_results[ds_name] = {
            "baseline": r_baseline_test,
            "validation": {
                "baseline_val": r_baseline_val,
                "real_ktm_val": r_real_val,
                "delta_real": round(delta_real, 4),
                "null_mean": round(null_mean, 4),
                "null_std": round(null_std, 4),
                "null_median": round(null_median, 4),
                "null_95th": round(null_95th, 4),
                "null_deltas": [round(float(d), 4) for d in null_deltas],
                "p_value": round(p_value, 4),
            },
            "decision": decision,
            "final": r_final,
        }

        # Incremental save (AFTER full_results is populated)
        save_all_results(log, full_results, all_decisions, all_validation_real,
                         all_validation_null, all_final_test, all_top_k,
                         all_runtime, all_diagnostics)
        log(f"  Saved after {ds_name}")

    # ===================================================================
    # Figures
    # ===================================================================
    log(f"\n{'='*70}")
    log("  GENERATING FIGURES")
    log(f"{'='*70}")

    fig_dir = os.path.join(OUT_DIR, "figures")

    # Figure 1
    fig1, axes1 = plt.subplots(1, 4, figsize=(20, 5))
    for idx, (ds_name, _, _) in enumerate(DATASETS):
        ax = axes1[idx]
        vr = full_results[ds_name]["validation"]
        null_d = np.array(vr["null_deltas"])
        real_d = vr["delta_real"]
        ax.hist(null_d, bins=min(15, N_NULL_SHIFTS), color="steelblue",
                alpha=0.7, edgecolor="black", label="Null D")
        ax.axvline(real_d, color="red", linewidth=2, linestyle="--",
                   label=f"Real D={real_d:+.4f}")
        ax.set_title(ds_name, fontsize=11)
        ax.set_xlabel("D Macro-F1")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
    fig1.suptitle("Figure 1: Real vs Null Delta Distribution", fontsize=13, fontweight="bold")
    fig1.tight_layout(rect=[0, 0, 1, 0.93])
    fig1.savefig(os.path.join(fig_dir, "fig1_real_vs_null_delta.png"), dpi=150, bbox_inches="tight")
    fig1.savefig(os.path.join(fig_dir, "fig1_real_vs_null_delta.pdf"), bbox_inches="tight")
    plt.close(fig1)

    # Figure 2
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
    axes2 = axes2.flatten()
    for idx, (ds_name, _, _) in enumerate(DATASETS):
        ax = axes2[idx]
        vr = full_results[ds_name]["validation"]
        null_d = np.array(vr["null_deltas"])
        real_d = vr["delta_real"]
        p_val = vr["p_value"]
        ax.hist(null_d, bins=min(15, N_NULL_SHIFTS), color="lightcoral",
                alpha=0.7, edgecolor="black", label=f"Null (n={N_NULL_SHIFTS})")
        ax.axvline(real_d, color="darkgreen", linewidth=2.5,
                   label=f"Real D={real_d:+.4f}")
        ax.axvline(vr["null_mean"], color="gray", linewidth=1, linestyle=":",
                   label=f"Null mean={vr['null_mean']:+.4f}")
        decision = full_results[ds_name]["decision"]
        color = "green" if "KEEP" in decision else "red"
        ax.set_title(f"{ds_name}\np={p_val:.4f} -> {decision}",
                     fontsize=11, color=color, fontweight="bold")
        ax.set_xlabel("D Macro-F1")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
    fig2.suptitle("Figure 2: Null Distribution with Real Delta", fontsize=13, fontweight="bold")
    fig2.tight_layout(rect=[0, 0, 1, 0.93])
    fig2.savefig(os.path.join(fig_dir, "fig2_null_histogram.png"), dpi=150, bbox_inches="tight")
    fig2.savefig(os.path.join(fig_dir, "fig2_null_histogram.pdf"), bbox_inches="tight")
    plt.close(fig2)

    # Figure 3
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 6))
    ds_names = [d[0] for d in DATASETS]
    mr_scores = [full_results[ds]["baseline"]["macro_f1"] for ds in ds_names]
    final_scores = [full_results[ds]["final"]["macro_f1"] for ds in ds_names]
    decisions_list = [full_results[ds]["decision"] for ds in ds_names]
    x_pos = np.arange(len(ds_names))
    width = 0.35
    bars1 = ax3.bar(x_pos - width/2, mr_scores, width, label="Canonical MR",
                    color="steelblue", edgecolor="black")
    bars2 = ax3.bar(x_pos + width/2, final_scores, width, label="Final Model",
                    color="darkorange", edgecolor="black")
    for i, (b1, b2, dec) in enumerate(zip(bars1, bars2, decisions_list)):
        ax3.text(b1.get_x() + b1.get_width()/2, b1.get_height() + 0.005,
                 f"{b1.get_height():.4f}", ha="center", va="bottom", fontsize=9)
        ax3.text(b2.get_x() + b2.get_width()/2, b2.get_height() + 0.005,
                 f"{b2.get_height():.4f}", ha="center", va="bottom", fontsize=9)
        color = "green" if "KEEP" in dec else "red"
        ax3.text(b2.get_x() + b2.get_width()/2, b2.get_height() - 0.03,
                 dec.replace("_", "\n"), ha="center", va="top", fontsize=7,
                 color=color, fontweight="bold")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(ds_names)
    ax3.set_ylabel("Macro-F1")
    ax3.set_title("Figure 3: Final Test Performance\nCanonical MR vs TURS-KTM-AV",
                   fontsize=12, fontweight="bold")
    ax3.legend()
    ax3.set_ylim(bottom=max(0, min(min(mr_scores), min(final_scores)) - 0.05))
    fig3.tight_layout()
    fig3.savefig(os.path.join(fig_dir, "fig3_final_test_performance.png"), dpi=150, bbox_inches="tight")
    fig3.savefig(os.path.join(fig_dir, "fig3_final_test_performance.pdf"), bbox_inches="tight")
    plt.close(fig3)
    log("  Figures saved")

    # Decision table
    log(f"\n  DECISION TABLE:")
    log(f"  {'Dataset':<18} {'D_real':>8} {'Null mean':>10} {'Null 95%':>10} {'p':>8} {'Decision':<14}")
    log("  " + "-" * 70)
    for d in all_decisions:
        log(f"  {d['dataset']:<18} {d['delta_real']:>+8.4f} {d['null_mean']:>+10.4f} "
            f"{d['null_95th']:>+10.4f} {d['p_value']:>8.4f} {d['decision']:<14}")

    # Final test table
    log(f"\n  FINAL TEST RESULTS:")
    log(f"  {'Dataset':<18} {'MR base':>8} {'Decision':<14} {'Model':<12} {'MF1':>8} {'D':>8}")
    log("  " + "-" * 70)
    for t in all_final_test:
        log(f"  {t['dataset']:<18} {t['mr_baseline_test_mf1']:>8.4f} {t['decision']:<14} "
            f"{t['final_model']:<12} {t['final_test_mf1']:>8.4f} {t['delta_test']:>+8.4f}")

    grand_t = time.time() - grand_t0
    log(f"\n  Total: {grand_t/60:.1f} min")
    log_fh.close()

    return full_results, all_decisions


if __name__ == "__main__":
    main()
