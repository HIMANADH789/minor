"""
TURS-KTM Experiment Runner
===========================
Runs the controlled feature-augmentation experiment:

  MR-2K       vs MR-2K + KTM(μ, W)
  MR-10K      vs MR-10K + KTM(μ, W)

on 4 biomedical datasets (ECG5000_UNBAL, ECG5000_BAL, CWRU_UNBAL, CWRU_BAL).

Canonical protocol (from benchmark_baselines.py / provenance doc):
  - MiniRocket(random_state=42, n_jobs=-1)
  - Per-sample z-normalization
  - RidgeClassifierCV(alphas=np.logspace(-4,4,20)) on train+val
  - NO post-hoc standardization for the baseline (raw features)
  - KTM blocks (μ, W) are standardized (fit on train+val, transform test)
    to ensure scale parity with PPV ∈ [0,1]

Outputs saved to results/turs_ktm/.
"""
import os, sys, json, time
import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.features.ktm import (
    ktm_transform, activation_count_summary, block_stats,
    feature_owners, concatenate_blocks, standardize_blocks,
)

SEED = 42
DATASETS = [
    ("ECG5000_UNBAL", "data/ecg5000_resplit.npz", 5),
    ("ECG5000_BAL",   "data/ecg5000_fair_balanced.npz", 5),
    ("CWRU_UNBAL",    "data/cwru_unbalanced.npz", 4),
    ("CWRU_BAL",      "data/cwru_balanced.npz", 4),
]
CAPACITIES = {"2K": 2016, "10K": 10000}

REF_VALUES = {
    "ECG5000_UNBAL": 0.5938,
    "ECG5000_BAL":   0.6553,
    "CWRU_UNBAL":    0.9917,
    "CWRU_BAL":      0.9947,
}

OUT_DIR = os.path.join(ROOT, "results", "turs_ktm")


def znorm(X):
    mu = X.mean(axis=-1, keepdims=True)
    sig = X.std(axis=-1, keepdims=True) + 1e-8
    return ((X - mu) / sig).astype(np.float32)


def load_split(ds_file):
    """Canonical split from benchmark_baselines.py."""
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
    """RidgeClassifierCV on train+val, evaluate on test."""
    clf = RidgeClassifierCV(alphas=np.logspace(-4, 4, 20))
    clf.fit(X_trva, y_trva)
    preds = clf.predict(X_te)
    acc = accuracy_score(y_te, preds)
    mf1 = f1_score(y_te, preds, average="macro", zero_division=0)
    wf1 = f1_score(y_te, preds, average="weighted", zero_division=0)
    per_class = f1_score(y_te, preds, average=None, zero_division=0,
                         labels=list(range(int(y_te.max()) + 1)))
    cm = confusion_matrix(y_te, preds).tolist()
    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(mf1), 4),
        "weighted_f1": round(float(wf1), 4),
        "class_f1s": [round(float(f), 4) for f in per_class],
        "confusion_matrix": cm,
        "alpha": round(float(clf.alpha_), 6),
        "n_features": int(X_trva.shape[1]),
    }


def main():
    os.makedirs(os.path.join(OUT_DIR, "configs"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "reports"), exist_ok=True)

    all_results = {}
    all_runtime = {}
    all_diagnostics = {}

    for ds_name, ds_file, n_cls in DATASETS:
        print(f"\n{'='*60}\n  {ds_name}\n{'='*60}", flush=True)

        # --- Load and preprocess ---
        t0 = time.time()
        Xtr, Xva, Xte, ytr, yva, yte = load_split(ds_file)
        Xn_tr, Xn_va, Xn_te = znorm(Xtr), znorm(Xva), znorm(Xte)
        y_trva = np.concatenate([ytr, yva])
        ds_results = {}
        ds_runtime = {}

        for cap_label, n_kernels in CAPACITIES.items():
            print(f"\n  --- {cap_label} (n_kernels={n_kernels}) ---", flush=True)
            cap_key = f"MR-{cap_label}"
            ktm_key = f"MR-{cap_label}+KTM"
            ds_results[cap_key] = {}
            ds_results[ktm_key] = {}

            # --- Fit MiniRocket ---
            from aeon.transformations.collection.convolution_based import MiniRocket
            t_mr_fit = time.time()
            mr = MiniRocket(n_kernels=n_kernels, random_state=SEED, n_jobs=-1)
            mr.fit_transform(Xn_tr[:, None, :])
            t_mr_fit = time.time() - t_mr_fit

            # --- Transform ---
            t_mr_tx = time.time()
            Z_tr = mr.transform(Xn_tr[:, None, :])
            Z_va = mr.transform(Xn_va[:, None, :])
            Z_te = mr.transform(Xn_te[:, None, :])
            t_mr_tx = time.time() - t_mr_tx

            # --- KTM extraction ---
            t_ktm = time.time()
            ppv_tr, mu_tr, W_tr, ct_tr = ktm_transform(
                Xn_tr, mr, return_counts=True)
            ppv_va, mu_va, W_va, ct_va = ktm_transform(
                Xn_va, mr, return_counts=True)
            ppv_te, mu_te, W_te, ct_te = ktm_transform(
                Xn_te, mr, return_counts=True)
            t_ktm = time.time() - t_ktm

            # --- Audit: PPV from KTM path vs aeon transform ---
            ppv_audit_max = max(np.abs(ppv_tr - Z_tr).max(),
                                np.abs(ppv_va - Z_va).max(),
                                np.abs(ppv_te - Z_te).max())
            print(f"    PPV audit max diff: {ppv_audit_max:.2e}", flush=True)
            # Count zero-activation pairs
            zero_frac = (ct_tr == 0).mean()
            mean_count = ct_tr.mean()
            print(f"    Zero-act fraction: {zero_frac:.3f}, "
                  f"mean count: {mean_count:.1f}", flush=True)

            # --- Variant A/B0: raw MR features (canonical, no standardization) ---
            print(f"    Running {cap_key} (baseline)...", flush=True)
            t_ridge = time.time()
            r_base = run_ridge(
                np.concatenate([Z_tr, Z_va]), y_trva,
                Z_te, yte)
            ds_results[cap_key]["raw"] = r_base
            ds_results[cap_key]["raw"]["time_s"] = round(time.time() - t_ridge, 1)
            print(f"    {cap_key}: MF1={r_base['macro_f1']:.4f}", flush=True)

            # --- Variant A/B1: raw MR || standardized(μ, W) ---
            print(f"    Running {ktm_key}...", flush=True)
            t_ridge = time.time()
            # Build blocks: raw MR || standardized μ || standardized W
            # Standardize μ and W using train+val statistics
            mu_trva = np.concatenate([mu_tr, mu_va])
            W_trva = np.concatenate([W_tr, W_va])
            mu_te_arr = mu_te
            W_te_arr = W_te

            # Fit scaler on train+val for μ block
            mu_mean, mu_std = mu_trva.mean(0, keepdims=True), mu_trva.std(0, keepdims=True)
            mu_std = np.where(mu_std < 1e-8, 1.0, mu_std)
            mu_tr_s = (mu_tr - mu_mean) / mu_std
            mu_va_s = (mu_va - mu_mean) / mu_std
            mu_te_s = (mu_te_arr - mu_mean) / mu_std

            # Fit scaler on train+val for W block
            W_mean, W_std = W_trva.mean(0, keepdims=True), W_trva.std(0, keepdims=True)
            W_std = np.where(W_std < 1e-8, 1.0, W_std)
            W_tr_s = (W_tr - W_mean) / W_std
            W_va_s = (W_va - W_mean) / W_std
            W_te_s = (W_te_arr - W_mean) / W_std

            # Combine: raw MR || std(μ) || std(W)
            F_tr_ktm = np.concatenate([Z_tr, mu_tr_s, W_tr_s], axis=1)
            F_va_ktm = np.concatenate([Z_va, mu_va_s, W_va_s], axis=1)
            F_te_ktm = np.concatenate([Z_te, mu_te_s, W_te_s], axis=1)

            r_ktm = run_ridge(
                np.concatenate([F_tr_ktm, F_va_ktm]), y_trva,
                F_te_ktm, yte)
            ds_results[ktm_key]["raw_ktm"] = r_ktm
            ds_results[ktm_key]["raw_ktm"]["time_s"] = round(
                time.time() - t_ridge, 1)
            print(f"    {ktm_key}: MF1={r_ktm['macro_f1']:.4f}", flush=True)

            # --- Runtime ---
            ds_runtime[cap_key] = {
                "mr_fit_s": round(t_mr_fit, 1),
                "mr_transform_s": round(t_mr_tx, 1),
                "ktm_extract_s": round(t_ktm, 1),
                "total_s": round(t_mr_fit + t_mr_tx + t_ktm, 1),
            }

            # --- Diagnostics ---
            F_mu_block = np.concatenate([mu_tr, mu_va])
            F_W_block = np.concatenate([W_tr, W_va])
            F_mr_block = np.concatenate([Z_tr, Z_va])
            diag = {
                "mr_block": block_stats(F_mr_block.astype(np.float64)),
                "mu_block": block_stats(F_mu_block.astype(np.float64)),
                "W_block": block_stats(F_W_block.astype(np.float64)),
                "counts": {
                    "zero_frac": float(zero_frac),
                    "mean_count": float(mean_count),
                },
            }
            all_diagnostics[f"{ds_name}_{cap_label}"] = diag

        # --- B0 baseline verification ---
        ref_10k = REF_VALUES.get(ds_name)
        if ref_10k is not None:
            actual_10k = ds_results["MR-10K"]["raw"]["macro_f1"]
            delta = abs(actual_10k - ref_10k)
            print(f"\n  Baseline check: {actual_10k:.4f} vs ref {ref_10k:.4f} "
                  f"(delta={delta:.4f})", flush=True)
            if delta > 0.02:
                print("  WARNING: B0 deviates materially from reference!",
                      flush=True)
            ds_results["baseline_check"] = {
                "ref": ref_10k, "actual": actual_10k, "delta": round(delta, 4)}

        all_results[ds_name] = ds_results
        all_runtime[ds_name] = ds_runtime

    # --- Primary results table ---
    print(f"\n{'='*60}\n  PRIMARY RESULTS TABLE\n{'='*60}")
    header = f"{'Dataset':<18} {'MR-2K':>7} {'MR-2K+KTM':>10} {'D2K':>7} {'MR-10K':>7} {'MR-10K+KTM':>11} {'D10K':>7}"
    print(header)
    print("-" * len(header))
    for ds_name, _, _ in DATASETS:
        r = all_results[ds_name]
        mr2k = r["MR-2K"]["raw"]["macro_f1"]
        mr2k_ktm = r["MR-2K+KTM"]["raw_ktm"]["macro_f1"]
        mr10k = r["MR-10K"]["raw"]["macro_f1"]
        mr10k_ktm = r["MR-10K+KTM"]["raw_ktm"]["macro_f1"]
        d2k = mr2k_ktm - mr2k
        d10k = mr10k_ktm - mr10k
        print(f"{ds_name:<18} {mr2k:>7.4f} {mr2k_ktm:>10.4f} {d2k:>+7.4f}"
              f" {mr10k:>7.4f} {mr10k_ktm:>11.4f} {d10k:>+7.4f}")
    # Mean row
    means = {}
    for cap in ["2K", "10K"]:
        base_vals = [all_results[ds]["MR-" + cap]["raw"]["macro_f1"]
                     for ds, _, _ in DATASETS]
        ktm_vals = [all_results[ds]["MR-" + cap + "+KTM"]["raw_ktm"]["macro_f1"]
                    for ds, _, _ in DATASETS]
        means[cap] = (np.mean(base_vals), np.mean(ktm_vals),
                      np.mean(ktm_vals) - np.mean(base_vals))
    print("-" * len(header))
    print(f"{'Mean':<18} {means['2K'][0]:>7.4f} {means['2K'][1]:>10.4f}"
          f" {means['2K'][2]:>+7.4f}"
          f" {means['10K'][0]:>7.4f} {means['10K'][1]:>11.4f}"
          f" {means['10K'][2]:>+7.4f}")

    # --- Save ---
    with open(os.path.join(OUT_DIR, "full_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    with open(os.path.join(OUT_DIR, "runtime.json"), "w") as f:
        json.dump(all_runtime, f, indent=2)
    with open(os.path.join(OUT_DIR, "diagnostics.json"), "w") as f:
        json.dump(all_diagnostics, f, indent=2)
    print(f"\n  Results saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
