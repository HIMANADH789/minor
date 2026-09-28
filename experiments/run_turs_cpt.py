"""
TURS-CPT Experiment Runner
===========================
Tests whether intrinsic-cycle-anchored phase timing provides complementary
information to MiniROCKET PPV features.

Conditions:
  A0: MiniROCKET(2K)
  A1: MiniROCKET(2K) + CPT
  B0: MiniROCKET(10K canonical)
  B1: MiniROCKET(10K canonical) + CPT

Primary result: B0 vs B1.
"""
import os, sys, json, time
import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.features.cpt import (
    estimate_period, cpt_transform,
    rank_kernels_by_discriminative_power, select_top_k_transformers,
    period_summary,
)
from src.features.ktm import ktm_transform, block_stats, feature_owners

SEED = 42
K_TOP = 1024  # number of kernels to select

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

OUT_DIR = os.path.join(ROOT, "results", "turs_cpt")


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
    mf1 = float(f1_score(y_te, preds, average="macro", zero_division=0))
    acc = float(accuracy_score(y_te, preds))
    wf1 = float(f1_score(y_te, preds, average="weighted", zero_division=0))
    per_class = f1_score(y_te, preds, average=None, zero_division=0,
                         labels=list(range(int(y_te.max()) + 1)))
    cm = confusion_matrix(y_te, preds).tolist()
    return {
        "macro_f1": round(mf1, 4),
        "accuracy": round(acc, 4),
        "weighted_f1": round(wf1, 4),
        "class_f1s": [round(float(f), 4) for f in per_class],
        "confusion_matrix": cm,
        "alpha": round(float(clf.alpha_), 6),
        "n_features": int(X_trva.shape[1]),
    }


def std_fit(train, val, test):
    """Standardize using train+val statistics."""
    trva = np.concatenate([train, val])
    m = trva.mean(0, keepdims=True)
    s = trva.std(0, keepdims=True)
    s = np.where(s < 1e-8, 1.0, s)
    return (train - m) / s, (val - m) / s, (test - m) / s


def main():
    os.makedirs(os.path.join(OUT_DIR, "configs"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "reports"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "figures"), exist_ok=True)

    all_results = {}
    all_runtime = {}
    all_diagnostics = {}
    all_periods = {}

    for ds_name, ds_file, n_cls in DATASETS:
        print(f"\n{'='*60}\n  {ds_name}\n{'='*60}", flush=True)

        t0 = time.time()
        Xtr, Xva, Xte, ytr, yva, yte = load_split(ds_file)
        Xn_tr, Xn_va, Xn_te = znorm(Xtr), znorm(Xva), znorm(Xte)
        y_trva = np.concatenate([ytr, yva])
        ds_results = {}
        ds_runtime = {}
        ds_periods = {}

        for cap_label, n_kernels in CAPACITIES.items():
            print(f"\n  --- {cap_label} (n_kernels={n_kernels}) ---", flush=True)
            cap_key = f"MR-{cap_label}"
            cpt_key = f"MR-{cap_label}+CPT"

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

            # --- Verify baseline ---
            r_base = run_ridge(
                np.concatenate([Z_tr, Z_va]), y_trva, Z_te, yte)
            ds_results[cap_key] = {"raw": r_base}
            print(f"    {cap_key}: MF1={r_base['macro_f1']:.4f}", flush=True)

            # --- Estimate periods (data-driven, no labels) ---
            t_period = time.time()
            X_all = np.concatenate([Xn_tr, Xn_va, Xn_te])
            periods_all = np.array([estimate_period(X_all[i])[0]
                                    for i in range(len(X_all))])
            methods_all = np.array([estimate_period(X_all[i])[1]
                                    for i in range(len(X_all))])
            t_period = time.time() - t_period

            periods_tr = periods_all[:len(Xn_tr)]
            periods_va = periods_all[len(Xn_tr):len(Xn_tr) + len(Xn_va)]
            periods_te = periods_all[len(Xn_tr) + len(Xn_va):]

            ds_periods[cap_label] = period_summary(periods_all, methods_all)
            print(f"    Period: mean={ds_periods[cap_label]['mean']:.1f}, "
                  f"ACF={ds_periods[cap_label]['acf_frac']:.1%}", flush=True)

            # --- Top-K kernel selection (training data only) ---
            t_rank = time.time()
            ranks, scores = rank_kernels_by_discriminative_power(Z_tr, ytr)
            K_actual = min(K_TOP, Z_tr.shape[1] // 84 *
                           len(mr.parameters[2]))  # total kernels
            kernel_mask, feat_mask = select_top_k_transformers(
                mr, ranks, K=min(K_TOP, K_actual))
            n_selected_kernels = kernel_mask.sum()
            print(f"    Top-K: selected {n_selected_kernels} kernels "
                  f"({feat_mask.sum()} features)", flush=True)
            t_rank = time.time() - t_rank

            # --- CPT extraction ---
            t_cpt = time.time()
            conc_tr, phase_tr, ct_tr = cpt_transform(
                Xn_tr, mr, periods_tr, return_counts=True)
            conc_va, phase_va, ct_va = cpt_transform(
                Xn_va, mr, periods_va, return_counts=True)
            conc_te, phase_te, ct_te = cpt_transform(
                Xn_te, mr, periods_te, return_counts=True)

            # Apply top-K mask
            conc_tr_k = conc_tr[:, feat_mask]
            phase_tr_k = phase_tr[:, feat_mask]
            conc_va_k = conc_va[:, feat_mask]
            phase_va_k = phase_va[:, feat_mask]
            conc_te_k = conc_te[:, feat_mask]
            phase_te_k = phase_te[:, feat_mask]
            t_cpt = time.time() - t_cpt

            # --- Variant A1/B1: MR || standardized(CPT) ---
            print(f"    Running {cpt_key}...", flush=True)
            t_ridge = time.time()

            # Standardize blocks
            conc_tr_s, conc_va_s, conc_te_s = std_fit(
                conc_tr_k, conc_va_k, conc_te_k)
            phase_tr_s, phase_va_s, phase_te_s = std_fit(
                phase_tr_k, phase_va_k, phase_te_k)

            # Combine: raw MR || std(concentration) || std(phase)
            F_tr_cpt = np.concatenate(
                [Z_tr, conc_tr_s, phase_tr_s], axis=1)
            F_va_cpt = np.concatenate(
                [Z_va, conc_va_s, phase_va_s], axis=1)
            F_te_cpt = np.concatenate(
                [Z_te, conc_te_s, phase_te_s], axis=1)

            r_cpt = run_ridge(
                np.concatenate([F_tr_cpt, F_va_cpt]), y_trva,
                F_te_cpt, yte)
            ds_results[cpt_key] = {"raw_cpt": r_cpt}
            ds_results[cpt_key]["raw_cpt"]["time_s"] = round(
                time.time() - t_ridge, 1)
            print(f"    {cpt_key}: MF1={r_cpt['macro_f1']:.4f}", flush=True)

            # --- Runtime ---
            ds_runtime[cap_key] = {
                "mr_fit_s": round(t_mr_fit, 1),
                "mr_transform_s": round(t_mr_tx, 1),
                "period_est_s": round(t_period, 1),
                "kernel_rank_s": round(t_rank, 1),
                "cpt_extract_s": round(t_cpt, 1),
                "total_s": round(t_mr_fit + t_mr_tx + t_period +
                                 t_rank + t_cpt, 1),
            }

            # --- Diagnostics ---
            diag = {
                "mr_block": block_stats(Z_tr.astype(np.float64)),
                "conc_block": block_stats(conc_tr_k.astype(np.float64)),
                "phase_block": block_stats(phase_tr_k.astype(np.float64)),
                "counts": {
                    "zero_frac": float((ct_tr == 0).mean()),
                    "mean_count": float(ct_tr.mean()),
                },
                "n_selected_kernels": int(n_selected_kernels),
                "n_selected_features": int(feat_mask.sum()),
                "period": ds_periods[cap_label],
            }
            all_diagnostics[f"{ds_name}_{cap_label}"] = diag

        all_results[ds_name] = ds_results
        all_runtime[ds_name] = ds_runtime
        all_periods[ds_name] = ds_periods

    # --- Primary results table ---
    print(f"\n{'='*60}\n  PRIMARY RESULTS TABLE\n{'='*60}")
    header = (f"{'Dataset':<18} {'MR-2K':>7} {'MR-2K+CPT':>10} {'D2K':>7} "
              f"{'MR-10K':>7} {'MR-10K+CPT':>11} {'D10K':>7}")
    print(header)
    print("-" * len(header))
    for ds_name, _, _ in DATASETS:
        r = all_results[ds_name]
        mr2k = r["MR-2K"]["raw"]["macro_f1"]
        mr2k_cpt = r["MR-2K+CPT"]["raw_cpt"]["macro_f1"]
        mr10k = r["MR-10K"]["raw"]["macro_f1"]
        mr10k_cpt = r["MR-10K+CPT"]["raw_cpt"]["macro_f1"]
        d2k = mr2k_cpt - mr2k
        d10k = mr10k_cpt - mr10k
        print(f"{ds_name:<18} {mr2k:>7.4f} {mr2k_cpt:>10.4f} {d2k:>+7.4f}"
              f" {mr10k:>7.4f} {mr10k_cpt:>11.4f} {d10k:>+7.4f}")
    # Mean row
    base_2k = [all_results[ds]["MR-2K"]["raw"]["macro_f1"]
               for ds, _, _ in DATASETS]
    cpt_2k = [all_results[ds]["MR-2K+CPT"]["raw_cpt"]["macro_f1"]
              for ds, _, _ in DATASETS]
    base_10k = [all_results[ds]["MR-10K"]["raw"]["macro_f1"]
                for ds, _, _ in DATASETS]
    cpt_10k = [all_results[ds]["MR-10K+CPT"]["raw_cpt"]["macro_f1"]
               for ds, _, _ in DATASETS]
    print("-" * len(header))
    print(f"{'Mean':<18} {np.mean(base_2k):>7.4f} {np.mean(cpt_2k):>10.4f}"
          f" {np.mean(cpt_2k)-np.mean(base_2k):>+7.4f}"
          f" {np.mean(base_10k):>7.4f} {np.mean(cpt_10k):>11.4f}"
          f" {np.mean(cpt_10k)-np.mean(base_10k):>+7.4f}")

    # --- Save ---
    with open(os.path.join(OUT_DIR, "full_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    with open(os.path.join(OUT_DIR, "runtime.json"), "w") as f:
        json.dump(all_runtime, f, indent=2)
    with open(os.path.join(OUT_DIR, "diagnostics.json"), "w") as f:
        json.dump(all_diagnostics, f, indent=2)
    with open(os.path.join(OUT_DIR, "period_diagnostics.json"), "w") as f:
        json.dump(all_periods, f, indent=2)
    print(f"\n  Results saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
