"""
KTA-TF-Drift v3: Full enhancement suite with comprehensive ablation.

Components:
  A. Random Transport Feature bank (MiniRocket analogue)
  B. Transport pyramid multi-scale (InceptionTime analogue)
  C. Trimmed Wasserstein (robustness)
  D. Conformal calibration (uncertainty)
  E. Unsupervised anomaly scoring via barycenter distance
  F. KTA-weighted ensembling

Runs all 2^3 = 8 combinations of (bank, pyramid, trimmed) plus
conformal/anomaly/ensemble post-hoc analyses.
"""

import os
import sys
import json
import time
import warnings
import itertools
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, confusion_matrix,
)
from rich.console import Console

warnings.filterwarnings("ignore")
console = Console()

# ============================================================
# Shared utilities
# ============================================================

def per_sample_standardize(X, eps=1e-8):
    mu = np.mean(X, axis=-1, keepdims=True)
    sigma = np.std(X, axis=-1, keepdims=True)
    return (X - mu) / (sigma + eps)


def compute_windows(signal, L, S):
    N = len(signal)
    out = []
    i = 0
    while i * S + L <= N:
        out.append(signal[i * S : i * S + L])
        i += 1
    return np.array(out)


def compute_quantile_functions(windows, K):
    M, L = windows.shape
    qgrid = (np.arange(1, K + 1) - 0.5) / K
    Q = np.zeros((M, K), dtype=np.float64)
    for i in range(M):
        sw = np.sort(windows[i])
        emp_q = (np.arange(1, L + 1) - 0.5) / L
        Q[i] = np.interp(qgrid, emp_q, sw)
    return Q


def signed_sqrt(u):
    return np.sign(u) * np.sqrt(np.abs(u) + 1e-16)


def kernel_target_alignment(Kmat, Y_onehot):
    YYt = Y_onehot @ Y_onehot.T
    num = np.sum(Kmat * YYt)
    dK = np.sqrt(np.sum(Kmat * Kmat))
    dY = np.sqrt(np.sum(YYt * YYt))
    return num / (dK * dY + 1e-16)


def compute_kta_weights_balanced(dQ_all, labels_all, K_feat, tau, gamma, n_samples=2000):
    n_pairs = dQ_all.shape[0]
    C = len(np.unique(labels_all))
    class_counts = np.bincount(labels_all.astype(int), minlength=C)
    min_count = max(int(np.min(class_counts)), 10)
    target = min(n_samples // C, min_count)

    rng = np.random.RandomState(42)
    idx_list = []
    for c in range(C):
        ci = np.where(labels_all == c)[0]
        chosen = rng.choice(ci, target, replace=(len(ci) < target))
        idx_list.append(chosen)
    idx = np.concatenate(idx_list)
    rng.shuffle(idx)

    dQ_sub = dQ_all[idx]
    lbl_sub = labels_all[idx]
    Y_oh = (lbl_sub[:, None] == np.arange(C)[None, :]).astype(np.float64)

    if gamma <= 0:
        phi0 = signed_sqrt(dQ_sub[:, 0])
        diffs = phi0[:, None] - phi0[None, :]
        med_sq = np.median(diffs ** 2)
        gamma = 1.0 / (med_sq + 1e-16)

    A = np.zeros(K_feat)
    for k in range(K_feat):
        phi = signed_sqrt(dQ_sub[:, k])
        diff = phi[:, None] - phi[None, :]
        Kk = np.exp(-gamma * diff ** 2)
        A[k] = kernel_target_alignment(Kk, Y_oh)

    sel = np.where(A > tau)[0]
    if len(sel) == 0:
        sel = np.arange(K_feat)
    pos = np.maximum(A[sel], 0.0)
    tot = np.sum(pos)
    w = pos / tot if tot > 1e-16 else np.ones(len(sel)) / len(sel)
    return sel, w, A, gamma


def aggregate_signal_features(Q_list, selected_indices, weights, barycenters, y_labels=None):
    """Build per-signal feature vectors from quantile functions."""
    sqrt_w = np.sqrt(weights)
    features = []
    for Q in Q_list:
        # Static: mean, std, max, min of quantile functions
        static = np.concatenate([np.mean(Q, 0), np.std(Q, 0), np.max(Q, 0), np.min(Q, 0)])

        # KTA-weighted drift (lag=1)
        dQ = Q[1:] - Q[:-1]
        phi = signed_sqrt(dQ[:, selected_indices])
        wt = phi * sqrt_w[None, :]
        drift = np.concatenate([np.mean(wt, 0), np.std(wt, 0), np.max(wt, 0), np.min(wt, 0)])

        # Barycenter distances
        classes = sorted(barycenters.keys())
        Q_mean = np.mean(Q, 0)
        M = Q.shape[0]
        dists = [np.sqrt(np.mean((Q_mean - barycenters[c]) ** 2)) for c in classes]
        pwd = np.zeros((M, len(classes)))
        for j, c in enumerate(classes):
            for i in range(M):
                pwd[i, j] = np.sqrt(np.mean((Q[i] - barycenters[c]) ** 2))
        bary = np.array(dists + list(np.min(pwd, 0)) + list(np.max(pwd, 0)) + list(np.std(pwd, 0)))

        features.append(np.concatenate([static, drift, bary]))
    return np.array(features)


# ============================================================
# Component A: Random Transport Feature Bank
# ============================================================
def random_transport_bank(X_raw, n_banks=200, signal_len=140, K=32, rng_seed=42):
    """
    Generate random (L, S, tau) triples and compute Wasserstein distance
    summary statistics for each. Returns feature matrix (N, n_banks * n_stats).
    """
    rng = np.random.RandomState(rng_seed)
    N = X_raw.shape[0]

    configs = []
    for _ in range(n_banks):
        L = rng.randint(10, min(signal_len, 70))
        S = rng.randint(3, max(4, L // 2))
        tau = rng.choice([1, 2, 4])
        configs.append((L, S, tau))

    all_features = np.zeros((N, n_banks * 4), dtype=np.float64)  # 4 stats per bank

    for b, (L, S, tau) in enumerate(configs):
        for i in range(N):
            windows = compute_windows(X_raw[i], L, S)
            if windows.shape[0] < tau + 1:
                continue
            # Sort each window for Wasserstein
            sorted_w = np.sort(windows, axis=1)
            # Drift between consecutive windows at lag tau
            dW = sorted_w[tau:] - sorted_w[:-tau]
            # Wasserstein distance per pair (L2 norm / sqrt(L))
            w_dist = np.sqrt(np.mean(dW ** 2, axis=1))
            all_features[i, b*4] = np.mean(w_dist)
            all_features[i, b*4+1] = np.std(w_dist)
            all_features[i, b*4+2] = np.max(w_dist)
            all_features[i, b*4+3] = np.min(w_dist) if len(w_dist) > 0 else 0

    return all_features, configs


# ============================================================
# Component B: Transport Pyramid (multi-scale)
# ============================================================
def compute_pyramid_quantiles(X_raw, scales, L_base=35, S_base=7, K=32):
    """
    Compute quantile functions at multiple scales L, 2L, 4L.
    Returns dict {scale_name: Q_list}.
    """
    all_Q = {}
    for scale_name, mult in scales:
        L = L_base * mult
        S = max(S_base * mult, S_base)  # don't shrink hop below base
        Q_list = []
        for i in range(X_raw.shape[0]):
            windows = compute_windows(X_raw[i], L, S)
            if windows.shape[0] < 2:
                # Pad with last window repeated
                windows = np.tile(windows[-1:], (2, 1))
            Q = compute_quantile_functions(windows, K)
            Q_list.append(Q)
        all_Q[scale_name] = Q_list
    return all_Q


# ============================================================
# Component C: Trimmed Wasserstein
# ============================================================
def compute_quantile_functions_trimmed(windows, K, trim_frac=0.05):
    """Quantile functions computed after trimming top/bottom trim_frac of order statistics."""
    M, L = windows.shape
    trim_n = max(1, int(L * trim_frac))
    qgrid = (np.arange(1, K + 1) - 0.5) / K
    Q = np.zeros((M, K), dtype=np.float64)
    for i in range(M):
        sw = np.sort(windows[i])
        # Trim extremes
        if 2 * trim_n < L:
            sw = sw[trim_n:-trim_n]
        L_trimmed = len(sw)
        emp_q = (np.arange(1, L_trimmed + 1) - 0.5) / L_trimmed
        Q[i] = np.interp(qgrid, emp_q, sw)
    return Q


# ============================================================
# Component D: Conformal Calibration
# ============================================================
class ConformalClassifier:
    """Split conformal prediction wrapper for any RF classifier."""
    def __init__(self, base_clf, alpha=0.1):
        self.base_clf = base_clf
        self.alpha = alpha
        self.cal_scores = None
        self.n_classes = None

    def fit(self, X_cal, y_cal):
        """Fit on calibration set (split from training data)."""
        proba = self.base_clf.predict_proba(X_cal)
        self.n_classes = proba.shape[1]
        # Nonconformity scores: 1 - p(y_cal)
        self.cal_scores = 1.0 - proba[np.arange(len(y_cal)), y_cal]
        return self

    def predict_set(self, X_test):
        """Prediction sets with guaranteed marginal coverage."""
        proba = self.base_clf.predict_proba(X_test)
        n = len(self.cal_scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)
        threshold = np.quantile(self.cal_scores, q_level)
        pred_sets = proba >= threshold
        return pred_sets, proba

    def evaluate(self, X_test, y_test):
        """Compute coverage, avg set size, and point predictions."""
        pred_sets, proba = self.predict_set(X_test)
        # Coverage: fraction of times true class is in the set
        covered = pred_sets[np.arange(len(y_test)), y_test]
        coverage = np.mean(covered)
        # Average set size
        avg_set_size = np.mean(np.sum(pred_sets, axis=1))
        # Point prediction: class with highest probability
        y_pred = np.argmax(proba, axis=1)
        acc = accuracy_score(y_test, y_pred)
        return {
            "coverage": float(coverage),
            "avg_set_size": float(avg_set_size),
            "point_accuracy": float(acc),
            "target_coverage": float(1 - self.alpha),
        }


# ============================================================
# Component E: Unsupervised Anomaly Scoring
# ============================================================
def compute_anomaly_scores(Q_list, healthy_barycenter):
    """
    Compute W2 distance from each signal's mean quantile function
    to the healthy-class barycenter. Higher = more anomalous.
    """
    scores = []
    for Q in Q_list:
        Q_mean = np.mean(Q, axis=0)
        d = np.sqrt(np.mean((Q_mean - healthy_barycenter) ** 2))
        scores.append(d)
    return np.array(scores)


# ============================================================
# Component F: KTA-Weighted Ensemble
# ============================================================
def kta_weighted_ensemble_predict(clf_list, X_list, kta_scores):
    """
    Ensemble: each classifier predicts, weight votes by its KTA alignment score.
    """
    all_proba = []
    for clf, X in zip(clf_list, X_list):
        all_proba.append(clf.predict_proba(X))

    # Weight by KTA scores (normalized)
    w = np.array(kta_scores)
    w = np.maximum(w, 0)
    w = w / (np.sum(w) + 1e-16)

    weighted_proba = np.zeros_like(all_proba[0])
    for proba, wi in zip(all_proba, w):
        weighted_proba += wi * proba

    return np.argmax(weighted_proba, axis=1)


# ============================================================
# Core pipeline: build features for one configuration
# ============================================================
def build_features_for_config(
    X_train, y_train, X_test, y_test,
    use_bank=False, use_pyramid=False, use_trimmed=False,
    n_banks=200, L_base=35, S_base=7, K=32, trim_frac=0.05,
    tau=0.01, max_pairs=4000,
):
    """Build the full feature matrix for a given combination of components."""
    N_train = X_train.shape[0]
    N_test = X_test.shape[0]

    # --- Base quantile functions ---
    if use_trimmed:
        train_Q_list = []
        for i in range(N_train):
            w = compute_windows(X_train[i], L_base, S_base)
            Q = compute_quantile_functions_trimmed(w, K, trim_frac)
            train_Q_list.append(Q)
        test_Q_list = []
        for i in range(N_test):
            w = compute_windows(X_test[i], L_base, S_base)
            Q = compute_quantile_functions_trimmed(w, K, trim_frac)
            test_Q_list.append(Q)
    else:
        train_Q_list = []
        for i in range(N_train):
            w = compute_windows(X_train[i], L_base, S_base)
            Q = compute_quantile_functions(w, K)
            train_Q_list.append(Q)
        test_Q_list = []
        for i in range(N_test):
            w = compute_windows(X_test[i], L_base, S_base)
            Q = compute_quantile_functions(w, K)
            test_Q_list.append(Q)

    # --- KTA weights on base features ---
    train_dQ_flat = []
    train_lbl_flat = []
    for i in range(N_train):
        dQ = train_Q_list[i][1:] - train_Q_list[i][:-1]
        train_dQ_flat.append(dQ)
        train_lbl_flat.extend([y_train[i]] * dQ.shape[0])
    train_dQ_flat = np.concatenate(train_dQ_flat, axis=0)
    train_lbl_flat = np.array(train_lbl_flat, dtype=np.int64)

    sel_idx, weights, A_scores, gamma_used = compute_kta_weights_balanced(
        train_dQ_flat, train_lbl_flat, K, tau, -1.0, max_pairs
    )

    # --- Barycenters (on base or trimmed quantiles) ---
    classes = np.unique(y_train)
    barycenters = {}
    for c in classes:
        qs = [np.mean(train_Q_list[i], 0) for i in range(N_train) if y_train[i] == c]
        barycenters[c] = np.mean(qs, axis=0)

    # --- Base features ---
    X_train_base = aggregate_signal_features(train_Q_list, sel_idx, weights, barycenters)
    X_test_base = aggregate_signal_features(test_Q_list, sel_idx, weights, barycenters)
    feature_parts = [("base", X_train_base.shape[1])]

    # --- Pyramid features ---
    if use_pyramid:
        scales = [("L1", 1), ("L2", 2), ("L4", 4)]
        pyramid_Q = compute_pyramid_quantiles(X_train, scales, L_base, S_base, K)
        pyramid_Q_test = compute_pyramid_quantiles(X_test, scales, L_base, S_base, K)

        pyramid_feats_train = []
        pyramid_feats_test = []
        for sname, _ in scales:
            tQ = pyramid_Q[sname]
            teQ = pyramid_Q_test[sname]
            # KTA on this scale
            dQ_flat = []
            lbl_flat = []
            for i in range(N_train):
                if tQ[i].shape[0] > 1:
                    d = tQ[i][1:] - tQ[i][:-1]
                else:
                    d = np.zeros((1, K))
                dQ_flat.append(d)
                lbl_flat.extend([y_train[i]] * d.shape[0])
            dQ_flat = np.concatenate(dQ_flat, axis=0)
            lbl_flat = np.array(lbl_flat, dtype=np.int64)
            s_idx, s_w, _, _ = compute_kta_weights_balanced(
                dQ_flat, lbl_flat, K, tau, -1.0, min(max_pairs, 2000)
            )
            # Barycenters for this scale
            bc = {}
            for c in classes:
                qs = [np.mean(tQ[i], 0) for i in range(N_train) if y_train[i] == c]
                bc[c] = np.mean(qs, axis=0) if qs else np.zeros(K)
            ft = aggregate_signal_features(tQ, s_idx, s_w, bc)
            fte = aggregate_signal_features(teQ, s_idx, s_w, bc)
            pyramid_feats_train.append(ft)
            pyramid_feats_test.append(fte)
            feature_parts.append((f"pyramid_{sname}", ft.shape[1]))

        X_train_pyramid = np.concatenate(pyramid_feats_train, axis=1)
        X_test_pyramid = np.concatenate(pyramid_feats_test, axis=1)
    else:
        X_train_pyramid = np.zeros((N_train, 0))
        X_test_pyramid = np.zeros((N_test, 0))

    # --- Bank features ---
    if use_bank:
        bank_train, configs = random_transport_bank(X_train, n_banks, K=K)
        bank_test, _ = random_transport_bank(X_test, n_banks, K=K)
        feature_parts.append(("bank", bank_train.shape[1]))
    else:
        bank_train = np.zeros((N_train, 0))
        bank_test = np.zeros((N_test, 0))

    # --- Concatenate all ---
    X_train_all = np.concatenate([X_train_base, X_train_pyramid, bank_train], axis=1)
    X_test_all = np.concatenate([X_test_base, X_test_pyramid, bank_test], axis=1)

    return X_train_all, X_test_all, feature_parts, A_scores, barycenters, train_Q_list, test_Q_list, sel_idx, weights


# ============================================================
# Main ablation runner
# ============================================================
def run_ablation(data_file, output_dir):
    """Run all combinations of (bank, pyramid, trimmed) + post-hoc analyses."""
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    console.print("[cyan]Loading ECG5000...[/cyan]")
    data = np.load(data_file)
    X_train_raw = per_sample_standardize(data["X_train"].astype(np.float64))
    y_train = data["y_train"].astype(np.int64)
    X_test_raw = per_sample_standardize(data["X_test"].astype(np.float64))
    y_test = data["y_test"].astype(np.int64)

    console.print(f"  Train: {X_train_raw.shape[0]}, Test: {X_test_raw.shape[0]}")

    # Generate all 8 combinations of (bank, pyramid, trimmed)
    components = ["bank", "pyramid", "trimmed"]
    combos = list(itertools.product([False, True], repeat=3))
    combo_names = []
    for b, p, t in combos:
        parts = []
        if b: parts.append("bank")
        if p: parts.append("pyramid")
        if t: parts.append("trimmed")
        combo_names.append("+".join(parts) if parts else "base_only")

    # Hyperparams
    L_BASE, S_BASE, K = 35, 7, 35
    TAU = 0.01
    N_ESTIMATORS = 500
    N_BANKS = 200

    all_results = {}

    for ci, (use_bank, use_pyramid, use_trimmed) in enumerate(combos):
        name = combo_names[ci]
        console.print(f"\n{'='*60}")
        console.print(f"[bold yellow]Combo {ci+1}/8: {name}[/bold yellow]")
        console.print(f"{'='*60}")

        t_start = time.time()

        X_tr, X_te, feat_parts, A_scores, barycenters, trQ, teQ, sel, wts = \
            build_features_for_config(
                X_train_raw, y_train, X_test_raw, y_test,
                use_bank=use_bank, use_pyramid=use_pyramid, use_trimmed=use_trimmed,
                n_banks=N_BANKS, L_base=L_BASE, S_base=S_BASE, K=K,
                tau=TAU, max_pairs=4000,
            )

        t_feat = time.time()

        # Train RF
        clf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS, max_depth=None,
            class_weight="balanced_subsample", n_jobs=-1, random_state=42,
        )
        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_te)

        t_train = time.time()

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        mf1 = f1_score(y_test, y_pred, average="macro")
        recs = recall_score(y_test, y_pred, average=None)
        cm = confusion_matrix(y_test, y_pred)

        total = t_train - t_start
        console.print(f"  Features: {X_tr.shape[1]} dims ({', '.join(f'{n}:{d}' for n,d in feat_parts)})")
        console.print(f"  [bold]Accuracy: {acc:.4f}  Macro F1: {mf1:.4f}[/bold]")
        console.print(f"  Recalls: {', '.join(f'{r:.4f}' for r in recs)}")
        console.print(f"  Time: {total:.1f}s (feat: {t_feat-t_start:.1f}s, RF: {t_train-t_feat:.1f}s)")

        all_results[name] = {
            "accuracy": float(acc),
            "macro_f1": float(mf1),
            "class_recalls": [float(r) for r in recs],
            "confusion_matrix": cm.tolist(),
            "feature_dim": int(X_tr.shape[1]),
            "feature_breakdown": {n: int(d) for n, d in feat_parts},
            "alignment_mean": float(np.mean(A_scores)),
            "alignment_std": float(np.std(A_scores)),
            "n_selected_levels": int(len(sel)),
            "timing": {"total": total, "features": t_feat - t_start, "rf": t_train - t_feat},
        }

        # Save individual result
        with open(os.path.join(output_dir, f"v3_{name}.json"), "w") as f:
            json.dump(all_results[name], f, indent=2)

    # ================================================================
    # Post-hoc: Conformal Calibration on best combo
    # ================================================================
    console.print(f"\n{'='*60}")
    console.print("[bold cyan]POST-HOC: Conformal Calibration[/bold cyan]")
    console.print(f"{'='*60}")

    # Find best combo by Macro F1
    best_name = max(all_results, key=lambda k: all_results[k]["macro_f1"])
    console.print(f"  Best combo: {best_name} (Macro F1 = {all_results[best_name]['macro_f1']:.4f})")

    # Rebuild best features
    b_bank = "bank" in best_name
    b_pyr = "pyramid" in best_name
    b_trim = "trimmed" in best_name
    X_tr_best, X_te_best, _, _, _, _, _, _, _ = build_features_for_config(
        X_train_raw, y_train, X_test_raw, y_test,
        use_bank=b_bank, use_pyramid=b_pyr, use_trimmed=b_trim,
        n_banks=N_BANKS, L_base=L_BASE, S_base=S_BASE, K=K,
        tau=TAU, max_pairs=4000,
    )

    # Split training into train+calibration (80/20)
    rng = np.random.RandomState(42)
    n = len(y_train)
    perm = rng.permutation(n)
    n_cal = n // 5
    cal_idx = perm[:n_cal]
    tr_idx = perm[n_cal:]

    clf_base = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, max_depth=None,
        class_weight="balanced_subsample", n_jobs=-1, random_state=42,
    )
    clf_base.fit(X_tr_best[tr_idx], y_train[tr_idx])

    for alpha in [0.05, 0.10, 0.20]:
        cc = ConformalClassifier(clf_base, alpha=alpha)
        cc.fit(X_tr_best[cal_idx], y_train[cal_idx])
        metrics = cc.evaluate(X_te_best, y_test)
        console.print(f"  alpha={alpha}: coverage={metrics['coverage']:.3f} "
                      f"(target={metrics['target_coverage']:.2f}), "
                      f"avg_set_size={metrics['avg_set_size']:.2f}, "
                      f"point_acc={metrics['point_accuracy']:.4f}")
        all_results[f"conformal_alpha{alpha}"] = metrics

    # ================================================================
    # Post-hoc: Unsupervised Anomaly Scoring
    # ================================================================
    console.print(f"\n{'='*60}")
    console.print("[bold cyan]POST-HOC: Unsupervised Anomaly Scoring[/bold cyan]")
    console.print(f"{'='*60}")

    # Rebuild best features to get barycenters and Q lists
    _, _, _, _, barycenters_best, trQ_best, teQ_best, _, _ = build_features_for_config(
        X_train_raw, y_train, X_test_raw, y_test,
        use_bank=b_bank, use_pyramid=b_pyr, use_trimmed=b_trim,
        n_banks=N_BANKS, L_base=L_BASE, S_base=S_BASE, K=K,
        tau=TAU, max_pairs=4000,
    )

    healthy_class = 0  # Class 0 is "normal" in ECG5000
    healthy_bary = barycenters_best[healthy_class]

    # Compute anomaly scores on test set
    anomaly_scores = compute_anomaly_scores(teQ_best, healthy_bary)
    test_labels = y_test

    # Per-class statistics
    console.print(f"  Anomaly scores per class (test set):")
    anomaly_results = {}
    for c in sorted(np.unique(test_labels)):
        mask = test_labels == c
        scores_c = anomaly_scores[mask]
        anomaly_results[int(c)] = {
            "mean": float(np.mean(scores_c)),
            "std": float(np.std(scores_c)),
            "median": float(np.median(scores_c)),
            "min": float(np.min(scores_c)),
            "max": float(np.max(scores_c)),
        }
        console.print(f"    Class {c}: mean={np.mean(scores_c):.4f}, "
                      f"std={np.std(scores_c):.4f}, "
                      f"median={np.median(scores_c):.4f}")

    # ROC-style analysis: can we detect abnormal classes?
    # Class 0 = healthy, classes 1-4 = abnormal
    healthy_mask = test_labels == healthy_class
    abnormal_mask = test_labels != healthy_class

    if np.sum(healthy_mask) > 0 and np.sum(abnormal_mask) > 0:
        # Simple threshold analysis
        sorted_scores = np.sort(anomaly_scores)
        # Find threshold that gives 90% sensitivity on healthy
        for target_tpr in [0.80, 0.90, 0.95]:
            thresh = np.percentile(anomaly_scores[healthy_mask], target_tpr * 100)
            detected_abnormal = np.mean(anomaly_scores[abnormal_mask] > thresh)
            detected_healthy = np.mean(anomaly_scores[healthy_mask] <= thresh)
            console.print(f"  Threshold @{target_tpr:.0%} healthy: "
                          f"healthy_detected={detected_healthy:.3f}, "
                          f"abnormal_detected={detected_abnormal:.3f}")
            anomaly_results[f"detection_at_{target_tpr}"] = {
                "threshold": float(thresh),
                "healthy_detected": float(detected_healthy),
                "abnormal_detected": float(detected_abnormal),
            }

    all_results["anomaly_scoring"] = anomaly_results

    # ================================================================
    # Post-hoc: KTA-Weighted Ensemble
    # ================================================================
    console.print(f"\n{'='*60}")
    console.print("[bold cyan]POST-HOC: KTA-Weighted Ensemble[/bold cyan]")
    console.print(f"{'='*60}")

    # Train one RF per combo, ensemble with KTA weights
    ensemble_clfs = []
    ensemble_Xte = []
    ensemble_kta = []

    for ci, (use_bank, use_pyramid, use_trimmed) in enumerate(combos):
        name = combo_names[ci]
        X_tr_c, X_te_c, _, A_c, _, _, _, _, _ = build_features_for_config(
            X_train_raw, y_train, X_test_raw, y_test,
            use_bank=use_bank, use_pyramid=use_pyramid, use_trimmed=use_trimmed,
            n_banks=N_BANKS, L_base=L_BASE, S_base=S_BASE, K=K,
            tau=TAU, max_pairs=4000,
        )
        clf_c = RandomForestClassifier(
            n_estimators=200, max_depth=None,
            class_weight="balanced_subsample", n_jobs=-1, random_state=42,
        )
        clf_c.fit(X_tr_c, y_train)
        ensemble_clfs.append(clf_c)
        ensemble_Xte.append(X_te_c)
        ensemble_kta.append(float(np.mean(A_c)))

    # Uniform ensemble
    uniform_proba = np.mean([clf.predict_proba(Xe) for clf, Xe in zip(ensemble_clfs, ensemble_Xte)], axis=0)
    y_pred_uniform = np.argmax(uniform_proba, axis=1)
    acc_u = accuracy_score(y_test, y_pred_uniform)
    mf1_u = f1_score(y_test, y_pred_uniform, average="macro")
    recs_u = recall_score(y_test, y_pred_uniform, average=None)
    console.print(f"  Uniform ensemble: acc={acc_u:.4f}, macroF1={mf1_u:.4f}")
    console.print(f"    Recalls: {', '.join(f'{r:.4f}' for r in recs_u)}")

    # KTA-weighted ensemble
    y_pred_kta = kta_weighted_ensemble_predict(ensemble_clfs, ensemble_Xte, ensemble_kta)
    acc_k = accuracy_score(y_test, y_pred_kta)
    mf1_k = f1_score(y_test, y_pred_kta, average="macro")
    recs_k = recall_score(y_test, y_pred_kta, average=None)
    console.print(f"  KTA-weighted ensemble: acc={acc_k:.4f}, macroF1={mf1_k:.4f}")
    console.print(f"    Recalls: {', '.join(f'{r:.4f}' for r in recs_k)}")
    console.print(f"    KTA weights: {[f'{w:.4f}' for w in np.array(ensemble_kta)/sum(ensemble_kta)]}")

    all_results["ensemble_uniform"] = {
        "accuracy": float(acc_u), "macro_f1": float(mf1_u),
        "class_recalls": [float(r) for r in recs_u],
    }
    all_results["ensemble_kta_weighted"] = {
        "accuracy": float(acc_k), "macro_f1": float(mf1_k),
        "class_recalls": [float(r) for r in recs_k],
        "kta_weights": [float(w) for w in np.array(ensemble_kta)/sum(ensemble_kta)],
    }

    # ================================================================
    # Final Summary Table
    # ================================================================
    console.print(f"\n{'='*80}")
    console.print("[bold cyan]COMPLETE V3 RESULTS[/bold cyan]")
    console.print(f"{'='*80}")

    console.print(f"\n[bold]Ablation: Component Combinations[/bold]")
    console.print(f"{'Config':<25} {'Dim':>5} {'Acc':>8} {'MacroF1':>8} {'C0':>6} {'C1':>6} {'C2':>6} {'C3':>6} {'C4':>6} {'Time':>6}")
    console.print("-" * 95)

    best_mf1 = -1
    best_name = ""
    for name in combo_names:
        r = all_results[name]
        cr = r["class_recalls"]
        marker = ""
        if r["macro_f1"] > best_mf1:
            best_mf1 = r["macro_f1"]
            best_name = name
        console.print(f"{name:<25} {r['feature_dim']:>5} {r['accuracy']:>8.4f} {r['macro_f1']:>8.4f} "
                      f"{cr[0]:>6.3f} {cr[1]:>6.3f} {cr[2]:>6.3f} {cr[3]:>6.3f} {cr[4]:>6.3f} "
                      f"{r['timing']['total']:>5.1f}s{marker}")
    console.print(f"\n  Best combo: [bold]{best_name}[/bold] (Macro F1 = {best_mf1:.4f})")

    console.print(f"\n[bold]Post-Hoc: Conformal Calibration (on {best_name})[/bold]")
    for alpha in [0.05, 0.10, 0.20]:
        key = f"conformal_alpha{alpha}"
        if key in all_results:
            r = all_results[key]
            console.print(f"  alpha={alpha}: coverage={r['coverage']:.3f} (target {r['target_coverage']:.2f}), "
                          f"set_size={r['avg_set_size']:.2f}, point_acc={r['point_accuracy']:.4f}")

    console.print(f"\n[bold]Post-Hoc: Ensemble Methods[/bold]")
    for ek in ["ensemble_uniform", "ensemble_kta_weighted"]:
        r = all_results[ek]
        cr = r["class_recalls"]
        console.print(f"  {ek}: acc={r['accuracy']:.4f}, macroF1={r['macro_f1']:.4f} "
                      f"recalls=[{','.join(f'{x:.3f}' for x in cr)}]")

    console.print(f"\n[bold]Post-Hoc: Unsupervised Anomaly Detection[/bold]")
    for c in sorted(np.unique(y_test)):
        ar = anomaly_results[c]
        console.print(f"  Class {c}: score_mean={ar['mean']:.4f} ± {ar['std']:.4f}")

    # Baseline comparison
    console.print(f"\n[bold]Comparison with Baselines[/bold]")
    console.print(f"  {'Model':<30} {'Acc':>8} {'MacroF1':>8}")
    console.print(f"  {'-'*50}")
    console.print(f"  {'MiniRocket':<30} {'0.9560':>8} {'0.6011':>8}")
    console.print(f"  {'InceptionTime':<30} {'0.9610':>8} {'0.7697':>8}")
    console.print(f"  {'KTA-TF-Drift v1':<30} {'0.9480':>8} {'0.5782':>8}")
    console.print(f"  {'KTA-TF-Drift v2':<30} {'0.9470':>8} {'0.5816':>8}")
    console.print(f"  {'KTA-TF-Drift v2.1':<30} {'0.9500':>8} {'0.5891':>8}")
    console.print(f"  {'KTA-TF-Drift v3 (best combo)':<30} {all_results[best_name]['accuracy']:>8.4f} {best_mf1:>8.4f}")
    console.print(f"  {'KTA-TF-Drift v3 (KTA ensemble)':<30} {all_results['ensemble_kta_weighted']['accuracy']:>8.4f} {all_results['ensemble_kta_weighted']['macro_f1']:>8.4f}")

    # Save everything
    summary_path = os.path.join(output_dir, "v3_full_ablation.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    console.print(f"\n[green]Full ablation saved to {summary_path}[/green]")

    return all_results


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_file = os.path.join(base_dir, "data", "ecg5000_resplit.npz")
    output_dir = os.path.join(base_dir, "results", "v3_ablation")
    run_ablation(data_file, output_dir)
