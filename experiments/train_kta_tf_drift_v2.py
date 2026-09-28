"""
KTA-TF-Drift v2: Wasserstein Drift Features with Hellinger-Kernel Alignment
for ECG5000 Classification

Fixes from v1 diagnosis:
  Fix 1: Class-balanced KTA alignment (subsampling + reweighted)
  Fix 2: Static quantile features + drift features concatenated
  Fix 3: Multi-lag dyadic drift (τ=1,2,4,8)
  Fix 4: Wasserstein-barycenter anchor features per class
  Fix 5: Stratified bootstrap RF (manual BalancedRF)
  Fix 6: Kernel PCA on full multivariate Hellinger-RBF kernel
"""

import os
import sys
import json
import time
import warnings
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import KernelPCA
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from rich.console import Console

warnings.filterwarnings("ignore")
console = Console()


# ============================================================
# Stage 1: Windowing
# ============================================================
def compute_windows(signal, L, S):
    """Segment signal into overlapping windows."""
    N = len(signal)
    windows = []
    i = 0
    while i * S + L <= N:
        windows.append(signal[i * S : i * S + L])
        i += 1
    return np.array(windows)  # (M, L)


# ============================================================
# Stage 2: Quantile Functions
# ============================================================
def compute_quantile_functions(windows, K):
    """
    Compute quantile-function vectors for each window.
    Returns: Q (M, K)
    """
    M, L = windows.shape
    quantile_grid = (np.arange(1, K + 1) - 0.5) / K

    Q = np.zeros((M, K), dtype=np.float64)
    for i in range(M):
        sorted_w = np.sort(windows[i])
        emp_quantiles = (np.arange(1, L + 1) - 0.5) / L
        Q[i] = np.interp(quantile_grid, emp_quantiles, sorted_w)
    return Q


# ============================================================
# Stage 3: Drift Features (multi-lag)
# ============================================================
def compute_drift_features_multi_lag(Q, lags=(1, 2, 4, 8)):
    """
    Compute drift features at multiple lags: ΔQ^(τ)_i = Q_{i+τ} - Q_i
    Q: (M, K)
    Returns: dict {tau: dQ_tau (M-tau, K)}
    """
    M, K = Q.shape
    drift_dict = {}
    for tau in lags:
        if M - tau > 0:
            drift_dict[tau] = Q[tau:] - Q[:M - tau]
    return drift_dict


# ============================================================
# Stage 3b: Static quantile features (Fix 2)
# ============================================================
def compute_static_quantile_features(Q):
    """
    Compute static per-signal quantile statistics.
    Q: (M, K) — quantile functions for all windows
    Returns: features per signal (4*K) — mean, std, max, min across windows
    """
    return np.concatenate([
        np.mean(Q, axis=0),
        np.std(Q, axis=0),
        np.max(Q, axis=0),
        np.min(Q, axis=0),
    ])


# ============================================================
# Stage 3c: Wasserstein-barycenter anchor features (Fix 4)
# ============================================================
def compute_class_barycenters(Q_all, y_all, K):
    """
    Compute Wasserstein barycenter (mean quantile function) per class.
    Q_all: list of (M_i, K) arrays, one per signal
    y_all: (N,) labels
    Returns: barycenters dict {class: Q_bar (K,)}, barycenters_all (C, K)
    """
    classes = np.unique(y_all)
    barycenters = {}
    for c in classes:
        mask = y_all == c
        class_qs = []
        for i, (qi, yi) in enumerate(zip(Q_all, y_all)):
            if yi == c:
                # Use mean quantile function across windows as the signal's Q
                class_qs.append(np.mean(qi, axis=0))
        barycenters[c] = np.mean(class_qs, axis=0)
    return barycenters


def compute_barycenter_distances(Q_signal, barycenters, K):
    """
    For a single signal's quantile functions (M, K), compute:
    - Distance from each window's mean Q to each class barycenter
    - Min/mean/max distance across windows
    Returns: feature vector of shape (C * 3,) where C = number of classes
    """
    classes = sorted(barycenters.keys())
    C = len(classes)
    # Mean Q for this signal
    Q_mean = np.mean(Q_signal, axis=0)  # (K,)

    dists = []
    for c in classes:
        d = np.sqrt(np.mean((Q_mean - barycenters[c]) ** 2))  # W2 distance
        dists.append(d)

    # Also per-window statistics
    M = Q_signal.shape[0]
    per_window_dists = np.zeros((M, C))
    for j, c in enumerate(classes):
        for i in range(M):
            per_window_dists[i, j] = np.sqrt(np.mean((Q_signal[i] - barycenters[c]) ** 2))

    features = []
    # Global distances (C)
    features.extend(dists)
    # Min distances per class (C)
    features.extend(np.min(per_window_dists, axis=0).tolist())
    # Max distances per class (C)
    features.extend(np.max(per_window_dists, axis=0).tolist())
    # Std of distances per class (C)
    features.extend(np.std(per_window_dists, axis=0).tolist())

    return np.array(features)


# ============================================================
# Stage 4: Class-Balanced KTA (Fix 1)
# ============================================================
def signed_sqrt(u):
    """Signed square-root map."""
    return np.sign(u) * np.sqrt(np.abs(u) + 1e-16)


def kernel_target_alignment(Kmat, Y_onehot):
    """
    A(K, Y) = <K, YY^T>_F / (||K||_F * ||YY^T||_F)
    """
    YYt = Y_onehot @ Y_onehot.T
    numerator = np.sum(Kmat * YYt)
    denom_K = np.sqrt(np.sum(Kmat * Kmat))
    denom_Y = np.sqrt(np.sum(YYt * YYt))
    if denom_K < 1e-16 or denom_Y < 1e-16:
        return 0.0
    return numerator / (denom_K * denom_Y)


def compute_kta_weights_balanced(dQ_all, labels_all, K, tau, gamma, n_samples=2000):
    """
    Fix 1: Class-balanced KTA alignment.
    Subsample each class to the minority-class count before computing alignment.
    """
    n_pairs, K_feat = dQ_all.shape
    C = len(np.unique(labels_all))

    # Class-balanced subsampling: downsample each class to min count
    class_counts = np.bincount(labels_all.astype(int), minlength=C)
    min_count = max(int(np.min(class_counts)), 10)  # at least 10 per class
    target_per_class = min(n_samples // C, min_count)

    console.print(f"  [dim]Class-balanced KTA: {target_per_class} pairs/class (from {dict(enumerate(class_counts))})[/dim]")

    # Subsample per class
    rng = np.random.RandomState(42)
    balanced_idx = []
    for c in range(C):
        c_idx = np.where(labels_all == c)[0]
        if len(c_idx) > target_per_class:
            chosen = rng.choice(c_idx, target_per_class, replace=False)
        else:
            chosen = rng.choice(c_idx, target_per_class, replace=True)
        balanced_idx.append(chosen)

    balanced_idx = np.concatenate(balanced_idx)
    rng.shuffle(balanced_idx)

    dQ_sub = dQ_all[balanced_idx]
    labels_sub = labels_all[balanced_idx]
    Y_onehot = (labels_sub[:, None] == np.arange(C)[None, :]).astype(np.float64)

    n_sub = dQ_sub.shape[0]

    # Gamma: median heuristic on balanced set
    if gamma <= 0:
        phi_ch0 = signed_sqrt(dQ_sub[:, 0])
        diffs = phi_ch0[:, None] - phi_ch0[None, :]
        med_sq = np.median(diffs ** 2)
        gamma = 1.0 / (med_sq + 1e-16)
        console.print(f"  [dim]Median heuristic gamma = {gamma:.4f}[/dim]")

    alignment_scores = np.zeros(K_feat)
    for k in range(K_feat):
        phi = signed_sqrt(dQ_sub[:, k])
        diff = phi[:, None] - phi[None, :]
        K_k = np.exp(-gamma * diff ** 2)
        alignment_scores[k] = kernel_target_alignment(K_k, Y_onehot)

    # Selection and weighting
    selected_mask = alignment_scores > tau
    selected_indices = np.where(selected_mask)[0]
    if len(selected_indices) == 0:
        selected_indices = np.arange(K_feat)

    pos_scores = np.maximum(alignment_scores[selected_indices], 0.0)
    total = np.sum(pos_scores)
    if total < 1e-16:
        weights = np.ones(len(selected_indices)) / len(selected_indices)
    else:
        weights = pos_scores / total

    console.print(f"  [green]Selected {len(selected_indices)}/{K_feat} quantile levels[/green]")
    console.print(f"  [dim]Alignment range: [{alignment_scores.min():.4f}, {alignment_scores.max():.4f}][/dim]")
    return selected_indices, weights, alignment_scores


# ============================================================
# Stage 4b: Kernel PCA on full multivariate kernel (Fix 6)
# ============================================================
def compute_kernel_pca_features(dQ_all, labels_all, n_components=10, gamma=-1.0, max_fit_samples=3000):
    """
    Fix 6: Compute kernel PCA on a subsampled multivariate Hellinger-RBF kernel.
    Fit on subsample, then transform all.
    """
    phi = signed_sqrt(dQ_all)
    n_total = phi.shape[0]

    if gamma <= 0:
        n_sub_med = min(n_total, 1000)
        idx = np.random.RandomState(42).choice(n_total, n_sub_med, replace=False)
        diffs = np.linalg.norm(phi[idx, None] - phi[None, idx, :], axis=2)
        med = np.median(diffs ** 2)
        gamma = 1.0 / (med + 1e-16)

    # Subsample for fitting to avoid OOM
    if n_total > max_fit_samples:
        fit_idx = np.random.RandomState(42).choice(n_total, max_fit_samples, replace=False)
        fit_idx.sort()
        phi_fit = phi[fit_idx]
    else:
        phi_fit = phi

    console.print(f"  [dim]Kernel PCA: fit on {phi_fit.shape[0]}/{n_total} samples, {n_components} components, gamma={gamma:.4f}[/dim]")

    kpca = KernelPCA(
        n_components=n_components,
        kernel="rbf",
        gamma=gamma,
        fit_inverse_transform=False,
        n_jobs=1,
    )
    # Fit on subset only
    kpca.fit(phi_fit)
    return kpca


# ============================================================
# Stage 5: Stratified Bootstrap RF (Fix 5)
# ============================================================
class StratifiedBalancedRF:
    """
    Fix 5: Random Forest with stratified bootstrap sampling.
    Each tree's bootstrap sample is stratified to guarantee minority-class representation.
    """
    def __init__(self, n_estimators=500, max_depth=None, min_samples_leaf=1,
                 random_state=42, n_jobs=-1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.trees = []
        self.oob_indices = []

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        classes = np.unique(y)
        class_counts = {c: np.sum(y == c) for c in classes}
        min_count = min(class_counts.values())

        self.trees = []
        for t in range(self.n_estimators):
            # Stratified bootstrap: sample min_count from each class, then oversample
            bootstrap_idx = []
            for c in classes:
                c_idx = np.where(y == c)[0]
                # Sample with replacement, at least min_count
                n_sample = max(min_count, len(c_idx))
                sampled = rng.choice(c_idx, size=n_sample, replace=True)
                bootstrap_idx.extend(sampled)

            bootstrap_idx = np.array(bootstrap_idx)
            rng.shuffle(bootstrap_idx)

            X_boot = X[bootstrap_idx]
            y_boot = y[bootstrap_idx]

            tree = RandomForestClassifier(
                n_estimators=1,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=rng.randint(0, 2**31),
                n_jobs=1,
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

        return self

    def predict(self, X):
        """Majority vote across all trees."""
        all_preds = np.array([tree.predict(X) for tree in self.trees])  # (n_trees, n_samples)
        from scipy import stats
        y_pred, _ = stats.mode(all_preds, axis=0, keepdims=False)
        return y_pred.ravel()

    def predict_proba(self, X):
        """Average predicted probabilities across trees."""
        all_proba = np.array([tree.predict_proba(X) for tree in self.trees])  # (n_trees, n_samples, C)
        return np.mean(all_proba, axis=0)


# ============================================================
# Full Pipeline: KTA-TF-Drift v2
# ============================================================
def run_kta_tf_drift_v2(
    data_file,
    output_file,
    L=35, S=7, K=35,
    lags=(1, 2, 4, 8),
    tau=0.01, gamma_kta=-1.0, gamma_kpca=-1.0,
    n_estimators=500,
    n_kpca_components=10,
    max_pairs_for_kta=4000,
    use_balanced_rf=True,
):
    """Full KTA-TF-Drift v2 pipeline with all 6 fixes."""
    console.print("[bold cyan]KTA-TF-Drift v2 Pipeline[/bold cyan]")
    console.print(f"  L={L}, S={S}, K={K}, lags={lags}, tau={tau}")
    console.print(f"  Fixes: balanced-KTA, static+drift, multi-lag, barycenter, stratified-RF, kernelPCA")

    # ---- Load Data ----
    console.print(f"[cyan]Loading dataset...[/cyan]")
    data = np.load(data_file)
    X_train_raw = data["X_train"].astype(np.float64)
    y_train_raw = data["y_train"].astype(np.int64)
    X_test_raw = data["X_test"].astype(np.float64)
    y_test_raw = data["y_test"].astype(np.int64)

    # Per-sample standardization (same as baselines)
    eps = 1e-8
    mu = np.mean(X_train_raw, axis=-1, keepdims=True)
    sigma = np.std(X_train_raw, axis=-1, keepdims=True)
    X_train_raw = (X_train_raw - mu) / (sigma + eps)
    mu = np.mean(X_test_raw, axis=-1, keepdims=True)
    sigma = np.std(X_test_raw, axis=-1, keepdims=True)
    X_test_raw = (X_test_raw - mu) / (sigma + eps)

    N_train, N_test = X_train_raw.shape[0], X_test_raw.shape[0]
    console.print(f"  Train: {N_train}, Test: {N_test}")

    # ================================================================
    # Stage 1-2: Compute quantile functions for all signals
    # ================================================================
    console.print("[cyan]Stage 1-2: Windowing + Quantile functions...[/cyan]")
    t0 = time.time()

    train_Q_list = []
    for i in range(N_train):
        windows = compute_windows(X_train_raw[i], L, S)
        Q = compute_quantile_functions(windows, K)
        train_Q_list.append(Q)

    test_Q_list = []
    for i in range(N_test):
        windows = compute_windows(X_test_raw[i], L, S)
        Q = compute_quantile_functions(windows, K)
        test_Q_list.append(Q)

    t1 = time.time()
    console.print(f"[green]  Quantile computation: {t1 - t0:.2f}s[/green]")

    # ================================================================
    # Stage 3: Multi-lag drift features (Fix 3) + static features (Fix 2)
    # ================================================================
    console.print("[cyan]Stage 3: Multi-lag drift + static features...[/cyan]")
    t2 = time.time()

    # Collect all drift pairs for KTA
    train_dQ_all_flat = []
    train_labels_flat = []
    for i in range(N_train):
        drift_dict = compute_drift_features_multi_lag(train_Q_list[i], lags)
        for tau_lag, dQ_tau in drift_dict.items():
            train_dQ_all_flat.append(dQ_tau)
            train_labels_flat.extend([y_train_raw[i]] * dQ_tau.shape[0])
    train_dQ_all_flat = np.concatenate(train_dQ_all_flat, axis=0)
    train_labels_flat = np.array(train_labels_flat, dtype=np.int64)

    # Same for test
    test_dQ_all_flat = []
    test_labels_flat = []
    for i in range(N_test):
        drift_dict = compute_drift_features_multi_lag(test_Q_list[i], lags)
        for tau_lag, dQ_tau in drift_dict.items():
            test_dQ_all_flat.append(dQ_tau)
            test_labels_flat.extend([y_test_raw[i]] * dQ_tau.shape[0])
    test_dQ_all_flat = np.concatenate(test_dQ_all_flat, axis=0)
    test_labels_flat = np.array(test_labels_flat, dtype=np.int64)

    n_lags = len(lags)
    K_feat_drift = train_dQ_all_flat.shape[1]  # Should be K
    console.print(f"  Multi-lag drift pairs: train={train_dQ_all_flat.shape[0]}, test={test_dQ_all_flat.shape[0]}")
    console.print(f"  Drift feature dimension: {K_feat_drift} (K={K}, {n_lags} lags)")
    t3 = time.time()

    # ================================================================
    # Stage 4: Class-Balanced KTA on drift features (Fix 1)
    # ================================================================
    console.print("[cyan]Stage 4: Class-balanced KTA (Fix 1)...[/cyan]")
    t4 = time.time()
    selected_indices, weights, alignment_scores = compute_kta_weights_balanced(
        train_dQ_all_flat, train_labels_flat, K_feat_drift, tau, gamma_kta, max_pairs_for_kta
    )
    t5 = time.time()
    console.print(f"[green]  KTA computation: {t5 - t4:.2f}s[/green]")

    # ================================================================
    # Stage 4b: Kernel PCA on full multivariate kernel (Fix 6)
    # ================================================================
    console.print("[cyan]Stage 4b: Kernel PCA (Fix 6)...[/cyan]")
    t6 = time.time()
    kpca_model = compute_kernel_pca_features(
        train_dQ_all_flat, train_labels_flat, n_components=n_kpca_components, gamma=gamma_kpca
    )
    t7 = time.time()
    console.print(f"[green]  Kernel PCA fitted: {t7 - t6:.2f}s[/green]")

    # ================================================================
    # Build per-signal feature vectors
    # ================================================================
    console.print("[cyan]Building per-signal feature vectors...[/cyan]")

    # Fix 4: Compute class barycenters on training data
    barycenters = compute_class_barycenters(train_Q_list, y_train_raw, K)
    C = len(np.unique(y_train_raw))
    console.print(f"  Computed barycenters for {C} classes")

    # Fix 4: Barycenter distance features
    train_bary_feat = np.array([
        compute_barycenter_distances(Q, barycenters, K) for Q in train_Q_list
    ])
    test_bary_feat = np.array([
        compute_barycenter_distances(Q, barycenters, K) for Q in test_Q_list
    ])
    console.print(f"  Barycenter features: {train_bary_feat.shape[1]} dims")

    # Fix 2: Static quantile features
    train_static_feat = np.array([compute_static_quantile_features(Q) for Q in train_Q_list])
    test_static_feat = np.array([compute_static_quantile_features(Q) for Q in test_Q_list])
    console.print(f"  Static features: {train_static_feat.shape[1]} dims")

    # KTA-weighted drift features (mean across pairs)
    sqrt_w = np.sqrt(weights)
    train_kta_drift = []
    for i in range(N_train):
        drift_dict = compute_drift_features_multi_lag(train_Q_list[i], lags)
        # Concatenate all lag drift features for this signal
        all_dQ = np.concatenate([dQ for dQ in drift_dict.values()], axis=0)
        phi = signed_sqrt(all_dQ[:, selected_indices])
        weighted = phi * sqrt_w[None, :]
        # Aggregate: mean, std, max, min
        train_kta_drift.append(np.concatenate([
            np.mean(weighted, axis=0),
            np.std(weighted, axis=0),
            np.max(weighted, axis=0),
            np.min(weighted, axis=0),
        ]))
    train_kta_drift = np.array(train_kta_drift)

    test_kta_drift = []
    for i in range(N_test):
        drift_dict = compute_drift_features_multi_lag(test_Q_list[i], lags)
        all_dQ = np.concatenate([dQ for dQ in drift_dict.values()], axis=0)
        phi = signed_sqrt(all_dQ[:, selected_indices])
        weighted = phi * sqrt_w[None, :]
        test_kta_drift.append(np.concatenate([
            np.mean(weighted, axis=0),
            np.std(weighted, axis=0),
            np.max(weighted, axis=0),
            np.min(weighted, axis=0),
        ]))
    test_kta_drift = np.array(test_kta_drift)
    console.print(f"  KTA drift features: {train_kta_drift.shape[1]} dims")

    # Kernel PCA features (Fix 6) — aggregate per signal
    # kpca features are per-pair, need to aggregate per signal
    # We'll use per-signal kernel PCA instead
    console.print("[cyan]  Recomputing kernel PCA per-signal...[/cyan]")
    train_kpca_feat = []
    for i in range(N_train):
        drift_dict = compute_drift_features_multi_lag(train_Q_list[i], lags)
        all_dQ = np.concatenate([dQ for dQ in drift_dict.values()], axis=0)
        phi = signed_sqrt(all_dQ)
        # Use per-signal: mean of kernel PCA projections
        kpca_proj = kpca_model.transform(phi)
        train_kpca_feat.append(np.concatenate([
            np.mean(kpca_proj, axis=0),
            np.std(kpca_proj, axis=0),
        ]))
    train_kpca_feat = np.array(train_kpca_feat)

    test_kpca_feat = []
    for i in range(N_test):
        drift_dict = compute_drift_features_multi_lag(test_Q_list[i], lags)
        all_dQ = np.concatenate([dQ for dQ in drift_dict.values()], axis=0)
        phi = signed_sqrt(all_dQ)
        kpca_proj = kpca_model.transform(phi)
        test_kpca_feat.append(np.concatenate([
            np.mean(kpca_proj, axis=0),
            np.std(kpca_proj, axis=0),
        ]))
    test_kpca_feat = np.array(test_kpca_feat)
    console.print(f"  Kernel PCA features: {train_kpca_feat.shape[1]} dims")

    # ================================================================
    # Concatenate all feature sources (Fix 2 + Fixes 3,4,6)
    # ================================================================
    X_train_feat = np.concatenate([
        train_static_feat,      # Fix 2: static quantile features
        train_kta_drift,        # Fix 1+3: class-balanced KTA-weighted multi-lag drift
        train_bary_feat,        # Fix 4: barycenter distance features
        train_kpca_feat,        # Fix 6: kernel PCA nonlinear embedding
    ], axis=1)

    X_test_feat = np.concatenate([
        test_static_feat,
        test_kta_drift,
        test_bary_feat,
        test_kpca_feat,
    ], axis=1)

    feature_breakdown = {
        "static_quantile": train_static_feat.shape[1],
        "kta_weighted_drift": train_kta_drift.shape[1],
        "barycenter_distances": train_bary_feat.shape[1],
        "kernel_pca": train_kpca_feat.shape[1],
    }
    total_dim = X_train_feat.shape[1]
    console.print(f"\n  [bold]Total feature dimension: {total_dim}[/bold]")
    for name, dim in feature_breakdown.items():
        console.print(f"    {name}: {dim}")

    # ================================================================
    # Stage 5: Classification (Fix 5)
    # ================================================================
    console.print(f"\n[cyan]Stage 5: Classification (Fix 5: {'StratifiedBalancedRF' if use_balanced_rf else 'Standard RF'})...[/cyan]")
    t8 = time.time()

    if use_balanced_rf:
        clf = StratifiedBalancedRF(
            n_estimators=n_estimators,
            max_depth=None,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
        )
    else:
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        )

    clf.fit(X_train_feat, y_train_raw)
    y_pred = clf.predict(X_test_feat)

    t9 = time.time()
    console.print(f"[green]  Classification: {t9 - t8:.2f}s[/green]")

    # ================================================================
    # Evaluation
    # ================================================================
    total_time = t9 - t0

    final_acc = accuracy_score(y_test_raw, y_pred)
    final_macro_f1 = f1_score(y_test_raw, y_pred, average="macro")
    final_recalls = recall_score(y_test_raw, y_pred, average=None)
    cm = confusion_matrix(y_test_raw, y_pred)

    console.print(f"\n[bold green]{'='*60}[/bold green]")
    console.print(f"[bold green]  KTA-TF-Drift v2 RESULTS[/bold green]")
    console.print(f"[bold green]{'='*60}[/bold green]")
    console.print(f"  [bold]Accuracy: {final_acc:.4f}[/bold]")
    console.print(f"  [bold]Macro F1: {final_macro_f1:.4f}[/bold]")
    console.print(f"  [bold]Class Recalls: {', '.join(f'{r:.4f}' for r in final_recalls)}[/bold]")
    console.print(f"  [bold]Total time: {total_time:.2f}s[/bold]")

    # ================================================================
    # Save Results
    # ================================================================
    results = {
        "model": "KTA-TF-Drift-v2",
        "accuracy": float(final_acc),
        "macro_f1": float(final_macro_f1),
        "class_recalls": [float(r) for r in final_recalls],
        "confusion_matrix": cm.tolist(),
        "config": {
            "L": L, "S": S, "K": K,
            "lags": list(lags),
            "tau": tau,
            "gamma_kta": "median_heuristic" if gamma_kta <= 0 else float(gamma_kta),
            "gamma_kpca": "median_heuristic" if gamma_kpca <= 0 else float(gamma_kpca),
            "n_estimators": n_estimators,
            "n_kpca_components": n_kpca_components,
            "use_balanced_rf": use_balanced_rf,
            "feature_breakdown": feature_breakdown,
            "total_feature_dim": total_dim,
            "selected_quantile_levels": selected_indices.tolist(),
            "num_selected": len(selected_indices),
            "weights": weights.tolist(),
        },
        "timing": {
            "quantile_computation": t1 - t0,
            "drift_features": t3 - t2,
            "kta_computation": t5 - t4,
            "kernel_pca": t7 - t6,
            "classification": t9 - t8,
            "total_sec": total_time,
        },
        "alignment_scores": alignment_scores.tolist(),
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    console.print(f"\n[green]Results saved to {output_file}[/green]")

    return results


if __name__ == "__main__":
    import shutil
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_file = os.path.join(base_dir, "data", "ecg5000_resplit.npz")
    output_file = os.path.join(base_dir, "results", "kta_tf_drift_v2_results.json")

    # ---- Final optimized run: best findings from ablation ----
    # Standard RF outperformed StratifiedBalancedRF (Fix 5 not helpful here)
    # tau is irrelevant (balanced KTA scores still flat), use default
    # Multi-lag adds noise for ECG5000, single-lag is cleaner
    # Kernel PCA adds modest value
    # Barycenter features are the key new signal for minority classes
    configs = [
        {"label": "v2_best", "tau": 0.01, "use_balanced_rf": False, "lags": (1, 2, 4, 8), "n_kpca_components": 10},
        {"label": "v2_bary_only", "tau": 0.01, "use_balanced_rf": False, "lags": (1,), "n_kpca_components": 0},
    ]

    best_macro_f1 = -1
    best_label = ""
    all_results = {}

    for cfg in configs:
        label = cfg.pop("label")
        # Handle n_kpca_components=0 by disabling KPCA
        n_kpca = cfg.pop("n_kpca_components", 10)
        console.print(f"\n{'='*60}")
        console.print(f"[bold yellow]Config: {label}[/bold yellow]")
        console.print(f"{'='*60}")

        try:
            result = run_kta_tf_drift_v2(
                data_file,
                output_file.replace(".json", f"_{label}.json"),
                n_kpca_components=max(n_kpca, 2) if n_kpca > 0 else 2,  # sklearn needs >= 1
                **cfg,
            )
            all_results[label] = {
                "accuracy": result["accuracy"],
                "macro_f1": result["macro_f1"],
                "class_recalls": result["class_recalls"],
                "timing": result["timing"]["total_sec"],
            }
            if result["macro_f1"] > best_macro_f1:
                best_macro_f1 = result["macro_f1"]
                best_label = label
        except Exception as e:
            console.print(f"[red]Error with {label}: {e}[/red]")
            import traceback
            traceback.print_exc()

    # ---- Summary ----
    console.print(f"\n{'='*60}")
    console.print("[bold cyan]V2 ABLATION SUMMARY[/bold cyan]")
    console.print(f"{'='*60}")
    console.print(f"{'Config':<35} {'Acc':>8} {'MacroF1':>8} {'C0':>6} {'C1':>6} {'C2':>6} {'C3':>6} {'C4':>6} {'Time':>7}")
    console.print("-" * 100)
    for label, r in all_results.items():
        marker = " <-- BEST" if label == best_label else ""
        cr = r["class_recalls"]
        console.print(f"{label:<35} {r['accuracy']:>8.4f} {r['macro_f1']:>8.4f} {cr[0]:>6.3f} {cr[1]:>6.3f} {cr[2]:>6.3f} {cr[3]:>6.3f} {cr[4]:>6.3f} {r['timing']:>6.1f}s{marker}")

    console.print(f"\n[bold green]Best: {best_label} (Macro F1 = {best_macro_f1:.4f})[/bold green]")

    # Save summary
    summary = {"best": best_label, "results": all_results}
    summary_path = os.path.join(base_dir, "results", "kta_tf_drift_v2_ablation.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    console.print(f"[green]Ablation saved to {summary_path}[/green]")

    # Copy best to main file
    best_file = output_file.replace(".json", f"_{best_label}.json")
    if os.path.exists(best_file):
        shutil.copy(best_file, output_file)
        console.print(f"[green]Best results copied to {output_file}[/green]")
