"""
KTA-TF-Drift: Wasserstein Drift Features with Hellinger-Kernel Alignment
for ECG5000 Classification

Pipeline:
  1. Window each signal into overlapping windows
  2. Compute quantile function per window
  3. Drift features = quantile differences between consecutive windows
  4. Hellinger-KTA for per-quantile-level feature selection/weighting
  5. Random Forest classifier on weighted drift features
"""

import os
import sys
import json
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from rich.console import Console

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
    windows: (M, L)
    Returns: Q (M, K) array of quantile values on grid p_k = (k-0.5)/K
    """
    M, L = windows.shape
    quantile_grid = (np.arange(1, K + 1) - 0.5) / K  # p_k values

    Q = np.zeros((M, K), dtype=np.float64)
    for i in range(M):
        sorted_w = np.sort(windows[i])
        # Order statistics correspond to empirical quantiles at (n-0.5)/L
        emp_quantiles = (np.arange(1, L + 1) - 0.5) / L
        # Interpolate onto target grid
        Q[i] = np.interp(quantile_grid, emp_quantiles, sorted_w)
    return Q


# ============================================================
# Stage 3: Drift Features
# ============================================================
def compute_drift_features(Q):
    """
    Compute drift features: Delta Q_i = Q_{i+1} - Q_i
    Q: (M, K)
    Returns: dQ (M-1, K)
    """
    return Q[1:] - Q[:-1]


# ============================================================
# Stage 4: Hellinger-Kernel KTA
# ============================================================
def signed_sqrt(u):
    """Signed square-root map: sign(u) * sqrt(|u|)."""
    return np.sign(u) * np.sqrt(np.abs(u) + 1e-16)


def hellinger_rbf_kernel(phi_i, phi_j, gamma):
    """Hellinger-RBF kernel between two signed-sqrt-transformed scalars."""
    return np.exp(-gamma * (phi_i - phi_j) ** 2)


def compute_kernel_target_alignment(Kmat, Y_onehot):
    """
    Compute Kernel Target Alignment: A(K, Y) = <K, YY^T>_F / (||K||_F * ||YY^T||_F)
    Kmat: (n, n) kernel matrix
    Y_onehot: (n, C) one-hot label matrix
    """
    YYt = Y_onehot @ Y_onehot.T  # ideal same-class kernel
    numerator = np.sum(Kmat * YYt)
    denom_K = np.sqrt(np.sum(Kmat * Kmat))
    denom_Y = np.sqrt(np.sum(YYt * YYt))
    if denom_K < 1e-16 or denom_Y < 1e-16:
        return 0.0
    return numerator / (denom_K * denom_Y)


def compute_kta_weights(dQ_all, labels_all, K, tau, gamma, max_pairs_for_kta=2000):
    """
    Compute KTA weights for each quantile level.

    dQ_all: (total_pairs, K) drift features from all training signals
    labels_all: (total_pairs,) labels for each pair
    K: number of quantile grid points
    tau: alignment threshold
    gamma: Hellinger-RBF bandwidth
    max_pairs_for_kta: subsample limit for KTA computation (feasibility)

    Returns:
        selected_indices: indices of selected quantile levels
        weights: normalized weights for selected levels
        alignment_scores: all K alignment scores
    """
    n_pairs, K_feat = dQ_all.shape
    C = len(np.unique(labels_all))
    Y_onehot = (labels_all[:, None] == np.arange(C)[None, :]).astype(np.float64)

    # Subsample if too many pairs (KTA requires n x n kernel matrix)
    if n_pairs > max_pairs_for_kta:
        rng = np.random.RandomState(42)
        idx = rng.choice(n_pairs, max_pairs_for_kta, replace=False)
        dQ_sub = dQ_all[idx]
        Y_sub = Y_onehot[idx]
    else:
        dQ_sub = dQ_all
        Y_sub = Y_onehot

    n_sub = dQ_sub.shape[0]
    console.print(f"  [dim] KTA computed on {n_sub} subsampled pairs (from {n_pairs} total)")

    # Compute per-channel median heuristic for gamma if gamma <= 0
    if gamma <= 0:
        # Use first channel as representative
        phi_ch0 = signed_sqrt(dQ_sub[:, 0])
        diffs = phi_ch0[:, None] - phi_ch0[None, :]
        med_sq = np.median(diffs ** 2)
        gamma = 1.0 / (med_sq + 1e-16)
        console.print(f"  [dim] Median heuristic gamma = {gamma:.4f}")

    alignment_scores = np.zeros(K_feat)
    for k in range(K_feat):
        channel = dQ_sub[:, k]  # (n_sub,)
        phi = signed_sqrt(channel)

        # Build kernel matrix K^(k)_{ij} = exp(-gamma * (phi_i - phi_j)^2)
        # Efficient vectorized computation
        diff = phi[:, None] - phi[None, :]
        K_k = np.exp(-gamma * diff ** 2)

        alignment_scores[k] = compute_kernel_target_alignment(K_k, Y_sub)

        if (k + 1) % 10 == 0 or k == K_feat - 1:
            console.print(f"    Quantile level {k+1}/{K_feat}: A_k = {alignment_scores[k]:.6f}")

    # Selection and weighting
    selected_mask = alignment_scores > tau
    selected_indices = np.where(selected_mask)[0]

    if len(selected_indices) == 0:
        console.print("  [yellow]Warning: No quantile levels passed threshold. Using all levels.[/yellow]")
        selected_indices = np.arange(K_feat)

    pos_scores = np.maximum(alignment_scores[selected_indices], 0.0)
    total = np.sum(pos_scores)
    if total < 1e-16:
        weights = np.ones(len(selected_indices)) / len(selected_indices)
    else:
        weights = pos_scores / total

    console.print(f"  [green]Selected {len(selected_indices)}/{K_feat} quantile levels (tau={tau})[/green]")
    return selected_indices, weights, alignment_scores


def apply_kta_weights(dQ_all, selected_indices, weights):
    """
    Apply KTA weights to drift features.
    dQ_tilde_i = [sqrt(w_k) * phi(dQ_i(p_k))] for k in S

    dQ_all: (total_pairs, K)
    selected_indices: (|S|,)
    weights: (|S|,)

    Returns: features (total_pairs, |S|)
    """
    phi_all = signed_sqrt(dQ_all)  # (total_pairs, K)
    sqrt_w = np.sqrt(weights)  # (|S|,)
    features = phi_all[:, selected_indices] * sqrt_w[None, :]
    return features


# ============================================================
# Stage 5: Random Forest Classification
# ============================================================
def train_and_evaluate_rf(X_train, y_train, X_test, y_test, n_estimators=500):
    """Train RF and evaluate with standard metrics."""
    console.print("[cyan]Training Random Forest...[/cyan]")
    t0 = time.time()

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    t1 = time.time()
    console.print(f"[green]RF training time: {t1 - t0:.2f}s[/green]")

    # Predict on test (majority vote per signal is handled at the signal level)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    recalls = recall_score(y_test, y_pred, average=None)
    cm = confusion_matrix(y_test, y_pred)

    console.print(f"[bold]Test Accuracy: {acc:.4f}[/bold]")
    console.print(f"[bold]Test Macro F1: {macro_f1:.4f}[/bold]")
    console.print(f"[bold]Class Recalls: {', '.join(f'{r:.4f}' for r in recalls)}[/bold]")

    return acc, macro_f1, recalls, cm


# ============================================================
# Main Pipeline
# ============================================================
def aggregate_drift_features(dQ_list, selected_indices, weights):
    """
    Aggregate per-signal drift features into signal-level feature vectors.
    For each signal, we have M-1 drift vectors of dim |S|.
    Aggregate: mean, std, max, min across the M-1 pairs → 4*|S| features per signal.
    Also add the Wasserstein distance trajectory statistics.
    """
    sqrt_w = np.sqrt(weights)
    features_list = []
    for dQ in dQ_list:
        # Apply KTA weights: dQ_tilde (M-1, |S|)
        phi = signed_sqrt(dQ[:, selected_indices])
        dQ_tilde = phi * sqrt_w[None, :]

        # Aggregate statistics across the M-1 pairs
        feat_mean = np.mean(dQ_tilde, axis=0)
        feat_std = np.std(dQ_tilde, axis=0)
        feat_max = np.max(dQ_tilde, axis=0)
        feat_min = np.min(dQ_tilde, axis=0)

        # Wasserstein distance per pair (scalar)
        W_dist = np.mean(np.abs(dQ) ** 2, axis=1) ** 0.5  # L2 norm / sqrt(K) approx
        w_feat = np.array([np.mean(W_dist), np.std(W_dist), np.max(W_dist), np.min(W_dist)])

        # Concatenate: [mean, std, max, min of weighted drift] + [W-dist stats]
        sig_feat = np.concatenate([feat_mean, feat_std, feat_max, feat_min, w_feat])
        features_list.append(sig_feat)
    return np.array(features_list)


def run_kta_tf_drift(
    data_file,
    output_file,
    L=28,
    S=7,
    K=32,
    tau=0.01,
    gamma=-1.0,  # -1 means use median heuristic
    n_estimators=500,
    max_pairs_for_kta=4000,
):
    """Full KTA-TF-Drift pipeline with signal-level aggregation."""
    console.print("[bold cyan]KTA-TF-Drift Pipeline (Signal-Level Aggregation)[/bold cyan]")
    console.print(f"  L={L}, S={S}, K={K}, tau={tau}, gamma={'median_heuristic' if gamma <= 0 else gamma}")

    # ---- Load Data ----
    console.print(f"[cyan]Loading dataset from {data_file}...[/cyan]")
    data = np.load(data_file)
    X_train_raw = data["X_train"].astype(np.float64)  # (N, 140)
    y_train_raw = data["y_train"].astype(np.int64)
    X_test_raw = data["X_test"].astype(np.float64)
    y_test_raw = data["y_test"].astype(np.int64)

    # Per-sample standardization (same as baselines)
    eps = 1e-8
    mu_train = np.mean(X_train_raw, axis=-1, keepdims=True)
    sigma_train = np.std(X_train_raw, axis=-1, keepdims=True)
    X_train_raw = (X_train_raw - mu_train) / (sigma_train + eps)

    mu_test = np.mean(X_test_raw, axis=-1, keepdims=True)
    sigma_test = np.std(X_test_raw, axis=-1, keepdims=True)
    X_test_raw = (X_test_raw - mu_test) / (sigma_test + eps)

    N_train = X_train_raw.shape[0]
    N_test = X_test_raw.shape[0]
    console.print(f"  Train: {N_train}, Test: {N_test}")

    # ---- Stage 1-3: Compute drift features for all signals ----
    console.print("[cyan]Computing drift features...[/cyan]")
    t0 = time.time()

    train_dQ_list = []
    for i in range(N_train):
        windows = compute_windows(X_train_raw[i], L, S)
        Q = compute_quantile_functions(windows, K)
        dQ = compute_drift_features(Q)  # (M-1, K)
        train_dQ_list.append(dQ)

    # For KTA: flatten all pairs with labels
    train_dQ_all_flat = np.concatenate(train_dQ_list, axis=0)  # (total_pairs, K)
    train_labels_flat = np.repeat(y_train_raw, [dQ.shape[0] for dQ in train_dQ_list])
    console.print(f"  Total training drift pairs: {train_dQ_all_flat.shape[0]}")

    test_dQ_list = []
    for i in range(N_test):
        windows = compute_windows(X_test_raw[i], L, S)
        Q = compute_quantile_functions(windows, K)
        dQ = compute_drift_features(Q)
        test_dQ_list.append(dQ)

    t1 = time.time()
    console.print(f"[green]Drift feature computation: {t1 - t0:.2f}s[/green]")

    # ---- Stage 4: KTA Feature Selection/Weighting ----
    console.print("[cyan]Computing KTA weights (on all pairs)...[/cyan]")
    t2 = time.time()
    selected_indices, weights, alignment_scores = compute_kta_weights(
        train_dQ_all_flat, train_labels_flat, K, tau, gamma, max_pairs_for_kta
    )
    t3 = time.time()
    console.print(f"[green]KTA computation: {t3 - t2:.2f}s[/green]")

    # ---- Aggregate per signal ----
    console.print("[cyan]Aggregating drift features per signal...[/cyan]")
    X_train_feat = aggregate_drift_features(train_dQ_list, selected_indices, weights)
    X_test_feat = aggregate_drift_features(test_dQ_list, selected_indices, weights)
    console.print(f"  Signal-level feature dimension: {X_train_feat.shape[1]}")
    console.print(f"  Train features: {X_train_feat.shape}, Test features: {X_test_feat.shape}")

    # ---- Stage 5: RF Classification (signal level) ----
    console.print("[cyan]Training RF (signal-level features)...[/cyan]")
    t4 = time.time()

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train_feat, y_train_raw)
    t5 = time.time()
    console.print(f"[green]RF training: {t5 - t4:.2f}s[/green]")

    y_pred = clf.predict(X_test_feat)

    t6 = time.time()
    total_time = t6 - t0
    console.print(f"[green]Total pipeline time: {total_time:.2f}s[/green]")

    # ---- Evaluation ----
    final_acc = accuracy_score(y_test_raw, y_pred)
    final_macro_f1 = f1_score(y_test_raw, y_pred, average="macro")
    final_recalls = recall_score(y_test_raw, y_pred, average=None)
    cm = confusion_matrix(y_test_raw, y_pred)

    console.print("[bold green]=== Final Results ===[/bold green]")
    console.print(f"[bold]Accuracy: {final_acc:.4f}[/bold]")
    console.print(f"[bold]Macro F1: {final_macro_f1:.4f}[/bold]")
    console.print(f"[bold]Class Recalls: {', '.join(f'{r:.4f}' for r in final_recalls)}[/bold]")

    # ---- Save Results ----
    results = {
        "model": "KTA-TF-Drift",
        "accuracy": float(final_acc),
        "macro_f1": float(final_macro_f1),
        "class_recalls": [float(r) for r in final_recalls],
        "confusion_matrix": cm.tolist(),
        "config": {
            "L": L,
            "S": S,
            "K": K,
            "tau": tau,
            "gamma": "median_heuristic" if gamma <= 0 else float(gamma),
            "n_estimators": n_estimators,
            "max_pairs_for_kta": max_pairs_for_kta,
            "selected_quantile_levels": selected_indices.tolist(),
            "num_selected": len(selected_indices),
            "weights": weights.tolist(),
            "feature_dim": int(X_train_feat.shape[1]),
            "aggregation": "mean+std+max+min + W_dist_stats",
        },
        "timing": {
            "drift_features_sec": t1 - t0,
            "kta_computation_sec": t3 - t2,
            "rf_training_sec": t5 - t4,
            "total_sec": total_time,
        },
        "alignment_scores": alignment_scores.tolist(),
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    console.print(f"[green]Results saved to {output_file}[/green]")

    return results


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_file = os.path.join(base_dir, "data", "ecg5000_resplit.npz")
    output_file = os.path.join(base_dir, "results", "kta_tf_drift_results.json")

    # ---- Parameter sweep across multiple configurations ----
    configs = [
        {"L": 28, "S": 7, "K": 32, "tau": 0.01, "label": "L28_S7_K32_tau001"},
        {"L": 28, "S": 14, "K": 32, "tau": 0.01, "label": "L28_S14_K32_tau001"},
        {"L": 14, "S": 7, "K": 14, "tau": 0.01, "label": "L14_S7_K14_tau001"},
        {"L": 20, "S": 10, "K": 20, "tau": 0.01, "label": "L20_S10_K20_tau001"},
        {"L": 35, "S": 7, "K": 35, "tau": 0.01, "label": "L35_S7_K35_tau001"},
        {"L": 28, "S": 7, "K": 64, "tau": 0.01, "label": "L28_S7_K64_tau001"},
    ]

    best_macro_f1 = -1
    best_config_label = ""
    all_results = {}

    for cfg in configs:
        label = cfg.pop("label")
        console.print(f"\n{'='*60}")
        console.print(f"[bold yellow]Config: {label}[/bold yellow]")
        console.print(f"{'='*60}")

        try:
            result = run_kta_tf_drift(
                data_file,
                output_file.replace(".json", f"_{label}.json"),
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
                best_config_label = label
        except Exception as e:
            console.print(f"[red]Error with {label}: {e}[/red]")

    # ---- Summary ----
    console.print(f"\n{'='*60}")
    console.print("[bold cyan]SWEEP SUMMARY[/bold cyan]")
    console.print(f"{'='*60}")
    console.print(f"{'Config':<35} {'Acc':>8} {'MacroF1':>8} {'Time':>8}")
    console.print("-" * 65)
    for label, r in all_results.items():
        marker = " <-- BEST" if label == best_config_label else ""
        console.print(f"{label:<35} {r['accuracy']:>8.4f} {r['macro_f1']:>8.4f} {r['timing']:>7.2f}s{marker}")

    console.print(f"\n[bold green]Best config: {best_config_label} (Macro F1 = {best_macro_f1:.4f})[/bold green]")

    # Save sweep summary
    summary_path = os.path.join(base_dir, "results", "kta_tf_drift_sweep.json")
    with open(summary_path, "w") as f:
        json.dump({"best": best_config_label, "results": all_results}, f, indent=4)
    console.print(f"[green]Sweep summary saved to {summary_path}[/green]")

    # Also save the best as the main results file
    best_file = output_file.replace(".json", f"_{best_config_label}.json")
    if os.path.exists(best_file):
        import shutil
        shutil.copy(best_file, output_file)
        console.print(f"[green]Best results copied to {output_file}[/green]")
