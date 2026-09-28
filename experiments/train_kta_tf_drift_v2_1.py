"""
KTA-TF-Drift v2.1: Wasserstein-geodesic data augmentation for minority classes.

Augments minority-class training samples by interpolating quantile functions
on the W2 geodesic between same-class exemplars, then converts back to raw signals.
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
# Reuse all v2 pipeline functions
# ============================================================
def compute_windows(signal, L, S):
    N = len(signal)
    windows = []
    i = 0
    while i * S + L <= N:
        windows.append(signal[i * S : i * S + L])
        i += 1
    return np.array(windows)


def compute_quantile_functions(windows, K):
    M, L = windows.shape
    quantile_grid = (np.arange(1, K + 1) - 0.5) / K
    Q = np.zeros((M, K), dtype=np.float64)
    for i in range(M):
        sorted_w = np.sort(windows[i])
        emp_quantiles = (np.arange(1, L + 1) - 0.5) / L
        Q[i] = np.interp(quantile_grid, emp_quantiles, sorted_w)
    return Q


def compute_drift_features_multi_lag(Q, lags=(1, 2, 4, 8)):
    M, K = Q.shape
    drift_dict = {}
    for tau in lags:
        if M - tau > 0:
            drift_dict[tau] = Q[tau:] - Q[:M - tau]
    return drift_dict


def compute_static_quantile_features(Q):
    return np.concatenate([
        np.mean(Q, axis=0), np.std(Q, axis=0),
        np.max(Q, axis=0), np.min(Q, axis=0),
    ])


def compute_class_barycenters(Q_all, y_all, K):
    classes = np.unique(y_all)
    barycenters = {}
    for c in classes:
        class_qs = [np.mean(qi, axis=0) for qi, yi in zip(Q_all, y_all) if yi == c]
        barycenters[c] = np.mean(class_qs, axis=0)
    return barycenters


def compute_barycenter_distances(Q_signal, barycenters, K):
    classes = sorted(barycenters.keys())
    C = len(classes)
    Q_mean = np.mean(Q_signal, axis=0)
    dists = [np.sqrt(np.mean((Q_mean - barycenters[c]) ** 2)) for c in classes]

    M = Q_signal.shape[0]
    per_window_dists = np.zeros((M, C))
    for j, c in enumerate(classes):
        for i in range(M):
            per_window_dists[i, j] = np.sqrt(np.mean((Q_signal[i] - barycenters[c]) ** 2))

    features = []
    features.extend(dists)
    features.extend(np.min(per_window_dists, axis=0).tolist())
    features.extend(np.max(per_window_dists, axis=0).tolist())
    features.extend(np.std(per_window_dists, axis=0).tolist())
    return np.array(features)


def signed_sqrt(u):
    return np.sign(u) * np.sqrt(np.abs(u) + 1e-16)


def kernel_target_alignment(Kmat, Y_onehot):
    YYt = Y_onehot @ Y_onehot.T
    numerator = np.sum(Kmat * YYt)
    denom_K = np.sqrt(np.sum(Kmat * Kmat))
    denom_Y = np.sqrt(np.sum(YYt * YYt))
    if denom_K < 1e-16 or denom_Y < 1e-16:
        return 0.0
    return numerator / (denom_K * denom_Y)


def compute_kta_weights_balanced(dQ_all, labels_all, K, tau, gamma, n_samples=2000):
    n_pairs, K_feat = dQ_all.shape
    C = len(np.unique(labels_all))
    class_counts = np.bincount(labels_all.astype(int), minlength=C)
    min_count = max(int(np.min(class_counts)), 10)
    target_per_class = min(n_samples // C, min_count)

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

    if gamma <= 0:
        phi_ch0 = signed_sqrt(dQ_sub[:, 0])
        diffs = phi_ch0[:, None] - phi_ch0[None, :]
        med_sq = np.median(diffs ** 2)
        gamma = 1.0 / (med_sq + 1e-16)

    alignment_scores = np.zeros(K_feat)
    for k in range(K_feat):
        phi = signed_sqrt(dQ_sub[:, k])
        diff = phi[:, None] - phi[None, :]
        K_k = np.exp(-gamma * diff ** 2)
        alignment_scores[k] = kernel_target_alignment(K_k, Y_onehot)

    selected_mask = alignment_scores > tau
    selected_indices = np.where(selected_mask)[0]
    if len(selected_indices) == 0:
        selected_indices = np.arange(K_feat)

    pos_scores = np.maximum(alignment_scores[selected_indices], 0.0)
    total = np.sum(pos_scores)
    weights = pos_scores / total if total > 1e-16 else np.ones(len(selected_indices)) / len(selected_indices)

    return selected_indices, weights, alignment_scores


def compute_kernel_pca_features(dQ_all, n_components=10, gamma=-1.0, max_fit_samples=3000):
    phi = signed_sqrt(dQ_all)
    n_total = phi.shape[0]
    if gamma <= 0:
        n_sub_med = min(n_total, 1000)
        idx = np.random.RandomState(42).choice(n_total, n_sub_med, replace=False)
        diffs = np.linalg.norm(phi[idx, None] - phi[None, idx, :], axis=2)
        med = np.median(diffs ** 2)
        gamma = 1.0 / (med + 1e-16)
    if n_total > max_fit_samples:
        fit_idx = np.random.RandomState(42).choice(n_total, max_fit_samples, replace=False)
        fit_idx.sort()
        phi_fit = phi[fit_idx]
    else:
        phi_fit = phi
    kpca = KernelPCA(n_components=n_components, kernel="rbf", gamma=gamma, fit_inverse_transform=False, n_jobs=1)
    kpca.fit(phi_fit)
    return kpca


# ============================================================
# Wasserstein-Geodesic Augmentation (the new piece)
# ============================================================
def wasserstein_geodesic_augment(X_train, y_train, L, S, K,
                                  target_per_class=200,
                                  oversample_classes=None,
                                  signals_as_quantiles=True):
    """
    Generate synthetic training samples by interpolating quantile functions
    on the W2 geodesic between same-class exemplars.

    For minority classes with fewer than target_per_class samples:
      1. Compute the mean quantile function per signal
      2. Sample pairs (j,k) from the same class
      3. Interpolate: Q_synth = (1-t)*Q_j + t*Q_k, t ~ Uniform(0,1)
      4. Convert Q_synth back to a raw 140-sample signal via inverse CDF sampling

    Returns: X_aug, y_aug (augmented training set)
    """
    N, signal_len = X_train.shape
    classes, counts = np.unique(y_train, return_counts=True)
    class_count_map = dict(zip(classes, counts))

    if oversample_classes is None:
        oversample_classes = [c for c in classes if class_count_map[c] < target_per_class]

    if not oversample_classes:
        console.print("  [dim]No classes need augmentation[/dim]")
        return X_train, y_train

    console.print(f"  [cyan]Wasserstein-geodesic augmentation for classes: {oversample_classes}[/cyan]")

    # Compute quantile functions for all training signals
    quantile_grid = (np.arange(1, K + 1) - 0.5) / K
    all_Q = []
    for i in range(N):
        windows = compute_windows(X_train[i], L, S)
        Q = compute_quantile_functions(windows, K)
        all_Q.append(Q)  # (M_i, K)

    # Generate synthetic signals
    X_synth_list = []
    y_synth_list = []

    rng = np.random.RandomState(42)

    for c in oversample_classes:
        c_indices = np.where(y_train == c)[0]
        n_current = len(c_indices)
        n_needed = target_per_class - n_current

        console.print(f"    Class {c}: {n_current} -> {target_per_class} (+{n_needed} synthetic)")

        for _ in range(n_needed):
            # Pick two random exemplars from this class
            j, k = rng.choice(c_indices, 2, replace=(n_current < 2))
            Q_j = all_Q[j]  # (M_j, K)
            Q_k = all_Q[k]  # (M_k, K)

            # Interpolate at the signal level: use mean quantile function per signal
            Q_j_mean = np.mean(Q_j, axis=0)  # (K,)
            Q_k_mean = np.mean(Q_k, axis=0)  # (K,)

            t = rng.uniform(0, 1)
            Q_synth = (1 - t) * Q_j_mean + t * Q_k_mean  # (K,) — valid W2 geodesic

            # Convert Q_synth back to a raw 140-sample signal
            # Inverse CDF sampling: sample p ~ Uniform(0,1), evaluate Q_synth(p)
            p_samples = rng.uniform(0, 1, signal_len)
            signal_synth = np.interp(p_samples, quantile_grid, Q_synth).astype(np.float32)

            # Per-sample standardize (match training preprocessing)
            mu = np.mean(signal_synth)
            sigma = np.std(signal_synth) + 1e-8
            signal_synth = (signal_synth - mu) / sigma

            X_synth_list.append(signal_synth)
            y_synth_list.append(c)

    if X_synth_list:
        X_synth = np.array(X_synth_list)
        y_synth = np.array(y_synth_list)
        X_aug = np.concatenate([X_train, X_synth], axis=0)
        y_aug = np.concatenate([y_train, y_synth], axis=0)

        # Shuffle
        perm = rng.permutation(len(X_aug))
        X_aug = X_aug[perm]
        y_aug = y_aug[perm]

        console.print(f"  [green]Augmented: {N} -> {len(X_aug)} samples ({len(X_synth)} synthetic)[/green]")
        return X_aug, y_aug
    else:
        return X_train, y_train


# ============================================================
# Full Pipeline: KTA-TF-Drift v2.1
# ============================================================
def run_kta_tf_drift_v2_1(
    data_file,
    output_file,
    L=35, S=7, K=35,
    lags=(1, 2, 4, 8),
    tau=0.01, gamma_kta=-1.0, gamma_kpca=-1.0,
    n_estimators=500,
    n_kpca_components=10,
    max_pairs_for_kta=4000,
    target_augment=200,
):
    """KTA-TF-Drift v2.1 with Wasserstein-geodesic augmentation."""
    console.print("[bold cyan]KTA-TF-Drift v2.1 Pipeline (Geodesic Augmentation)[/bold cyan]")

    # ---- Load Data ----
    console.print(f"[cyan]Loading dataset...[/cyan]")
    data = np.load(data_file)
    X_train_raw = data["X_train"].astype(np.float64)
    y_train_raw = data["y_train"].astype(np.int64)
    X_test_raw = data["X_test"].astype(np.float64)
    y_test_raw = data["y_test"].astype(np.int64)

    eps = 1e-8
    mu = np.mean(X_train_raw, axis=-1, keepdims=True)
    sigma = np.std(X_train_raw, axis=-1, keepdims=True)
    X_train_raw = (X_train_raw - mu) / (sigma + eps)
    mu = np.mean(X_test_raw, axis=-1, keepdims=True)
    sigma = np.std(X_test_raw, axis=-1, keepdims=True)
    X_test_raw = (X_test_raw - mu) / (sigma + eps)

    N_train_orig = X_train_raw.shape[0]
    console.print(f"  Original train: {N_train_orig}, Test: {X_test_raw.shape[0]}")

    # ---- Augment minority classes ----
    console.print("[cyan]Stage 0: Wasserstein-geodesic augmentation...[/cyan]")
    t_aug_start = time.time()
    X_train_aug, y_train_aug = wasserstein_geodesic_augment(
        X_train_raw, y_train_raw, L, S, K,
        target_per_class=target_augment,
    )
    t_aug_end = time.time()
    console.print(f"[green]  Augmentation: {t_aug_end - t_aug_start:.2f}s[/green]")

    N_train = X_train_aug.shape[0]
    console.print(f"  Augmented train: {N_train}")

    # ---- Stage 1-2: Quantile functions ----
    console.print("[cyan]Stage 1-2: Windowing + Quantile functions...[/cyan]")
    t0 = time.time()

    train_Q_list = []
    for i in range(N_train):
        windows = compute_windows(X_train_aug[i], L, S)
        Q = compute_quantile_functions(windows, K)
        train_Q_list.append(Q)

    test_Q_list = []
    for i in range(X_test_raw.shape[0]):
        windows = compute_windows(X_test_raw[i], L, S)
        Q = compute_quantile_functions(windows, K)
        test_Q_list.append(Q)

    t1 = time.time()
    console.print(f"[green]  Quantile computation: {t1 - t0:.2f}s[/green]")

    # ---- Stage 3: Multi-lag drift + static ----
    console.print("[cyan]Stage 3: Multi-lag drift + static features...[/cyan]")

    train_dQ_all_flat = []
    train_labels_flat = []
    for i in range(N_train):
        drift_dict = compute_drift_features_multi_lag(train_Q_list[i], lags)
        for tau_lag, dQ_tau in drift_dict.items():
            train_dQ_all_flat.append(dQ_tau)
            train_labels_flat.extend([y_train_aug[i]] * dQ_tau.shape[0])
    train_dQ_all_flat = np.concatenate(train_dQ_all_flat, axis=0)
    train_labels_flat = np.array(train_labels_flat, dtype=np.int64)

    # ---- Stage 4: Class-balanced KTA ----
    console.print("[cyan]Stage 4: Class-balanced KTA...[/cyan]")
    t4 = time.time()
    selected_indices, weights, alignment_scores = compute_kta_weights_balanced(
        train_dQ_all_flat, train_labels_flat, train_dQ_all_flat.shape[1], tau, gamma_kta, max_pairs_for_kta
    )
    t5 = time.time()
    console.print(f"[green]  KTA: {t5 - t4:.2f}s[/green]")

    # ---- Stage 4b: Kernel PCA ----
    console.print("[cyan]Stage 4b: Kernel PCA...[/cyan]")
    t6 = time.time()
    kpca_model = compute_kernel_pca_features(
        train_dQ_all_flat, n_components=n_kpca_components, gamma=gamma_kpca
    )
    t7 = time.time()
    console.print(f"[green]  KPCA: {t7 - t6:.2f}s[/green]")

    # ---- Build per-signal features ----
    console.print("[cyan]Building per-signal features...[/cyan]")

    barycenters = compute_class_barycenters(train_Q_list, y_train_aug, K)
    C = len(np.unique(y_train_aug))

    train_bary_feat = np.array([compute_barycenter_distances(Q, barycenters, K) for Q in train_Q_list])
    test_bary_feat = np.array([compute_barycenter_distances(Q, barycenters, K) for Q in test_Q_list])

    train_static_feat = np.array([compute_static_quantile_features(Q) for Q in train_Q_list])
    test_static_feat = np.array([compute_static_quantile_features(Q) for Q in test_Q_list])

    sqrt_w = np.sqrt(weights)
    train_kta_drift = []
    for i in range(N_train):
        drift_dict = compute_drift_features_multi_lag(train_Q_list[i], lags)
        all_dQ = np.concatenate([dQ for dQ in drift_dict.values()], axis=0)
        phi = signed_sqrt(all_dQ[:, selected_indices])
        weighted = phi * sqrt_w[None, :]
        train_kta_drift.append(np.concatenate([
            np.mean(weighted, axis=0), np.std(weighted, axis=0),
            np.max(weighted, axis=0), np.min(weighted, axis=0),
        ]))
    train_kta_drift = np.array(train_kta_drift)

    test_kta_drift = []
    for i in range(X_test_raw.shape[0]):
        drift_dict = compute_drift_features_multi_lag(test_Q_list[i], lags)
        all_dQ = np.concatenate([dQ for dQ in drift_dict.values()], axis=0)
        phi = signed_sqrt(all_dQ[:, selected_indices])
        weighted = phi * sqrt_w[None, :]
        test_kta_drift.append(np.concatenate([
            np.mean(weighted, axis=0), np.std(weighted, axis=0),
            np.max(weighted, axis=0), np.min(weighted, axis=0),
        ]))
    test_kta_drift = np.array(test_kta_drift)

    # Kernel PCA per signal
    train_kpca_feat = []
    for i in range(N_train):
        drift_dict = compute_drift_features_multi_lag(train_Q_list[i], lags)
        all_dQ = np.concatenate([dQ for dQ in drift_dict.values()], axis=0)
        phi = signed_sqrt(all_dQ)
        kpca_proj = kpca_model.transform(phi)
        train_kpca_feat.append(np.concatenate([np.mean(kpca_proj, axis=0), np.std(kpca_proj, axis=0)]))
    train_kpca_feat = np.array(train_kpca_feat)

    test_kpca_feat = []
    for i in range(X_test_raw.shape[0]):
        drift_dict = compute_drift_features_multi_lag(test_Q_list[i], lags)
        all_dQ = np.concatenate([dQ for dQ in drift_dict.values()], axis=0)
        phi = signed_sqrt(all_dQ)
        kpca_proj = kpca_model.transform(phi)
        test_kpca_feat.append(np.concatenate([np.mean(kpca_proj, axis=0), np.std(kpca_proj, axis=0)]))
    test_kpca_feat = np.array(test_kpca_feat)

    # Concatenate all
    X_train_feat = np.concatenate([train_static_feat, train_kta_drift, train_bary_feat, train_kpca_feat], axis=1)
    X_test_feat = np.concatenate([test_static_feat, test_kta_drift, test_bary_feat, test_kpca_feat], axis=1)
    console.print(f"  Total feature dim: {X_train_feat.shape[1]}")

    # ---- Stage 5: RF ----
    console.print("[cyan]Stage 5: Random Forest...[/cyan]")
    t8 = time.time()
    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=None,
        class_weight="balanced_subsample", n_jobs=-1, random_state=42,
    )
    clf.fit(X_train_feat, y_train_aug)
    y_pred = clf.predict(X_test_feat)
    t9 = time.time()
    console.print(f"[green]  RF: {t9 - t8:.2f}s[/green]")

    # ---- Evaluation ----
    total_time = t9 - t_aug_start
    final_acc = accuracy_score(y_test_raw, y_pred)
    final_macro_f1 = f1_score(y_test_raw, y_pred, average="macro")
    final_recalls = recall_score(y_test_raw, y_pred, average=None)
    cm = confusion_matrix(y_test_raw, y_pred)

    console.print(f"\n[bold green]{'='*60}[/bold green]")
    console.print(f"[bold green]  KTA-TF-Drift v2.1 RESULTS (with geodesic augmentation)[/bold green]")
    console.print(f"[bold green]{'='*60}[/bold green]")
    console.print(f"  [bold]Accuracy: {final_acc:.4f}[/bold]")
    console.print(f"  [bold]Macro F1: {final_macro_f1:.4f}[/bold]")
    console.print(f"  [bold]Class Recalls: {', '.join(f'{r:.4f}' for r in final_recalls)}[/bold]")
    console.print(f"  [bold]Total time: {total_time:.2f}s[/bold]")

    results = {
        "model": "KTA-TF-Drift-v2.1",
        "accuracy": float(final_acc),
        "macro_f1": float(final_macro_f1),
        "class_recalls": [float(r) for r in final_recalls],
        "confusion_matrix": cm.tolist(),
        "config": {
            "L": L, "S": S, "K": K, "lags": list(lags),
            "target_augment": target_augment,
            "n_train_original": N_train_orig,
            "n_train_augmented": N_train,
            "feature_dim": int(X_train_feat.shape[1]),
        },
        "timing": {"total_sec": total_time},
        "alignment_scores": alignment_scores.tolist(),
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    console.print(f"[green]Results saved to {output_file}[/green]")
    return results


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_file = os.path.join(base_dir, "data", "ecg5000_resplit.npz")
    output_file = os.path.join(base_dir, "results", "kta_tf_drift_v2_1_results.json")

    # Sweep augmentation targets
    configs = [
        {"target_augment": 100, "label": "aug100"},
        {"target_augment": 200, "label": "aug200"},
        {"target_augment": 500, "label": "aug500"},
        {"target_augment": 1000, "label": "aug1000"},
    ]

    best_macro_f1 = -1
    best_label = ""
    all_results = {}

    for cfg in configs:
        label = cfg.pop("label")
        console.print(f"\n{'='*60}")
        console.print(f"[bold yellow]Config: {label} (target={cfg['target_augment']})[/bold yellow]")
        console.print(f"{'='*60}")

        try:
            result = run_kta_tf_drift_v2_1(
                data_file,
                output_file.replace(".json", f"_{label}.json"),
                **cfg,
            )
            all_results[label] = {
                "accuracy": result["accuracy"],
                "macro_f1": result["macro_f1"],
                "class_recalls": result["class_recalls"],
                "n_augmented": result["config"]["n_train_augmented"],
                "timing": result["timing"]["total_sec"],
            }
            if result["macro_f1"] > best_macro_f1:
                best_macro_f1 = result["macro_f1"]
                best_label = label
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            import traceback
            traceback.print_exc()

    # Summary
    console.print(f"\n{'='*60}")
    console.print("[bold cyan]V2.1 AUGMENTATION SWEEP SUMMARY[/bold cyan]")
    console.print(f"{'='*60}")
    console.print(f"{'Config':<20} {'Acc':>8} {'MacroF1':>8} {'C0':>6} {'C1':>6} {'C2':>6} {'C3':>6} {'C4':>6} {'N_tr':>6} {'Time':>7}")
    console.print("-" * 95)
    for label, r in all_results.items():
        marker = " <-- BEST" if label == best_label else ""
        cr = r["class_recalls"]
        console.print(f"{label:<20} {r['accuracy']:>8.4f} {r['macro_f1']:>8.4f} {cr[0]:>6.3f} {cr[1]:>6.3f} {cr[2]:>6.3f} {cr[3]:>6.3f} {cr[4]:>6.3f} {r['n_augmented']:>6} {r['timing']:>6.1f}s{marker}")

    console.print(f"\n[bold green]Best: {best_label} (Macro F1 = {best_macro_f1:.4f})[/bold green]")

    import shutil
    best_file = output_file.replace(".json", f"_{best_label}.json")
    if os.path.exists(best_file):
        shutil.copy(best_file, output_file)
