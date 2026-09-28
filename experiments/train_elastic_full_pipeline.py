"""
Full KTA-TF-Drift Pipeline on Elastic-Registered Signals

The definitive experiment: combine SRVF elastic registration with the
full downstream pipeline (quantile + drift + barycenter + balanced-KTA + RF),
under synthetic warping conditions.

Comparison:
  A) Full pipeline on raw (unregistered) signals — amplitude-only baseline
  B) Full pipeline on elastic-registered signals — elastic amplitude+phase
"""

import os
import sys
import json
import time
import warnings
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from rich.console import Console

warnings.filterwarnings("ignore")
console = Console()


# ============================================================
# SRVF Elastic Registration (compact version)
# ============================================================

def resample(f, N):
    return np.interp(np.linspace(0, 1, N), np.linspace(0, 1, len(f)), f)

def compute_srvf(f):
    N = len(f)
    dt = 1.0 / max(N - 1, 1)
    df = np.diff(f) / dt
    q = np.zeros(N)
    q[1:] = np.sign(df) * np.sqrt(np.abs(df))
    return q

def dp_align(q1, q2):
    N = len(q1)
    C = np.full((N + 1, N + 1), np.inf)
    C[0, 0] = 0.0
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            c = (q1[i - 1] - q2[j - 1]) ** 2
            C[i, j] = min(C[i - 1, j] + c, C[i, j - 1] + c, C[i - 1, j - 1] + c)
    path_m, path_n = [], []
    i, j = N, N
    while i > 0 or j > 0:
        path_m.append(i); path_n.append(j)
        if i == 0: j -= 1
        elif j == 0: i -= 1
        else:
            idx = np.argmin([C[i-1,j], C[i,j-1], C[i-1,j-1]])
            if idx == 0: i -= 1
            elif idx == 1: j -= 1
            else: i -= 1; j -= 1
    gamma = np.interp(np.arange(N + 1), np.array(path_m[::-1], dtype=float),
                       np.array(path_n[::-1], dtype=float))
    gamma = np.clip(gamma / max(N - 1, 1), 0, 1)
    return gamma, C[N, N] / N

def warp_by_gamma(f, gamma):
    warped = np.interp(gamma, np.linspace(0, 1, len(f)), f)
    return warped[1:]

def compute_psi(gamma):
    dg = np.maximum(np.diff(gamma), 1e-10)
    psi = np.sqrt(dg)
    return psi / (np.sqrt(np.sum(psi ** 2)) + 1e-10)

def karcher_mean(curves, N, tol=1e-3, max_iter=10):
    mu = curves[0].copy()
    q_mu = compute_srvf(mu)
    for _ in range(max_iter):
        aligned = []
        for f in curves:
            q_f = compute_srvf(f)
            gamma, _ = dp_align(q_mu, q_f)
            aligned.append(warp_by_gamma(f, gamma))
        mu_new = np.mean(aligned, axis=0)
        if np.sqrt(np.mean((mu_new - mu) ** 2)) < tol:
            return mu_new
        mu = mu_new
        q_mu = compute_srvf(mu)
    return mu


# ============================================================
# Full downstream pipeline (quantile + drift + barycenter + KTA + RF)
# ============================================================

def signed_sqrt(u):
    return np.sign(u) * np.sqrt(np.abs(u) + 1e-16)

def kernel_target_alignment(Kmat, Y_oh):
    YYt = Y_oh @ Y_oh.T
    return np.sum(Kmat * YYt) / (np.sqrt(np.sum(Kmat**2)) * np.sqrt(np.sum(YYt**2)) + 1e-16)

def balanced_kta(dQ_all, labels_all, K_feat, n_samples=2000):
    C = len(np.unique(labels_all))
    cc = np.bincount(labels_all.astype(int), minlength=C)
    target = min(n_samples // C, max(int(np.min(cc)), 10))
    rng = np.random.RandomState(42)
    idx = np.concatenate([rng.choice(np.where(labels_all == c)[0],
                          target, replace=(cc[c] < target)) for c in range(C)])
    rng.shuffle(idx)
    dQ_sub, lbl_sub = dQ_all[idx], labels_all[idx]
    Y_oh = (lbl_sub[:, None] == np.arange(C)[None, :]).astype(np.float64)
    phi0 = signed_sqrt(dQ_sub[:, 0])
    med_sq = np.median((phi0[:, None] - phi0[None, :]) ** 2)
    gamma_rbf = 1.0 / (med_sq + 1e-16)
    A = np.zeros(K_feat)
    for k in range(K_feat):
        phi = signed_sqrt(dQ_sub[:, k])
        Kk = np.exp(-gamma_rbf * (phi[:, None] - phi[None, :]) ** 2)
        A[k] = kernel_target_alignment(Kk, Y_oh)
    sel = np.where(A > 0)[0]
    if len(sel) == 0: sel = np.arange(K_feat)
    pos = np.maximum(A[sel], 0)
    tot = np.sum(pos)
    w = pos / tot if tot > 1e-16 else np.ones(len(sel)) / len(sel)
    return sel, w, A

def quantile_fn(signal, K):
    s = np.sort(signal)
    emp_q = (np.arange(1, len(s) + 1) - 0.5) / len(s)
    return np.interp((np.arange(1, K + 1) - 0.5) / K, emp_q, s)


def full_pipeline_features(X_tr, y_tr, X_te, y_te, K=32):
    """
    Full downstream: quantile -> drift -> barycenter -> balanced-KTA -> RF features.
    Returns feature matrices and classification results.
    """
    N_tr, N_te = len(X_tr), len(X_te)

    # --- Quantile functions ---
    Q_tr = np.array([quantile_fn(X_tr[i], K) for i in range(N_tr)])
    Q_te = np.array([quantile_fn(X_te[i], K) for i in range(N_te)])

    # --- Drift features (lag=1 quantile differences) ---
    dQ_tr = Q_tr[1:] - Q_tr[:-1] if N_tr > 1 else np.zeros((1, K))
    # For per-signal features, we use Q directly (drift needs windows, not signals)
    # Use Q and Q^2 as multi-resolution representation
    # Drift: consecutive quantile differences (pad last row)
    dQ_tr = np.diff(Q_tr, axis=0) if N_tr > 1 else np.zeros((1, K))
    dQ_tr = np.vstack([dQ_tr, dQ_tr[-1:]])  # pad to N_tr rows
    dQ_te = np.diff(Q_te, axis=0) if N_te > 1 else np.zeros((1, K))
    dQ_te = np.vstack([dQ_te, dQ_te[-1:]])

    dQ_tr_full = np.concatenate([Q_tr, Q_tr ** 2, dQ_tr], axis=1)
    dQ_te_full = np.concatenate([Q_te, Q_te ** 2, dQ_te], axis=1)

    K_feat = dQ_tr_full.shape[1]

    # --- Balanced KTA for feature selection ---
    labels_flat = np.repeat(y_tr, 1)  # one-to-one
    sel, w, A_scores = balanced_kta(dQ_tr_full, y_tr, K_feat)

    sqrt_w = np.sqrt(w)

    # --- Barycenter anchor features ---
    classes = np.unique(y_tr)
    Q_bary = {}
    for c in classes:
        Q_bary[c] = np.mean(Q_tr[y_tr == c], axis=0)

    def bary_features(Q_mat, Q_bary, classes):
        feats = []
        for Q in Q_mat:
            dists = [np.sqrt(np.mean((Q - Q_bary[c]) ** 2)) for c in classes]
            feats.append(dists)
        return np.array(feats)

    bary_tr = bary_features(Q_tr, Q_bary, classes)
    bary_te = bary_features(Q_te, Q_bary, classes)

    # --- Apply KTA weights ---
    phi_tr = signed_sqrt(dQ_tr_full[:, sel])
    phi_te = signed_sqrt(dQ_te_full[:, sel])
    wt_tr = phi_tr * sqrt_w[None, :]
    wt_te = phi_te * sqrt_w[None, :]

    # --- Static + weighted features ---
    static_tr = np.column_stack([np.mean(Q_tr, 1), np.std(Q_tr, 1),
                                  np.max(Q_tr, 1) - np.min(Q_tr, 1)])
    static_te = np.column_stack([np.mean(Q_te, 1), np.std(Q_te, 1),
                                  np.max(Q_te, 1) - np.min(Q_te, 1)])

    Xf_tr = np.column_stack([static_tr, wt_tr, bary_tr])
    Xf_te = np.column_stack([static_te, wt_te, bary_te])

    # --- RF ---
    clf = RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample",
                                  n_jobs=-1, random_state=42)
    clf.fit(Xf_tr, y_tr)
    y_pred = clf.predict(Xf_te)

    return {
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "macro_f1": float(f1_score(y_te, y_pred, average="macro")),
        "class_recalls": [float(r) for r in recall_score(y_te, y_pred, average=None)],
        "confusion_matrix": confusion_matrix(y_te, y_pred).tolist(),
        "feature_dim": Xf_tr.shape[1],
        "kta_selected": int(len(sel)),
        "kta_alignment_range": [float(A_scores.min()), float(A_scores.max())],
    }


def full_pipeline_elastic(X_tr, y_tr, X_te, y_te, T=80, K=32):
    """
    Elastic registration -> full downstream pipeline.
    Amplitude stream: de-warped signal -> full pipeline features
    Phase stream: psi features -> additional RF features
    """
    N_tr, N_te = len(X_tr), len(X_te)

    # Resample for DP alignment
    Xr_tr = np.array([resample(X_tr[i], T) for i in range(N_tr)])
    Xr_te = np.array([resample(X_te[i], T) for i in range(N_te)])

    # Karcher mean template
    console.print("    [dim]Computing Karcher mean template...[/dim]")
    t0 = time.time()
    mu = karcher_mean(Xr_tr[:min(200, N_tr)].tolist(), T, tol=1e-3, max_iter=8)
    console.print(f"    [dim]Template: {time.time()-t0:.1f}s[/dim]")

    # Register all signals
    console.print("    [dim]Registering all signals...[/dim]")
    t1 = time.time()

    def register_batch(Xr):
        f_tilde_list, psi_list = [], []
        q_mu = compute_srvf(mu)
        for f in Xr:
            q_f = compute_srvf(f)
            gamma, _ = dp_align(q_mu, q_f)
            f_tilde_list.append(warp_by_gamma(f, gamma))
            psi_list.append(compute_psi(gamma))
        return np.array(f_tilde_list), np.array(psi_list)

    amp_tr, psi_tr = register_batch(Xr_tr)
    amp_te, psi_te = register_batch(Xr_te)
    console.print(f"    [dim]Registration: {time.time()-t1:.1f}s[/dim]")

    # --- Amplitude stream: full downstream on de-warped signals ---
    console.print("    [dim]Amplitude stream features...[/dim]")
    amp_result = full_pipeline_features(amp_tr, y_tr, amp_te, y_te, K)

    # --- Phase stream features ---
    psi_h = np.mean(psi_tr[y_tr == 0][:min(100, np.sum(y_tr == 0))], axis=0)

    phase_tr = np.column_stack([
        np.mean(psi_tr, 1), np.std(psi_tr, 1),
        (np.max(psi_tr, 1) - np.min(psi_tr, 1)),
        np.sqrt(np.mean((psi_tr - psi_h) ** 2, 1)),
    ])
    phase_te = np.column_stack([
        np.mean(psi_te, 1), np.std(psi_te, 1),
        (np.max(psi_te, 1) - np.min(psi_te, 1)),
        np.sqrt(np.mean((psi_te - psi_h) ** 2, 1)),
    ])

    # --- Combined: amplitude features + phase features ---
    # Rebuild amplitude feature matrices for combination
    Q_tr = np.array([quantile_fn(amp_tr[i], K) for i in range(N_tr)])
    Q_te = np.array([quantile_fn(amp_te[i], K) for i in range(N_te)])
    classes = np.unique(y_tr)
    Q_bary = {c: np.mean(Q_tr[y_tr == c], axis=0) for c in classes}

    bary_tr = np.array([[np.sqrt(np.mean((Q_tr[i] - Q_bary[c])**2)) for c in classes] for i in range(N_tr)])
    bary_te = np.array([[np.sqrt(np.mean((Q_te[i] - Q_bary[c])**2)) for c in classes] for i in range(N_te)])

    static_tr = np.column_stack([np.mean(Q_tr, 1), np.std(Q_tr, 1), np.max(Q_tr, 1) - np.min(Q_tr, 1)])
    static_te = np.column_stack([np.mean(Q_te, 1), np.std(Q_te, 1), np.max(Q_te, 1) - np.min(Q_te, 1)])

    # KTA on amplitude features
    dQ_tr_full = np.concatenate([Q_tr, Q_tr**2], axis=1)
    dQ_te_full = np.concatenate([Q_te, Q_te**2], axis=1)
    K_feat = dQ_tr_full.shape[1]
    sel, w, A_scores = balanced_kta(dQ_tr_full, y_tr, K_feat)
    sqrt_w = np.sqrt(w)
    phi_tr = signed_sqrt(dQ_tr_full[:, sel]) * sqrt_w[None, :]
    phi_te = signed_sqrt(dQ_te_full[:, sel]) * sqrt_w[None, :]

    # Combined features
    Xf_tr = np.column_stack([static_tr, phi_tr, bary_tr, phase_tr])
    Xf_te = np.column_stack([static_te, phi_te, bary_te, phase_te])

    clf = RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample",
                                  n_jobs=-1, random_state=42)
    clf.fit(Xf_tr, y_tr)
    y_pred = clf.predict(Xf_te)

    return {
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "macro_f1": float(f1_score(y_te, y_pred, average="macro")),
        "class_recalls": [float(r) for r in recall_score(y_te, y_pred, average=None)],
        "confusion_matrix": confusion_matrix(y_te, y_pred).tolist(),
        "feature_dim": Xf_tr.shape[1],
        "amp_feature_dim": static_tr.shape[1] + phi_tr.shape[1] + bary_tr.shape[1],
        "phase_feature_dim": phase_tr.shape[1],
        "kta_selected": int(len(sel)),
    }


# ============================================================
# Synthetic warping
# ============================================================

def random_warping(N, strength, rng):
    n_ctrl = 8
    cx = np.linspace(0, 1, n_ctrl)
    cy = cx + strength * rng.randn(n_ctrl) * cx * (1 - cx) * 4
    cy = np.sort(cy); cy[0], cy[-1] = 0, 1
    cy = np.clip(cy, 0, 1)
    gamma = np.interp(np.linspace(0, 1, N), cx, cy)
    gamma = np.sort(gamma)
    return (gamma - gamma[0]) / (gamma[-1] - gamma[0] + 1e-10)

def apply_warping(X, strength, seed):
    rng = np.random.RandomState(seed)
    Xw = np.zeros_like(X)
    for i in range(len(X)):
        g = random_warping(X.shape[1], strength, rng)
        Xw[i] = np.interp(g, np.linspace(0, 1, X.shape[1]), X[i])
    return Xw


# ============================================================
# Main experiment
# ============================================================

def run_experiment(data_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    console.print("[bold cyan]DEFINITIVE EXPERIMENT: Full Pipeline + Elastic Registration[/bold cyan]")
    console.print("[cyan]Does the full KTA-TF-Drift architecture benefit from elastic registration?[/cyan]\n")

    data = np.load(data_file)
    X_tr = data["X_train"].astype(np.float64)
    y_tr = data["y_train"].astype(np.int64)
    X_te = data["X_test"].astype(np.float64)
    y_te = data["y_test"].astype(np.int64)

    eps = 1e-8
    m, s = np.mean(X_tr, -1, keepdims=True), np.std(X_tr, -1, keepdims=True)
    X_tr = (X_tr - m) / (s + eps)
    m, s = np.mean(X_te, -1, keepdims=True), np.std(X_te, -1, keepdims=True)
    X_te = (X_te - m) / (s + eps)

    console.print(f"Data: Train={len(X_tr)}, Test={len(X_te)}, Length={X_tr.shape[1]}\n")
    results = {}
    T = 80

    warping_strengths = [0.0, 0.15, 0.30, 0.50]

    for ws in warping_strengths:
        console.print(f"{'='*60}")
        if ws == 0.0:
            console.print(f"[bold]No warping (original ECG5000)[/bold]")
        else:
            console.print(f"[bold]Synthetic warping (strength={ws})[/bold]")
        console.print("=" * 60)

        if ws > 0.0:
            X_tr_w = apply_warping(X_tr, ws, seed=42)
            X_te_w = apply_warping(X_te, ws, seed=123)
            m, s = np.mean(X_tr_w, -1, keepdims=True), np.std(X_tr_w, -1, keepdims=True)
            X_tr_w = (X_tr_w - m) / (s + eps)
            m, s = np.mean(X_te_w, -1, keepdims=True), np.std(X_te_w, -1, keepdims=True)
            X_te_w = (X_te_w - m) / (s + eps)
        else:
            X_tr_w, X_te_w = X_tr, X_te

        # A) Full pipeline, amplitude-only (no registration)
        console.print(f"\n  [bold]A) Full pipeline, amplitude-only (no registration):[/bold]")
        t0 = time.time()
        r = full_pipeline_features(X_tr_w, y_tr, X_te_w, y_te)
        dt = time.time() - t0
        console.print(f"  Acc={r['accuracy']:.4f}, MacroF1={r['macro_f1']:.4f}, "
                      f"dim={r['feature_dim']}, KTA_sel={r['kta_selected']} ({dt:.1f}s)")
        results[f"ws{ws}_amp_only"] = r

        # B) Full pipeline + elastic registration (amplitude + phase)
        console.print(f"\n  [bold]B) Full pipeline + elastic registration (amplitude + phase):[/bold]")
        t0 = time.time()
        r = full_pipeline_elastic(X_tr_w, y_tr, X_te_w, y_te, T=T)
        dt = time.time() - t0
        console.print(f"  Acc={r['accuracy']:.4f}, MacroF1={r['macro_f1']:.4f}, "
                      f"dim={r['feature_dim']} (amp={r['amp_feature_dim']}, phase={r['phase_feature_dim']}), "
                      f"KTA_sel={r['kta_selected']} ({dt:.1f}s)")
        results[f"ws{ws}_elastic"] = r

    # ================================================================
    # Summary
    # ================================================================
    console.print(f"\n{'='*80}")
    console.print("[bold cyan]DEFINITIVE RESULTS: Full Pipeline + Elastic Registration[/bold cyan]")
    console.print("=" * 80)

    base_amp = results["ws0.0_amp_only"]["macro_f1"]
    base_el = results["ws0.0_elastic"]["macro_f1"]

    console.print(f"\n{'Warp':<8} {'Method':<28} {'Acc':>8} {'MacroF1':>8} {'dF1':>8} {'C0':>6} {'C2':>6} {'C3':>6} {'C4':>6}")
    console.print("-" * 90)

    for ws in warping_strengths:
        ra = results[f"ws{ws}_amp_only"]
        re = results[f"ws{ws}_elastic"]
        cr_a = ra["class_recalls"]
        cr_e = re["class_recalls"]
        label = "none" if ws == 0 else str(ws)
        console.print(f"{label:<8} {'Full pipeline (amp-only)':<28} {ra['accuracy']:>8.4f} "
                      f"{ra['macro_f1']:>8.4f} {ra['macro_f1']-base_amp:>+8.4f} "
                      f"{cr_a[0]:>6.3f} {cr_a[2]:>6.3f} {cr_a[3]:>6.3f} {cr_a[4]:>6.3f}")
        console.print(f"{'':8} {'Full pipeline + elastic':<28} {re['accuracy']:>8.4f} "
                      f"{re['macro_f1']:>8.4f} {re['macro_f1']-base_amp:>+8.4f} "
                      f"{cr_e[0]:>6.3f} {cr_e[2]:>6.3f} {cr_e[3]:>6.3f} {cr_e[4]:>6.3f}")
        if ws > 0:
            drop_a = base_amp - ra["macro_f1"]
            drop_e = base_amp - re["macro_f1"]
            recovery = max(0, (1 - drop_e / (drop_a + 1e-10)) * 100) if drop_a > 0.001 else 0
            console.print(f"{'':8} {'>>> Recovery:':<28} {recovery:>8.1f}% of warp-induced drop recovered")
        console.print()

    # Final summary
    console.print(f"[bold]Baseline comparisons (no warping):[/bold]")
    console.print(f"  MiniRocket:            Acc=0.956, MacroF1=0.601")
    console.print(f"  InceptionTime:         Acc=0.961, MacroF1=0.770")
    console.print(f"  KTA-TF-Drift v2.1:     Acc=0.950, MacroF1=0.589")
    console.print(f"  Full pipeline (amp):   Acc={results['ws0.0_amp_only']['accuracy']:.4f}, "
                  f"MacroF1={results['ws0.0_amp_only']['macro_f1']:.4f}")
    console.print(f"  Full pipeline (elastic):Acc={results['ws0.0_elastic']['accuracy']:.4f}, "
                  f"MacroF1={results['ws0.0_elastic']['macro_f1']:.4f}")

    # Verdict
    console.print(f"\n[bold]VERDICT:[/bold]")
    for ws in [0.15, 0.30, 0.50]:
        ra = results[f"ws{ws}_amp_only"]
        re = results[f"ws{ws}_elastic"]
        delta = re["macro_f1"] - ra["macro_f1"]
        console.print(f"  Warping {ws}: elastic {'+'if delta>=0 else ''}{delta:.4f} "
                      f"({'WINS' if delta > 0 else 'slightly behind'})")

    with open(os.path.join(output_dir, "elastic_full_pipeline.json"), "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n[green]Results saved to {output_dir}/elastic_full_pipeline.json[/green]")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    run_experiment(
        os.path.join(base_dir, "data", "ecg5000_resplit.npz"),
        os.path.join(base_dir, "results", "elastic_full_pipeline"),
    )
