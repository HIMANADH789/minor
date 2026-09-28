"""
SRVF Elastic Amplitude-Phase Separation for ECG5000 Classification

Validates the core claim: explicit amplitude-phase separation recovers
accuracy that naive amplitude-only features lose under time warping.
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
# SRVF and Elastic Alignment (all functions use length N consistently)
# ============================================================

def resample_signal(f, N):
    """Resample signal to N points on [0,1]."""
    x_old = np.linspace(0, 1, len(f))
    x_new = np.linspace(0, 1, N)
    return np.interp(x_new, x_old, f)


def compute_srvf(f):
    """SRVF of signal f of length N. Returns array of length N with q[0]=0."""
    N = len(f)
    dt = 1.0 / max(N - 1, 1)
    df = np.diff(f) / dt
    q = np.zeros(N)
    q[1:] = np.sign(df) * np.sqrt(np.abs(df))
    return q


def dp_align(q1, q2):
    """
    DP elastic alignment of q1 to q2. Both length N.
    Returns: gamma of length N (values in [0,1]), cost
    """
    N = len(q1)
    # Cost matrix: C[i,j] = cost to align q1[:i] to q2[:j]
    C = np.full((N + 1, N + 1), np.inf)
    C[0, 0] = 0.0

    for i in range(1, N + 1):
        for j in range(1, N + 1):
            c = (q1[i - 1] - q2[j - 1]) ** 2
            C[i, j] = min(
                C[i - 1, j] + c,
                C[i, j - 1] + c,
                C[i - 1, j - 1] + c,
            )

    # Backtrack
    path_m, path_n = [], []
    i, j = N, N
    while i > 0 or j > 0:
        path_m.append(i)
        path_n.append(j)
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            idx = np.argmin([C[i - 1, j], C[i, j - 1], C[i - 1, j - 1]])
            if idx == 0:
                i -= 1
            elif idx == 1:
                j -= 1
            else:
                i -= 1
                j -= 1

    path_m = np.array(path_m[::-1], dtype=float)
    path_n = np.array(path_n[::-1], dtype=float)
    gamma = np.interp(np.arange(N + 1), path_m, path_n)
    gamma = np.clip(gamma / max(N - 1, 1), 0, 1)
    return gamma, C[N, N] / N


def warp_by_gamma(f, gamma):
    """Apply gamma (length N+1, values [0,1]) to signal f (length N). Returns length N."""
    warped = np.interp(gamma, np.linspace(0, 1, len(f)), f)
    return warped[1:]  # drop the leading point to match original length


def compute_psi(gamma):
    """Fisher-Rao psi = sqrt(gamma') from gamma (length N+1). Returns length N."""
    dg = np.diff(gamma)
    dg = np.maximum(dg, 1e-10)
    psi = np.sqrt(dg)
    norm = np.sqrt(np.sum(psi ** 2))
    return psi / (norm + 1e-10)


def warp_signal_to_length(f, gamma, N_out):
    """Apply gamma (length N+1) to f (length N_in), return length N_out."""
    warped_full = np.interp(gamma, np.linspace(0, 1, len(f)), f)
    # Resample to N_out points
    return np.interp(np.linspace(0, 1, N_out), np.linspace(0, 1, len(warped_full)), warped_full)


# ============================================================
# Karcher Mean
# ============================================================

def karcher_mean(curves, N, tol=1e-3, max_iter=10, verbose=False):
    """Iterative elastic mean. curves: list of arrays length N. Returns template length N."""
    n = len(curves)
    mu = curves[0].copy()
    q_mu = compute_srvf(mu)

    for it in range(max_iter):
        aligned = []
        for f in curves:
            q_f = compute_srvf(f)
            gamma, _ = dp_align(q_mu, q_f)
            # warp_by_gamma returns length N (matches mu)
            aligned.append(warp_by_gamma(f, gamma))

        mu_new = np.mean(aligned, axis=0)
        change = np.sqrt(np.mean((mu_new - mu) ** 2))
        if verbose:
            console.print(f"    Iter {it+1}: change={change:.6f}")
        if change < tol:
            mu = mu_new
            break
        mu = mu_new
        q_mu = compute_srvf(mu)
    return mu


# ============================================================
# Quantile and features
# ============================================================

def quantile_function(signal, K):
    """K-point quantile function."""
    s = np.sort(signal)
    emp_q = (np.arange(1, len(s) + 1) - 0.5) / len(s)
    qgrid = (np.arange(1, K + 1) - 0.5) / K
    return np.interp(qgrid, emp_q, s)


def signed_sqrt(u):
    return np.sign(u) * np.sqrt(np.abs(u) + 1e-16)


def kta_alignment(Kmat, Y_oh):
    YYt = Y_oh @ Y_oh.T
    return np.sum(Kmat * YYt) / (np.sqrt(np.sum(Kmat * Kmat)) * np.sqrt(np.sum(YYt * YYt)) + 1e-16)


# ============================================================
# Synthetic warping
# ============================================================

def random_warping(N, strength=0.3, rng=None):
    """Random smooth monotone warping of N points."""
    if rng is None:
        rng = np.random.RandomState()
    n_ctrl = 8
    cx = np.linspace(0, 1, n_ctrl)
    cy = cx + strength * rng.randn(n_ctrl) * cx * (1 - cx) * 4
    cy = np.sort(cy)
    cy[0], cy[-1] = 0, 1
    cy = np.clip(cy, 0, 1)
    gamma = np.interp(np.linspace(0, 1, N), cx, cy)
    gamma = np.sort(gamma)
    return (gamma - gamma[0]) / (gamma[-1] - gamma[0] + 1e-10)


def apply_warping(X, N, strength, seed=42):
    """Apply random warping to each row of X."""
    rng = np.random.RandomState(seed)
    Xw = np.zeros_like(X)
    for i in range(len(X)):
        gamma = random_warping(N, strength, rng)
        Xw[i] = np.interp(gamma, np.linspace(0, 1, len(X[i])), X[i])
    return Xw


# ============================================================
# Pipelines
# ============================================================

def amp_only_pipeline(X_tr, y_tr, X_te, y_te, K=32):
    """Amplitude-only: quantile features -> RF."""
    N_tr, N_te = len(X_tr), len(X_te)
    Q_tr = np.array([quantile_function(X_tr[i], K) for i in range(N_tr)])
    Q_te = np.array([quantile_function(X_te[i], K) for i in range(N_te)])
    Q_h = np.mean(Q_tr[y_tr == 0], axis=0)

    Xf_tr = np.column_stack([Q_tr, Q_tr**2, np.mean(Q_tr, 1), np.std(Q_tr, 1),
                              np.sqrt(np.mean((Q_tr - Q_h)**2, 1))])
    Xf_te = np.column_stack([Q_te, Q_te**2, np.mean(Q_te, 1), np.std(Q_te, 1),
                              np.sqrt(np.mean((Q_te - Q_h)**2, 1))])

    clf = RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample",
                                  n_jobs=-1, random_state=42)
    clf.fit(Xf_tr, y_tr)
    y_pred = clf.predict(Xf_te)
    return _metrics(y_te, y_pred, Xf_tr.shape[1])


def elastic_pipeline(X_tr, y_tr, X_te, y_te, T=80, K=32):
    """Elastic amplitude+phase pipeline."""
    N_tr, N_te = len(X_tr), len(X_te)

    # Resample to T points
    Xr_tr = np.array([resample_signal(X_tr[i], T) for i in range(N_tr)])
    Xr_te = np.array([resample_signal(X_te[i], T) for i in range(N_te)])

    # Karcher mean template (use subset for speed)
    console.print("    [dim]Computing Karcher mean...[/dim]")
    t0 = time.time()
    mu = karcher_mean(Xr_tr[:min(200, N_tr)].tolist(), T, tol=1e-3, max_iter=8)
    console.print(f"    [dim]Template: {time.time()-t0:.1f}s[/dim]")

    # Per-class templates
    classes = np.unique(y_tr)
    mu_c = {}
    for c in classes:
        mask = y_tr == c
        mu_c[c] = karcher_mean(Xr_tr[mask][:min(100, np.sum(mask))].tolist(), T, tol=1e-2, max_iter=5)

    # Register all signals
    console.print("    [dim]Registering signals...[/dim]")
    t1 = time.time()

    def register_batch(Xr):
        amp_Q, phase_psi, amp_anom = [], [], []
        q_mu = compute_srvf(mu)
        Q_h = quantile_function(mu_c.get(0, mu), K)
        for f in Xr:
            q_f = compute_srvf(f)
            gamma, _ = dp_align(q_mu, q_f)
            f_tilde = warp_by_gamma(f, gamma)
            Q = quantile_function(f_tilde, K)
            psi = compute_psi(gamma)
            amp_Q.append(Q)
            phase_psi.append(psi)
            amp_anom.append(np.sqrt(np.mean((Q - Q_h) ** 2)))
        return np.array(amp_Q), np.array(phase_psi), np.array(amp_anom)

    Q_tr, psi_tr, anom_tr = register_batch(Xr_tr)
    Q_te, psi_te, anom_te = register_batch(Xr_te)
    console.print(f"    [dim]Registration: {time.time()-t1:.1f}s[/dim]")

    # Phase template: mean psi of healthy-class registrations
    psi_healthy_all = []
    q_mu = compute_srvf(mu)
    healthy_mask = y_tr == 0
    for f in Xr_tr[healthy_mask][:min(100, np.sum(healthy_mask))]:
        q_f = compute_srvf(f)
        gamma, _ = dp_align(q_mu, q_f)
        psi_healthy_all.append(compute_psi(gamma))
    psi_h = np.mean(psi_healthy_all, axis=0)

    # Build features
    def build_feats(Q, psi, anom):
        amp = np.column_stack([Q, Q**2, np.mean(Q, 1)[:, None], np.std(Q, 1)[:, None], anom[:, None]])
        ph = np.column_stack([
            np.mean(psi, 1)[:, None], np.std(psi, 1)[:, None],
            (np.max(psi, 1) - np.min(psi, 1))[:, None],
            np.sqrt(np.mean((psi - psi_h)**2, 1))[:, None],
        ])
        return np.column_stack([amp, ph])

    Xf_tr = build_feats(Q_tr, psi_tr, anom_tr)
    Xf_te = build_feats(Q_te, psi_te, anom_te)

    clf = RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample",
                                  n_jobs=-1, random_state=42)
    clf.fit(Xf_tr, y_tr)
    y_pred = clf.predict(Xf_te)
    return _metrics(y_te, y_pred, Xf_tr.shape[1])


def _metrics(y_true, y_pred, feat_dim):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "class_recalls": [float(r) for r in recall_score(y_true, y_pred, average=None)],
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "feature_dim": feat_dim,
    }


# ============================================================
# Main experiment
# ============================================================

def run_experiment(data_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    console.print("[bold cyan]SRVF Elastic Amplitude-Phase Validation on ECG5000[/bold cyan]")
    console.print("[cyan]Core claim: elastic registration recovers accuracy lost under synthetic warping[/cyan]\n")

    data = np.load(data_file)
    X_tr = data["X_train"].astype(np.float64)
    y_tr = data["y_train"].astype(np.int64)
    X_te = data["X_test"].astype(np.float64)
    y_te = data["y_test"].astype(np.int64)

    eps = 1e-8
    mu_, sig_ = np.mean(X_tr, -1, keepdims=True), np.std(X_tr, -1, keepdims=True)
    X_tr = (X_tr - mu_) / (sig_ + eps)
    mu_, sig_ = np.mean(X_te, -1, keepdims=True), np.std(X_te, -1, keepdims=True)
    X_te = (X_te - mu_) / (sig_ + eps)

    console.print(f"Data: Train={len(X_tr)}, Test={len(X_te)}, Length={X_tr.shape[1]}\n")
    results = {}

    # Experiment 1: No warping
    console.print("=" * 60)
    console.print("[bold]Experiment 1: No warping (original ECG5000)[/bold]")
    console.print("=" * 60)

    console.print("\n  [bold]Amplitude-only:[/bold]")
    t0 = time.time()
    r = amp_only_pipeline(X_tr, y_tr, X_te, y_te)
    console.print(f"  Acc={r['accuracy']:.4f}, MacroF1={r['macro_f1']:.4f} ({time.time()-t0:.1f}s)")
    results["no_warp_amp"] = r

    console.print("\n  [bold]Elastic (amplitude + phase):[/bold]")
    t0 = time.time()
    r = elastic_pipeline(X_tr, y_tr, X_te, y_te)
    console.print(f"  Acc={r['accuracy']:.4f}, MacroF1={r['macro_f1']:.4f} ({time.time()-t0:.1f}s)")
    results["no_warp_elastic"] = r

    base_f1_amp = results["no_warp_amp"]["macro_f1"]
    base_f1_el = results["no_warp_elastic"]["macro_f1"]

    # Experiments 2-4: Synthetic warping at increasing strengths
    for ws in [0.15, 0.30, 0.50]:
        console.print(f"\n{'=' * 60}")
        console.print(f"[bold]Synthetic warping (strength={ws})[/bold]")
        console.print("=" * 60)

        Xw_tr = apply_warping(X_tr, X_tr.shape[1], ws, seed=42)
        Xw_te = apply_warping(X_te, X_te.shape[1], ws, seed=123)

        # Re-standardize
        mu_, sig_ = np.mean(Xw_tr, -1, keepdims=True), np.std(Xw_tr, -1, keepdims=True)
        Xw_tr = (Xw_tr - mu_) / (sig_ + eps)
        mu_, sig_ = np.mean(Xw_te, -1, keepdims=True), np.std(Xw_te, -1, keepdims=True)
        Xw_te = (Xw_te - mu_) / (sig_ + eps)

        console.print(f"\n  [bold]Amplitude-only on warped data:[/bold]")
        t0 = time.time()
        r = amp_only_pipeline(Xw_tr, y_tr, Xw_te, y_te)
        console.print(f"  Acc={r['accuracy']:.4f}, MacroF1={r['macro_f1']:.4f} ({time.time()-t0:.1f}s)")
        results[f"warp{ws}_amp"] = r

        console.print(f"\n  [bold]Elastic on warped data:[/bold]")
        t0 = time.time()
        r = elastic_pipeline(Xw_tr, y_tr, Xw_te, y_te)
        console.print(f"  Acc={r['accuracy']:.4f}, MacroF1={r['macro_f1']:.4f} ({time.time()-t0:.1f}s)")
        results[f"warp{ws}_elastic"] = r

    # ================================================================
    # Summary
    # ================================================================
    console.print(f"\n{'=' * 80}")
    console.print("[bold cyan]VALIDATION SUMMARY[/bold cyan]")
    console.print("=" * 80)
    console.print("Core claim: elastic registration recovers accuracy lost under synthetic warping\n")

    console.print(f"{'Warping':<18} {'Method':<20} {'Accuracy':>10} {'MacroF1':>10} {'dF1':>8}")
    console.print("-" * 66)

    console.print(f"{'None':<18} {'Amp-only':<20} {results['no_warp_amp']['accuracy']:>10.4f} "
                  f"{results['no_warp_amp']['macro_f1']:>10.4f} {'base':>8}")
    console.print(f"{'':18} {'Elastic (A+P)':<20} {results['no_warp_elastic']['accuracy']:>10.4f} "
                  f"{results['no_warp_elastic']['macro_f1']:>10.4f} "
                  f"{results['no_warp_elastic']['macro_f1']-base_f1_amp:>+8.4f}")

    for ws in [0.15, 0.30, 0.50]:
        ra = results[f"warp{ws}_amp"]
        re = results[f"warp{ws}_elastic"]
        console.print(f"{'str='+str(ws):<18} {'Amp-only':<20} {ra['accuracy']:>10.4f} "
                      f"{ra['macro_f1']:>10.4f} {ra['macro_f1']-base_f1_amp:>+8.4f}")
        console.print(f"{'':18} {'Elastic (A+P)':<20} {re['accuracy']:>10.4f} "
                      f"{re['macro_f1']:>10.4f} {re['macro_f1']-base_f1_amp:>+8.4f}")

    console.print(f"\n[bold]Recovery Analysis (how much of the warp-induced drop does elastic recover?)[/bold]")
    for ws in [0.15, 0.30, 0.50]:
        amp_drop = base_f1_amp - results[f"warp{ws}_amp"]["macro_f1"]
        el_drop = base_f1_amp - results[f"warp{ws}_elastic"]["macro_f1"]
        recovery = max(0, (1 - el_drop / (amp_drop + 1e-10)) * 100) if amp_drop > 0.001 else float('nan')
        console.print(f"  Warping {ws}: amp_drop={amp_drop:+.4f}, elastic_drop={el_drop:+.4f}, "
                      f"recovery={recovery:.0f}%")

    # Baselines comparison
    console.print(f"\n[bold]Comparison with baselines (no warping):[/bold]")
    console.print(f"  MiniRocket:          Acc=0.9560, MacroF1=0.6011")
    console.print(f"  InceptionTime:       Acc=0.9610, MacroF1=0.7697")
    console.print(f"  KTA-TF-Drift v2.1:   Acc=0.9500, MacroF1=0.5891")
    console.print(f"  Amp-only (this run): Acc={results['no_warp_amp']['accuracy']:.4f}, "
                  f"MacroF1={results['no_warp_amp']['macro_f1']:.4f}")
    console.print(f"  Elastic (this run):  Acc={results['no_warp_elastic']['accuracy']:.4f}, "
                  f"MacroF1={results['no_warp_elastic']['macro_f1']:.4f}")

    # Save
    with open(os.path.join(output_dir, "elastic_validation.json"), "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n[green]Results saved to {output_dir}/elastic_validation.json[/green]")

    return results


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    run_experiment(
        os.path.join(base_dir, "data", "ecg5000_resplit.npz"),
        os.path.join(base_dir, "results", "elastic_validation"),
    )
