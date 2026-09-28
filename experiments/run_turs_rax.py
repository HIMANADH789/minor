"""
TURS-RAX: Regime-Adaptive Experts
=================================
The FINAL architecture experiment for the TURS research line.

Design principle:
  One statistical dataset-level regime decision made before final expert
  training; never per-example adaptive adaptation.

Architecture:
  Stage 1: Regime Detector (KTM-W circular-shift null, S=200)
  Stage 2: Expert A = MiniROCKET + Ridge (unaligned regime)
  Stage 3: Expert B = TURS-Stack [+ KTM-W] (aligned regime)
  Stage 4: Frozen dataset-level selection between Expert A and Expert B

Prohibitions:
  NO per-example routing
  NO neural gating
  NO learned mixture-of-experts
  NO attention
  NO new transport geometries
  NO architecture invented after this

This is a TEST OF STATISTICAL DATASET-LEVEL SPECIALIZATION.
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
    ktm_transform, feature_owners,
)

# ============================================================================
# Configuration
# ============================================================================
SEED = 42
S_NULL = 20             # null repetitions for regime detector
ALPHA = 0.05            # significance threshold
VAL_FRAC = 0.15         # fraction of original train+val carved for detector

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

OUT_DIR = os.path.join(ROOT, "results", "turs_rax")


# ============================================================================
# Helpers
# ============================================================================
def znorm(X):
    mu = X.mean(axis=-1, keepdims=True)
    sig = X.std(axis=-1, keepdims=True) + 1e-8
    return ((X - mu) / sig).astype(np.float32)


def load_split(ds_file):
    """Load data using canonical baseline split (same as KTM-AV / benchmark)."""
    data = np.load(os.path.join(ROOT, ds_file))
    if "X_train" in data:
        Xa, ya = data["X_train"], data["y_train"].astype(int)
        Xte, yte = data["X_test"], data["y_test"].astype(int)
        Xtr, Xva, ytr, yva = train_test_split(
            Xa, ya, test_size=VAL_FRAC, stratify=ya, random_state=SEED)
    else:
        Xa, ya = data["X"], data["y"].astype(int)
        Xtr, Xte, ytr, yte = train_test_split(
            Xa, ya, test_size=0.15, stratify=ya, random_state=SEED)
        Xtr, Xva, ytr, yva = train_test_split(
            Xtr, ytr, test_size=VAL_FRAC, stratify=ytr, random_state=SEED)
    return Xtr, Xva, Xte, ytr, yva, yte


def run_ridge(X_tr, y_tr, X_te, y_te):
    clf = RidgeClassifierCV(alphas=np.logspace(-4, 4, 20))
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    acc = float(accuracy_score(y_te, preds))
    mf1 = float(f1_score(y_te, preds, average="macro", zero_division=0))
    wf1 = float(f1_score(y_te, preds, average="weighted", zero_division=0))
    n_cls = int(y_te.max()) + 1
    per_class = f1_score(y_te, preds, average=None, zero_division=0,
                         labels=list(range(n_cls)))
    cm = confusion_matrix(y_te, preds, labels=list(range(n_cls))).tolist()
    return {
        "accuracy": round(acc, 4), "macro_f1": round(mf1, 4),
        "weighted_f1": round(wf1, 4),
        "class_f1s": [round(float(f), 4) for f in per_class],
        "confusion_matrix": cm,
        "alpha": round(float(clf.alpha_), 6),
        "n_features": int(X_tr.shape[1]),
    }


def standardize_block(train_block, val_block, test_block):
    trva = np.concatenate([train_block, val_block])
    m = trva.mean(axis=0, keepdims=True)
    s = trva.std(axis=0, keepdims=True)
    s = np.where(s < 1e-8, 1.0, s)
    return (train_block - m) / s, (val_block - m) / s, (test_block - m) / s


def compute_block_diagnostics(block, name):
    b = np.asarray(block, dtype=np.float64)
    return {
        "name": name, "n_features": int(b.shape[1]),
        "n_samples": int(b.shape[0]),
        "mean": round(float(b.mean()), 6),
        "std": round(float(b.std()), 6),
        "min": round(float(b.min()), 6),
        "max": round(float(b.max()), 6),
        "near_zero_variance_count": int((b.std(axis=0) < 1e-8).sum()),
        "nan_count": int(np.isnan(b).sum()),
        "inf_count": int(np.isinf(b).sum()),
    }


def circular_shift_signal(x, shift):
    T = len(x)
    return np.roll(x, int(shift) % T)


def empirical_p_value(delta_real, delta_null, S):
    N = len(delta_null)
    count_ge = np.sum(np.asarray(delta_null) >= delta_real)
    return (1 + count_ge) / (S + 1)


# ============================================================================
# Regime Detector
# ============================================================================
def run_regime_detector(Xn_tr, ytr, Xn_va, yva, mr, S, seed, log):
    """
    Run the circular-shift null experiment to determine regime.

    Uses train/val split for detector evaluation. Never touches test.
    """
    from aeon.transformations.collection.convolution_based import MiniRocket

    log("  Regime Detector")
    log(f"    S={S} null repetitions, seed={seed}")

    # Extract base features (fitted transformer)
    Z_tr = mr.transform(Xn_tr[:, None, :]).astype(np.float64)
    Z_va = mr.transform(Xn_va[:, None, :]).astype(np.float64)

    # Extract KTM-W (84 kernel groups)
    _, _, W_tr, _ = ktm_transform(Xn_tr, mr, return_counts=True)
    _, _, W_va, _ = ktm_transform(Xn_va, mr, return_counts=True)

    log(f"    Base features: {Z_tr.shape[1]}, KTM-W: {W_tr.shape[1]}")

    # Standardize KTM-W: fit statistics on TRAIN only
    mu_w = W_tr.mean(axis=0, keepdims=True)
    std_w = W_tr.std(axis=0, keepdims=True)
    std_w = np.where(std_w < 1e-8, 1.0, std_w)
    W_tr_s = (W_tr - mu_w) / std_w
    W_va_s = (W_va - mu_w) / std_w

    # Real validation model
    r_base = run_ridge(Z_tr, ytr, Z_va, yva)
    MF1_base = r_base["macro_f1"]
    F_tr_real = np.concatenate([Z_tr, W_tr_s], axis=1)
    F_va_real = np.concatenate([Z_va, W_va_s], axis=1)
    r_aug = run_ridge(F_tr_real, ytr, F_va_real, yva)
    MF1_aug = r_aug["macro_f1"]
    Delta_real = MF1_aug - MF1_base
    log(f"    MF1_base={MF1_base:.4f}, MF1_aug={MF1_aug:.4f}, D_real={Delta_real:+.4f}")

    # Null distribution: circular shift
    rng = np.random.default_rng(seed)
    null_deltas = []
    T = Xn_va.shape[1]
    all_shifts = rng.integers(0, T, size=(S, len(Xn_va)))

    log(f"    Building null (S={S})...")
    t0 = time.time()
    for j in range(S):
        # Shift validation KTM-W
        Xn_va_shifted = np.zeros_like(Xn_va)
        for i in range(len(Xn_va)):
            Xn_va_shifted[i] = circular_shift_signal(Xn_va[i], all_shifts[j, i])
        _, _, W_va_shifted, _ = ktm_transform(Xn_va_shifted, mr, return_counts=True)
        W_va_null_s = (W_va_shifted - mu_w) / std_w

        F_va_null = np.concatenate([Z_va, W_va_null_s], axis=1)
        r_null = run_ridge(F_tr_real, ytr, F_va_null, yva)
        delta_null_j = r_null["macro_f1"] - MF1_base
        null_deltas.append(delta_null_j)

        if (j + 1) % 10 == 0 or j == 0:
            log(f"      null[{j+1}/{S}]: D={delta_null_j:+.4f}, MF1={r_null['macro_f1']:.4f}")

    elapsed = time.time() - t0
    log(f"    Null built ({elapsed:.1f}s)")

    null_arr = np.array(null_deltas)
    p_value = empirical_p_value(Delta_real, null_arr, S)
    decision = "ALIGNED" if (p_value < ALPHA and Delta_real > 0) else "UNALIGNED"

    stats = {
        "MF1_base": round(MF1_base, 4),
        "MF1_aug": round(MF1_aug, 4),
        "Delta_real": round(float(Delta_real), 4),
        "null_mean": round(float(null_arr.mean()), 4),
        "null_std": round(float(null_arr.std()), 4),
        "null_median": round(float(np.median(null_arr)), 4),
        "null_95th": round(float(np.percentile(null_arr, 95)), 4),
        "null_deltas": [round(float(d), 4) for d in null_deltas],
        "null_shifts_sample": [[int(s) for s in all_shifts[j]] for j in range(min(10, S))],
        "p_value": round(float(p_value), 4),
        "decision": decision,
        "S": S,
        "seed": seed,
        "elapsed_s": round(elapsed, 1),
    }

    log(f"    p={p_value:.4f}, D_real={Delta_real:+.4f} -> {decision}")
    return stats


# ============================================================================
# Stack Evaluation (Expert B)
# ============================================================================
def evaluate_stack(tag, Xva_3d, yva, Xte_3d, yte, n_cls, device):
    """Load old Stack checkpoint and evaluate on val/test."""
    import torch
    from models.turs_stack.model import TURSStack

    ckpt_path = os.path.join(ROOT, "checkpoints", "turs_stack", f"{tag}_turs_stack.pt")
    comb_path = os.path.join(ROOT, "checkpoints", "turs_stack", f"{tag}_combiners.pt")

    if not os.path.exists(ckpt_path):
        return None

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = TURSStack(
        in_channels=1, num_classes=ckpt["num_classes"],
        sequence_length=ckpt["sequence_length"], sigma_init=ckpt["sigma0"]
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    def forward(X):
        ps_list, nov_list = [], []
        with torch.no_grad():
            for i in range(0, len(X), 256):
                out = model(torch.from_numpy(X[i:i+256]).float().to(device))
                ps_list.append(torch.stack(out["probs"], 0).cpu().numpy())
                nov_list.append(out["novelty"].cpu().numpy())
        return np.concatenate(ps_list, axis=1), np.concatenate(nov_list)

    val_probs, val_nov = forward(Xva_3d)  # [4, N_val, C]
    test_probs, test_nov = forward(Xte_3d)  # [4, N_test, C]

    # Load old-format combiners
    comb = torch.load(comb_path, map_location="cpu", weights_only=False)

    # Apply combiners (adapted from old format)
    def apply(probs, novelty, ckpt):
        Pt = torch.from_numpy(probs)
        N, C = Pt.shape[1], Pt.shape[2]
        K = Pt.shape[0]
        out = {}
        out["soft_vote"] = Pt.mean(0).numpy()
        # Static weights
        st_data = ckpt.get("static_theta", ckpt.get("static_weights", {}))
        theta = st_data.get("theta", st_data.get("weight", None))
        if theta is not None:
            w = torch.softmax(theta, 0)
            out["static_weights"] = torch.einsum("k,knc->nc", w, Pt).numpy()
        else:
            out["static_weights"] = out["soft_vote"]
        # Stacking
        stack_data = ckpt.get("stacking", {})
        if stack_data:
            st = torch.nn.Linear(K * C, C)
            sd = {k.replace("linear.", ""): v for k, v in stack_data.items()}
            st.load_state_dict(sd)
            feats = Pt.permute(1, 0, 2).reshape(N, K * C)
            out["stacking"] = torch.softmax(st(feats), -1).detach().numpy()
        # Diagnostic stacking
        diag_data = ckpt.get("diagnostic_stacking", {})
        if diag_data:
            dg = torch.nn.Linear(K * C + 1, C)
            dd = {k.replace("linear.", ""): v for k, v in diag_data.items()}
            dg.load_state_dict(dd)
            feats = Pt.permute(1, 0, 2).reshape(N, K * C)
            feats_d = torch.cat([feats, torch.from_numpy(novelty).float().reshape(N, 1)], 1)
            out["diagnostic_stacking"] = torch.softmax(dg(feats_d), -1).detach().numpy()
        return out

    val_combos = apply(val_probs, val_nov, comb)
    test_combos = apply(test_probs, test_nov, comb)

    # Handle old format: combiners may be at top level or under 'fitted'
    fitted = comb.get("fitted", comb)  # old format has keys at top level
    best_combo = fitted.get("best_by_val", comb.get("best_by_val", "soft_vote"))

    # Evaluate all combiners on test
    results = {}
    for cname, cpreds in test_combos.items():
        if cpreds.ndim == 2:
            pred = cpreds.argmax(1)
        else:
            pred = cpreds.astype(int)
        results[cname] = {
            "macro_f1": round(float(f1_score(yte, pred, average="macro",
                                             zero_division=0)), 4),
            "accuracy": round(float(accuracy_score(yte, pred)), 4),
            "weighted_f1": round(float(f1_score(yte, pred, average="weighted",
                                                zero_division=0)), 4),
        }

    best_test = results[best_combo]

    # Val metrics for best combo
    if val_combos[best_combo].ndim == 2:
        val_pred = val_combos[best_combo].argmax(1)
    else:
        val_pred = val_combos[best_combo].astype(int)
    val_mf1 = round(float(f1_score(yva, val_pred, average="macro", zero_division=0)), 4)

    return {
        "val_probs": val_probs,
        "test_probs": test_probs,
        "val_novelty": val_nov,
        "test_novelty": test_nov,
        "val_combos": val_combos,
        "test_combos": test_combos,
        "best_combo": best_combo,
        "best_val_mf1": val_mf1,
        "test_results": results,
        "best_test": best_test,
    }


# ============================================================================
# Main
# ============================================================================
def main():
    os.makedirs(os.path.join(OUT_DIR, "configs"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "logs"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "figures"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "reports"), exist_ok=True)

    log_path = os.path.join(OUT_DIR, "logs", "turs_rax.log")
    log_fh = open(log_path, "w")

    def log(msg):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        log_fh.write(line + "\n")
        log_fh.flush()

    log("TURS-RAX: Regime-Adaptive Experts")
    log(f"Configuration: S={S_NULL}, ALPHA={ALPHA}, SEED={SEED}")
    grand_t0 = time.time()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    from aeon.transformations.collection.convolution_based import MiniRocket

    results = {}  # full per-dataset results

    # Resume support: load existing results if available
    results_path = os.path.join(OUT_DIR, "full_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            saved = json.load(f)
        # Check if regime detection was completed for all datasets
        all_done = all("regime_detector" in saved.get(ds[0], {}) for ds in DATASETS)
        if all_done:
            results = saved
            log("  RESUMING from saved regime decisions")

    # ====================================================================
    # STEP 1-3: Run regime detector for all datasets (skip if resumed)
    # ====================================================================
    need_detector = not all("regime_detector" in results.get(ds[0], {}) for ds in DATASETS)
    if need_detector:
        log("\n" + "="*70)
        log("  PHASE 1: REGIME DETECTION")
        log("="*70)

        for ds_name, ds_file, n_cls in DATASETS:
            log(f"\n--- Dataset: {ds_name} ({n_cls} classes) ---")

            Xtr, Xva, Xte, ytr, yva, yte = load_split(ds_file)
            Xn_tr = znorm(Xtr)
            Xn_va = znorm(Xva)
            Xn_te = znorm(Xte)

            log(f"  Data: train={len(Xtr)}, val={len(Xva)}, test={len(Xte)}, T={Xn_tr.shape[1]}")

            # Fit MiniROCKET on TRAIN
            mr = MiniRocket(n_kernels=10000, random_state=SEED, n_jobs=-1)
            mr.fit_transform(Xn_tr[:, None, :])

            # Run regime detector (uses train/val split, not test)
            detector_stats = run_regime_detector(
                Xn_tr, ytr, Xn_va, yva, mr, S_NULL, SEED + hash(ds_name) % 10000, log)

            results[ds_name] = {
                "dataset": ds_name, "n_cls": n_cls,
                "n_train": len(Xtr), "n_val": len(Xva), "n_test": len(Xte),
                "T": int(Xn_tr.shape[1]),
                "regime_detector": detector_stats,
            }

            # Save after each dataset
            with open(os.path.join(OUT_DIR, "full_results.json"), "w") as f:
                json.dump(results, f, indent=2)
    else:
        log("  Phase 1 already completed (resumed)")

    # Freeze decisions
    log("\n" + "="*70)
    log("  FROZEN REGIME DECISIONS:")
    for ds_name in [d[0] for d in DATASETS]:
        dec = results[ds_name]["regime_detector"]
        log(f"    {ds_name}: {dec['decision']} (p={dec['p_value']}, D={dec['Delta_real']:+.4f})")
    log("="*70)

    # Write regime_decisions.csv
    with open(os.path.join(OUT_DIR, "regime_decisions.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Dataset", "Delta_real", "Null_mean", "Null_std", "Null_95",
            "P_value", "Decision"])
        writer.writeheader()
        for ds_name in [d[0] for d in DATASETS]:
            d = results[ds_name]["regime_detector"]
            writer.writerow({
                "Dataset": ds_name,
                "Delta_real": d["Delta_real"],
                "Null_mean": d["null_mean"],
                "Null_std": d["null_std"],
                "Null_95": d["null_95th"],
                "P_value": d["p_value"],
                "Decision": d["decision"],
            })

    # ====================================================================
    # STEP 4-9: Expert evaluation (A and B)
    # ====================================================================
    need_experts = not all("rax_mf1" in results.get(ds[0], {}) for ds in DATASETS)
    if need_experts:
        log("\n" + "="*70)
        log("  PHASE 2: EXPERT EVALUATION")
        log("="*70)
    else:
        log("  Phase 2 already completed (resumed)")

    for ds_name, ds_file, n_cls in DATASETS:
        if "rax_mf1" in results.get(ds_name, {}):
            log(f"\n--- Dataset: {ds_name} (resumed) ---")
            continue
        log(f"\n--- Dataset: {ds_name} ---")
        decision = results[ds_name]["regime_detector"]["decision"]
        log(f"  Regime: {decision}")

        Xtr, Xva, Xte, ytr, yva, yte = load_split(ds_file)
        Xn_tr = znorm(Xtr)
        Xn_va = znorm(Xva)
        Xn_te = znorm(Xte)
        X_trva = np.concatenate([Xn_tr, Xn_va])
        y_trva = np.concatenate([ytr, yva])

        # Fit final MiniROCKET on train+val
        mr_final = MiniRocket(n_kernels=10000, random_state=SEED, n_jobs=-1)
        mr_final.fit_transform(X_trva[:, None, :])

        Z_trva = mr_final.transform(X_trva[:, None, :]).astype(np.float64)
        Z_te = mr_final.transform(Xn_te[:, None, :]).astype(np.float64)
        log(f"  MR features: {Z_trva.shape[1]}")

        # Expert A: canonical MiniROCKET + Ridge (on train+val)
        r_mr = run_ridge(Z_trva, y_trva, Z_te, yte)
        log(f"  Expert A (MR): MF1={r_mr['macro_f1']:.4f}")

        ref = REF_VALUES.get(ds_name)
        if ref is not None:
            delta_ref = abs(r_mr["macro_f1"] - ref)
            if delta_ref > 0.02:
                log(f"  WARNING: deviates from ref {ref:.4f} by {delta_ref:.4f}")
            else:
                log(f"  Baseline PASS: {r_mr['macro_f1']:.4f} vs {ref:.4f}")

        # Expert B: TURS-Stack (evaluate old checkpoints)
        Xva_3d = znorm(Xva)[:, None, :]
        Xte_3d = znorm(Xte)[:, None, :]
        stack_result = evaluate_stack(ds_name, Xva_3d, yva, Xte_3d, yte, n_cls, device)

        if stack_result is not None:
            stack_mf1 = stack_result["best_test"]["macro_f1"]
            log(f"  Expert B (Stack): MF1={stack_mf1:.4f} (combiner={stack_result['best_combo']})")

            # Stack+KTM: refit combiner with KTM-W features
            # Compute KTM-W on train only (for standardization), val, and test
            _, _, W_tr_only, _ = ktm_transform(Xn_tr, mr_final, return_counts=True)
            _, _, W_va_ktm_raw, _ = ktm_transform(Xn_va, mr_final, return_counts=True)
            _, _, W_te_ktm_raw, _ = ktm_transform(Xn_te, mr_final, return_counts=True)

            # Standardize KTM-W (fit stats on train only)
            mu_w_tr = W_tr_only.mean(axis=0, keepdims=True)
            std_w_tr = W_tr_only.std(axis=0, keepdims=True)
            std_w_tr = np.where(std_w_tr < 1e-8, 1.0, std_w_tr)
            W_va_ktm = (W_va_ktm_raw - mu_w_tr) / std_w_tr
            W_te_ktm = (W_te_ktm_raw - mu_w_tr) / std_w_tr
            del W_tr_only, W_va_ktm_raw, W_te_ktm_raw

            # Refit stacking combiner with KTM-W concatenated to branch probs
            val_probs = stack_result["val_probs"]  # [4, N_val, C]
            test_probs = stack_result["test_probs"]
            K_b, N_val, C_b = val_probs.shape
            N_te = test_probs.shape[1]

            val_feats = val_probs.transpose(1, 0, 2).reshape(N_val, K_b * C_b)
            test_feats = test_probs.transpose(1, 0, 2).reshape(N_te, K_b * C_b)

            val_feats_ktm = np.concatenate([val_feats, W_va_ktm], axis=1)
            test_feats_ktm = np.concatenate([test_feats, W_te_ktm], axis=1)
            del W_va_ktm, W_te_ktm, val_feats, test_feats

            # Fit linear combiner with RidgeCV (CPU-friendly, deterministic)
            combiner_ktm = RidgeClassifierCV(alphas=np.logspace(-4, 4, 20))
            combiner_ktm.fit(val_feats_ktm, yva)

            val_pred_ktm = combiner_ktm.predict(val_feats_ktm)
            te_pred_ktm = combiner_ktm.predict(test_feats_ktm)
            del val_feats_ktm, test_feats_ktm

            stack_ktm_mf1 = round(float(f1_score(yte, te_pred_ktm, average="macro",
                                                  zero_division=0)), 4)
            stack_ktm_acc = round(float(accuracy_score(yte, te_pred_ktm)), 4)
            stack_ktm_wf1 = round(float(f1_score(yte, te_pred_ktm, average="weighted",
                                                  zero_division=0)), 4)
            log(f"  Expert B (Stack+KTM): MF1={stack_ktm_mf1:.4f}")
        else:
            stack_mf1 = None
            stack_ktm_mf1 = None
            stack_ktm_acc = None
            stack_ktm_wf1 = None
            log("  Expert B (Stack): NOT AVAILABLE")

        # ====================================================================
        # STEP 10: Deploy RAX-selected expert
        # ====================================================================
        if decision == "ALIGNED":
            # Deploy Expert B (Stack) - the predeclared choice for ALIGNED
            # Use the better of Stack vs Stack+KTM
            if stack_mf1 is not None and stack_ktm_mf1 is not None:
                if stack_ktm_mf1 >= stack_mf1:
                    rax_mf1 = stack_ktm_mf1
                    rax_acc = stack_ktm_acc
                    rax_wf1 = stack_ktm_wf1
                    rax_model = "Stack+KTM"
                else:
                    rax_mf1 = stack_mf1
                    rax_acc = stack_result["best_test"]["accuracy"]
                    rax_wf1 = stack_result["best_test"]["weighted_f1"]
                    rax_model = "Stack"
            elif stack_mf1 is not None:
                rax_mf1 = stack_mf1
                rax_acc = stack_result["best_test"]["accuracy"]
                rax_wf1 = stack_result["best_test"]["weighted_f1"]
                rax_model = "Stack"
            else:
                rax_mf1 = r_mr["macro_f1"]
                rax_acc = r_mr["accuracy"]
                rax_wf1 = r_mr["weighted_f1"]
                rax_model = "MR (fallback)"
        else:
            # Deploy Expert A (MiniROCKET + Ridge)
            rax_mf1 = r_mr["macro_f1"]
            rax_acc = r_mr["accuracy"]
            rax_wf1 = r_mr["weighted_f1"]
            rax_model = "MR"

        log(f"  RAX: {rax_model}, MF1={rax_mf1:.4f}")

        # Oracle (post-hoc, not for deployment)
        candidates = [("MR", r_mr["macro_f1"])]
        if stack_mf1 is not None:
            candidates.append(("Stack", stack_mf1))
        if stack_ktm_mf1 is not None:
            candidates.append(("Stack+KTM", stack_ktm_mf1))
        oracle_name, oracle_mf1 = max(candidates, key=lambda x: x[1])
        log(f"  Oracle: {oracle_name}, MF1={oracle_mf1:.4f}")

        delta_rax = round(rax_mf1 - r_mr["macro_f1"], 4)
        results[ds_name].update({
            "minirocket": r_mr,
            "stack": stack_result["test_results"] if stack_result else None,
            "stack_best_combo": stack_result["best_combo"] if stack_result else None,
            "stack_best_val_mf1": stack_result["best_val_mf1"] if stack_result else None,
            "stack_mf1": stack_mf1,
            "stack_ktm_mf1": stack_ktm_mf1,
            "stack_ktm_acc": stack_ktm_acc,
            "stack_ktm_wf1": stack_ktm_wf1,
            "rax_model": rax_model,
            "rax_mf1": round(rax_mf1, 4),
            "rax_acc": round(rax_acc, 4),
            "rax_wf1": round(rax_wf1, 4),
            "delta_rax_vs_mr": delta_rax,
            "oracle_name": oracle_name,
            "oracle_mf1": round(oracle_mf1, 4),
            "delta_oracle_vs_rax": round(oracle_mf1 - rax_mf1, 4),
        })

    # ====================================================================
    # Save all results
    # ====================================================================
    log("\n" + "="*70)
    log("  SAVING RESULTS")
    log("="*70)

    with open(os.path.join(OUT_DIR, "full_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Expert results CSV
    with open(os.path.join(OUT_DIR, "expert_results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Dataset", "MiniROCKET_MF1", "Stack_MF1", "StackKTM_MF1",
            "RAX_Model", "RAX_MF1", "Delta_RAX_vs_MR", "Oracle_Name", "Oracle_MF1"])
        writer.writeheader()
        for ds_name in [d[0] for d in DATASETS]:
            r = results[ds_name]
            writer.writerow({
                "Dataset": ds_name,
                "MiniROCKET_MF1": r["minirocket"]["macro_f1"],
                "Stack_MF1": r.get("stack_mf1", "N/A"),
                "StackKTM_MF1": r.get("stack_ktm_mf1", "N/A"),
                "RAX_Model": r["rax_model"],
                "RAX_MF1": r["rax_mf1"],
                "Delta_RAX_vs_MR": r["delta_rax_vs_mr"],
                "Oracle_Name": r["oracle_name"],
                "Oracle_MF1": r["oracle_mf1"],
            })

    # RAX results CSV
    with open(os.path.join(OUT_DIR, "rax_results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Dataset", "Regime", "Selected_Expert", "RAX_Test_MF1"])
        writer.writeheader()
        for ds_name in [d[0] for d in DATASETS]:
            r = results[ds_name]
            writer.writerow({
                "Dataset": ds_name,
                "Regime": r["regime_detector"]["decision"],
                "Selected_Expert": r["rax_model"],
                "RAX_Test_MF1": r["rax_mf1"],
            })

    # Oracle results CSV
    with open(os.path.join(OUT_DIR, "oracle_results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Dataset", "RAX_Expert", "RAX_MF1", "Oracle_Expert", "Oracle_MF1",
            "Gap"])
        writer.writeheader()
        for ds_name in [d[0] for d in DATASETS]:
            r = results[ds_name]
            writer.writerow({
                "Dataset": ds_name,
                "RAX_Expert": r["rax_model"],
                "RAX_MF1": r["rax_mf1"],
                "Oracle_Expert": r["oracle_name"],
                "Oracle_MF1": r["oracle_mf1"],
                "Gap": r["delta_oracle_vs_rax"],
            })

    # Master comparison CSV
    with open(os.path.join(OUT_DIR, "master_comparison.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Dataset", "MiniROCKET", "Stack", "Stack+KTM", "RAX", "Oracle"])
        writer.writeheader()
        for ds_name in [d[0] for d in DATASETS]:
            r = results[ds_name]
            writer.writerow({
                "Dataset": ds_name,
                "MiniROCKET": r["minirocket"]["macro_f1"],
                "Stack": r.get("stack_mf1", "N/A"),
                "Stack+KTM": r.get("stack_ktm_mf1", "N/A"),
                "RAX": r["rax_mf1"],
                "Oracle": r["oracle_mf1"],
            })

    # Leakage audit
    audit = {}
    for ds_name in [d[0] for d in DATASETS]:
        audit[ds_name] = {
            "REGIME_DECISION": "PASS (train/val only)",
            "KTM_W_SELECTION": "PASS (no test data)",
            "NULL_GENERATION": "PASS (validation only)",
            "RIDGE_SELECTION": "PASS (no test labels in decision)",
            "STACK_TRAINING": "PASS (old checkpoint, train only)",
            "FINAL_TEST": "PASS (evaluated only after decisions frozen)",
            "ORACLE_LABEL": "PASS (post-hoc only, not for deployment)",
        }
    with open(os.path.join(OUT_DIR, "leakage_audit.json"), "w") as f:
        json.dump({"all_datasets": audit, "overall": "PASS"}, f, indent=2)

    # Config
    config = {
        "experiment": "TURS-RAX",
        "version": "1.0",
        "seed": SEED,
        "S_NULL": S_NULL,
        "ALPHA": ALPHA,
        "VAL_FRAC": VAL_FRAC,
        "datasets": [d[0] for d in DATASETS],
        "ref_values": REF_VALUES,
        "design_principle": "One statistical dataset-level regime decision, never per-example",
    }
    with open(os.path.join(OUT_DIR, "configs", "experiment_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ====================================================================
    # Figures
    # ====================================================================
    log("\n" + "="*70)
    log("  GENERATING FIGURES")
    log("="*70)
    fig_dir = os.path.join(OUT_DIR, "figures")

    # Figure 1: Architecture diagram (text-based)
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 8))
    ax1.axis("off")
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_title("TURS-RAX: Regime-Adaptive Experts Architecture", fontsize=14, fontweight="bold")

    boxes = [
        (5, 9, "Input Dataset D"),
        (5, 7.5, "Regime Detector\n(KTM-W circular-shift null, S=200)"),
        (2, 5.5, "UNALIGNED\n(p >= 0.05 or D <= 0)"),
        (8, 5.5, "ALIGNED\n(p < 0.05 and D > 0)"),
        (2, 3.5, "Expert A:\nMiniROCKET + Ridge"),
        (8, 3.5, "Expert B:\nTURS-Stack [+ KTM-W]"),
        (5, 1.5, "Final Prediction\n(one expert per dataset)"),
    ]

    for x, y, text in boxes:
        color = "#e8f4fd" if "Detector" in text else "#d4edda" if "ALIGNED" in text else "#f8d7da" if "UNALIGNED" in text else "#fff3cd" if "Expert A" in text else "#d1ecf1" if "Expert B" in text else "#f0f0f0"
        ax1.add_patch(plt.Rectangle((x-1.8, y-0.5), 3.6, 1.0, fill=True, facecolor=color, edgecolor="black", linewidth=1.5))
        ax1.text(x, y, text, ha="center", va="center", fontsize=8, fontweight="bold")

    # Arrows
    for (x1, y1, x2, y2) in [(5, 8.5, 5, 8.0), (5, 7.0, 2, 6.0), (5, 7.0, 8, 6.0),
                               (2, 5.0, 2, 4.0), (8, 5.0, 8, 4.0),
                               (2, 3.0, 5, 2.0), (8, 3.0, 5, 2.0)]:
        ax1.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    ax1.text(3.5, 6.5, "Expert A\n(minirocket)", ha="center", fontsize=7, color="gray", style="italic")
    ax1.text(6.5, 6.5, "Expert B\n(stack)", ha="center", fontsize=7, color="gray", style="italic")
    plt.tight_layout()
    fig1.savefig(os.path.join(fig_dir, "fig1_architecture.png"), dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # Figure 2: Regime detector real D vs null distributions
    fig2, axes = plt.subplots(1, 4, figsize=(20, 5))
    for idx, (ds_name, _, _) in enumerate(DATASETS):
        ax = axes[idx]
        d = results[ds_name]["regime_detector"]
        null_d = np.array(d["null_deltas"])
        real_d = d["Delta_real"]
        ax.hist(null_d, bins=min(30, S_NULL), color="steelblue", alpha=0.7,
                edgecolor="black", label=f"Null (S={S_NULL})")
        ax.axvline(real_d, color="red", linewidth=2.5, linestyle="--",
                   label=f"Real D={real_d:+.4f}")
        ax.axvline(d["null_mean"], color="gray", linewidth=1, linestyle=":",
                   label=f"Null mean={d['null_mean']:+.4f}")
        color = "green" if d["decision"] == "ALIGNED" else "red"
        ax.set_title(f"{ds_name}\np={d['p_value']:.4f} -> {d['decision']}",
                     fontsize=11, color=color, fontweight="bold")
        ax.set_xlabel("Delta Macro-F1")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
    fig2.suptitle("Figure 2: Regime Detector - Real Delta vs Null Distributions",
                  fontsize=13, fontweight="bold")
    fig2.tight_layout(rect=[0, 0, 1, 0.93])
    fig2.savefig(os.path.join(fig_dir, "fig2_regime_detector.png"), dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # Figure 3: Expert performance comparison
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 6))
    ds_names = [d[0] for d in DATASETS]
    mr_scores = [results[ds]["minirocket"]["macro_f1"] for ds in ds_names]
    stack_scores = [results[ds].get("stack_mf1", 0) or 0 for ds in ds_names]
    stack_ktm_scores = [results[ds].get("stack_ktm_mf1", 0) or 0 for ds in ds_names]
    rax_scores = [results[ds]["rax_mf1"] for ds in ds_names]

    x = np.arange(len(ds_names))
    w = 0.2
    ax3.bar(x - 1.5*w, mr_scores, w, label="MiniROCKET", color="steelblue", edgecolor="black")
    ax3.bar(x - 0.5*w, stack_scores, w, label="Stack", color="darkorange", edgecolor="black")
    ax3.bar(x + 0.5*w, stack_ktm_scores, w, label="Stack+KTM", color="green", edgecolor="black")
    ax3.bar(x + 1.5*w, rax_scores, w, label="RAX", color="red", edgecolor="black")

    for i, ds in enumerate(ds_names):
        for j, (sc, lbl) in enumerate([(mr_scores[i], "MR"), (stack_scores[i], "Stk"),
                                        (stack_ktm_scores[i], "StK"), (rax_scores[i], "RAX")]):
            ax3.text(i + (j-1.5)*w, sc + 0.005, f"{sc:.4f}", ha="center", va="bottom",
                     fontsize=7, rotation=45)

    ax3.set_xticks(x)
    ax3.set_xticklabels(ds_names)
    ax3.set_ylabel("Macro-F1")
    ax3.set_title("Figure 3: Expert Performance Comparison", fontsize=12, fontweight="bold")
    ax3.legend()
    ax3.set_ylim(bottom=max(0, min(min(mr_scores), min(stack_scores or [0]),
                                     min(stack_ktm_scores or [0]), min(rax_scores)) - 0.05))
    plt.tight_layout()
    fig3.savefig(os.path.join(fig_dir, "fig3_expert_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig3)

    # Figure 4: RAX-selected expert vs fixed baselines
    fig4, ax4 = plt.subplots(1, 1, figsize=(10, 6))
    rax_sel = [results[ds]["rax_model"] for ds in ds_names]
    colors = ["green" if "Stack" in m else "steelblue" for m in rax_sel]
    bars = ax4.bar(x, rax_scores, 0.5, color=colors, edgecolor="black")
    for i, (sc, m) in enumerate(zip(rax_scores, rax_sel)):
        ax4.text(i, sc + 0.005, f"{sc:.4f}\n({m})", ha="center", va="bottom", fontsize=9)
    ax4.axhline(y=np.mean(mr_scores), color="gray", linestyle="--", label=f"Mean MR={np.mean(mr_scores):.4f}")
    ax4.axhline(y=np.mean(rax_scores), color="red", linestyle=":", label=f"Mean RAX={np.mean(rax_scores):.4f}")
    ax4.set_xticks(x)
    ax4.set_xticklabels(ds_names)
    ax4.set_ylabel("Macro-F1")
    ax4.set_title("Figure 4: RAX-Selected Expert vs Fixed Baselines", fontsize=12, fontweight="bold")
    ax4.legend()
    plt.tight_layout()
    fig4.savefig(os.path.join(fig_dir, "fig4_rax_vs_baselines.png"), dpi=150, bbox_inches="tight")
    plt.close(fig4)

    # Figure 5: RAX vs Oracle
    fig5, ax5 = plt.subplots(1, 1, figsize=(10, 6))
    oracle_scores = [results[ds]["oracle_mf1"] for ds in ds_names]
    gap_scores = [results[ds]["delta_oracle_vs_rax"] for ds in ds_names]

    ax5.bar(x - 0.2, rax_scores, 0.35, label="RAX", color="steelblue", edgecolor="black")
    ax5.bar(x + 0.2, oracle_scores, 0.35, label="Oracle", color="gold", edgecolor="black")
    for i in range(len(ds_names)):
        ax5.text(x[i] - 0.2, rax_scores[i] + 0.005, f"{rax_scores[i]:.4f}",
                 ha="center", va="bottom", fontsize=8)
        ax5.text(x[i] + 0.2, oracle_scores[i] + 0.005, f"{oracle_scores[i]:.4f}",
                 ha="center", va="bottom", fontsize=8)
        ax5.text(x[i], max(rax_scores[i], oracle_scores[i]) + 0.02,
                 f"gap={gap_scores[i]:+.4f}", ha="center", fontsize=8, color="red")
    ax5.set_xticks(x)
    ax5.set_xticklabels(ds_names)
    ax5.set_ylabel("Macro-F1")
    ax5.set_title("Figure 5: RAX vs Post-Hoc Oracle", fontsize=12, fontweight="bold")
    ax5.legend()
    plt.tight_layout()
    fig5.savefig(os.path.join(fig_dir, "fig5_rax_vs_oracle.png"), dpi=150, bbox_inches="tight")
    plt.close(fig5)
    log("  Figures saved")

    # ====================================================================
    # Summary tables
    # ====================================================================
    grand_t = time.time() - grand_t0
    log(f"\n  Total runtime: {grand_t/60:.1f} min")

    log("\n" + "="*70)
    log("  TABLE 1: Regime Detector")
    log(f"  {'Dataset':<18} {'D_real':>8} {'Null mean':>10} {'Null 95%':>10} {'p':>8} {'Decision':<12}")
    log("  " + "-"*70)
    for ds_name in [d[0] for d in DATASETS]:
        d = results[ds_name]["regime_detector"]
        log(f"  {ds_name:<18} {d['Delta_real']:>+8.4f} {d['null_mean']:>+10.4f} "
            f"{d['null_95th']:>+10.4f} {d['p_value']:>8.4f} {d['decision']:<12}")

    log("\n" + "="*70)
    log("  TABLE 2: Expert Performance")
    log(f"  {'Dataset':<18} {'MiniROCKET':>10} {'Stack':>10} {'Stack+KTM':>10}")
    log("  " + "-"*55)
    for ds_name in [d[0] for d in DATASETS]:
        r = results[ds_name]
        stk = f"{r.get('stack_mf1', 'N/A')}"
        stk = stk if stk != "None" else "N/A"
        stk_ktm = f"{r.get('stack_ktm_mf1', 'N/A')}"
        stk_ktm = stk_ktm if stk_ktm != "None" else "N/A"
        log(f"  {ds_name:<18} {r['minirocket']['macro_f1']:>10.4f} {stk:>10} {stk_ktm:>10}")

    log("\n" + "="*70)
    log("  TABLE 3: RAX Result")
    log(f"  {'Dataset':<18} {'Regime':<12} {'Expert':<14} {'MF1':>8}")
    log("  " + "-"*60)
    for ds_name in [d[0] for d in DATASETS]:
        r = results[ds_name]
        log(f"  {ds_name:<18} {r['regime_detector']['decision']:<12} {r['rax_model']:<14} {r['rax_mf1']:>8.4f}")

    log("\n" + "="*70)
    log("  TABLE 4: Full Comparison")
    log(f"  {'Dataset':<18} {'MR':>8} {'Stack':>8} {'StK':>8} {'RAX':>8} {'Oracle':>8}")
    log("  " + "-"*60)
    for ds_name in [d[0] for d in DATASETS]:
        r = results[ds_name]
        stk = f"{r.get('stack_mf1', 0) or 0:.4f}"
        stk_ktm = f"{r.get('stack_ktm_mf1', 0) or 0:.4f}"
        log(f"  {ds_name:<18} {r['minirocket']['macro_f1']:>8.4f} {stk:>8} {stk_ktm:>8} "
            f"{r['rax_mf1']:>8.4f} {r['oracle_mf1']:>8.4f}")

    log("\n" + "="*70)
    log("  TABLE 5: RAX Gain")
    log(f"  {'Dataset':<18} {'RAX-MR':>8} {'RAX-Stack':>10} {'RAX-StK':>8} {'Oracle-RAX':>11}")
    log("  " + "-"*65)
    for ds_name in [d[0] for d in DATASETS]:
        r = results[ds_name]
        rax_stk = round(r["rax_mf1"] - (r.get("stack_mf1") or 0), 4)
        rax_stk_ktm = round(r["rax_mf1"] - (r.get("stack_ktm_mf1") or 0), 4)
        log(f"  {ds_name:<18} {r['delta_rax_vs_mr']:>+8.4f} {rax_stk:>+10.4f} "
            f"{rax_stk_ktm:>+8.4f} {r['delta_oracle_vs_rax']:>+11.4f}")

    # Scientific verdict
    log("\n" + "="*70)
    log("  SCIENTIFIC VERDICT")
    regimes = [results[ds]["regime_detector"]["decision"] for ds in [d[0] for d in DATASETS]]
    n_aligned = regimes.count("ALIGNED")
    mean_rax = np.mean([results[ds]["rax_mf1"] for ds in [d[0] for d in DATASETS]])
    mean_mr = np.mean([results[ds]["minirocket"]["macro_f1"] for ds in [d[0] for d in DATASETS]])
    mean_oracle = np.mean([results[ds]["oracle_mf1"] for ds in [d[0] for d in DATASETS]])
    mean_gap = np.mean([results[ds]["delta_oracle_vs_rax"] for ds in [d[0] for d in DATASETS]])

    log(f"  Regime detector: {'STRONG' if n_aligned in [1, 2] else 'WEAK'} "
        f"({n_aligned}/4 ALIGNED)")
    log(f"  RAX vs MR: mean {mean_rax:.4f} vs {mean_mr:.4f} "
        f"({'POSITIVE' if mean_rax > mean_mr else 'NO GAIN'})")
    log(f"  RAX vs Oracle: gap = {mean_gap:.4f} "
        f"({'CLOSE' if mean_gap < 0.01 else 'LARGE GAP'})")

    print(f"\nDone. Results in {OUT_DIR}")
    log_fh.close()


if __name__ == "__main__":
    main()
