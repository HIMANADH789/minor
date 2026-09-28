"""
3-Seed Confirmation: DRTN-Conditioned MiniROCKET on Haptics
===========================================================

Runs seeds 42, 43, 44 with M0/M1/M2/M3 variants each.

Phase A: Train DRTN R5 for seeds 43, 44 (seed 42 already exists).
Phase B: Run M0-M3 for all three seeds.

Usage:
    python -m experiments.drtn_conditioned_minirocket_haptics_3seed.runner
    python -m experiments.drtn_conditioned_minirocket_haptics_3seed.runner --seeds 42 43 44
    python -m experiments.drtn_conditioned_minirocket_haptics_3seed.runner --skip-drtn-train
    python -m experiments.drtn_conditioned_minirocket_haptics_3seed.runner --smoke
"""
import argparse
import copy
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from experiments.external_stack_generalization.data import load_dataset
from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (
    compute_raw_activations,
    ppv_from_activations,
    heterogeneity_features,
    independent_heterogeneity_recompute,
)  # noqa: E402
from models.drtn.model import build_model  # noqa: E402

# ---- Constants ----
DATASET = "Haptics"
BASE_DIR = os.path.join(ROOT, "results", "drtn_conditioned_minirocket_haptics_3seed")
ALPHAS = np.logspace(-4, 4, 20)
N_FEATURES = 9996
N_GLOBAL = N_FEATURES // 2  # 4998
N_HETEROGENEITY = N_FEATURES - N_GLOBAL  # 4998

# ---- DRTN training config (must match existing protocol) ----
D_MODEL = 64
K_CODES = 8
TAU = 0.5
EMA_DECAY = 0.99
BETA_COMMIT = 0.25
LAM_DIV = 0.01
DEAD_THRESHOLD = 1e-3
REVIVAL_PATIENCE = 100
BATCH_SIZE = 16
LR = 1e-3
WD = 1e-4
MAX_EPOCHS = 60
PATIENCE = 10


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


# ====================================================================
# PHASE A: DRTN Training
# ====================================================================

def train_drtn_for_seed(seed, data, device, smoke=False):
    """Train DRTN R5 for a given seed. Returns checkpoint dict and result."""
    max_ep = 8 if smoke else MAX_EPOCHS
    patience = 3 if smoke else PATIENCE

    set_seed(seed)

    zn = lambda X: ((X - X.mean(-1, keepdims=True)) /
                    (X.std(-1, keepdims=True) + 1e-8)).astype(np.float32)

    mk = lambda X, y, sh: DataLoader(
        TensorDataset(torch.from_numpy(zn(X))[:, None, :],
                      torch.from_numpy(y)),
        batch_size=BATCH_SIZE if not smoke else 8,
        shuffle=sh, num_workers=0,
        generator=torch.Generator().manual_seed(seed) if sh else None)

    tr_dl = mk(data["Xtr"], data["ytr"], True)
    va_dl = mk(data["Xva"], data["yva"], False)
    te_dl = mk(data["Xte"], data["yte"], False)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(
        "R5", c_in=1, n_classes=data["n_classes"], d_model=D_MODEL,
        n_codes=K_CODES, tau=TAU, ema_decay=EMA_DECAY, beta=BETA_COMMIT,
        lam_div=LAM_DIV, dead_threshold=DEAD_THRESHOLD,
        revival_patience=REVIVAL_PATIENCE,
    ).to(device)

    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=LR, weight_decay=WD)
    steps_per_epoch = len(tr_dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, steps_per_epoch=steps_per_epoch, epochs=max_ep)

    best_val, best_state, best_ep, no_imp = -1.0, None, 0, 0
    step = 0
    t0 = time.time()

    for ep in range(max_ep):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            z = model.encoder(xb)
            q_st, assign, commit_raw = model.vq.quantize(z)
            h, _ = model.pool(model.trajectory(q_st))
            logits = model.classifier(h)
            ce = F.cross_entropy(logits, yb)
            commit = BETA_COMMIT * commit_raw
            div = model.diversity_loss_from_assign(assign)
            loss = ce + commit + LAM_DIV * div
            model.vq.ema_step(
                z.detach().reshape(-1, z.shape[-1]),
                assign.reshape(-1), step=step)
            step += 1
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

        # Validation
        model.eval()
        preds, tgts = [], []
        with torch.no_grad():
            for xb, yb in va_dl:
                xb = xb.to(device)
                logits, _ = model.forward_with_assign(xb)
                preds.append(logits.argmax(-1).cpu().numpy())
                tgts.append(yb.numpy())
        vp = np.concatenate(preds)
        vt = np.concatenate(tgts)
        val_mf1 = float(f1_score(vt, vp, average="macro", zero_division=0))

        marker = ""
        if val_mf1 > best_val:
            best_val, best_ep, no_imp = val_mf1, ep + 1, 0
            best_state = copy.deepcopy(model.state_dict())
            marker = " *"
        else:
            no_imp += 1

        if (ep + 1) % 5 == 0 or marker:
            print(f"    [seed {seed}] ep{ep+1}/{max_ep} valMF1={val_mf1:.4f} "
                  f"(best {best_val:.4f}@{best_ep}) [{time.time()-t0:.0f}s]{marker}",
                  flush=True)

        if no_imp >= patience:
            print(f"    [seed {seed}] early stop at ep{ep+1}", flush=True)
            break

    train_time = time.time() - t0
    model.load_state_dict(best_state)

    # Final test evaluation
    preds, tgts = [], []
    with torch.no_grad():
        for xb, yb in te_dl:
            xb = xb.to(device)
            logits, _ = model.forward_with_assign(xb)
            preds.append(logits.argmax(-1).cpu().numpy())
            tgts.append(yb.numpy())
    tp = np.concatenate(preds)
    tt = np.concatenate(tgts)
    test_mf1 = float(f1_score(tt, tp, average="macro", zero_division=0))

    ckpt = {
        "rung": "R5", "config": {
            "dataset": DATASET, "seed": seed, "d_model": D_MODEL,
            "n_codes": K_CODES, "tau": TAU, "ema_decay": EMA_DECAY,
            "beta_commit": BETA_COMMIT, "lam_div": LAM_DIV,
            "dead_threshold": DEAD_THRESHOLD, "revival_patience": REVIVAL_PATIENCE,
            "batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WD,
            "max_epochs": max_ep, "patience": patience,
            "trajectory": {"layers": 2, "heads": 4, "ffn": 128, "dropout": 0.1},
        }, "epoch": best_ep, "best_val_mf1": best_val, "seed": seed,
        "dataset": DATASET, "model_state": model.state_dict(),
    }

    result = {
        "seed": seed, "best_val_mf1": round(best_val, 4),
        "best_epoch": best_ep, "test_mf1": round(test_mf1, 4),
        "train_time_s": round(train_time, 1),
    }
    return ckpt, result


def load_drtn_checkpoint(seed, device):
    """Load a pre-trained DRTN R5 checkpoint for a given seed."""
    rdir = os.path.join(ROOT, "results", f"drtn_haptics_seed{seed}", "R5")
    if not os.path.exists(os.path.join(rdir, "checkpoint.pt")):
        return None, None
    with open(os.path.join(rdir, "result.json")) as f:
        official = json.load(f)
    ck = torch.load(os.path.join(rdir, "checkpoint.pt"),
                    map_location=device, weights_only=False)
    cfg = ck["config"]
    model = build_model("R5", c_in=1, n_classes=5, d_model=cfg["d_model"],
                        n_codes=cfg["n_codes"], tau=cfg["tau"],
                        ema_decay=cfg["ema_decay"], beta=cfg["beta_commit"],
                        lam_div=cfg["lam_div"], dead_threshold=cfg["dead_threshold"],
                        revival_patience=cfg["revival_patience"],
                        traj_layers=cfg["trajectory"]["layers"],
                        traj_heads=cfg["trajectory"]["heads"],
                        traj_ffn=cfg["trajectory"]["ffn"],
                        traj_dropout=cfg["trajectory"]["dropout"])
    model.load_state_dict(ck["model_state"])
    model.to(device).eval()
    return model, official


# ====================================================================
# MiniROCKET Feature Computation
# ====================================================================


def compute_regime_heterogeneity(act_het, valid_het, regimes, K=8,
                                 min_occupancy=0.01):
    """Compute regime-conditioned activation heterogeneity features.

    CORRECTED (Stage A audit, DOC-10 item 9): the heterogeneity statistic is
    computed over each feature's aeon VALID region [padding, T - padding)
    only, never over padded/invalid positions. Uses the validated transfer
    core implementation (formula and renormalization unchanged):

        H_m = sum_k q_k * (PPV_{m,k} - PPV_m)^2

    with q_k = n_{k,valid} / n_valid, PPV_{m,k} the activation rate over
    valid positions in regime k, PPV_m the global valid-region activation
    rate, and regimes with n_{k,valid} < ceil(min_occupancy * T) excluded
    with renormalized weights.

    act_het  : (N, F_het, T) bool   raw activation indicators
    valid_het: (F_het, T)   bool    per-feature valid-region mask
    regimes  : (N, T)       int     hard regime assignment k_t
    """
    return heterogeneity_features(act_het, valid_het, regimes, K=K,
                                  min_occupancy=min_occupancy)


# ====================================================================
# Regime Controls
# ====================================================================

M2_RNG_OFFSET = 900001   # fixed, documented stream offset for M2
M3_RNG_OFFSET = 900002   # fixed, documented stream offset for M3


def create_random_regime_control(regime_assignments, seed, K=8):
    """SPEC-EXACT M2 (Stage A audit fix, A6/DOC-10 item 4).

    For each sample INDEPENDENTLY: generate a new random regime array with
    EXACTLY the per-sample occupancy histogram of the actual DRTN labels
    (same length, same counts per regime, same regime support) and random
    temporal placement.

    Construction deliberately differs from M3's index permutation: positions
    for each regime are drawn WITHOUT REPLACEMENT from the free-position
    pool in regime order. This is uniform over all arrangements with fixed
    per-sample counts (mathematically equivalent to a uniform permutation of
    the multiset -- the explicit reason allowed by audit A6) while being an
    independent code path and an independent RNG stream (seed + 900001,
    disjoint from M3's seed + 900002).

    No labels are used. Deterministic under a fixed seed.
    """
    rng = np.random.RandomState(seed + M2_RNG_OFFSET)
    n_samples, T = regime_assignments.shape
    out = np.empty_like(regime_assignments)
    for i in range(n_samples):
        counts = np.bincount(regime_assignments[i], minlength=K)
        free = np.arange(T)
        for k in np.flatnonzero(counts):
            pos = rng.choice(free, size=int(counts[k]), replace=False)
            out[i, pos] = k
            free = free[~np.isin(free, pos)]
    return out


def create_shuffled_regime_control(regime_assignments, seed):
    """M3: per-sample deterministic permutation of the ACTUAL DRTN labels.

    Preserves the exact per-sample regime histogram and destroys temporal
    alignment. Uses an independent RNG stream (seed + 900002), disjoint
    from M2's stream (seed + 900001) -- audit A7.
    """
    rng = np.random.RandomState(seed + M3_RNG_OFFSET)
    n_samples, T = regime_assignments.shape
    shuffled = np.zeros_like(regime_assignments)
    for i in range(n_samples):
        seq = regime_assignments[i].copy()
        perm = rng.permutation(T)
        shuffled[i] = seq[perm]
    return shuffled


def extract_drtn_regimes(model, X, device, batch_size=64):
    """Extract hard regime assignments from frozen DRTN model."""
    zn = ((X - X.mean(-1, keepdims=True)) /
          (X.std(-1, keepdims=True) + 1e-8)).astype(np.float32)
    dl = DataLoader(
        TensorDataset(torch.from_numpy(zn)[:, None, :]),
        batch_size=batch_size, shuffle=False)
    all_assignments = []
    model.eval()
    with torch.no_grad():
        for (xb,) in dl:
            xb = xb.to(device)
            z = model.encoder(xb)
            _, assign, _ = model.vq.quantize(z)
            all_assignments.append(assign.cpu().numpy())
    return np.concatenate(all_assignments, axis=0)


def compute_regime_occupancy_stats(regimes, K=8):
    """Compute regime usage statistics."""
    import math
    all_assignments = regimes.flatten()
    total = len(all_assignments)
    counts = np.bincount(all_assignments, minlength=K).astype(np.float64)
    q = counts / total
    nz = q[q > 0]
    H = float(-(nz * np.log(nz)).sum()) if len(nz) else 0.0
    return {
        "usage": q.tolist(),
        "counts": counts.tolist(),
        "entropy": H,
        "normalized_entropy": H / math.log(K) if K > 1 else 1.0,
        "perplexity": float(np.exp(min(H, 700.0))),
        "active_codes": int((q > 0).sum()),
        "dominant_fraction": float(q.max()),
    }


# ====================================================================
# Main Experiment Runner
# ====================================================================

def run_one_seed(seed, data, device, smoke=False):
    """Run M0-M3 for a single seed. Returns results dict."""
    print(f"\n{'='*70}")
    print(f"  SEED {seed}")
    print(f"{'='*70}")

    seed_dir = os.path.join(BASE_DIR, f"seed{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    Xtr, ytr = data["Xtr"], data["ytr"]
    Xva, yva = data["Xva"], data["yva"]
    Xte, yte = data["Xte"], data["yte"]
    Xtrva = np.vstack([Xtr, Xva])
    ytrva = np.concatenate([ytr, yva])

    # ---- Load or train DRTN ----
    print(f"\n  [DRTN] Loading checkpoint for seed {seed}...")
    drtn, drtn_info = load_drtn_checkpoint(seed, device)
    if drtn is None:
        print(f"  [DRTN] No existing checkpoint for seed {seed}. Training...")
        ckpt, drtn_info = train_drtn_for_seed(seed, data, device, smoke=smoke)
        drtn_dir = os.path.join(ROOT, "results", f"drtn_haptics_seed{seed}", "R5")
        os.makedirs(drtn_dir, exist_ok=True)
        torch.save(ckpt, os.path.join(drtn_dir, "checkpoint.pt"))
        with open(os.path.join(drtn_dir, "result.json"), "w") as f:
            json.dump(drtn_info, f, indent=2)
        # Reload
        drtn, drtn_info = load_drtn_checkpoint(seed, device)
    print(f"  [DRTN] val MF1: {drtn_info.get('best_val_mf1', drtn_info.get('test_mf1'))}")

    # ---- Extract DRTN regimes ----
    print(f"  [REGIMES] Extracting DRTN regimes...")
    t0 = time.time()
    regimes_trva = extract_drtn_regimes(drtn, Xtrva, device=device)
    regimes_te = extract_drtn_regimes(drtn, Xte, device=device)
    regimes_tr = regimes_trva[:len(Xtr)]
    regimes_va = regimes_trva[len(Xtr):]
    print(f"  [REGIMES] Done in {time.time()-t0:.1f}s, "
          f"values: {np.unique(regimes_te)}")

    # ---- MiniROCKET ----
    print(f"  [MR] Fitting MiniRocket with seed={seed}...")
    from aeon.transformations.collection.convolution_based import MiniRocket
    extractor = MiniRocket(random_state=seed, n_jobs=-1)
    extractor.fit(Xtr[:, None, :].astype(np.float32))

    Ftr = extractor.transform(Xtr[:, None, :].astype(np.float32))
    Fva = extractor.transform(Xva[:, None, :].astype(np.float32))
    Fte = extractor.transform(Xte[:, None, :].astype(np.float32))
    Ftrva = extractor.transform(Xtrva[:, None, :].astype(np.float32))
    n_features = Ftr.shape[1]
    assert n_features == 9996, f"Expected 9996 features, got {n_features}"
    print(f"  [MR] Features: {n_features}")

    # ---- Raw activations (valid-region convention, Stage A fix) ----
    # DOC-10 item 9: activations are evaluated ONLY on each feature's aeon
    # valid region [padding, T-padding); padded positions never enter any
    # activation rate. compute_raw_minirocket_features (legacy, all-T) has
    # been removed per BUG GUARD 5.
    print(f"  [RAW] Computing raw activations (valid-region convention)...")
    t0 = time.time()
    act_tr, valid = compute_raw_activations(extractor, Xtr)
    act_va, _ = compute_raw_activations(extractor, Xva)
    act_te, _ = compute_raw_activations(extractor, Xte)
    act_trva, _ = compute_raw_activations(extractor, Xtrva)
    print(f"  [RAW] Done in {time.time()-t0:.1f}s")

    # ---- DOC-10 item 1: raw-extractor PPV identity with canonical aeon ----
    PPV_trva_check = ppv_from_activations(act_trva, valid)
    mr_identity = float(np.max(np.abs(PPV_trva_check - Ftrva)))
    print(f"  [CHECK] raw-extractor PPV vs aeon transform: max|diff|={mr_identity:.2e}")
    assert mr_identity < 1e-5, "raw extractor disagrees with canonical aeon MiniRocket"

    # ---- Feature blocks ----
    Ftr_global = Ftr[:, :N_GLOBAL]
    Fte_global = Fte[:, :N_GLOBAL]
    Ftrva_global = Ftrva[:, :N_GLOBAL]
    # BUG GUARD 1: slice the FEATURE axis (axis 1), never the sample axis.
    act_het_tr = act_tr[:, N_GLOBAL:, :]
    act_het_va = act_va[:, N_GLOBAL:, :]
    act_het_te = act_te[:, N_GLOBAL:, :]
    act_het_trva = act_trva[:, N_GLOBAL:, :]
    valid_het = valid[N_GLOBAL:]

    # M0: Canonical MiniROCKET
    M0_tr = Ftrva
    M0_te = Fte

    # M1: DRTN-conditioned
    Ftrva_het_m1 = compute_regime_heterogeneity(
        act_het_trva, valid_het, regimes_trva)
    Fte_het_m1 = compute_regime_heterogeneity(
        act_het_te, valid_het, regimes_te)
    M1_tr = np.hstack([Ftrva_global, Ftrva_het_m1])
    M1_te = np.hstack([Fte_global, Fte_het_m1])

    # M2: Random-regime control (per-sample occupancy, independent RNG)
    rng_regime = create_random_regime_control(regimes_trva, seed=seed)
    rng_regime_te = create_random_regime_control(regimes_te, seed=seed)
    Ftrva_het_m2 = compute_regime_heterogeneity(
        act_het_trva, valid_het, rng_regime)
    Fte_het_m2 = compute_regime_heterogeneity(
        act_het_te, valid_het, rng_regime_te)
    M2_tr = np.hstack([Ftrva_global, Ftrva_het_m2])
    M2_te = np.hstack([Fte_global, Fte_het_m2])

    # M3: Shuffled-regime control (independent RNG stream)
    shuf_regime = create_shuffled_regime_control(regimes_trva, seed=seed)
    shuf_regime_te = create_shuffled_regime_control(regimes_te, seed=seed)
    Ftrva_het_m3 = compute_regime_heterogeneity(
        act_het_trva, valid_het, shuf_regime)
    Fte_het_m3 = compute_regime_heterogeneity(
        act_het_te, valid_het, shuf_regime_te)
    M3_tr = np.hstack([Ftrva_global, Ftrva_het_m3])
    M3_te = np.hstack([Fte_global, Fte_het_m3])

    # ---- Stage A audit assertions (A11/A13 acceptance criteria) ----
    # (a) M2 and M3 regime arrays are distinct and occupancy-preserving
    assert not np.array_equal(rng_regime_te, shuf_regime_te), \
        "M2 and M3 regime arrays are identical (audit failure)"
    assert not np.array_equal(rng_regime, shuf_regime), \
        "M2 and M3 trainva regime arrays are identical (audit failure)"
    assert not np.shares_memory(rng_regime_te, shuf_regime_te)
    for _ctrl, _src, _nm in [(rng_regime_te, regimes_te, "M2"),
                             (shuf_regime_te, regimes_te, "M3"),
                             (rng_regime, regimes_trva, "M2"),
                             (shuf_regime, regimes_trva, "M3")]:
        for i in range(len(_src)):
            assert np.array_equal(np.bincount(_ctrl[i], minlength=8),
                                  np.bincount(_src[i], minlength=8)), \
                f"{_nm} per-sample occupancy not preserved (sample {i})"
    # (b) control heterogeneity features are not trivially identical
    assert not np.array_equal(Fte_het_m2, Fte_het_m3), \
        "M2/M3 heterogeneity feature matrices are exactly identical"
    # (c) DOC-10 item 2/9: independent recomputation of H_m for two
    #     deterministic (sample, feature) pairs on the SAME raw responses
    recompute_checks = {}
    for i_s, f_off in [(0, 0), (min(5, len(act_het_te) - 1), 1234)]:
        h_impl = float(Fte_het_m1[i_s, f_off])
        h_ref = independent_heterogeneity_recompute(
            act_het_te[i_s, f_off], valid_het[f_off], regimes_te[i_s])
        recompute_checks[f"sample{i_s}_kernel{f_off}"] = {
            "implemented": h_impl, "independent": h_ref,
            "abs_diff": abs(h_impl - h_ref)}
        # float32 vectorized accumulation vs float64 reference: observed
        # deviation <= 1e-9; a wrong formula would differ at >= 1e-6 scale
        assert abs(h_impl - h_ref) < 1e-7, \
            f"H_m recomputation mismatch at ({i_s},{f_off})"
    print(f"  [CHECK] independent H_m recomputation: {recompute_checks}")
    # (d) M2/M3 divergence statistics on the test features
    d23 = np.abs(Fte_het_m2 - Fte_het_m3)
    m2m3_feature_diff = {
        "max_abs_diff": float(d23.max()),
        "mean_abs_diff": float(d23.mean()),
        "n_exactly_equal": int((d23 == 0).sum()),
        "n_elements": int(d23.size),
    }
    print(f"  [CHECK] M2 vs M3 hetero features: max|d|={m2m3_feature_diff['max_abs_diff']:.3e} "
          f"n_equal={m2m3_feature_diff['n_exactly_equal']}/{m2m3_feature_diff['n_elements']}")

    # Verify dimensions
    for nm, Mt, Me in [("M0", M0_tr, M0_te), ("M1", M1_tr, M1_te),
                        ("M2", M2_tr, M2_te), ("M3", M3_tr, M3_te)]:
        assert Mt.shape[1] == 9996, f"{nm} train features: {Mt.shape[1]}"
        assert Me.shape[1] == 9996, f"{nm} test features: {Me.shape[1]}"

    # Verify global block identity (within-variant, by construction) AND
    # against the independent aeon transform (DOC-10 item 1, second half)
    assert np.allclose(Ftrva_global, M0_tr[:, :N_GLOBAL])
    assert np.array_equal(M1_te[:, :N_GLOBAL], M0_te[:, :N_GLOBAL])

    # Verify heterogeneity is not all zeros
    het_nonzero = float((Fte_het_m1 != 0).mean())
    print(f"  [CHECK] M1 heterogeneity nonzero rate: {het_nonzero:.4f}")
    assert het_nonzero > 0, "M1 heterogeneity features are all zeros!"

    # ---- Per-variant validation features ----
    Fva_global = Fva[:, :N_GLOBAL]
    M0_va = Fva
    Fva_het_m1 = compute_regime_heterogeneity(act_het_va, valid_het, regimes_va)
    M1_va = np.hstack([Fva_global, Fva_het_m1])
    rng_regime_va = create_random_regime_control(regimes_va, seed=seed)
    Fva_het_m2 = compute_regime_heterogeneity(act_het_va, valid_het, rng_regime_va)
    M2_va = np.hstack([Fva_global, Fva_het_m2])
    shuf_regime_va = create_shuffled_regime_control(regimes_va, seed=seed)
    Fva_het_m3 = compute_regime_heterogeneity(act_het_va, valid_het, shuf_regime_va)
    M3_va = np.hstack([Fva_global, Fva_het_m3])

    # ---- Classifiers ----
    print(f"  [CLF] Fitting classifiers...")
    results = {}
    all_preds = {}
    all_val_mf1 = {}

    for name, F_tr, F_te, F_v in [
        ("M0", M0_tr, M0_te, M0_va),
        ("M1", M1_tr, M1_te, M1_va),
        ("M2", M2_tr, M2_te, M2_va),
        ("M3", M3_tr, M3_te, M3_va),
    ]:
        ridge = RidgeClassifierCV(alphas=ALPHAS)
        ridge.fit(F_tr, ytrva)
        pred_va = ridge.predict(F_v)
        pred_te = ridge.predict(F_te)
        val_mf1 = macro_f1(yva, pred_va)
        test_mf1 = macro_f1(yte, pred_te)

        results[name] = {
            "val_macro_f1": round(val_mf1, 4),
            "test_macro_f1": round(test_mf1, 4),
            "accuracy": round(float(accuracy_score(yte, pred_te)), 4),
            "selected_alpha": float(ridge.alpha_),
            "class_f1s": [round(float(x), 4) for x in f1_score(
                yte, pred_te, average=None, zero_division=0,
                labels=list(range(data["n_classes"])))],
        }
        all_preds[name] = pred_te
        print(f"    {name}: val={val_mf1:.4f} test={test_mf1:.4f} "
              f"alpha={ridge.alpha_:.6f}")

    # ---- Regime diagnostics ----
    regime_diag = compute_regime_occupancy_stats(regimes_te, K=8)

    # ---- Complementarity ----
    m0_correct = all_preds["M0"] == yte
    m1_correct = all_preds["M1"] == yte
    m3_correct = all_preds["M3"] == yte
    complementarity = {
        "M0_only_correct": int(np.sum(m0_correct & ~m1_correct)),
        "M1_only_correct": int(np.sum(~m0_correct & m1_correct)),
        "both_correct": int(np.sum(m0_correct & m1_correct)),
        "both_wrong": int(np.sum(~m0_correct & ~m1_correct)),
        "n_samples": int(len(yte)),
    }
    m0_m3_correct = all_preds["M3"] == yte
    complementarity_m3 = {
        "M0_only_correct": int(np.sum(m0_correct & ~m0_m3_correct)),
        "M3_only_correct": int(np.sum(~m0_correct & m0_m3_correct)),
        "both_correct": int(np.sum(m0_correct & m0_m3_correct)),
        "both_wrong": int(np.sum(~m0_correct & ~m0_m3_correct)),
        "n_samples": int(len(yte)),
    }

    # ---- Save seed results ----
    seed_result = {
        "seed": seed,
        "results": results,
        "deltas": {
            "M1_M0": round(results["M1"]["test_macro_f1"] - results["M0"]["test_macro_f1"], 4),
            "M1_M2": round(results["M1"]["test_macro_f1"] - results["M2"]["test_macro_f1"], 4),
            "M1_M3": round(results["M1"]["test_macro_f1"] - results["M3"]["test_macro_f1"], 4),
        },
        "regime_diagnostics": regime_diag,
        "complementarity_M0_M1": complementarity,
        "complementarity_M0_M3": complementarity_m3,
        "drtn_info": {k: v for k, v in drtn_info.items() if k != "history"},
        "feature_check": {
            "heterogeneity_nonzero_rate": het_nonzero,
            "heterogeneity_mean": float(Fte_het_m1.mean()),
            "heterogeneity_std": float(Fte_het_m1.std()),
            "raw_extractor_vs_aeon_max_diff": mr_identity,
            "m2_m3_feature_diff": m2m3_feature_diff,
            "independent_recompute": recompute_checks,
            "m2_rng_offset": M2_RNG_OFFSET,
            "m3_rng_offset": M3_RNG_OFFSET,
        },
    }

    with open(os.path.join(seed_dir, "result.json"), "w") as f:
        json.dump(seed_result, f, indent=2)

    # Save predictions
    with open(os.path.join(seed_dir, "predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_index", "true_class", "M0_pred", "M1_pred",
                     "M2_pred", "M3_pred"])
        for i in range(len(yte)):
            w.writerow([i, int(yte[i]),
                         int(all_preds["M0"][i]), int(all_preds["M1"][i]),
                         int(all_preds["M2"][i]), int(all_preds["M3"][i])])

    print(f"\n  [SUMMARY] Seed {seed}:")
    print(f"    M0 = {results['M0']['test_macro_f1']:.4f}")
    print(f"    M1 = {results['M1']['test_macro_f1']:.4f}  (delta M1-M0 = {seed_result['deltas']['M1_M0']:+.4f})")
    print(f"    M2 = {results['M2']['test_macro_f1']:.4f}")
    print(f"    M3 = {results['M3']['test_macro_f1']:.4f}  (delta M1-M3 = {seed_result['deltas']['M1_M3']:+.4f})")

    return seed_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--skip-drtn-train", action="store_true",
                        help="Skip DRTN training, use existing checkpoints only")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick smoke test with reduced epochs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(BASE_DIR, exist_ok=True)

    print("=" * 70)
    print("DRTN-CONDITIONED MINIROCKET -- HAPTICS 3-SEED CONFIRMATION")
    print("=" * 70)
    print(f"Seeds: {args.seeds}")
    print(f"Device: {device}")

    # Load dataset once
    print("\n[DATA] Loading Haptics...")
    data = load_dataset(DATASET)
    assert len(data["Xtr"]) == 132
    assert len(data["Xva"]) == 23
    assert len(data["Xte"]) == 308
    assert data["L"] == 1092
    assert data["n_classes"] == 5
    print(f"  Train: {len(data['Xtr'])}, Val: {len(data['Xva'])}, Test: {len(data['Xte'])}")
    print(f"  T: {data['L']}, Classes: {data['n_classes']}")

    # Phase A: Train DRTN for seeds that don't have checkpoints
    if not args.skip_drtn_train:
        print("\n" + "=" * 70)
        print("PHASE A: DRTN TRAINING")
        print("=" * 70)
        for seed in args.seeds:
            rdir = os.path.join(ROOT, "results", f"drtn_haptics_seed{seed}", "R5")
            if os.path.exists(os.path.join(rdir, "checkpoint.pt")):
                print(f"\n  Seed {seed}: checkpoint exists, skipping training")
            else:
                print(f"\n  Seed {seed}: training DRTN R5...")
                ckpt, result = train_drtn_for_seed(seed, data, device, smoke=args.smoke)
                os.makedirs(rdir, exist_ok=True)
                torch.save(ckpt, os.path.join(rdir, "checkpoint.pt"))
                with open(os.path.join(rdir, "result.json"), "w") as f:
                    json.dump(result, f, indent=2)
                print(f"  Seed {seed}: DRTN trained, val MF1={result['best_val_mf1']}")

    # Phase B: Run M0-M3 for all seeds
    print("\n" + "=" * 70)
    print("PHASE B: M0-M3 EVALUATION")
    print("=" * 70)

    all_seed_results = []
    for seed in args.seeds:
        seed_result = run_one_seed(seed, data, device, smoke=args.smoke)
        all_seed_results.append(seed_result)

    # ---- Aggregate results ----
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)

    variants = ["M0", "M1", "M2", "M3"]
    aggregate = {}
    for v in variants:
        test_mf1s = [r["results"][v]["test_macro_f1"] for r in all_seed_results]
        aggregate[v] = {
            "mean": round(float(np.mean(test_mf1s)), 4),
            "std": round(float(np.std(test_mf1s)), 4),
            "se": round(float(np.std(test_mf1s) / np.sqrt(len(test_mf1s))), 4),
            "min": round(float(np.min(test_mf1s)), 4),
            "max": round(float(np.max(test_mf1s)), 4),
            "values": [round(x, 4) for x in test_mf1s],
        }

    deltas_m1_m0 = [r["deltas"]["M1_M0"] for r in all_seed_results]
    deltas_m1_m3 = [r["deltas"]["M1_M3"] for r in all_seed_results]
    deltas_m1_m2 = [r["deltas"]["M1_M2"] for r in all_seed_results]

    delta_agg = {
        "M1_M0": {
            "mean": round(float(np.mean(deltas_m1_m0)), 4),
            "std": round(float(np.std(deltas_m1_m0)), 4),
            "se": round(float(np.std(deltas_m1_m0) / np.sqrt(len(deltas_m1_m0))), 4),
            "n_positive": sum(1 for d in deltas_m1_m0 if d > 0),
            "values": [round(x, 4) for x in deltas_m1_m0],
        },
        "M1_M3": {
            "mean": round(float(np.mean(deltas_m1_m3)), 4),
            "std": round(float(np.std(deltas_m1_m3)), 4),
            "se": round(float(np.std(deltas_m1_m3) / np.sqrt(len(deltas_m1_m3))), 4),
            "n_positive": sum(1 for d in deltas_m1_m3 if d > 0),
            "values": [round(x, 4) for x in deltas_m1_m3],
        },
        "M1_M2": {
            "mean": round(float(np.mean(deltas_m1_m2)), 4),
            "std": round(float(np.std(deltas_m1_m2)), 4),
            "se": round(float(np.std(deltas_m1_m2) / np.sqrt(len(deltas_m1_m2))), 4),
            "n_positive": sum(1 for d in deltas_m1_m2 if d > 0),
            "values": [round(x, 4) for x in deltas_m1_m2],
        },
    }

    print("\nVariant results (3-seed aggregate):")
    for v in variants:
        a = aggregate[v]
        print(f"  {v}: mean={a['mean']:.4f} std={a['std']:.4f} "
              f"se={a['se']:.4f} range=[{a['min']:.4f}, {a['max']:.4f}]")
        print(f"       seeds: {a['values']}")

    print("\nDelta results:")
    for dk, dv in delta_agg.items():
        print(f"  {dk}: mean={dv['mean']:+.4f} std={dv['std']:.4f} "
              f"se={dv['se']:.4f} n_positive={dv['n_positive']}/3")
        print(f"       seeds: {dv['values']}")

    # ---- Determine verdict ----
    m1_m0_mean = delta_agg["M1_M0"]["mean"]
    m1_m3_mean = delta_agg["M1_M3"]["mean"]

    if m1_m0_mean > 0 and m1_m3_mean > 0:
        verdict = "SUPPORTED"
        conclusion = ("M1 beats M0 and M3 on average across 3 seeds. "
                      "Learned DRTN temporal regimes provide useful context.")
    elif m1_m0_mean > 0 and abs(m1_m3_mean) < 0.01:
        verdict = "PARTIALLY SUPPORTED"
        conclusion = ("M1 beats M0 but not clearly above M3. "
                      "Regime conditioning helps but DRTN alignment not confirmed.")
    elif abs(m1_m0_mean) < 0.01:
        verdict = "INCONCLUSIVE"
        conclusion = "M1 ~ M0 on average. Regime conditioning does not clearly improve."
    else:
        verdict = "NOT SUPPORTED"
        conclusion = "M1 does not beat M0 on average."

    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"CONCLUSION: {conclusion}")
    print(f"{'='*70}")

    # ---- Save final report ----
    report = {
        "title": "DRTN-CONDITIONED MINIROCKET -- HAPTICS 3-SEED CONFIRMATION",
        "verdict": verdict,
        "conclusion": conclusion,
        "seeds": args.seeds,
        "aggregate": aggregate,
        "deltas": delta_agg,
        "seed_results": all_seed_results,
        "complementarity": {
            "M0_M1": [r["complementarity_M0_M1"] for r in all_seed_results],
            "M0_M3": [r["complementarity_M0_M3"] for r in all_seed_results],
        },
        "regime_diagnostics": [r["regime_diagnostics"] for r in all_seed_results],
    }

    with open(os.path.join(BASE_DIR, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # ---- Console summary ----
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    for seed, r in zip(args.seeds, all_seed_results):
        print(f"\n  Seed {seed}:")
        for v in variants:
            print(f"    {v}: {r['results'][v]['test_macro_f1']:.4f}")
        print(f"    Delta M1-M0: {r['deltas']['M1_M0']:+.4f}")
        print(f"    Delta M1-M3: {r['deltas']['M1_M3']:+.4f}")

    print(f"\n  Aggregate:")
    for v in variants:
        a = aggregate[v]
        print(f"    {v}: {a['mean']:.4f} +/- {a['std']:.4f}")
    print(f"    Delta M1-M0: {delta_agg['M1_M0']['mean']:+.4f} "
          f"(n_pos={delta_agg['M1_M0']['n_positive']}/3)")
    print(f"    Delta M1-M3: {delta_agg['M1_M3']['mean']:+.4f} "
          f"(n_pos={delta_agg['M1_M3']['n_positive']}/3)")
    print(f"\n  VERDICT: {verdict}")
    print(f"  {conclusion}")
    print(f"{'='*70}")

    return report


if __name__ == "__main__":
    main()
