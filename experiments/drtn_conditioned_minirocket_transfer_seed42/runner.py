"""
Seed-42 cross-dataset transfer screen: DRTN-conditioned MiniROCKET.

Datasets (all with the repository's canonical protocol):
    external .ts : EpilepticSeizures, Phoneme  (load_dataset)
    internal NPZ : ECG5000_UNBAL, ECG5000_BAL, CWRU_UNBAL, CWRU_BAL

For each dataset, seed 42, four variants x one test evaluation:
    M0  canonical MiniROCKET (9,996 PPV)
    M1  4,998 canonical PPV + 4,998 DRTN-regime heterogeneity
    M2  same as M1 but random regimes (occupancy-preserving)
    M3  same as M1 but per-sample temporally shuffled regimes

Fairness invariants enforced in code:
    * variant-LOCAL feature variables F_tr / F_va / F_te only
      (guards the historical Fte/F_te shadowing bug)
    * every variant exactly 9,996 features
    * global block of M1/M2/M3 identical to canonical M0 features
    * heterogeneity features from ACTUAL raw responses (non-placeholder)
    * RidgeClassifierCV(alphas=np.logspace(-4,4,20)) everywhere
    * test labels touched exactly once, after all decisions frozen

Usage:
    python -m experiments.drtn_conditioned_minirocket_transfer_seed42.runner
    python -m experiments.drtn_conditioned_minirocket_transfer_seed42.runner --dataset EpilepticSeizures
    python -m experiments.drtn_conditioned_minirocket_transfer_seed42.runner --smoke
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.external_stack_generalization.data import (  # noqa: E402
    load_dataset, znorm as znorm_external,
)
from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (  # noqa: E402
    compute_raw_activations,
    ppv_from_activations,
    heterogeneity_features,
    independent_heterogeneity_recompute,
    dataset_stats,
    compute_regime_occupancy_stats,
    create_random_regime_control,
    create_shuffled_regime_control,
    extract_drtn_regimes,
)
from models.drtn.model import build_model  # noqa: E402

SEED = 42
N_FEATURES = 9996
N_GLOBAL = N_FEATURES // 2          # 4,998
N_HET = N_FEATURES - N_GLOBAL       # 4,998
ALPHAS = np.logspace(-4, 4, 20)
K_CODES = 8
MIN_OCCUPANCY = 0.01

BASELINE_TOLERANCE = 0.0011         # reproduction guard for canonical M0

OUT_DIR = os.path.join(ROOT, "results",
                       "drtn_conditioned_minirocket_transfer_seed42")

# Canonical reference M0 (test Macro-F1) read from repository artifacts.
EXTERNAL_REFS = {
    "EpilepticSeizures": 0.9194,   # results/external_stack_generalization
    "Phoneme": 0.0808,
}
NPZ_REFS = {
    "ECG5000_UNBAL": 0.5938,       # results/baseline_bench
    "ECG5000_BAL": 0.6553,
    "CWRU_UNBAL": 0.9917,
    "CWRU_BAL": 0.9947,
}
REF_VALUES = {**EXTERNAL_REFS, **NPZ_REFS}

# Internal NPZ dataset definitions (canonical benchmark files + splits).
NPZ_SPECS = {
    "ECG5000_UNBAL": ("data/ecg5000_resplit.npz", 5),
    "ECG5000_BAL": ("data/ecg5000_fair_balanced.npz", 5),
    "CWRU_UNBAL": ("data/cwru_unbalanced.npz", 4),
    "CWRU_BAL": ("data/cwru_balanced.npz", 4),
}

EXTERNAL_DATASETS = ["EpilepticSeizures", "Phoneme"]
ALL_DATASETS = EXTERNAL_DATASETS + list(NPZ_SPECS.keys())

DRTN_CFG = dict(d_model=64, n_codes=K_CODES, tau=0.5, ema_decay=0.99,
                beta=0.25, lam_div=0.01, dead_threshold=1e-3,
                revival_patience=100)
BATCH_SIZE = 16
LR = 1e-3
WD = 1e-4
MAX_EPOCHS = 60
PATIENCE = 10


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Dataset loading (canonical protocols, verbatim conventions)
# ---------------------------------------------------------------------------
def znorm(X):
    mu = X.mean(axis=-1, keepdims=True)
    sig = X.std(axis=-1, keepdims=True) + 1e-8
    return ((X - mu) / sig).astype(np.float32)


def load_npz_dataset(ds_name):
    """Canonical benchmark split (identical to run_turs_skb.load_split)."""
    npz_rel, n_cls = NPZ_SPECS[ds_name]
    data = np.load(os.path.join(ROOT, npz_rel))
    if "X_train" in data:
        Xa, ya = data["X_train"], data["y_train"].astype(int)
        Xte, yte = data["X_test"], data["y_test"].astype(int)
        Xtr, Xva, ytr, yva = train_test_split(
            Xa, ya, test_size=0.15, stratify=ya, random_state=SEED)
        val_source = "stratified_15pct_of_train_seed42"
    else:
        Xa, ya = data["X"], data["y"].astype(int)
        Xtr, Xte, ytr, yte = train_test_split(
            Xa, ya, test_size=0.15, stratify=ya, random_state=SEED)
        Xtr, Xva, ytr, yva = train_test_split(
            Xtr, ytr, test_size=0.15, stratify=ytr, random_state=SEED)
        val_source = "stratified_85_15_then_15pct_seed42"
    return {
        "name": ds_name, "Xtr": Xtr, "ytr": ytr, "Xva": Xva, "yva": yva,
        "Xte": Xte, "yte": yte, "n_classes": n_cls,
        "L": int(Xtr.shape[1]), "val_source": val_source,
        "provenance": npz_rel,
    }


def load_any_dataset(ds_name):
    if ds_name in NPZ_SPECS:
        return load_npz_dataset(ds_name)
    d = load_dataset(ds_name)
    return {
        "name": ds_name, "Xtr": d["Xtr"], "ytr": d["ytr"],
        "Xva": d["Xva"], "yva": d["yva"], "Xte": d["Xte"], "yte": d["yte"],
        "n_classes": d["n_classes"], "L": d["L"],
        "val_source": d["val_source"], "provenance": str(d["paths"]),
    }


# ---------------------------------------------------------------------------
# DRTN training / loading (existing R5 config, per-dataset checkpoint)
# ---------------------------------------------------------------------------
def _zn(X):
    return ((X - X.mean(-1, keepdims=True)) /
            (X.std(-1, keepdims=True) + 1e-8)).astype(np.float32)


def drtn_checkpoint_path(ds_name):
    return os.path.join(OUT_DIR, ds_name, "drtn_R5_seed42_checkpoint.pt")


def train_or_load_drtn(ds_name, data, device, smoke=False):
    """Train DRTN R5 (seed 42) on TRAIN only, select on VAL, freeze."""
    path = drtn_checkpoint_path(ds_name)
    if os.path.exists(path) and not smoke:
        ck = torch.load(path, map_location=device, weights_only=False)
        model = build_model("R5", c_in=1, n_classes=data["n_classes"],
                            **{**DRTN_CFG,
                               "traj_layers": ck["config"]["trajectory"]["layers"],
                               "traj_heads": ck["config"]["trajectory"]["heads"],
                               "traj_ffn": ck["config"]["trajectory"]["ffn"],
                               "traj_dropout": ck["config"]["trajectory"]["dropout"]})
        model.load_state_dict(ck["model_state"])
        model.to(device).eval()
        log(f"  [DRTN] loaded frozen checkpoint (val MF1 "
            f"{ck['best_val_mf1']:.4f}, epoch {ck['epoch']})")
        return model, {"best_val_mf1": ck["best_val_mf1"],
                       "epoch": ck["epoch"], "source": "checkpoint"}

    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    max_ep = 6 if smoke else MAX_EPOCHS
    patience = 2 if smoke else PATIENCE
    set_seed(SEED)

    mk = lambda X, y, sh: DataLoader(
        TensorDataset(torch.from_numpy(_zn(X))[:, None, :],
                      torch.from_numpy(np.asarray(y))),
        batch_size=BATCH_SIZE if not smoke else 8, shuffle=sh, num_workers=0,
        generator=torch.Generator().manual_seed(SEED) if sh else None)
    tr_dl = mk(data["Xtr"], data["ytr"], True)
    va_dl = mk(data["Xva"], data["yva"], False)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = build_model("R5", c_in=1, n_classes=data["n_classes"],
                        **DRTN_CFG).to(device)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, steps_per_epoch=len(tr_dl), epochs=max_ep)

    best_val, best_state, best_ep, no_imp, step = -1.0, None, 0, 0, 0
    t0 = time.time()
    for ep in range(max_ep):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            z = model.encoder(xb)
            q_st, assign, commit_raw = model.vq.quantize(z)
            h, _ = model.pool(model.trajectory(q_st))
            logits = model.classifier(h)
            loss = (F.cross_entropy(logits, yb)
                    + DRTN_CFG["beta"] * commit_raw
                    + DRTN_CFG["lam_div"] * model.diversity_loss_from_assign(assign))
            model.vq.ema_step(z.detach().reshape(-1, z.shape[-1]),
                              assign.reshape(-1), step=step)
            step += 1
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

        model.eval()
        preds, tgts = [], []
        with torch.no_grad():
            for xb, yb in va_dl:
                xb = xb.to(device)
                logits, _ = model.forward_with_assign(xb)
                preds.append(logits.argmax(-1).cpu().numpy())
                tgts.append(yb.numpy())
        vp, vt = np.concatenate(preds), np.concatenate(tgts)
        val_mf1 = macro_f1(vt, vp)
        if val_mf1 > best_val:
            best_val, best_ep, no_imp = val_mf1, ep + 1, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            no_imp += 1
        if (ep + 1) % 2 == 0 or ep == 0:
            log(f"    [DRTN {ds_name}] ep{ep+1}/{max_ep} valMF1={val_mf1:.4f} "
                f"(best {best_val:.4f}@{best_ep}) [{time.time()-t0:.0f}s]")
        if no_imp >= patience:
            log(f"    [DRTN {ds_name}] early stop ep{ep+1}")
            break

    model.load_state_dict(best_state)
    ck = {
        "rung": "R5", "seed": SEED, "dataset": ds_name,
        "config": {**DRTN_CFG, "batch_size": BATCH_SIZE, "lr": LR,
                   "weight_decay": WD, "max_epochs": max_ep,
                   "patience": patience,
                   "trajectory": {"layers": 2, "heads": 4, "ffn": 128,
                                  "dropout": 0.1}},
        "epoch": best_ep, "best_val_mf1": best_val,
        "model_state": model.state_dict(),
    }
    if not smoke:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(ck, path)
        log(f"  [DRTN] trained (val MF1 {best_val:.4f}@ep{best_ep}, "
            f"{time.time()-t0:.0f}s), checkpoint frozen")
    else:
        log(f"  [DRTN] smoke-trained (val MF1 {best_val:.4f}), NOT saved")
    return model, {"best_val_mf1": round(best_val, 4), "epoch": best_ep,
                   "source": "trained" if not smoke else "trained_smoke"}


# ---------------------------------------------------------------------------
# Per-dataset experiment
# ---------------------------------------------------------------------------
def run_dataset(ds_name, device, smoke=False):
    log(f"\n{'='*74}\nDATASET {ds_name}\n{'='*74}")
    ds_dir = os.path.join(OUT_DIR, ds_name)
    diag_dir = os.path.join(ds_dir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)

    # ---------------- data + protocol validation ----------------
    data = load_any_dataset(ds_name)
    n_cls, T = data["n_classes"], data["L"]
    Xtr, Xva, Xte = data["Xtr"], data["Xva"], data["Xte"]
    ytr, yva, yte = data["ytr"], data["yva"], data["yte"]
    log(f"  train={len(Xtr)} val={len(Xva)} test={len(Xte)} T={T} "
        f"n_cls={n_cls} val_source={data['val_source']}")
    assert Xtr.shape[1] == T and Xva.shape[1] == T and Xte.shape[1] == T
    assert not np.isnan(Xtr).any() and not np.isnan(Xte).any()

    Xtr_z, Xva_z, Xte_z = znorm(Xtr), znorm(Xva), znorm(Xte)

    # ---------------- DRTN (train on train, select on val, freeze) --------
    drtn, drtn_info = train_or_load_drtn(ds_name, data, device, smoke=smoke)
    t0 = time.time()
    regimes_tr = extract_drtn_regimes(drtn, Xtr_z, device=device)
    regimes_va = extract_drtn_regimes(drtn, Xva_z, device=device)
    regimes_te = extract_drtn_regimes(drtn, Xte_z, device=device)
    log(f"  [REGIMES] extracted in {time.time()-t0:.1f}s; "
        f"test values {np.unique(regimes_te).tolist()}")
    assert regimes_tr.shape == (len(Xtr), T)
    assert regimes_te.min() >= 0 and regimes_te.max() < K_CODES
    # freeze check: parameters identical after extraction
    sd1 = {k: v.clone() for k, v in drtn.state_dict().items()}
    _ = extract_drtn_regimes(drtn, Xte_z[:2], device=device)
    sd2 = drtn.state_dict()
    drtn_frozen = all(torch.equal(sd1[k], sd2[k]) for k in sd1)

    regime_diag = {
        "train": compute_regime_occupancy_stats(regimes_tr, K=K_CODES),
        "val": compute_regime_occupancy_stats(regimes_va, K=K_CODES),
        "test": compute_regime_occupancy_stats(regimes_te, K=K_CODES),
        "drtn": drtn_info, "drtn_frozen_verified": bool(drtn_frozen),
    }

    # ---------------- canonical MiniROCKET (fit on TRAIN) ----------------
    from aeon.transformations.collection.convolution_based import MiniRocket
    log("  [MR] fitting MiniRocket(random_state=42) on TRAIN...")
    t0 = time.time()
    extractor = MiniRocket(random_state=SEED, n_jobs=-1)
    extractor.fit(Xtr_z[:, None, :].astype(np.float32))
    log(f"  [MR] fitted in {time.time()-t0:.1f}s")

    # ---------------- raw activations + canonical PPV ----------------
    log("  [RAW] computing raw activation tensors...")
    t0 = time.time()
    act_tr, valid = compute_raw_activations(extractor, Xtr_z)
    act_va, _ = compute_raw_activations(extractor, Xva_z)
    act_te, _ = compute_raw_activations(extractor, Xte_z)
    log(f"  [RAW] done in {time.time()-t0:.1f}s, shape {act_tr.shape}")

    PPV_tr = ppv_from_activations(act_tr, valid)
    PPV_va = ppv_from_activations(act_va, valid)
    PPV_te = ppv_from_activations(act_te, valid)
    assert PPV_tr.shape == (len(Xtr), N_FEATURES)

    # Cross-check our extractor against the canonical aeon transform.
    Ftr_aeon = extractor.transform(Xtr_z[:, None, :].astype(np.float32))
    mr_match = float(np.max(np.abs(PPV_tr - Ftr_aeon)))
    log(f"  [MR] raw-activation PPV vs aeon transform: max|diff|={mr_match:.2e}")
    assert mr_match < 1e-5, "raw extractor disagrees with canonical aeon PPV"

    # ---------------- variant-LOCAL feature construction ----------------
    biases = extractor.parameters[-1]
    # Global block identity check: our PPV == canonical features (first half)
    # aeon returns float32 PPV; our block A uses the SAME array.

    act_g_tr, act_h_tr = act_tr[:, :N_GLOBAL, :], act_tr[:, N_GLOBAL:, :]
    act_g_va, act_h_va = act_va[:, :N_GLOBAL, :], act_va[:, N_GLOBAL:, :]
    act_g_te, act_h_te = act_te[:, :N_GLOBAL, :], act_te[:, N_GLOBAL:, :]

    # Block A: canonical PPV (bit-identical to aeon features)
    A_tr, A_va, A_te = PPV_tr[:, :N_GLOBAL], PPV_va[:, :N_GLOBAL], PPV_te[:, :N_GLOBAL]

    # Heterogeneity blocks per variant (variant-LOCAL names)
    H1_tr = heterogeneity_features(act_h_tr, valid[N_GLOBAL:], regimes_tr, K=K_CODES)
    H1_va = heterogeneity_features(act_h_va, valid[N_GLOBAL:], regimes_va, K=K_CODES)
    H1_te = heterogeneity_features(act_h_te, valid[N_GLOBAL:], regimes_te, K=K_CODES)

    rng_tr = create_random_regime_control(regimes_tr, seed=SEED)
    rng_va = create_random_regime_control(regimes_va, seed=SEED)
    rng_te = create_random_regime_control(regimes_te, seed=SEED)
    H2_tr = heterogeneity_features(act_h_tr, valid[N_GLOBAL:], rng_tr, K=K_CODES)
    H2_va = heterogeneity_features(act_h_va, valid[N_GLOBAL:], rng_va, K=K_CODES)
    H2_te = heterogeneity_features(act_h_te, valid[N_GLOBAL:], rng_te, K=K_CODES)

    shf_tr = create_shuffled_regime_control(regimes_tr, seed=SEED)
    shf_va = create_shuffled_regime_control(regimes_va, seed=SEED)
    shf_te = create_shuffled_regime_control(regimes_te, seed=SEED)
    H3_tr = heterogeneity_features(act_h_tr, valid[N_GLOBAL:], shf_tr, K=K_CODES)
    H3_va = heterogeneity_features(act_h_va, valid[N_GLOBAL:], shf_va, K=K_CODES)
    H3_te = heterogeneity_features(act_h_te, valid[N_GLOBAL:], shf_te, K=K_CODES)

    # ---------------- mandatory pre-test audits ----------------
    # (4-7) feature budget + global identity
    # (8) heterogeneity non-placeholder
    audits = {}
    nz1 = float((H1_te != 0).mean())
    audits["heterogeneity_nonzero_fraction_M1"] = nz1
    assert nz1 > 0.5, "M1 heterogeneity looks like placeholder zeros"
    # (9) independent recomputation for random sample/kernel pairs
    rng = np.random.RandomState(0)
    recomp_errs = []
    for _ in range(12):
        i = rng.randint(0, act_te.shape[0])
        m = rng.randint(0, act_te.shape[1] - N_GLOBAL)
        a = np.asarray(act_te[i, N_GLOBAL + m, :], dtype=np.bool_)
        v = np.asarray(valid[N_GLOBAL + m, :], dtype=np.bool_)
        h_ref = independent_heterogeneity_recompute(a, v, regimes_te[i], K=K_CODES)
        h_fast = H1_te[i, m]
        recomp_errs.append(abs(h_ref - h_fast))
    audits["independent_recompute_max_abs_err"] = float(np.max(recomp_errs))
    # float32 vectorized matmul vs float64 reference: ~1e-8 tolerance;
    # a real bug (e.g. placeholder zeros) produces errors of order 1e-2+
    assert np.max(recomp_errs) < 1e-7, "heterogeneity recomputation mismatch"
    # (10/11) occupancy preservation (M2 global hist; M3 per-sample hist)
    K_ = K_CODES
    glob_tr = np.bincount(regimes_tr.ravel(), minlength=K_) / regimes_tr.size
    glob_m2 = np.bincount(rng_tr.ravel(), minlength=K_) / rng_tr.size
    audits["m2_global_occupancy_max_abs_diff"] = float(np.max(np.abs(glob_tr - glob_m2)))
    assert audits["m2_global_occupancy_max_abs_diff"] < 0.02
    per_sample_max = 0.0
    for i in range(len(regimes_te)):
        h_a = np.bincount(regimes_te[i], minlength=K_)
        h_b = np.bincount(shf_te[i], minlength=K_)
        per_sample_max = max(per_sample_max, float(np.max(np.abs(h_a - h_b))))
    audits["m3_per_sample_hist_max_abs_diff"] = per_sample_max
    assert per_sample_max == 0.0, "M3 must preserve per-sample regime histogram"
    # (12) M3 destroys temporal alignment
    align_destroyed_frac = float(np.mean(
        [not np.array_equal(regimes_te[i], shf_te[i])
         for i in range(min(50, len(regimes_te)))]))
    audits["m3_alignment_changed_fraction"] = align_destroyed_frac
    assert align_destroyed_frac > 0.9
    # (13) frozen DRTN
    audits["drtn_frozen_verified"] = bool(drtn_frozen)
    assert drtn_frozen

    # deterministic controls check (rerun with same seed -> identical)
    rng_te_b = create_random_regime_control(regimes_te, seed=SEED)
    shf_te_b = create_shuffled_regime_control(regimes_te, seed=SEED)
    audits["m2_deterministic"] = bool(np.array_equal(rng_te, rng_te_b))
    audits["m3_deterministic"] = bool(np.array_equal(shf_te, shf_te_b))
    assert audits["m2_deterministic"] and audits["m3_deterministic"]

    # ---------------- assemble variant-LOCAL feature matrices --------------
    # (variable-shadowing guard: each variant builds F_tr/F_va/F_te fresh)
    F_tr_by_variant = {
        "M0": np.hstack([PPV_tr, PPV_tr[:, N_GLOBAL:]]),  # placeholder, fixed below
    }
    # NOTE: M0 uses the FULL canonical 9,996 PPV features directly.
    F_tr_by_variant["M0"] = PPV_tr
    F_va_by_variant = {"M0": PPV_va}
    F_te_by_variant = {"M0": PPV_te}

    F_tr_by_variant["M1"] = np.hstack([A_tr, H1_tr])
    F_va_by_variant["M1"] = np.hstack([A_va, H1_va])
    F_te_by_variant["M1"] = np.hstack([A_te, H1_te])

    F_tr_by_variant["M2"] = np.hstack([A_tr, H2_tr])
    F_va_by_variant["M2"] = np.hstack([A_va, H2_va])
    F_te_by_variant["M2"] = np.hstack([A_te, H2_te])

    F_tr_by_variant["M3"] = np.hstack([A_tr, H3_tr])
    F_va_by_variant["M3"] = np.hstack([A_va, H3_va])
    F_te_by_variant["M3"] = np.hstack([A_te, H3_te])

    # shape/global-identity audits
    for v in ["M0", "M1", "M2", "M3"]:
        assert F_tr_by_variant[v].shape[1] == N_FEATURES, v
        assert F_va_by_variant[v].shape[1] == N_FEATURES, v
        assert F_te_by_variant[v].shape[1] == N_FEATURES, v
    for v in ["M1", "M2", "M3"]:
        assert np.array_equal(F_tr_by_variant[v][:, :N_GLOBAL], PPV_tr[:, :N_GLOBAL]), v
        assert np.array_equal(F_te_by_variant[v][:, :N_GLOBAL], PPV_te[:, :N_GLOBAL]), v
    # global block identity with the canonical aeon transform
    assert np.allclose(F_tr_by_variant["M1"][:, :N_GLOBAL], Ftr_aeon[:, :N_GLOBAL])

    # ---------------- classifier: one fit per variant ----------------------
    ytrva = np.concatenate([ytr, yva])
    # Per-variant train+val feature matrices (variant-LOCAL naming: F_fit).
    # The classifier protocol is canonical:
    #   fit              = train+val features (RidgeCV selects alpha by
    #                      internal CV on these rows)
    #   test             = exactly once, same variant's test features
    #   val MF1 reported = diagnostic only (model already includes val)
    F_fit_by_variant = {
        v: np.vstack([F_tr_by_variant[v], F_va_by_variant[v]])
        for v in ["M0", "M1", "M2", "M3"]
    }
    for v in ["M0", "M1", "M2", "M3"]:
        assert F_fit_by_variant[v].shape == (len(ytrva), N_FEATURES), v
    results, preds_by_variant = {}, {}
    for v in ["M0", "M1", "M2", "M3"]:
        F_fit = F_fit_by_variant[v]         # variant-LOCAL names ONLY
        F_va = F_va_by_variant[v]
        F_te = F_te_by_variant[v]
        ridge = RidgeClassifierCV(alphas=ALPHAS)
        t0 = time.time()
        ridge.fit(F_fit, ytrva)             # train+val fit (canonical)
        fit_s = time.time() - t0
        pred_te = ridge.predict(F_te)       # SAME variant's test features
        pred_va = ridge.predict(F_va)       # diagnostic val MF1 (fit on tr+val)
        test_mf1 = macro_f1(yte, pred_te)
        results[v] = {
            "test_macro_f1": round(test_mf1, 4),
            "val_macro_f1_diag": round(macro_f1(yva, pred_va), 4),
            "accuracy": round(float(accuracy_score(yte, pred_te)), 4),
            "selected_alpha": float(ridge.alpha_),
            "class_f1s": [round(float(x), 4) for x in f1_score(
                yte, pred_te, average=None, zero_division=0,
                labels=list(range(n_cls)))],
            "ridge_fit_s": round(fit_s, 2),
        }
        preds_by_variant[v] = pred_te
        log(f"  [{v}] test MF1={test_mf1:.4f} alpha={ridge.alpha_:.4f}")

    # ---------------- canonical baseline reproduction ----------------------
    ref = REF_VALUES[ds_name]
    d0 = round(results["M0"]["test_macro_f1"] - ref, 4)
    reproduction = {"reference_mf1": ref, "reproduction_delta": d0,
                    "passed": abs(d0) <= BASELINE_TOLERANCE}
    log(f"  [M0 REPRODUCTION] {results['M0']['test_macro_f1']:.4f} vs ref "
        f"{ref:.4f} (delta {d0:+.4f}, tol {BASELINE_TOLERANCE}) "
        f"-> {'PASS' if reproduction['passed'] else 'FAIL'}")

    # ---------------- deltas + classification -----------------------------
    deltas = {
        "M1_M0": round(results["M1"]["test_macro_f1"] - results["M0"]["test_macro_f1"], 4),
        "M1_M2": round(results["M1"]["test_macro_f1"] - results["M2"]["test_macro_f1"], 4),
        "M1_M3": round(results["M1"]["test_macro_f1"] - results["M3"]["test_macro_f1"], 4),
    }
    if not reproduction["passed"]:
        category = "INVALID"
    elif deltas["M1_M0"] > 0.005 and deltas["M1_M3"] > 0.005:
        category = "PROMISING"
    elif deltas["M1_M0"] < -0.005 or deltas["M1_M3"] < -0.005:
        category = "NEGATIVE"
    else:
        category = "NEUTRAL"

    # ---------------- complementarity (descriptive) ------------------------
    m0c = preds_by_variant["M0"] == yte
    m1c = preds_by_variant["M1"] == yte
    complementarity = {
        "both_correct": int((m0c & m1c).sum()),
        "M0_only_correct": int((m0c & ~m1c).sum()),
        "M1_only_correct": int((~m0c & m1c).sum()),
        "both_wrong": int((~m0c & ~m1c).sum()),
        "n_samples": int(len(yte)),
    }

    # ---------------- diagnostics ------------------------------------------
    hetero_diag = {
        "M1_train": dataset_stats(H1_tr), "M1_test": dataset_stats(H1_te),
        "M2_test": dataset_stats(H2_te), "M3_test": dataset_stats(H3_te),
    }
    feature_diag = {
        "ppv_global_mean": float(PPV_te.mean()),
        "ppv_global_std": float(PPV_te.std()),
        "ppv_global_nonzero": float((PPV_te != 0).mean()),
        "raw_vs_aeon_max_diff": mr_match,
    }

    # ---------------- save artifacts ----------------------------------------
    with open(os.path.join(ds_dir, "predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_index", "true_class", "M0_pred", "M1_pred",
                    "M2_pred", "M3_pred", "M0_correct", "M1_correct",
                    "M2_correct", "M3_correct"])
        for i in range(len(yte)):
            w.writerow([i, int(yte[i])] +
                       [int(preds_by_variant[v][i]) for v in ["M0", "M1", "M2", "M3"]] +
                       [int(preds_by_variant[v][i] == yte[i])
                        for v in ["M0", "M1", "M2", "M3"]])

    with open(os.path.join(diag_dir, "regime_statistics.json"), "w") as f:
        json.dump(regime_diag, f, indent=2)
    with open(os.path.join(diag_dir, "feature_statistics.json"), "w") as f:
        json.dump({"heterogeneity": hetero_diag, "features": feature_diag,
                   "audits": audits}, f, indent=2)
    with open(os.path.join(diag_dir, "complementarity.json"), "w") as f:
        json.dump(complementarity, f, indent=2)

    result = {
        "dataset": ds_name, "seed": SEED, "T": int(T),
        "n_classes": n_cls,
        "split": {"train": int(len(Xtr)), "val": int(len(Xva)),
                  "test": int(len(Xte))},
        "val_source": data["val_source"], "provenance": data["provenance"],
        "n_features": N_FEATURES, "K_regimes": K_CODES,
        "min_occupancy": MIN_OCCUPANCY,
        "results": results, "deltas": deltas, "category": category,
        "canonical_reproduction": reproduction,
        "complementarity": complementarity,
        "regime_diagnostics": {
            "test": regime_diag["test"], "drtn": drtn_info,
            "drtn_frozen_verified": bool(drtn_frozen)},
        "heterogeneity_diagnostics": hetero_diag,
        "feature_statistics": feature_diag,
        "audits": audits,
    }
    with open(os.path.join(ds_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    log(f"  [CATEGORY] {category}  (M1-M0 {deltas['M1_M0']:+.4f}, "
        f"M1-M3 {deltas['M1_M3']:+.4f}, M1-M2 {deltas['M1_M2']:+.4f})")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="*", default=ALL_DATASETS)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    set_seed(SEED)

    log("=" * 74)
    log("DRTN-CONDITIONED MINIROCKET -- CROSS-DATASET TRANSFER SCREEN (SEED 42)")
    log("=" * 74)
    log(f"Datasets: {args.dataset}")
    log(f"Device: {device}")

    # config artifact
    config = {
        "seed": SEED,
        "n_features": N_FEATURES, "n_global": N_GLOBAL, "n_het": N_HET,
        "K_regimes": K_CODES, "min_occupancy": MIN_OCCUPANCY,
        "ridge": "RidgeClassifierCV(alphas=np.logspace(-4,4,20))",
        "minirocket": "aeon MiniRocket(random_state=42, n_jobs=-1)",
        "drtn": "official R5 config, trained per dataset on TRAIN, "
                "checkpoint selection on VAL, frozen before regime extraction",
        "datasets": args.dataset,
        "baseline_tolerance": BASELINE_TOLERANCE,
        "reference_values": REF_VALUES,
        "smoke": bool(args.smoke),
    }
    with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    all_results = []
    for ds in args.dataset:
        res = run_dataset(ds, device, smoke=args.smoke)
        all_results.append(res)
        # incremental master save
        with open(os.path.join(OUT_DIR, "report.json"), "w") as f:
            json.dump({"config": config, "datasets": all_results}, f, indent=2)

    # -------- master table --------
    log("\n" + "=" * 74)
    log("MASTER TABLE (test Macro-F1)")
    log(f"{'Dataset':<20} {'M0':>7} {'M1':>7} {'M2':>7} {'M3':>7} "
        f"{'M1-M0':>7} {'M1-M3':>7} {'M1-M2':>7}  {'Category':<10}")
    for r in all_results:
        rr = r["results"]
        d = r["deltas"]
        log(f"{r['dataset']:<20} {rr['M0']['test_macro_f1']:>7.4f} "
            f"{rr['M1']['test_macro_f1']:>7.4f} {rr['M2']['test_macro_f1']:>7.4f} "
            f"{rr['M3']['test_macro_f1']:>7.4f} {d['M1_M0']:>+7.4f} "
            f"{d['M1_M3']:>+7.4f} {d['M1_M2']:>+7.4f}  {r['category']:<10}")

    n_pass = sum(r["canonical_reproduction"]["passed"] for r in all_results)
    log(f"\nCanonical M0 reproduction: {n_pass}/{len(all_results)} datasets "
        f"within tolerance {BASELINE_TOLERANCE}")
    log("DONE.")


if __name__ == "__main__":
    main()
