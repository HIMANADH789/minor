"""
Stage B runner: ECG5000_BAL 3-seed confirmation (seeds 42/43/44) of the
DRTN-conditioned MiniROCKET method, using the EXACT audited implementation
from Stage A.

Scientific definitions (identical to audited Haptics 3-seed):
    M0  canonical aeon MiniRocket, 9996 features, RidgeClassifierCV
    M1  4998 global PPV + 4998 DRTN regime-heterogeneity
    M2  4998 global PPV + 4998 heterogeneity, per-sample occupancy-
        preserving RANDOM regimes (RNG stream seed+900001)
    M3  4998 global PPV + 4998 heterogeneity, per-sample SHUFFLED actual
        DRTN labels (RNG stream seed+900002)

    H_m = sum_k q_k (PPV_{m,k} - PPV_m)^2 over aeon valid regions only.

Canonical seed-42 M0 reference (results/baseline_bench): Macro-F1 = 0.6553.
Test-once policy per seed x variant (12 official test evaluations total).

Usage:
    python -m experiments.drtn_conditioned_minirocket_ecg5000_bal_3seed.runner \
        [--seeds 42 43 44] [--train-drtn] [--smoke]
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
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.drtn_conditioned_minirocket_haptics_3seed.runner import (  # noqa: E402
    create_random_regime_control,     # audited Stage A implementation
    create_shuffled_regime_control,   # audited Stage A implementation
    compute_regime_heterogeneity,     # audited: valid-region convention
    compute_regime_occupancy_stats,
)
from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (  # noqa: E402
    compute_raw_activations,
    ppv_from_activations,
    independent_heterogeneity_recompute,
)
from models.drtn.model import build_model  # noqa: E402

# ---------------------------------------------------------------- constants
DATASET = "ECG5000_BAL"
NPZ_PATH = os.path.join(ROOT, "data", "ecg5000_fair_balanced.npz")
N_CLASSES = 5
SEEDS = [42, 43, 44]

N_FEATURES = 9996
N_GLOBAL = N_FEATURES // 2
N_HETEROGENEITY = N_FEATURES - N_GLOBAL
ALPHAS = np.logspace(-4, 4, 20)

K_CODES = 8
D_MODEL = 64
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

CANONICAL_M0_REF = 0.6553          # results/baseline_bench ECG5000_BAL
M0_TOLERANCE = 0.0011              # repository reproduction tolerance

BASE_DIR = os.path.join(ROOT, "results", "drtn_conditioned_minirocket_ecg5000_bal_3seed")
DRTN_DIR = os.path.join(ROOT, "results", "drtn_ecg5000_bal_3seed")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def znorm(X):
    return ((X - X.mean(-1, keepdims=True)) /
            (X.std(-1, keepdims=True) + 1e-8)).astype(np.float32)


def load_data():
    """Canonical benchmark split (identical to run_turs_skb.load_split)."""
    d = np.load(NPZ_PATH)
    if "X_train" in d:
        Xa, ya = d["X_train"], d["y_train"].astype(int)
        Xte, yte = d["X_test"], d["y_test"].astype(int)
        Xtr, Xva, ytr, yva = train_test_split(
            Xa, ya, test_size=0.15, stratify=ya, random_state=SEEDS[0])
        val_source = "stratified_15pct_of_train_seed42"
    else:
        Xa, ya = d["X"], d["y"].astype(int)
        Xtr, Xte, ytr, yte = train_test_split(
            Xa, ya, test_size=0.15, stratify=ya, random_state=SEEDS[0])
        Xtr, Xva, ytr, yva = train_test_split(
            Xtr, ytr, test_size=0.15, stratify=ytr, random_state=SEEDS[0])
        val_source = "stratified_85_15_then_15pct_seed42"
    return {
        "Xtr": Xtr, "ytr": ytr, "Xva": Xva, "yva": yva, "Xte": Xte, "yte": yte,
        "n_classes": N_CLASSES, "L": int(Xtr.shape[1]), "val_source": val_source,
    }


# ------------------------------------------------------------------- DRTN
def drtn_ckpt_path(seed):
    return os.path.join(DRTN_DIR, f"seed{seed}", "checkpoint.pt")


def train_drtn_for_seed(seed, data, device, smoke=False):
    """Train DRTN R5 K=8 (train only, validation selection) -- same protocol
    as the audited Haptics implementation."""
    max_ep = 8 if smoke else MAX_EPOCHS
    patience = 3 if smoke else PATIENCE
    set_seed(seed)

    mk = lambda X, y, sh: DataLoader(
        TensorDataset(torch.from_numpy(znorm(X))[:, None, :],
                      torch.from_numpy(np.asarray(y))),
        batch_size=BATCH_SIZE if not smoke else 8, shuffle=sh, num_workers=0,
        generator=torch.Generator().manual_seed(seed) if sh else None)
    tr_dl = mk(data["Xtr"], data["ytr"], True)
    va_dl = mk(data["Xva"], data["yva"], False)

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
                    + BETA_COMMIT * commit_raw
                    + LAM_DIV * model.diversity_loss_from_assign(assign))
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
            best_state = copy.deepcopy(model.state_dict())
        else:
            no_imp += 1
        print(f"    [DRTN seed{seed}] ep{ep+1}/{max_ep} valMF1={val_mf1:.4f} "
              f"(best {best_val:.4f}@{best_ep}) [{time.time()-t0:.0f}s]",
              flush=True)
        if no_imp >= patience:
            print(f"    [DRTN seed{seed}] early stop ep{ep+1}", flush=True)
            break

    model.load_state_dict(best_state)
    ckpt = {
        "rung": "R5", "config": {
            "dataset": DATASET, "seed": seed, "d_model": D_MODEL,
            "n_codes": K_CODES, "tau": TAU, "ema_decay": EMA_DECAY,
            "beta_commit": BETA_COMMIT, "lam_div": LAM_DIV,
            "dead_threshold": DEAD_THRESHOLD, "revival_patience": REVIVAL_PATIENCE,
            "batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WD,
            "max_epochs": max_ep, "patience": patience,
            "trajectory": {"layers": 2, "heads": 4, "ffn": 128, "dropout": 0.1},
        },
        "epoch": best_ep, "best_val_mf1": best_val, "seed": seed,
        "dataset": DATASET, "model_state": model.state_dict(),
    }
    return ckpt, {"best_val_mf1": round(best_val, 4), "best_epoch": best_ep}


def load_drtn_checkpoint(seed, device):
    path = drtn_ckpt_path(seed)
    if not os.path.exists(path):
        return None, None
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = build_model(
        "R5", c_in=1, n_classes=N_CLASSES, d_model=cfg["d_model"],
        n_codes=cfg["n_codes"], tau=cfg["tau"], ema_decay=cfg["ema_decay"],
        beta=cfg["beta_commit"], lam_div=cfg["lam_div"],
        dead_threshold=cfg["dead_threshold"],
        revival_patience=cfg["revival_patience"])
    model.load_state_dict(ck["model_state"])
    model.to(device).eval()
    return model, {"best_val_mf1": ck["best_val_mf1"], "epoch": ck["epoch"],
                   "source": "checkpoint"}


def extract_drtn_regimes(model, X, device, batch_size=64):
    zn = znorm(X)
    dl = DataLoader(TensorDataset(torch.from_numpy(zn)[:, None, :]),
                    batch_size=batch_size, shuffle=False)
    out = []
    model.eval()
    with torch.no_grad():
        for (xb,) in dl:
            xb = xb.to(device)
            z = model.encoder(xb)
            _, assign, _ = model.vq.quantize(z)
            out.append(assign.cpu().numpy())
    return np.concatenate(out, axis=0)


# ------------------------------------------------------------- per-seed run
def run_one_seed(seed, data, device, smoke=False):
    print(f"\n{'='*70}\n  SEED {seed}\n{'='*70}", flush=True)
    seed_dir = os.path.join(BASE_DIR, f"seed{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    Xtr, ytr = data["Xtr"], data["ytr"]
    Xva, yva = data["Xva"], data["yva"]
    Xte, yte = data["Xte"], data["yte"]
    Xtrva = np.vstack([Xtr, Xva])
    ytrva = np.concatenate([ytr, yva])

    # ---- DRTN: frozen checkpoint (train once per seed, never during eval) --
    drtn, drtn_info = load_drtn_checkpoint(seed, device)
    if drtn is None:
        print(f"  [DRTN] no checkpoint for seed {seed}; training...")
        ckpt, info = train_drtn_for_seed(seed, data, device, smoke=smoke)
        os.makedirs(os.path.dirname(drtn_ckpt_path(seed)), exist_ok=True)
        torch.save(ckpt, drtn_ckpt_path(seed))
        drtn, drtn_info = load_drtn_checkpoint(seed, device)
    print(f"  [DRTN] val MF1: {drtn_info['best_val_mf1']} (frozen)")

    regimes_trva = extract_drtn_regimes(drtn, Xtrva, device=device)
    regimes_te = extract_drtn_regimes(drtn, Xte, device=device)
    regimes_tr = regimes_trva[:len(Xtr)]
    regimes_va = regimes_trva[len(Xtr):]
    assert regimes_trva.shape == (len(Xtrva), data["L"])
    assert regimes_te.min() >= 0 and regimes_te.max() < K_CODES

    # freeze verification: re-extract and compare state dicts
    sd1 = {k: v.clone() for k, v in drtn.state_dict().items()}
    _ = extract_drtn_regimes(drtn, Xte[:2], device=device)
    sd2 = drtn.state_dict()
    drtn_frozen = all(torch.equal(sd1[k], sd2[k]) for k in sd1)
    assert drtn_frozen, "DRTN parameters changed during regime extraction"

    # ---- canonical MiniROCKET (z-normalized input, canonical benchmark
    #      convention: benchmark_baselines.znorm BEFORE MiniRocket) ----
    from aeon.transformations.collection.convolution_based import MiniRocket
    set_seed(seed)
    extractor = MiniRocket(random_state=seed, n_jobs=-1)
    extractor.fit(znorm(Xtr)[:, None, :].astype(np.float32))
    Ftr = extractor.transform(znorm(Xtr)[:, None, :].astype(np.float32))
    Fva = extractor.transform(znorm(Xva)[:, None, :].astype(np.float32))
    Fte = extractor.transform(znorm(Xte)[:, None, :].astype(np.float32))
    Ftrva = extractor.transform(znorm(Xtrva)[:, None, :].astype(np.float32))
    assert Ftr.shape[1] == N_FEATURES

    # ---- raw activations (valid-region convention) ----
    # NOTE: the extractor is FIT on znorm(Xtr) but we compute raw activations
    # from the RAW series. This matches the Haptics reference implementation,
    # where activations and PPV are computed from the same arrays fed to the
    # extractor. To keep the extractor's fitted kernels identical, activations
    # must be computed on the SAME normalized series as the transform.
    act_tr, valid = compute_raw_activations(extractor, znorm(Xtr))
    act_va, _ = compute_raw_activations(extractor, znorm(Xva))
    act_te, _ = compute_raw_activations(extractor, znorm(Xte))
    act_trva, _ = compute_raw_activations(extractor, znorm(Xtrva))

    PPV_trva_check = ppv_from_activations(act_trva, valid)
    mr_identity = float(np.max(np.abs(PPV_trva_check - Ftrva)))
    print(f"  [CHECK] raw-extractor PPV vs aeon transform: max|diff|={mr_identity:.2e}")
    assert mr_identity < 1e-5, "raw extractor disagrees with canonical aeon MiniRocket"

    Ftr_global = Ftr[:, :N_GLOBAL]
    Fte_global = Fte[:, :N_GLOBAL]
    Ftrva_global = Ftrva[:, :N_GLOBAL]
    act_het_tr = act_tr[:, N_GLOBAL:, :]
    act_het_va = act_va[:, N_GLOBAL:, :]
    act_het_te = act_te[:, N_GLOBAL:, :]
    act_het_trva = act_trva[:, N_GLOBAL:, :]
    valid_het = valid[N_GLOBAL:]

    # ---- variants (variant-LOCAL feature matrices only) ----
    M0_tr, M0_va, M0_te = Ftrva, Fva, Fte

    Ftrva_het_m1 = compute_regime_heterogeneity(act_het_trva, valid_het, regimes_trva)
    Fte_het_m1 = compute_regime_heterogeneity(act_het_te, valid_het, regimes_te)
    Fva_het_m1 = compute_regime_heterogeneity(act_het_va, valid_het, regimes_va)
    M1_tr = np.hstack([Ftrva_global, Ftrva_het_m1])
    M1_va = np.hstack([Fva[:, :N_GLOBAL], Fva_het_m1])
    M1_te = np.hstack([Fte_global, Fte_het_m1])

    rng_regime = create_random_regime_control(regimes_trva, seed=seed)
    rng_regime_va = create_random_regime_control(regimes_va, seed=seed)
    rng_regime_te = create_random_regime_control(regimes_te, seed=seed)
    Ftrva_het_m2 = compute_regime_heterogeneity(act_het_trva, valid_het, rng_regime)
    Fte_het_m2 = compute_regime_heterogeneity(act_het_te, valid_het, rng_regime_te)
    Fva_het_m2 = compute_regime_heterogeneity(act_het_va, valid_het, rng_regime_va)
    M2_tr = np.hstack([Ftrva_global, Ftrva_het_m2])
    M2_va = np.hstack([Fva[:, :N_GLOBAL], Fva_het_m2])
    M2_te = np.hstack([Fte_global, Fte_het_m2])

    shuf_regime = create_shuffled_regime_control(regimes_trva, seed=seed)
    shuf_regime_va = create_shuffled_regime_control(regimes_va, seed=seed)
    shuf_regime_te = create_shuffled_regime_control(regimes_te, seed=seed)
    Ftrva_het_m3 = compute_regime_heterogeneity(act_het_trva, valid_het, shuf_regime)
    Fte_het_m3 = compute_regime_heterogeneity(act_het_te, valid_het, shuf_regime_te)
    Fva_het_m3 = compute_regime_heterogeneity(act_het_va, valid_het, shuf_regime_va)
    M3_tr = np.hstack([Ftrva_global, Ftrva_het_m3])
    M3_va = np.hstack([Fva[:, :N_GLOBAL], Fva_het_m3])
    M3_te = np.hstack([Fte_global, Fte_het_m3])

    # ---- Stage A audit assertions (same as audited Haptics run) ----
    assert not np.array_equal(rng_regime_te, shuf_regime_te), "M2 == M3 arrays!"
    assert not np.shares_memory(rng_regime_te, shuf_regime_te)
    for _ctrl, _src, _nm in [(rng_regime_te, regimes_te, "M2"),
                             (shuf_regime_te, regimes_te, "M3"),
                             (rng_regime, regimes_trva, "M2"),
                             (shuf_regime, regimes_trva, "M3")]:
        for i in range(len(_src)):
            assert np.array_equal(np.bincount(_ctrl[i], minlength=K_CODES),
                                  np.bincount(_src[i], minlength=K_CODES)), \
                f"{_nm} per-sample occupancy not preserved (sample {i})"
    assert not np.array_equal(Fte_het_m2, Fte_het_m3), \
        "M2/M3 heterogeneity feature matrices are exactly identical"

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
        assert abs(h_impl - h_ref) < 1e-7, f"H_m mismatch at ({i_s},{f_off})"

    d23 = np.abs(Fte_het_m2 - Fte_het_m3)
    m2m3_feature_diff = {
        "max_abs_diff": float(d23.max()), "mean_abs_diff": float(d23.mean()),
        "n_exactly_equal": int((d23 == 0).sum()), "n_elements": int(d23.size)}

    for nm, Mt, Mv, Me in [("M0", M0_tr, M0_va, M0_te), ("M1", M1_tr, M1_va, M1_te),
                           ("M2", M2_tr, M2_va, M2_te), ("M3", M3_tr, M3_va, M3_te)]:
        assert Mt.shape[1] == N_FEATURES, f"{nm} train"
        assert Mv.shape[1] == N_FEATURES, f"{nm} val"
        assert Me.shape[1] == N_FEATURES, f"{nm} test"
        assert np.array_equal(Me[:, :N_GLOBAL], M0_te[:, :N_GLOBAL])
    assert np.allclose(Ftrva_global, M0_tr[:, :N_GLOBAL])

    het_nonzero = float((Fte_het_m1 != 0).mean())
    assert het_nonzero > 0, "M1 heterogeneity all zeros"

    # ---- classifier: fit train+val, ONE test evaluation per variant ----
    results, all_preds = {}, {}
    for name, F_tr, F_va, F_te_v in [
        ("M0", M0_tr, M0_va, M0_te),
        ("M1", M1_tr, M1_va, M1_te),
        ("M2", M2_tr, M2_va, M2_te),
        ("M3", M3_tr, M3_va, M3_te),
    ]:
        ridge = RidgeClassifierCV(alphas=ALPHAS)
        ridge.fit(F_tr, ytrva)              # canonical train+validation fit
        pred_va = ridge.predict(F_va)
        pred_te = ridge.predict(F_te_v)     # SAME variant's test features
        results[name] = {
            "val_macro_f1": round(macro_f1(yva, pred_va), 4),
            "test_macro_f1": round(macro_f1(yte, pred_te), 4),
            "accuracy": round(float(accuracy_score(yte, pred_te)), 4),
            "selected_alpha": float(ridge.alpha_),
            "class_f1s": [round(float(x), 4) for x in f1_score(
                yte, pred_te, average=None, zero_division=0,
                labels=list(range(data["n_classes"])))],
        }
        all_preds[name] = pred_te
        print(f"    {name}: val={results[name]['val_macro_f1']:.4f} "
              f"test={results[name]['test_macro_f1']:.4f} "
              f"alpha={ridge.alpha_:.6f}", flush=True)

    regime_diag = {
        "train": compute_regime_occupancy_stats(regimes_tr, K=K_CODES),
        "val": compute_regime_occupancy_stats(regimes_va, K=K_CODES),
        "test": compute_regime_occupancy_stats(regimes_te, K=K_CODES),
        "drtn": drtn_info, "drtn_frozen_verified": bool(drtn_frozen),
    }

    m0c = all_preds["M0"] == yte
    m1c = all_preds["M1"] == yte
    m3c = all_preds["M3"] == yte
    complementarity = {
        "M0_M1": {"both_correct": int((m0c & m1c).sum()),
                  "M0_only": int((m0c & ~m1c).sum()),
                  "M1_only": int((~m0c & m1c).sum()),
                  "both_wrong": int((~m0c & ~m1c).sum())},
        "M0_M3": {"both_correct": int((m0c & m3c).sum()),
                  "M0_only": int((m0c & ~m3c).sum()),
                  "M3_only": int((~m0c & m3c).sum()),
                  "both_wrong": int((~m0c & ~m3c).sum())},
    }

    seed_result = {
        "seed": seed, "results": results,
        "deltas": {
            "M1_M0": round(results["M1"]["test_macro_f1"] - results["M0"]["test_macro_f1"], 4),
            "M1_M2": round(results["M1"]["test_macro_f1"] - results["M2"]["test_macro_f1"], 4),
            "M1_M3": round(results["M1"]["test_macro_f1"] - results["M3"]["test_macro_f1"], 4),
            "M2_M3": round(results["M2"]["test_macro_f1"] - results["M3"]["test_macro_f1"], 4),
        },
        "regime_diagnostics": regime_diag,
        "complementarity": complementarity,
        "feature_check": {
            "heterogeneity_nonzero_rate": het_nonzero,
            "heterogeneity_mean": float(Fte_het_m1.mean()),
            "heterogeneity_std": float(Fte_het_m1.std()),
            "raw_extractor_vs_aeon_max_diff": mr_identity,
            "m2_m3_feature_diff": m2m3_feature_diff,
            "independent_recompute": recompute_checks,
        },
    }

    with open(os.path.join(seed_dir, "result.json"), "w") as f:
        json.dump(seed_result, f, indent=2)
    with open(os.path.join(seed_dir, "predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_index", "true_class",
                    "M0_pred", "M1_pred", "M2_pred", "M3_pred",
                    "M0_correct", "M1_correct", "M2_correct", "M3_correct"])
        for i in range(len(yte)):
            w.writerow([i, int(yte[i])] +
                       [int(all_preds[v][i]) for v in ["M0", "M1", "M2", "M3"]] +
                       [int(all_preds[v][i] == yte[i]) for v in ["M0", "M1", "M2", "M3"]])

    print(f"\n  [SUMMARY] Seed {seed}:")
    for v in ["M0", "M1", "M2", "M3"]:
        print(f"    {v} = {results[v]['test_macro_f1']:.4f}")
    print(f"    M1-M0 = {seed_result['deltas']['M1_M0']:+.4f}  "
          f"M1-M3 = {seed_result['deltas']['M1_M3']:+.4f}  "
          f"M1-M2 = {seed_result['deltas']['M1_M2']:+.4f}")
    return seed_result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--train-drtn", action="store_true",
                    help="Train DRTN for seeds without checkpoints, then run")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(BASE_DIR, exist_ok=True)

    print("=" * 70)
    print("STAGE B: DRTN-CONDITIONED MINIROCKET -- ECG5000_BAL 3-SEED CONFIRMATION")
    print("=" * 70)
    data = load_data()
    print(f"  train={len(data['Xtr'])} val={len(data['Xva'])} test={len(data['Xte'])} "
          f"T={data['L']} classes={data['n_classes']} val_source={data['val_source']}")

    if args.train_drtn:
        for seed in args.seeds:
            if not os.path.exists(drtn_ckpt_path(seed)):
                print(f"\n[DRTN] training seed {seed}...")
                ckpt, info = train_drtn_for_seed(seed, data, device, smoke=args.smoke)
                os.makedirs(os.path.dirname(drtn_ckpt_path(seed)), exist_ok=True)
                torch.save(ckpt, drtn_ckpt_path(seed))
                json.dump(info, open(os.path.join(
                    os.path.dirname(drtn_ckpt_path(seed)), "train_result.json"), "w"),
                    indent=2)
                print(f"[DRTN] seed {seed} done: {info}")

    all_results = [run_one_seed(s, data, device, smoke=args.smoke) for s in args.seeds]

    # ---- aggregate ----
    variants = ["M0", "M1", "M2", "M3"]
    aggregate = {}
    for v in variants:
        vals = [r["results"][v]["test_macro_f1"] for r in all_results]
        aggregate[v] = {
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
            "se": round(float(np.std(vals) / np.sqrt(len(vals))), 4),
            "min": round(float(np.min(vals)), 4),
            "max": round(float(np.max(vals)), 4),
            "values": [round(x, 4) for x in vals],
        }
    delta_agg = {}
    for dk in ["M1_M0", "M1_M2", "M1_M3", "M2_M3"]:
        vals = [r["deltas"][dk] for r in all_results]
        delta_agg[dk] = {
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
            "se": round(float(np.std(vals) / np.sqrt(len(vals))), 4),
            "n_positive": int(sum(1 for x in vals if x > 0)),
            "values": [round(x, 4) for x in vals],
        }

    print("\n" + "=" * 70)
    print("AGGREGATE (ECG5000_BAL, 3 seeds)")
    print("=" * 70)
    for v in variants:
        a = aggregate[v]
        print(f"  {v}: {a['mean']:.4f} +/- {a['std']:.4f}  seeds={a['values']}")
    for dk, dv in delta_agg.items():
        print(f"  {dk}: mean={dv['mean']:+.4f} se={dv['se']:.4f} "
              f"n_positive={dv['n_positive']}/3 seeds={dv['values']}")

    report = {
        "title": "DRTN-CONDITIONED MINIROCKET -- ECG5000_BAL 3-SEED CONFIRMATION (Stage B)",
        "dataset": DATASET, "seeds": args.seeds,
        "protocol": {
            "features": N_FEATURES, "n_global": N_GLOBAL,
            "n_heterogeneity": N_HETEROGENEITY, "alphas": [float(a) for a in ALPHAS],
            "classifier": "RidgeClassifierCV", "fit": "train+validation",
            "canonical_M0_reference": CANONICAL_M0_REF,
            "M0_tolerance": M0_TOLERANCE,
            "m2_rng_offset": 900001, "m3_rng_offset": 900002,
            "min_occupancy": 0.01, "K": K_CODES,
        },
        "aggregate": aggregate, "deltas": delta_agg,
        "seed_results": all_results,
    }
    with open(os.path.join(BASE_DIR, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {os.path.join(BASE_DIR, 'report.json')}")
    return report


if __name__ == "__main__":
    main()
