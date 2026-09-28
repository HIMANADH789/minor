"""Context-dependence transfer screen (seed 42): DRTN-conditioned MiniROCKET
on GunPoint, ItalyPowerDemand, FordA (verified Kaggle UCRArchive_2018 copies).

Variants (all 9996 features, RidgeClassifierCV on np.logspace(-4,4,20),
final fit on train+validation, test touched exactly once per variant):
    M0     canonical aeon MiniROCKET
    M1     4998 global + 4998 DRTN-regime heterogeneity (valid region)
    M2     same, per-sample occupancy-matched RANDOM regimes (RNG seed+900001)
    M3     same, per-sample SHUFFLED DRTN regimes      (RNG seed+900002)
    A_SOFT same, soft-assignment conditional rates H_soft (frozen codebook,
           softmax(-d2/tau), tau = configured value; no architecture change)

Audited pieces are imported unchanged:
    * controls + occupancy stats: drtn_conditioned_minirocket_haptics_3seed.runner
    * raw extractor / canonical PPV / heterogeneity: transfer_seed42.core
    * DRTN training constants: transfer_seed42.runner (protocol copied verbatim
      into train_or_load_drtn so checkpoints live in THIS namespace's results)

Usage:
    python -m experiments.drtn_conditioned_minirocket_context3_seed42.runner [--smoke]
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.drtn_conditioned_minirocket_haptics_3seed.runner import (  # noqa: E402
    create_random_regime_control,     # audited: per-sample occupancy, seed+900001
    create_shuffled_regime_control,   # audited: per-sample permutation, seed+900002
    compute_regime_heterogeneity,     # audited: valid-region H_m
    compute_regime_occupancy_stats,
    M2_RNG_OFFSET, M3_RNG_OFFSET,
)
from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (  # noqa: E402
    compute_raw_activations, ppv_from_activations,
    independent_heterogeneity_recompute,
)
import experiments.drtn_conditioned_minirocket_transfer_seed42.runner as transfer  # noqa: E402
from experiments.drtn_conditioned_minirocket_context3_seed42.core import (  # noqa: E402
    DATASET_SPECS, load_kaggle_ucr, znorm,
    extract_soft_assignments, soft_heterogeneity_features,
    independent_soft_recompute,
)
from experiments.drtn_conditioned_minirocket_transfer_seed42.runner import (  # noqa: E402
    DRTN_CFG, BATCH_SIZE, LR, WD, MAX_EPOCHS, PATIENCE, set_seed,
)
from models.drtn.model import build_model  # noqa: E402

SEED = 42
N_FEATURES = 9996
N_GLOBAL = N_FEATURES // 2
N_HETEROGENEITY = N_FEATURES - N_GLOBAL
ALPHAS = np.logspace(-4, 4, 20)
K_CODES = 8

DATASETS = ["GunPoint", "ItalyPowerDemand", "FordA"]
OUT_DIR = os.path.join(ROOT, "results", "drtn_conditioned_minirocket_context3_seed42")
SOFT_TAU = DRTN_CFG["tau"]   # frozen-config tau for the softmax(-d2/tau) conversion


def arr_hash(a):
    return hashlib.sha1(np.ascontiguousarray(a, dtype=np.int64).tobytes()).hexdigest()[:16]


def macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# DRTN training -- protocol copied verbatim from the audited transfer runner
# (same constants, same config, same selection rule); only the checkpoint
# location differs so artifacts stay inside this namespace.
# ---------------------------------------------------------------------------
def drtn_checkpoint_path(ds_name):
    return os.path.join(OUT_DIR, ds_name, "drtn_R5_seed42_checkpoint.pt")


def train_or_load_drtn(ds_name, data, device, smoke=False):
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
        return model, {"best_val_mf1": ck["best_val_mf1"], "epoch": ck["epoch"],
                       "source": "checkpoint"}

    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    max_ep = 6 if smoke else MAX_EPOCHS
    patience = 2 if smoke else PATIENCE
    set_seed(SEED)

    mk = lambda X, y, sh: DataLoader(
        TensorDataset(torch.from_numpy(znorm(X))[:, None, :],
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
def run_dataset(ds_name, device, smoke=False):
    log(f"\n{'='*74}\nDATASET {ds_name} (context3 screen, seed {SEED})\n{'='*74}")
    ds_dir = os.path.join(OUT_DIR, ds_name)
    diag_dir = os.path.join(ds_dir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)

    # ---------------- data: verified Kaggle UCR + canonical val split -------
    data = load_kaggle_ucr(ds_name, seed=SEED)
    T = data["L"]
    Xtr, Xva, Xte = data["Xtr"], data["Xva"], data["Xte"]
    ytr, yva, yte = data["ytr"], data["yva"], data["yte"]
    ytrva = np.concatenate([ytr, yva])
    log(f"  source={data['source_path']} train={len(Xtr)} val={len(Xva)} "
        f"test={len(Xte)} T={T} n_cls={data['n_classes']} "
        f"labels={data['label_set_original']} val_source={data['val_source']}")
    with open(os.path.join(diag_dir, "dataset_verification.json"), "w") as f:
        json.dump({k: v for k, v in data.items()
                   if k not in ("Xtr", "Xva", "Xte", "ytr", "yva", "yte")}, f,
                  indent=2, default=str)

    Xtr_z, Xva_z, Xte_z = znorm(Xtr), znorm(Xva), znorm(Xte)

    drtn, drtn_info = train_or_load_drtn(ds_name, data, device, smoke=smoke)
    ck_path = drtn_checkpoint_path(ds_name)
    ck_sha = hashlib.sha1(open(ck_path, "rb").read()).hexdigest()[:16] \
        if os.path.exists(ck_path) else None

    from experiments.drtn_conditioned_minirocket_haptics_3seed.runner import (
        extract_drtn_regimes)
    regimes_tr = extract_drtn_regimes(drtn, Xtr_z, device=device)
    regimes_va = extract_drtn_regimes(drtn, Xva_z, device=device)
    regimes_te = extract_drtn_regimes(drtn, Xte_z, device=device)
    assert regimes_tr.shape == (len(Xtr), T)
    assert regimes_te.min() >= 0 and regimes_te.max() < K_CODES

    sd1 = {k: v.clone() for k, v in drtn.state_dict().items()}
    _ = extract_drtn_regimes(drtn, Xte_z[:2], device=device)
    _ = extract_soft_assignments(drtn, Xte_z[:2], device=device, tau=SOFT_TAU)
    drtn_frozen = all(torch.equal(sd1[k], drtn.state_dict()[k]) for k in sd1)
    assert drtn_frozen, "DRTN parameters changed during regime extraction"

    regime_diag = {
        "train": compute_regime_occupancy_stats(regimes_tr, K=K_CODES),
        "val": compute_regime_occupancy_stats(regimes_va, K=K_CODES),
        "test": compute_regime_occupancy_stats(regimes_te, K=K_CODES),
        "drtn": drtn_info, "drtn_checkpoint_sha1": ck_sha,
        "drtn_frozen_verified": bool(drtn_frozen),
    }

    # ---------------- canonical MiniROCKET (z-normalized input) -------------
    from aeon.transformations.collection.convolution_based import MiniRocket
    extractor = MiniRocket(random_state=SEED, n_jobs=-1)
    Xtrva_z = np.vstack([Xtr_z, Xva_z])
    extractor.fit(Xtr_z[:, None, :].astype(np.float32))
    Ftr = extractor.transform(Xtr_z[:, None, :].astype(np.float32))
    Fva = extractor.transform(Xva_z[:, None, :].astype(np.float32))
    Fte = extractor.transform(Xte_z[:, None, :].astype(np.float32))
    Ftrva = extractor.transform(Xtrva_z[:, None, :].astype(np.float32))
    assert Ftr.shape[1] == N_FEATURES, f"MiniRocket gave {Ftr.shape[1]} features"
    # M0 smoke reproduction: aeon transform is deterministic under seed 42
    Ftrva_re = extractor.transform(Xtrva_z[:, None, :].astype(np.float32))
    m0_deterministic = bool(np.array_equal(Ftrva, Ftrva_re))

    # ---------------- raw activations (valid-region convention) -------------
    # Memory strategy (FordA's train+val activation tensor would need ~21 GiB):
    # test/val activations are materialized once; all TRAIN+VAL quantities
    # (PPV identity check + the four heterogeneity blocks) are computed in
    # bounded sample-chunks. H and PPV are row-independent across samples,
    # so the chunked results are mathematically identical to a full pass.
    act_va, valid = compute_raw_activations(extractor, Xva_z)
    act_te, _ = compute_raw_activations(extractor, Xte_z)

    log(f"  [AUDIT3] global block = features [0, {N_GLOBAL}), "
        f"heterogeneity block = features [{N_GLOBAL}, {N_FEATURES})")
    act_h = lambda a: a[:, N_GLOBAL:, :]          # feature axis only (axis 1)
    valid_het = valid[N_GLOBAL:]
    assert valid_het.shape == (N_HETEROGENEITY, T)

    # ================= AUDIT 5: M2/M3 regime arrays =========================
    regimes_trva = np.vstack([regimes_tr, regimes_va])
    m2_trva = create_random_regime_control(regimes_trva, seed=SEED)
    m2_va = create_random_regime_control(regimes_va, seed=SEED)
    m2_te = create_random_regime_control(regimes_te, seed=SEED)
    m3_trva = create_shuffled_regime_control(regimes_trva, seed=SEED)
    m3_va = create_shuffled_regime_control(regimes_va, seed=SEED)
    m3_te = create_shuffled_regime_control(regimes_te, seed=SEED)

    for nm, a, b in [("trva", m2_trva, m3_trva), ("va", m2_va, m3_va),
                     ("te", m2_te, m3_te)]:
        assert not np.array_equal(a, b), f"M2 == M3 arrays on {nm}!"
        assert not np.shares_memory(a, b), f"M2/M3 share memory on {nm}!"
    # ================= AUDIT 4: per-sample occupancy preservation ===========
    occ_fail = 0
    for ctrl, src in [(m2_trva, regimes_trva), (m3_trva, regimes_trva),
                      (m2_va, regimes_va), (m3_va, regimes_va),
                      (m2_te, regimes_te), (m3_te, regimes_te)]:
        for i in range(len(src)):
            if not np.array_equal(np.bincount(ctrl[i], minlength=K_CODES),
                                  np.bincount(src[i], minlength=K_CODES)):
                occ_fail += 1
    assert occ_fail == 0, f"occupancy preservation failed on {occ_fail} samples"
    log(f"  [AUDIT4] per-sample occupancy preserved: all splits, 0 failures")

    m2m3_diff = {
        "m2_hash": arr_hash(m2_te), "m3_hash": arr_hash(m3_te),
        "n_positions": int(m2_te.size),
        "n_different_positions": int((m2_te != m3_te).sum()),
        "fraction_different": round(float((m2_te != m3_te).mean()), 6),
        "shared_memory": bool(np.shares_memory(m2_te, m3_te)),
        "rng_offsets": {"M2": M2_RNG_OFFSET, "M3": M3_RNG_OFFSET},
    }
    log(f"  [AUDIT5] M2 hash {m2m3_diff['m2_hash']} vs M3 hash "
        f"{m2m3_diff['m3_hash']}; differing positions "
        f"{m2m3_diff['n_different_positions']}/{m2m3_diff['n_positions']}")

    # ================= AUDIT 1 (chunked) + trainva heterogeneity blocks =====
    soft_trva = extract_soft_assignments(drtn, Xtrva_z, device=device, tau=SOFT_TAU)
    n_trva = Xtrva_z.shape[0]
    H1_trva = np.empty((n_trva, N_HETEROGENEITY), dtype=np.float64)
    H2_trva = np.empty_like(H1_trva)
    H3_trva = np.empty_like(H1_trva)
    H4_trva = np.empty_like(H1_trva)
    mr_identity, CHUNK = 0.0, 128
    for c0 in range(0, n_trva, CHUNK):
        c1 = min(c0 + CHUNK, n_trva)
        act_chunk, _ = compute_raw_activations(extractor, Xtrva_z[c0:c1])
        mr_identity = max(mr_identity, float(np.max(np.abs(
            ppv_from_activations(act_chunk, valid) - Ftrva[c0:c1]))))
        a_h = act_h(act_chunk)
        H1_trva[c0:c1] = compute_regime_heterogeneity(
            a_h, valid_het, regimes_trva[c0:c1])
        H2_trva[c0:c1] = compute_regime_heterogeneity(
            a_h, valid_het, m2_trva[c0:c1])
        H3_trva[c0:c1] = compute_regime_heterogeneity(
            a_h, valid_het, m3_trva[c0:c1])
        H4_trva[c0:c1] = soft_heterogeneity_features(
            a_h, valid_het, soft_trva[c0:c1])
        del act_chunk
    log(f"  [AUDIT1] raw-extractor PPV vs aeon (trainva, chunked): "
        f"max|diff|={mr_identity:.2e}; aeon transform deterministic: "
        f"{m0_deterministic}")
    assert mr_identity < 1e-5, "raw extractor disagrees with canonical aeon"
    assert m0_deterministic, "aeon transform not deterministic"
    del soft_trva

    # ================= AUDIT 2 + 6 + 7: heterogeneity path (test) ===========
    H1_te = compute_regime_heterogeneity(act_h(act_te), valid_het, regimes_te)
    H2_te = compute_regime_heterogeneity(act_h(act_te), valid_het, m2_te)
    H3_te = compute_regime_heterogeneity(act_h(act_te), valid_het, m3_te)

    recompute_checks = {}
    for i_s, f_off in [(0, 0), (min(5, len(act_te) - 1), 1234),
                       (min(9, len(act_te) - 1), 4997)]:
        h_impl = float(H1_te[i_s, f_off])
        h_ref = independent_heterogeneity_recompute(
            act_h(act_te)[i_s, f_off], valid_het[f_off], regimes_te[i_s])
        recompute_checks[f"sample{i_s}_kernel{f_off}"] = {
            "implemented": h_impl, "independent": h_ref,
            "abs_diff": abs(h_impl - h_ref)}
        assert abs(h_impl - h_ref) < 1e-7, f"H_m mismatch at ({i_s},{f_off})"
    log(f"  [AUDIT2/7] independent H_m recompute: "
        f"max diff {max(v['abs_diff'] for v in recompute_checks.values()):.2e}")

    # valid-region invariance (AUDIT 6): per-feature padding -- flipping
    # activations OUTSIDE each feature's own valid mask must never change H_m.
    # Processed in sample-chunks to stay within memory on large datasets
    # (mathematically identical: H is row-independent across samples).
    vr_invariance = True
    n_flipped = 0
    CHUNK = 128
    for c0 in range(0, act_te.shape[0], CHUNK):
        act_chunk = act_te[c0:c0 + CHUNK].copy()
        het_act = act_chunk[:, N_GLOBAL:, :]
        outside = np.broadcast_to(~valid_het[None, :, :], het_act.shape)
        het_act[outside] = ~het_act[outside]
        n_flipped += int(outside.sum())
        if n_flipped:
            H1_c = compute_regime_heterogeneity(
                act_h(act_chunk), valid_het, regimes_te[c0:c0 + CHUNK])
            if not np.array_equal(H1_te[c0:c0 + CHUNK], H1_c):
                vr_invariance = False
                break
        del act_chunk
    assert vr_invariance, "padded positions influenced H_m (AUDIT 6 failure)"
    log(f"  [AUDIT6] valid-region invariance: flipped {n_flipped} "
        f"out-of-valid activations -> H unchanged: {vr_invariance}")

    # ================= A_SOFT: test/val soft-regime heterogeneity ==========
    soft_va = extract_soft_assignments(drtn, Xva_z, device=device, tau=SOFT_TAU)
    soft_te = extract_soft_assignments(drtn, Xte_z, device=device, tau=SOFT_TAU)
    H4_va = soft_heterogeneity_features(act_h(act_va), valid_het, soft_va)
    H4_te = soft_heterogeneity_features(act_h(act_te), valid_het, soft_te)

    soft_checks = {}
    for i_s, f_off in [(0, 0), (min(5, len(act_te) - 1), 1234)]:
        h_impl = float(H4_te[i_s, f_off])
        h_ref = independent_soft_recompute(
            act_h(act_te)[i_s, f_off], valid_het[f_off], soft_te[i_s])
        soft_checks[f"sample{i_s}_kernel{f_off}"] = {
            "implemented": h_impl, "independent": h_ref,
            "abs_diff": abs(h_impl - h_ref)}
        assert abs(h_impl - h_ref) < 1e-7, f"H_soft mismatch at ({i_s},{f_off})"
    soft_nontrivial = {
        "nonzero_fraction": float((H4_te != 0).mean()),
        "max": float(H4_te.max()), "mean": float(H4_te.mean()),
    }
    assert soft_nontrivial["nonzero_fraction"] > 0.01, \
        "H_soft is (near-)placeholder: non-triviality check failed"
    assert not np.allclose(H4_te, H1_te), "H_soft identical to hard H"
    log(f"  [AUDIT-SOFT] H_soft recompute max diff "
        f"{max(v['abs_diff'] for v in soft_checks.values()):.2e}; "
        f"nonzero fraction {soft_nontrivial['nonzero_fraction']:.3f}")

    # ================= AUDIT 3: feature budget / allocation =================
    M0_tr, M0_va, M0_te = Ftrva, Fva, Fte
    M1_tr = np.hstack([Ftrva[:, :N_GLOBAL], H1_trva])
    M1_va = np.hstack([Fva[:, :N_GLOBAL],
                       compute_regime_heterogeneity(act_h(act_va), valid_het, regimes_va)])
    M1_te = np.hstack([Fte[:, :N_GLOBAL], H1_te])
    M2_tr = np.hstack([Ftrva[:, :N_GLOBAL], H2_trva])
    M2_va = np.hstack([Fva[:, :N_GLOBAL],
                       compute_regime_heterogeneity(act_h(act_va), valid_het, m2_va)])
    M2_te = np.hstack([Fte[:, :N_GLOBAL], H2_te])
    M3_tr = np.hstack([Ftrva[:, :N_GLOBAL], H3_trva])
    M3_va = np.hstack([Fva[:, :N_GLOBAL],
                       compute_regime_heterogeneity(act_h(act_va), valid_het, m3_va)])
    M3_te = np.hstack([Fte[:, :N_GLOBAL], H3_te])
    M4_tr = np.hstack([Ftrva[:, :N_GLOBAL], H4_trva])
    M4_va = np.hstack([Fva[:, :N_GLOBAL], H4_va])
    M4_te = np.hstack([Fte[:, :N_GLOBAL], H4_te])

    feature_diffs = {
        f"H{a}vsH{b}": {
            "max_abs_diff": float(np.abs(x - y).max()),
            "mean_abs_diff": float(np.abs(x - y).mean()),
            "n_exactly_equal": int((x == y).sum()), "n_elements": int(x.size)}
        for a, b, x, y in [("1", "2", H1_te, H2_te), ("1", "3", H1_te, H3_te),
                           ("2", "3", H2_te, H3_te), ("1", "4", H1_te, H4_te),
                           ("2", "4", H2_te, H4_te), ("3", "4", H3_te, H4_te)]
    }
    assert not np.array_equal(H2_te, H3_te), "M2/M3 features identical"
    assert (H1_te != 0).mean() > 0, "M1 heterogeneity all zeros"

    variants = [("M0", M0_tr, M0_va, M0_te), ("M1", M1_tr, M1_va, M1_te),
                ("M2", M2_tr, M2_va, M2_te), ("M3", M3_tr, M3_va, M3_te),
                ("A_SOFT", M4_tr, M4_va, M4_te)]
    for nm, Mt, Mv, Me in variants:
        assert Mt.shape[1] == N_FEATURES and Mv.shape[1] == N_FEATURES \
            and Me.shape[1] == N_FEATURES, f"{nm} budget"
        assert np.array_equal(Me[:, :N_GLOBAL], M0_te[:, :N_GLOBAL]), f"{nm} global"
    assert np.allclose(M0_tr[:, :N_GLOBAL], Ftrva[:, :N_GLOBAL])
    log(f"  [AUDIT3] budgets: M0/M1/M2/M3/A_SOFT all 9996 = 4998 + 4998; "
        f"global blocks identical to M0")

    # ================= classifier: train+val fit, test once =================
    results, preds = {}, {}
    for name, F_tr, F_va, F_te_v in variants:
        ridge = RidgeClassifierCV(alphas=ALPHAS)
        ridge.fit(F_tr, ytrva)               # train + validation
        pred_va = ridge.predict(F_va)
        pred_te = ridge.predict(F_te_v)      # same variant's test features
        results[name] = {
            "val_macro_f1": round(macro_f1(yva, pred_va), 4),
            "test_macro_f1": round(macro_f1(yte, pred_te), 4),
            "accuracy": round(float(accuracy_score(yte, pred_te)), 4),
            "selected_alpha": float(ridge.alpha_),
            "class_f1s": [round(float(x), 4) for x in f1_score(
                yte, pred_te, average=None, zero_division=0,
                labels=list(range(data["n_classes"])))],
        }
        preds[name] = pred_te
        log(f"    {name}: val={results[name]['val_macro_f1']:.4f} "
            f"test={results[name]['test_macro_f1']:.4f} "
            f"alpha={ridge.alpha_:.6f}")

    deltas = {
        f"{a}_minus_{b}": round(results[a]["test_macro_f1"]
                                - results[b]["test_macro_f1"], 4)
        for a, b in [("M1", "M0"), ("M1", "M2"), ("M1", "M3"),
                     ("M2", "M3"), ("M1", "A_SOFT"), ("A_SOFT", "M0")]}

    # ---------------- predictions + feature dims ----------------------------
    os.makedirs(os.path.join(OUT_DIR, "predictions"), exist_ok=True)
    with open(os.path.join(OUT_DIR, "predictions", f"{ds_name}_seed42.csv"),
              "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_index", "true_class", "M0_pred", "M1_pred",
                    "M2_pred", "M3_pred", "A_SOFT_pred"])
        for i in range(len(yte)):
            w.writerow([i, int(yte[i])] + [int(preds[v][i])
                        for v in ["M0", "M1", "M2", "M3", "A_SOFT"]])

    with open(os.path.join(diag_dir, "feature_dimensions.json"), "w") as f:
        json.dump({nm: {"train": list(Mt.shape), "val": list(Mv.shape),
                        "test": list(Me.shape)}
                   for nm, Mt, Mv, Me in variants}, f, indent=2)

    het_stats = {
        v: {"mean": float(h.mean()), "std": float(h.std()),
            "nonzero_fraction": float((h != 0).mean()), "max": float(h.max())}
        for v, h in [("M1", H1_te), ("M2", H2_te), ("M3", H3_te),
                     ("A_SOFT", H4_te)]}

    result = {
        "dataset": ds_name, "seed": SEED,
        "protocol": {"source": data["source_path"],
                     "train": len(Xtr), "val": len(Xva), "test": len(Xte),
                     "trainva_fit": int(len(ytrva)), "T": T,
                     "n_classes": data["n_classes"],
                     "label_set_original": data["label_set_original"],
                     "label_map": data["label_map"],
                     "val_source": data["val_source"],
                     "file_sha256": data["file_sha256"]},
        "results": results, "deltas": deltas,
        "m0_smoke": {"n_features": N_FEATURES,
                     "aeon_transform_deterministic": m0_deterministic,
                     "extractor_identity_max_diff": mr_identity},
        "regime_diagnostics": regime_diag,
        "audits": {
            "mr_identity_max_diff": mr_identity,
            "occupancy_failures": occ_fail,
            "m2_m3_arrays": m2m3_diff,
            "feature_diffs": feature_diffs,
            "independent_recompute": recompute_checks,
            "soft_recompute": soft_checks,
            "soft_nontrivial": soft_nontrivial,
            "valid_region_invariance": vr_invariance,
        },
        "heterogeneity_stats": het_stats,
        "context_dependence": {
            "H1_mean_over_H2_mean": float(H1_te.mean() / max(H2_te.mean(), 1e-12)),
            "H1_mean_over_H3_mean": float(H1_te.mean() / max(H3_te.mean(), 1e-12)),
            "H4_mean_over_H1_mean": float(H4_te.mean() / max(H1_te.mean(), 1e-12)),
        },
    }
    with open(os.path.join(ds_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(diag_dir, "regime_occupancy.json"), "w") as f:
        json.dump({k: v for k, v in regime_diag.items()
                   if k in ("train", "val", "test")}, f, indent=2)

    log(f"\n  [SUMMARY] {ds_name}: "
        + " ".join(f"{v}={results[v]['test_macro_f1']:.4f}"
                   for v in ["M0", "M1", "M2", "M3", "A_SOFT"]))
    log(f"            deltas: {deltas}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    log("=" * 74)
    log("DRTN-CONDITIONED MINIROCKET -- CONTEXT-DEPENDENCE SCREEN (seed 42)")
    log("Datasets: verified Kaggle UCRArchive_2018 copies (GunPoint, "
        "ItalyPowerDemand, FordA)")
    log("=" * 74)

    all_results = [run_dataset(ds, device, smoke=args.smoke)
                   for ds in args.datasets]

    log("\n" + "=" * 74)
    log("RESULTS (seed 42, test Macro-F1)")
    log("=" * 74)
    hdr = f"{'dataset':<18}" + "".join(f"{v:>8}" for v in
                                       ["M0", "M1", "M2", "M3", "A_SOFT"])
    log(hdr)
    for r in all_results:
        log(f"{r['dataset']:<18}"
            + "".join(f"{r['results'][v]['test_macro_f1']:>8.4f}"
                      for v in ["M0", "M1", "M2", "M3", "A_SOFT"]))

    report = {
        "title": "DRTN-conditioned MiniROCKET context-dependence screen",
        "seed": SEED, "datasets": args.datasets,
        "implementation": "audited Stage A (per-sample-occupancy M2/M3, "
                          "valid-region H) + A_SOFT soft-assignment ablation",
        "results": all_results,
    }
    with open(os.path.join(OUT_DIR, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    log(f"\nSaved: {os.path.join(OUT_DIR, 'report.json')}")
    return report


if __name__ == "__main__":
    main()
