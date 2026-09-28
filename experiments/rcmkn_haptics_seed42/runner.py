"""RCMKN first experiment: Haptics, seed 42, full ablation ladder.

Variants (every variant's test set is touched EXACTLY ONCE):
    R0  audited DRTN-conditioned MiniROCKET reference (frozen official
        checkpoint; 4998 global + 4998 heterogeneity) -- reproduction gate
    R1  R0 + Hydra competitive/count block (2048 features; 12044 total)
    R2  new SSL causal encoder + HardVQ context -> heterogeneity
        (4998 global + 4998 heterogeneity)
    R3  full RCMKN: R2 context + Hydra (12044 total)
    C1  R2 with occupancy-matched random regimes (audited M2 construction)
    C2  R2 with per-sample shuffled regimes (audited M3 construction)

Usage:
    python -m experiments.rcmkn_haptics_seed42.runner [--smoke]
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
import torch.nn.functional as F
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.external_stack_generalization.data import load_dataset  # noqa: E402
from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (  # noqa: E402
    compute_raw_activations, ppv_from_activations,
    independent_heterogeneity_recompute,
)
from experiments.drtn_conditioned_minirocket_haptics_3seed.runner import (  # noqa: E402
    create_random_regime_control, create_shuffled_regime_control,
    compute_regime_heterogeneity, compute_regime_occupancy_stats,
    extract_drtn_regimes, M2_RNG_OFFSET, M3_RNG_OFFSET,
)
from experiments.rcmkn_haptics_seed42.config import (
    SEED, N_FEATURES, N_GLOBAL, N_HET, K_CODES, ENCODER, VQ, JOINT, HYDRA,
    MIN_OCCUPANCY, R0_REFERENCE, R0_TOLERANCE, versions,
)
from experiments.rcmkn_haptics_seed42.kernel_features import (
    compute_heterogeneity_chunked, compute_hydra_features, hydra_dimension,
    scale_hydra_features,
)
from experiments.rcmkn_haptics_seed42.regime_encoder import count_parameters
from experiments.rcmkn_haptics_seed42.model import RCMKNContextModel, parameter_report
from experiments.rcmkn_haptics_seed42.vq import hard_assign, occupancy_stats

ALPHAS = np.logspace(-4, 4, 20)
OUT_DIR = os.path.join(ROOT, "results", "rcmkn_haptics_seed42")


def log(msg):
    print(msg, flush=True)


def arr_hash(a):
    return hashlib.sha1(np.ascontiguousarray(a, dtype=np.int64).tobytes()).hexdigest()[:16]


def macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# R0: audited DRTN reference (frozen official checkpoint)
# ---------------------------------------------------------------------------
def load_official_drtn(device):
    from models.drtn.model import build_model
    rdir = os.path.join(ROOT, "results", "drtn_haptics_seed42", "R5")
    ck = torch.load(os.path.join(rdir, "checkpoint.pt"),
                    map_location=device, weights_only=False)
    cfg = ck["config"]
    model = build_model("R5", c_in=1, n_classes=5, d_model=cfg["d_model"],
                        n_codes=cfg["n_codes"], tau=cfg["tau"],
                        ema_decay=cfg["ema_decay"], beta=cfg["beta_commit"],
                        lam_div=cfg["lam_div"],
                        dead_threshold=cfg["dead_threshold"],
                        revival_patience=cfg["revival_patience"],
                        traj_layers=cfg["trajectory"]["layers"],
                        traj_heads=cfg["trajectory"]["heads"],
                        traj_ffn=cfg["trajectory"]["ffn"],
                        traj_dropout=cfg["trajectory"]["dropout"])
    model.load_state_dict(ck["model_state"])
    model.to(device).eval()
    return model, ck


# ---------------------------------------------------------------------------
# R2/R3/C1/C2: SSL context model training (train only, val for selection)
# ---------------------------------------------------------------------------
def train_context_model(model, Xtr, ytr, Xva, yva, device, smoke=False):
    from torch.utils.data import DataLoader, TensorDataset
    enc_cfg = dict(ENCODER)
    joint_cfg = dict(JOINT)
    if smoke:
        enc_cfg["ssl_epochs"], enc_cfg["patience"] = 3, 1
        joint_cfg["epochs"], joint_cfg["patience"] = 3, 1

    def dl(X, y, shuffle):
        return DataLoader(
            TensorDataset(torch.from_numpy(X)[:, None, :],
                          torch.from_numpy(np.asarray(y))),
            batch_size=enc_cfg["batch_size"], shuffle=shuffle, num_workers=0,
            generator=torch.Generator().manual_seed(SEED) if shuffle else None)

    tr_dl, va_dl = dl(Xtr, ytr, True), dl(Xva, yva, False)

    def _ssl_batch(xb):
        return xb.to(device).squeeze(1)
    opt = torch.optim.AdamW(model.parameters(), lr=enc_cfg["lr"],
                            weight_decay=enc_cfg["weight_decay"])

    # ---- phase 1: pure SSL (no labels) ----
    best, best_state, no_imp = float("inf"), None, 0
    step = 0
    t0 = time.time()
    for ep in range(enc_cfg["ssl_epochs"]):
        model.train()
        ep_loss = 0.0
        for xb, _y in tr_dl:
            xb = _ssl_batch(xb)
            loss = model.ssl_loss(xb, enc_cfg["mask_ratio"], enc_cfg["span_len"])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); ep_loss += float(loss); step += 1
        model.eval()
        va_loss, nb = 0.0, 0
        with torch.no_grad():
            for xb, _y in va_dl:
                xb = _ssl_batch(xb)
                va_loss += float(model.ssl_loss(xb, enc_cfg["mask_ratio"],
                                                enc_cfg["span_len"]))
                nb += 1
        va_loss /= max(nb, 1)
        if va_loss < best - 1e-6:
            best, no_imp = va_loss, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            no_imp += 1
        if (ep + 1) % 10 == 0 or ep == 0 or smoke:
            log(f"    [SSL] ep{ep+1}/{enc_cfg['ssl_epochs']} "
                f"train={ep_loss:.5f} val={va_loss:.5f} (best {best:.5f}) "
                f"[{time.time()-t0:.0f}s]")
        if no_imp >= enc_cfg["patience"]:
            log(f"    [SSL] early stop ep{ep+1}")
            break
    ssl_best = best
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- phase 2: joint fine-tune (SSL + VQ + div + lambda_cls aux CE) ----
    opt = torch.optim.AdamW(model.parameters(), lr=joint_cfg["lr"],
                            weight_decay=enc_cfg["weight_decay"])
    best_cls, best_state2, no_imp = -1.0, None, 0
    losses = {"commit": 0.0, "div": 0.0, "aux": 0.0, "ssl": 0.0}
    t0 = time.time()
    for ep in range(joint_cfg["epochs"]):
        model.train()
        ep_loss = 0.0
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            logits, z, q_st, assign, commit = model(xb)
            l_ssl = model.ssl_loss(xb.squeeze(1))
            l_div = model.diversity_loss(assign)
            l_aux = F.cross_entropy(logits, yb)
            loss = (l_ssl + VQ["beta_vq"] * commit + VQ["lam_div"] * l_div
                    + joint_cfg["lambda_cls"] * l_aux)
            model.ema_step(z, assign, step); step += 1
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += float(loss)
            losses["commit"] += float(commit); losses["div"] += float(l_div)
            losses["aux"] += float(l_aux); losses["ssl"] += float(l_ssl)
        model.eval()
        preds, tgts = [], []
        with torch.no_grad():
            for xb, yb in va_dl:
                logits, *_ = model(xb.to(device))
                preds.append(logits.argmax(-1).cpu().numpy()); tgts.append(yb.numpy())
        val_mf1 = macro_f1(np.concatenate(tgts), np.concatenate(preds))
        if val_mf1 > best_cls:
            best_cls, no_imp = val_mf1, 0
            best_state2 = {k: v.detach().cpu().clone()
                           for k, v in model.state_dict().items()}
        else:
            no_imp += 1
        if (ep + 1) % 10 == 0 or ep == 0 or smoke:
            log(f"    [JOINT] ep{ep+1}/{joint_cfg['epochs']} "
                f"train={ep_loss:.4f} valMF1={val_mf1:.4f} (best {best_cls:.4f}) "
                f"[{time.time()-t0:.0f}s]")
        if no_imp >= joint_cfg["patience"]:
            log(f"    [JOINT] early stop ep{ep+1}")
            break
    if best_state2 is not None:
        model.load_state_dict(best_state2)
    n_batches = max(1, (len(Xtr) + enc_cfg["batch_size"] - 1) // enc_cfg["batch_size"])
    train_info = {
        "ssl_best_val": round(ssl_best, 6),
        "joint_best_val_mf1": round(best_cls, 4),
        "avg_joint_losses": {k: round(v / n_batches, 5) for k, v in losses.items()},
    }
    return train_info


@torch.no_grad()
def extract_context_regimes(model, X, device, batch=32):
    """Frozen context model -> hard regime sequences (N, T)."""
    model.eval()
    out = []
    for c0 in range(0, len(X), batch):
        xb = torch.from_numpy(X[c0:c0 + batch])[:, None, :].to(device)
        z = model.encoder(xb)
        out.append(hard_assign(model.vq, z).cpu().numpy())
    return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
def run_variant(name, blocks, yva, yte, ytrva, n_classes, diag_dir):
    """Fit Ridge on train+val, evaluate val + test once. blocks: dict of
    variant-local (F_tr, F_va, F_te) per block, concatenated in order."""
    F_tr = np.hstack([b[0] for b in blocks.values()])
    F_va = np.hstack([b[1] for b in blocks.values()])
    F_te = np.hstack([b[2] for b in blocks.values()])
    assert F_tr.shape[1] == F_va.shape[1] == F_te.shape[1], f"{name} budget"
    ridge = RidgeClassifierCV(alphas=ALPHAS)
    ridge.fit(F_tr, ytrva)                       # train + validation
    pred_va = ridge.predict(F_va)
    pred_te = ridge.predict(F_te)                # same variant's test features
    res = {
        "val_macro_f1": round(macro_f1(yva, pred_va), 4),
        "test_macro_f1": round(macro_f1(yte, pred_te), 4),
        "accuracy": round(float(accuracy_score(yte, pred_te)), 4),
        "selected_alpha": float(ridge.alpha_),
        "feature_dim": int(F_tr.shape[1]),
        "class_f1s": [round(float(x), 4) for x in f1_score(
            yte, pred_te, average=None, zero_division=0,
            labels=list(range(n_classes)))],
    }
    log(f"    {name}: val={res['val_macro_f1']:.4f} test={res['test_macro_f1']:.4f} "
        f"alpha={ridge.alpha_:.6f} dim={res['feature_dim']}")
    return res, pred_te, (F_tr.shape, F_va.shape, F_te.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--variants", nargs="+",
                    default=["R0", "R1", "R2", "R3", "C1", "C2"])
    args = ap.parse_args()
    want = set(args.variants)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    log("=" * 74)
    log("RCMKN — Regime-Conditioned Multi-View Kernel Network (Haptics, seed 42)")
    log("=" * 74)

    # ---------------- data (canonical Haptics, untouched split) -------------
    data = load_dataset("Haptics")
    Xtr, ytr = data["Xtr"], data["ytr"]
    Xva, yva = data["Xva"], data["yva"]
    Xte, yte = data["Xte"], data["yte"]
    Xtrva = np.vstack([Xtr, Xva]); ytrva = np.concatenate([ytr, yva])
    T = data["L"]; n_classes = data["n_classes"]
    log(f"  train={len(Xtr)} val={len(Xva)} test={len(Xte)} T={T} "
        f"n_cls={n_classes} val_source={data['val_source']}")

    def znorm(X):
        return ((X - X.mean(-1, keepdims=True)) /
                (X.std(-1, keepdims=True) + 1e-8)).astype(np.float32)

    Xtr_z, Xva_z, Xte_z = znorm(Xtr), znorm(Xva), znorm(Xte)
    Xtrva_z = np.vstack([Xtr_z, Xva_z])

    # ---------------- canonical MiniROCKET (fit on train, per protocol) -----
    from aeon.transformations.collection.convolution_based import MiniRocket
    set_seed(SEED)
    extractor = MiniRocket(random_state=SEED, n_jobs=-1)
    extractor.fit(Xtr_z[:, None, :].astype(np.float32))
    F_trva_mr = extractor.transform(Xtrva_z[:, None, :].astype(np.float32))
    F_va_mr = extractor.transform(Xva_z[:, None, :].astype(np.float32))
    F_te_mr = extractor.transform(Xte_z[:, None, :].astype(np.float32))
    assert F_trva_mr.shape[1] == N_FEATURES
    G_tr, G_va, G_te = (F_trva_mr[:, :N_GLOBAL], F_va_mr[:, :N_GLOBAL],
                        F_te_mr[:, :N_GLOBAL])

    # AUDIT 1: bit-exact extractor identity (chunked, trainva)
    audits = {"control_occupancy_failures": None,
              "hydra_determinism_same_batch": None}
    valid = None
    mr_identity = 0.0
    for c0 in range(0, len(Xtrva_z), 64):
        act, valid = compute_raw_activations(extractor, Xtrva_z[c0:c0 + 64])
        mr_identity = max(mr_identity, float(np.max(np.abs(
            ppv_from_activations(act, valid) - F_trva_mr[c0:c0 + 64]))))
        del act
    log(f"  [AUDIT1] raw-extractor PPV vs aeon: max|diff|={mr_identity:.2e}")
    assert mr_identity < 1e-5
    audits["audit1_extractor_identity_maxdiff"] = float(mr_identity)
    valid_het = valid[N_GLOBAL:]
    act_h = lambda a: a[:, N_GLOBAL:, :]

    hydra_dim, hydra_struct = hydra_dimension(T, k=HYDRA["k"], g=HYDRA["g"])
    log(f"  [HYDRA] structure: {hydra_struct} -> features={hydra_dim}")

    results, preds, meta = {}, {}, {}

    # ================= R0: audited DRTN reference ===========================
    if "R0" in want:
        log("\n[R0] audited DRTN-conditioned MiniROCKET (frozen checkpoint)")
        drtn, ck = load_official_drtn(device)
        set_seed(SEED)
        regimes_trva = extract_drtn_regimes(drtn, Xtrva_z, device=device)
        regimes_te = extract_drtn_regimes(drtn, Xte_z, device=device)
        sd1 = {k: v.clone() for k, v in drtn.state_dict().items()}
        _ = extract_drtn_regimes(drtn, Xte_z[:2], device=device)
        assert all(torch.equal(sd1[k], drtn.state_dict()[k]) for k in sd1)
        H_tr = compute_heterogeneity_chunked(extractor, Xtrva_z, regimes_trva,
                                             valid_het, act_h=act_h)
        act_va, _ = compute_raw_activations(extractor, Xva_z)
        act_te, _ = compute_raw_activations(extractor, Xte_z)
        H_va = compute_regime_heterogeneity(act_h(act_va), valid_het, regimes_trva[len(Xtr):])
        H_te = compute_regime_heterogeneity(act_h(act_te), valid_het, regimes_te)
        del act_va, act_te
        blocks = {"global": (G_tr, G_va, G_te), "het": (H_tr, H_va, H_te)}
        res, pred_te, shapes = run_variant("R0", blocks, yva, yte, ytrva,
                                           n_classes, OUT_DIR)
        results["R0"], preds["R0"], meta["R0"] = res, pred_te, {"shapes": shapes}
        # ---- R0 GATE ----
        gate = abs(res["test_macro_f1"] - R0_REFERENCE) <= R0_TOLERANCE
        log(f"  [R0 GATE] {res['test_macro_f1']} vs reference {R0_REFERENCE} "
            f"-> {'PASS' if gate else 'FAIL'}")
        audits["r0_gate"] = {"test_macro_f1": res["test_macro_f1"],
                             "reference": R0_REFERENCE, "pass": bool(gate)}
        assert gate, "R0 failed to reproduce the audited reference. STOP."
        # diagnostics for figures
        results["R0"]["regime_diagnostics"] = {
            "train": occupancy_stats(regimes_trva), "test": occupancy_stats(regimes_te)}
        np.save(os.path.join(OUT_DIR, "_regimes_R0_te.npy"), regimes_te[:20])

    # ================= Hydra features (shared by R1/R3) =====================
    H_d_tr = H_d_va = H_d_te = None
    if {"R1", "R3"} & want:
        log("\n[HYDRA] competitive/count features (k=8, g=16)")
        Hd_tr, hydra_mod = compute_hydra_features(Xtrva_z, T, k=HYDRA["k"], g=HYDRA["g"])
        Hd_va, _ = compute_hydra_features(Xva_z, T, k=HYDRA["k"], g=HYDRA["g"])
        Hd_te, _ = compute_hydra_features(Xte_z, T, k=HYDRA["k"], g=HYDRA["g"])
        assert Hd_tr.shape[1] == hydra_dim == Hd_te.shape[1]
        # canonical Hydra scaling, fit on train+val only; applied to the
        # Hydra block only (PPV blocks keep their audited scale)
        H_d_tr, H_d_va, H_d_te = scale_hydra_features(Hd_tr, Hd_va, Hd_te)
        # AUDIT: determinism under fixed torch seed (same-batch recompute
        # must be exact; cross-batch float32 reduction order differs at
        # ulp level only, so subset comparison uses a tight tolerance)
        set_seed(SEED)
        Hd_te2, _ = compute_hydra_features(Xte_z, T, k=HYDRA["k"], g=HYDRA["g"])
        audits["hydra_determinism_same_batch"] = bool(np.array_equal(Hd_te2, Hd_te))
        assert audits["hydra_determinism_same_batch"], "Hydra not deterministic (same batch)"
        set_seed(SEED)
        Hd_sub, _ = compute_hydra_features(Xte_z[:16], T, k=HYDRA["k"], g=HYDRA["g"])
        audits["hydra_cross_batch_maxdiff"] = float(np.max(np.abs(Hd_sub - Hd_te[:16])))
        assert np.allclose(Hd_sub, Hd_te[:16], atol=1e-4), "Hydra cross-batch drift"
        log(f"  [HYDRA] dim={hydra_dim}, deterministic: True (exact same-batch, "
            f"ulp-level cross-batch)")

    # ================= R1: R0 + Hydra =======================================
    if "R1" in want:
        log("\n[R1] R0 + Hydra")
        # R0 regimes recomputed identically (frozen checkpoint, deterministic)
        drtn, _ = load_official_drtn(device)
        set_seed(SEED)
        regimes_trva = extract_drtn_regimes(drtn, Xtrva_z, device=device)
        regimes_te = extract_drtn_regimes(drtn, Xte_z, device=device)
        H_tr = compute_heterogeneity_chunked(extractor, Xtrva_z, regimes_trva,
                                             valid_het, act_h=act_h)
        act_va, _ = compute_raw_activations(extractor, Xva_z)
        act_te, _ = compute_raw_activations(extractor, Xte_z)
        H_va = compute_regime_heterogeneity(act_h(act_va), valid_het, regimes_trva[len(Xtr):])
        H_te = compute_regime_heterogeneity(act_h(act_te), valid_het, regimes_te)
        del act_va, act_te
        blocks = {"global": (G_tr, G_va, G_te), "het": (H_tr, H_va, H_te),
                  "hydra": (H_d_tr, H_d_va, H_d_te)}
        res, pred_te, shapes = run_variant("R1", blocks, yva, yte, ytrva,
                                           n_classes, OUT_DIR)
        results["R1"], preds["R1"], meta["R1"] = res, pred_te, {"shapes": shapes}
        results["R1"]["hydra_structure"] = hydra_struct

    # ================= R2/R3/C1/C2: SSL context model =======================
    if {"R2", "R3", "C1", "C2"} & want:
        log("\n[SSL] training context model (encoder + HardVQ + decoder)")
        set_seed(SEED)
        model = RCMKNContextModel(n_classes=n_classes).to(device)
        pre_params = parameter_report(model)
        log(f"  parameters: {pre_params}")
        train_info = train_context_model(model, Xtr_z, ytr, Xva_z, yva,
                                         device, smoke=args.smoke)
        log(f"  context training: {train_info}")
        # freeze
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        torch.save({"model_state": model.state_dict(),
                    "config": {"encoder": ENCODER, "vq": VQ, "joint": JOINT},
                    "train_info": train_info, "params": pre_params},
                   os.path.join(OUT_DIR, "context_model_seed42.pt"))

        regimes_trva = extract_context_regimes(model, Xtrva_z, device)
        regimes_te = extract_context_regimes(model, Xte_z, device)
        occ = {"train": occupancy_stats(regimes_trva, K_CODES),
               "test": occupancy_stats(regimes_te, K_CODES)}
        log(f"  [VQ] train occupancy: active={occ['train']['active_codes']} "
            f"entropy={occ['train']['normalized_entropy']:.3f} "
            f"perplexity={occ['train']['perplexity']:.2f} "
            f"dominant={occ['train']['dominant_fraction']:.3f} "
            f"revivals={len(model.vq.revival_log)}")

        H_tr = compute_heterogeneity_chunked(extractor, Xtrva_z, regimes_trva,
                                             valid_het, act_h=act_h)
        act_va, _ = compute_raw_activations(extractor, Xva_z)
        act_te, _ = compute_raw_activations(extractor, Xte_z)
        H_va = compute_regime_heterogeneity(act_h(act_va), valid_het, regimes_trva[len(Xtr):])
        H_te = compute_regime_heterogeneity(act_h(act_te), valid_het, regimes_te)

        if "R2" in want:
            log("\n[R2] SSL context + DRTN-style conditioning (no Hydra)")
            blocks = {"global": (G_tr, G_va, G_te), "het": (H_tr, H_va, H_te)}
            res, pred_te, shapes = run_variant("R2", blocks, yva, yte, ytrva,
                                               n_classes, OUT_DIR)
            results["R2"], preds["R2"], meta["R2"] = res, pred_te, {"shapes": shapes}
            results["R2"]["regime_diagnostics"] = occ
            np.save(os.path.join(OUT_DIR, "_regimes_R2_te.npy"), regimes_te[:20])
            results["R2"]["context_train"] = train_info
            results["R2"]["params"] = pre_params
        if "R3" in want:
            log("\n[R3] full RCMKN (SSL context + conditioning + Hydra)")
            blocks = {"global": (G_tr, G_va, G_te), "het": (H_tr, H_va, H_te),
                      "hydra": (H_d_tr, H_d_va, H_d_te)}
            res, pred_te, shapes = run_variant("R3", blocks, yva, yte, ytrva,
                                               n_classes, OUT_DIR)
            results["R3"], preds["R3"], meta["R3"] = res, pred_te, {"shapes": shapes}
            results["R3"]["regime_diagnostics"] = occ
            results["R3"]["context_train"] = train_info
            results["R3"]["params"] = pre_params
            results["R3"]["hydra_structure"] = hydra_struct

        # ---- C1/C2 controls (audited constructions) ----
        if {"C1", "C2"} & want:
            m2_trva = create_random_regime_control(regimes_trva, seed=SEED)
            m2_te = create_random_regime_control(regimes_te, seed=SEED)
            m3_trva = create_shuffled_regime_control(regimes_trva, seed=SEED)
            m3_te = create_shuffled_regime_control(regimes_te, seed=SEED)
            for nm, a, b in [("C1", m2_trva, m3_trva), ("C2", m2_te, m3_te)]:
                assert not np.array_equal(a, b)
                assert not np.shares_memory(a, b)
            occ_fail = 0
            for ctrl, src in [(m2_trva, regimes_trva), (m3_trva, regimes_trva),
                              (m2_te, regimes_te), (m3_te, regimes_te)]:
                for i in range(len(src)):
                    if not np.array_equal(np.bincount(ctrl[i], minlength=K_CODES),
                                          np.bincount(src[i], minlength=K_CODES)):
                        occ_fail += 1
            assert occ_fail == 0, "control occupancy broken"
            audits["control_occupancy_failures"] = occ_fail
            log("  [CTRL] M2/M3 constructions audited (occupancy 0 failures, "
                "distinct arrays, no shared memory)")

            if "C1" in want:
                log("\n[C1] R2 with occupancy-matched random regimes")
                Hc1_tr = compute_heterogeneity_chunked(extractor, Xtrva_z, m2_trva,
                                                       valid_het, act_h=act_h)
                Hc1_va = compute_regime_heterogeneity(act_h(act_va), valid_het,
                                                      m2_trva[len(Xtr):])
                Hc1_te = compute_regime_heterogeneity(act_h(act_te), valid_het, m2_te)
                blocks = {"global": (G_tr, G_va, G_te),
                          "het": (Hc1_tr, Hc1_va, Hc1_te)}
                res, pred_te, shapes = run_variant("C1", blocks, yva, yte, ytrva,
                                                   n_classes, OUT_DIR)
                results["C1"], preds["C1"], meta["C1"] = res, pred_te, {"shapes": shapes}
                results["C1"]["regime_diagnostics"] = {
                    "train": occupancy_stats(m2_trva, K_CODES),
                    "test": occupancy_stats(m2_te, K_CODES)}
            if "C2" in want:
                log("\n[C2] R2 with per-sample shuffled regimes")
                Hc2_tr = compute_heterogeneity_chunked(extractor, Xtrva_z, m3_trva,
                                                       valid_het, act_h=act_h)
                Hc2_va = compute_regime_heterogeneity(act_h(act_va), valid_het,
                                                      m3_trva[len(Xtr):])
                Hc2_te = compute_regime_heterogeneity(act_h(act_te), valid_het, m3_te)
                blocks = {"global": (G_tr, G_va, G_te),
                          "het": (Hc2_tr, Hc2_va, Hc2_te)}
                res, pred_te, shapes = run_variant("C2", blocks, yva, yte, ytrva,
                                                   n_classes, OUT_DIR)
                results["C2"], preds["C2"], meta["C2"] = res, pred_te, {"shapes": shapes}
                results["C2"]["regime_diagnostics"] = {
                    "train": occupancy_stats(m3_trva, K_CODES),
                    "test": occupancy_stats(m3_te, K_CODES)}
        del act_va, act_te

    # ================= mechanistic diagnostics ==============================
    log("\n[DIAG] heterogeneity stats are stored per variant in result.json")

    # ================= save =================================================
    os.makedirs(os.path.join(OUT_DIR, "predictions"), exist_ok=True)
    with open(os.path.join(OUT_DIR, "predictions", "haptics_seed42.csv"),
              "w", newline="") as f:
        w = csv.writer(f)
        vs = [v for v in ["R0", "R1", "R2", "R3", "C1", "C2"] if v in preds]
        w.writerow(["sample_index", "true_class"] + vs)
        for i in range(len(yte)):
            w.writerow([i, int(yte[i])] + [int(preds[v][i]) for v in vs])

    report = {
        "title": "RCMKN Haptics seed-42 ablation study",
        "seed": SEED, "dataset": "Haptics", "T": T,
        "protocol": {"train": len(Xtr), "val": len(Xva), "test": len(Xte),
                     "val_source": data["val_source"], "znorm": "per-sample, before all branches"},
        "versions": versions(),
        "config": {"encoder": ENCODER, "vq": VQ, "joint": JOINT,
                   "hydra": {**HYDRA, **hydra_struct, "feature_dim": hydra_dim},
                   "alphas": "logspace(-4,4,20)"},
        "results": results,
        "audits": audits,
    }
    with open(os.path.join(OUT_DIR, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    log(f"\nSaved {os.path.join(OUT_DIR, 'report.json')}")

    log("\n" + "=" * 74)
    log("ABLATION TABLE (test Macro-F1, seed 42)")
    log("=" * 74)
    for v in ["R0", "R1", "R2", "R3", "C1", "C2"]:
        if v in results:
            log(f"  {v}: {results[v]['test_macro_f1']:.4f} "
                f"(dim {results[v]['feature_dim']})")
    return report


if __name__ == "__main__":
    main()
