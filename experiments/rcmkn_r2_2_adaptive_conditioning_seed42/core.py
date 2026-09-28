"""R2.2 experiment core: frozen-R2 context, margin histograms, exact
conditioned heterogeneity, variant construction.

Two-pass design (memory-safe on 16 GB RAM):
  PASS A (per split): one margin-kernel pass -> chunk-0 identity audit vs
        the audited extractor, then per-chunk u-histograms ->
        (N, 4998, 8, 27) uint16 memmap (training sufficient statistics).
  PASS B (per split): one margin-kernel pass -> exact conditioned
        activations for ALL variants simultaneously -> exact H via the
        audited heterogeneity_features, in the same pass.

Variants (all 9996 features = 4998 global PPV + 4998 heterogeneity):
  B0   exact unconditioned H from ORIGINAL activations (R2 reproduction)
  B1   conditioned path with delta == 0 (must equal B0: identity audit)
  B2   exact conditioned H with the learned modulation table
  B3A  B2 table + occupancy-matched random codes (seed+900001 stream)
  B3B  B2 table + per-sample shuffled codes (seed+900002 stream)

The context model (encoder + VQ) is FROZEN from the validated R2
checkpoints; only the modulation table (a, b) and the surrogate aux head
are trained (train-only fitting, val-based checkpoint selection).
"""

import hashlib
import json
import os
import time

import numpy as np
import torch

from experiments.drtn_conditioned_minirocket_haptics_3seed.runner import (
    create_random_regime_control,     # audited B3A construction (seed+900001)
    create_shuffled_regime_control,   # audited B3B construction (seed+900002)
    compute_regime_heterogeneity,     # audited valid-region H
)
from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (
    compute_raw_activations,
    ppv_from_activations,
    independent_heterogeneity_recompute,
)
import experiments.drtn_conditioned_minirocket_transfer_seed42.runner as transfer

from experiments.rcmkn_haptics_seed42.model import RCMKNContextModel
from experiments.rcmkn_haptics_seed42.runner import (
    extract_context_regimes, macro_f1, set_seed,
)
from experiments.rcmkn_r2_2_adaptive_conditioning_seed42 import conditioning as cond

SEED = 42
K_CODES = 8
N_GLOBAL = cond.N_GLOBAL
N_FEATURES = cond.N_FEATURES
N_HET = cond.N_HET
MIN_OCCUPANCY = 0.01
ALPHAS = np.logspace(-4, 4, 20)
RESULTS_ROOT = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "results")
OUT_DIR = os.path.join(RESULTS_ROOT, "rcmkn_r2_2_adaptive_conditioning_seed42")

# ---- predeclared training configuration (NOT tuned) ------------------------
TRAIN = {
    "hist_bins": cond.HIST_N_INTERIOR,
    "soft_w": cond.SOFT_W,
    "lambda_cls": 0.10,          # same as the R2 joint phase
    "lr": 1e-3,
    "epochs": 150,
    "patience": 25,
    "batch_size": 64,            # over precomputed (sample) histograms
}

DATASETS = {
    "Haptics": {
        "ckpt": os.path.join(RESULTS_ROOT, "rcmkn_haptics_seed42",
                             "context_model_seed42.pt"),
        "expected": {"train": 132, "val": 23, "test": 308, "T": 1092,
                     "n_classes": 5},
        "R2_ref": 0.5500, "R0_ref": 0.5366, "M0_ref": 0.4974,
    },
    "CWRU_BAL": {
        "ckpt": os.path.join(RESULTS_ROOT,
                             "rcmkn_ssl_context_important2_seed42", "CWRU_BAL",
                             "context_model_seed42.pt"),
        "expected": {"train": 2727, "val": 482, "test": 567, "T": 1024,
                     "n_classes": 4},
        "R2_ref": 0.9982, "R0_ref": 0.9930, "M0_ref": 0.9947,
    },
}
VARIANT_ORDER = ["B0", "B1", "B2", "B3A", "B3B"]


def log(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------
# Context model loading (FROZEN; identical to the R2 experiments)
# ------------------------------------------------------------------
def load_frozen_context(ds_name, device):
    """Load the frozen R2 context model + hard codes for all splits."""
    info = DATASETS[ds_name]
    data = transfer.load_any_dataset(ds_name)
    Xtr, ytr = data["Xtr"], data["ytr"]
    Xva, yva = data["Xva"], data["yva"]
    Xte, yte = data["Xte"], data["yte"]

    Xtr_z, Xva_z, Xte_z = (transfer.znorm(Xtr), transfer.znorm(Xva),
                           transfer.znorm(Xte))
    Xtrva_z = np.vstack([Xtr_z, Xva_z]).astype(np.float32)
    ytrva = np.concatenate([ytr, yva])

    model = RCMKNContextModel(n_classes=info["expected"]["n_classes"])
    ck = torch.load(info["ckpt"], map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state"])
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    set_seed(SEED)
    codes_trva = extract_context_regimes(model, Xtrva_z, device, batch=32)
    codes_te = extract_context_regimes(model, Xte_z, device, batch=32)
    return {
        "data": data, "model": model, "codes_trva": codes_trva,
        "codes_te": codes_te, "n_train": len(Xtr),
        "Xtrva_z": Xtrva_z, "Xte_z": Xte_z, "ytrva": ytrva,
        "yva": yva, "yte": yte,
        "T": int(Xtr.shape[1]), "n_classes": int(data["n_classes"]),
        "split": {"train": len(Xtr), "val": len(Xva), "test": len(Xte)},
    }


def build_extractor(ctx):
    """Canonical MiniRocket, fit on z-normed TRAIN only (project convention)."""
    from aeon.transformations.collection.convolution_based import MiniRocket
    set_seed(SEED)
    extractor = MiniRocket(random_state=SEED, n_jobs=-1)
    extractor.fit(ctx["Xtrva_z"][:ctx["n_train"]][:, None, :]
                  .astype(np.float32))
    return extractor


def global_blocks(extractor, ctx):
    """Canonical global PPV block for trainva / val / test (aeon transform)."""
    F_trva = extractor.transform(
        ctx["Xtrva_z"][:, None, :].astype(np.float32))
    F_va = extractor.transform(
        ctx["Xtrva_z"][ctx["n_train"]:][:, None, :].astype(np.float32))
    F_te = extractor.transform(ctx["Xte_z"][:, None, :].astype(np.float32))
    assert F_trva.shape[1] == N_FEATURES
    assert not np.isnan(F_trva).any() and not np.isinf(F_trva).any()
    return (F_trva[:, :N_GLOBAL], F_va[:, :N_GLOBAL], F_te[:, :N_GLOBAL])


# ------------------------------------------------------------------
# PASS A: identity audit + margin histograms (training sufficient stats)
# ------------------------------------------------------------------
def pass_a_split(extractor, X_z, codes, split_name, device, chunk=16):
    """One margin-kernel pass per split.

    Chunk 0 doubles as AUDIT 2: exact bool equality of the margin kernel's
    activations with the audited raw extractor on the same rows.

    Returns (hist_path, audit_dict). hist: (N, 4998, K, 27) uint16 memmap
    of per-(sample, feature, code, bin) margin counts over VALID positions.
    """
    N, T = X_z.shape
    hist_dir = os.path.join(OUT_DIR, "hist")
    os.makedirs(hist_dir, exist_ok=True)
    hist_path = os.path.join(hist_dir, f"{split_name}_hist.npy")
    hist = np.lib.format.open_memmap(hist_path, mode="w+", dtype=np.uint16,
                                     shape=(N, N_HET, K_CODES, cond.HIST_S))
    audit = {"split": split_name, "n_rows": int(N), "identity_n_diff": None,
             "u_finite": True, "u_abs1_frac": 0.0}
    t0 = time.time()
    for c0, (act, u_het, valid_het) in enumerate(
            cond.compute_activations_and_margins(extractor, X_z, chunk=chunk)):
        c1 = min(c0 * chunk + chunk, N)
        if c0 == 0:
            act_ref, _ = compute_raw_activations(extractor, X_z[c0:c1])
            audit["identity_n_diff"] = int(np.count_nonzero(act != act_ref))
            del act_ref
            assert audit["identity_n_diff"] == 0, \
                f"AUDIT 2 FAILED: margin kernel != audited extractor " \
                f"({audit['identity_n_diff']} positions)"
        if not np.isfinite(u_het).all():
            audit["u_finite"] = False
            raise AssertionError("PASS A: non-finite margins")
        audit["u_abs1_frac"] += float(
            np.count_nonzero(
                np.abs(u_het[:, valid_het[cond.N_GLOBAL:]]) >= 1.0))
        h = cond.build_histograms(u_het, codes[c0 * chunk:c1], valid_het,
                                  device)
        hist[c0 * chunk:c1] = h.astype(np.uint16)
        del act, u_het, h
    audit["u_abs1_frac"] /= float(
        N * cond.N_HET * int(valid_het[cond.N_GLOBAL:].sum()))
    hist.flush()
    log(f"  [PASS A:{split_name}] {N} rows in {time.time()-t0:.0f}s; "
        f"identity diffs={audit['identity_n_diff']}; "
        f"|u|>=1 frac={audit['u_abs1_frac']:.4f}")
    return hist_path, audit


# ------------------------------------------------------------------
# Modulation training on precomputed histograms (train-only, val-selected)
# ------------------------------------------------------------------
def train_modulation(ds_name, ctx, extractor, device, smoke=False):
    """Train (a, b) + surrogate aux head on PASS-A histograms.

    Objective: lambda_cls * CE on the surrogate R2.2 representation
    [G | H2(tau(a,b))].  With the context model frozen, the SSL/commit/div
    terms of the R2 joint objective carry no gradient to the modulation, so
    this is exactly the spec's L_total restricted to learnable parameters.
    Checkpoint selection: best VAL macro-F1 of the surrogate head.
    No test data is touched anywhere in this function.
    """
    cfg = dict(TRAIN)
    if smoke:
        cfg["epochs"], cfg["patience"] = 3, 1
    dev = torch.device(device)

    n_trva = len(ctx["Xtrva_z"])
    n_tr = ctx["n_train"]
    hist_tr = np.load(os.path.join(OUT_DIR, "hist", "trainva_hist.npy"),
                      mmap_mode="r")
    G_trva = ctx["_G_trva"]
    G_mean = G_trva[:n_tr].mean(0, keepdims=True)
    G_std = G_trva[:n_tr].std(0, keepdims=True) + 1e-6
    G_mean_dev = torch.from_numpy(np.ascontiguousarray(G_mean)).to(dev)
    G_std_dev = torch.from_numpy(np.ascontiguousarray(G_std)).to(dev)
    y_tr = ctx["ytrva"][:n_tr]
    y_va = ctx["ytrva"][n_tr:]

    mod = cond.CodeModulation(K=K_CODES, M=N_HET).to(dev)
    head = torch.nn.Linear(N_FEATURES, ctx["n_classes"]).to(dev)
    params = list(mod.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    ceil_min_count = int(np.ceil(MIN_OCCUPANCY * ctx["T"]))
    bs = np.sign(np.asarray(extractor.parameters[4], dtype=np.float32)
                 [N_GLOBAL:])                    # sign(b_f), het block

    def batch_tensors(idx):
        cnt = torch.from_numpy(
            np.asarray(hist_tr[idx], dtype=np.float32)).to(dev)
        g = torch.from_numpy(np.ascontiguousarray(G_trva[idx])).to(dev)
        g = (g - G_mean_dev) / G_std_dev
        return cnt, g

    rng = np.random.RandomState(SEED)
    best_val, best_state, no_imp = -1.0, None, 0
    t0 = time.time()
    for ep in range(cfg["epochs"]):
        mod.train(); head.train()
        perm = rng.permutation(n_tr)
        ep_loss = nb = 0.0
        for b0 in range(0, n_tr, cfg["batch_size"]):
            idx = perm[b0:b0 + cfg["batch_size"]]
            cnt, g = batch_tensors(idx)
            H2, _ = cond.surrogate_H2(cnt, mod.delta(), bs, ceil_min_count)
            feats = torch.cat([g, H2], dim=1)                 # (B, 9996)
            logits = head(feats)
            loss = cfg["lambda_cls"] * torch.nn.functional.cross_entropy(
                logits,
                torch.from_numpy(np.ascontiguousarray(y_tr[idx])).to(dev))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            ep_loss += float(loss); nb += 1
        # validation (surrogate head macro-F1)
        mod.eval(); head.eval()
        preds = []
        with torch.no_grad():
            for b0 in range(n_tr, n_trva, cfg["batch_size"]):
                cnt, g = batch_tensors(
                    np.arange(b0, min(b0 + cfg["batch_size"], n_trva)))
                H2, _ = cond.surrogate_H2(cnt, mod.delta(), bs, ceil_min_count)
                feats = torch.cat([g, H2], dim=1)
                preds.append(head(feats).argmax(-1).cpu().numpy())
        val_mf1 = macro_f1(y_va, np.concatenate(preds)) if preds else 0.0
        if val_mf1 > best_val + 1e-4:
            best_val, no_imp = val_mf1, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in mod.state_dict().items()}
        else:
            no_imp += 1
        if (ep + 1) % 10 == 0 or ep == 0 or smoke:
            log(f"    [MOD] ep{ep+1}/{cfg['epochs']} "
                f"train={ep_loss/max(nb,1):.4f} valMF1={val_mf1:.4f} "
                f"(best {best_val:.4f}) [{time.time()-t0:.0f}s]")
        if not np.isfinite(ep_loss):
            raise AssertionError("NaN/Inf in modulation training loss")
        if no_imp >= cfg["patience"]:
            log(f"    [MOD] early stop ep{ep+1}")
            break
    if best_state is not None:
        mod.load_state_dict(best_state)
    diag = cond.modulation_diagnostics(mod)
    log(f"  [MOD] beta={diag['beta']:.4f} mean|s|={diag['mean_abs_s']:.4f} "
        f"max|s|={diag['max_abs_s']:.4f} "
        f"cos(mean)={diag['pairwise_cos_mean']:.3f}")
    return mod, diag, {"best_val_mf1": round(best_val, 4),
                       "epochs_run": ep + 1}


# ------------------------------------------------------------------
# PASS B: exact conditioned features for all variants in one pass
# ------------------------------------------------------------------
def pass_b_split(extractor, X_z, code_arrays, tau_tables, chunk=8,
                 need_unconditioned=True):
    """One margin-kernel pass -> exact H per variant.

    code_arrays: dict variant -> (N, T) int codes that index the variant's
                 tau table (B3A/B3B pass their control code arrays; B1/B2
                 the real codes; B0 handled via need_unconditioned).
    tau_tables : dict variant -> (K, 4998) SIGNED effective thresholds
                 (fold_bias_sign applied). B1 must pass the exact-zero
                 table so its conditioned activations are u > 0 == C > b.
    Returns dict variant -> (N, 4998) float64 H block.
    """
    N, T = X_z.shape
    outs = {name: np.empty((N, N_HET), dtype=np.float64)
            for name in code_arrays}
    if need_unconditioned:
        outs["B0"] = np.empty((N, N_HET), dtype=np.float64)
    t0 = time.time()
    for c0, (act, u_het, valid_het) in enumerate(
            cond.compute_activations_and_margins(extractor, X_z, chunk=chunk)):
        lo = c0 * chunk
        hi = min(lo + chunk, N)
        act_het = act[:, N_GLOBAL:]
        valid_h = valid_het[N_GLOBAL:]      # het-block valid mask (4998, T)
        if need_unconditioned:
            outs["B0"][lo:hi] = compute_regime_heterogeneity(
                act_het, valid_h, code_arrays["B2"][lo:hi])
        for name, regs in code_arrays.items():
            act_cond = cond.cond_act_from_u(u_het, tau_tables[name],
                                            regs[lo:hi], valid_h)
            outs[name][lo:hi] = compute_regime_heterogeneity(
                act_cond, valid_h, regs[lo:hi])
            del act_cond
        del act, act_het, u_het
    log(f"  [PASS B] {N} rows x {len(outs)} variant-H blocks "
        f"in {time.time()-t0:.0f}s")
    return outs


# ------------------------------------------------------------------
# Ridge protocol (train+val fit; identical to prior experiments)
# ------------------------------------------------------------------
def run_variant(name, blocks, yva, yte, ytrva, n_classes):
    F_tr = np.hstack([b[0] for b in blocks])
    F_va = np.hstack([b[1] for b in blocks])
    F_te = np.hstack([b[2] for b in blocks])
    assert F_tr.shape[1] == F_va.shape[1] == F_te.shape[1] == N_FEATURES, \
        f"{name}: budget {F_tr.shape[1]} != {N_FEATURES}"
    assert F_tr.shape[0] == len(ytrva), f"{name}: rows != trainva"
    from sklearn.linear_model import RidgeClassifierCV
    from sklearn.metrics import accuracy_score, f1_score
    ridge = RidgeClassifierCV(alphas=ALPHAS)
    t0 = time.time()
    ridge.fit(F_tr, ytrva)                    # AUDIT 20: train + validation
    pred_va = ridge.predict(F_va)
    pred_te = ridge.predict(F_te)
    return {
        "val_macro_f1": round(macro_f1(yva, pred_va), 4),
        "test_macro_f1": round(macro_f1(yte, pred_te), 4),
        "accuracy": round(float(accuracy_score(yte, pred_te)), 4),
        "selected_alpha": float(ridge.alpha_),
        "feature_dim": int(F_tr.shape[1]),
        "ridge_fit_s": round(time.time() - t0, 2),
        "class_f1s": [round(x, 4) for x in f1_score(
            yte, pred_te, average=None,
            labels=list(range(n_classes)), zero_division=0).tolist()],
    }, pred_te


def regime_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.int64).tobytes()
                          ).hexdigest()[:16]
