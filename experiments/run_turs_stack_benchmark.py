"""
TURS-Stack benchmark runner — shared-trunk, multi-regime-head stacking
======================================================================

Follows the repository's validated biomedical protocol (TURS-Lite / TURS-CS):
  seed=42, stratified 85/15 train/val from the official train split (or 85/15
  split when no official split exists), per-sample z-normalization,
  batch_size=64, AdamW lr=3e-4 wd=1e-2, OneCycleLR, early stopping on val
  Macro-F1 (patience=8), max 30 epochs, joint CE loss over the 4 branches
  (mean, matching the previous runner's convention), grad clip 1.0.

Stacking protocol (spec Parts 33-43):
  1. Train shared trunk + 4 branches jointly (end-to-end).
  2. Restore best-val checkpoint. FREEZE everything.
  3. Generate held-out VALIDATION predictions p1..p4 (+ novelty e_t).
  4. Fit combiners on validation ONLY:
       soft vote (no fit), hard vote (no fit),
       static weights (theta, 4 params), linear stacking (4C->C),
       diagnostic-conditioned stacking (4C+1 -> C).
  5. Evaluate ALL five methods on the FINAL TEST set.
  The test set is never used for combiner fitting or method selection.

Datasets (strictly): ECG5000_UNBAL, ECG5000_BAL, CWRU_UNBAL, CWRU_BAL.
FI-2010 is excluded. No historical baselines are touched.
"""

import os
import sys
import time
import json
import copy
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.turs_stack.model import (
    TURSStack,
    StaticWeightCombiner,
    StackingCombiner,
    DiagnosticConditionedCombiner,
    soft_vote,
    hard_vote,
)

SEED = 42
VAL_FRAC = 0.15
MAX_EPOCHS = 30
PATIENCE = 8
LR = 3e-4
WD = 1e-2
BATCH_SIZE = 64
LAMBDA_I = 0.1

ALL_DATASETS = [
    ("ECG5000_UNBAL", "data/ecg5000_resplit.npz", 5),
    ("ECG5000_BAL", "data/ecg5000_fair_balanced.npz", 6149),
    ("CWRU_UNBAL", "data/cwru_unbalanced.npz", 1600),
    ("CWRU_BAL", "data/cwru_balanced.npz", 3776),
]
NUM_CLASSES = {"ECG5000_UNBAL": 5, "ECG5000_BAL": 5, "CWRU_UNBAL": 4, "CWRU_BAL": 4}

RESULT_DIR = os.path.join(ROOT, "results", "turs_stack")
CKPT_DIR = os.path.join(ROOT, "checkpoints", "turs_stack")


# ============================================================
# Utilities
# ============================================================
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def znorm(X):
    mu = X.mean(axis=-1, keepdims=True)
    sig = X.std(axis=-1, keepdims=True)
    return ((X - mu) / (sig + 1e-8)).astype(np.float32)


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def group_param_count(model):
    """A. shared trunk, B-E. per-branch, F. total, G. combiners."""
    trunk = count_params(model.transport_encoder) + count_params(model.backbone)
    branches = {
        "lite": count_params(model.lite),
        "rv": count_params(model.rv),
        "cs": count_params(model.cs),
        "cmr": count_params(model.cmr),
    }
    return trunk, branches


def compute_sigma0(X_train, max_lag=50):
    """Characteristic timescale from TRAIN data only (autocorrelation 1/e decay)."""
    if X_train.ndim == 3:
        X_train = X_train.squeeze(1)
    Xf = X_train.astype(np.float64)
    T = Xf.shape[1]
    max_lag = min(max_lag, T // 2)
    mean_sig = Xf.mean(axis=0)
    mean_sig = mean_sig - mean_sig.mean()
    var = np.var(mean_sig)
    acf = None
    if var > 1e-10:
        c = np.correlate(mean_sig, mean_sig, mode="full")
        acf = c[len(c) // 2:] / (var * len(mean_sig))
    else:
        acfs = []
        for i in range(min(len(Xf), 100)):
            s = Xf[i] - Xf[i].mean()
            v = np.var(s)
            if v < 1e-10:
                continue
            c = np.correlate(s, s, mode="full")
            acfs.append(c[len(c) // 2:] / (v * len(s)))
        if not acfs:
            return 5.0
        acf = np.mean(acfs, axis=0)
    acf = acf[: max_lag]
    threshold = 1.0 / np.e
    sigma0 = float(max_lag)
    for lag in range(1, len(acf)):
        if acf[lag] < threshold:
            if acf[lag - 1] > threshold:
                frac = (acf[lag - 1] - threshold) / (acf[lag - 1] - acf[lag] + 1e-10)
                sigma0 = (lag - 1) + frac
            else:
                sigma0 = float(lag)
            break
    return float(max(1.5, min(sigma0, T / 4.0)))


# ============================================================
# Correctness gates (Parts 53-61)
# ============================================================
def run_correctness_gates(model, device, C_class):
    report = {}
    model.eval()
    B, T = 3, 140
    x = torch.randn(B, 1, T, device=device)

    # -- shared computation test (Part 54) --
    model.reset_call_counters()
    with torch.no_grad():
        out = model(x)
    report["transport_encoder_calls_per_forward"] = model._transport_calls
    report["backbone_calls_per_forward"] = model._backbone_calls
    assert model._transport_calls == 1, "transport encoder must be called once"
    assert model._backbone_calls == 1, "backbone must be called once"

    # -- output dimensions --
    assert len(out["branch_logits"]) == 4
    for l in out["branch_logits"]:
        assert l.shape == (B, C_class)
    assert out["novelty"].shape == (B,)
    report["output_shapes_ok"] = True

    # -- branch independence (Part 55): perturb H, check only that branch's
    #    logits change consistently is not directly testable since all see the
    #    same H; instead verify each branch uses ITS OWN projections by
    #    perturbing each branch's parameters and checking isolation of logits.
    with torch.no_grad():
        base = [l.clone() for l in out["branch_logits"]]
    for i, branch_mod in enumerate([model.lite, model.rv, model.cs, model.cmr]):
        saved = copy.deepcopy(branch_mod.state_dict())
        with torch.no_grad():
            for p_ in branch_mod.parameters():
                p_.add_(0.05)
        with torch.no_grad():
            out2 = model(x)
        diff_others = max(
            float((out2["branch_logits"][j] - base[j]).abs().max())
            for j in range(4) if j != i
        )
        diff_self = float((out2["branch_logits"][i] - base[i]).abs().max())
        assert diff_self > 1e-6, f"branch {i} perturbation had no effect"
        assert diff_others < 1e-5, (
            f"branch {i} perturbation leaked into other branches ({diff_others})")
        branch_mod.load_state_dict(saved)
    report["branch_independence_ok"] = True

    # -- beta simplex tests (Part 56) --
    beta_cs = out["beta_cs"]
    beta_cmr = out["beta_cmr"]
    assert torch.allclose(beta_cs.sum(-1), torch.ones_like(beta_cs[..., 0]), atol=1e-5)
    assert torch.allclose(beta_cmr.sum(-1), torch.ones_like(beta_cmr[..., 0]), atol=1e-5)
    report["beta_simplex_ok"] = True

    # -- RV response actually present (Part 56) --
    H = out["H"]
    with torch.no_grad():
        rv_out_a = model.rv(out["F_T"], H)
        H2 = H + 0.5 * torch.randn_like(H)
        rv_out_b = model.rv(out["F_T"], H2)
    report["rv_response_active"] = bool(
        (rv_out_a - rv_out_b).abs().max() > 1e-6)

    # -- CMR response exists independently (Part 56) --
    assert hasattr(model.cmr, "raw_gamma") and hasattr(model.cmr, "resp1")
    report["cmr_bounded_response_ok"] = True

    # -- probability simplex (Part 58) --
    for p_ in out["probs"]:
        s = p_.sum(-1)
        assert torch.allclose(s, torch.ones_like(s), atol=1e-5)
    report["probs_sum_to_one_ok"] = True

    # -- soft/hard vote --
    sv = soft_vote(out["probs"])
    hv = hard_vote(out["probs"])
    assert sv.shape == (B, C_class)
    assert hv.shape == (B,)
    report["vote_shapes_ok"] = True

    # -- combiners forward --
    sw = StaticWeightCombiner().to(device)
    sc = StackingCombiner(C_class).to(device)
    dc = DiagnosticConditionedCombiner(C_class).to(device)
    pf, w = sw(out["probs"])
    assert pf.shape == (B, C_class) and w.shape == (4,)
    assert sc(out["probs"]).shape == (B, C_class)
    e_t = out["novelty"]
    assert dc(out["probs"], e_t).shape == (B, C_class)
    report["combiners_forward_ok"] = True

    # -- no cross-window state (Part 57) --
    with torch.no_grad():
        o1 = model(x)
        o2 = model(x)
        same = all(
            torch.allclose(a, b, atol=1e-5)
            for a, b in zip(o1["branch_logits"], o2["branch_logits"])
        )
    assert same, "model must be stateless across windows (eval mode)"
    report["no_cross_window_state_ok"] = True

    # -- gradient flow (Part 59) --
    model.train()
    outg = model(x)
    y = torch.randint(0, C_class, (B,), device=device)
    loss = sum(nn.functional.cross_entropy(l, y) for l in outg["branch_logits"]) / 4.0
    loss.backward()
    grads_ok = {"trunk": [], "branches": {}}
    for name in ["transport_encoder", "backbone"]:
        gmax = max(
            (p_.grad.abs().max().item() for p_ in getattr(model, name).parameters()
             if p_.grad is not None),
            default=0.0,
        )
        grads_ok["trunk"].append(gmax)
    for bname, bmod in [("lite", model.lite), ("rv", model.rv),
                        ("cs", model.cs), ("cmr", model.cmr)]:
        gmax = max(
            (p_.grad.abs().max().item() for p_ in bmod.parameters() if p_.grad is not None),
            default=0.0,
        )
        grads_ok["branches"][bname] = gmax
    assert min(grads_ok["trunk"]) > 0, "no gradient reached shared trunk"
    assert min(grads_ok["branches"].values()) > 0, "no gradient reached a branch"
    report["gradient_flow_ok"] = grads_ok
    model.zero_grad(set_to_none=True)

    # -- serialization (Part 61) --
    ckpt = {
        "model_state_dict": model.state_dict(),
        "opt_state_dict": torch.optim.AdamW(model.parameters()).state_dict(),
        "epoch": 0,
        "best_metric": -1.0,
    }
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "ckpt.pt")
        torch.save(ckpt, path)
        loaded = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(loaded["model_state_dict"])
    report["serialization_ok"] = True
    return report


# ============================================================
# Prediction / eval helpers
# ============================================================
@torch.no_grad()
def predict_branch_probs(model, loader, device, return_novelty=False):
    """Frozen-model predictions: per-branch probs, argmax preds, novelty e_t."""
    model.eval()
    Ps, ys, Es = [], [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        out = model(xb)
        Ps.append(torch.stack(out["probs"], dim=1).cpu())   # [b, 4, C]
        Es.append(out["novelty"].cpu())
        ys.append(yb)
    P = torch.cat(Ps)                                       # [N, 4, C]
    y = torch.cat(ys)
    e_t = torch.cat(Es)
    if return_novelty:
        return P, y, e_t
    return P, y


def mf1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


# ============================================================
# Training
# ============================================================
def train_model(model, tr_dl, va_dl, te_dl, y_va, y_te, n_cls, tag, device,
                epochs=MAX_EPOCHS, patience=PATIENCE):
    """Joint training of shared trunk + 4 branch heads. Returns history."""
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR * 10, steps_per_epoch=len(tr_dl), epochs=epochs)

    best_mf1 = -1.0
    best_state = None
    best_ep = 0
    no_improve = 0
    history = []
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        ep_loss, branch_sums = 0.0, [0.0] * 4
        ep_t0 = time.time()
        for xb, yb in tr_dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            out = model(xb)
            losses = [criterion(l, yb) for l in out["branch_logits"]]
            loss = sum(losses) / 4.0
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            ep_loss += loss.item()
            for k in range(4):
                branch_sums[k] += losses[k].item()

        # validation: per-branch + soft-vote MF1
        P_va, y_va_t = predict_branch_probs(model, va_dl, device)
        val_stats = {}
        for k, name in enumerate(["lite", "rv", "cs", "cmr"]):
            val_stats[f"val_mf1_{name}"] = round(mf1(y_va_t, P_va[:, k].argmax(-1).numpy()), 4)
        val_soft = soft_vote([P_va[:, k] for k in range(4)]).argmax(-1).numpy()
        val_mf1_soft = mf1(y_va_t, val_soft)

        rec = {
            "epoch": ep + 1,
            "train_loss": round(ep_loss / len(tr_dl), 4),
            "branch_losses": [round(b / len(tr_dl), 4) for b in branch_sums],
            **val_stats,
            "val_mf1_softvote": round(val_mf1_soft, 4),
            "epoch_time_s": round(time.time() - ep_t0, 2),
        }
        history.append(rec)
        print(f"  [{tag}] Ep {ep+1:02d}/{epochs} loss={rec['train_loss']:.4f} "
              f"L={[f'{b:.3f}' for b in rec['branch_losses']]} "
              f"valMF1(lite/rv/cs/cmr)={rec['val_mf1_lite']:.3f}/{rec['val_mf1_rv']:.3f}/"
              f"{rec['val_mf1_cs']:.3f}/{rec['val_mf1_cmr']:.3f} soft={val_mf1_soft:.4f} "
              f"({rec['epoch_time_s']:.1f}s)", flush=True)

        if val_mf1_soft > best_mf1:
            best_mf1 = val_mf1_soft
            best_state = copy.deepcopy(model.state_dict())
            best_ep = ep + 1
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  [{tag}] Early stopping at epoch {ep+1} "
                      f"(best soft-vote val MF1 {best_mf1:.4f} @ ep {best_ep})")
                break

    total_time = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best_mf1, best_ep, total_time


# ============================================================
# Per-dataset experiment
# ============================================================
def run_one_dataset(tag, npz_rel, device, smoke=False, profile=False):
    print(f"\n{'='*70}\nTURS-Stack: {tag}\n{'='*70}", flush=True)
    set_seed(SEED)
    n_cls = NUM_CLASSES[tag]
    ds_res_dir = os.path.join(RESULT_DIR, tag)
    os.makedirs(ds_res_dir, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, f"{tag}_turs_stack.pt")
    stack_path = os.path.join(CKPT_DIR, f"{tag}_combiners.pt")

    # ---------------- data (repository protocol) ----------------
    data = np.load(os.path.join(ROOT, npz_rel))
    if "X_train" in data:
        # official split: validation carved from the train portion
        X_trval, y_trval = data["X_train"].astype(np.float32), data["y_train"].astype(np.int64)
        X_te, y_te = data["X_test"].astype(np.float32), data["y_test"].astype(np.int64)
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_trval, y_trval, test_size=VAL_FRAC / 0.85, random_state=SEED,
            stratify=y_trval)
    else:
        # no official split (CWRU): stratified 85/15 trainval/test, then val
        X_all, y_all = data["X"].astype(np.float32), data["y"].astype(np.int64)
        idx = np.arange(len(X_all))
        trval_idx, te_idx = train_test_split(idx, test_size=0.15, random_state=SEED,
                                             stratify=y_all)
        X_trval, y_trval = X_all[trval_idx], y_all[trval_idx]
        X_te, y_te = X_all[te_idx], y_all[te_idx]
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_trval, y_trval, test_size=VAL_FRAC / 0.85, random_state=SEED,
            stratify=y_trval)

    T = X_tr.shape[1]
    C_in = 1
    print(f"  C={C_in} T={T} num_classes={n_cls} | "
          f"train={len(X_tr)} val={len(X_va)} test={len(X_te)}", flush=True)
    print(f"  train dist: {np.bincount(y_tr, minlength=n_cls).tolist()} | "
          f"test dist: {np.bincount(y_te, minlength=n_cls).tolist()}", flush=True)

    X_tr, X_va, X_te = znorm(X_tr), znorm(X_va), znorm(X_te)
    X_tr = X_tr[:, None, :].astype(np.float32)
    X_va = X_va[:, None, :].astype(np.float32)
    X_te = X_te[:, None, :].astype(np.float32)

    tr_dl = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                       batch_size=BATCH_SIZE, shuffle=True,
                       num_workers=0, pin_memory=(device.type == "cuda"))
    va_dl = DataLoader(TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
                       batch_size=256, shuffle=False)
    te_dl = DataLoader(TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te)),
                       batch_size=256, shuffle=False)

    # ---------------- model ----------------
    model = TURSStack(in_channels=C_in, num_classes=n_cls, sequence_length=T,
                      lambda_I=LAMBDA_I).to(device)

    # ---------------- correctness gates ----------------
    print("  [gates] correctness tests...", flush=True)
    gate_report = run_correctness_gates(model, device, n_cls)
    print(f"  [gates] passed: {json.dumps({k: v for k, v in gate_report.items() if not isinstance(v, dict)}, default=str)}", flush=True)

    # ---------------- calibration warm start (train data only) ----------------
    sigma0 = compute_sigma0(X_tr)
    model.cs.scale_params.raw_sigma.data.fill_(float(np.log(np.exp(sigma0 - 1.0) - 1)))
    model.cmr.scale_params.raw_sigma.data.fill_(float(np.log(np.exp(sigma0 - 1.0) - 1)))
    print(f"  [calibration] sigma0 (train-only autocorrelation): {sigma0:.2f}", flush=True)

    param_report = {}
    trunk_p, branch_p = group_param_count(model)
    total_p = count_params(model)
    param_report = {
        "shared_trunk": trunk_p,
        "branch_lite": branch_p["lite"],
        "branch_rv": branch_p["rv"],
        "branch_cs": branch_p["cs"],
        "branch_cmr": branch_p["cmr"],
        "total": total_p,
        "independent_4x_estimate": trunk_p * 4 + sum(branch_p.values()) * 4,
        "saving_fraction": 1.0 - total_p / (trunk_p * 4 + sum(branch_p.values()) * 4),
    }
    print(f"  [params] trunk={trunk_p:,} | lite={branch_p['lite']:,} rv={branch_p['rv']:,} "
          f"cs={branch_p['cs']:,} cmr={branch_p['cmr']:,} | total={total_p:,} | "
          f"4x-independent est={param_report['independent_4x_estimate']:,}", flush=True)

    if smoke:
        # ------- smoke: 1 train batch, 1 val, combiner fit on tiny subset -------
        model.train()
        xb, yb = next(iter(tr_dl))
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        loss = sum(nn.functional.cross_entropy(l, yb) for l in out["branch_logits"]) / 4.0
        loss.backward()
        opt_smoke = torch.optim.AdamW(model.parameters(), lr=1e-4)
        opt_smoke.step()
        P_sm, y_sm = predict_branch_probs(model, va_dl, device)  # already CPU
        # tiny combiner fit (structure check only) — combiners live on CPU here
        swc = StaticWeightCombiner()
        sc = StackingCombiner(n_cls)
        dcc = DiagnosticConditionedCombiner(n_cls)
        Pv = P_sm[:64]
        yv = y_sm[:64]
        if len(torch.unique(yv)) > 1:
            o = torch.optim.Adam(list(swc.parameters()) + list(sc.parameters())
                                 + list(dcc.parameters()), lr=0.05)
            for _ in range(5):
                o.zero_grad()
                l = (nn.functional.cross_entropy(swc([Pv[:, k] for k in range(4)])[0], yv)
                     + nn.functional.cross_entropy(sc([Pv[:, k] for k in range(4)]), yv)
                     + nn.functional.cross_entropy(dcc([Pv[:, k] for k in range(4)], 0.5 * torch.ones(len(yv))), yv))
                l.backward()
                o.step()
        torch.save({"model_state_dict": model.state_dict()}, ckpt_path + ".smoke")
        torch.load(ckpt_path + ".smoke", map_location=device, weights_only=False)
        os.remove(ckpt_path + ".smoke")
        print(f"  [SMOKE] {tag} OK — forward/backward/step/val-preds/combiner-fit/serialize", flush=True)
        return {"smoke": True}

    if profile:
        # ------- profile: ~100 batches, measure fwd/bwd, VRAM, samples/sec -------
        import torch.cuda as cuda
        model.train()
        n_batches = min(100, len(tr_dl))
        fwd_t, bwd_t = [], []
        if device.type == "cuda":
            cuda.reset_peak_memory_stats()
        it = iter(tr_dl)
        # warmup
        for _ in range(3):
            xb, yb = next(it)
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = sum(nn.functional.cross_entropy(l, yb) for l in out["branch_logits"]) / 4.0
            model.zero_grad(set_to_none=True)
            loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_all0 = time.time()
        for i in range(n_batches):
            try:
                xb, yb = next(it)
            except StopIteration:
                it = iter(tr_dl)
                xb, yb = next(it)
            xb, yb = xb.to(device), yb.to(device)
            torch.cuda.synchronize() if device.type == "cuda" else None
            tf0 = time.time()
            out = model(xb)
            loss = sum(nn.functional.cross_entropy(l, yb) for l in out["branch_logits"]) / 4.0
            torch.cuda.synchronize() if device.type == "cuda" else None
            fwd_t.append(time.time() - tf0)
            tb0 = time.time()
            model.zero_grad(set_to_none=True)
            loss.backward()
            torch.cuda.synchronize() if device.type == "cuda" else None
            bwd_t.append(time.time() - tb0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_t = time.time() - t_all0
        prof = {
            "n_batches": n_batches,
            "batch_size": BATCH_SIZE,
            "forward_ms_mean": round(1000 * float(np.mean(fwd_t)), 2),
            "backward_ms_mean": round(1000 * float(np.mean(bwd_t)), 2),
            "train_step_ms_mean": round(1000 * total_t / n_batches, 2),
            "samples_per_sec": round(n_batches * BATCH_SIZE / total_t, 1),
            "peak_vram_mb": round(cuda.max_memory_allocated() / 1e6, 1) if device.type == "cuda" else None,
            "peak_vram_reserved_mb": round(cuda.max_memory_reserved() / 1e6, 1) if device.type == "cuda" else None,
            "params": total_p,
        }
        with open(os.path.join(ds_res_dir, "profile.json"), "w") as f:
            json.dump(prof, f, indent=2)
        print(f"  [profile] {json.dumps(prof)}", flush=True)
        return {"profile": prof}

    # ---------------- resume support ----------------
    start_ep = 0
    best_mf1_prev = -1.0
    history_prev = []
    if os.path.exists(ckpt_path):
        try:
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ck["model_state_dict"])
            start_ep = ck.get("epoch", 0)
            best_mf1_prev = ck.get("best_metric", -1.0)
            history_prev = ck.get("history", [])
            print(f"  [resume] loaded checkpoint at epoch {start_ep}", flush=True)
        except Exception as e:  # noqa
            print(f"  [resume] failed ({e}); training fresh", flush=True)
            start_ep = 0

    if start_ep >= MAX_EPOCHS:
        history, best_val, best_ep, train_time = history_prev, best_mf1_prev, ck.get("best_ep", 0), ck.get("time_s", 0.0)
    else:
        history, best_val, best_ep, train_time = train_model(
            model, tr_dl, va_dl, te_dl, y_va, y_te, n_cls, tag, device)
        # persist resumable checkpoint
        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": MAX_EPOCHS,
            "best_metric": best_val,
            "best_ep": best_ep,
            "time_s": train_time,
            "history": history,
            "sigma0": sigma0,
            "num_classes": n_cls,
            "sequence_length": T,
        }, ckpt_path)

    # ---------------- freeze & held-out predictions ----------------
    for p_ in model.parameters():
        p_.requires_grad_(False)
    model.eval()

    P_va, y_va_t, e_va = predict_branch_probs(model, va_dl, device, return_novelty=True)
    P_te, y_te_t, e_te = predict_branch_probs(model, te_dl, device, return_novelty=True)

    # leakage test: combiner fitting uses ONLY validation tensors
    leakage_test = {"combiner_fit_source": "validation_only",
                    "val_n": int(len(y_va_t)), "test_n": int(len(y_te_t))}

    # ---------------- combiner fitting (validation only) ----------------
    # NOTE: probabilities are computed on GPU then moved to CPU; combiners are
    # tiny (<= 4C+1 params) so they are fitted on CPU for device consistency.
    P_va = P_va.cpu()
    P_te = P_te.cpu()
    e_va = e_va.cpu()
    e_te = e_te.cpu()
    probs_va = [P_va[:, k] for k in range(4)]
    probs_te = [P_te[:, k] for k in range(4)]

    fit_results = {}

    # C: static weights (combiners stay on CPU — probabilities are CPU tensors)
    swc = StaticWeightCombiner()
    o = torch.optim.Adam(swc.parameters(), lr=0.05)
    for _ in range(300):
        o.zero_grad()
        pf, _ = swc(probs_va)
        loss = nn.functional.cross_entropy(pf, y_va_t)
        loss.backward()
        o.step()
    w_learned = torch.softmax(swc.theta.detach(), dim=0).cpu().numpy()

    # D: linear stacking
    sc = StackingCombiner(n_cls)
    o = torch.optim.Adam(sc.parameters(), lr=0.05)
    for _ in range(300):
        o.zero_grad()
        loss = nn.functional.cross_entropy(sc(probs_va), y_va_t)
        loss.backward()
        o.step()

    # E: diagnostic-conditioned stacking
    dcc = DiagnosticConditionedCombiner(n_cls)
    o = torch.optim.Adam(dcc.parameters(), lr=0.05)
    for _ in range(300):
        o.zero_grad()
        loss = nn.functional.cross_entropy(dcc(probs_va, e_va), y_va_t)
        loss.backward()
        o.step()

    # ---------------- evaluate all five on TEST ----------------
    with torch.no_grad():
        sv_te = soft_vote(probs_te).argmax(-1).numpy()
        hv_te = hard_vote(probs_te).numpy()
        pf_te, _ = swc(probs_te)
        st_te = pf_te.argmax(-1).numpy()
        stck_te = sc(probs_te).argmax(-1).numpy()
        dstack_te = dcc(probs_te, e_te).argmax(-1).numpy()

        # validation selection metrics (for pre-selection transparency)
        sv_va = soft_vote(probs_va).argmax(-1).numpy()
        hv_va = hard_vote(probs_va).numpy()
        pf_va, _ = swc(probs_va)
        st_va = pf_va.argmax(-1).numpy()
        stck_va = sc(probs_va).argmax(-1).numpy()
        dstack_va = dcc(probs_va, e_va).argmax(-1).numpy()

    combo_val = {
        "soft_vote": mf1(y_va_t, sv_va), "hard_vote": mf1(y_va_t, hv_va),
        "static_weights": mf1(y_va_t, st_va), "stacking": mf1(y_va_t, stck_va),
        "diagnostic_stacking": mf1(y_va_t, dstack_va),
    }
    combo_test = {
        "soft_vote": mf1(y_te_t, sv_te), "hard_vote": mf1(y_te_t, hv_te),
        "static_weights": mf1(y_te_t, st_te), "stacking": mf1(y_te_t, stck_te),
        "diagnostic_stacking": mf1(y_te_t, dstack_te),
    }
    best_combo_val = max(combo_val, key=combo_val.get)
    preselected_test = combo_test[best_combo_val]

    # ---------------- branch results + diagnostics ----------------
    branch_test, branch_val = {}, {}
    for k, name in enumerate(["lite", "rv", "cs", "cmr"]):
        branch_val[name] = mf1(y_va_t, P_va[:, k].argmax(-1).numpy())
        branch_test[name] = mf1(y_te_t, P_te[:, k].argmax(-1).numpy())

    def conf_stats(P, y):
        conf = P.max(-1).values.numpy()
        preds = P.argmax(-1).numpy()
        return {"mean_conf": round(float(conf.mean()), 4),
                "pred_dist": np.bincount(preds, minlength=P.shape[-1]).tolist(),
                "true_dist": np.bincount(y, minlength=P.shape[-1]).tolist()}

    diagnostics = {"val": {n: conf_stats(P_va[:, k], y_va_t.numpy())
                           for k, n in enumerate(["lite", "rv", "cs", "cmr"])},
                   "test": {n: conf_stats(P_te[:, k], y_te_t.numpy())
                            for k, n in enumerate(["lite", "rv", "cs", "cmr"])}}

    # RV correction/velocity diagnostic (Part 13, Part 69)
    rv_diag_batches = []
    model.eval()
    with torch.no_grad():
        for xb, _ in te_dl:
            xb = xb.to(device)
            out = model(xb)
            H = out["H"]
            z_t = model.rv.P_z(H).permute(0, 2, 1)
            v_t = model.rv.vel_linear(z_t)
            R7 = model.rv.dw7(H).permute(0, 2, 1)
            R15 = model.rv.dw15(H).permute(0, 2, 1)
            R31 = model.rv.dw31(H).permute(0, 2, 1)
            R_t = torch.cat([R7, R15, R31], dim=-1)
            DR = torch.zeros_like(R_t)
            DR[:, 1:, :] = R_t[:, 1:, :] - R_t[:, :-1, :]
            g = torch.sigmoid(model.rv.P_g(torch.cat([z_t, v_t, R_t], dim=-1)))
            corr = g * model.rv.P_Delta(DR)
            rv_diag_batches.append({
                "corr_norm": float(corr.norm(dim=-1).mean()),
                "v_norm": float(v_t.norm(dim=-1).mean()),
            })
            break  # one representative batch is sufficient for the diagnostic
    rv_corr_ratio = rv_diag_batches[0]["corr_norm"] / max(rv_diag_batches[0]["v_norm"], 1e-8)

    # Branch 3/4 calibration + beta stats (Part 69)
    with torch.no_grad():
        out_probe = model(va_dl.dataset.tensors[0][:256].to(device))
    beta_cs_all = out_probe["beta_cs"].cpu().numpy()
    beta_cmr_all = out_probe["beta_cmr"].cpu().numpy()

    def beta_stats(b):
        return {"mean": b.mean(axis=(0, 1)).round(4).tolist(),
                "std": b.std(axis=(0, 1)).round(4).tolist(),
                "entropy": float(-(b * np.log(b + 1e-9)).sum(-1).mean())}

    sigma_cs = float(model.cs.scale_params.sigma.item())
    rho_cs = [float(r) for r in model.cs.scale_params.rho]
    sigma_cmr = float(model.cmr.scale_params.sigma.item())
    rho_cmr = [float(r) for r in model.cmr.scale_params.rho]
    gamma_cmr = float(model.cmr.gamma.item())

    novelty_stats = {"val_mean": float(e_va.mean()), "val_std": float(e_va.std()),
                     "test_mean": float(e_te.mean()), "test_std": float(e_te.std())}

    # ---------------- complementarity analysis (Part 70) ----------------
    def complementarity(P, y):
        preds = P.argmax(-1).numpy()          # [N, 4]
        names = ["lite", "rv", "cs", "cmr"]
        agree, disagree, overlap, corr = {}, {}, {}, {}
        for i in range(4):
            for j in range(i + 1, 4):
                a, b = preds[:, i], preds[:, j]
                pair = f"{names[i]}-{names[j]}"
                agree[pair] = round(float((a == b).mean()), 4)
                err_i, err_j = (a != y.numpy()), (b != y.numpy())
                # error overlap: of samples at least one errs, fraction both err
                both = err_i & err_j
                either = err_i | err_j
                overlap[pair] = round(float(both.sum() / max(either.sum(), 1)), 4)
                pi, pj = P[:, i].numpy(), P[:, j].numpy()
                ci = pi - pi.mean(0, keepdims=True)
                cj = pj - pj.mean(0, keepdims=True)
                denom = (np.linalg.norm(ci) * np.linalg.norm(cj) + 1e-9)
                corr[pair] = round(float((ci * cj).sum() / denom), 4)
        return {"agreement": agree, "error_overlap": overlap, "prob_corr": corr}

    comp_val = complementarity(P_va, y_va_t)
    comp_te = complementarity(P_te, y_te_t)

    # ---------------- result payload ----------------
    test_soft_acc = accuracy_score(y_te_t.numpy(), sv_te)
    res = {
        "dataset": tag,
        "num_classes": n_cls,
        "T": int(T),
        "n_train": len(y_tr), "n_val": len(y_va), "n_test": len(y_te),
        "seed": SEED,
        "params": param_report,
        "gates": {k: v for k, v in gate_report.items() if not isinstance(v, dict)},
        "calibration": {"sigma0_train_only": round(sigma0, 3),
                        "sigma_cs": round(sigma_cs, 3), "rho_cs": [round(r, 4) for r in rho_cs],
                        "sigma_cmr": round(sigma_cmr, 3), "rho_cmr": [round(r, 4) for r in rho_cmr],
                        "gamma_cmr": round(gamma_cmr, 4)},
        "training": {"best_val_softvote_mf1": round(float(best_val), 4),
                     "best_epoch": best_ep, "train_time_s": round(train_time, 1),
                     "history": history},
        "branch_val_mf1": {k: round(v, 4) for k, v in branch_val.items()},
        "branch_test_mf1": {k: round(v, 4) for k, v in branch_test.items()},
        "combination_val_mf1": {k: round(v, 4) for k, v in combo_val.items()},
        "combination_test_mf1": {k: round(v, 4) for k, v in combo_test.items()},
        "best_combiner_by_val": best_combo_val,
        "preselected_test_mf1": round(float(preselected_test), 4),
        "static_weights_learned": [round(float(w), 4) for w in w_learned],
        "rv_diag": {"corr_norm": round(rv_diag_batches[0]["corr_norm"], 4),
                    "v_norm": round(rv_diag_batches[0]["v_norm"], 4),
                    "ratio": round(float(rv_corr_ratio), 4)},
        "beta_stats": {"cs": beta_stats(beta_cs_all), "cmr": beta_stats(beta_cmr_all)},
        "novelty_feature": {k: round(v, 4) for k, v in novelty_stats.items()},
        "complementarity": {"val": comp_val, "test": comp_te},
        "diagnostics": diagnostics,
        "leakage_test": leakage_test,
        "test_softvote_accuracy": round(float(test_soft_acc), 4),
    }
    out_path = os.path.join(ds_res_dir, "full_results.json")
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)

    # combiner serialization (Part 68)
    torch.save({
        "static_theta": swc.state_dict(),
        "stacking": sc.state_dict(),
        "diagnostic_stacking": dcc.state_dict(),
        "best_by_val": best_combo_val,
    }, stack_path)

    print(f"  [{tag}] branches val(lite/rv/cs/cmr)="
          f"{branch_val['lite']:.4f}/{branch_val['rv']:.4f}/{branch_val['cs']:.4f}/{branch_val['cmr']:.4f}")
    print(f"  [{tag}] branches test="
          f"{branch_test['lite']:.4f}/{branch_test['rv']:.4f}/{branch_test['cs']:.4f}/{branch_test['cmr']:.4f}")
    print(f"  [{tag}] combos test: {json.dumps({k: round(v, 4) for k, v in combo_test.items()})}")
    print(f"  [{tag}] best-by-val: {best_combo_val} -> test {preselected_test:.4f}")
    print(f"  [{tag}] saved -> {out_path}", flush=True)
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="subset of dataset tags to run (default: all four, sequentially)")
    parser.add_argument("--smoke", action="store_true", help="per-dataset smoke test only")
    parser.add_argument("--profile", action="store_true", help="~100-batch profile only")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | seed={SEED} | max_epochs={MAX_EPOCHS} patience={PATIENCE}", flush=True)

    tags = args.datasets if args.datasets else [d[0] for d in ALL_DATASETS]
    # STRICT order: ECG5000_UNBAL -> ECG5000_BAL -> CWRU_UNBAL -> CWRU_BAL
    order = [d[0] for d in ALL_DATASETS]
    tags = sorted(tags, key=order.index)

    if args.smoke:
        for t in tags:
            run_one_dataset(t, dict((d[0], d[1]) for d in ALL_DATASETS)[t], device, smoke=True)
        print("\nAll smoke tests passed.")
        return
    if args.profile:
        run_one_dataset("ECG5000_BAL", "data/ecg5000_fair_balanced.npz", device, profile=True)
        print("\nProfile complete.")
        return

    results = {}
    ds_map = {d[0]: d[1] for d in ALL_DATASETS}
    for t in tags:
        results[t] = run_one_dataset(t, ds_map[t], device)
    print("\nSequential TURS-Stack benchmark complete:", list(results.keys()))


if __name__ == "__main__":
    main()
