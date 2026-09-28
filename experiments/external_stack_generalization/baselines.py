"""
Standard baselines for the external generalization experiment.

PHASE 5-7: ONLY established project baselines are run here, reusing the
repository's canonical implementations and configurations verbatim:

  - MiniROCKET : aeon MiniRocket(random_state=42, n_jobs=-1) — canonical ~10K
                 configuration (NOT the 2016 variant), RidgeClassifierCV
                 (alphas=np.logspace(-4, 4, 20)) — same protocol as
                 experiments/benchmark_baselines.py / results/baseline_bench.
  - ResNet1D / FCN / PatchTSTCls : models/external_baselines.py (canonical).
  - InceptionTime : models/inceptiontime.py (canonical).

Training protocol (identical to experiments/benchmark_baselines.py, the
repository's canonical neural-baseline protocol): seed 42, per-sample
z-norm, batch 64, AdamW lr 3e-4 / wd 1e-2, OneCycleLR, CrossEntropyLoss,
max 15 epochs, early stopping on VALIDATION macro-F1 (patience 6),
grad clip 1.0. Test evaluated once with the best-val checkpoint.

No test labels influence anything (PHASE 10).
"""
import copy
import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score)

NEURAL_CONFIG = {
    "LR": 3e-4, "WD": 1e-2, "BS": 64, "MAX_EP": 15, "PAT": 6, "SEED": 42,
}


# ----------------------------------------------------------------------
# metrics (superset of the canonical eval_one: adds balanced accuracy
# and returns predictions for the complementarity analysis)
# ----------------------------------------------------------------------
def full_metrics(y_true, preds, n_cls):
    per_f1 = f1_score(y_true, preds, average=None, zero_division=0,
                      labels=list(range(n_cls)))
    prec, rec = [], []
    for c in range(n_cls):
        tp = int(((preds == c) & (y_true == c)).sum())
        fp = int(((preds == c) & (y_true != c)).sum())
        fn = int(((preds != c) & (y_true == c)).sum())
        prec.append(round(tp / (tp + fp), 4) if (tp + fp) else 0.0)
        rec.append(round(tp / (tp + fn), 4) if (tp + fn) else 0.0)
    return {
        "accuracy": round(float(accuracy_score(y_true, preds)), 4),
        "macro_f1": round(float(f1_score(y_true, preds, average="macro",
                                         zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, preds, average="weighted",
                                            zero_division=0)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true,
                                                                 preds)), 4),
        "class_f1s": [round(float(f), 4) for f in per_f1],
        "class_precision": prec,
        "class_recall": rec,
        "confusion_matrix": confusion_matrix(
            y_true, preds, labels=list(range(n_cls))).tolist(),
    }


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------
# MiniROCKET (PHASE 6)
# ----------------------------------------------------------------------
def run_minirocket(d, device="cpu", seed=42):
    """Canonical aeon MiniRocket ~10K + RidgeClassifierCV.

    Validation MF1 convention: RidgeCV fitted on TRAIN only -> VAL macro-F1
    (train-only fit, no leakage). Final test classifier fitted on TRAIN+VAL
    exactly like results/baseline_bench. Test touched once, at the end.
    """
    from aeon.transformations.collection.convolution_based import MiniRocket

    Xtr, Xva, Xte = d["Xtr"], d["Xva"], d["Xte"]
    ytr, yva, yte = d["ytr"], d["yva"], d["yte"]
    t0 = time.time()

    tr3 = Xtr[:, None, :].astype(np.float32)
    va3 = Xva[:, None, :].astype(np.float32)
    te3 = Xte[:, None, :].astype(np.float32)

    transformer = MiniRocket(random_state=seed, n_jobs=-1)
    Ztr = transformer.fit_transform(tr3)          # fit on TRAIN only
    t_fit = time.time() - t0
    Zva = transformer.transform(va3)
    Zte = transformer.transform(te3)
    t_trans = time.time() - t0 - t_fit

    # validation MF1 (train-only ridge — selection-free diagnostic)
    clf_val = RidgeClassifierCV(alphas=np.logspace(-4, 4, 20))
    clf_val.fit(Ztr, ytr)
    val_preds = clf_val.predict(Zva)
    val_mf1 = float(f1_score(yva, val_preds, average="macro", zero_division=0))

    # final canonical fit on TRAIN+VAL, single test evaluation
    Ztrva = np.concatenate([Ztr, Zva], axis=0)
    ytrva = np.concatenate([ytr, yva], axis=0)
    clf = RidgeClassifierCV(alphas=np.logspace(-4, 4, 20))
    clf.fit(Ztrva, ytrva)
    preds = clf.predict(Zte).astype(np.int64)
    t_total = time.time() - t0

    res = full_metrics(yte, preds, d["n_classes"])
    res.update({
        "model": "MiniROCKET",
        "n_kernels": int(getattr(transformer, "n_kernels", -1)),
        "n_features": int(Ztr.shape[1]),
        "selected_alpha": float(clf.alpha_),
        "val_macro_f1": round(val_mf1, 4),
        "params_trainable": 0,
        "time_transform_fit_s": round(t_fit, 2),
        "time_transform_all_s": round(t_fit + t_trans, 2),
        "time_total_s": round(t_total, 2),
        "device": "cpu",
    })
    return res, preds


# ----------------------------------------------------------------------
# Neural baselines (PHASE 7) — canonical protocol from benchmark_baselines
# ----------------------------------------------------------------------
def _train_neural(model, Xtr, ytr, Xva, yva, device, label, log):
    cfg = NEURAL_CONFIG
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["LR"],
                            weight_decay=cfg["WD"])
    from torch.utils.data import DataLoader, TensorDataset
    tr_dl = DataLoader(TensorDataset(torch.from_numpy(Xtr).float(),
                                     torch.from_numpy(ytr).long()),
                       batch_size=cfg["BS"], shuffle=True)
    va_dl = DataLoader(TensorDataset(torch.from_numpy(Xva).float(),
                                     torch.from_numpy(yva).long()),
                       batch_size=256)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg["LR"], steps_per_epoch=len(tr_dl), epochs=cfg["MAX_EP"])

    best_mf1, best_state, best_ep, no_imp = -1.0, None, 0, 0
    t0 = time.time()
    for ep in range(cfg["MAX_EP"]):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        # validation (early stopping uses VALIDATION only)
        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for xb, yb in va_dl:
                vp.append(model(xb.to(device)).argmax(-1).cpu().numpy())
                vt.append(yb.numpy())
        vmf1 = f1_score(np.concatenate(vt), np.concatenate(vp),
                        average="macro", zero_division=0)
        if vmf1 > best_mf1:
            best_mf1, best_ep, no_imp = vmf1, ep + 1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            no_imp += 1
        log(f"    {label} ep{ep+1}: val MF1={vmf1:.4f} (best={best_mf1:.4f}) "
            f"[{time.time()-t0:.0f}s]")
        if no_imp >= cfg["PAT"]:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_mf1, best_ep, time.time() - t0


def _predict(model, X, device, batch=256):
    from torch.utils.data import DataLoader, TensorDataset
    dl = DataLoader(TensorDataset(torch.from_numpy(X.astype(np.float32))),
                    batch_size=batch)
    model.eval()
    out = []
    with torch.no_grad():
        for (xb,) in dl:
            out.append(model(xb.to(device)).argmax(-1).cpu().numpy())
    return np.concatenate(out).astype(np.int64)


def run_neural_baseline(model_name, d, device, log=print, seed=42):
    """Run one canonical neural baseline end-to-end. Returns (res, preds)."""
    from models.external_baselines import FCN, ResNet1D, PatchTSTCls
    from models.inceptiontime import InceptionTime

    factories = {
        "InceptionTime": lambda: InceptionTime(num_classes=d["n_classes"],
                                               in_channels=1),
        "ResNet-1D": lambda: ResNet1D(1, d["n_classes"]),
        "FCN": lambda: FCN(1, d["n_classes"]),
        "PatchTST": lambda: PatchTSTCls(1, d["n_classes"]),
    }
    if model_name not in factories:
        raise ValueError(f"unknown baseline {model_name}")

    set_seed(seed)
    model = factories[model_name]().to(device)
    n_params = sum(p.numel() for p in model.parameters())

    Xtr = d["Xtr"][:, None, :]
    Xva = d["Xva"][:, None, :]
    Xte = d["Xte"][:, None, :]
    t0 = time.time()
    val_mf1, best_ep, train_t = _train_neural(
        model, Xtr, d["ytr"], Xva, d["yva"], device, model_name, log)
    preds = _predict(model, Xte, device)
    total_t = time.time() - t0

    res = full_metrics(d["yte"], preds, d["n_classes"])
    res.update({
        "model": model_name,
        "val_macro_f1": round(float(val_mf1), 4),
        "best_epoch": int(best_ep),
        "params_trainable": int(n_params),
        "time_train_s": round(train_t, 2),
        "time_total_s": round(total_t, 2),
        "device": str(device),
    })
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return res, preds
