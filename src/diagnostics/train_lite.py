"""Canonical standalone TURS-Lite training (Phases 1/3/4/5).

Semantics copied EXACTLY from experiments/fair_turs.py::train_and_evaluate
for the TURS-Lite configuration:
  - model  = TURSNet(in_channels=1, regime_dim=16, variant="lite",
                     multi_scale_fusion=False)
  - loss   = TURSLoss(CE + aux regularizers, lambdas 0.01/0.005/0.01/0.005)
  - AdamW lr=3e-4 wd=1e-2 over model params + turs_loss params
  - OneCycleLR(max_lr=3e-4, epochs=30, steps_per_epoch=len(train_loader))
  - grad clip 1.0 on model parameters
  - early stopping on validation Macro-F1, patience 8, best-epoch snapshot
  - checkpoint {model_state_dict, time, best_epoch} at <ROOT>/checkpoints/<tag>.pt
"""
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (f1_score, accuracy_score, confusion_matrix,
                             balanced_accuracy_score, matthews_corrcoef)

from . import config as C


def make_turs_lite(n_cls):
    from models.tursnet import TURSNet
    return TURSNet(in_channels=1, num_classes=n_cls, regime_dim=16,
                   variant="lite", multi_scale_fusion=False)


def evaluate_full(model, X, y, n_cls, device, batch_size=256):
    """Full metric suite on a frozen model. X: [N,1,L] normalized."""
    from models.tursnet import TURSNet  # noqa: F401  (import check)
    model.eval()
    dl = DataLoader(TensorDataset(torch.from_numpy(X).float(),
                                  torch.from_numpy(np.asarray(y)).long()),
                    batch_size=batch_size)
    probs_l, logits_l = [], []
    with torch.no_grad():
        for xb, _ in dl:
            out = model(xb.to(device))
            logits_l.append(out.float().cpu())
            probs_l.append(torch.softmax(out, dim=1).cpu())
    logits = torch.cat(logits_l).numpy()
    probs = torch.cat(probs_l).numpy()
    preds = probs.argmax(1)
    y = np.asarray(y)
    return dict(
        accuracy=float(accuracy_score(y, preds)),
        macro_f1=float(f1_score(y, preds, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y, preds, average="weighted", zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(y, preds)),
        mcc=float(matthews_corrcoef(y, preds)),
        class_f1s=[float(v) for v in f1_score(y, preds, average=None,
                                              labels=list(range(n_cls)),
                                              zero_division=0)],
        confusion_matrix=confusion_matrix(y, preds, labels=list(range(n_cls))).tolist(),
        probs=probs, logits=logits, preds=preds,
    )


def train_turs_lite(ds, device, log=print, force=False):
    """Train (or reuse) canonical TURS-Lite for one dataset dict from data.load_split."""
    tag = ds["tag"]
    os.makedirs(C.CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(C.CKPT_DIR, f"{tag}_TURS_Lite.pt")
    meta_path = os.path.join(C.CKPT_DIR, f"{tag}_TURS_Lite_meta.json")

    if os.path.exists(ckpt_path) and os.path.exists(meta_path) and not force:
        log(f"  [train] checkpoint exists, skipping training: {ckpt_path}")
        return ckpt_path, json.load(open(meta_path)), None

    from models.tursnet import TURSLoss
    from . import config as CC

    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    n_cls = ds["n_cls"]
    model = make_turs_lite(n_cls).to(device)
    nparams = sum(p.numel() for p in model.parameters())

    tr_dl = DataLoader(TensorDataset(torch.from_numpy(ds["Xtr"]).float(),
                                     torch.from_numpy(ds["y_train"]).long()),
                       batch_size=C.BATCH_SIZE, shuffle=True)

    turs_loss = TURSLoss(num_classes=n_cls, focal_gamma=1.0, use_focal=False,
                         **CC.TURS_LOSS_LAMBDAS).to(device)
    params = list(model.parameters()) + list(turs_loss.parameters())
    opt = torch.optim.AdamW(params, lr=C.LR, weight_decay=C.WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=C.LR,
                                                steps_per_epoch=len(tr_dl),
                                                epochs=C.MAX_EPOCHS)

    best_val_mf1, best_state, best_ep, no_improve = -1, None, 0, 0
    history = []
    t0 = time.time()

    for ep in range(C.MAX_EPOCHS):
        model.train()
        ep_loss, ep_terms = 0.0, {}
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            logits, aux = model(xb, return_aux=True)
            loss, ldict = turs_loss(logits, yb, aux)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            ep_loss += float(loss) * len(xb)
            for k, v in ldict.items():
                ep_terms[k] = ep_terms.get(k, 0.0) + float(v) * len(xb)
        n_tr = len(ds["y_train"])

        model.eval()
        vp, vt = [], []
        va_dl = DataLoader(TensorDataset(torch.from_numpy(ds["Xva"]).float(),
                                         torch.from_numpy(ds["y_val"]).long()),
                           batch_size=256)
        with torch.no_grad():
            for xb, yb in va_dl:
                logits, _ = model(xb.to(device), return_aux=True)
                vp.append(logits.argmax(-1).cpu().numpy())
                vt.append(yb.numpy())
        val_mf1 = float(f1_score(np.concatenate(vt), np.concatenate(vp),
                                 average="macro", zero_division=0))

        history.append(dict(epoch=ep + 1, train_loss=round(ep_loss / n_tr, 6),
                            term_losses={k: round(v / n_tr, 6) for k, v in ep_terms.items()},
                            val_mf1=round(val_mf1, 6),
                            lr=round(opt.param_groups[0]["lr"], 8)))
        improved = val_mf1 > best_val_mf1
        if improved:
            best_val_mf1, best_ep, no_improve = val_mf1, ep + 1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            no_improve += 1
        log(f"    [train {tag}] Ep {ep+1:2d}/{C.MAX_EPOCHS} loss={ep_loss/n_tr:.4f} "
            f"val_mf1={val_mf1:.4f} best={best_val_mf1:.4f}")
        if no_improve >= C.PATIENCE:
            log(f"    [train {tag}] early stop at ep {ep+1}")
            break

    elapsed = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({"model_state_dict": model.state_dict(), "time": elapsed,
                "best_epoch": best_ep}, ckpt_path)
    meta = dict(tag=tag, params=nparams, best_epoch=best_ep,
                best_val_mf1=best_val_mf1, total_epochs=len(history),
                elapsed_s=round(elapsed, 1), seed=C.SEED,
                protocol=dict(lr=C.LR, wd=C.WD, batch=C.BATCH_SIZE,
                              max_epochs=C.MAX_EPOCHS, patience=C.PATIENCE,
                              loss="TURSLoss CE + aux (0.01/0.005/0.01/0.005)",
                              optimizer="AdamW", scheduler="OneCycleLR",
                              grad_clip=1.0),
                history=history)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log(f"    [train {tag}] done: best_val_mf1={best_val_mf1:.4f} "
        f"best_ep={best_ep} {elapsed:.0f}s params={nparams:,}")
    return ckpt_path, meta, history


def load_frozen(tag, device):
    """Load frozen canonical checkpoint (Phase 5)."""
    from .train_lite import make_turs_lite
    ckpt_path = os.path.join(C.CKPT_DIR, f"{tag}_TURS_Lite.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    n_cls = C.NUM_CLASSES[tag]
    model = make_turs_lite(n_cls).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt
