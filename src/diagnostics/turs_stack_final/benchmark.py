"""Fresh canonical TURS-Stack benchmark (Phases 0-5, 7).

Training semantics copied EXACTLY from experiments/run_turs_stack_benchmark.py:
  - set_seed(42), data via the canonical split (VAL_FRAC/0.85 carve), z-norm
  - TURSStack(in_channels=1, num_classes, sequence_length, sigma_init=sigma0)
    sigma0 from TRAIN only (autocorrelation 1/e decay)
  - joint training: loss = mean CE over 4 branch logits
  - AdamW lr=3e-4 wd=1e-2, OneCycleLR(max_lr, steps=len(train), epochs=30)
  - grad clip 1.0, batch 64, max 30 epochs, early stop patience 8 on val MF1
    of the soft-vote combiner (canonical selection criterion)
  - after training: freeze trunk -> collect val branch probs -> fit combiners
    on VAL ONLY (static theta 4, stacking 4C->C, diagnostic stacking 4C+1->C;
    hard vote needs no fitting) -> evaluate ALL combiners on val, select best
    by val MF1, then evaluate on test ONCE.
Fresh artifacts under results/turs_stack_final/benchmark + checkpoints/turs_stack_final.
"""
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (f1_score, accuracy_score, confusion_matrix,
                             balanced_accuracy_score, matthews_corrcoef)

from . import config as C


def znorm(X):
    mu = X.mean(axis=-1, keepdims=True)
    sig = X.std(axis=-1, keepdims=True) + 1e-8
    return ((X - mu) / (sig + 1e-8)).astype(np.float32)


def compute_sigma0(X_train, max_lag=50):
    """TRAIN-only autocorrelation 1/e decay (canonical sigma0 warm start)."""
    X = X_train[:, 0, :] if X_train.ndim == 3 else X_train
    n = min(400, len(X))
    Xs = X[np.random.default_rng(C.SEED).choice(len(X), n, replace=False)]
    ac = np.zeros(max_lag)
    for x in Xs:
        x = x - x.mean()
        var = x.var() + 1e-12
        for lag in range(1, max_lag):
            ac[lag - 1] += float((x[:-lag] * x[lag:]).mean() / var)
    ac /= n
    below = np.where(ac < 1.0 / np.e)[0]
    return float(below[0] + 1 if len(below) else max_lag)


def load_split(tag):
    path = os.path.join(C.ROOT, C.DATA_FILE[tag])
    data = np.load(path)
    if "X_train" in data:
        X_trval, y_trval = data["X_train"].astype(np.float32), data["y_train"].astype(np.int64)
        X_te, y_te = data["X_test"].astype(np.float32), data["y_test"].astype(np.int64)
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_trval, y_trval, test_size=C.VAL_FRAC / 0.85, random_state=C.SEED,
            stratify=y_trval)
    else:
        X_all, y_all = data["X"].astype(np.float32), data["y"].astype(np.int64)
        idx = np.arange(len(X_all))
        trval_idx, te_idx = train_test_split(idx, test_size=0.15, random_state=C.SEED,
                                             stratify=y_all)
        X_trval, y_trval = X_all[trval_idx], y_all[trval_idx]
        X_te, y_te = X_all[te_idx], y_all[te_idx]
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_trval, y_trval, test_size=C.VAL_FRAC / 0.85, random_state=C.SEED,
            stratify=y_trval)
    return dict(tag=tag, n_cls=C.NUM_CLASSES[tag], L=int(X_tr.shape[1]),
                Xtr=znorm(X_tr)[:, None, :], Xva=znorm(X_va)[:, None, :],
                Xte=znorm(X_te)[:, None, :], y_train=y_tr, y_val=y_va, y_test=y_te)


def _sha(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def _mf1(y, pred, n_cls):
    return float(f1_score(y, pred, average="macro", zero_division=0,
                          labels=list(range(n_cls))))


def full_metrics(y, pred, n_cls):
    if getattr(pred, "ndim", 2) == 2:
        pred = pred.argmax(1)
    return dict(
        accuracy=float(accuracy_score(y, pred)),
        macro_f1=_mf1(y, pred, n_cls),
        weighted_f1=float(f1_score(y, pred, average="weighted", zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        mcc=float(matthews_corrcoef(y, pred)),
        confusion_matrix=confusion_matrix(y, pred, labels=list(range(n_cls))).tolist(),
    )


# ---------------------------------------------------------------- combiners
def fit_combiners_on_val(val_probs, val_y, val_novelty, n_cls):
    """Fit static/stacking/diag-stacking on VAL ONLY (canonical protocol)."""
    K, N, Cl = val_probs.shape
    P = torch.from_numpy(val_probs)                    # [K, N, C]
    yv = torch.from_numpy(val_y).long()
    soft = P.mean(0)
    static = torch.zeros(K, requires_grad=True)
    opt = torch.optim.Adam([static], lr=0.05)
    for _ in range(300):
        w = torch.softmax(static, 0)
        p = torch.einsum("k,knc->nc", w, P)
        loss = F.cross_entropy(p.log(), yv)
        opt.zero_grad(); loss.backward(); opt.step()
    stacking = torch.nn.Linear(K * Cl, Cl)
    opt2 = torch.optim.Adam(stacking.parameters(), lr=0.01)
    feats = P.permute(1, 0, 2).reshape(N, K * Cl)
    for _ in range(300):
        loss = F.cross_entropy(stacking(feats), yv)
        opt2.zero_grad(); loss.backward(); opt2.step()
    diag = torch.nn.Linear(K * Cl + 1, Cl)
    opt3 = torch.optim.Adam(diag.parameters(), lr=0.01)
    feats_d = torch.cat([feats, torch.from_numpy(val_novelty).float().reshape(N, 1)], 1)
    for _ in range(300):
        loss = F.cross_entropy(diag(feats_d), yv)
        opt3.zero_grad(); loss.backward(); opt3.step()
    return dict(static_weights=static.detach(), stacking=stacking.state_dict(),
                diagnostic_stacking=diag.state_dict())


def _preds(v):
    """Combiner output -> predictions. Prob arrays [N,C] argmax; hard-vote [N] as-is."""
    v = np.asarray(v)
    return v.argmax(1) if v.ndim == 2 else v.astype(np.int64)


def apply_combiners(probs, novelty, n_cls, fitted):
    """probs [K,N,C], novelty [N] -> dict name -> probs [N,C] (hard_vote: preds [N]).

    hard_vote matches models/turs_stack/model.py::hard_vote exactly (majority
    vote, summed-confidence tie-break) but returns [N] predictions."""
    import torch.nn.functional as Ff
    Pt = torch.from_numpy(probs)
    N = Pt.shape[1]
    out = {}
    out["soft_vote"] = Pt.mean(0).numpy()
    votes = Pt.argmax(-1)                              # [K, N]
    onehot = F.one_hot(votes, n_cls).float()
    counts = onehot.sum(0)
    maxv = counts.max(-1, keepdim=True).values
    conf = Pt.sum(0)
    out["hard_vote"] = torch.where(counts == maxv, conf,
                                   torch.full_like(conf, -1.0)).argmax(-1).numpy()
    w = torch.softmax(fitted["static_weights"], 0)
    out["static_weights"] = torch.einsum("k,knc->nc", w, Pt).numpy()
    st = torch.nn.Linear(Pt.shape[0] * n_cls, n_cls)
    st.load_state_dict(fitted["stacking"])
    feats = Pt.permute(1, 0, 2).reshape(N, -1)
    out["stacking"] = torch.softmax(st(feats), -1).detach().numpy()
    dg = torch.nn.Linear(Pt.shape[0] * n_cls + 1, n_cls)
    dg.load_state_dict(fitted["diagnostic_stacking"])
    feats_d = torch.cat([feats, torch.from_numpy(novelty).float().reshape(N, 1)], 1)
    out["diagnostic_stacking"] = torch.softmax(dg(feats_d), -1).detach().numpy()
    return out


# ---------------------------------------------------------------- training
def train_stack(ds, device, log=print, force=False):
    """Fresh canonical training for one dataset. Returns (ckpt_path, meta)."""
    tag = ds["tag"]
    os.makedirs(C.CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(C.CKPT_DIR, f"{tag}_turs_stack.pt")
    meta_path = os.path.join(C.CKPT_DIR, f"{tag}_meta.json")
    comb_path = os.path.join(C.CKPT_DIR, f"{tag}_combiners.pt")
    if all(os.path.exists(p) for p in [ckpt_path, meta_path, comb_path]) and not force:
        log(f"  [bench] fresh checkpoints exist, skipping training ({tag})")
        return ckpt_path, json.load(open(meta_path))

    from models.turs_stack.model import TURSStack
    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    n_cls, MAX_EPOCHS, PATIENCE = ds["n_cls"], 30, 8
    sigma0 = compute_sigma0(ds["Xtr"])
    model = TURSStack(in_channels=1, num_classes=n_cls,
                      sequence_length=ds["L"], sigma_init=sigma0).to(device)
    nparams = sum(p.numel() for p in model.parameters())

    tr_dl = DataLoader(TensorDataset(torch.from_numpy(ds["Xtr"]).float(),
                                     torch.from_numpy(ds["y_train"]).long()),
                       batch_size=64, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4,
                                                steps_per_epoch=len(tr_dl),
                                                epochs=MAX_EPOCHS)
    history, best_val, best_state, best_ep, no_imp = [], -1, None, 0, 0
    t0 = time.time()

    def branch_probs(X):
        model.eval()
        ps, nov = [], []
        with torch.no_grad():
            for i in range(0, len(X), 256):
                o = model(torch.from_numpy(X[i:i + 256]).float().to(device))
                ps.append(torch.stack(o["probs"], 0).cpu().numpy())
                nov.append(o["novelty"].cpu().numpy())
        return np.concatenate(ps), np.concatenate(nov)

    for ep in range(MAX_EPOCHS):
        model.train()
        ep_loss = 0.0
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            o = model(xb)
            loss = sum(F.cross_entropy(l, yb) for l in o["branch_logits"]) / 4.0
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            ep_loss += float(loss) * len(xb)
        vp, _ = branch_probs(ds["Xva"])
        val_mf1 = _mf1(ds["y_val"], vp.mean(0).argmax(1), n_cls)
        history.append(dict(epoch=ep + 1, train_loss=round(ep_loss / len(ds["y_train"]), 6),
                            val_mf1_soft=round(val_mf1, 6)))
        improved = val_mf1 > best_val
        if improved:
            best_val, best_ep, no_imp = val_mf1, ep + 1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            no_imp += 1
        log(f"    [bench {tag}] Ep {ep+1:2d}/30 loss={ep_loss/len(ds['y_train']):.4f} "
            f"val_soft_mf1={val_mf1:.4f} best={best_val:.4f}")
        if no_imp >= PATIENCE:
            log(f"    [bench {tag}] early stop ep {ep+1}")
            break
    elapsed = time.time() - t0
    model.load_state_dict(best_state)

    # freeze -> fit combiners on VAL only (canonical protocol)
    val_probs, val_nov = branch_probs(ds["Xva"])
    fitted = fit_combiners_on_val(val_probs, ds["y_val"], val_nov, n_cls)
    combos_val = apply_combiners(val_probs, val_nov, n_cls, fitted)
    combo_val_mf1 = {k: _mf1(ds["y_val"], _preds(v), n_cls)
                     for k, v in combos_val.items()}
    best_combo = max(combo_val_mf1, key=combo_val_mf1.get)

    torch.save(dict(model_state_dict=model.state_dict(), epoch=len(history),
                    best_metric=best_val, best_ep=best_ep, time_s=elapsed,
                    sigma0=sigma0, num_classes=n_cls, sequence_length=ds["L"]),
               ckpt_path)
    torch.save(dict(fitted=fitted, best_by_val=best_combo,
                    combo_val_mf1=combo_val_mf1), comb_path)
    meta = dict(tag=tag, params=nparams, sigma0=sigma0, best_epoch=best_ep,
                best_val_soft_mf1=best_val, total_epochs=len(history),
                elapsed_s=round(elapsed, 1), seed=C.SEED,
                selected_combiner=best_combo,
                combiner_val_mf1={k: round(v, 4) for k, v in combo_val_mf1.items()},
                history=history)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log(f"    [bench {tag}] done: val_soft={best_val:.4f} best_ep={best_ep} "
        f"{elapsed:.0f}s combiner={best_combo}")
    return ckpt_path, meta


def evaluate_fresh(tag, ds, device):
    """Test evaluation ONCE with the val-selected combiner + all combiners."""
    from models.turs_stack.model import TURSStack
    ckpt = torch.load(os.path.join(C.CKPT_DIR, f"{tag}_turs_stack.pt"),
                      map_location="cpu", weights_only=False)
    model = TURSStack(in_channels=1, num_classes=ckpt["num_classes"],
                      sequence_length=ckpt["sequence_length"],
                      sigma_init=ckpt["sigma0"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    def forward(X):
        ps, nov, bs = [], [], []
        with torch.no_grad():
            for i in range(0, len(X), 256):
                o = model(torch.from_numpy(X[i:i + 256]).float().to(device))
                ps.append(torch.stack(o["probs"], 0).cpu().numpy())
                nov.append(o["novelty"].cpu().numpy())
                bs.append(torch.stack([l.float() for l in o["branch_logits"]], 0).cpu().numpy())
        return np.concatenate(ps, 1), np.concatenate(nov), np.concatenate(bs, 1)

    out = {}
    for split, X in [("val", ds["Xva"]), ("test", ds["Xte"]),
                     ("train", ds["Xtr"])]:
        out[split] = dict(zip(["probs", "novelty", "logits"], forward(X)))

    comb_path = os.path.join(C.CKPT_DIR, f"{tag}_combiners.pt")
    comb_art = torch.load(comb_path, map_location="cpu", weights_only=False)
    fitted, best_combo = comb_art["fitted"], comb_art["best_by_val"]

    y, n_cls = ds["y_test"], ds["n_cls"]
    combos_test = apply_combiners(out["test"]["probs"], out["test"]["novelty"],
                                  n_cls, fitted)
    branch_mf1 = {b: _mf1(y, out["test"]["probs"][k].argmax(1), n_cls)
                  for k, b in enumerate(C.BRANCH_NAMES)}
    official_pred = _preds(combos_test[best_combo])
    result = dict(
        tag=tag,
        selected_combiner=best_combo,
        final=full_metrics(y, official_pred, n_cls),
        branch_test_mf1=branch_mf1,
        combiners_test={k: round(_mf1(y, _preds(v), n_cls), 4)
                        for k, v in combos_test.items()},
        combiner_val_mf1={k: round(v, 4) for k, v in comb_art["combo_val_mf1"].items()},
        train_soft_acc=float((out["train"]["probs"].mean(0).argmax(1)
                              == ds["y_train"]).mean()),
    )
    S_ = None
    return result, model, out, combos_test, best_combo
