"""Data split (EXACT run_turs_stack_benchmark.py semantics), checkpoint
loading, and Phase-2 deterministic replay verification.

Split semantics (run_turs_stack_benchmark.py lines 415-433):
  official split files: val carved with test_size=VAL_FRAC/0.85, seed 42
  no official split:    85/15 stratified trainval/test, then val carved
Per-sample z-normalization (mean/std over time, eps 1e-8), then [N,1,L].
"""
import hashlib
import json
import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from . import config as C


def znorm(X):
    mu = X.mean(axis=-1, keepdims=True)
    sig = X.std(axis=-1, keepdims=True) + 1e-8
    return ((X - mu) / (sig + 1e-8)).astype(np.float32)


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
                Xte=znorm(X_te)[:, None, :],
                y_train=y_tr, y_val=y_va, y_test=y_te,
                raw_shapes=dict(train=list(X_tr.shape), val=list(X_va.shape),
                                test=list(X_te.shape)),
                data_sha256=_sha256(path))


def load_model(tag, device):
    from models.turs_stack.model import TURSStack
    ckpt_path = os.path.join(C.CKPT_DIR, f"{tag}_turs_stack.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = TURSStack(in_channels=1, num_classes=ckpt["num_classes"],
                      sequence_length=ckpt["sequence_length"],
                      sigma_init=ckpt["sigma0"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt


def load_combiners(tag):
    path = os.path.join(C.CKPT_DIR, f"{tag}_combiners.pt")
    return torch.load(path, map_location="cpu", weights_only=False)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _mf1(y, pred, n_cls):
    return float(f1_score(y, pred, average="macro", zero_division=0,
                          labels=list(range(n_cls))))


def _forward_all(model, X, device, batch=256):
    """Forward over X in equal batches; returns stacked per-output arrays
    where every output has shape [4, N, ...] aligned across outputs."""
    n = len(X)
    bs = min(batch, n)
    chunks = []
    with torch.no_grad():
        for i in range(0, n, bs):
            chunks.append(model(torch.from_numpy(X[i:i + bs]).float().to(device)))
    out = {}
    out["probs"] = np.stack([np.stack(torch.stack(
        [torch.stack(c["probs"], 0) for c in chunks]).numpy(), 0)]) if False else \
        np.concatenate([torch.stack(c["probs"], 0).cpu().numpy() for c in chunks], axis=1)
    out["novelty"] = np.concatenate([c["novelty"].cpu().numpy() for c in chunks])
    out["beta_cs"] = np.concatenate([c["beta_cs"].cpu().numpy() for c in chunks])
    out["beta_cmr"] = np.concatenate([c["beta_cmr"].cpu().numpy() for c in chunks])
    return out


def replay_dataset(tag, model, ds, device, tol=5e-4):
    """Phase 2: reproduce saved benchmark metrics from frozen checkpoint."""
    saved = json.load(open(os.path.join(C.STACK_RESULTS, tag, "full_results.json")))

    # forward once over all splits (batch size divides N so outputs align)
    out = {}
    for split, X in [("train", ds["Xtr"]), ("val", ds["Xva"]), ("test", ds["Xte"])]:
        out[split] = _forward_all(model, X, device)

    y = ds["y_test"]
    P = out["test"]["probs"]                      # [4, N, C]
    n_cls = ds["n_cls"]
    branch_mf1 = {b: _mf1(y, P[k].argmax(1), n_cls) for k, b in enumerate(C.BRANCH_NAMES)}
    soft = _mf1(y, P.mean(0).argmax(1), n_cls)
    hard_votes = P.argmax(-1)                     # [4, N]
    onehot = np.eye(n_cls)[hard_votes]
    counts = onehot.sum(0)
    maxv = counts.max(1, keepdims=True)
    conf = P.sum(0)
    hard = np.where(counts == maxv, conf, -1).argmax(1)
    hard_mf1 = _mf1(y, hard, n_cls)

    checks = []
    def add(name, saved_v, replay_v):
        sv = saved.get("combination_test_mf1", {}).get(name) if name in (
            "soft_vote", "hard_vote", "stacking", "diagnostic_stacking",
            "static_weights") else None
        if sv is None:
            return
        diff = abs(replay_v - sv)
        checks.append(dict(metric=f"combination_test_mf1.{name}",
                           saved=sv, replayed=round(replay_v, 6),
                           difference=round(diff, 8), tolerance=tol,
                           passed=bool(diff <= tol)))

    for k, b in enumerate(C.BRANCH_NAMES):
        sv = saved.get("branch_test_mf1", {}).get(b)
        if sv is not None:
            diff = abs(branch_mf1[b] - sv)
            checks.append(dict(metric=f"branch_test_mf1.{b}", saved=sv,
                               replayed=round(branch_mf1[b], 6),
                               difference=round(diff, 8), tolerance=tol,
                               passed=bool(diff <= tol)))
    add("soft_vote", None, soft)
    add("hard_vote", None, hard_mf1)

    replay = dict(dataset=tag, branch_test_mf1=branch_mf1,
                  soft_vote_mf1=soft, hard_vote_mf1=hard_mf1,
                  soft_vote_accuracy=float((P.mean(0).argmax(1) == y).mean()),
                  checks=checks,
                  all_passed=bool(all(c["passed"] for c in checks)),
                  saved_best_combiner=saved.get("best_combiner_by_val"),
                  test_softvote_accuracy_saved=saved.get("test_softvote_accuracy"))
    os.makedirs(C.REPLAY_DIR, exist_ok=True)
    with open(os.path.join(C.REPLAY_DIR, f"{tag}_replay.json"), "w") as f:
        json.dump(replay, f, indent=2)
    return replay, out
