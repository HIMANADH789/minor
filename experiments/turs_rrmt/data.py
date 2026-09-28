"""Canonical data protocol for TURS-RRMT (identical to the established
biomedical benchmark: seed 42, stratified splits, per-sample z-norm)."""
import hashlib
import os

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

SEED = 42
VAL_FRAC = 0.15
NUM_CLASSES = {"ECG5000_UNBAL": 5, "ECG5000_BAL": 5, "CWRU_UNBAL": 4, "CWRU_BAL": 4}
DATA_FILE = {
    "ECG5000_UNBAL": "data/ecg5000_resplit.npz",
    "ECG5000_BAL": "data/ecg5000_fair_balanced.npz",
    "CWRU_UNBAL": "data/cwru_unbalanced.npz",
    "CWRU_BAL": "data/cwru_balanced.npz",
}
DATASETS = list(NUM_CLASSES)


def znorm(X):
    mu = X.mean(axis=-1, keepdims=True)
    sig = X.std(axis=-1, keepdims=True) + 1e-8
    return ((X - mu) / (sig + 1e-8)).astype(np.float32)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def load_split(tag):
    """Exact canonical split semantics (same as run_turs_stack_benchmark.py)."""
    path = os.path.join(ROOT, DATA_FILE[tag])
    data = np.load(path)
    if "X_train" in data:
        X_trval, y_trval = data["X_train"].astype(np.float32), data["y_train"].astype(np.int64)
        X_te, y_te = data["X_test"].astype(np.float32), data["y_test"].astype(np.int64)
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_trval, y_trval, test_size=VAL_FRAC / 0.85, random_state=SEED,
            stratify=y_trval)
    else:
        X_all, y_all = data["X"].astype(np.float32), data["y"].astype(np.int64)
        idx = np.arange(len(X_all))
        trval_idx, te_idx = train_test_split(idx, test_size=0.15, random_state=SEED,
                                             stratify=y_all)
        X_trval, y_trval = X_all[trval_idx], y_all[trval_idx]
        X_te, y_te = X_all[te_idx], y_all[te_idx]
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_trval, y_trval, test_size=VAL_FRAC / 0.85, random_state=SEED,
            stratify=y_trval)
    return dict(tag=tag, n_cls=NUM_CLASSES[tag], L=int(X_tr.shape[1]),
                Xtr=znorm(X_tr)[:, None, :], Xva=znorm(X_va)[:, None, :],
                Xte=znorm(X_te)[:, None, :],
                y_train=y_tr, y_val=y_va, y_test=y_te,
                data_sha256=_sha256(path))


def manifest(tag, ds):
    n_cls = ds["n_cls"]
    return dict(dataset=tag, source=DATA_FILE[tag], sha256=ds["data_sha256"],
                signal_length=ds["L"], num_classes=n_cls,
                split=dict(train=int(len(ds["y_train"])), val=int(len(ds["y_val"])),
                           test=int(len(ds["y_test"]))),
                class_counts={k: np.bincount(ds[k], minlength=n_cls).tolist()
                              for k in ["y_train", "y_val", "y_test"]},
                preprocessing="per-sample z-norm (eps 1e-8)",
                seed=SEED, deterministic=True)
