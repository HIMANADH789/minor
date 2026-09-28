"""Data loading + dataset integrity manifest (Phase 2).
Split semantics copied EXACTLY from experiments/fair_turs.py main():"""
import json
import os

import numpy as np
from sklearn.model_selection import train_test_split

from . import config as C


def znorm(X):
    mu = X.mean(axis=-1, keepdims=True)
    sig = X.std(axis=-1, keepdims=True) + 1e-8
    return ((X - mu) / sig).astype(np.float32)


def load_split(tag):
    """Canonical split. Returns dict with raw + normalized + channel data."""
    path = os.path.join(C.ROOT, C.DATA_FILE[tag])
    data = np.load(path)
    n_cls = C.NUM_CLASSES[tag]
    if "X_train" in data:
        X_all, y_all = data["X_train"], data["y_train"].astype(int)
        X_test, y_test = data["X_test"], data["y_test"].astype(int)
        X_train, X_val, y_train, y_val = train_test_split(
            X_all, y_all, test_size=VAL_FRAC_, stratify=y_all, random_state=SEED_)
    else:
        X_all, y_all = data["X"], data["y"].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.15, stratify=y_all, random_state=SEED_)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=VAL_FRAC_, stratify=y_train, random_state=SEED_)

    X_train_n, X_val_n, X_test_n = znorm(X_train), znorm(X_val), znorm(X_test)
    L = X_train.shape[1]
    labels = np.unique(np.concatenate([y_train, y_val, y_test]))
    manifest = dict(
        dataset=tag,
        source_file=C.DATA_FILE[tag],
        source_sha256=_sha256(path),
        n_samples_total=int(len(X_all) if "X_train" in data else len(data["X"])),
        signal_length=int(L),
        num_classes=int(n_cls),
        label_mapping={"labels_present": labels.tolist(),
                       "contiguous": bool(np.array_equal(labels, np.arange(n_cls)))},
        split={"protocol": ("official_train/test + val carved from train "
                            if "X_train" in data else "stratified 85/15 then val"),
               "seed": SEED_, "val_frac_of_train": VAL_FRAC_,
               "train": int(len(X_train)), "val": int(len(X_val)),
               "test": int(len(X_test))},
        class_counts={
            "train": np.bincount(y_train, minlength=n_cls).tolist(),
            "val": np.bincount(y_val, minlength=n_cls).tolist(),
            "test": np.bincount(y_test, minlength=n_cls).tolist()},
        normalization="per-sample z-normalization (mean/std over time, eps=1e-8)",
        leakage_checks={
            "test_disjoint_from_trainval": True,   # by construction (index splits)
            "split_deterministic_given_seed": True,
            "normalization_fitted_per_sample": "no train-statistics leakage",
        },
        raw_examples_shapes={
            "train": list(X_train.shape), "val": list(X_val.shape),
            "test": list(X_test.shape)},
    )
    return dict(
        tag=tag, n_cls=n_cls, L=int(L),
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        Xtr=X_train_n[:, None, :], Xva=X_val_n[:, None, :], Xte=X_test_n[:, None, :],
        manifest=manifest,
    )


SEED_ = C.SEED
VAL_FRAC_ = C.VAL_FRAC


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(tag, manifest):
    path = os.path.join(C.AUDIT_DIR, f"dataset_manifest_{tag}.json")
    os.makedirs(C.AUDIT_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path
