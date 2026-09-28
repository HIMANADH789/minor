"""UWave dataset location, loading, verification, and split management.

Finds the four local UWave .ts dataset copies (project root or its parent),
verifies the canonical UCR structure from the actual files, converts labels
to contiguous class IDs preserving sorted original-label order, creates the
established stratified 15%-of-TRAIN validation split (seed 42), and saves
the split indices for reuse by both M0 and R2.
"""
import os

import numpy as np
from sklearn.model_selection import train_test_split

from experiments.rcmkn_r2_uwave_seed42.config import CANONICAL, ROOT, SEED, VAL_FRAC

_DATASET_DIRS = [
    ROOT,                                   # inside the repo
    os.path.dirname(ROOT),                  # main project directory (parent)
]

_TS_PATHS = {}


def find_datasets():
    """Locate the four local UWave folders / .ts file pairs.

    The user instruction names folders (UWaveGestureLibraryAll/ ...); the
    local copies are flat .ts file pairs (<NAME>_TRAIN.ts / <NAME>_TEST.ts)
    in the main project directory. Both layouts are accepted; original files
    are never modified.
    """
    global _TS_PATHS
    found = {}
    for ds in CANONICAL:
        paths = {}
        for base in _DATASET_DIRS:
            folder = os.path.join(base, ds)
            flat = base
            for root_dir in (folder, flat):
                tr = os.path.join(root_dir, f"{ds}_TRAIN.ts")
                te = os.path.join(root_dir, f"{ds}_TEST.ts")
                if os.path.exists(tr) and os.path.exists(te):
                    paths = {"train": tr, "test": te,
                             "layout": ("folder" if root_dir == folder
                                        else "flat")}
                    break
            if paths:
                break
        if not paths:
            raise FileNotFoundError(
                f"{ds}: no <{ds}_TRAIN.ts>/<{ds}_TEST.ts> pair found in "
                f"{_DATASET_DIRS}")
        found[ds] = paths
    _TS_PATHS = found
    return found


def _to_2d(X):
    """aeon returns (n, 1, T) for univariate .ts; flatten to (n, T)."""
    X = np.asarray(X)
    if X.ndim == 3 and X.shape[1] == 1:
        return X[:, 0, :]
    if X.ndim == 2:
        return X
    raise ValueError(f"unexpected aeon shape {X.shape}")


def load_ucr_split(path):
    """Load one .ts split; map sorted original labels to 0..K-1."""
    from aeon.datasets import load_from_ts_file
    X, y = load_from_ts_file(path)
    X = _to_2d(X).astype(np.float32)
    y = np.asarray(y).astype(str)
    classes = sorted(np.unique(y).tolist())
    cmap = {c: i for i, c in enumerate(classes)}
    y = np.array([cmap[v] for v in y], dtype=np.int64)
    return X, y, classes, cmap


def znorm(X):
    """Per-sample z-normalization (the repository's canonical convention)."""
    mu = X.mean(axis=-1, keepdims=True)
    sig = X.std(axis=-1, keepdims=True) + 1e-8
    return ((X - mu) / sig).astype(np.float32)


def resolve_path(rel_or_abs):
    """Resolve a recorded relative path back to an absolute file path."""
    if os.path.isabs(rel_or_abs):
        return rel_or_abs
    return os.path.join(ROOT, rel_or_abs)


def load_and_split(ds_name, out_dir):
    """Load + verify + split one UWave dataset. Returns data dict and audit.

    Split policy (established R2 transfer protocol):
      official UCR TRAIN (896) -> stratified 15% val (seed 42) + internal train
      official UCR TEST (3582) -> untouched official test
    Indices are saved as train_indices.npy / val_indices.npy /
    test_indices.npy (test indices = arange(n_test), i.e. the official set).
    """
    paths = _TS_PATHS.get(ds_name) or find_datasets()[ds_name]
    Xtr_raw, ytr_raw, classes, cmap = load_ucr_split(paths["train"])
    Xte_raw, yte_raw, classes_te, _ = load_ucr_split(paths["test"])

    spec = CANONICAL[ds_name]
    got = {"train": len(Xtr_raw), "test": len(Xte_raw), "T": Xtr_raw.shape[1],
           "n_classes": len(classes)}
    assert got == spec, \
        f"{ds_name}: canonical mismatch {got} != {spec} -- STOP"
    assert classes == classes_te, f"{ds_name}: train/test label sets differ"

    # stratified 15% of the official TRAIN, seed 42 (indices into the
    # ORIGINAL 896-row train array)
    idx = np.arange(len(Xtr_raw))
    tr_idx, va_idx = train_test_split(
        idx, test_size=VAL_FRAC, stratify=ytr_raw, random_state=SEED)
    tr_idx, va_idx = np.sort(tr_idx), np.sort(va_idx)

    Xtr, ytr = Xtr_raw[tr_idx], ytr_raw[tr_idx]
    Xva, yva = Xtr_raw[va_idx], ytr_raw[va_idx]
    Xte, yte = Xte_raw, yte_raw
    assert not set(tr_idx.tolist()) & set(va_idx.tolist()), "val/train overlap"
    assert len(Xtr) + len(Xva) == spec["train"]

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "train_indices.npy"), tr_idx)
    np.save(os.path.join(out_dir, "val_indices.npy"), va_idx)
    np.save(os.path.join(out_dir, "test_indices.npy"),
            np.arange(len(Xte_raw)))

    # class counts (original label -> count)
    def counts(y):
        return {classes[i]: int((y == i).sum()) for i in range(len(classes))}
    audit = {
        "dataset": ds_name,
        "paths": {k: (os.path.relpath(v, ROOT) if k != "layout" else v)
                  for k, v in paths.items()},
        "canonical_spec": spec,
        "verified": got,
        "original_labels": classes,
        "label_map": {str(k): v for k, v in cmap.items()},
        "class_counts_train": counts(ytr),
        "class_counts_val": counts(yva),
        "class_counts_test": counts(yte),
        "split": {"internal_train": len(Xtr), "val": len(Xva),
                  "official_test": len(Xte)},
        "val_source": f"stratified_{int(VAL_FRAC*100)}pct_of_train_seed{SEED}",
    }
    return {"Xtr": Xtr, "ytr": ytr, "Xva": Xva, "yva": yva,
            "Xte": Xte, "yte": yte, "n_classes": len(classes),
            "T": got["T"], "classes": classes, "audit": audit}
