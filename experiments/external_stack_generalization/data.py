"""
Dataset loaders for the external generalization experiment.

Datasets (canonical UCR/aeon .ts files, downloaded from
timeseriesclassification.com/aeon-toolkit — the same canonical host the
repository uses for ECG5000):

  EpilepticSeizures : TRAIN 80 / PROVIDED val 20 / TEST 11420, L=178, 2 classes
  Haptics           : TRAIN 155 / TEST 308, L=1092, 5 classes (no provided val)
  Phoneme           : TRAIN 214 / TEST 1896, L=1024, 39 classes (no provided val)

Validation split policy (spec Phase 3):
  - EpilepticSeizures: USE the provided canonical val.ts (do not create a
    second validation split).
  - Haptics, Phoneme: stratified 15% of TRAIN (random_state=42). TEST is
    never touched.

Preprocessing (spec Phase 4): reuses the repository's canonical per-sample
z-normalization convention exactly (same formula as
experiments/run_turs_stack_benchmark.py::znorm and
experiments/benchmark_baselines.py::znorm).
"""
import os

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT, "data", "external", "extracted")

SEED = 42
VAL_FRAC = 0.15

DATASETS = ["EpilepticSeizures", "Haptics", "Phoneme"]
PRIMARY_DATASET = "EpilepticSeizures"

# cache of resolved .ts paths
_TS_PATHS = {}


def _find_ts(ds_name):
    """Locate the canonical TRAIN/TEST (and optional val) .ts files."""
    if ds_name in _TS_PATHS:
        return _TS_PATHS[ds_name]
    found = {}
    for root, _, files in os.walk(os.path.join(DATA_DIR, ds_name)):
        for f in files:
            lf = f.lower()
            if not lf.endswith(".ts"):
                continue
            p = os.path.join(root, f)
            if lf.endswith("_train.ts"):
                found["train"] = p
            elif lf.endswith("_test.ts"):
                found["test"] = p
            elif lf == "val.ts":
                found["val"] = p
    missing = {"train", "test"} - set(found)
    if missing:
        raise FileNotFoundError(
            f"{ds_name}: canonical .ts files not found (missing {missing}) "
            f"under {DATA_DIR}")
    _TS_PATHS[ds_name] = found
    return found


def znorm(X):
    """Canonical per-sample z-normalization (project convention)."""
    mu = X.mean(axis=-1, keepdims=True)
    sig = X.std(axis=-1, keepdims=True) + 1e-8
    return ((X - mu) / sig).astype(np.float32)


def _to_2d(X):
    """aeon returns [N, channels, L]; the project works on [N, L] univariate."""
    X = np.asarray(X)
    if X.dtype == object:
        X = np.stack([np.asarray(xi, dtype=np.float64) for xi in X])
    if X.ndim == 3:
        assert X.shape[1] == 1, f"expected univariate, got {X.shape[1]} channels"
        X = X[:, 0, :]
    return X.astype(np.float32)


def _load_split(path):
    from aeon.datasets import load_from_ts_file
    X, y = load_from_ts_file(path)
    X = _to_2d(X)
    y = np.asarray(y).astype(str)
    # canonical class order: sorted label strings -> contiguous ints
    classes = sorted(np.unique(y).tolist())
    cmap = {c: i for i, c in enumerate(classes)}
    y = np.array([cmap[v] for v in y], dtype=np.int64)
    return X, y, classes


def stratified_val_split(y, val_frac, seed, min_class_for_val=2):
    """Deterministic stratified val allocation that tolerates singleton
    classes (Phoneme has 8 classes with exactly 1 train member, which makes
    sklearn's stratified split impossible).

    Rule: per class c with n_c members, n_val = round(val_frac * n_c) if
    n_c >= min_class_for_val else 0; val members chosen by a seeded RNG
    permutation; a class is never emptied (n_val <= n_c - 1 enforced).
    Documented, non-leaky (uses labels + indices only, no test data).
    """
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    val_idx = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        perm = rng.permutation(len(idx))
        idx = idx[perm]
        n_val = int(round(val_frac * len(idx))) if len(idx) >= \
            min_class_for_val else 0
        n_val = min(n_val, len(idx) - 1)
        val_idx.extend(idx[:n_val].tolist())
    mask = np.zeros(len(y), dtype=bool)
    mask[val_idx] = True
    return ~mask, mask


def load_dataset(ds_name, audit=False):
    """Load one external dataset with the frozen split policy.

    Returns dict with Xtr/Xva/Xte (raw, unnormalized), ytr/yva/yte,
    class_names, and provenance flags.
    """
    paths = _find_ts(ds_name)
    Xtr, ytr, classes = _load_split(paths["train"])
    Xte, yte, _ = _load_split(paths["test"])

    if "val" in paths:
        # PHASE 3: canonical provided validation split — use it as-is.
        Xva, yva, cls_v = _load_split(paths["val"])
        assert cls_v == classes, "provided val.ts has different class space"
        val_source = "provided_canonical_val_ts"
    else:
        tr_m, va_m = stratified_val_split(ytr, VAL_FRAC, SEED)
        Xva, yva = Xtr[va_m], ytr[va_m]
        Xtr, ytr = Xtr[tr_m], ytr[tr_m]
        val_source = (f"per_class_stratified_{int(VAL_FRAC*100)}pct_of_train_"
                      f"seed{SEED}_singletons_kept_in_train")

    out = {
        "name": ds_name,
        "Xtr": Xtr, "ytr": ytr,
        "Xva": Xva, "yva": yva,
        "Xte": Xte, "yte": yte,
        "class_names": classes,
        "n_classes": len(classes),
        "L": int(Xtr.shape[1]),
        "val_source": val_source,
        "paths": paths,
    }
    if audit:
        out["audit"] = dataset_audit_stats(out)
    return out


def dataset_audit_stats(d):
    """PHASE 2 integrity numbers for one loaded dataset."""
    def stats(X, y):
        Xd = X.reshape(len(X), -1)
        uni, cnt = np.unique(y, return_counts=True)
        return {
            "n": int(len(X)),
            "class_distribution": {str(c): int(n) for c, n in zip(uni, cnt)},
            "nan_count": int(np.isnan(Xd).sum()),
            "inf_count": int(np.isinf(Xd).sum()),
            "duplicated_samples": int(len(Xd) - len(np.unique(Xd, axis=0))),
        }
    return {
        "train": stats(d["Xtr"], d["ytr"]),
        "val": stats(d["Xva"], d["yva"]),
        "test": stats(d["Xte"], d["yte"]),
        "val_source": d["val_source"],
        "sequence_length": d["L"],
        "n_classes": d["n_classes"],
        "univariate": True,
        "class_names": d["class_names"],
    }


def class_entropy(y):
    """Shannon entropy (bits) of the label distribution."""
    _, cnt = np.unique(y, return_counts=True)
    p = cnt / cnt.sum()
    return float(-(p * np.log2(p)).sum())
