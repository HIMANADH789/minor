"""Verification of the Kaggle UCRArchive_2018 copies against canonical aeon data.

Loads each dataset from both sources and compares, without any normalization
or other alteration of the raw values:

    1. structural agreement: shapes, sequence length, class count, label sets
    2. content agreement: sorted per-sample SHA-256 hashes of the raw values
       (ordering-independent), plus a nearest-row match and max/mean abs diff
    3. label agreement: canonical UCR labels, with any encoding mapping
       reported explicitly

Run:  python -m experiments.kaggle_ucr_context_datasets.verify
"""
import hashlib
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
KAGGLE_ROOT = os.path.join(ROOT, "data", "kaggle")
OUT_DIR = os.path.join(ROOT, "results", "kaggle_ucr_context_datasets")

DATASETS = {
    "GunPoint": {
        "kaggle_dir": "GunPoint", "train": "GunPoint_TRAIN.tsv", "test": "GunPoint_TEST.tsv",
        "expected": {"train": 50, "test": 150, "T": 150, "n_classes": 2},
    },
    "ItalyPowerDemand": {
        "kaggle_dir": "ItalyPowerDemand", "train": "ItalyPowerDemand_TRAIN.tsv",
        "test": "ItalyPowerDemand_TEST.tsv",
        "expected": {"train": 67, "test": 1029, "T": 24, "n_classes": 2},
    },
    "FordA": {
        "kaggle_dir": "FordA", "train": "FordA_TRAIN.tsv", "test": "FordA_TEST.tsv",
        "expected": {"train": 3601, "test": 1320, "T": 500, "n_classes": 2},
    },
}


def load_ucr_tsv(path):
    """UCRArchive_2018 TSV: first column = class label, remaining = values,
    tab-separated, NaN allowed."""
    raw = np.loadtxt(path, delimiter="\t", ndmin=2)
    return raw[:, 0], raw[:, 1:]


def sample_hashes(X, y):
    """Deterministic per-sample hash of (label, raw values); ordering-independent."""
    out = []
    for i in range(X.shape[0]):
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(X[i], dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(np.array([y[i]]), dtype=np.float64).tobytes())
        out.append(h.hexdigest())
    return out


def verify_one(name):
    spec = DATASETS[name]
    k_dir = os.path.join(KAGGLE_ROOT, spec["kaggle_dir"])
    yk_tr, Xk_tr = load_ucr_tsv(os.path.join(k_dir, spec["train"]))
    yk_te, Xk_te = load_ucr_tsv(os.path.join(k_dir, spec["test"]))

    from aeon.datasets import load_classification
    Xa, ya = load_classification(name)
    # aeon returns (n, 1, T); flatten channel dim for these univariate datasets
    Xa2 = np.asarray(Xa, dtype=np.float64)[:, 0, :]
    ya2 = np.asarray(ya)
    # aeon canonical split: first n_train rows = train
    n_tr = spec["expected"]["train"]
    Xa_tr, ya_tr = Xa2[:n_tr], ya2[:n_tr]
    Xa_te, ya_te = Xa2[n_tr:], ya2[n_tr:]

    res = {"dataset": name,
           "kaggle_path": os.path.relpath(k_dir, ROOT).replace("\\", "/")}

    # ---- structure ----
    struct = {}
    for split, (Xk, yk, Xa_s, ya_s) in {
        "train": (Xk_tr, yk_tr, Xa_tr, ya_tr), "test": (Xk_te, yk_te, Xa_te, ya_te)
    }.items():
        assert Xk.shape[1] == Xa_s.shape[1], \
            f"{name} {split}: T mismatch {Xk.shape[1]} vs {Xa_s.shape[1]}"
        struct[split] = {
            "kaggle_shape": [int(Xk.shape[0]), int(Xk.shape[1])],
            "aeon_shape": [int(Xa_s.shape[0]), int(Xa_s.shape[1])],
            "kaggle_label_dtype": str(yk.dtype),
            "aeon_label_dtype": str(np.asarray(ya_s).dtype),
            "kaggle_labels": sorted(set(yk.tolist())),
            "aeon_labels": sorted(set(ya_s.tolist())),
        }
        # aeon returns UCR labels as STRINGS ('1', '2', '-1'); the TSV parses
        # to floats. Same canonical label set, different encoding -> compare
        # via the explicit mapping float->int->str (documented in the report).
        struct[split]["labels_match_via_mapping"] = bool(
            {f"{int(v)}" for v in yk} == {str(v) for v in ya_s})
        assert Xk.shape[0] == Xa_s.shape[0] and Xk.shape[1] == Xa_s.shape[1], \
            f"{name} {split}: shape mismatch {Xk.shape} vs {Xa_s.shape}"
        struct[split]["n_match"] = True
        struct[split]["T_match"] = True
    res["structure"] = struct
    res["n_classes"] = int(len(set(yk_tr.tolist()) | set(yk_te.tolist())))

    # ---- content: sorted per-sample hashes (ordering-independent) ----
    content = {}
    for split, (Xk, yk, Xa_s, ya_s) in {
        "train": (Xk_tr, yk_tr, Xa_tr, ya_tr), "test": (Xk_te, yk_te, Xa_te, ya_te)
    }.items():
        hk = sorted(sample_hashes(Xk.astype(np.float64), yk.astype(np.float64)))
        ha = sorted(sample_hashes(Xa_s, ya_s))
        n_hash_match = int(len(set(hk) & set(ha)))
        content[split] = {
            "hash_identical_multiset": bool(hk == ha),
            "n_hash_match": n_hash_match, "n_samples": len(hk),
        }
    res["content_hashes"] = content

    # ---- content: nearest-row numeric comparison (first 100 kaggle rows) ----
    # handles float32-vs-float64 serialization differences; reports mapping too
    numeric = {}
    label_map = set()
    for split, (Xk, yk, Xa_s, ya_s) in {
        "train": (Xk_tr, yk_tr, Xa_tr, ya_tr), "test": (Xk_te, yk_te, Xa_te, ya_te)
    }.items():
        diffs, matched = [], 0
        for i in range(Xk.shape[0]):          # ALL rows, not a subsample
            d = np.abs(Xa_s - Xk[i]).max(axis=1)
            j = int(d.argmin())
            diffs.append(float(d[j]))
            if float(d[j]) == 0.0:
                matched += 1
            label_map.add((float(yk[i]), str(ya_s[j])))
        diffs = np.array(diffs)
        numeric[split] = {
            "n_compared": int(len(diffs)), "n_exact_match": matched,
            "max_abs_diff": float(diffs.max()), "mean_abs_diff": float(diffs.mean()),
            "all_rows_exactly_matched_once": bool(matched == len(diffs) == Xa_s.shape[0]),
        }
    res["numeric_nearest_row"] = numeric
    res["label_pairs_observed"] = sorted(label_map)

    # ---- verdict ----
    ok = all(
        struct[s]["n_match"] and struct[s]["T_match"]
        and struct[s]["labels_match_via_mapping"]
        and content[s]["hash_identical_multiset"] for s in ("train", "test"))
    ok = ok and all(numeric[s]["all_rows_exactly_matched_once"] for s in ("train", "test"))
    res["verified"] = bool(ok)
    res["expected"] = spec["expected"]
    return res


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = [verify_one(name) for name in DATASETS]
    with open(os.path.join(OUT_DIR, "verification_details.json"), "w") as f:
        json.dump(results, f, indent=2)
    for r in results:
        print(f"{r['dataset']}: VERIFIED={r['verified']} "
              f"train={r['structure']['train']['kaggle_shape']} "
              f"test={r['structure']['test']['kaggle_shape']} "
              f"labels={r['label_pairs_observed'][:4]}")
    return results


if __name__ == "__main__":
    main()
