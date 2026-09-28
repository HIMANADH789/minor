"""
TURS-SKB: Structured Kernel Bank
=================================
Can mathematically structured FIXED temporal kernels provide complementary
information to canonical MiniROCKET, while retaining the efficient
fixed-feature + linear-readout framework?

Models (spec sec. 2), all sharing the SAME canonical MiniROCKET base:
  M0 = MiniROCKET 10K (canonical reference)
  M1..M5 = M0 + one structured family (morph/deriv/wavelet/gabor/energy)
  M6  = M0 + all five families
  Pair models = M0 + two promising families (selected on VALIDATION only)

Downstream philosophy unchanged: fixed kernel bank -> response -> threshold
(PPV) -> concat -> block standardization -> one RidgeClassifierCV.
NO router / gate / attention / adaptation / MoE / stacking / learned weights.
0 trainable neural parameters; kernel grids fully predefined and
deterministic; thresholds mirror aeon MiniROCKET's 9 golden-ratio quantile
biases fitted on TRAIN responses only.

Protocol (frozen before any result):
  seed 42 (primary), canonical benchmark split, znorm, MiniROCKET fit on
  TRAIN, ridge fit on TRAIN+VAL (identical to results/baseline_bench),
  single test evaluation per model. Screening (pair selection) uses
  train-fitted ridge + VALIDATION MF1 only. Multi-seed robustness:
  seeds 42-46 for M0 vs M6 (family features are split-deterministic and
  identical across seeds; only MiniROCKET's random draw varies).

Output: results/turs_skb/
"""
import csv
import json
import math
import os
import shutil
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.features.turs_skb.features import (
    SKBExtractor, BlockStandardizer, assemble_blocks, linear_cka,
    mean_abs_cross_correlation,
)
from src.features.turs_skb.kernels import (
    SIGMA_GRID, DILATION_GRID, ALPHA_DO_G_GRID, DERIV_GRID, WAVELET_TYPES,
    WAVELET_SCALES, GABOR_F_GRID, GABOR_SIGMA_GRID, GABOR_PHASE_GRID,
    ENERGY_W_GRID, ENERGY_LAMBDA_GRID, BANK_BUILDERS,
)
from src.diagnostics.statistics import (
    mcnemar, benjamini_hochberg, paired_permutation_test, cohens_d_paired,
)

# ============================================================================
# FROZEN CONFIGURATION (pre-registered; never changed after seeing results)
# ============================================================================
SEED = 42
SEEDS_ROBUST = [42, 43, 44, 45, 46]   # spec sec. 25/28
N_KERNELS = 10000
BASELINE_TOLERANCE = 0.02             # M0 verification gate (sec. 4)
ALPHA = 0.05
FAMILIES = ["morphological", "derivative", "wavelet", "gabor", "energy"]
FAM_TAG = {"morphological": "Morph", "derivative": "Deriv",
           "wavelet": "Wavelet", "gabor": "Gabor", "energy": "Energy"}

DATASETS = [
    ("ECG5000_UNBAL", "data/ecg5000_resplit.npz"),
    ("ECG5000_BAL",   "data/ecg5000_fair_balanced.npz"),
    ("CWRU_UNBAL",    "data/cwru_unbalanced.npz"),
    ("CWRU_BAL",      "data/cwru_balanced.npz"),
]
REF_VALUES = {
    "ECG5000_UNBAL": 0.5938,
    "ECG5000_BAL":   0.6553,
    "CWRU_UNBAL":    0.9917,
    "CWRU_BAL":      0.9947,
}

# Pre-registered representative kernels for the response-shape figure
# (fixed BEFORE any performance result was seen; spec sec. 23: illustrate
# what each family mathematically detects - no cherry-picking).
REPRESENTATIVE_KERNELS = {
    "morphological": ("dog", {"sigma": 4.0, "alpha": 1.0, "dilation": 1}),
    "derivative": ("deriv", {"order": 1, "dilation": 4}),
    "wavelet": ("wavelet", {"type": "mexicanhat", "scale": 4, "dilation": 1}),
    "gabor": ("gabor", {"freq": 1.0 / 16, "sigma": 8.0, "phase": 0.0,
                        "dilation": 1}),
    "energy": ("energy", {"w_short": 8, "lambda": 1.0}),
}

# Pre-registered mechanistic expectations (spec sec. 34)
MECH_EXPECTATIONS = {
    "morphological": (["ECG5000_UNBAL", "ECG5000_BAL"],
                      "shape-rich ECG beat morphology"),
    "derivative": (["ECG5000_UNBAL", "ECG5000_BAL", "CWRU_UNBAL",
                    "CWRU_BAL"], "transitions/edges (QRS edges, fault steps)"),
    "wavelet": (["ECG5000_UNBAL", "ECG5000_BAL"],
                "multiscale morphology of ECG beats"),
    "gabor": (["CWRU_UNBAL", "CWRU_BAL"],
              "oscillatory vibration structure of bearings"),
    "energy": (["CWRU_UNBAL", "CWRU_BAL"], "impulsive/burst fault energy"),
}

OUT_DIR = os.environ.get("TURS_SKB_OUT_DIR",
                         os.path.join(ROOT, "results", "turs_skb"))
CACHE_DIR = os.path.join(OUT_DIR, "cache")
for sub in ("configs", "logs", "figures", "tests", "reports", "cache"):
    os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)

_SUBSET = os.environ.get("TURS_SKB_DATASETS")
if _SUBSET:
    _want = {s.strip() for s in _SUBSET.split(",") if s.strip()}
    DATASETS = [d for d in DATASETS if d[0] in _want]

RUN_MULTISEED = os.environ.get("TURS_SKB_SKIP_MULTISEED", "") != "1"

LOG_PATH = os.path.join(OUT_DIR, "logs", "full_run_console.log")
_logf = open(LOG_PATH, "a", encoding="utf-8")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    _logf.write(line + "\n")
    _logf.flush()


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)


def write_csv(path, rows):
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


# ============================================================================
# Canonical benchmark protocol helpers (verbatim conventions)
# ============================================================================
def znorm(X):
    mu = X.mean(axis=-1, keepdims=True)
    sig = X.std(axis=-1, keepdims=True) + 1e-8
    return ((X - mu) / sig).astype(np.float32)


def load_split(ds_file):
    """Canonical benchmark split (identical to benchmark_baselines.py)."""
    data = np.load(os.path.join(ROOT, ds_file))
    if "X_train" in data:
        Xa, ya = data["X_train"], data["y_train"].astype(int)
        Xte, yte = data["X_test"], data["y_test"].astype(int)
        Xtr, Xva, ytr, yva = train_test_split(
            Xa, ya, test_size=0.15, stratify=ya, random_state=SEED)
    else:
        Xa, ya = data["X"], data["y"].astype(int)
        Xtr, Xte, ytr, yte = train_test_split(
            Xa, ya, test_size=0.15, stratify=ya, random_state=SEED)
        Xtr, Xva, ytr, yva = train_test_split(
            Xtr, ytr, test_size=0.15, stratify=ytr, random_state=SEED)
    return Xtr, Xva, Xte, ytr, yva, yte


def full_metrics(y_eval, preds, n_features, ridge_fit_s, infer_s, alpha):
    n_cls = int(max(int(y_eval.max()), 0)) + 1
    labels = list(range(n_cls))
    return {
        "accuracy": round(float(accuracy_score(y_eval, preds)), 4),
        "macro_f1": round(float(f1_score(y_eval, preds, average="macro",
                                         zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_eval, preds,
                                            average="weighted",
                                            zero_division=0)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_eval,
                                                                 preds)), 4),
        "class_f1s": [round(float(f), 4) for f in f1_score(
            y_eval, preds, average=None, zero_division=0, labels=labels)],
        "class_precision": [round(float(p), 4) for p in precision_score(
            y_eval, preds, average=None, zero_division=0, labels=labels)],
        "class_recall": [round(float(r), 4) for r in recall_score(
            y_eval, preds, average=None, zero_division=0, labels=labels)],
        "confusion_matrix": confusion_matrix(
            y_eval, preds, labels=labels).tolist(),
        "alpha": round(float(alpha), 6),
        "n_features": int(n_features),
        "ridge_fit_s": round(float(ridge_fit_s), 2),
        "inference_s": round(float(infer_s), 4),
    }


def fit_ridge(X_fit, y_fit, X_eval, y_eval):
    """Canonical RidgeClassifierCV(logspace(-4,4,20)) + predictions."""
    clf = RidgeClassifierCV(alphas=np.logspace(-4, 4, 20))
    t0 = time.time()
    clf.fit(X_fit, y_fit)
    fit_s = time.time() - t0
    t0 = time.time()
    preds = clf.predict(X_eval)
    infer_s = time.time() - t0
    m = full_metrics(y_eval, preds, X_fit.shape[1], fit_s, infer_s,
                     clf.alpha_)
    correct = np.asarray(preds) == np.asarray(y_eval)
    return m, preds, correct


def mr_transform(Xn_tr, Xn_va, Xn_te, seed):
    """Canonical MiniROCKET: fit on TRAIN only; transform all splits."""
    from aeon.transformations.collection.convolution_based import MiniRocket
    t0 = time.time()
    mr = MiniRocket(n_kernels=N_KERNELS, random_state=seed, n_jobs=-1)
    mr.fit(Xn_tr[:, None, :])
    fit_s = time.time() - t0
    t0 = time.time()
    Z_tr = mr.transform(Xn_tr[:, None, :]).astype(np.float64)
    Z_va = mr.transform(Xn_va[:, None, :]).astype(np.float64)
    Z_te = mr.transform(Xn_te[:, None, :]).astype(np.float64)
    tr_s = time.time() - t0
    return Z_tr, Z_va, Z_te, {"mr_fit_s": round(fit_s, 2),
                              "mr_transform_s": round(tr_s, 2)}


# ============================================================================
# Model assembly: [raw canonical MR block] + [standardized family blocks]
# ============================================================================
def assemble_model_blocks(Z_tr, Z_va, Z_te, fam_store, stds, fams):
    """MR block kept RAW (bit-identical features to canonical M0, sec. 3);
    each structured-family block standardized by its train-fitted
    BlockStandardizer."""
    bt, bv, be = [Z_tr], [Z_va], [Z_te]
    for f in fams:
        Ftr, Fva, Fte = fam_store[f]
        bt.append(stds[f].transform(Ftr))
        bv.append(stds[f].transform(Fva))
        be.append(stds[f].transform(Fte))
    Xtr, dims = assemble_blocks(bt)
    Xva, _ = assemble_blocks(bv)
    Xte, _ = assemble_blocks(be)
    return Xtr, Xva, Xte, dims


def eval_model(Z_tr, Z_va, Z_te, fam_store, stds, fams, ytr, yva, yte,
               final_protocol=True):
    """Final protocol: ridge fit on TRAIN+VAL, single TEST evaluation.
    Screening protocol (final_protocol=False): ridge fit on TRAIN only,
    evaluated on VAL (never touches TEST)."""
    Xtr, Xva, Xte, dims = assemble_model_blocks(
        Z_tr, Z_va, Z_te, fam_store, stds, fams)
    if final_protocol:
        X_fit = np.concatenate([Xtr, Xva], axis=0)
        y_fit = np.concatenate([ytr, yva])
        m, preds, correct = fit_ridge(X_fit, y_fit, Xte, yte)
        m["n_train_rows"] = int(X_fit.shape[0])
    else:
        m, preds, correct = fit_ridge(Xtr, ytr, Xva, yva)
        m["n_train_rows"] = int(Xtr.shape[0])
    m["feature_dim"] = int(sum(dims))
    m["block_dims"] = [int(d) for d in dims]
    return m, preds, correct


def block_sanity(blocks):
    rows = []
    for name, B in blocks:
        if B.shape[1] == 0:
            rows.append({"block": name, "n_features": 0,
                         "status": "EMPTY (family does not fit this T)"})
            continue
        finite = bool(np.isfinite(B).all())
        status = "OK" if finite else "FAIL(nan/inf)"
        if float(B.std()) < 1e-8:
            status += "+NEAR-ZERO-VARIANCE"
        rows.append({"block": name, "n_features": int(B.shape[1]),
                     "mean": round(float(B.mean()), 6),
                     "std": round(float(B.std()), 6),
                     "min": round(float(B.min()), 6),
                     "max": round(float(B.max()), 6),
                     "finite": finite, "status": status})
    return rows


# ============================================================================
# Per-dataset primary pipeline (seed 42)
# ============================================================================
def run_dataset_primary(ds_name, ds_file):
    cache = os.path.join(CACHE_DIR, f"primary_{ds_name}.json")
    if os.path.exists(cache):
        log(f"[cache] primary result found for {ds_name}")
        with open(cache, encoding="utf-8") as f:
            return json.load(f)

    log(f"=== {ds_name} ===")
    Xtr, Xva, Xte, ytr, yva, yte = load_split(ds_file)
    Xn_tr, Xn_va, Xn_te = znorm(Xtr), znorm(Xva), znorm(Xte)
    timings = {}

    # ---- canonical MiniROCKET base (once, seed 42) ----
    Z_tr, Z_va, Z_te, tmr = mr_transform(Xn_tr, Xn_va, Xn_te, SEED)
    timings.update(tmr)

    # ---- M0 canonical (must reproduce reference; sec. 4) ----
    m0, p0, c0 = eval_model(Z_tr, Z_va, Z_te, {}, {}, [], ytr, yva, yte)
    m0["n_train_rows"] = int(Z_tr.shape[0] + Z_va.shape[0])
    ref = REF_VALUES[ds_name]
    m0["reference_mf1"] = ref
    m0["reproduction_delta"] = round(m0["macro_f1"] - ref, 4)
    log(f"  M0 canonical: MF1={m0['macro_f1']:.4f} "
        f"(ref {ref:.4f}, delta {m0['reproduction_delta']:+.4f})")
    if abs(m0["macro_f1"] - ref) > BASELINE_TOLERANCE:
        raise RuntimeError(
            f"M0 FAILED to reproduce canonical baseline for {ds_name}: "
            f"{m0['macro_f1']} vs {ref} (tol {BASELINE_TOLERANCE}). "
            "STOP per spec sec. 4 - debug before interpreting anything.")

    # ---- M0 screening fit (train-only -> val) for family screening ----
    m0s, _, _ = fit_ridge(Z_tr, ytr, Z_va, yva)
    val_m0 = m0s["macro_f1"]

    # ---- structured family features (biases fitted on TRAIN only) ----
    t0 = time.time()
    ext = SKBExtractor(families=FAMILIES).fit(Xn_tr)
    skb_fit_s = time.time() - t0
    t0 = time.time()
    F_tr = ext.transform(Xn_tr)
    F_va = ext.transform(Xn_va)
    t_ext_tr = time.time() - t0
    t0 = time.time()
    F_te = ext.transform(Xn_te)
    t_ext_te = time.time() - t0
    fam_store = {f: (F_tr[f], F_va[f], F_te[f]) for f in FAMILIES}
    stds = {f: BlockStandardizer([F_tr[f].shape[1]]).fit(F_tr[f])
            for f in FAMILIES}
    fam_dims = {f: int(F_tr[f].shape[1]) for f in FAMILIES}
    log(f"  SKB fit {skb_fit_s:.1f}s, extract tr+va {t_ext_tr:.1f}s, "
        f"te {t_ext_te:.1f}s; dims {fam_dims} "
        f"(duplicates collapsed {ext.duplicates_})")

    # sanity blocks on train (mandatory, sec. 14)
    sanity_rows = block_sanity(
        [("MiniROCKET", Z_tr)] + [(f, F_tr[f]) for f in FAMILIES])

    # ---- screening (VALIDATION ONLY; sec. 20) ----
    val_screen = {"M0": val_m0}
    for f in FAMILIES:
        ms, _, _ = eval_model(Z_tr, Z_va, Z_te, fam_store, stds, [f],
                              ytr, yva, yte, final_protocol=False)
        val_screen[f] = ms["macro_f1"]
    promising = [f for f in FAMILIES if val_screen[f] > val_screen["M0"]]
    log("  screening (val MF1): " + ", ".join(
        f"{k}={v:.4f}" for k, v in val_screen.items()) +
        f" -> promising: {promising}")

    # ---- primary single-family models M1-M5 ----
    per_family, correct_map = {}, {"M0": c0}
    for i, f in enumerate(FAMILIES, start=1):
        t0 = time.time()
        m, preds, correct = eval_model(Z_tr, Z_va, Z_te, fam_store, stds,
                                       [f], ytr, yva, yte)
        wall = time.time() - t0
        m["val_screening_mf1"] = val_screen[f]
        m["delta_mf1"] = round(m["macro_f1"] - m0["macro_f1"], 4)
        m["wall_s"] = round(wall, 2)
        per_family[f] = m
        correct_map[f"M{i}_{FAM_TAG[f]}"] = correct
        log(f"  M{i} (+{f}): MF1={m['macro_f1']:.4f} "
            f"(d={m['delta_mf1']:+.4f}, dim={m['feature_dim']}, {wall:.0f}s)")

    # ---- pairwise models (only promising x promising; sec. 20) ----
    pairwise, pairs = {}, []
    if len(promising) >= 2:
        pairs = [(a, b) for i, a in enumerate(promising)
                 for b in promising[i + 1:]]
        for a, b in pairs:
            tag = f"{FAM_TAG[a]}+{FAM_TAG[b]}"
            ms, _, _ = eval_model(Z_tr, Z_va, Z_te, fam_store, stds, [a, b],
                                  ytr, yva, yte, final_protocol=False)
            m, preds, correct = eval_model(Z_tr, Z_va, Z_te, fam_store,
                                           stds, [a, b], ytr, yva, yte)
            m["val_screening_mf1"] = ms["macro_f1"]
            m["delta_mf1"] = round(m["macro_f1"] - m0["macro_f1"], 4)
            pairwise[tag] = m
            correct_map[f"pair_{tag}"] = correct
            log(f"  pair ({tag}): MF1={m['macro_f1']:.4f} "
                f"(d={m['delta_mf1']:+.4f})")

    # ---- M6 full structured bank ----
    t0 = time.time()
    m6, p6, c6 = eval_model(Z_tr, Z_va, Z_te, fam_store, stds, FAMILIES,
                            ytr, yva, yte)
    m6_wall = time.time() - t0
    m6s, _, _ = eval_model(Z_tr, Z_va, Z_te, fam_store, stds, FAMILIES,
                           ytr, yva, yte, final_protocol=False)
    m6["val_screening_mf1"] = m6s["macro_f1"]
    m6["delta_mf1"] = round(m6["macro_f1"] - m0["macro_f1"], 4)
    m6["wall_s"] = round(m6_wall, 2)
    correct_map["M6_ALL5"] = c6
    log(f"  M6 (+all5): MF1={m6['macro_f1']:.4f} (d={m6['delta_mf1']:+.4f}, "
        f"dim={m6['feature_dim']}, {m6_wall:.0f}s)")

    # ---- statistical comparisons on held-out test (sec. 25-26) ----
    stat_rows = []
    n_cmp = len(FAMILIES) + 1  # M1..M5 + M6, per dataset
    pvals = []
    for i, f in enumerate(FAMILIES, start=1):
        tag = f"M{i}_{FAM_TAG[f]}"
        chi2, p = mcnemar(correct_map[tag], correct_map["M0"])
        stat_rows.append({"dataset": ds_name, "comparison": f"M0 vs {tag}",
                          "delta_mf1": per_family[f]["delta_mf1"],
                          "chi2": round(chi2, 4), "p_raw": round(p, 6)})
        pvals.append(p)
    chi2_6, p_6 = mcnemar(correct_map["M6_ALL5"], correct_map["M0"])
    stat_rows.append({"dataset": ds_name, "comparison": "M0 vs M6_ALL5",
                      "delta_mf1": m6["delta_mf1"], "chi2": round(chi2_6, 4),
                      "p_raw": round(p_6, 6)})
    pvals.append(p_6)
    for tag in pairwise:
        chi2_p, p_p = mcnemar(correct_map[f"pair_{tag}"], correct_map["M0"])
        stat_rows.append({"dataset": ds_name,
                          "comparison": f"M0 vs pair({tag})",
                          "delta_mf1": pairwise[tag]["delta_mf1"],
                          "chi2": round(chi2_p, 4), "p_raw": round(p_p, 6),
                          "note": "secondary (post-screening)"})
    # BH-FDR per dataset over the 6 PRIMARY comparisons only; screening-
    # gated pairwise tests form a separate secondary family.
    q = benjamini_hochberg(pvals[:n_cmp])
    for r, qq in zip(stat_rows[:n_cmp], q):
        r["q_fdr"] = round(float(qq), 6)
        r["effect_size_d"] = ""
        r["verdict"] = ("SIGNIFICANT" if r["q_fdr"] < ALPHA and
                        r["delta_mf1"] > 0 else
                        "significant-but-negative" if r["q_fdr"] < ALPHA
                        else "not significant")
    n_sec = len(stat_rows) - n_cmp
    if n_sec > 0:
        q_sec = benjamini_hochberg(pvals[n_cmp:])
        for r, qq in zip(stat_rows[n_cmp:], q_sec):
            r["q_fdr"] = round(float(qq), 6)
            r["effect_size_d"] = ""
            r["verdict"] = ("SIGNIFICANT" if r["q_fdr"] < ALPHA and
                            r["delta_mf1"] > 0 else "not significant")

    # ---- representation diagnostics (sec. 22) ----
    diag = {"cka_vs_mr": {}, "mean_abs_corr_vs_mr": {},
            "family_self_cka": {}, "duplicates_collapsed": ext.duplicates_}
    for f in FAMILIES:
        diag["cka_vs_mr"][f] = round(linear_cka(Z_tr, F_tr[f]), 6)
        diag["mean_abs_corr_vs_mr"][f] = round(
            mean_abs_cross_correlation(Z_tr, F_tr[f]), 6)
    for i, a in enumerate(FAMILIES):
        for b in FAMILIES[i + 1:]:
            diag["family_self_cka"][f"{a}|{b}"] = round(
                linear_cka(F_tr[a], F_tr[b]), 6)

    out = {
        "dataset": ds_name,
        "seed": SEED,
        "m0": m0,
        "per_family": per_family,
        "pairwise": pairwise,
        "m6": m6,
        "val_screening": val_screen,
        "promising": promising,
        "pairs_run": [f"{a}+{b}" for a, b in pairs],
        "stat_rows": stat_rows,
        "sanity_blocks": sanity_rows,
        "diagnostics": diag,
        "family_dims": fam_dims,
        "duplicates_collapsed": ext.duplicates_,
        "timings": timings,
        "skb_fit_s": round(skb_fit_s, 2),
        "skb_extract_trva_s": round(t_ext_tr, 2),
        "skb_extract_te_s": round(t_ext_te, 2),
        "signal_length": int(Xtr.shape[1]),
        "kernel_metadata": ext.metadata(),
    }
    save_json(out, cache)
    return out


# ============================================================================
# Multi-seed robustness: M0 vs M6, seeds 42-46 (spec sec. 28)
# ============================================================================
def run_multiseed(ds_name, ds_file, fam_dims_ref):
    cache = os.path.join(CACHE_DIR, f"multiseed_{ds_name}.json")
    if os.path.exists(cache):
        log(f"[cache] multiseed result found for {ds_name}")
        with open(cache, encoding="utf-8") as f:
            return json.load(f)
    Xtr, Xva, Xte, ytr, yva, yte = load_split(ds_file)
    Xn_tr, Xn_va, Xn_te = znorm(Xtr), znorm(Xva), znorm(Xte)
    # Family features are deterministic given the split; fit once.
    ext = SKBExtractor(families=FAMILIES).fit(Xn_tr)
    F_tr, F_va, F_te = ext.transform(Xn_tr), ext.transform(Xn_va), \
        ext.transform(Xn_te)
    fam_store = {f: (F_tr[f], F_va[f], F_te[f]) for f in FAMILIES}
    stds = {f: BlockStandardizer([F_tr[f].shape[1]]).fit(F_tr[f])
            for f in FAMILIES}
    rows = []
    for seed in SEEDS_ROBUST:
        Z_tr, Z_va, Z_te, _ = mr_transform(Xn_tr, Xn_va, Xn_te, seed)
        m0, _, _ = eval_model(Z_tr, Z_va, Z_te, {}, {}, [], ytr, yva, yte)
        m6, _, _ = eval_model(Z_tr, Z_va, Z_te, fam_store, stds, FAMILIES,
                              ytr, yva, yte)
        rows.append({"dataset": ds_name, "seed": seed,
                     "m0_mf1": m0["macro_f1"], "m6_mf1": m6["macro_f1"],
                     "delta": round(m6["macro_f1"] - m0["macro_f1"], 4)})
        log(f"  [multiseed {ds_name}] seed {seed}: "
            f"M0={m0['macro_f1']:.4f} M6={m6['macro_f1']:.4f} "
            f"(d={rows[-1]['delta']:+.4f})")
    d = [r["delta"] for r in rows]
    t, p = paired_permutation_test([r["m6_mf1"] for r in rows],
                                   [r["m0_mf1"] for r in rows])
    summ = {"rows": rows,
            "mean_delta": round(float(np.mean(d)), 5),
            "std_delta": round(float(np.std(d, ddof=1)), 5) if len(d) > 1
            else 0.0,
            "median_delta": round(float(np.median(d)), 5),
            "min_delta": round(float(np.min(d)), 5),
            "max_delta": round(float(np.max(d)), 5),
            "n_positive": int(sum(1 for x in d if x > 0)),
            "n_negative": int(sum(1 for x in d if x < 0)),
            "paired_permutation_p": round(p, 6),
            "effect_size_d": round(cohens_d_paired(
                [r["m6_mf1"] for r in rows],
                [r["m0_mf1"] for r in rows]), 4)}
    save_json(summ, cache)
    return summ


# ============================================================================
# Figures
# ============================================================================
def fig1_architecture(path):
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.axis("off")
    boxes = [
        (0.02, "Canonical\nMiniROCKET 10K\n(fixed, seed 42)", "#9ecae1"),
        (0.19, "Structured fixed\nkernel banks F1-F5\n(predefined grids)", "#fdd0a2"),
        (0.36, "Dilated response\nr(t) = sum_j k_j x_{t+jd}\n(+ energy operator)", "#fdd0a2"),
        (0.53, "9 golden-quantile\nbiases per kernel\n(TRAIN-fit, PPV)", "#c7e9c0"),
        (0.70, "Concat + block\nstandardization\n(train-only stats)", "#c7e9c0"),
        (0.87, "One\nRidgeClassifierCV\n(logspace -4..4)", "#fcbba1"),
    ]
    for x, txt, c in boxes:
        ax.add_patch(plt.Rectangle((x, 0.35), 0.16, 0.32, color=c,
                                   ec="k", lw=1.2))
        ax.text(x + 0.08, 0.51, txt, ha="center", va="center", fontsize=8.5)
    for x in [0.18, 0.35, 0.52, 0.69, 0.86]:
        ax.annotate("", xy=(x + 0.01, 0.51), xytext=(x - 0.01, 0.51),
                    arrowprops=dict(arrowstyle="->", lw=1.6))
    ax.text(0.5, 0.15, "M0: left block only | M1-M5: + one family | "
            "M6: + all five | 0 trainable kernel parameters",
            ha="center", fontsize=9, style="italic")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1)
    fig.suptitle("TURS-SKB architecture (MiniROCKET philosophy unchanged; "
                 "kernel space enriched)", fontsize=11)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig2_main(res_by_ds, path):
    ds = list(res_by_ds.keys())
    models = [("m0", "MR-10K")] + \
             [(f, f"+{FAM_TAG[f]}") for f in FAMILIES] + [("m6", "+All5")]
    x = np.arange(len(ds))
    w = 0.11
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for j, (key, lab) in enumerate(models):
        vals = [res_by_ds[d]["per_family"][key]["macro_f1"]
                if key in FAMILIES else res_by_ds[d][key]["macro_f1"]
                for d in ds]
        ax.bar(x + (j - 3) * w, vals, w, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(ds, fontsize=9)
    ax.set_ylabel("Test Macro-F1")
    ax.set_ylim(min(0.5, min(res_by_ds[d]["m0"]["macro_f1"] for d in ds)
                    - 0.05), 1.02)
    ax.legend(ncol=7, fontsize=8, loc="upper left")
    ax.set_title("TURS-SKB: M0-M6 test Macro-F1 (seed 42)")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig3_deltas(res_by_ds, path):
    ds = list(res_by_ds.keys())
    fig, axes = plt.subplots(1, len(ds), figsize=(3.1 * len(ds), 3.6),
                             sharey=False)
    if len(ds) == 1:
        axes = [axes]
    for ax, d in zip(axes, ds):
        fams = list(res_by_ds[d]["per_family"].keys())
        deltas = [res_by_ds[d]["per_family"][f]["delta_mf1"] for f in fams]
        deltas.append(res_by_ds[d]["m6"]["delta_mf1"])
        labs = [FAM_TAG[f] for f in fams] + ["All5"]
        cols = ["tab:blue" if v >= 0 else "tab:red" for v in deltas]
        ax.bar(labs, deltas, color=cols)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(d, fontsize=9)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
    fig.suptitle("Per-family incremental test dMF1 vs M0 (seed 42)")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig4_cka(res_by_ds, path):
    fig, ax = plt.subplots(figsize=(8.5, 4))
    fams = FAMILIES
    w = 0.8 / len(res_by_ds)
    for j, d in enumerate(res_by_ds):
        vals = [res_by_ds[d]["diagnostics"]["cka_vs_mr"][f] for f in fams]
        ax.bar(np.arange(len(fams)) + j * w - 0.4 + w / 2, vals, w, label=d)
    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels([FAM_TAG[f] for f in fams])
    ax.set_ylabel("Linear CKA vs MiniROCKET block (train)")
    ax.legend(fontsize=8)
    ax.set_title("Representation similarity of family blocks vs MR "
                 "(low CKA = distinct, NOT automatically useful)")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig5_responses(res_by_ds, path):
    """Representative response shapes: one example per family, kernels
    pre-registered before any performance result was seen."""
    # pick example signals: one ECG, one CWRU, from TRAIN only
    ex = {}
    for ds_name, ds_file in DATASETS:
        if ds_name.startswith("ECG") and "ecg" not in ex:
            Xtr, _, _, _, _, _ = load_split(ds_file)
            ex["ecg"] = znorm(Xtr)[0]
        if ds_name.startswith("CWRU") and "cwru" not in ex:
            Xtr, _, _, _, _, _ = load_split(ds_file)
            ex["cwru"] = znorm(Xtr)[0]
    from src.features.turs_skb.kernels import (
        difference_of_gaussians_kernel, derivative_kernel,
        wavelet_kernel, gabor_kernel, energy_kernel,
    )
    from src.features.turs_skb.features import (
        linear_kernel_response, energy_kernel_response, tent_window,
    )
    fig, axes = plt.subplots(2, 6, figsize=(17, 6))
    for r, (tag, sig) in enumerate([("ECG", ex.get("ecg")),
                                    ("CWRU", ex.get("cwru"))]):
        if sig is None:
            continue
        T = sig.size
        sig_plot = sig[:T]
        axes[r, 0].plot(sig_plot, lw=0.7, color="k")
        axes[r, 0].set_title(f"{tag} raw (train, znorm)", fontsize=9)
        # MiniROCKET representative: first kernel of the fitted transformer
        # is not retrievable portably; use deterministic sparse ±1 kernel
        # (MiniROCKET's family shape) at dilation 8 as illustration.
        k = np.array([1] * 4 + [-1] * 5, dtype=np.float32)
        resp = linear_kernel_response(sig_plot[None, :], k, 8)[0]
        axes[r, 1].plot(resp, lw=0.7, color="#3182bd")
        axes[r, 1].set_title("MR-style sparse kernel (dil 8)", fontsize=9)
        spec = REPRESENTATIVE_KERNELS
        kd, meta = spec["morphological"]
        resp = linear_kernel_response(
            sig_plot[None, :], difference_of_gaussians_kernel(
                meta["sigma"], meta["alpha"]), meta["dilation"])[0]
        axes[r, 2].plot(resp, lw=0.7, color="#e6550d")
        axes[r, 2].set_title("Morph (DoG s4 a1 dil1)", fontsize=9)
        kd, meta = spec["derivative"]
        resp = linear_kernel_response(
            sig_plot[None, :], derivative_kernel(meta["order"]),
            meta["dilation"])[0]
        axes[r, 3].plot(resp, lw=0.7, color="#31a354")
        axes[r, 3].set_title("Deriv (1st, dil 4)", fontsize=9)
        kd, meta = spec["wavelet"]
        resp = linear_kernel_response(
            sig_plot[None, :], wavelet_kernel(meta["type"], meta["scale"]),
            meta["dilation"])[0]
        axes[r, 4].plot(resp, lw=0.7, color="#756bb1")
        axes[r, 4].set_title("Wavelet (MexHat s4 dil1)", fontsize=9)
        kd, meta = spec["gabor"]
        resp = linear_kernel_response(
            sig_plot[None, :], gabor_kernel(meta["freq"], meta["sigma"],
                                            meta["phase"]),
            meta["dilation"])[0]
        axes[r, 5].plot(resp, lw=0.7, color="#d6616b")
        axes[r, 5].set_title("Gabor (f1/16 s8 ph0 dil1)", fontsize=9)
        # energy row appended below as separate small figure? keep 6 cols
    fig.suptitle("Representative kernel responses (pre-registered kernels; "
                 "train signals)", fontsize=11)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    # energy panel separately (nonlinear operator)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3))
    for ax, (tag, sig) in zip(axes, [("ECG", ex.get("ecg")),
                                     ("CWRU", ex.get("cwru"))]):
        if sig is None:
            continue
        meta = REPRESENTATIVE_KERNELS["energy"][1]
        qs, ws, wl, lam = energy_kernel(meta["w_short"], meta["lambda"])
        resp = energy_kernel_response(sig[None, :], qs, tent_window(wl), lam)
        if resp is not None:
            ax.plot(resp[0], lw=0.7, color="#8c6d31")
        ax.set_title(f"{tag} energy E8 - 1.0*E32 (nonlinear operator)",
                     fontsize=9)
    fig.savefig(path.replace(".png", "_energy.png"), dpi=160,
                bbox_inches="tight")
    plt.close(fig)


def fig6_cost(res_by_ds, path):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    markers = {"m0": "o", "m6": "*"}
    for d, res in res_by_ds.items():
        m0, m6 = res["m0"], res["m6"]
        ax.scatter(m0["ridge_fit_s"], m0["macro_f1"] * 100, marker="o",
                   s=70, label=f"M0 {d}")
        ax.scatter(m6["ridge_fit_s"], m6["macro_f1"] * 100, marker="*",
                   s=160, label=f"M6 {d}")
        ax.annotate("", xy=(m6["ridge_fit_s"], m6["macro_f1"] * 100),
                    xytext=(m0["ridge_fit_s"], m0["macro_f1"] * 100),
                    arrowprops=dict(arrowstyle="->", lw=0.8, alpha=0.5))
    ax.set_xlabel("Ridge fit time on train+val (s) - feature-extraction "
                  "runtime in runtime.csv")
    ax.set_ylabel("Test Macro-F1 (%)")
    ax.legend(fontsize=7)
    ax.set_title("Accuracy vs computational cost: M0 -> M6")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# REPORT.md (structure mandated by spec sec. 42)
# ============================================================================
def fmt(x, nd=4):
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def build_report(res_by_ds, seed_rows, param_rows, mech_verdicts):
    L = []
    A = L.append
    ds_names = list(res_by_ds.keys())
    m0_mean = float(np.mean([res_by_ds[d]["m0"]["macro_f1"]
                             for d in ds_names]))
    m6_mean = float(np.mean([res_by_ds[d]["m6"]["macro_f1"]
                             for d in ds_names]))
    fam_mean_deltas = {}
    for f in FAMILIES:
        fam_mean_deltas[f] = round(float(np.mean(
            [res_by_ds[d]["per_family"][f]["delta_mf1"] for d in ds_names])),
            4)
    all_d6 = [res_by_ds[d]["m6"]["delta_mf1"] for d in ds_names]
    d6_mean = round(float(np.mean(all_d6)), 4)
    n_pos6 = sum(1 for x in all_d6 if x > 0)
    n_sig6 = sum(1 for d in ds_names
                 for r in res_by_ds[d]["stat_rows"]
                 if r["comparison"] == "M0 vs M6_ALL5" and
                 r.get("q_fdr", 1) < ALPHA)
    best_fam = max(fam_mean_deltas, key=lambda k: fam_mean_deltas[k])
    mean_dim_m6 = int(np.mean([res_by_ds[d]["m6"]["feature_dim"]
                               for d in ds_names]))
    mean_dim_m0 = int(np.mean([res_by_ds[d]["m0"]["feature_dim"]
                               for d in ds_names]))

    A("# TURS-SKB - Structured Kernel Bank: Final Report\n")
    A(f"Run completed {time.strftime('%Y-%m-%d %H:%M')} | seed 42 primary | "
      f"canonical aeon MiniROCKET 10K base | outputs: results/turs_skb/\n")

    A("## 1. Executive summary\n")
    A(f"- M0 canonical reproduced the reference benchmark on all "
      f"{len(ds_names)} datasets (max |delta| "
      f"{max(abs(res_by_ds[d]['m0']['reproduction_delta']) for d in ds_names):.4f}"
      f" <= tol {BASELINE_TOLERANCE}).\n")
    A(f"- M6 (MR + all five structured families, "
      f"{mean_dim_m6} features vs {mean_dim_m0}) mean Macro-F1 "
      f"{m6_mean:.4f} vs M0 {m0_mean:.4f} (mean delta {d6_mean:+.4f}; "
      f"{n_pos6}/{len(all_d6)} datasets positive; {n_sig6} FDR-significant "
      f"dataset-level M6 improvements).\n")
    A(f"- Best single family by mean delta: **{best_fam}** "
      f"({fam_mean_deltas[best_fam]:+.4f}).\n")
    A(f"- Final classification: **{classify(res_by_ds, seed_rows, d6_mean, n_sig6, best_fam)}**.\n")

    A("## 2. Research question\n")
    A("Can mathematically structured fixed temporal kernels provide "
      "complementary information to canonical MiniROCKET while retaining "
      "its efficient fixed-feature + linear-readout framework? The novelty "
      "tested is ONLY the kernel family; the downstream pipeline is "
      "unchanged.\n")

    A("## 3. Why MiniROCKET kernels may be limiting\n")
    A("Canonical MiniROCKET draws sparse two-valued kernels (groups of +1 "
      "and -1 taps) with random lengths {7,9,11} and data-driven dilation; "
      "its bias set is 9 train-response quantiles. This family is random, "
      "sparse, two-valued, and optimized for generic shape matching - it "
      "does not deliberately encode smooth bump morphology (F1), exact "
      "finite-difference operators (F2), analytic wavelets (F3), "
      "band-localized oscillations (F4), or burst energy (F5). If those "
      "structured function classes carry predictive signal that the sparse "
      "family misses, a fixed structured bank should add it.\n")

    A("## 4. Mathematical definition of each family\n")
    A("- **F1 Morphological/shape**: unit-L2 Gaussian bumps "
      "`k(u)=exp(-u^2/2s^2)`, difference-of-Gaussians "
      "`k(u)=exp(-u^2/2s1^2)-a*exp(-u^2/2s2^2)` (s2=s1*1.6), and "
      "Gaussian-derivative operators (1st/2nd). Grid: sigma in "
      f"{SIGMA_GRID} x dilation in {DILATION_GRID} x polarity for bumps; "
      f"alpha in {ALPHA_DO_G_GRID} for DoG; orders {DERIV_GRID}.\n")
    A("- **F2 Derivative**: discrete operators `[-1,0,+1]` (1st) and "
      f"`[+1,-2,+1]` (2nd), unit-L2, dilations {1,2,4,8,16,32,64,128}, "
      "both polarities. Deliberately small (32 kernels) to avoid "
      "feature-count dominance.\n")
    A("- **F3 Wavelet**: `k_a(u) = a^{-1/2} psi(u/a)` discretized for psi "
      f"in {{Haar, Mexican hat, Morlet}} x a in {WAVELET_SCALES} x "
      "dilations {1..128} x polarity (192 kernels).\n")
    A("- **F4 Gabor**: `k(u)=exp(-u^2/(2s^2)) cos(2 pi f u + phi)`, f in "
      f"{GABOR_F_GRID} (periods 8/16/32 taps), sigma in {GABOR_SIGMA_GRID}, "
      f"phi in {GABOR_PHASE_GRID} (sine quadrature via phase), dilations "
      f"{DILATION_GRID} (192 kernels).\n")
    A("- **F5 Energy/burst (NONLINEAR)**: `B(t) = E_w(t) - lam*E_4w(t)` "
      "with `E_w(t) = sum_j q_j x_{t+j}^2`, q = sum-normalized tent "
      "window (NOT a signed linear convolution; documented as a nonlinear "
      f"energy operator). Grid: w in {ENERGY_W_GRID} x lam in "
      f"{ENERGY_LAMBDA_GRID} (12 operators).\n")

    A("## 5. Exact kernel-generation procedure\n")
    A("All banks are built from the predefined grids above with NO "
      "randomness, NO fitting, and NO label access; every base kernel is "
      "unit-L2-normalized; dilation is recorded per bank entry and capped "
      "at fit time to the largest power of two that fits the series "
      "length (aeon's clip rule). Identical (kernel, effective-dilation) "
      "pairs are collapsed and counted, never duplicated. Bank counts: "
      + ", ".join(f"{k}={len(BANK_BUILDERS[k]())}" for k in FAMILIES) +
      ".\n")

    A("## 6. Feature extraction\n")
    A("Response `r_k(t) = sum_j k_j x_{t + j d}` (kernel taps strided by "
      "dilation d, evaluated at every valid position) - identical response "
      "model to MiniROCKET. Each kernel yields 9 PPV features against its "
      "9 golden-ratio quantile biases (the aeon MiniROCKET threshold "
      "abstraction, mirrored exactly): `PPV = mean_t 1[r(t) > b]`, biases "
      "fitted on TRAIN responses only. No mu/W/variance/phase features "
      "(spec sec. 12).\n")

    A("## 7. Standardization\n")
    A("MR block: RAW (bit-identical to canonical M0 features - the "
      "canonical baseline itself applies no scaler to MiniROCKET PPV "
      "features; fairness requires M1-M6 to inherit that exact block). "
      "Each structured family block: z-standardized with TRAIN-only "
      "statistics (BlockStandardizer, near-zero-variance columns left "
      "unscaled). All blocks verified finite with mean/std/min/max logged "
      "(sanity_blocks in full_results.json).\n")

    A("## 8. Ridge protocol\n")
    A("RidgeClassifierCV(alphas=np.logspace(-4, 4, 20)) fitted on "
      "TRAIN+VAL (canonical baseline_bench protocol), single TEST "
      "evaluation. Screening fits use TRAIN-only ridge evaluated on VAL "
      "and never touch TEST.\n")

    A("## 9. Dataset protocol\n")
    A("Canonical four-dataset benchmark split (train_test_split seed 42, "
      "stratified; per-signal z-norm). Reference values: " +
      ", ".join(f"{d}={REF_VALUES[d]}" for d in REF_VALUES) + ".\n")

    A("## 10. Experimental variants\n")
    A("| Model | Blocks | Feature dim (mean) |")
    A("|---|---|---:|")
    A(f"| M0 | MR | {mean_dim_m0} |")
    for i, f in enumerate(FAMILIES, start=1):
        dm = int(np.mean([res_by_ds[d]["per_family"][f]["feature_dim"]
                          for d in ds_names]))
        A(f"| M{i} | MR + {f} | {dm} |")
    for tag in sorted({t for d in ds_names
                       for t in res_by_ds[d]["pairwise"]}):
        dm = int(np.mean([res_by_ds[d]["pairwise"][tag]["feature_dim"]
                          for d in ds_names if tag in
                          res_by_ds[d]["pairwise"]]))
        A(f"| pair | MR + {tag} | {dm} |")
    A(f"| M6 | MR + all five | {mean_dim_m6} |\n")

    A("## 11. Main results (test Macro-F1, seed 42)\n")
    A("| Dataset | MR-10K | +Morph | +Deriv | +Wavelet | +Gabor | +Energy "
      "| +All5 |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|")
    for d in ds_names:
        r = res_by_ds[d]
        A(f"| {d} | {r['m0']['macro_f1']:.4f} | " +
          " | ".join(f"{r['per_family'][f]['macro_f1']:.4f}"
                     for f in FAMILIES) +
          f" | {r['m6']['macro_f1']:.4f} |")
    A("")

    A("## 12. Per-family gains (dMF1 vs M0)\n")
    A("| Dataset | dMorph | dDeriv | dWavelet | dGabor | dEnergy | dAll5 |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for d in ds_names:
        r = res_by_ds[d]
        A(f"| {d} | " + " | ".join(
            f"{r['per_family'][f]['delta_mf1']:+.4f}" for f in FAMILIES) +
          f" | {r['m6']['delta_mf1']:+.4f} |")
    A("")
    A("Mean over datasets: " + ", ".join(
        f"d{FAM_TAG[f]}={fam_mean_deltas[f]:+.4f}" for f in FAMILIES) +
      f", dAll5={d6_mean:+.4f} (median "
      f"{float(np.median(all_d6)):+.4f}).\n")
    A("Pairwise (screening-gated) results: " +
      ("; ".join(f"{d}: " + ", ".join(
          f"{t} {res_by_ds[d]['pairwise'][t]['delta_mf1']:+.4f}"
          for t in sorted(res_by_ds[d]["pairwise"]))
          for d in ds_names if res_by_ds[d]["pairwise"]) or "none run") +
      ".\n")

    A("## 13. Statistical inference (paired McNemar on held-out test "
      "correctness)\n")
    A("| Comparison | Dataset | dMF1 | p | q(FDR) | Effect size | Verdict |")
    A("|---|---|---:|---:|---:|---:|---|")
    for d in ds_names:
        for r in res_by_ds[d]["stat_rows"]:
            eff = seed_rows.get(d, {}).get("effect_size_d", "") \
                if r["comparison"] == "M0 vs M6_ALL5" else ""
            qv = r.get("q_fdr", "")
            qv = f"{float(qv):.4g}" if isinstance(qv, (int, float)) else "-"
            A(f"| {r['comparison']} | {d} | {r['delta_mf1']:+.4f} | "
              f"{float(r['p_raw']):.4g} | {qv} | {eff} | "
              f"{r.get('verdict','')} |")
    A("")
    A("Six primary comparisons per dataset (M1-M5, M6) corrected with "
      "BH-FDR within dataset; screening-gated pairwise tests corrected "
      "separately. Effect size for the primary M0-vs-M6 comparison is "
      "Cohen's d over the 5-seed MF1 pairs (sec. 18).\n")

    A("## 14. Multiple-comparison correction\n")
    A(f"BH-FDR at alpha={ALPHA}. Verdicts above use q, not raw p. No "
      "family is declared significant on raw p alone.\n")

    A("## 15. Representation diversity\n")
    A("| Family | Feature count (mean) | CKA vs MR (mean) | Mean |corr| "
      "vs MR (mean) |")
    A("|---|---:|---:|---:|")
    for f in FAMILIES:
        cka = float(np.mean([res_by_ds[d]["diagnostics"]["cka_vs_mr"][f]
                             for d in ds_names]))
        mc = float(np.mean(
            [res_by_ds[d]["diagnostics"]["mean_abs_corr_vs_mr"][f]
             for d in ds_names]))
        fc = int(np.mean([res_by_ds[d]["family_dims"][f]
                          for d in ds_names]))
        A(f"| {f} | {fc} | {cka:.3f} | {mc:.3f} |")
    A("")
    A("CKA is a DIVERSITY diagnostic only: low CKA means the block is "
      "mathematically distinct, not that it is useful; predictive "
      "increment is judged in sections 12-13.\n")

    A("## 16. Kernel-response diagnostics\n")
    A("Figure 5 (fig5_kernel_responses.png / _energy.png) shows the "
      "pre-registered representative kernels' responses on one TRAIN ECG "
      "and one TRAIN CWRU signal (selected before any test result was "
      "seen; no cherry-picking). Mechanistic expectations (pre-registered) "
      "vs outcome:\n")
    for f, (targets, why) in MECH_EXPECTATIONS.items():
        sup = mech_verdicts.get(f, {})
        A(f"- {f} ({why}): pre-registered target datasets "
          f"{targets}; observed mean delta on targets "
          f"{sup.get('target_mean', float('nan')):+.4f}, "
          f"elsewhere {sup.get('other_mean', float('nan')):+.4f} -> "
          f"**{sup.get('verdict', 'n/a')}**.")
    A("")

    A("## 17. Computational cost\n")
    A("| Dataset | Model | Feature dim | Ridge fit (s) | Inference (s) | "
      "SKB extract (s) |")
    A("|---|---|---:|---:|---:|---:|")
    for d in ds_names:
        r = res_by_ds[d]
        A(f"| {d} | M0 | {r['m0']['feature_dim']} | "
          f"{r['m0']['ridge_fit_s']:.1f} | {r['m0']['inference_s']:.2f} | "
          "- |")
        A(f"| {d} | M6 | {r['m6']['feature_dim']} | "
          f"{r['m6']['ridge_fit_s']:.1f} | {r['m6']['inference_s']:.2f} | "
          f"{r['skb_extract_te_s']:.1f} |")
    A("")
    A("Feature-extraction cost of the full structured bank is "
      "seconds-scale per split; the ridge fit on the enlarged matrix "
      "dominates. Gain per added feature is reported per dataset in "
      "feature_dimensions.csv (delta MF1 per 1K added features).\n")

    if seed_rows:
        A("## 18. Seed robustness (M0 vs M6, seeds 42-46)\n")
        A("| Dataset | mean d | std | median | min | max | n_pos | n_neg | "
          "permutation p | Cohen's d |")
        A("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for d, s in seed_rows.items():
            A(f"| {d} | {s['mean_delta']:+.4f} | {s['std_delta']:.4f} | "
              f"{s['median_delta']:+.4f} | {s['min_delta']:+.4f} | "
              f"{s['max_delta']:+.4f} | {s['n_positive']} | "
              f"{s['n_negative']} | {s['paired_permutation_p']:.4g} | "
              f"{s['effect_size_d']:+.3f} |")
        A("")
    else:
        A("## 18. Seed robustness\n")
        A("Skipped (TURS_SKB_SKIP_MULTISEED=1); primary benchmark is "
          "labeled a deterministic/single-seed benchmark.\n")

    A("## 19. Failure analysis\n")
    fails = []
    for f in FAMILIES:
        n_neg = sum(1 for d in ds_names
                    if res_by_ds[d]["per_family"][f]["delta_mf1"] < 0)
        if n_neg:
            fails.append(f"{f}: negative delta on {n_neg}/{len(ds_names)} "
                         "datasets")
    A(("Families with negative increments: " + "; ".join(fails) +
       ". Structured blocks can shift the ridge solution away from the "
       "canonical optimum when their added features are noise-dominated "
       "at near-ceiling accuracy (CWRU) or when family scale dominates "
       "after standardization.") if fails else
      "No family produced a negative mean increment; the structured bank "
      "is uniformly non-harmful on this benchmark.")
    A("")

    A("## 20. Limitations\n")
    A("- Single canonical split per dataset for primary results (5-seed "
      "robustness covers MiniROCKET draw variance only; the split itself "
      "is fixed by the benchmark protocol).\n"
      "- Family feature counts differ by design (deriv/energy smaller, "
      "documented per spec sec. 13); per-family comparisons therefore "
      "conflate family identity with feature budget, mitigated by the "
      "matched ~9-features-per-kernel abstraction and the M6 "
      "gain-per-feature accounting.\n"
      "- Biases are train-quantile-fitted (MiniROCKET's own mechanism); a "
      "fully predefined deterministic threshold convention was the "
      "alternative and is documented as such (spec sec. 6).\n"
      "- The energy family is a nonlinear operator; its comparison to "
      "linear families is by feature semantics, not by convolution "
      "algebra.\n")

    A("## 21. Final scientific conclusion\n")
    A(final_conclusion(res_by_ds, seed_rows, d6_mean, n_sig6, m0_mean,
                       m6_mean, fam_mean_deltas, best_fam) + "\n")

    A("## Final decision-rule classification (sec. 44)\n")
    A(f"**{classify(res_by_ds, seed_rows, d6_mean, n_sig6, best_fam)}**\n")

    A("## Final scientific questions\n")
    A(f"- **Q1** Does any structured family significantly improve over "
      f"canonical MiniROCKET? "
      f"{q1_answer(res_by_ds, fam_mean_deltas)}\n")
    A(f"- **Q2** Which family is most useful? **{best_fam}** (mean delta "
      f"{fam_mean_deltas[best_fam]:+.4f}).\n")
    A(f"- **Q3** Consistent across ECG and CWRU? {q3_answer(res_by_ds)}\n")
    A(f"- **Q4** Genuinely different representations? Mean CKA vs MR: " +
      ", ".join(f"{f}={np.mean([res_by_ds[d]['diagnostics']['cka_vs_mr'][f] for d in ds_names]):.3f}"
                for f in FAMILIES) +
      " (distinct blocks; usefulness judged by predictive increment).\n")
    A(f"- **Q5** Additive benefit from combining families? dAll5="
      f"{d6_mean:+.4f} vs best single family "
      f"{fam_mean_deltas[best_fam]:+.4f}.\n")
    A(f"- **Q6** Justified by cost? Feature dim {mean_dim_m0} -> "
      f"{mean_dim_m6} (~{mean_dim_m6/mean_dim_m0:.1f}x) for "
      f"{d6_mean:+.4f} mean MF1; gain per 1K added features: " +
      ", ".join(f"{d}={res_by_ds[d]['m6']['delta_mf1']/max((res_by_ds[d]['m6']['feature_dim']-res_by_ds[d]['m0']['feature_dim'])/1000,1e-9):+.3f}"
                for d in ds_names) + ".\n")
    A(f"- **Q7** M6 vs M0 under controlled evaluation? Mean "
      f"{m6_mean:.4f} vs {m0_mean:.4f}; {n_sig6} dataset(s) "
      f"FDR-significant; seed-robustness p-values in sec. 18.\n")

    return "\n".join(L)


def q1_answer(res_by_ds, fam_mean_deltas):
    sigs = []
    for d, res in res_by_ds.items():
        for r in res["stat_rows"]:
            if r.get("q_fdr", 1) < ALPHA and r["delta_mf1"] > 0 and \
                    "pair" not in r["comparison"]:
                sigs.append(f"{r['comparison']} on {d} (q={r['q_fdr']:.3f})")
    if sigs:
        return "Yes - " + "; ".join(sigs) + "."
    any_pos = [f for f, v in fam_mean_deltas.items() if v > 0]
    if any_pos:
        return ("No family survives FDR correction, but positive mean "
                "increments for " + ", ".join(any_pos) +
                " (numerical, not statistically significant).")
    return "No - no family improves canonical MiniROCKET."


def q3_answer(res_by_ds):
    parts = []
    for f in FAMILIES:
        e = [res_by_ds[d]["per_family"][f]["delta_mf1"] for d in res_by_ds
             if d.startswith("ECG")]
        c = [res_by_ds[d]["per_family"][f]["delta_mf1"] for d in res_by_ds
             if d.startswith("CWRU")]
        me, mc = float(np.mean(e)), float(np.mean(c))
        tag = "consistent" if (me > 0) == (mc > 0) else "DATASET-DEPENDENT"
        parts.append(f"{f}: ECG {me:+.4f} vs CWRU {mc:+.4f} ({tag})")
    return "; ".join(parts) + "."


def final_conclusion(res_by_ds, seed_rows, d6_mean, n_sig6, m0_mean,
                     m6_mean, fam_mean_deltas, best_fam):
    if d6_mean > 0.005 and n_sig6 >= 2:
        return ("The structured kernel bank provides a genuine, "
                "statistically supported improvement over canonical "
                "MiniROCKET on this benchmark.")
    if d6_mean > 0 and (n_sig6 >= 1 or any(
            v > 0.002 for v in fam_mean_deltas.values())):
        return (f"The structured kernel hypothesis is PARTIALLY supported: "
                f"{best_fam} gives the largest mean increment "
                f"({fam_mean_deltas[best_fam]:+.4f}) and M6 reaches "
                f"{m6_mean:.4f} vs {m0_mean:.4f}, but evidence is "
                f"dataset-dependent/limited; treated as a numerical "
                f"improvement without strong universal statistical "
                f"support.")
    if d6_mean > -0.002:
        return ("The structured kernel hypothesis is NOT supported: "
                "mathematically distinct fixed families yield no material "
                "improvement over the canonical sparse MiniROCKET family "
                "under matched protocol. Canonical MiniROCKET's sparse "
                "kernels + quantile PPV pooling already capture the "
                "predictive temporal structure available to linear "
                "fixed-feature pipelines at this scale.")
    return ("The structured kernel bank slightly DEGRADES canonical "
            "MiniROCKET performance; the hypothesis is not supported.")


def classify(res_by_ds, seed_rows, d6_mean, n_sig6, best_fam):
    all_pos = all(res_by_ds[d]["m6"]["delta_mf1"] > 0.002
                  for d in res_by_ds)
    if d6_mean > 0.01 and n_sig6 >= 2 and all_pos:
        return "MODERATE"
    if d6_mean > 0.005 and n_sig6 >= 1:
        return "WEAK (positive, limited statistical support)"
    if d6_mean > 0.001:
        return "NO IMPROVEMENT to WEAK (numerical improvement only)"
    if d6_mean >= -0.002:
        return "NO IMPROVEMENT"
    return "NEGATIVE"


# ============================================================================
# Main
# ============================================================================
def main():
    t_start = time.time()
    log("TURS-SKB starting")
    log(f"config: seed={SEED}, kernels={N_KERNELS}, tolerance="
        f"{BASELINE_TOLERANCE}, multiseed={RUN_MULTISEED}, datasets="
        f"{[d[0] for d in DATASETS]}")

    save_config = {
        "seed": SEED, "seeds_robust": SEEDS_ROBUST, "n_kernels": N_KERNELS,
        "baseline_tolerance": BASELINE_TOLERANCE, "alpha": ALPHA,
        "families": FAMILIES,
        "grids": {
            "sigma": SIGMA_GRID, "dilation": DILATION_GRID,
            "dog_alpha": ALPHA_DO_G_GRID, "deriv_orders": DERIV_GRID,
            "wavelet_types": WAVELET_TYPES, "wavelet_scales":
                WAVELET_SCALES,
            "gabor_freq": GABOR_F_GRID, "gabor_sigma": GABOR_SIGMA_GRID,
            "gabor_phase": GABOR_PHASE_GRID, "energy_w": ENERGY_W_GRID,
            "energy_lambda": ENERGY_LAMBDA_GRID,
        },
        "representative_kernels": {k: list(v) for k, v in
                                   REPRESENTATIVE_KERNELS.items()},
        "mech_expectations": {k: v[0] for k, v in
                              MECH_EXPECTATIONS.items()},
        "ridge": "RidgeClassifierCV(alphas=logspace(-4,4,20)), fit on "
                 "TRAIN+VAL (canonical)",
        "thresholds": "aeon MiniROCKET 9 golden-quantile biases, "
                      "TRAIN-fitted per kernel",
        "protocol": "canonical benchmark split + znorm; MiniROCKET fit on "
                    "TRAIN; MR block raw; family blocks train-standardized",
    }
    save_json(save_config, os.path.join(OUT_DIR, "configs",
                                        "frozen_config.json"))

    # STEP 1-5 + 8-10: primary per-dataset runs
    res_by_ds = {}
    for ds_name, ds_file in DATASETS:
        res_by_ds[ds_name] = run_dataset_primary(ds_name, ds_file)

    # STEP 11: multi-seed robustness M0 vs M6
    seed_rows = {}
    if RUN_MULTISEED:
        for ds_name, ds_file in DATASETS:
            seed_rows[ds_name] = run_multiseed(ds_name, ds_file, None)
    else:
        log("multi-seed skipped (TURS_SKB_SKIP_MULTISEED=1)")

    # STEP 6-7 handled inside primary (stats + diagnostics)

    # ---- CSV outputs ----
    primary_rows = []
    for d, r in res_by_ds.items():
        primary_rows.append({
            "dataset": d, "model": "M0_MiniROCKET",
            "macro_f1": r["m0"]["macro_f1"],
            "accuracy": r["m0"]["accuracy"],
            "weighted_f1": r["m0"]["weighted_f1"],
            "balanced_accuracy": r["m0"]["balanced_accuracy"],
            "feature_dim": r["m0"]["feature_dim"],
            "delta_vs_m0": 0.0, "reference": r["m0"]["reference_mf1"],
            "reproduction_delta": r["m0"]["reproduction_delta"],
            "ridge_fit_s": r["m0"]["ridge_fit_s"],
            "class_f1s": r["m0"]["class_f1s"],
            "class_precision": r["m0"]["class_precision"],
            "class_recall": r["m0"]["class_recall"]})
        for i, f in enumerate(FAMILIES, start=1):
            m = r["per_family"][f]
            primary_rows.append({
                "dataset": d, "model": f"M{i}_+{f}",
                "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
                "weighted_f1": m["weighted_f1"],
                "balanced_accuracy": m["balanced_accuracy"],
                "feature_dim": m["feature_dim"],
                "delta_vs_m0": m["delta_mf1"], "reference": "",
                "reproduction_delta": "",
                "ridge_fit_s": m["ridge_fit_s"], "class_f1s":
                    m["class_f1s"], "class_precision": m["class_precision"],
                "class_recall": m["class_recall"]})
        m = r["m6"]
        primary_rows.append({
            "dataset": d, "model": "M6_+All5", "macro_f1": m["macro_f1"],
            "accuracy": m["accuracy"], "weighted_f1": m["weighted_f1"],
            "balanced_accuracy": m["balanced_accuracy"],
            "feature_dim": m["feature_dim"],
            "delta_vs_m0": m["delta_mf1"], "reference": "",
            "reproduction_delta": "", "ridge_fit_s": m["ridge_fit_s"],
            "class_f1s": m["class_f1s"], "class_precision":
                m["class_precision"], "class_recall": m["class_recall"]})
    write_csv(os.path.join(OUT_DIR, "primary_results.csv"), primary_rows)

    family_rows = []
    for d, r in res_by_ds.items():
        for f in FAMILIES:
            family_rows.append({
                "dataset": d, "family": f,
                "val_screening_mf1": r["per_family"][f]
                ["val_screening_mf1"],
                "test_mf1": r["per_family"][f]["macro_f1"],
                "delta_mf1": r["per_family"][f]["delta_mf1"],
                "feature_dim": r["per_family"][f]["feature_dim"],
                "n_kernels_bank": len(BANK_BUILDERS[f]()),
                "duplicates_collapsed": r["duplicates_collapsed"][f]})
    write_csv(os.path.join(OUT_DIR, "family_results.csv"), family_rows)

    pair_rows = []
    for d, r in res_by_ds.items():
        for tag, m in r["pairwise"].items():
            pair_rows.append({"dataset": d, "pair": tag,
                              "val_screening_mf1": m["val_screening_mf1"],
                              "test_mf1": m["macro_f1"],
                              "delta_mf1": m["delta_mf1"],
                              "feature_dim": m["feature_dim"]})
    write_csv(os.path.join(OUT_DIR, "pairwise_results.csv"), pair_rows)

    stat_csv = []
    for d, r in res_by_ds.items():
        for row in r["stat_rows"]:
            stat_csv.append(dict(row))
    write_csv(os.path.join(OUT_DIR, "statistical_tests.csv"), stat_csv)

    seed_csv = []
    for d, s in seed_rows.items():
        for row in s["rows"]:
            seed_csv.append(row)
    write_csv(os.path.join(OUT_DIR, "seed_results.csv"), seed_csv)

    fdim_rows = []
    for d, r in res_by_ds.items():
        added = r["m6"]["feature_dim"] - r["m0"]["feature_dim"]
        fdim_rows.append({
            "dataset": d, "m0_dim": r["m0"]["feature_dim"],
            "m6_dim": r["m6"]["feature_dim"], "added_features": added,
            "delta_mf1_all5": r["m6"]["delta_mf1"],
            "delta_mf1_per_1k_features": round(
                r["m6"]["delta_mf1"] / max(added / 1000, 1e-9), 5),
            "family_dims": json.dumps(r["family_dims"])})
    write_csv(os.path.join(OUT_DIR, "feature_dimensions.csv"), fdim_rows)

    rt_rows = []
    for d, r in res_by_ds.items():
        rt_rows.append({
            "dataset": d, "mr_fit_s": r["timings"]["mr_fit_s"],
            "mr_transform_s": r["timings"]["mr_transform_s"],
            "skb_bank_fit_s": r["skb_fit_s"],
            "skb_extract_trva_s": r["skb_extract_trva_s"],
            "skb_extract_te_s": r["skb_extract_te_s"],
            "m0_ridge_fit_s": r["m0"]["ridge_fit_s"],
            "m6_ridge_fit_s": r["m6"]["ridge_fit_s"],
            "m0_inference_s": r["m0"]["inference_s"],
            "m6_inference_s": r["m6"]["inference_s"]})
    write_csv(os.path.join(OUT_DIR, "runtime.csv"), rt_rows)

    km_rows = []
    for d, r in res_by_ds.items():
        for row in r.get("kernel_metadata", []):
            km_rows.append({"dataset": d, **row})
    write_csv(os.path.join(OUT_DIR, "kernel_metadata.csv"), km_rows)

    diag_all = {d: {"sanity_blocks": r["sanity_blocks"],
                    "diagnostics": r["diagnostics"],
                    "val_screening": r["val_screening"],
                    "promising": r["promising"]}
                for d, r in res_by_ds.items()}
    save_json(diag_all, os.path.join(OUT_DIR, "diagnostics.json"))

    # mechanistic verdicts (pre-registered expectations vs outcome)
    mech_verdicts = {}
    for f, (targets, _) in MECH_EXPECTATIONS.items():
        tgt = [res_by_ds[d]["per_family"][f]["delta_mf1"] for d in
               res_by_ds if d in targets]
        oth = [res_by_ds[d]["per_family"][f]["delta_mf1"] for d in
               res_by_ds if d not in targets]
        tm = float(np.mean(tgt)) if tgt else float("nan")
        om = float(np.mean(oth)) if oth else float("nan")
        if not oth:
            verdict = ("SUPPORTED" if tm > 0 else "REJECTED") + \
                " (single-domain family)"
        elif tm > 0 and tm >= om:
            verdict = "SUPPORTED"
        elif tm > 0:
            verdict = "PARTIALLY SUPPORTED (positive but not targeted)"
        else:
            verdict = "REJECTED"
        mech_verdicts[f] = {"target_mean": round(tm, 4),
                            "other_mean": round(om, 4),
                            "verdict": verdict}

    # ---- figures ----
    fig1_architecture(os.path.join(OUT_DIR, "figures",
                                   "fig1_architecture.png"))
    fig2_main(res_by_ds, os.path.join(OUT_DIR, "figures",
                                      "fig2_main_results.png"))
    fig3_deltas(res_by_ds, os.path.join(OUT_DIR, "figures",
                                        "fig3_family_deltas.png"))
    fig4_cka(res_by_ds, os.path.join(OUT_DIR, "figures",
                                     "fig4_cka.png"))
    fig5_responses(res_by_ds, os.path.join(OUT_DIR, "figures",
                                           "fig5_kernel_responses.png"))
    fig6_cost(res_by_ds, os.path.join(OUT_DIR, "figures",
                                      "fig6_cost.png"))

    # ---- full results + summary ----
    save_json(res_by_ds, os.path.join(OUT_DIR, "full_results.json"))
    # headline aggregates shared with the report/classification
    ds_names = list(res_by_ds.keys())
    m0_mean = float(np.mean([res_by_ds[d]["m0"]["macro_f1"]
                             for d in ds_names]))
    m6_mean = float(np.mean([res_by_ds[d]["m6"]["macro_f1"]
                             for d in ds_names]))
    d6_mean = float(np.mean([res_by_ds[d]["m6"]["delta_mf1"]
                             for d in ds_names]))
    fam_mean_deltas = {
        f: round(float(np.mean(
            [res_by_ds[d]["per_family"][f]["delta_mf1"]
             for d in ds_names])), 4)
        for f in FAMILIES}
    best_fam = max(fam_mean_deltas, key=lambda k: fam_mean_deltas[k])
    n_sig6 = sum(1 for d in ds_names for r in res_by_ds[d]["stat_rows"]
                 if r["comparison"] == "M0 vs M6_ALL5" and
                 r.get("q_fdr", 1) < ALPHA)
    summary = {
        "experiment": "TURS-SKB", "completed": time.strftime(
            "%Y-%m-%d %H:%M:%S"),
        "seed": SEED,
        "mean_m0_mf1": round(m0_mean, 4),
        "mean_m6_mf1": round(m6_mean, 4),
        "mean_delta_all5": round(d6_mean, 4),
        "mean_family_deltas": fam_mean_deltas,
        "best_family": best_fam,
        "n_fdr_significant_m6": n_sig6,
        "mech_verdicts": mech_verdicts,
        "multiseed": {d: {k: v for k, v in s.items() if k != "rows"}
                      for d, s in seed_rows.items()},
        "sanity_all_ok": all(
            all(b.get("finite", True) for b in r["sanity_blocks"])
            for r in res_by_ds.values()),
    }
    classification = classify(
        res_by_ds, seed_rows, d6_mean, n_sig6, best_fam) if res_by_ds \
        else "n/a"
    summary["classification"] = classification
    save_json(summary, os.path.join(OUT_DIR, "summary.json"))

    csv_summary = []
    for d, r in res_by_ds.items():
        row = {"dataset": d, "m0_mf1": r["m0"]["macro_f1"]}
        for f in FAMILIES:
            row[f"d_{FAM_TAG[f]}"] = r["per_family"][f]["delta_mf1"]
        row["d_all5"] = r["m6"]["delta_mf1"]
        csv_summary.append(row)
    write_csv(os.path.join(OUT_DIR, "summary.csv"), csv_summary)

    # ---- report ----
    report = build_report(res_by_ds, seed_rows, [], mech_verdicts)
    with open(os.path.join(OUT_DIR, "reports", "REPORT.md"), "w",
              encoding="utf-8") as f:
        f.write(report)

    # ---- copy tests for provenance ----
    tdir = os.path.join(ROOT, "tests")
    for tf in ("test_turs_skb.py",):
        src = os.path.join(tdir, tf)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(OUT_DIR, "tests", tf))

    log(f"TURS-SKB complete in {(time.time()-t_start)/60:.1f} min; "
        f"classification={classification}")
    _logf.close()


if __name__ == "__main__":
    main()
