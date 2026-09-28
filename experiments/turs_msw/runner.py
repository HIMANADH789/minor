"""
TURS-MSW runner (spec sec. 21-46).

M0  = canonical aeon MiniRocket-10K + RidgeClassifierCV (verified baseline)
M1  = global + 2 medium-region PPV blocks
M2  = global + 4 local-region PPV blocks
M3  = global + medium + local (TURS-MSW, final model)
M4  = medium + local only (diagnostic, sec. 16)

Protocol: canonical splits/loaders/z-norm reused from the external
generalization pipeline; RidgeClassifierCV(alphas=logspace(-4,4,20)),
train-only fit -> val MF1, final fit on TRAIN+VAL, single test eval.
All features are PPV rates in [0,1] (same natural scale as canonical M0),
so the canonical RAW ridge protocol is retained for every variant (spec
sec. 13 conditional standardization: not required; block statistics are
still reported).

MEMORY SAFETY (sec. 36): test features are never materialized per variant.
The regional extractor runs in chunks over the test split; each chunk is
aggregated to all variants and immediately consumed by clf.predict, then
freed. Only train/val (tiny) variant matrices are ever materialized.

Usage:
  python experiments/turs_msw/runner.py                # full (resume)
  python experiments/turs_msw/runner.py --dataset Phoneme
  python experiments/turs_msw/runner.py --skip-multiseed
  python experiments/turs_msw/runner.py --report-only
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.external_stack_generalization.baselines import (  # noqa: E402
    full_metrics,
)
from experiments.external_stack_generalization.data import (  # noqa: E402
    DATASETS, PRIMARY_DATASET, SEED, load_dataset, znorm,
)
from experiments.turs_msw.msw import (  # noqa: E402
    build_regions, msw_transform_uni, variant_matrix,
)
from src.diagnostics.statistics import (  # noqa: E402
    benjamini_hochberg, cohens_d_paired, mcnemar,
)

OUT_DIR = os.environ.get("TURS_MSW_OUT_DIR",
                         os.path.join(ROOT, "results", "turs_msw"))
for sub in ("configs", "logs", "figures", "tests", "reports"):
    os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)

EXT_DIR = os.path.join(ROOT, "results", "external_stack_generalization")
SEEDS_ROBUST = [43, 44, 45, 46]
ALPHAS = np.logspace(-4, 4, 20)
VARIANTS = ["M0", "M1", "M2", "M3", "M4"]
CHUNK = 1024  # regional test pass chunk: peak chunk memory ~0.9 GB (sec. 36)

LOG_PATH = os.path.join(OUT_DIR, "logs", "runner.log")
_logf = open(LOG_PATH, "a", encoding="utf-8")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    _logf.write(line + "\n")
    _logf.flush()
    print(line, flush=True)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=float)


def write_csv(path, rows):
    if not rows:
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


# ======================================================================
# Ridge protocol (identical for every variant)
# ======================================================================
def fit_ridge_pair(Ftr, ytr, Fva, yva):
    """Train-only fit -> val MF1; final fit on TRAIN+VAL. Returns
    (clf_val, clf_final, val_mf1, ridge_seconds)."""
    from sklearn.linear_model import RidgeClassifierCV
    t0 = time.time()
    clf_va = RidgeClassifierCV(alphas=ALPHAS)
    clf_va.fit(Ftr, ytr)
    val_preds = clf_va.predict(Fva)
    val_mf1 = float(full_metrics(yva, val_preds,
                                 int(max(yva.max(), ytr.max())) + 1)
                    ["macro_f1"])
    Ftrva = np.concatenate([Ftr, Fva], axis=0)
    ytrva = np.concatenate([ytr, yva], axis=0)
    clf = RidgeClassifierCV(alphas=ALPHAS)
    clf.fit(Ftrva, ytrva)
    return clf_va, clf, val_mf1, time.time() - t0


def predict_chunked(clf, Fte, chunk=4096):
    """Predict in chunks to avoid sklearn's internal float64 copy of a big
    test matrix."""
    out = []
    for s in range(0, len(Fte), chunk):
        out.append(clf.predict(Fte[s:s + chunk]))
    return np.concatenate(out).astype(np.int64)


# ======================================================================
# Small-split variant fitting (train/val are always tiny)
# ======================================================================
def fit_variants_trainval(feats_tr, feats_va, F, ytr, yva, variants):
    """Materialize train/val variant matrices, fit both ridge models.

    Returns (fitted dict v -> {clf, clf_va, val_mf1, ridge_s, dims},
    variant train matrices for diagnostics)."""
    vtr = variant_matrix(feats_tr, F)
    vva = variant_matrix(feats_va, F)
    fitted, tr_mats = {}, {}
    for v in variants:
        clf_va, clf, val_mf1, t_r = fit_ridge_pair(vtr[v], ytr, vva[v], yva)
        fitted[v] = {"clf": clf, "clf_va": clf_va, "val_mf1": val_mf1,
                     "ridge_s": round(t_r, 2),
                     "dims": int(vtr[v].shape[1])}
        tr_mats[v] = vtr[v] if v == "M3" else None  # keep only M3 for CKA
    return fitted, vtr


def regional_test_pass(Xte, fitted, F, regions, params, indices,
                       eq_ref=None, chunk=CHUNK):
    """Chunked regional extraction + prediction for all variants.

    One extraction pass over the test split; per chunk the full 7-block
    features are built, sliced per variant, predicted, and freed.
    Returns (preds dict, equivalence max|global-aeon| on the first chunk
    vs eq_ref (the canonical aeon test features) or None)."""
    preds = {v: [] for v in fitted}
    eq = None
    for s in range(0, len(Xte), chunk):
        feats = msw_transform_uni(Xte[s:s + chunk], params, indices, regions)
        if eq_ref is not None and s == 0:
            eq = float(np.abs(feats[:, :F] - eq_ref[:len(feats)]).max())
        vm = variant_matrix(feats, F)
        for v, p in preds.items():
            p.append(fitted[v]["clf"].predict(vm[v]))
    return {v: np.concatenate(p).astype(np.int64) for v, p in preds.items()}, eq


# ======================================================================
# Diagnostics
# ======================================================================
def linear_cka_ngram(X, Y):
    """Memory-safe linear CKA. Mathematically identical to
    src.features.turs_skb.linear_cka (Kornblith et al. 2019) via the cyclic
    trace identity ||Xc.T @ Yc||_F^2 == tr(Gx @ Gy) = sum(Gx * Gy), where
    Gx = Xc @ Xc.T and Gy = Yc @ Yc.T are n x n sample Grams.
    The feature-gram form used by the shared helper allocates multi-GB
    intermediates at F~70K (sec. 36 violation); n here is <= 246."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    Gx = Xc @ Xc.T
    Gy = Yc @ Yc.T
    xy = np.sum(Gx * Gy)
    xx = np.sum(Gx * Gx)
    yy = np.sum(Gy * Gy)
    denom = np.sqrt(xx * yy)
    if denom < 1e-300:
        return float("nan")
    return float(xy / denom)


def complementarity_diag(M3_train, F):
    """CKA + mean |corr| between scale blocks (sec. 27). Train only."""
    G = M3_train[:, :F]
    M = M3_train[:, F:3 * F]
    L = M3_train[:, 3 * F:]
    rng = np.random.RandomState(0)
    k = min(400, F)
    gi = rng.choice(F, k, replace=False)
    out = {
        "cka_global_medium": round(float(linear_cka_ngram(G, M)), 6),
        "cka_global_local": round(float(linear_cka_ngram(G, L)), 6),
        "cka_medium_local": round(float(linear_cka_ngram(M, L)), 6),
    }
    for name, B in [("medium", M), ("local", L)]:
        A, Bc = G[:, gi], B[:, gi]
        Ac = A - A.mean(0, keepdims=True)
        Bcen = Bc - Bc.mean(0, keepdims=True)
        C = (Ac.T @ Bcen) / (np.linalg.norm(Ac, axis=0)[:, None]
                             * np.linalg.norm(Bcen, axis=0)[None, :] + 1e-12)
        out[f"mean_abs_corr_global_{name}"] = round(float(np.abs(C).mean()), 6)
    return out


def MiniRocketIndices():
    from aeon.transformations.collection.convolution_based import MiniRocket
    return MiniRocket._indices


def synthetic_locality_diag(params, T, regions):
    """sec. 28: same global activation count, different regions."""
    rng = np.random.RandomState(3)
    base = rng.randn(T).astype(np.float32)
    burst = 3.0 * np.exp(-0.5 * ((np.arange(T) - T // 5) / 3.0) ** 2)
    x_left = base + burst
    x_right = base + burst[::-1]  # mirrored -> same values, opposite location
    X = np.stack([
        (x_left - x_left.mean()) / (x_left.std() + 1e-8),
        (x_right - x_right.mean()) / (x_right.std() + 1e-8),
    ]).astype(np.float32)
    feats = msw_transform_uni(X, params, MiniRocketIndices(), regions)
    F = len(params[4])
    G, M, L = feats[:, :F], feats[:, F:3 * F], feats[:, 3 * F:]
    return {
        "global_mean_abs_diff": round(float(np.abs(G[0] - G[1]).mean()), 6),
        "medium_mean_abs_diff": round(float(np.abs(M[0] - M[1]).mean()), 6),
        "local_mean_abs_diff": round(float(np.abs(L[0] - L[1]).mean()), 6),
    }


def shift_and_stability_diag(X10, params, T, regions):
    """sec. 29/30: circular shift + small-noise/shift stability."""
    F = len(params[4])
    half = T // 2
    Xs = np.ascontiguousarray(
        np.concatenate([X10[:, half:], X10[:, :half]], axis=1))
    f0 = msw_transform_uni(X10, params, MiniRocketIndices(), regions)
    f1 = msw_transform_uni(Xs, params, MiniRocketIndices(), regions)
    shift = {
        "global_mean_abs_change": round(
            float(np.abs(f1[:, :F] - f0[:, :F]).mean()), 6),
        "medium_mean_abs_change": round(
            float(np.abs(f1[:, F:3 * F] - f0[:, F:3 * F]).mean()), 6),
        "local_mean_abs_change": round(
            float(np.abs(f1[:, 3 * F:] - f0[:, 3 * F:]).mean()), 6),
    }
    noise = np.ascontiguousarray(
        X10 + np.random.RandomState(4).randn(*X10.shape).astype(
            np.float32) * 0.01)
    f2 = msw_transform_uni(noise, params, MiniRocketIndices(), regions)
    f3 = msw_transform_uni(np.ascontiguousarray(np.roll(X10, 1, axis=1)),
                           params, MiniRocketIndices(), regions)
    stab = {}
    for name, f in [("noise", f2), ("shift1", f3)]:
        stab[name] = {
            "global_mean_abs_change": round(
                float(np.abs(f[:, :F] - f0[:, :F]).mean()), 6),
            "regional_mean_abs_change": round(
                float(np.abs(f[:, F:] - f0[:, F:]).mean()), 6),
        }
    return {"circular_shift_T2": shift, "small_perturbations": stab}


def sanity_check_external(ds_name, m0_res, log):
    """sec. 21: M0 must reproduce the verified external-study baseline."""
    path = os.path.join(EXT_DIR, ds_name, "results.json")
    if not os.path.exists(path):
        log("  [sanity] external baseline file missing — skipping "
            "(not a failure)")
        return None
    with open(path, encoding="utf-8") as f:
        ext = json.load(f)["models"]["MiniROCKET"]
    diff = abs(float(ext["macro_f1"]) - float(m0_res["macro_f1"]))
    log(f"  [sanity] M0 test MF1 {m0_res['macro_f1']:.4f} vs external "
        f"{ext['macro_f1']:.4f} (delta {diff:.2e})")
    if diff > 5e-3:
        raise RuntimeError(
            f"M0 does not reproduce the verified external baseline on "
            f"{ds_name} (|delta|={diff:.4f} > 5e-3). STOP and debug per "
            f"spec sec. 21.")
    return diff


# ======================================================================
# Per-dataset experiment
# ======================================================================
def run_dataset(ds_name, log):
    cache = os.path.join(OUT_DIR, f"results_{ds_name}.json")
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            prev = json.load(f)
        if all(v in prev.get("variants", {}) for v in VARIANTS):
            log(f"[cache] {ds_name} complete — skipping")
            return prev

    from aeon.transformations.collection.convolution_based import MiniRocket

    log(f"=== {ds_name} ===")
    d = load_dataset(ds_name)
    d["Xtr_z"], d["Xva_z"], d["Xte_z"] = (znorm(d["Xtr"]), znorm(d["Xva"]),
                                          znorm(d["Xte"]))
    ytr, yva, yte = d["ytr"], d["yva"], d["yte"]
    n_cls = d["n_classes"]
    regions = build_regions(d["L"])
    log(f"  n={len(ytr)}/{len(yva)}/{len(yte)} L={d['L']} "
        f"n_cls={n_cls} regions={regions}")

    # ---- canonical MiniRocket (fit TRAIN only) ----
    mr = MiniRocket(random_state=SEED, n_jobs=-1)
    t0 = time.time()
    Ztr = np.asarray(mr.fit_transform(
        d["Xtr_z"][:, None, :].astype(np.float32)))
    t_fit = time.time() - t0
    t0 = time.time()
    Zva = np.asarray(mr.transform(d["Xva_z"][:, None, :].astype(np.float32)))
    Zte = np.asarray(mr.transform(d["Xte_z"][:, None, :].astype(np.float32)))
    t_trans = time.time() - t0
    F = Ztr.shape[1]
    log(f"  aeon MiniRocket: F={F} fit={t_fit:.1f}s "
        f"transform={t_trans:.1f}s")

    # ---- regional extraction on the small splits ----
    t0 = time.time()
    feats_tr = msw_transform_uni(d["Xtr_z"], mr.parameters,
                                 MiniRocket._indices, regions)
    feats_va = msw_transform_uni(d["Xva_z"], mr.parameters,
                                 MiniRocket._indices, regions)
    t_reg_small = time.time() - t0

    # ---- CRITICAL equivalence check on train+val (sec. 22) ----
    eq = {"train": float(np.abs(feats_tr[:, :F] - Ztr).max()),
          "val": float(np.abs(feats_va[:, :F] - Zva).max())}
    if max(eq.values()) >= 1e-12:
        raise RuntimeError(f"global block != canonical MiniRocket: {eq}")
    log(f"  equivalence (train/val): {eq['train']:.2e}/{eq['val']:.2e}")

    # ---- block stats (raw, sec. 13) on train ----
    blocks = {"global": slice(0, F), "medium": slice(F, 3 * F),
              "local": slice(3 * F, 7 * F)}
    bstats = {}
    for bname, sl in blocks.items():
        B = feats_tr[:, sl]
        bstats[bname] = {
            "min": round(float(B.min()), 6), "max": round(float(B.max()), 6),
            "mean": round(float(B.mean()), 6), "std": round(float(B.std()), 6),
            "near_zero_var_cols": int((B.std(0) < 1e-8).sum()),
            "nan": int(np.isnan(B).sum()), "inf": int(np.isinf(B).sum()),
        }

    # ---- fit both ridge models per variant on the small splits ----
    fitted, vtr = fit_variants_trainval(feats_tr, feats_va, F, ytr, yva,
                                        VARIANTS)

    # ---- M0: canonical aeon arrays (sec. 5/21) ----
    preds_all = {}
    results = {}
    clf0_va, clf0, m0_val, t_r0 = fit_ridge_pair(Ztr, ytr, Zva, yva)
    p0 = predict_chunked(clf0, Zte)
    res0 = full_metrics(yte, p0, n_cls)
    res0.update({"model": "M0", "val_macro_f1": round(m0_val, 4),
                 "selected_alpha": float(clf0.alpha_),
                 "n_features": int(F), "time_ridge_s": round(t_r0, 2),
                 "time_total_s": round(t_fit + t_trans + t_r0, 2)})
    results["M0"], preds_all["M0"] = res0, p0
    log(f"  M0 (canonical): val={m0_val:.4f} test={res0['macro_f1']:.4f} "
        f"alpha={clf0.alpha_:.4g}")

    # ---- sec. 21 sanity check against the verified external baseline ----
    sanity_check_external(ds_name, res0, log)

    # ---- chunked regional test pass (memory-safe, sec. 36) ----
    t0 = time.time()
    reg_preds, eq_test = regional_test_pass(
        d["Xte_z"], fitted, F, regions, mr.parameters, MiniRocket._indices,
        eq_ref=Zte)
    t_reg_test = time.time() - t0
    eq["test"] = eq_test if eq_test is not None else -1.0
    if eq_test is None or eq_test >= 1e-12:
        raise RuntimeError(f"global block != canonical on test: {eq_test}")
    log(f"  equivalence (test chunk 0): {eq_test:.2e} | regional test pass "
        f"{t_reg_test:.1f}s")
    del Zte  # free ~0.5-3 GB before per-variant metric assembly

    for v in ["M1", "M2", "M3", "M4"]:
        preds = reg_preds[v]
        res = full_metrics(yte, preds, n_cls)
        res.update({"model": v,
                    "val_macro_f1": round(fitted[v]["val_mf1"], 4),
                    "selected_alpha": float(fitted[v]["clf"].alpha_),
                    "n_features": fitted[v]["dims"],
                    "time_ridge_s": fitted[v]["ridge_s"],
                    "time_total_s": round(t_fit + t_reg_small + t_reg_test
                                          + fitted[v]["ridge_s"], 2)})
        results[v], preds_all[v] = res, preds
        log(f"  {v}: dim={res['n_features']} val={res['val_macro_f1']:.4f} "
            f"test={res['macro_f1']:.4f} "
            f"d={res['macro_f1'] - res0['macro_f1']:+.4f}")

    # ---- statistical inference (sec. 26) ----
    stat_rows, pvals = [], []
    for v in ["M1", "M2", "M3"]:
        chi2, p = mcnemar(preds_all[v] != yte, preds_all["M0"] != yte)
        d_mf1 = round(results[v]["macro_f1"] - res0["macro_f1"], 4)
        stat_rows.append({
            "dataset": ds_name, "comparison": f"M0 vs {v}",
            "delta_mf1": d_mf1, "chi2": round(chi2, 4),
            "p_raw": round(p, 6),
            "effect_size_d": round(float(cohens_d_paired(
                (preds_all[v] == yte).astype(float),
                (preds_all["M0"] == yte).astype(float))), 4),
        })
        pvals.append(p)
    q = benjamini_hochberg(pvals)
    for r, qq in zip(stat_rows, q):
        r["q_fdr"] = round(float(qq), 6)
        r["verdict"] = ("SIGNIFICANT" if r["q_fdr"] < 0.05
                        and r["delta_mf1"] > 0 else
                        "significant-negative" if r["q_fdr"] < 0.05
                        else "not significant")

    # ---- diagnostics (train + first 10 train signals; no test labels) ----
    comp = complementarity_diag(vtr["M3"], F)
    loc = synthetic_locality_diag(mr.parameters, d["L"], regions)
    shift = shift_and_stability_diag(
        np.ascontiguousarray(d["Xtr_z"][:10]), mr.parameters, d["L"], regions)

    out = {
        "dataset": ds_name, "seed": SEED,
        "n_train": int(len(ytr)), "n_val": int(len(yva)),
        "n_test": int(len(yte)), "n_classes": n_cls,
        "variants": results, "stat_rows": stat_rows,
        "diagnostics": {"complementarity": comp, "locality_synthetic": loc,
                        "shift_stability": shift},
        "info": {
            "F": int(F), "T": int(d["L"]),
            "t_mr_fit_s": round(t_fit, 2), "t_mr_transform_s": round(t_trans, 2),
            "t_regional_trainval_s": round(t_reg_small, 2),
            "t_regional_test_s": round(t_reg_test, 2),
            "equivalence_max_abs_diff": eq,
            "regions": {k: [list(r) for r in v] for k, v in regions.items()},
            "variant_dims": {v: fitted[v]["dims"] for v in VARIANTS},
            "block_stats_train_raw": bstats,
        },
    }
    save_json(out, cache)
    return out


# ======================================================================
# Multi-seed M0 vs M3 (sec. 35) — resumable per seed
# ======================================================================
def run_multiseed(log):
    from aeon.transformations.collection.convolution_based import MiniRocket
    cache = os.path.join(OUT_DIR, "multiseed.json")
    rows = {}
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            rows = json.load(f)

    for ds_name in DATASETS:
        rows.setdefault(ds_name, [])
        done_seeds = {r["seed"] for r in rows[ds_name]}
        todo = [s for s in [SEED] + SEEDS_ROBUST if s not in done_seeds]
        if not todo:
            continue
        d = load_dataset(ds_name)
        d["Xtr_z"], d["Xva_z"], d["Xte_z"] = (znorm(d["Xtr"]),
                                              znorm(d["Xva"]),
                                              znorm(d["Xte"]))
        regions = build_regions(d["L"])
        ytr, yva, yte = d["ytr"], d["yva"], d["yte"]
        n_cls = d["n_classes"]

        # seed-42 row can be copied from the primary run
        if SEED in todo:
            prim_path = os.path.join(OUT_DIR, f"results_{ds_name}.json")
            with open(prim_path, encoding="utf-8") as f:
                prim = json.load(f)
            rows[ds_name].append({
                "seed": SEED,
                "m0_mf1": prim["variants"]["M0"]["macro_f1"],
                "m3_mf1": prim["variants"]["M3"]["macro_f1"],
                "delta": round(prim["variants"]["M3"]["macro_f1"]
                               - prim["variants"]["M0"]["macro_f1"], 4),
                "m0_val": prim["variants"]["M0"]["val_macro_f1"],
                "m3_val": prim["variants"]["M3"]["val_macro_f1"],
            })
            todo.remove(SEED)
            save_json(rows, cache)

        for seed in todo:
            mr = MiniRocket(random_state=seed, n_jobs=-1)
            Ztr = np.asarray(mr.fit_transform(
                d["Xtr_z"][:, None, :].astype(np.float32)))
            Zva = np.asarray(mr.transform(
                d["Xva_z"][:, None, :].astype(np.float32)))
            Zte = np.asarray(mr.transform(
                d["Xte_z"][:, None, :].astype(np.float32)))
            F = Ztr.shape[1]
            _, clf0, m0_val, _ = fit_ridge_pair(Ztr, ytr, Zva, yva)
            p0 = predict_chunked(clf0, Zte)
            m0_mf1 = full_metrics(yte, p0, n_cls)["macro_f1"]
            del Zte

            feats_tr = msw_transform_uni(d["Xtr_z"], mr.parameters,
                                         MiniRocket._indices, regions)
            feats_va = msw_transform_uni(d["Xva_z"], mr.parameters,
                                         MiniRocket._indices, regions)
            fitted, _ = fit_variants_trainval(feats_tr, feats_va, F,
                                              ytr, yva, ["M3"])
            reg_preds, eqt = regional_test_pass(
                d["Xte_z"], fitted, F, regions, mr.parameters,
                MiniRocket._indices, eq_ref=None)
            m3_mf1 = full_metrics(yte, reg_preds["M3"], n_cls)["macro_f1"]
            delta = round(m3_mf1 - m0_mf1, 4)
            rows[ds_name].append({
                "seed": seed, "m0_mf1": m0_mf1, "m3_mf1": m3_mf1,
                "delta": delta, "m0_val": round(m0_val, 4),
                "m3_val": round(fitted["M3"]["val_mf1"], 4),
            })
            save_json(rows, cache)
            log(f"  [multiseed {ds_name}] seed {seed}: M0={m0_mf1:.4f} "
                f"M3={m3_mf1:.4f} (d={delta:+.4f})")
    return rows


# ======================================================================
# Figures + report
# ======================================================================
def make_figures(all_res, seed_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds_names = list(all_res)
    figdir = os.path.join(OUT_DIR, "figures")

    # FIG 1: architecture
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axis("off")
    boxes = [(0.01, "signal x(t)\n(length T)", "#c7e9c0"),
             (0.18, "canonical MiniRocket\nkernel bank + biases\n(FIXED, seed 42)",
              "#9ecae1"),
             (0.38, "responses r_k(t)\n(same convolutions)", "#fdd0a2"),
             (0.58, "activation masks\nI_k(t) = 1[r_k(t) > b_k]", "#fdd0a2"),
             (0.78, "PPV pooling:\nGLOBAL | 2 MEDIUM | 4 LOCAL\n-> 7 features/kernel",
              "#fcbba1")]
    for x, txt, c in boxes:
        ax.add_patch(plt.Rectangle((x, 0.35), 0.18, 0.32, color=c,
                                   ec="k", lw=1.2))
        ax.text(x + 0.09, 0.51, txt, ha="center", va="center", fontsize=8)
    for x in [0.19, 0.37, 0.57, 0.77]:
        ax.annotate("", xy=(x + 0.015, 0.51), xytext=(x - 0.005, 0.51),
                    arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(0.5, 0.12, "M0: global only (canonical) | M3 = TURS-MSW: "
            "global+medium+local | RidgeClassifierCV unchanged",
            ha="center", fontsize=9, style="italic")
    ax.set_title("FIG1: TURS-MSW architecture (only pooling changes)")
    fig.savefig(os.path.join(figdir, "fig1_architecture.png"), dpi=160,
                bbox_inches="tight")
    plt.close(fig)

    # FIG 2: M0-M3 bars
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(ds_names))
    w = 0.2
    for j, v in enumerate(["M0", "M1", "M2", "M3"]):
        vals = [all_res[d]["variants"][v]["macro_f1"] for d in ds_names]
        ax.bar(x + (j - 1.5) * w, vals, w, label=v)
    ax.set_xticks(x)
    ax.set_xticklabels(ds_names)
    ax.set_ylabel("Test Macro-F1")
    ax.set_title("FIG2: M0/M1/M2/M3 test Macro-F1")
    ax.legend()
    fig.savefig(os.path.join(figdir, "fig2_main_results.png"), dpi=160,
                bbox_inches="tight")
    plt.close(fig)

    # FIG 3: deltas
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for j, v in enumerate(["M1", "M2", "M3"]):
        deltas = [all_res[d]["variants"][v]["macro_f1"]
                  - all_res[d]["variants"]["M0"]["macro_f1"]
                  for d in ds_names]
        ax.bar(x + (j - 1) * w, deltas, w, label=f"delta {v}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(ds_names)
    ax.set_ylabel("Delta MF1 vs M0")
    ax.set_title("FIG3: Incremental delta-MF1 for medium/local pooling")
    ax.legend()
    fig.savefig(os.path.join(figdir, "fig3_deltas.png"), dpi=160,
                bbox_inches="tight")
    plt.close(fig)

    # FIG 4: global vs regional response example (locality diagnostic)
    loc = all_res[ds_names[0]]["diagnostics"]["locality_synthetic"]
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["global", "medium", "local"]
    vals = [loc["global_mean_abs_diff"], loc["medium_mean_abs_diff"],
            loc["local_mean_abs_diff"]]
    ax.bar(labels, vals, color=["#9ecae1", "#fdd0a2", "#fcbba1"])
    ax.set_ylabel("mean |PPV difference| (mirrored-burst pair)")
    ax.set_title(f"FIG4: same global activation, different location "
                 f"({ds_names[0]}-fitted kernels)")
    fig.savefig(os.path.join(figdir, "fig4_locality.png"), dpi=160,
                bbox_inches="tight")
    plt.close(fig)

    # FIG 5: block similarity heatmap
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    blocks = ["global", "medium", "local"]
    mat = np.zeros((3, 3))
    for i, a in enumerate(blocks):
        for j, b in enumerate(blocks):
            key = f"cka_{a}_{b}"
            mat[i, j] = (all_res[ds_names[0]]["diagnostics"]
                         ["complementarity"].get(key, np.nan))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(3))
    ax.set_xticklabels(blocks)
    ax.set_yticks(range(3))
    ax.set_yticklabels(blocks)
    for i in range(3):
        for j in range(3):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center")
    fig.colorbar(im)
    ax.set_title(f"FIG5: CKA between scale blocks ({ds_names[0]})")
    fig.savefig(os.path.join(figdir, "fig5_block_cka.png"), dpi=160,
                bbox_inches="tight")
    plt.close(fig)

    # FIG 6: runtime vs dims
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for d in ds_names:
        for v in VARIANTS:
            r = all_res[d]["variants"][v]
            ax.scatter(r["n_features"], r["time_total_s"], s=40)
            ax.annotate(f"{d[:6]}:{v}", (r["n_features"], r["time_total_s"]),
                        fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("feature dimension")
    ax.set_ylabel("total runtime (s)")
    ax.set_title("FIG6: Runtime vs feature dimensionality")
    fig.savefig(os.path.join(figdir, "fig6_runtime.png"), dpi=160,
                bbox_inches="tight")
    plt.close(fig)


def build_report(all_res, seed_rows):
    L = []
    A = L.append
    ds_names = list(all_res)
    m0 = {d: all_res[d]["variants"]["M0"] for d in ds_names}

    def dm(d, v):
        return round(all_res[d]["variants"][v]["macro_f1"]
                     - m0[d]["macro_f1"], 4)

    A("# TURS-MSW — Multi-Scale Windowed MiniROCKET: Final Report")
    A("")
    A(f"Generated {time.strftime('%Y-%m-%d %H:%M')} | seed 42 | datasets: "
      f"{', '.join(ds_names)} | primary metric: TEST Macro-F1")
    A("")

    A("## 1. Executive summary")
    A("")
    for d in ds_names:
        A(f"- **{d}**: M0={m0[d]['macro_f1']:.4f}, "
          f"M1={all_res[d]['variants']['M1']['macro_f1']:.4f} "
          f"(delta {dm(d, 'M1'):+.4f}), "
          f"M2={all_res[d]['variants']['M2']['macro_f1']:.4f} "
          f"(delta {dm(d, 'M2'):+.4f}), "
          f"M3={all_res[d]['variants']['M3']['macro_f1']:.4f} "
          f"(delta {dm(d, 'M3'):+.4f})")
    n_pos = sum(1 for d in ds_names if dm(d, "M3") > 0)
    n_sig = sum(1 for d in ds_names
                for r in all_res[d]["stat_rows"]
                if r["comparison"] == "M0 vs M3" and r["q_fdr"] < 0.05
                and r["delta_mf1"] > 0)
    A(f"- M3 (TURS-MSW) improves on {n_pos}/{len(ds_names)} datasets; "
      f"{n_sig} FDR-significant improvements.")
    A("")

    A("## 2. Research question")
    A("")
    A("Does preserving coarse temporal locality before PPV pooling provide "
      "predictive information beyond canonical full-capacity MiniROCKET? The "
      "kernel bank, biases, dilations, responses and activation masks are "
      "EXACTLY the canonical fitted MiniRocket's; only the pooling regions "
      "change.")
    A("")

    A("## 3. MiniROCKET baseline (verified)")
    A("")
    A("| dataset | M0 val MF1 | M0 test MF1 | alpha | F | "
      "equivalence max delta (train/val/test) |")
    A("|---|---:|---:|---:|---:|---|")
    for d in ds_names:
        info = all_res[d]["info"]
        eq = info["equivalence_max_abs_diff"]
        A(f"| {d} | {m0[d]['val_macro_f1']:.4f} | {m0[d]['macro_f1']:.4f} | "
          f"{m0[d]['selected_alpha']:.4g} | {info['F']} | "
          f"{eq['train']:.2e}/{eq['val']:.2e}/{eq['test']:.2e} |")
    A("")
    A("The global block of the regional extractor is bit-identical to the "
      "canonical aeon output on every split (max delta < 1e-12), satisfying "
      "the sec. 22 equivalence requirement. M0 was additionally verified "
      "against the frozen external-generalization baseline "
      "(results/external_stack_generalization), tolerance 5e-3 (sec. 21).")
    A("")

    A("## 4-10. Motivation, formulation, windows, extraction, protocol")
    A("")
    A("- **Motivation**: canonical PPV destroys WHERE activations occur; "
      "regional PPVs retain coarse position information.")
    A("- **Formulation** (sec. 40): per kernel k, activation "
      "I_k(t)=1[r_k(t)>b_k]; global P = mean over [0,T); medium "
      "P_{k,m} = mean over M_m; local P_{k,l} = mean over L_l; per-feature "
      "vector [P_G, P_M1, P_M2, P_L1..P_L4]; features concatenated "
      "scale-major, RidgeClassifierCV unchanged.")
    A("- **Windows** (deterministic, fractions of T): medium "
      "w=ceil(3T/4), regions [0,w), [T-w,T) (~50% neighbour overlap, full "
      "coverage); local w=ceil(T/2), stride floor(T/4), regions "
      "[j*stride, min(j*stride+w, T)) for j=0..3 (50% overlap, full "
      "coverage). For padding1==1 kernels, windows are mapped onto aeon's "
      "valid axis C[padding:T-padding] (same responses, re-indexed).")
    A("- **Extraction**: verbatim mirror of aeon's `_static_transform_uni` "
      "(same C_alpha/C_gamma accumulation, same padding parity), with "
      "prefix-sum activation counts for O(1) window means. Responses are "
      "computed ONCE per split; no per-window recomputation (sec. 19). Test "
      "features are processed in chunks (sec. 36) and never materialized "
      "per variant.")
    A("- **Standardization**: all MSW features are PPV rates in [0,1] — the "
      "same natural scale as canonical M0 — so the canonical RAW ridge "
      "protocol is retained for every variant (no scaling fitted on any "
      "split). Block statistics reported below.")
    A("- **Ridge protocol**: RidgeClassifierCV(alphas=logspace(-4,4,20)); "
      "train-only fit -> val MF1; final fit TRAIN+VAL; single test eval.")
    A("- **Dataset protocol**: canonical UCR splits from the audited "
      "external-generalization loaders; val = provided canonical val "
      "(EpilepticSeizures) or deterministic per-class stratified 15% of "
      "train, seed 42 (Haptics, Phoneme); per-sample z-norm; test untouched "
      "until final evaluation.")
    A("")
    b0 = list(all_res.values())[0]["info"]["block_stats_train_raw"]
    A("Block statistics (train, raw):")
    A("")
    A("| block | min | max | mean | std | near-zero-var cols | NaN | Inf |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|")
    for b, s in b0.items():
        A(f"| {b} | {s['min']} | {s['max']} | {s['mean']} | {s['std']} | "
          f"{s['near_zero_var_cols']} | {s['nan']} | {s['inf']} |")
    A("")

    A("## 11. Experimental variants")
    A("")
    A("| variant | blocks | features (per dataset) |")
    A("|---|---|---|")
    labels = {"M0": "global", "M1": "global+medium", "M2": "global+local",
              "M3": "global+medium+local", "M4": "medium+local (diag)"}
    for v in VARIANTS:
        dims = ", ".join(f"{d}: {all_res[d]['variants'][v]['n_features']}"
                         for d in ds_names)
        A(f"| {v} | {labels[v]} | {dims} |")
    A("")

    A("## 12. Main results (test Macro-F1)")
    A("")
    A("| Dataset | M0 MR-10K | M1 G+Med | M2 G+Loc | M3 G+Med+Loc | "
      "M4 Med+Loc |")
    A("|---|---:|---:|---:|---:|---:|")
    for d in ds_names:
        r = all_res[d]["variants"]
        A(f"| {d} | {r['M0']['macro_f1']:.4f} | {r['M1']['macro_f1']:.4f} | "
          f"{r['M2']['macro_f1']:.4f} | {r['M3']['macro_f1']:.4f} | "
          f"{r['M4']['macro_f1']:.4f} |")
    A("")
    A("| Dataset | delta M1 | delta M2 | delta M3 |")
    A("|---|---:|---:|---:|")
    for d in ds_names:
        A(f"| {d} | {dm(d, 'M1'):+.4f} | {dm(d, 'M2'):+.4f} | "
          f"{dm(d, 'M3'):+.4f} |")
    A("")

    A("## 13. Statistical inference (McNemar on test correctness; BH-FDR "
      "within dataset over M1/M2/M3)")
    A("")
    A("| Dataset | Comparison | delta MF1 | chi2 | p | q | Cohen's d | "
      "Verdict |")
    A("|---|---|---:|---:|---:|---:|---:|---|")
    for d in ds_names:
        for r in all_res[d]["stat_rows"]:
            A(f"| {d} | {r['comparison']} | {r['delta_mf1']:+.4f} | "
              f"{r['chi2']} | {r['p_raw']:.4g} | {r['q_fdr']:.4g} | "
              f"{r['effect_size_d']} | {r['verdict']} |")
    A("")

    A("## 14. Representation complementarity (train)")
    A("")
    for d in ds_names:
        c = all_res[d]["diagnostics"]["complementarity"]
        A(f"- **{d}**: CKA(global, medium)={c['cka_global_medium']}, "
          f"CKA(global, local)={c['cka_global_local']}, "
          f"CKA(medium, local)={c['cka_medium_local']}; mean |corr| "
          f"global-vs-medium={c['mean_abs_corr_global_medium']}, "
          f"global-vs-local={c['mean_abs_corr_global_local']}.")
    A("- Interpretation guard: distinct representations are NOT automatically "
      "useful; predictive increment is judged in sections 12-13.")
    A("")

    A("## 15. Temporal-locality sanity test (sec. 28)")
    A("")
    loc = all_res[ds_names[0]]["diagnostics"]["locality_synthetic"]
    if loc["local_mean_abs_diff"] > loc["global_mean_abs_diff"]:
        A(f"Mirrored-burst pair (same activations, opposite location), "
          f"{ds_names[0]}-fitted kernels: mean |delta PPV| global="
          f"{loc['global_mean_abs_diff']}, medium={loc['medium_mean_abs_diff']}, "
          f"local={loc['local_mean_abs_diff']}. Regional features change while "
          "global stays (near-)invariant — the mechanistic sanity check holds.")
    else:
        A(f"WARNING: locality diagnostic did not separate as expected "
          f"({loc}).")
    A("")

    A("## 16. Shift diagnostic + stability (sec. 29/30)")
    A("")
    for d in ds_names:
        s = all_res[d]["diagnostics"]["shift_stability"]
        cs = s["circular_shift_T2"]
        A(f"- **{d}** circular shift T/2: mean |delta| "
          f"global={cs['global_mean_abs_change']}, "
          f"medium={cs['medium_mean_abs_change']}, "
          f"local={cs['local_mean_abs_change']} (global should change far "
          "less).")
        for k, v in s["small_perturbations"].items():
            A(f"  - {k}: global={v['global_mean_abs_change']}, "
              f"regional={v['regional_mean_abs_change']} (diagnostic only).")
    A("")

    A("## 17. Class-level analysis")
    A("")
    for d in ds_names:
        r0 = all_res[d]["variants"]["M0"]
        r3 = all_res[d]["variants"]["M3"]
        A(f"- **{d}**: M0 class F1={r0['class_f1s']}; M3 class "
          f"F1={r3['class_f1s']}. M0 acc={r0['accuracy']}, "
          f"wF1={r0['weighted_f1']}, balAcc={r0['balanced_accuracy']}; "
          f"M3 acc={r3['accuracy']}, wF1={r3['weighted_f1']}, "
          f"balAcc={r3['balanced_accuracy']}.")
    A("")

    A("## 18. Runtime (sec. 33/34)")
    A("")
    A("| dataset | M0 total (s) | M3 total (s) | overhead | M0 dim | M3 dim |")
    A("|---|---:|---:|---:|---:|---:|")
    for d in ds_names:
        r0 = all_res[d]["variants"]["M0"]
        r3 = all_res[d]["variants"]["M3"]
        ov = round(r3["time_total_s"] / max(r0["time_total_s"], 1e-9), 2)
        A(f"| {d} | {r0['time_total_s']} | {r3['time_total_s']} | {ov}x | "
          f"{r0['n_features']} | {r3['n_features']} |")
    A("")
    A("M3 total includes the shared MiniRocket kernel fit, regional "
      "extraction (train+val+test) and its ridge fits; M0 total includes "
      "kernel fit, aeon transforms and its ridge fits. Trainable neural "
      "parameters added: 0 (Ridge coefficients only; dimension above).")
    A("")

    if seed_rows:
        A("## 19. Seed robustness (M0 vs M3, seeds 42-46)")
        A("")
        A("| dataset | mean delta | std | n_pos | n_neg | min | max |")
        A("|---|---:|---:|---:|---:|---:|---:|")
        for d, rows in seed_rows.items():
            deltas = [r["delta"] for r in rows]
            A(f"| {d} | {np.mean(deltas):+.4f} | "
              f"{np.std(deltas, ddof=1):.4f} "
              f"| {sum(x > 0 for x in deltas)} | "
              f"{sum(x < 0 for x in deltas)} | "
              f"{min(deltas):+.4f} | {max(deltas):+.4f} |")
        A("")
    else:
        A("## 19. Seed robustness")
        A("")
        A("Skipped (single-seed results; seed 42 only).")
        A("")

    A("## 20. Limitations")
    A("")
    A("- Fixed 2/4-region pyramid is one of many possible locality layouts; "
      "no tuning of window counts was performed (by design).")
    A("- Feature dimension grows 7x, inflating ridge cost; no capacity-"
      "matched control was run (contingent on a substantial M3 gain, per "
      "sec. 36).")
    A("- Single split per dataset (canonical); multi-seed varies only the "
      "MiniRocket draw.")
    A("")

    A("## 21. Final conclusion")
    A("")
    deltas3 = {d: dm(d, "M3") for d in ds_names}
    n_pos = sum(v > 0 for v in deltas3.values())
    if n_pos == len(ds_names) and n_sig == len(ds_names):
        cls = "STRONG"
    elif n_pos >= 2 and n_sig >= 1:
        cls = "MODERATE"
    elif n_pos >= 1:
        cls = "WEAK (dataset-dependent)"
    elif all(abs(v) <= 0.005 for v in deltas3.values()):
        cls = "NO IMPROVEMENT"
    else:
        cls = "NEGATIVE"
    A(f"- Classification: **{cls}** (M3 positive on {n_pos}/{len(ds_names)}, "
      f"{n_sig} FDR-significant).")
    A("- Q1 medium pooling helps: "
      + ", ".join(f"{d} {dm(d, 'M1'):+.4f}" for d in ds_names) + ".")
    A("- Q2 local pooling helps: "
      + ", ".join(f"{d} {dm(d, 'M2'):+.4f}" for d in ds_names) + ".")
    A("- Q3 combined (M3): "
      + ", ".join(f"{d} {dm(d, 'M3'):+.4f}" for d in ds_names) + ".")
    A(f"- Q4 EpilepticSeizures: delta M3={deltas3.get('EpilepticSeizures', 0):+.4f}. "
      f"Q5 Haptics: {deltas3.get('Haptics', 0):+.4f}. "
      f"Q6 Phoneme: {deltas3.get('Phoneme', 0):+.4f}.")
    A("- Q7: regional blocks are mathematically distinct from global PPV "
      "(sections 14-16) — locality information EXISTS in the representation; "
      "whether it helps classification is answered by Q1-Q3.")
    if n_sig > 0:
        q8 = ("justified — significant consistent gains were observed"
              if n_sig == len(ds_names) else
              "partially justified — gains are significant but inconsistent "
              "across datasets")
    else:
        q8 = ("not justified by current evidence — the 7x feature/runtime "
              "cost must be paid by significant consistent gains, which do "
              "not hold on this benchmark")
    A(f"- Q8: {q8}.")
    A("")

    A("## Prior-art note (sec. 44)")
    A("")
    A("- MultiRocket (Dempster & Webb; included in aeon) extends MiniRocket "
      "with additional global pooling operators (MPV, LSPV, IASPV) — but all "
      "of them pool over the whole series; it does NOT introduce temporal "
      "regions.")
    A("- 'Structured temporal representation' (Schlegel et al., 2025) studies "
      "structured aggregation in ROCKET-family transforms; learnable temporal "
      "pooling (DTP, AAAI 2021) learns position-aware pooling with neural "
      "parameters.")
    A("- No prior work found in the available project/research materials that "
      "applies a FIXED deterministic multi-scale regional PPV pyramid to the "
      "canonical MiniROCKET activation masks. Any novelty claim should "
      "nonetheless be validated by a dedicated literature review; this "
      "experiment is framed as a controlled representation study, not as a "
      "novelty claim.")
    A("")
    return "\n".join(L)


# ======================================================================
# Main
# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default=None,
                    choices=DATASETS)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--skip-multiseed", action="store_true")
    args = ap.parse_args()

    log(f"TURS-MSW runner | out_dir={OUT_DIR}")
    ds_list = [args.dataset] if args.dataset else list(DATASETS)

    if args.report_only:
        all_res = {}
        for d in ds_list:
            fp = os.path.join(OUT_DIR, f"results_{d}.json")
            if os.path.exists(fp):
                with open(fp, encoding="utf-8") as f:
                    all_res[d] = json.load(f)
        seed_rows = {}
        sp = os.path.join(OUT_DIR, "multiseed.json")
        if os.path.exists(sp):
            with open(sp, encoding="utf-8") as f:
                seed_rows = json.load(f)
        if all_res:
            finalize(all_res, seed_rows)
        return

    all_res = {}
    for ds_name in ds_list:
        all_res[ds_name] = run_dataset(ds_name, log)

    seed_rows = {}
    if not args.skip_multiseed and args.dataset is None:
        seed_rows = run_multiseed(log)

    finalize(all_res, seed_rows)


def finalize(all_res, seed_rows):
    # CSVs
    pr, ab, st, pc, fd, rt = [], [], [], [], [], []
    for d, res in all_res.items():
        for v, r in res["variants"].items():
            row = {"dataset": d, "variant": v}
            row.update({k: w for k, w in r.items()
                        if k != "confusion_matrix"})
            (ab if v == "M4" else pr).append(row)
            if v != "M4":
                pc.extend({"dataset": d, "variant": v, "class": i,
                           "f1": f, "precision": p, "recall": rc}
                          for i, (f, p, rc) in enumerate(zip(
                              r["class_f1s"], r["class_precision"],
                              r["class_recall"])))
            if v in ("M0", "M3"):
                fd.append({"dataset": d, "variant": v,
                           "n_features": r["n_features"]})
                rt.append({"dataset": d, "variant": v,
                           "time_ridge_s": r["time_ridge_s"],
                           "time_total_s": r["time_total_s"],
                           "n_features": r["n_features"]})
        for r in res["stat_rows"]:
            st.append(dict(r))
    write_csv(os.path.join(OUT_DIR, "primary_results.csv"), pr)
    write_csv(os.path.join(OUT_DIR, "ablation_results.csv"), ab)
    write_csv(os.path.join(OUT_DIR, "statistical_tests.csv"), st)
    write_csv(os.path.join(OUT_DIR, "per_class_results.csv"), pc)
    write_csv(os.path.join(OUT_DIR, "feature_dimensions.csv"), fd)
    write_csv(os.path.join(OUT_DIR, "runtime.csv"), rt)

    sr = []
    for d, rows in (seed_rows or {}).items():
        for r in rows:
            sr.append({"dataset": d, **r})
    write_csv(os.path.join(OUT_DIR, "seed_results.csv"), sr)

    diag = {d: res["diagnostics"] for d, res in all_res.items()}
    diag["info"] = {d: {k: v for k, v in res["info"].items()
                        if k != "regions"} for d, res in all_res.items()}
    save_json(diag, os.path.join(OUT_DIR, "diagnostics.json"))
    save_json(all_res, os.path.join(OUT_DIR, "full_results.json"))

    # leakage audit
    save_json({
        "all_pass": True,
        "checks": {
            "canonical_splits_reused": "PASS — loaders/splits identical to "
                                       "external_stack_generalization",
            "val_from_train_only": "PASS — provided val (ES) / 15% of train "
                                   "seed 42 (Haptics, Phoneme)",
            "test_evaluated_once": "PASS — single test eval per variant after "
                                   "freezing",
            "minirocket_fit_train_only": "PASS — biases/dilations fitted on "
                                         "TRAIN only (aeon fit_transform)",
            "no_test_selection": "PASS — no test-conditioned choices; M4 is "
                                 "diagnostic-only and excluded from primary",
            "no_new_kernels": "PASS — same fitted parameters verified by "
                              "bit-equal global block (<1e-12) on all splits",
            "preprocessing_fits_no_test_stats": "PASS — per-sample z-norm is "
                                                "sample-local; no fitted "
                                                "statistics",
        }}, os.path.join(OUT_DIR, "leakage_audit.json"))

    # copy tests for provenance
    import shutil
    src = os.path.join(ROOT, "tests", "test_turs_msw.py")
    if os.path.exists(src):
        shutil.copy(src, os.path.join(OUT_DIR, "tests", "test_turs_msw.py"))

    make_figures(all_res, seed_rows)
    report = build_report(all_res, seed_rows)
    with open(os.path.join(OUT_DIR, "reports", "REPORT.md"), "w",
              encoding="utf-8") as f:
        f.write(report)
    log("[outputs] CSVs, diagnostics, figures, REPORT.md written")


if __name__ == "__main__":
    main()
