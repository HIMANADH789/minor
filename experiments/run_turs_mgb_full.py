"""TURS-MGB — Multi-Geometry Bank — master automated pipeline.

Phases (spec): 0 audit | 1 protocol | 2-5 views+features | 6 scaling |
7-8 dual ridge + lambda | 9-11 baselines + primary | 12 ablation ladder |
13-26 diagnostics | 27-29 comparisons + solver validation | 30-31 caching |
36-38 tables/figures/report.

Usage:
  python experiments/run_turs_mgb_full.py --all
  python experiments/run_turs_mgb_full.py --dataset ECG5000_UNBAL
  python experiments/run_turs_mgb_full.py --variant A7
  ... --resume --force --features-only --train-only --evaluate-only \
      --diagnostics-only --report-only --plots-only
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, ROOT + "/src")

from experiments.turs_rrmt.data import load_split, DATASETS, SEED
from experiments.turs_rrmt.train_eval import full_metrics, _mf1
from src.diagnostics import statistics as S
from src.diagnostics.statistics import HypothesisRegistry, dump_json, dump_csv
from src.diagnostics.calibration import nll as nll_fn

from models.turs_mgb.feature_extractor import MGBFeatureExtractor
from models.turs_mgb.grouped_scaler import GroupStandardizer
from models.turs_mgb.ridge_readout import DualRidge, dual_primal_check
from models.turs_mgb import diagnostics as MGBD

RESULTS = os.path.join(ROOT, "results", "turs_mgb")
TABLE_DIR = os.path.join(RESULTS, "tables")
FIG_DIR = os.path.join(RESULTS, "figures")
REPORT_DIR = os.path.join(RESULTS, "reports")

GEOMETRIES = ["G1_standard", "G2_tail", "G3_fine", "G4_multilag"]
LAM_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
ABLATION_KEYS = {
    "A0": ["raw"],
    "A1": ["G1_standard"], "A2": ["G2_tail"], "A3": ["G3_fine"],
    "A4": ["G4_multilag"],
    "A5": ["G1_standard", "G2_tail"],
    "A6": ["G1_standard", "G2_tail", "G3_fine"],
    "A7": GEOMETRIES,                       # PRIMARY MGB
}
# Phase 12 probes: A8 = A7 without group scaling, A9/A10 = bank-size probes.
DEFAULT_VARIANTS = list(ABLATION_KEYS) + ["A8", "A9", "A10"]
S.BOOTSTRAP_SEED = 9200
S.PERMUTATION_SEED = 9300
S.RNG_POOL = {}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =========================================================== feature caching
def _cache_paths(ds_tag, M):
    d = os.path.join(RESULTS, ds_tag, "features")
    os.makedirs(d, exist_ok=True)
    return (os.path.join(d, f"feats_train_M{M}.npz"),
            os.path.join(d, f"feats_val_M{M}.npz"),
            os.path.join(d, f"feats_test_M{M}.npz"))


def _save_feats(path, feats):
    flat = {k: v for k, v in feats.items() if isinstance(v, np.ndarray)}
    meta = {k: v for k, v in feats.items() if not isinstance(v, np.ndarray)}
    np.savez_compressed(path, **flat,
                        _meta=np.array(json.dumps(meta), dtype=object))


def _load_feats(path):
    z = np.load(path, allow_pickle=True)
    feats = {k: z[k] for k in z.files if k != "_meta"}
    feats.update(json.loads(str(z["_meta"])))
    return feats


def get_features(ds_tag, ds, M, device, force=False, log=log):
    """Phase 2-5 + 30: extract/cache features for all splits at bank size M."""
    ctr, cva, cte = _cache_paths(ds_tag, M)
    if all(os.path.exists(p) for p in (ctr, cva, cte)) and not force:
        log(f"  [{ds_tag}] features cached (M={M})")
        return _load_feats(ctr), _load_feats(cva), _load_feats(cte)

    ex = MGBFeatureExtractor(M=M, seed=SEED, stats=("ppv", "max"),
                             device=device)
    ex.fit_transport_references(ds["Xtr"])
    window = max(3, round(0.05 * ds["L"]))
    out = []
    t0 = time.time()
    for split, X in (("train", ds["Xtr"]), ("val", ds["Xva"]), ("test", ds["Xte"])):
        f = ex.extract_features(X, window=window)
        f["spec"] = ex.spec()
        out.append(f)
        log(f"  [{ds_tag}/{split}] features M={M}: dim={f['feature_dim']} "
            f"({time.time()-t0:.0f}s cum)")
    _save_feats(ctr, out[0]); _save_feats(cva, out[1]); _save_feats(cte, out[2])
    return out[0], out[1], out[2]


# =========================================================== variant fitting
def fit_variant(feats_tr, feats_va, feats_te, y_tr, y_va, y_te, n_cls,
                keys, device, lam_grid=LAM_GRID, group_scaling=True,
                variant="A?", hyp=None, save_dir=None):
    """Phase 6-8: group scaling (optional), dual ridge, lambda selection."""
    all_keys = ["raw"] + GEOMETRIES
    scaler = GroupStandardizer([feats_tr[k].shape[1] for k in all_keys])
    blocks_tr = [feats_tr[k] for k in all_keys]
    if group_scaling:
        scaler.fit(blocks_tr)
    else:
        # A8: identity scaler (fit on zeros/ones so transform is a no-op)
        scaler.fit([np.zeros_like(b) for b in blocks_tr])
        for j in range(len(blocks_tr)):
            scaler.mu[j] = np.zeros_like(scaler.mu[j])
            scaler.sd[j] = np.ones_like(scaler.sd[j])

    def assemble(feats):
        blocks = [scaler.transform_block(j, feats[k]) for j, k in enumerate(all_keys)]
        by_key = dict(zip(all_keys, blocks))
        return np.concatenate([by_key[k] for k in keys], axis=1)

    Ztr, Zva, Zte = assemble(feats_tr), assemble(feats_va), assemble(feats_te)

    # lambda selection on VAL (MF1 primary, NLL tie-break)
    best = dict(lam=None, val_mf1=-1, val_nll=float("inf"))
    for lam in lam_grid:
        rd = DualRidge(n_cls, lam=lam, device=device)
        rd.fit(Ztr, y_tr)
        pv = rd.predict_proba(Zva)
        m = _mf1(y_va, pv.argmax(1), n_cls)
        nl = float(nll_fn(pv, y_va))
        if (m > best["val_mf1"]) or (m == best["val_mf1"] and nl < best["val_nll"]):
            best = dict(lam=lam, val_mf1=round(m, 4), val_nll=round(nl, 4))
    rd = DualRidge(n_cls, lam=best["lam"], device=device)
    rd.fit(Ztr, y_tr)
    pt = rd.predict_proba(Zte)
    pred = pt.argmax(1)
    res = dict(
        variant=variant, keys=list(keys), lam=best["lam"],
        val_mf1=best["val_mf1"], val_nll=best["val_nll"],
        feature_dim=int(Ztr.shape[1]),
        test=full_metrics(y_te, pred, n_cls),
        test_nll=round(float(nll_fn(pt, y_te)), 4),
        test_ece=round(float(__import__("src.diagnostics.calibration",
                                        fromlist=["ece"]).ece(pt, y_te)), 4),
        probs=pt, pred=pred)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        np.savez_compressed(os.path.join(save_dir, f"{variant}_preds.npz"),
                            probs=pt, pred=pred)
        rd.save(os.path.join(save_dir, f"{variant}_ridge.npz"))
    return res, dict(scaler=scaler, Ztr=Ztr, Zva=Zva, Zte=Zte, readout=rd)


# ============================================================ per-dataset run
def run_dataset(ds_tag, device, force=False, variants=None, M=4096,
                hyp=None, log=log):
    ds = load_split(ds_tag)
    n_cls = ds["n_cls"]
    y_tr, y_va, y_te = ds["y_train"], ds["y_val"], ds["y_test"]
    ds_dir = os.path.join(RESULTS, ds_tag)
    out_path = os.path.join(ds_dir, "full_results.json")
    cached = None
    if os.path.exists(out_path) and not force:
        try:
            cached = json.load(open(out_path))
        except (OSError, ValueError):
            cached = None
    requested = variants or DEFAULT_VARIANTS
    if cached is not None:
        missing = [v for v in requested if v not in cached.get("variants", {})]
        diag_missing = not ("pairwise_geometry" in cached and "faithfulness" in cached)
        if not missing and not diag_missing:
            log(f"  [{ds_tag}] full results cached "
                f"({len(cached.get('variants', {}))} variants)")
            return cached
        variants = missing
        if diag_missing and "A7" not in variants:
            variants = variants + ["A7"]  # refit A7 to power diagnostics
        log(f"  [{ds_tag}] resume: refitting variants {variants}")

    os.makedirs(ds_dir, exist_ok=True)
    feats_tr, feats_va, feats_te = get_features(ds_tag, ds, M, device,
                                                force=force, log=log)

    variants = variants or DEFAULT_VARIANTS
    results = dict(cached) if cached else dict(dataset=ds_tag, M=M, timing={})
    results.setdefault("dataset", ds_tag)
    results.setdefault("variants", {})
    results.setdefault("timing", {})
    results["M"] = M

    # A9 fallback bank size (Phase 12): smaller bank, fresh features
    extra_banks = {}
    if "A9" in variants:
        f9tr, f9va, f9te = get_features(ds_tag, ds, 2048, device,
                                        force=force, log=log)
        extra_banks[2048] = (f9tr, f9va, f9te)
    if "A10" in variants:
        f10tr, f10va, f10te = get_features(ds_tag, ds, 8192, device,
                                           force=force, log=log)
        extra_banks[8192] = (f10tr, f10va, f10te)

    for v in variants:
        t0 = time.time()
        cfg = list(ABLATION_KEYS.get(v, GEOMETRIES))
        if v == "A8":
            cfg = list(ABLATION_KEYS["A7"])
        use_M = M
        if v == "A9":
            use_M = 2048
        elif v == "A10":
            use_M = 8192
        ftr, fva, fte = (extra_banks[use_M] if use_M != M
                         else (feats_tr, feats_va, feats_te))
        group_scaling = not (v == "A8")
        res, ctx = fit_variant(ftr, fva, fte, y_tr, y_va, y_te, n_cls,
                               cfg, device, group_scaling=group_scaling,
                               variant=v, hyp=hyp,
                               save_dir=os.path.join(ds_dir, "predictions"))
        res["elapsed_s"] = round(time.time() - t0, 1)
        res.pop("probs", None), res.pop("pred", None)
        results["variants"][v] = res
        log(f"  [{ds_tag}/{v}] test MF1={res['test']['macro_f1']:.4f} "
            f"(lam={res['lam']}, d={res['feature_dim']}, {res['elapsed_s']}s)")

    # ---- Phase 29: dual/primal validation (small-d proxy from A0 block)
    sub = slice(0, 400)
    Z0_tr = feats_tr["raw"][sub][:, :512]
    dp = dual_primal_check(Z0_tr, y_tr[sub], 1.0, feats_te["raw"][:100][:, :512])
    results["dual_solver_validation"] = dp
    log(f"  [{ds_tag}] dual/primal max logit diff = {dp['max_abs_logit_diff']:.2e} "
        f"({'PASS' if dp['passed'] else 'FAIL'})")

    # ---- keep the primary model object for diagnostics (skip on partial
    # resume when every diagnostic block is already cached)
    need_diag = not (cached is not None and "info_addition" in cached
                     and "pairwise_geometry" in cached
                     and "faithfulness" in cached)
    primary = None
    if "A7" in results["variants"] and need_diag:
        res7, ctx7 = fit_variant(feats_tr, feats_va, feats_te, y_tr, y_va,
                                 y_te, n_cls, ABLATION_KEYS["A7"], device,
                                 variant="A7", hyp=hyp)
        primary = dict(res=res7, ctx=ctx7)

    # ---- Phase 13/25: information addition (A0 vs A5/A6/A7)
    if primary is not None:
        probs_models = {}
        for v in ("A0", "A5", "A6", "A7"):
            if v in results["variants"]:
                # recompute probs for the comparison (cheap: cached readouts)
                cfg = ABLATION_KEYS.get(v, GEOMETRIES)
                r, _ = fit_variant(feats_tr, feats_va, feats_te, y_tr, y_va,
                                   y_te, n_cls, cfg, device, variant=v + "_re",
                                   hyp=None)
                probs_models[v] = r["probs"]
        results["info_addition"] = MGBD.info_addition(y_te, probs_models,
                                                      hyp or HypothesisRegistry())

        # ---- Phase 14: incremental geometry gain (leave-one-out)
        loo = {}
        full_cfg = ABLATION_KEYS["A7"]
        r_full, _ = fit_variant(feats_tr, feats_va, feats_te, y_tr, y_va,
                                y_te, n_cls, full_cfg, device, variant="ALL_re",
                                hyp=None)
        loo["ALL"] = r_full["probs"]
        for k in GEOMETRIES:
            cfg = [kk for kk in full_cfg if kk != k]
            r, _ = fit_variant(feats_tr, feats_va, feats_te, y_tr, y_va,
                               y_te, n_cls, cfg, device, variant=f"no_{k}",
                               hyp=None)
            loo[f"minus_{k}"] = r["probs"]
        results["incremental_gain"] = MGBD.incremental_gain(
            y_te, y_va, loo, hyp or HypothesisRegistry())

        # ---- Phase 14b: pairwise geometry complementarity (table 03)
        pairwise = {}
        a0_mf1 = results["variants"].get("A0", {}).get("test", {})\
            .get("macro_f1")
        for i in range(len(GEOMETRIES)):
            for j in range(i + 1, len(GEOMETRIES)):
                gi, gj = GEOMETRIES[i], GEOMETRIES[j]
                rp, _ = fit_variant(feats_tr, feats_va, feats_te, y_tr,
                                    y_va, y_te, n_cls, [gi, gj], device,
                                    variant=f"{gi}+{gj}", hyp=None)
                rec = dict(test_mf1=round(rp["test"]["macro_f1"], 4))
                if a0_mf1 is not None:
                    rec["delta_vs_A0"] = round(
                        rp["test"]["macro_f1"] - a0_mf1, 4)
                pairwise[f"{gi}|{gj}"] = rec
        results["pairwise_geometry"] = pairwise

        # ---- Phase 15: geometry similarity
        blocks = [feats_te[k] for k in GEOMETRIES]
        results["geometry_similarity"] = MGBD.geometry_similarity(
            blocks, GEOMETRIES)

        # ---- Phase 16: class-conditional (preds reloaded from saved npz;
        # the in-memory copies were dropped after dump)
        preds = {}
        for v in results["variants"]:
            pp = os.path.join(ds_dir, "predictions", f"{v}_preds.npz")
            if os.path.exists(pp):
                preds[v] = np.load(pp)["pred"]
        results["class_conditional"] = MGBD.class_conditional(
            y_te, preds, n_cls)

        # ---- Phase 17: group contributions (exact, linear readout)
        # a fitted extractor is needed for on-the-fly view/feature extraction
        ex_fit = MGBFeatureExtractor(M=M, seed=SEED, stats=("ppv", "max"),
                                     device=device)
        ex_fit.fit_transport_references(ds["Xtr"])
        model_stub = _ModelStub(primary["ctx"], n_cls, device, ex_fit)
        results["group_contributions"] = MGBD.group_contributions(
            model_stub, feats_te, y_te, hyp or HypothesisRegistry())

        # ---- Phase 18: zeroing ablation
        results["zeroing_ablation"] = MGBD.zeroing_ablation(
            model_stub, feats_te, y_te, hyp or HypothesisRegistry())

        # ---- Phase 21: geometry faithfulness (targeted vs random masking)
        results["faithfulness"] = MGBD.geometry_faithfulness(
            model_stub, ds["Xte"], y_te, hyp or HypothesisRegistry())

        # ---- Phase 20: perturbation response
        results["perturbation"] = MGBD.perturbation_response(
            model_stub, ds["Xte"], None, y_te)

        # ---- Phase 22: stability
        results["stability"] = MGBD.stability(model_stub, ds["Xte"], y_te)

        # ---- Phase 23/24: calibration + selective prediction
        probs_for_cal = {}
        for v in ("A0", "A1", "A2", "A3", "A4", "A7"):
            if v in results["variants"]:
                cfg = ABLATION_KEYS.get(v, GEOMETRIES)
                r, _ = fit_variant(feats_tr, feats_va, feats_te, y_tr, y_va,
                                   y_te, n_cls, cfg, device, variant=v + "_re",
                                   hyp=None)
                probs_for_cal[v] = r["probs"]
        results["calibration"] = MGBD.calibration_table(probs_for_cal, y_te)
        results["selective_prediction"] = MGBD.selective_prediction(
            probs_for_cal, y_te)

    dump_json(results, out_path)
    return results


class _ModelStub:
    """Minimal adapter exposing group_logits/predict_split/keys_/n_classes/
    ex/scaler_all over a fitted variant context (avoids re-fitting)."""

    def __init__(self, ctx, n_cls, device, fitted_extractor=None):
        self.ctx = ctx
        self.n_classes = n_cls
        self.device = device
        self.keys_ = GEOMETRIES
        self.scaler_all = ctx["scaler"]
        from models.turs_mgb.feature_extractor import MGBFeatureExtractor
        self.ex = fitted_extractor or MGBFeatureExtractor(
            M=4096, seed=SEED, stats=("ppv", "max"), device=device)
        # readout: reuse fitted ridge on the A7 assembly
        self._rd = ctx["readout"]

    def transform_split(self, feats):
        all_keys = ["raw"] + GEOMETRIES
        blocks = [self.scaler_all.transform_block(j, feats[k])
                  for j, k in enumerate(all_keys)]
        by_key = dict(zip(all_keys, blocks))
        return np.concatenate([by_key[k] for k in self.keys_], axis=1)

    def predict_split(self, feats):
        return self._rd.predict_proba(self.transform_split(feats))

    def group_logits(self, feats):
        Z = self.transform_split(feats)
        offs = np.concatenate([[0], np.cumsum(
            [feats[k].shape[1] for k in self.keys_])])
        return self._rd.group_logits(Z, offs)


# ================================================================ reporting
def write_tables(all_results, tags):
    os.makedirs(TABLE_DIR, exist_ok=True)
    # 01 model comparison (with historical baselines)
    baseline = {
        "MiniROCKET": {"ECG5000_UNBAL": 0.5938, "ECG5000_BAL": 0.6553,
                       "CWRU_UNBAL": 0.9917, "CWRU_BAL": 0.9947},
        "TURS-Lite": {"ECG5000_UNBAL": 0.6046, "ECG5000_BAL": 0.6377,
                      "CWRU_UNBAL": 0.8997, "CWRU_BAL": 0.9417},
        "TURS-Stack(best)": {"ECG5000_UNBAL": 0.629, "ECG5000_BAL": 0.673,
                             "CWRU_UNBAL": 0.958, "CWRU_BAL": 0.988},
        "TURS-GLR A8": {"ECG5000_UNBAL": 0.656, "ECG5000_BAL": 0.658,
                        "CWRU_UNBAL": 0.967, "CWRU_BAL": 0.984},
    }
    rows = []
    models = ["A0", "A7", "A9", "A10"]
    for ds in tags:
        r = all_results.get(ds, {})
        row = dict(dataset=ds)
        for m, bv in baseline.items():
            row[m] = bv.get(ds)
        for v in models:
            row[f"MGB_{v}"] = round(r.get("variants", {}).get(v, {})
                                    .get("test", {}).get("macro_f1", float("nan")), 4)
        rows.append(row)
    dump_csv(rows, os.path.join(TABLE_DIR, "01_model_comparison.csv"))

    # 02 geometry ablation
    rows = []
    for ds in tags:
        r = all_results.get(ds, {}).get("variants", {})
        for v, rec in r.items():
            rows.append(dict(dataset=ds, variant=v,
                             keys=";".join(rec.get("keys", [])),
                             lam=rec.get("lam"),
                             val_mf1=rec.get("val_mf1"),
                             test_mf1=round(rec["test"]["macro_f1"], 4),
                             test_acc=round(rec["test"]["accuracy"], 4),
                             feature_dim=rec.get("feature_dim"),
                             elapsed_s=rec.get("elapsed_s")))
    dump_csv(rows, os.path.join(TABLE_DIR, "02_geometry_ablation.csv"))

    # 03 pairwise geometry complementarity
    rows = []
    for ds in tags:
        pw = all_results.get(ds, {}).get("pairwise_geometry", {})
        for pair, rec in pw.items():
            rows.append(dict(dataset=ds, pair=pair, **rec))
    dump_csv(rows, os.path.join(TABLE_DIR, "03_geometry_pairwise.csv"))

    # 04 incremental geometry gain
    rows = []
    for ds in tags:
        ig = all_results.get(ds, {}).get("incremental_gain", {})
        for k, rec in ig.items():
            if isinstance(rec, dict):
                rows.append(dict(dataset=ds, geometry=k, **rec))
    dump_csv(rows, os.path.join(TABLE_DIR, "04_incremental_geometry_gain.csv"))

    # 05 group contribution
    rows = []
    for ds in tags:
        gc = all_results.get(ds, {}).get("group_contributions", {})
        for k, rec in gc.get("per_group", {}).items():
            rows.append(dict(dataset=ds, geometry=k, **rec))
    dump_csv(rows, os.path.join(TABLE_DIR, "05_group_contribution.csv"))

    # 06 feature redundancy
    rows = []
    for ds in tags:
        sim = all_results.get(ds, {}).get("geometry_similarity", {})
        for pair, rec in sim.get("pairs", {}).items():
            rows.append(dict(dataset=ds, pair=pair, **rec))
    dump_csv(rows, os.path.join(TABLE_DIR, "06_feature_redundancy.csv"))

    # 07 class conditioned
    rows = []
    for ds in tags:
        cc = all_results.get(ds, {}).get("class_conditional", {})
        for v, rec in cc.items():
            for ci, f1 in enumerate(rec["per_class_f1"]):
                rows.append(dict(dataset=ds, variant=v, class_idx=ci,
                                 per_class_f1=f1))
    dump_csv(rows, os.path.join(TABLE_DIR, "07_class_conditioned.csv"))

    # 08 calibration
    rows = []
    for ds in tags:
        for v, rec in all_results.get(ds, {}).get("calibration", {}).items():
            rows.append(dict(dataset=ds, variant=v, **rec))
    dump_csv(rows, os.path.join(TABLE_DIR, "08_calibration.csv"))

    # 09 selective prediction
    rows = []
    for ds in tags:
        for v, rec in all_results.get(ds, {}).get("selective_prediction", {}).items():
            for sig, m in rec.items():
                rows.append(dict(dataset=ds, variant=v, signal=sig, **m))
    dump_csv(rows, os.path.join(TABLE_DIR, "09_selective_prediction.csv"))

    # 10 perturbation
    rows = []
    for ds in tags:
        pert = all_results.get(ds, {}).get("perturbation", {}).get("kinds", {})
        for kind, rec in pert.items():
            for lv, m in rec["levels"].items():
                rows.append(dict(dataset=ds, kind=kind, level=lv, **m))
    dump_csv(rows, os.path.join(TABLE_DIR, "10_perturbation.csv"))

    # 12 stability
    rows = []
    for ds in tags:
        st = all_results.get(ds, {}).get("stability", {}).get("specs", {})
        for name, rec in st.items():
            for k, cosv in rec.get("per_geometry", {}).items():
                rows.append(dict(dataset=ds, spec=name, geometry=k,
                                 cosine=cosv,
                                 pred_agreement=rec.get("pred_agreement")))
    dump_csv(rows, os.path.join(TABLE_DIR, "12_stability.csv"))

    # 11 faithfulness (targeted vs random geometry-field masking)
    rows = []
    for ds in tags:
        fa = all_results.get(ds, {}).get("faithfulness", {})
        for k, rec in fa.items():
            rows.append(dict(dataset=ds, geometry=k, **rec))
    dump_csv(rows, os.path.join(TABLE_DIR, "11_faithfulness.csv"))

    # 13 significance (from hypothesis registry, written by caller)
    # 15 complexity
    rows = []
    for ds in tags:
        for v, rec in all_results.get(ds, {}).get("variants", {}).items():
            rows.append(dict(dataset=ds, variant=v,
                             feature_dim=rec.get("feature_dim"),
                             trainable_params=0,
                             fixed_feature_dim=rec.get("feature_dim"),
                             elapsed_s=rec.get("elapsed_s")))
    dump_csv(rows, os.path.join(TABLE_DIR, "15_complexity.csv"))

    # 16 dual solver validation
    rows = []
    for ds in tags:
        dp = all_results.get(ds, {}).get("dual_solver_validation", {})
        if dp:
            rows.append(dict(dataset=ds, **dp))
    dump_csv(rows, os.path.join(TABLE_DIR, "16_dual_solver_validation.csv"))


def write_report(all_results, tags, hyp_rows, elapsed_s):
    os.makedirs(REPORT_DIR, exist_ok=True)
    p = os.path.join(REPORT_DIR, "TURS_MGB_FULL_REPORT.md")
    a = []
    a.append("# TURS-MGB: Multi-Geometry Bank — Full Report")
    a.append("")
    a.append(f"*Runtime {elapsed_s/60:.1f} min · seed {SEED} · shared fixed "
             f"kernel bank (0 trainable params) · 4 transport geometries · "
             f"dual linear-kernel Ridge · canonical protocol*")
    a.append("")
    a.append("## 1. Executive Summary")
    a.append("")
    for ds in tags:
        r = all_results.get(ds, {}).get("variants", {})
        a0 = r.get("A0", {}).get("test", {}).get("macro_f1")
        a7 = r.get("A7", {}).get("test", {}).get("macro_f1")
        if a0 is None or a7 is None:
            continue
        d = a7 - a0
        verdict = ("MGB ADDS INFORMATION" if d > 0.01 else
                   "MGB ~ MiniROCKET baseline" if d > -0.01 else
                   "MGB DOES NOT ADD INFORMATION (negative result)")
        a.append(f"- **{ds}**: A0={a0:.4f} -> A7={a7:.4f} ({d:+.4f}). {verdict}.")
    a.append("")
    a.append("## 2-4. Motivation / prior limitations / concept")
    a.append("")
    a.append("Transport geometry is treated as a FEATURE-GENERATION AXIS: four "
             "fixed geometries (standard W1, tail-weighted, fine-grid, "
             "multi-lag) are each convolved with the SAME fixed kernel bank; "
             "features are concatenated and one dual Ridge decides. No router, "
             "gate, block lambda, or deep head (those components repeatedly "
             "failed to add value in RRMT/GLR).")
    a.append("")
    a.append("## 13. Ablation ladder (test Macro-F1)")
    a.append("")
    variants = ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10"]
    a.append("| Dataset | " + " | ".join(variants) + " |")
    a.append("|---|" + "---|" * len(variants))
    for ds in tags:
        r = all_results.get(ds, {}).get("variants", {})
        cells = []
        for v in variants:
            m = r.get(v, {}).get("test", {}).get("macro_f1")
            cells.append(_fmt(m))
        a.append(f"| {ds} | " + " | ".join(cells) + " |")
    a.append("")
    a.append("## 14. Incremental geometry gain (leave-one-geometry-out)")
    a.append("")
    a.append("| Dataset | full MF1 | -G1 | -G2 | -G3 | -G4 |")
    a.append("|---|---|---|---|---|---|")
    for ds in tags:
        ig = all_results.get(ds, {}).get("incremental_gain", {})
        if not ig:
            continue
        row = [ig.get("full_mf1", "n/a")]
        for g in GEOMETRIES:
            rec = ig.get(f"minus_{g}", {})
            row.append(rec.get("delta_j", "n/a"))
        a.append(f"| {ds} | " + " | ".join(str(x) for x in row) + " |")
    a.append("")
    a.append("## 14b. Pairwise geometry complementarity (2-geometry banks vs A0)")
    a.append("")
    a.append("| Dataset | G1+G2 | G1+G3 | G1+G4 | G2+G3 | G2+G4 | G3+G4 | A0 |")
    a.append("|---|---|---|---|---|---|---|---|")
    for ds in tags:
        pw = all_results.get(ds, {}).get("pairwise_geometry", {})
        if not pw:
            continue
        r = all_results.get(ds, {}).get("variants", {})
        a0 = r.get("A0", {}).get("test", {}).get("macro_f1")
        cells = [_fmt(pw.get(f"{GEOMETRIES[i]}|{GEOMETRIES[j]}", {})
                      .get("test_mf1"))
                 for i in range(4) for j in range(i + 1, 4)]
        a.append(f"| {ds} | " + " | ".join(cells) + f" | {_fmt(a0)} |")
    a.append("")
    a.append("## 21. Geometry faithfulness (targeted vs random field masking)")
    a.append("")
    a.append("| Dataset | Geometry | target drop | random drop | diff [95% CI] | p |")
    a.append("|---|---|---|---|---|---|")
    for ds in tags:
        fa = all_results.get(ds, {}).get("faithfulness", {})
        for k, rec in fa.items():
            ci = rec.get("ci95", [None, None])
            a.append(f"| {ds} | {k} | {rec.get('target_drop')} | "
                     f"{rec.get('random_drop')} | {rec.get('diff')} "
                     f"[{ci[0]}, {ci[1]}] | {rec.get('p_perm')} |")
    a.append("")
    a.append("## 19. Statistical significance (FDR within families)")
    a.append("")
    sig = [r for r in hyp_rows if r.get("significant")]
    a.append(f"- {len(hyp_rows)} registered hypothesis tests, "
             f"{len(sig)} significant after BH-FDR (alpha=0.05).")
    a.append("- See `tables/13_significance.csv` for the full registry.")
    a.append("")
    a.append("## 26. Limitations")
    a.append("")
    a.append("- Single seed; PPV/max statistics only; linear readout.")
    a.append("- A8 (no group scaling) and A10 (M=8192) probe design choices, "
             "not tuned competitors.")
    a.append("- Negative results are reported as such (see Executive Summary).")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(a) + "\n")
    return p


def _fmt(x, nd=4):
    import numpy as np
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def write_figures(all_results, tags):
    os.makedirs(FIG_DIR, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # 5. geometry ablation + 18. cross-dataset summary combined
    variants = ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A9", "A10"]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    width = 0.8 / len(variants)
    for i, v in enumerate(variants):
        vals = [all_results.get(t, {}).get("variants", {}).get(v, {})
                .get("test", {}).get("macro_f1", 0) for t in tags]
        pos = np.arange(len(tags)) + i * width - 0.4 + width / 2
        ax.bar(pos, vals, width, label=v)
    ax.set_xticks(np.arange(len(tags)))
    ax.set_xticklabels(tags, rotation=15)
    ax.set_ylabel("Test Macro-F1")
    ax.set_title("TURS-MGB: geometry ablation (A7 = full multi-geometry bank)")
    ax.legend(ncol=5, fontsize=8)
    ax.set_ylim(0, 1)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"mgb_ablation.{ext}"), dpi=150,
                    bbox_inches="tight")
    plt.close(fig)
    log(f"  figures -> {os.path.relpath(FIG_DIR, ROOT)}")


# ==================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", default=True)
    ap.add_argument("--dataset", type=str, default=None)
    ap.add_argument("--variant", type=str, default=None)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--features-only", action="store_true")
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--evaluate-only", action="store_true")
    ap.add_argument("--diagnostics-only", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--plots-only", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"TURS-MGB master pipeline. Device: {device}")
    tags = [args.dataset] if args.dataset else list(DATASETS)

    if not (args.report_only or args.plots_only):
        from experiments.turs_mgb.audit import run_audit
        run_audit(log=log)

    hyp = HypothesisRegistry()
    all_results = {}
    if not args.report_only and not args.plots_only:
        for ds in tags:
            log(f"=== DATASET {ds} ===")
            r = run_dataset(ds, device, force=args.force,
                            variants=[args.variant] if args.variant else None,
                            hyp=hyp, log=log)
            all_results[ds] = r
            if args.features_only:
                continue
        if args.features_only:
            log(f"FEATURES-ONLY complete in {(time.time()-t0)/60:.1f} min")
            return

    # reload full results from disk for reporting (cache-safe)
    for ds in tags:
        fp = os.path.join(RESULTS, ds, "full_results.json")
        if os.path.exists(fp) and ds not in all_results:
            all_results[ds] = json.load(open(fp))

    if args.train_only:
        log(f"TRAIN-ONLY complete in {(time.time()-t0)/60:.1f} min")
        return

    log("PHASE: tables / significance / figures / report")
    hyp_rows = hyp.finalize()
    dump_csv(hyp_rows, os.path.join(TABLE_DIR, "13_significance.csv"))
    es_rows = [dict(family=r["family"], test=r["test"],
                    comparison=r["comparison"], estimate=r.get("estimate"),
                    p=r.get("p_value"), q=r.get("q_value"))
               for r in hyp_rows]
    dump_csv(es_rows, os.path.join(TABLE_DIR, "14_effect_sizes.csv"))
    write_tables(all_results, tags)
    if not args.report_only:
        write_figures(all_results, tags)
    if not args.plots_only:
        rep = write_report(all_results, tags, hyp_rows, time.time() - t0)
        log(f"  report -> {os.path.relpath(rep, ROOT)}")
    dump_json(all_results, os.path.join(RESULTS, "all_results.json"))
    log(f"\nCOMPLETE in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
