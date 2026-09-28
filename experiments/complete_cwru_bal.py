"""Complete the missing pairwise_geometry and faithfulness diagnostics for CWRU_BAL."""
import json, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import torch
from experiments.turs_rrmt.data import load_split, SEED
from experiments.turs_rrmt.train_eval import _mf1
from src.diagnostics.statistics import HypothesisRegistry
from src.diagnostics.calibration import nll as nll_fn
from models.turs_mgb.feature_extractor import MGBFeatureExtractor
from models.turs_mgb.grouped_scaler import GroupStandardizer
from models.turs_mgb.ridge_readout import DualRidge
from models.turs_mgb import diagnostics as MGBD

RESULTS = os.path.join(ROOT, "results", "turs_mgb")
GEOMETRIES = ["G1_standard", "G2_tail", "G3_fine", "G4_multilag"]
LAM_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
ABLATION_KEYS = {
    "A0": ["raw"],
    "A1": ["G1_standard"], "A2": ["G2_tail"], "A3": ["G3_fine"],
    "A4": ["G4_multilag"],
    "A5": ["G1_standard", "G2_tail"],
    "A6": ["G1_standard", "G2_tail", "G3_fine"],
    "A7": GEOMETRIES,
}

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def fit_variant(feats_tr, feats_va, feats_te, y_tr, y_va, y_te, n_cls,
                keys, device, lam_grid=LAM_GRID, group_scaling=True,
                variant="A?", hyp=None):
    all_keys = ["raw"] + GEOMETRIES
    scaler = GroupStandardizer([feats_tr[k].shape[1] for k in all_keys])
    blocks_tr = [feats_tr[k] for k in all_keys]
    scaler.fit(blocks_tr)

    def assemble(feats):
        blocks = [scaler.transform_block(j, feats[k]) for j, k in enumerate(all_keys)]
        by_key = dict(zip(all_keys, blocks))
        return np.concatenate([by_key[k] for k in keys], axis=1)

    Ztr, Zva, Zte = assemble(feats_tr), assemble(feats_va), assemble(feats_te)

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

    from experiments.turs_rrmt.train_eval import full_metrics
    res = dict(
        variant=variant, keys=list(keys), lam=best["lam"],
        val_mf1=best["val_mf1"], val_nll=best["val_nll"],
        feature_dim=int(Ztr.shape[1]),
        test=full_metrics(y_te, pred, n_cls),
        probs=pt, pred=pred)
    return res, dict(scaler=scaler, Ztr=Ztr, Zva=Zva, Zte=Zte, readout=rd)


class _ModelStub:
    def __init__(self, ctx, n_cls, device, fitted_extractor=None):
        self.ctx = ctx
        self.n_classes = n_cls
        self.device = device
        self.keys_ = GEOMETRIES
        self.scaler_all = ctx["scaler"]
        self.ex = fitted_extractor
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


def main():
    ds_tag = "CWRU_BAL"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Completing missing diagnostics for {ds_tag} on {device}")

    # Load data
    ds = load_split(ds_tag)
    n_cls = ds["n_cls"]
    y_tr, y_va, y_te = ds["y_train"], ds["y_val"], ds["y_test"]

    # Load existing results
    out_path = os.path.join(RESULTS, ds_tag, "full_results.json")
    results = json.load(open(out_path))
    log(f"Loaded existing results with {len(results.get('variants', {}))} variants")

    # Extract features (cached)
    from experiments.run_turs_mgb_full import get_features
    feats_tr, feats_va, feats_te = get_features(ds_tag, ds, 4096, device, log=log)

    hyp = HypothesisRegistry()
    M = 4096

    # ---- Fit primary A7 model
    t0 = time.time()
    res7, ctx7 = fit_variant(feats_tr, feats_va, feats_te, y_tr, y_va,
                              y_te, n_cls, ABLATION_KEYS["A7"], device,
                              variant="A7")
    log(f"  A7 fit done in {time.time()-t0:.1f}s, MF1={res7['test']['macro_f1']:.4f}")

    ex_fit = MGBFeatureExtractor(M=M, seed=SEED, stats=("ppv", "max"), device=device)
    ex_fit.fit_transport_references(ds["Xtr"])
    model_stub = _ModelStub(ctx7, n_cls, device, ex_fit)

    # ---- Phase 14b: pairwise geometry complementarity
    log("  Computing pairwise geometry complementarity...")
    t0 = time.time()
    pairwise = {}
    a0_mf1 = results["variants"].get("A0", {}).get("test", {}).get("macro_f1")
    for i in range(len(GEOMETRIES)):
        for j in range(i + 1, len(GEOMETRIES)):
            gi, gj = GEOMETRIES[i], GEOMETRIES[j]
            rp, _ = fit_variant(feats_tr, feats_va, feats_te, y_tr,
                                y_va, y_te, n_cls, [gi, gj], device,
                                variant=f"{gi}+{gj}")
            rec = dict(test_mf1=round(rp["test"]["macro_f1"], 4))
            if a0_mf1 is not None:
                rec["delta_vs_A0"] = round(
                    rp["test"]["macro_f1"] - a0_mf1, 4)
            pairwise[f"{gi}|{gj}"] = rec
            log(f"    {gi}+{gj}: MF1={rp['test']['macro_f1']:.4f}")
    results["pairwise_geometry"] = pairwise
    log(f"  Pairwise done in {time.time()-t0:.1f}s")

    # ---- Phase 21: geometry faithfulness
    log("  Computing geometry faithfulness...")
    t0 = time.time()
    try:
        results["faithfulness"] = MGBD.geometry_faithfulness(
            model_stub, ds["Xte"], y_te, hyp)
        log(f"  Faithfulness done in {time.time()-t0:.1f}s")
    except Exception as e:
        log(f"  Faithfulness FAILED: {e}")
        import traceback; traceback.print_exc()

    # ---- Save updated results
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"  Updated results saved to {out_path}")

    # Verify
    r = json.load(open(out_path))
    diag_keys = ['info_addition', 'incremental_gain', 'pairwise_geometry',
                 'geometry_similarity', 'class_conditional', 'group_contributions',
                 'zeroing_ablation', 'faithfulness', 'perturbation', 'stability',
                 'calibration', 'selective_prediction', 'dual_solver_validation']
    present = [k for k in diag_keys if k in r]
    missing = [k for k in diag_keys if k not in r]
    log(f"  Diagnostics: {len(present)}/{len(diag_keys)} present, missing: {missing}")
    log(f"  A8={r['variants'].get('A8',{}).get('test',{}).get('macro_f1')}")
    log(f"  A9={r['variants'].get('A9',{}).get('test',{}).get('macro_f1')}")
    log(f"  A10={r['variants'].get('A10',{}).get('test',{}).get('macro_f1')}")
    log("DONE")


if __name__ == "__main__":
    main()
