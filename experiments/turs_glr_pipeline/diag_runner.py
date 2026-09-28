"""TURS-GLR diagnostics runner: block contributions, global-vs-local,
routing diagnostics, robustness, calibration, uncertainty, case studies."""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

from src.diagnostics import statistics as S
from src.diagnostics.calibration import ece, adaptive_ece, brier, nll as nll_fn
from models.turs_glr.feature_blocks import BlockStandardizer, build_local_block
from models.turs_glr.diagnostics import (
    block_contributions, block_contribution_summary, global_local_disagreement,
    routing_descriptor_correlations, routing_summary_stats,
    predict_with_modified_routing, routing_faithfulness, routing_stability,
    counterfactual_routing, uncertainty_analysis, calibration_record,
    degradation_analysis)
from experiments.turs_glr_pipeline.core import (
    RESULTS, CACHE, build_model, log, ARCH, SEED)
from experiments.turs_rrmt.data import load_split


def _Zg_fn(model, scaler):
    """Closure: raw X -> standardized global features (frozen, cacheable)."""
    cache = {}
    def fn(X):
        key = (id(X), X.shape)
        if key not in cache:
            cache[key] = model.extract_global(X)
        Zg = cache[key]
        Zgs, _ = scaler.transform(Zg, np.zeros((len(Zg), 0)))
        return Zgs
    return fn


def run_diagnostics(ds_tag, device, force=False, log=log):
    """Full diagnostic suite for one dataset on the frozen A3 primary model."""
    out_path = os.path.join(RESULTS, ds_tag, "diagnostics.json")
    if os.path.exists(out_path) and not force:
        log(f"  [{ds_tag}] diagnostics cached")
        return json.load(open(out_path))

    ds = load_split(ds_tag)
    n_cls = ds["n_cls"]
    ab = json.load(open(os.path.join(RESULTS, ds_tag, "ablation_results.json")))
    hyp = S.HypothesisRegistry()
    out = {}

    # ---------- rebuild the frozen A3 pipeline ----------
    model = build_model(ds, device)
    feat = np.load(os.path.join(CACHE, ds_tag, "features_A3.npz"))
    scaler = BlockStandardizer(int(feat["Zg_tr"].shape[1]),
                               int(feat["Zl_tr"].shape[1]))
    scaler.fit(feat["Zg_tr"], feat["Zl_tr"])
    Zg_tr, Zl_tr = scaler.transform(feat["Zg_tr"], feat["Zl_tr"])
    ridge = None
    sel = ab["variants"]["A3"]["selection"]
    from models.turs_glr.block_ridge import BlockRidge
    ridge = BlockRidge(n_cls, lam_g=sel["lam_g"], lam_l=sel["lam_l"])
    ridge.fit(Zg_tr, Zl_tr, feat["y_tr"])

    probs_te = np.load(os.path.join(RESULTS, ds_tag, "preds_A3.npz"))["probs_te"]
    y_te = feat["y_te"]

    # ---------- block contributions (exact linear decomposition) ----------
    Zg_te, Zl_te = scaler.transform(feat["Zg_te"], feat["Zl_te"])
    bc = block_contributions(ridge, Zg_te, Zl_te, y_te, probs_te)
    out["block_contributions"] = block_contribution_summary(bc)
    out["block_contributions"]["by_class"] = {}
    for c in range(n_cls):
        m = y_te == c
        if m.any():
            out["block_contributions"]["by_class"][int(c)] = dict(
                global_norm=float(bc["global_logit_norm"][m].mean()),
                local_norm=float(bc["local_logit_norm"][m].mean()),
                n=int(m.sum()))
    out["global_local_disagreement"] = global_local_disagreement(ridge, Zg_te, Zl_te)
    log(f"  [{ds_tag}] block contributions: local share median "
        f"{out['block_contributions']['local_share_median']:.3f}")

    # per-sample contribution decomposition saved for figures
    np.savez_compressed(os.path.join(RESULTS, ds_tag, "block_contrib.npz"),
                        global_norm=bc["global_logit_norm"],
                        local_norm=bc["local_logit_norm"],
                        ratio=bc["ratio"], y=y_te, pred=bc["pred"],
                        correct=bc["correct"],
                        fg=bc["fg"].astype(np.float32),
                        fl=bc["fl"].astype(np.float32))

    # ---------- lambda adaptation record ----------
    out["lambda_adaptation"] = dict(
        lam_g=sel["lam_g"], lam_l=sel["lam_l"],
        coef_norms=ab["variants"]["A3"]["coef_norms"],
        d_global=int(feat["Zg_tr"].shape[1]),
        d_local=int(feat["Zl_tr"].shape[1]),
        interpretation="lam pair is validation-selected; block contribution "
                       "is measured from the fitted solution, not from lambda "
                       "values alone")

    # ---------- routing diagnostics on the test split ----------
    Zg_fn = _Zg_fn(model, scaler)
    with torch.no_grad():
        d = model.forward_diag(torch.from_numpy(ds["Xte"]).float()
                               .to(next(model.parameters()).device))
        w_te = d["w"].cpu().numpy()
        Tm = min(w_te.shape[-1], d["Tv"].shape[-1])
        U_te = (d["w"] * d["Tv"])[..., :Tm].cpu().numpy()
    out["routing_summary"] = routing_summary_stats(w_te)

    # routing weight vs descriptor correlations
    log(f"  [{ds_tag}] routing descriptor correlations")
    out["routing_descriptors"] = routing_descriptor_correlations(
        model, ds["Xte"], w_te, hyp, family="D3_routing_validity")

    # routing interventions (frozen ridge, modified routing)
    log(f"  [{ds_tag}] routing interventions")
    interventions = {}
    for mode in ["uniform", "shuffled", "fixed_0", "fixed_1", "fixed_2", "fixed_3"]:
        p_mod = predict_with_modified_routing(model, ridge, scaler, Zg_fn,
                                              ds["Xte"], y_te, mode, seed=SEED)
        pred = p_mod.argmax(1)
        from experiments.turs_rrmt.train_eval import full_metrics as fm
        rec = fm(y_te, pred, n_cls)
        rec["nll"] = nll_fn(p_mod, y_te)
        rec["mean_conf"] = float(p_mod.max(1).mean())
        rec["flip_rate_vs_actual"] = float((pred != probs_te.argmax(1)).mean())
        interventions[mode] = rec
        if mode in ("uniform", "shuffled"):
            cA = (probs_te.argmax(1) == y_te).astype(float)
            cB = (pred == y_te).astype(float)
            chi2, pm_ = S.mcnemar(cA, cB)
            hyp.add("D4_routing_intervention", "McNemar paired",
                    f"{mode} vs actual routing accuracy",
                    float(cB.mean() - cA.mean()), pm_, {})
    out["routing_intervention"] = interventions

    # faithfulness / stability / counterfactual (hyp is keyword-only here;
    # the 6th positional slot is rng_seed)
    log(f"  [{ds_tag}] routing faithfulness")
    out["routing_faithfulness"] = routing_faithfulness(
        model, ridge, scaler, Zg_fn, ds["Xte"], y_te, hyp=hyp)
    log(f"  [{ds_tag}] routing stability")
    out["routing_stability"] = routing_stability(
        model, ridge, scaler, Zg_fn, ds["Xte"], y_te, hyp=hyp)
    log(f"  [{ds_tag}] counterfactual routing")
    out["counterfactual_routing"] = counterfactual_routing(
        model, ridge, scaler, Zg_fn, ds["Xte"], y_te)

    # ---------- uncertainty / selective prediction ----------
    out["uncertainty"] = uncertainty_analysis(probs_te, y_te, w_te, bc)

    # ---------- calibration ----------
    out["calibration"] = calibration_record(probs_te, y_te)

    # ---------- robustness ----------
    log(f"  [{ds_tag}] degradation robustness")
    out["robustness"] = degradation_analysis(model, ridge, scaler, Zg_fn,
                                             ds["Xte"], y_te, hyp)

    out["_hyp_rows"] = [dict(dataset=ds_tag, **r) for r in hyp.finalize()]
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    return out
