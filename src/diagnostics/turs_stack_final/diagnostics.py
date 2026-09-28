"""Diagnostic suite for the fresh frozen Stack (Phases 10-28).

Reuses the verified experiment implementations from
src/diagnostics/turs_stack/experiments.py and extraction.py (the model
class and internals are identical; only the artifact paths differ)."""
import json
import os

import numpy as np
import torch

from . import config as C
from src.diagnostics import statistics as _S
from src.diagnostics.turs_stack.statistics import HypothesisRegistry
from src.diagnostics import calibration as CAL
from src.diagnostics import perturb as P
from src.diagnostics.turs_stack import extraction as EX
from src.diagnostics.turs_stack import experiments as XP
from src.diagnostics.turs_stack.case_studies import build_case_studies as _build_cases
import src.diagnostics.turs_stack.config as OLDC

# NOTE: rebind AFTER all imports — importing turs_stack.statistics (via
# experiments) overwrites the shared-seed globals with the old study's
# seeds (5200/5300); the fresh study uses its own pre-registered seeds.
_S.BOOTSTRAP_SEED = C.BOOTSTRAP_SEED
_S.PERMUTATION_SEED = C.PERMUTATION_SEED
_S.RNG_POOL = {}

OLDC.SYNTH_N_PER_CLASS = C.SYNTH_N_PER_CLASS
OLDC.SYNTH_REGION_FRAC = C.SYNTH_REGION_FRAC
OLDC.SYNTH_SEED = C.SYNTH_SEED
OLDC.DEG_MAX_SAMPLES = C.DEG_MAX_SAMPLES
OLDC.FAITH_MAX_SAMPLES = C.FAITH_MAX_SAMPLES
OLDC.FAITH_PRIMARY_FRAC = C.FAITH_PRIMARY_FRAC
OLDC.FAITH_PERTURB = C.FAITH_PERTURB
OLDC.FAITH_N_RANDOM = C.FAITH_N_RANDOM
OLDC.BENIGN_SPECS = C.BENIGN_SPECS
OLDC.CASE_SEED = C.CASE_SEED


def extract_all(tag, model, ds, device, force=False):
    """Phase 10: extract once per split using the verified extraction layer."""
    old_dir, old_ckpt = OLDC.EXTRACT_DIR, OLDC.CKPT_DIR
    OLDC.EXTRACT_DIR = C.EXTRACT_DIR          # redirect cache into study layout
    OLDC.CKPT_DIR = C.CKPT_DIR                # hash the FRESH checkpoint
    try:
        ext = EX.extract_dataset(tag, model, ds, device, force=force)
    finally:
        OLDC.EXTRACT_DIR = old_dir
        OLDC.CKPT_DIR = old_ckpt
    return ext


def run_all_diagnostics(tag, model, ds, device, ext, replay_info):
    hyp = HypothesisRegistry()
    R = dict(tag=tag, replay=replay_info)

    R["latent"] = XP.exp_latent(ext, ds, hyp)
    R["velocity"] = XP.exp_velocity(tag, model, ds, device, hyp, ext)
    R["uncertainty"] = XP.exp_intrinsic_uncertainty(ext, hyp)
    R["ensemble"] = XP.exp_ensemble(ext, hyp)
    R["pairwise"] = XP.exp_pairwise(ext, hyp)
    R["calibration"] = CAL.calibration_summary(ext["test"])
    R["risk_coverage"] = XP.exp_risk_coverage(ext)
    R["degradation"] = XP.exp_degradation(tag, model, ds, device, hyp)
    R["alpha"] = XP.exp_alpha(ext, ds, hyp)
    R["novelty_beta"] = XP.exp_novelty_beta(ext, hyp)
    R["faithfulness"] = XP.exp_faithfulness(tag, model, ds, device, ext["val"], hyp)
    R["counterfactual"] = XP.exp_counterfactual(tag, model, ds, device, hyp)
    R["benign"] = XP.exp_benign(tag, model, ds, device, hyp)
    R["baseline"] = XP.exp_baseline_summary(ext, hyp, tag)
    R["typology"] = XP.exp_error_typology(ext)

    # Phase 15: complementarity (conditional correctness)
    e = ext["test"]
    bp = e["branch_correct"]; bpred = e["branch_pred"]
    comp = {"level": "ensemble", "pairs": {}}
    K = len(C.BRANCH_NAMES)
    for a in range(K):
        for b in range(a + 1, K):
            na, nb = C.BRANCH_NAMES[a], C.BRANCH_NAMES[b]
            comp["pairs"][f"{na}-{nb}"] = dict(
                agreement=float((bpred[a] == bpred[b]).mean()),
                p_a_correct_given_b_wrong=float(
                    bp[a][bp[b] == 0].mean()) if (bp[b] == 0).any() else None,
                p_b_correct_given_a_wrong=float(
                    bp[b][bp[a] == 0].mean()) if (bp[a] == 0).any() else None,
                error_overlap=float(((bp[a] == 0) & (bp[b] == 0)).sum()
                                    / max((bp[a] == 0).sum(), 1)))
    corr = np.corrcoef(e["branch_probs"].reshape(K, -1))
    comp["prob_corr_mean"] = float((corr.sum() - K) / (K * K - K))
    solo = {b: float(bp[k].mean()) for k, b in enumerate(C.BRANCH_NAMES)}
    oracle = float(bp.any(0).mean())
    comp["per_branch_acc"] = solo
    comp["oracle_any_branch_correct"] = oracle
    comp["final_soft_acc"] = float(e["final_correct"].mean())
    comp["complementarity_score"] = float(np.clip((oracle - max(solo.values())) * 2, 0, 1))
    R["complementarity"] = comp

    # Phase 24: reliability meta-model (LogisticRegression on internal signals,
    # TRAIN fit / VAL tune / TEST eval; predicts correct-vs-incorrect)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import roc_auc_score

    def feats(e):
        cols = [e["final_conf"], e["final_entropy"], e["js_disagreement"],
                e.get("u_cs_mean", e["final_entropy"]),
                e.get("u_cmr_mean", e["final_entropy"]),
                e.get("alpha_cs", np.zeros(len(e["y"])))]
        return np.stack(cols, 1)

    Xtr, ytr = feats(ext["train"]), (ext["train"]["final_correct"]).astype(int)
    Xva, yva = feats(ext["val"]), (ext["val"]["final_correct"]).astype(int)
    Xte, yte = feats(ext["test"]), (ext["test"]["final_correct"]).astype(int)
    meta_models = {}
    configs = {
        "A_conf": [0], "B_entropy": [1], "C_disagreement": [2], "D_u": [3],
        "E_u_conf": [3, 0], "F_disagreement_conf": [2, 0], "G_u_disagreement": [3, 2],
        "H_u_disagreement_conf": [3, 2, 0], "I_alpha_u_disagreement": [5, 3, 2],
    }
    best_name, best_val = None, -1
    for name, cols in configs.items():
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        m.fit(Xtr[:, cols], ytr)
        s = m.predict_proba(Xva[:, cols])[:, 1]
        auc = float(roc_auc_score(yva, s)) if 0 < yva.sum() < len(yva) else float("nan")
        meta_models[name] = dict(val_auroc=auc)
        if np.isfinite(auc) and auc > best_val:
            best_name, best_val = name, auc
    final_m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    final_m.fit(Xtr[:, configs[best_name]], ytr)
    s_te = final_m.predict_proba(Xte[:, configs[best_name]])[:, 1]
    auc_te = float(roc_auc_score(yte, s_te)) if 0 < yte.sum() < len(yte) else float("nan")
    base_te = float(roc_auc_score(yte, Xte[:, 0])) if 0 < yte.sum() < len(yte) else float("nan")
    R["reliability_meta"] = dict(level="ensemble", selected=best_name,
                                 val_auroc=best_val, test_auroc=auc_te,
                                 conf_only_test_auroc=base_te,
                                 all_val=configs and {k: v["val_auroc"]
                                                      for k, v in meta_models.items()})
    hyp.add("reliability_meta", "AUROC (val-selected, test-evaluated)",
            f"{tag} meta-model {best_name} vs conf-only", auc_te - base_te, None,
            {"meta_auc": auc_te, "conf_auc": base_te})

    rows = hyp.finalize()
    R["_hyp_rows"] = rows
    return R


def complexity_analysis(tag, model, ds, device, meta):
    """Phase 35: params, inference time, memory, vs TURS-Lite."""
    from src.diagnostics.train_lite import make_turs_lite
    import time as _t
    n_stack = sum(p.numel() for p in model.parameters())
    lite = make_turs_lite(ds["n_cls"]).to(device)
    lite_ckpt = os.path.join(C.LITE_CKPT_DIR, f"{tag}_TURS_Lite.pt")
    n_lite, lite_time = None, None
    if os.path.exists(lite_ckpt):
        lite.load_state_dict(torch.load(lite_ckpt, map_location="cpu",
                                        weights_only=False)["model_state_dict"])
        lite.eval()
        n_lite = sum(p.numel() for p in lite.parameters())
        x = torch.from_numpy(ds["Xte"][:256]).float().to(device)
        with torch.no_grad():
            lite(x); torch.cuda.synchronize() if device.type == "cuda" else None
            t0 = _t.time()
            for _ in range(5):
                lite(x)
            lite_time = (_t.time() - t0) / 5 / 256 * 1000
    x = torch.from_numpy(ds["Xte"][:256]).float().to(device)
    with torch.no_grad():
        model(x); torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = _t.time()
        for _ in range(5):
            model(x)
        stack_time = (_t.time() - t0) / 5 / 256 * 1000
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            model(x)
        mem = torch.cuda.max_memory_allocated() / 1e6
    else:
        mem = None
    return dict(level="final", stack_params=n_stack, lite_params=n_lite,
                params_ratio=(n_stack / n_lite) if n_lite else None,
                stack_ms_per_sample=stack_time, lite_ms_per_sample=lite_time,
                peak_mem_mb=mem, n_branches=len(C.BRANCH_NAMES),
                train_time_s=meta.get("elapsed_s"))
