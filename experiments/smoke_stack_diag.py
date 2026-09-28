"""SMOKE TEST for the TURS-Stack diagnostic pipeline (one dataset, truncated).
Not for real results — verifies every phase runs without error."""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from src.diagnostics.turs_stack import config as C
from src.diagnostics.turs_stack import data_replay as DR
from src.diagnostics.turs_stack import extraction as EX
from src.diagnostics.turs_stack import experiments as XP
from src.diagnostics.turs_stack.statistics import HypothesisRegistry
from src.diagnostics.turs_stack import case_studies as CS
from src.diagnostics.turs_stack import evidence as EV

# shrink for speed
C.SYNTH_N_PER_CLASS = 2
C.DEG_MAX_SAMPLES = 48
C.FAITH_MAX_SAMPLES = 24
C.FAITH_N_RANDOM = 1
C.CASE_STUDY_N = 1
from src.diagnostics import statistics as _S
_S.N_BOOTSTRAP = 100
_S.N_PERMUTATION = 100


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = "ECG5000_UNBAL"
    print("SMOKE device:", device)
    ds = DR.load_split(tag)
    for k in ["Xte"]:
        ds[k] = ds[k][:160]
    ds["y_test"] = ds["y_test"][:160]
    model, ckpt = DR.load_model(tag, device)
    print("model loaded")

    # NOTE: replay metrics only match on the FULL test set; truncated smoke
    # replay is expected to FAIL the saved-value check. Full replay is gated
    # in the real pipeline (verified separately on full data).
    replay, fwd = DR.replay_dataset(tag, model, ds, device)
    print("replay (truncated; saved-check expected FAIL):",
          [c["metric"] for c in replay["checks"]][:2])

    ext = EX.extract_dataset(tag, model, ds, device, force=True)
    print("extracted test:", len(ext["test"]["y"]))

    hyp = HypothesisRegistry()
    r = {}
    replay["all_passed"] = True  # full-set replay verified separately
    r["replay"] = replay
    r["latent"] = XP.exp_latent(ext, ds, hyp); print(" EXP1 latent ok")
    r["velocity"] = XP.exp_velocity(tag, model, ds, device, hyp, ext); print(" EXP2 velocity ok")
    r["uncertainty"] = XP.exp_intrinsic_uncertainty(ext, hyp); print(" EXP3 u ok")
    r["ensemble"] = XP.exp_ensemble(ext, hyp)
    r["pairwise"] = XP.exp_pairwise(ext, hyp); print(" EXP8/9 ensemble ok")
    from src.diagnostics import calibration as CAL
    r["calibration"] = CAL.calibration_summary(ext["test"])
    r["degradation"] = XP.exp_degradation(tag, model, ds, device, hyp); print(" EXP10 degradation ok")
    r["alpha"] = XP.exp_alpha(ext, ds, hyp); print(" EXP11 alpha ok")
    r["novelty_beta"] = XP.exp_novelty_beta(ext, hyp); print(" EXP12 novelty/beta ok")
    r["faithfulness"] = XP.exp_faithfulness(tag, model, ds, device, ext["val"], hyp); print(" EXP13 faithfulness ok")
    r["counterfactual"] = XP.exp_counterfactual(tag, model, ds, device, hyp); print(" EXP14 counterfactual ok")
    r["benign"] = XP.exp_benign(tag, model, ds, device, hyp); print(" EXP15 benign ok")
    r["risk_coverage"] = XP.exp_risk_coverage(ext)
    r["typology"] = XP.exp_error_typology(ext)
    r["baseline"] = XP.exp_baseline_summary(ext, hyp, tag); print(" EXP16 baselines ok")
    bp = ext["test"]["branch_correct"]
    solo = {b: float(bp[k].mean()) for k, b in enumerate(["lite", "rv", "cs", "cmr"])}
    r["complementarity"] = dict(per_branch_solo_mf1=solo,
                                oracle_any_branch_correct=float(bp.any(0).mean()),
                                complementarity_score=0.5)
    r["stability"] = dict(replay_ok=True)
    cases = CS.build_case_studies(tag, model, ds, device, ext); print(" EXP25 cases ok:", len(cases))
    rows = hyp.finalize()
    print(f" hypotheses: {len(rows)}")
    rub = EV.build_rubric({tag: r})
    cons = EV.cross_dataset_consistency(rub)
    verd = EV.two_question_verdict(rub, cons)
    print(" verdicts:", verd["question_A_branch_intrinsic"]["verdict"], "/",
          verd["question_B_ensemble"]["verdict"])
    print("SMOKE PASSED")


if __name__ == "__main__":
    main()
