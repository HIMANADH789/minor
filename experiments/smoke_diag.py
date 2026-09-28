"""SMOKE TEST for the TURS-Lite diagnostic validation pipeline.
Truncates datasets, reduces epochs, runs every phase. Not for real results."""
import os
import sys
import json

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from src.diagnostics import config as C
from src.diagnostics import data as D
from src.diagnostics import extraction as E
from src.diagnostics import statistics as S
from src.diagnostics.statistics import HypothesisRegistry
from src.diagnostics import experiments_a as A
from src.diagnostics import experiments_b as B
from src.diagnostics import calibration as CAL
from src.diagnostics import train_lite as T

# shrink everything
C.MAX_EPOCHS = 2
C.PATIENCE = 2
C.SYNTH_N_PER_CLASS = 2
C.N_BOOTSTRAP = 100
C.N_PERMUTATION = 100
S.N_BOOTSTRAP = 100
S.N_PERMUTATION = 100
C.FAITH_N_RANDOM = 1

import src.diagnostics.experiments_a as _a
import src.diagnostics.experiments_b as _b
import src.diagnostics.evidence as _e
_a.N_BOOTSTRAP = 100  # module-level copies if any

_trunc = {}


def trunc(X, n):
    return X[:n]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("SMOKE device:", device)
    for tag in ["ECG5000_UNBAL"]:
        print(f"--- {tag} ---")
        ds = D.load_split(tag)
        # truncate for speed
        for k in ["Xtr"]:
            ds[k] = ds[k][:256]
        ds["y_train"] = ds["y_train"][:256]
        ds["Xva"] = ds["Xva"][:128]; ds["y_val"] = ds["y_val"][:128]
        ds["Xte"] = ds["Xte"][:128]; ds["y_test"] = ds["y_test"][:128]

        import shutil
        ckpt = os.path.join(C.CKPT_DIR, f"{tag}_TURS_Lite.pt")
        if os.path.exists(ckpt):
            os.remove(ckpt)

        _, meta, _ = T.train_turs_lite(ds, device, log=lambda *a: None, force=True)
        model, _ = T.load_frozen(tag, device)
        ext = E.extract_dataset(tag, model, ds, device, force=True)

        hyp = HypothesisRegistry()
        r = {}
        r["latent"] = A.exp_latent_validity(ext, ds, device, hyp)
        print("  EXP1 ok")
        r["velocity"] = A.exp_velocity_localization(tag, model, ds, device, hyp)
        print("  EXP2 ok")
        r["uncertainty"] = A.exp_uncertainty_validity(ext, ds, hyp)
        r["calibration"] = {sp: CAL.calibration_summary(ext, sp) for sp in ["val", "test"]}
        r["risk_coverage"] = CAL.risk_coverage_comparison(ext, "test")
        print("  EXP3+calib ok")
        r["corruption"] = A.exp_controlled_degradation(tag, model, ds, device, hyp)
        print("  EXP4 ok")
        r["alpha"] = A.exp_alpha_validation(ext, ds, hyp)
        print("  EXP5 ok")
        r["faithfulness"] = B.exp_faithfulness(tag, model, ds, device, ext["val"], hyp)
        print("  EXP6 ok")
        r["benign"] = B.exp_benign_stability(tag, model, ds, device, hyp)
        print("  EXP7 ok")
        r["counterfactual"] = B.exp_counterfactual(tag, model, ds, device, hyp)
        print("  EXP8 ok")
        r["stability"] = B.exp_stability(tag, model, ds, device)
        r["baseline"] = B.exp_baseline_summary(ext, tag, hyp)
        print("  EXP9/10 ok")
        rows = hyp.finalize()
        print(f"  hypotheses: {len(rows)}; sample row: {rows[0]['comparison']} q={rows[0].get('q_value')}")
        # rubric build
        from src.diagnostics import evidence as EV
        rub = EV.build_rubric({tag: r})
        cons = EV.cross_dataset_consistency(rub)
        verd = EV.overall_verdict(rub, cons)
        print("  rubric OK, verdict:", verd["verdict"])
        print("SMOKE PASSED for", tag)


if __name__ == "__main__":
    main()
