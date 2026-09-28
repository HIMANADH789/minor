"""HAPTICS FINAL 5-SEED — Phase 2 (MR) + Phase 3 (HERAMBA seeds 45/46).

Phase 2 (MR): the canonical Haptics MiniRocket (M0) is DETERMINISTIC by repo
convention (aeon MiniROCKET random_state=42 fitted on train only;
RidgeClassifierCV LOO deterministic), so ONE run serves all five seeds
42-46.  This script re-runs the deterministic pipeline, gates it against the
stored canonical reference M0 = 0.4974 (tolerance = repo R2 gate scale 0.011),
and materializes the per-seed MR rows + predictions.

Phase 3 (HERAMBA): user-confirmed naming -- HERAMBA == the R5 family, with the
fixed-rho R2 subset counting as the same family.  The existing audited
3-seed run `results/rcmkn_r2_haptics_3seed` (seeds 42/43/44) is REUSED.
Seeds 45/46 are executed through the SAME audited per-seed entry point
(`run_one_seed`) with the SAME frozen config (MiniROCKET fixed at 42,
learned context retrained per outer seed) -- no existing file is modified;
new artifacts land under results/haptics_final_5seed/heramba_seed{45,46}.

No test-based tuning; no protocol changes; no reruns of valid seeds.
"""
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))) if "__file__" in globals() else os.getcwd()
ROOT = os.path.abspath(os.path.join(os.getcwd()))
sys.path.insert(0, ROOT)

OUT = os.path.join(ROOT, "results", "haptics_final_5seed")
os.makedirs(OUT, exist_ok=True)

M0_REF = 0.4974
M0_TOL = 0.011          # repo-established seed/gate tolerance (R2 gate scale)
SEEDS = [42, 43, 44, 45, 46]
ALPHAS = np.logspace(-4, 4, 20)


def log(m):
    print(m, flush=True)


def macro_f1(y, p):
    from sklearn.metrics import f1_score
    return f1_score(y, p, average="macro", zero_division=0)


# ---------------------------------------------------------------- Phase 2: MR
def phase2_mr(device):
    from experiments.rcmkn_final_validation.banks import znorm, load_canonical
    from experiments.rcmkn_haptics_seed42.runner import set_seed
    from sklearn.linear_model import RidgeClassifierCV

    log("=" * 70)
    log("PHASE 2 — MINIROCKET (M0, deterministic) gate + per-seed rows")
    log("=" * 70)
    d = load_canonical("Haptics")
    Xtr, ytr, Xva, yva, Xte, yte = (d["Xtr"], d["ytr"], d["Xva"], d["yva"],
                                    d["Xte"], d["yte"])
    n_train = len(Xtr)
    Xtrva_z = np.vstack([znorm(Xtr), znorm(Xva)])
    Xte_z = znorm(Xte)
    assert (len(Xtr), len(Xva), len(Xte)) == (132, 23, 308)

    from aeon.transformations.collection.convolution_based import MiniRocket
    set_seed(42)
    extractor = MiniRocket(random_state=42, n_jobs=-1)
    extractor.fit(Xtrva_z[:n_train][:, None, :].astype(np.float32))
    F_trva = extractor.transform(Xtrva_z[:, None, :].astype(np.float32))
    F_te = extractor.transform(Xte_z[:, None, :].astype(np.float32))
    assert F_trva.shape[1] == 9996, F_trva.shape

    y_dev = np.concatenate([ytr, yva])
    ridge = RidgeClassifierCV(alphas=ALPHAS)
    ridge.fit(F_trva, y_dev)
    m0_val = round(macro_f1(yva, ridge.predict(F_trva[n_train:])), 4)
    m0_test = round(macro_f1(yte, ridge.predict(F_te)), 4)
    m0_alpha = float(ridge.alpha_)
    ok = abs(m0_test - M0_REF) <= M0_TOL
    log(f"  [MR GATE] reproduced M0 = {m0_test} vs canonical {M0_REF} "
        f"(tol {M0_TOL}) -> {'PASS' if ok else 'FAIL'}")
    assert ok, "MR deterministic reproduction gate FAILED"

    preds_te = ridge.predict(F_te).astype(np.int64)
    np.save(os.path.join(OUT, "mr_test_predictions.npy"), preds_te)
    with open(os.path.join(OUT, "mr_seed42_gate.json"), "w") as f:
        json.dump({"reference": M0_REF, "observed": m0_test,
                   "tolerance": M0_TOL, "pass": bool(ok),
                   "val_macro_f1": m0_val, "alpha": m0_alpha,
                   "convention": ("MiniROCKET random_state=42 fit on train "
                                  "only; deterministic Ridge; ONE run reused "
                                  "for all five seeds (canonical repo "
                                  "convention)")}, f, indent=2)
    rows = [{"method": "MiniRocket", "seed": s, "val_macro_f1": m0_val,
             "test_macro_f1": m0_test, "selected_alpha": m0_alpha,
             "status": "REUSED_DETERMINISTIC",
             "source": "rcmkn_haptics_seed42 convention, gated rerun"}
            for s in SEEDS]
    with open(os.path.join(OUT, "phase2_mr.json"), "w") as f:
        json.dump({"rows": rows, "gate_pass": bool(ok)}, f, indent=2)
    log(f"  MR: val={m0_val} test={m0_test} alpha={m0_alpha:.4f} "
        f"(5 seed rows materialized)")
    return rows


# ---------------------------------------------------- Phase 3: HERAMBA 45/46
def phase3_heramba(device):
    log("=" * 70)
    log("PHASE 3 — HERAMBA (R2/R5 family) seeds 45/46 via audited path")
    log("=" * 70)
    from experiments.rcmkn_r2_haptics_3seed import core as c3
    from experiments.rcmkn_r2_haptics_3seed.runner import run_one_seed
    from experiments.rcmkn_r2_haptics_3seed.config import (
        N_FEATURES, N_GLOBAL, MINIROCKET_SEED)

    data = c3.load_data()
    Xtr, ytr = data["Xtr"], data["ytr"]
    Xva, yva = data["Xva"], data["yva"]
    Xte, yte = data["Xte"], data["yte"]
    ytrva = np.concatenate([ytr, yva])
    Xtr_z, Xva_z, Xte_z = (c3.znorm(Xtr), c3.znorm(Xva), c3.znorm(Xte))
    assert (len(Xtr), len(Xva), len(Xte)) == (132, 23, 308)
    extractor = c3.build_fixed_extractor(Xtr_z)
    Xtrva_z = c3.stack_trva(Xtr_z, Xva_z)
    F_trva = extractor.transform(Xtrva_z[:, None, :].astype(np.float32))
    F_te = extractor.transform(Xte_z[:, None, :].astype(np.float32))
    G_trva, G_te = F_trva[:, :N_GLOBAL], F_te[:, :N_GLOBAL]
    valid_het = c3.compute_valid_het(extractor, Xtr_z[:8])

    rows = []
    for seed in (45, 46):
        t0 = time.time()
        res, pred_te, model = run_one_seed(
            seed, data, extractor, valid_het, Xtr_z, Xva_z, Xte_z,
            ytr, yva, yte, ytrva, G_trva, G_te, device)
        rt = round(time.time() - t0, 1)
        r = res["results"]
        sdir = os.path.join(OUT, f"heramba_seed{seed}")
        os.makedirs(sdir, exist_ok=True)
        with open(os.path.join(sdir, "result.json"), "w") as f:
            json.dump(res, f, indent=2)
        np.save(os.path.join(sdir, "test_predictions.npy"),
                np.asarray(pred_te, dtype=np.int64))
        rows.append({"method": "HERAMBA", "seed": seed,
                     "val_macro_f1": r["val_macro_f1"],
                     "test_macro_f1": r["test_macro_f1"],
                     "selected_alpha": r["selected_alpha"],
                     "status": "NEW_RUN_AUDITED_PATH",
                     "runtime_s": rt,
                     "source": ("experiments/rcmkn_r2_haptics_3seed."
                                "run_one_seed (frozen config)")})
        # sanity: no single-class collapse in HERAMBA predictions
        import collections
        dist = dict(collections.Counter(np.asarray(pred_te).tolist()))
        log(f"  [HERAMBA seed{seed}] val={r['val_macro_f1']} "
            f"test={r['test_macro_f1']} alpha={r['selected_alpha']:.4f} "
            f"pred_classes={len(dist)} ({rt}s)")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    with open(os.path.join(OUT, "phase3_heramba_new.json"), "w") as f:
        json.dump(rows, f, indent=2)
    return rows


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"HAPTICS FINAL 5-SEED — Phases 2+3 — device={device}")
    mr_rows = phase2_mr(device)
    her_rows = phase3_heramba(device)
    log("=" * 70)
    log("PHASES 2+3 COMPLETE")
    for r in mr_rows[:1] + her_rows:
        log(f"  {r['method']} seed{r['seed']}: test={r['test_macro_f1']}")
    log("=" * 70)


if __name__ == "__main__":
    main()
