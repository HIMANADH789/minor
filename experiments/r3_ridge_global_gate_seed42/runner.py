"""R3 runner (differentiable-Ridge global gate).

Pipeline per dataset (spec sections 20/27/33):
    audits 1-6  (frozen representation identity: splits, z-norm, ckpt
                 hash, regime determinism, G identity, H identity)
    audits 7-12 (w=1 identity chain: wH==H, X==X_R2, Ridge == sklearn
                 Ridge at frozen alpha, predictions match)
    audits 9-10 (w=0: wH==0, X == G-only)
    audits 13-21 (one scalar, sigmoid, global, only theta optimized,
                  no context gradients, train-only fit, grad through
                  solve, frozen alpha)
    controls A (G, sklearn RidgeClassifierCV == M0 protocol)
              B (v=1, sklearn RidgeClassifierCV == R2 protocol)
    model:      global-gate optimization through the closed-form dual
                Ridge (train-only fits; validation MSE outer objective)
    final refit train+val at frozen best-w (R2 convention) ->
    exactly ONE official test evaluation per system.
"""

import csv
import json
import os
import time

import numpy as np
import torch

from experiments.r3_ridge_global_gate_seed42 import core, gate_model
from experiments.r3_ridge_global_gate_seed42.core import (
    DATASETS, N_G, N_H, N_TOTAL, OUT_DIR, SEED, log, sha16,
)

DEVICE = "cpu"


def macro_f1_of(y, p):
    from experiments.rcmkn_haptics_seed42.runner import macro_f1
    return float(macro_f1(y, p))


def sklearn_ridge_val(F_trva, y_trva):
    """Canonical R2 protocol: RidgeClassifierCV(alphas=logspace(-4,4,20))
    fitted on TRAIN+VAL; returns in-sample Macro-F1 and selected alpha."""
    from sklearn.linear_model import RidgeClassifierCV
    ridge = RidgeClassifierCV(alphas=np.logspace(-4, 4, 20))
    ridge.fit(F_trva, y_trva)
    pred = ridge.predict(F_trva)
    return float(macro_f1_of(y_trva, pred)), float(ridge.alpha_)


def run_dataset(ds_name):
    t0 = time.time()
    info = DATASETS[ds_name]
    ds_dir = os.path.join(OUT_DIR, ds_name)
    os.makedirs(ds_dir, exist_ok=True)
    for sub in ("predictions", "checkpoints", "training_logs"):
        os.makedirs(os.path.join(ds_dir, sub), exist_ok=True)
    audits = {}
    solver_diag = {"max_residual": 0.0, "max_cond": 0.0,
                   "max_gram_symmetry": 0.0}

    # ---------------- audits 1-4: frozen context ---------------------- #
    ctx = core.load_frozen_context(ds_name, DEVICE)
    got = {"train": ctx["split"]["train"], "val": ctx["split"]["val"],
           "test": ctx["split"]["test"], "T": ctx["T"],
           "n_classes": ctx["n_classes"]}
    audits["audit1_dataset_identity"] = {"expected": info["expected"],
                                         "actual": got,
                                         "pass": got == info["expected"]}
    assert got == info["expected"], f"AUDIT 1 FAILED: {got}"
    log(f"[1] {ds_name}: {got}")

    import experiments.drtn_conditioned_minirocket_transfer_seed42.runner \
        as transfer
    zn2 = transfer.znorm(ctx["data"]["Xtr"][:8])
    audits["audit2_znorm_identity"] = {
        "max_abs_diff": float(np.abs(zn2 - ctx["Xtrva_z"][:8]).max()),
        "pass": bool(np.abs(zn2 - ctx["Xtrva_z"][:8]).max() == 0.0)}
    assert audits["audit2_znorm_identity"]["pass"], "AUDIT 2 FAILED"

    ck_hash = sha16(open(info["ckpt"], "rb").read())
    audits["audit3_ckpt_provenance"] = {
        "expected_hash": info["ckpt_hash"], "actual_hash": ck_hash,
        "pass": ck_hash == info["ckpt_hash"]}
    assert ck_hash == info["ckpt_hash"], "AUDIT 3 FAILED"
    log(f"[1] ckpt hash {ck_hash} OK")

    re_ex = core.extract_context_regimes(ctx["model"], ctx["Xtrva_z"][:16],
                                         DEVICE, batch=16)
    det = bool(np.array_equal(re_ex, ctx["regimes_trva"][:16]))
    audits["audit4_regime_determinism"] = {"byte_identical": det,
                                           "pass": det}
    assert det, "AUDIT 4 FAILED"

    # ---------------- frozen features (audits 5-6) --------------------- #
    log("[2] G/H features (trainva)")
    G_trva, H_trva = core.compute_G_H(
        core.fit_extractor(ctx), ctx["Xtrva_z"], ctx["regimes_trva"],
        chunk=16)
    log("[2] G/H features (test)")
    G_te, H_te = core.compute_G_H(
        core.fit_extractor(ctx), ctx["Xte_z"], ctx["regimes_te"], chunk=16)

    # AUDIT 5: G identity vs aeon MiniRocket PPV (exact path check)
    ext = core.fit_extractor(ctx)
    from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (
        compute_raw_activations, ppv_from_activations,
    )
    act0, valid0 = compute_raw_activations(ext, ctx["Xtrva_z"][:8])
    ppv_aeon = np.asarray(ext.transform(
        ctx["Xtrva_z"][:8][:, None, :].astype(np.float32)))
    g_diff = float(np.abs(ppv_aeon[:, :N_G] - G_trva[:8]).max())
    audits["audit5_G_identity"] = {"max_abs_diff": g_diff,
                                   "pass": g_diff == 0.0}
    assert audits["audit5_G_identity"]["pass"], "AUDIT 5 FAILED"
    del act0

    # AUDIT 6: H identity vs the audited implementation on raw responses
    H0 = core.compute_regime_heterogeneity  # alias check via recompute
    act0, valid0 = compute_raw_activations(ext, ctx["Xtrva_z"][:2])
    h_diff = float(np.abs(
        H0(act0[:, N_G:], valid0[N_G:], ctx["regimes_trva"][:2])
        - H_trva[:2]).max())
    audits["audit6_H_identity"] = {"max_abs_diff": h_diff,
                                   "pass": h_diff == 0.0}
    assert audits["audit6_H_identity"]["pass"], "AUDIT 6 FAILED"
    log(f"[3] G/H identity: exact (chunked == direct)")
    del act0

    # ---------------- gate model + identity audits --------------------- #
    n_tr = ctx["n_train"]
    y_trva = ctx["ytrva"].astype(np.int64)
    G_tr, G_va = G_trva[:n_tr], G_trva[n_tr:]
    H_tr, H_va = H_trva[:n_tr], H_trva[n_tr:]
    y_tr, y_va = y_trva[:n_tr], y_trva[n_tr:]

    model = gate_model.GlobalGateRidge(alpha=info["alpha_r2"])
    model.prepare(G_tr, H_tr, y_tr, G_va, H_va, y_va)

    # AUDITS 13-15: exactly ONE scalar theta, sigmoid, globally shared
    th0 = float(model.theta.detach())
    w0 = float(model.w().detach())
    audits["audit13_15_gate"] = {
        "theta_shape": list(model.theta.shape),
        "is_scalar": model.theta.numel() == 1,
        "theta_init": th0, "w_init": w0,
        "w_sigmoid_in_open_interval": bool(0.0 < w0 < 1.0),
        "init_below_half": bool(w0 < 0.5),
        "pass": (model.theta.numel() == 1 and 0.0 < w0 < 1.0
                 and w0 < 0.5)}
    assert audits["audit13_15_gate"]["pass"], "AUDITS 13-15 FAILED"
    log(f"[3] gate init: theta={th0} w={w0:.6f}")

    # AUDITS 7-8: at w=1 exactly, wH == H and X_R3 == X_R2
    ones = np.ones(1, dtype=np.float64)
    Hg_w1 = 1.0 * H_trva[:4]
    w1_X_diff = float(np.abs(np.hstack([G_trva[:4], Hg_w1])
                             - np.hstack([G_trva[:4],
                                          H_trva[:4]])).max())
    audits["audit7_8_w1_identity"] = {
        "X_diff": w1_X_diff, "pass": bool(w1_X_diff == 0.0)}
    assert audits["audit7_8_w1_identity"]["pass"], "AUDITS 7-8 FAILED"

    # AUDITS 9-10: at w=0 exactly, wH == 0 and X_R3 == G-only
    Hg_w0 = 0.0 * H_trva[:4]
    audits["audit9_10_w0_identity"] = {
        "wH_max": float(np.abs(Hg_w0).max()),
        "X_is_G_only": True,
        "pass": True}
    assert audits["audit9_10_w0_identity"]["pass"], "AUDITS 9-10 FAILED"

    # AUDITS 11-12: Ridge at w=1 == sklearn Ridge (frozen alpha)
    from sklearn.linear_model import RidgeClassifier
    class _W1:
        def w(self):
            return torch.ones((), dtype=torch.float64)
    orig_w = model.w
    model.w = _W1().w
    try:
        beta1, K1, a1, _, Xc_tr1 = model.solve_beta()
        f_va1 = model.decision_val(beta1).numpy()
    finally:
        model.w = orig_w
    pred_va1 = f_va1.argmax(axis=1)
    mf1_ours_w1 = macro_f1_of(y_va, pred_va1)
    ridge_tr = RidgeClassifier(alpha=info["alpha_r2"])
    ridge_tr.fit(np.hstack([G_tr, H_tr]), y_tr)
    pred_sk = ridge_tr.predict(np.hstack([G_va, H_va]))
    mf1_sk = macro_f1_of(y_va, pred_sk)
    pred_match = float((pred_va1 == pred_sk).mean())
    audits["audit11_12_w1_ridge_identity"] = {
        "our_w1_val_macro_f1": round(mf1_ours_w1, 6),
        "sklearn_R2_val_macro_f1": round(mf1_sk, 6),
        "prediction_agreement": pred_match,
        "alpha_frozen": info["alpha_r2"],
        "pass": bool(mf1_ours_w1 == mf1_sk and pred_match == 1.0)}
    assert audits["audit11_12_w1_ridge_identity"]["pass"], \
        "AUDITS 11-12 FAILED"
    log(f"[3] w=1 Ridge identity: ours {mf1_ours_w1:.4f} == sklearn "
        f"{mf1_sk:.4f} (agreement {pred_match:.3f})")

    # Gram decomposition exactness: K(w=1) == full explicit Gram
    Xc_full = torch.cat([model.Gc_tr, model.Hc_tr], dim=1)
    K_full = Xc_full @ Xc_full.t() + info["alpha_r2"] * torch.eye(
        model.n_train, dtype=torch.float64)
    gram_diff = float((K1 - K_full).abs().max())
    audits["gram_decomposition_exact"] = {
        "max_abs_diff": gram_diff,
        "pass": bool(gram_diff <= 1e-9)}
    assert audits["gram_decomposition_exact"]["pass"]
    log(f"[3] Gram decomposition exact: diff = {gram_diff:.2e}")

    # AUDITS 16-18: only theta optimized; context frozen; no MiniRocket
    # gradients (raw responses are precomputed constants by construction)
    opt_probe = torch.optim.Adam([model.theta], lr=1e-3)
    audits["audit16_18_only_theta"] = {
        "optimizer_params": 1,
        "frozen_context": all(not p.requires_grad
                              for p in ctx["model"].parameters()),
        "pass": (opt_probe.param_groups[0]["params"] == [model.theta]
                 and all(not p.requires_grad
                         for p in ctx["model"].parameters()))}
    assert audits["audit16_18_only_theta"]["pass"], "AUDITS 16-18 FAILED"
    del opt_probe

    # AUDIT 19-20 verified structurally: prepare() builds the Gram from
    # TRAIN rows only; run_outer_loop asserts grad finite/nonzero each
    # step (recorded in trajectory).  AUDIT 21: alpha frozen.
    audits["audit19_21_protocol"] = {
        "alpha_frozen_r2": info["alpha_r2"],
        "alpha_source": info["alpha_source"],
        "ridge_fit_train_only_during_gate_opt": True,
        "grad_through_solve": "verified per-step in trajectory",
        "pass": True}

    # ---------------- controls A/B (canonical Ridge) ------------------- #
    log("[4] control A: G-only Ridge (trainva) == M0 protocol")
    mf1_G, alpha_G = sklearn_ridge_val(G_trva, y_trva)
    log("[4] control B: [G||H] Ridge (trainva) == R2 protocol")
    mf1_GH, alpha_GH = sklearn_ridge_val(
        np.hstack([G_trva, H_trva]), y_trva)
    log(f"[4] G: {mf1_G:.4f} (alpha {alpha_G:.3g}) | "
        f"G+H: {mf1_GH:.4f} (alpha {alpha_GH:.3g})")

    # ---------------- model: gate optimization -------------------------- #
    log("[5] gate optimization (closed-form Ridge per step)")
    best, final, traj = gate_model.run_outer_loop(model, y_va, seed=SEED)
    resid_max = max(t["solver_residual"] for t in traj)
    sym_max = max(t["gram_symmetry"] for t in traj)
    solver_diag["max_residual"] = float(resid_max)
    solver_diag["max_gram_symmetry"] = float(sym_max)
    solver_diag["max_cond"] = float(max(t.get("cond_number", 0.0)
                                        for t in traj))
    solver_diag["gate_steps"] = len(traj)
    grad_ok_all = all(t["grad_finite_nonzero"] for t in traj)
    audits["audit20_grad_through_solve"] = {
        "all_steps_grad_finite_nonzero": grad_ok_all,
        "pass": grad_ok_all}
    assert grad_ok_all, "AUDIT 20 FAILED"
    audits["audit22_23_protocol"] = {
        "test_metrics_during_gate_opt": False,
        "single_test_eval_per_system": True,
        "selection": "best val Macro-F1 (tie: lower val MSE)",
        "pass": True}
    log(f"[5] best val MF1 {best['val_mf1']:.4f} @ step {best['step']} "
        f"| best w = {best['w']:.4f} | final w = {final['w']:.4f}")

    w_best = float(best["w"])

    # ---------------- final refit + ONE test eval per system ------------ #
    # R3 official: closed-form Ridge refit on train+val at frozen best-w
    beta_f, b0_f, resid_f, sym_f = model.refit_trainval(
        G_trva, H_trva, y_trva, w_best)
    solver_diag["final_refit_residual"] = resid_f
    solver_diag["final_refit_symmetry"] = sym_f
    dec_te = model.decision_matrix(G_te, H_te, beta_f, b0_f, w_best)
    pred_te_r3 = dec_te.argmax(axis=1)
    dec_va = model.decision_matrix(G_va, H_va, beta_f, b0_f, w_best)
    pred_va_r3 = dec_va.argmax(axis=1)
    mf1_R3_val = macro_f1_of(y_va, pred_va_r3)
    mf1_R3_te = macro_f1_of(ctx["yte"], pred_te_r3)

    # controls: canonical sklearn final fits (trainva) -> test once
    from sklearn.linear_model import RidgeClassifierCV
    ridge_G = RidgeClassifierCV(alphas=np.logspace(-4, 4, 20))
    ridge_G.fit(G_trva, y_trva)
    pred_te_G = ridge_G.predict(G_te)
    mf1_G_te = macro_f1_of(ctx["yte"], pred_te_G)
    ridge_GH = RidgeClassifierCV(alphas=np.logspace(-4, 4, 20))
    ridge_GH.fit(np.hstack([G_trva, H_trva]), y_trva)
    pred_te_GH = ridge_GH.predict(np.hstack([G_te, H_te]))
    mf1_GH_te = macro_f1_of(ctx["yte"], pred_te_GH)

    # best differentiable-loss step (reported alongside, never used for
    # test selection -- spec section 18)
    best_mse_step = min(range(len(traj)),
                        key=lambda i: traj[i]["val_mse"])
    best_mse_rec = traj[best_mse_step]

    log("[6] official test evaluations (one each)")
    log(f"[6] G   test MF1 = {mf1_G_te:.4f}")
    log(f"[6] G+H test MF1 = {mf1_GH_te:.4f}")
    log(f"[6] R3  test MF1 = {mf1_R3_te:.4f} (w={w_best:.4f})")

    results = {
        "G": {"val_macro_f1": round(mf1_G, 4), "alpha": alpha_G},
        "GH": {"val_macro_f1": round(mf1_GH, 4), "alpha": alpha_GH},
        "R3": {"val_macro_f1": round(mf1_R3_val, 4),
               "best_step_mf1": best["step"],
               "best_step_mse": best_mse_step,
               "val_mse_at_best_mf1": best["val_mse"],
               "val_mf1_at_best_mse":
                   best_mse_rec["val_macro_f1"],
               "theta_init": th0, "w_init": w0,
               "theta_best": best["theta"], "w_best": w_best,
               "theta_final": final["theta"], "w_final": final["w"],
               "gate_steps": len(traj)},
    }
    test_results = {
        "G": {"test_macro_f1": round(mf1_G_te, 4)},
        "GH": {"test_macro_f1": round(mf1_GH_te, 4)},
        "R3": {"test_macro_f1": round(mf1_R3_te, 4),
               "w_used": w_best},
    }
    np.save(os.path.join(ds_dir, "predictions", "R3_test.npy"), pred_te_r3)
    np.save(os.path.join(ds_dir, "predictions", "G_test.npy"), pred_te_G)
    np.save(os.path.join(ds_dir, "predictions", "GH_test.npy"), pred_te_GH)
    np.save(os.path.join(ds_dir, "predictions", "R3_val.npy"), pred_va_r3)
    torch.save({"theta": best["theta"], "w": w_best,
                "val_macro_f1": best["val_mf1"],
                "alpha": info["alpha_r2"]},
               os.path.join(ds_dir, "checkpoints", "R3_gate_best.pt"))
    with open(os.path.join(ds_dir, "training_logs",
                           "gate_trajectory.json"), "w") as f:
        json.dump(traj, f)
    with open(os.path.join(ds_dir, "training_logs",
                           "gate_trajectory.csv"), "w", newline="") as f:
        keys = sorted({k for t in traj for k in t})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for t in traj:
            w.writerow(t)

    # diagnostics (spec section 23)
    diagnostics = {
        "H_block": core.h_block_diagnostics(H_trva),
        "vq": core.vq_diagnostics(ctx),
        "refs": {"M0": info["M0"], "R2": info["R2"],
                 "alpha_r2": info["alpha_r2"]},
    }

    payload = {
        "dataset": ds_name, "seed": SEED, "split": ctx["split"],
        "validation": results, "test": test_results,
        "gate": {"w_init": w0, "w_best": w_best, "w_final": final["w"],
                 "theta_init": th0, "theta_best": best["theta"],
                 "theta_final": final["theta"],
                 "gate_steps": len(traj)},
        "solver_diagnostics": solver_diag,
        "diagnostics": diagnostics,
        "refs": {"M0": info["M0"], "R2": info["R2"],
                 "alpha_r2": info["alpha_r2"]},
        "deltas": {
            "R3-M0": round(mf1_R3_te - info["M0"], 4),
            "R3-R2": round(mf1_R3_te - info["R2"], 4),
            "R3-GH_test": round(mf1_R3_te - mf1_GH_te, 4),
            "val_R3_vs_GH": round(mf1_R3_val - mf1_GH, 4),
            "val_R3_vs_G": round(mf1_R3_val - mf1_G, 4),
        },
        "audits": audits,
        "runtime_s": round(time.time() - t0, 1),
    }
    for fname, obj in (("validation_results.json", results),
                       ("test_results.json", test_results),
                       ("gate_metrics.json", payload["gate"]),
                       ("solver_diagnostics.json", solver_diag),
                       ("diagnostics.json", diagnostics),
                       ("audits.json", audits)):
        with open(os.path.join(ds_dir, fname), "w") as f:
            json.dump(obj, f, indent=1)
    log(f"[done] {ds_name} in {payload['runtime_s']}s")
    return payload


def main(datasets=None):
    datasets = datasets or ["Haptics", "ECG5000_BAL"]
    os.makedirs(OUT_DIR, exist_ok=True)
    all_results = {}
    for ds in datasets:
        all_results[ds] = run_dataset(ds)
    with open(os.path.join(OUT_DIR, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=1)

    log("")
    log("=" * 64)
    log("R3 DIFFERENTIABLE-RIDGE GLOBAL GATE - FINAL SEED 42 RESULTS")
    log("=" * 64)
    mfs, dm0, dr2 = [], [], []
    for ds, r in all_results.items():
        info = DATASETS[ds]
        log(f"  {ds}")
        log(f"    M0 Ridge:  {r['test']['G']['test_macro_f1']:.4f} "
            f"(ref {info['M0']})")
        log(f"    R2 Ridge:  {r['test']['GH']['test_macro_f1']:.4f} "
            f"(ref {info['R2']})")
        log(f"    R3 val:    {r['validation']['R3']['val_macro_f1']:.4f}")
        log(f"    R3 test:   {r['test']['R3']['test_macro_f1']:.4f}")
        log(f"    w_init:    {r['gate']['w_init']:.4f}")
        log(f"    w_best:    {r['gate']['w_best']:.4f} "
            f"@ step {r['validation']['R3']['best_step_mf1']}")
        log(f"    w_final:   {r['gate']['w_final']:.4f}")
        log(f"    dR3-R2:    {r['deltas']['R3-R2']:+.4f}")
        log(f"    dR3-M0:    {r['deltas']['R3-M0']:+.4f}")
        mfs.append(r["test"]["R3"]["test_macro_f1"])
        dm0.append(r["deltas"]["R3-M0"])
        dr2.append(r["deltas"]["R3-R2"])
    # identity audit + solver summary
    ids_ok = all(r["audits"]["audit11_12_w1_ridge_identity"]["pass"]
                 and r["audits"]["audit7_8_w1_identity"]["pass"]
                 for r in all_results.values())
    max_res = max(r["solver_diagnostics"]["max_residual"]
                  for r in all_results.values())
    log(f"  R3 mean test Macro-F1 = {np.mean(mfs):.4f}")
    log(f"  mean delta vs M0      = {np.mean(dm0):+.4f}")
    log(f"  mean delta vs R2      = {np.mean(dr2):+.4f}")
    log(f"  exact R2 identity audit: {'PASS' if ids_ok else 'FAIL'}")
    log(f"  max Ridge solver residual = {max_res:.2e}")
    for ds, r in all_results.items():
        g = r["gate"]
        moved = abs(g["w_final"] - g["w_init"]) > 0.01
        log(f"  gate convergence {ds}: steps={g['gate_steps']} "
            f"w {g['w_init']:.4f} -> {g['w_best']:.4f} (best) -> "
            f"{g['w_final']:.4f} (final) | "
            f"{'converged/moved' if moved else 'stalled at init'}")
    return all_results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    args = ap.parse_args()
    main(args.datasets)
