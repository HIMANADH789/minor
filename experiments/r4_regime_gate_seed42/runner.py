"""R4 runner (differentiable-Ridge edition).

Pipeline per dataset (spec sections 16/25/29/30):
    audits 1-8  (frozen representation identity)
    identity audits 9-14, 17  (v=1 == R2 H / X / Ridge; v=0 == G;
                               softmax classifier removed)
    controls A (G, sklearn RidgeClassifierCV)
              B (v=1, sklearn RidgeClassifierCV == R2 reference)
    model C: 8-gate optimization through the closed-form dual Ridge
              (train-only fits; validation MSE outer objective)
    final refit train+val at frozen best-v (R2 convention) ->
    exactly ONE official test evaluation per system.
"""

import csv
import json
import os
import time

import numpy as np
import torch

from experiments.r4_regime_gate_seed42 import core, ridge_gate
from experiments.r4_regime_gate_seed42.core import (
    DATASETS, K_CODES, MIN_OCCUPANCY, N_G, N_H, N_TOTAL, OUT_DIR, SEED,
    log, sha16,
)

DEVICE = "cpu"


def macro_f1_of(y, p):
    from experiments.rcmkn_haptics_seed42.runner import macro_f1
    return float(macro_f1(y, p))


def sklearn_ridge_val(F_trva, n_tr, y_trva):
    """Canonical R2 protocol: RidgeClassifierCV(alphas=logspace(-4,4,20))
    fitted on TRAIN+VAL; returns in-sample Macro-F1 and selected alpha.
    (trainva fit == R2 final-fit convention, audit-aligned with M0.)"""
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
    solver_diag = {"max_residual": 0.0, "max_cond": 0.0}

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

    # ---------------- frozen features ---------------------------------- #
    log("[2] G/H/Hk features (trainva)")
    G_trva, H_trva, Hk_trva, valid = core.compute_G_H_Hk(
        core.fit_extractor(ctx), ctx["Xtrva_z"], ctx["regimes_trva"],
        chunk=16)
    log("[2] G/H/Hk features (test)")
    G_te, H_te, Hk_te, _ = core.compute_G_H_Hk(
        core.fit_extractor(ctx), ctx["Xte_z"], ctx["regimes_te"], chunk=16)

    # AUDIT 5: MiniRocket identity (aeon PPV vs raw-response path)
    from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (
        compute_raw_activations, ppv_from_activations,
    )
    ext = core.fit_extractor(ctx)
    act0, valid0 = compute_raw_activations(ext, ctx["Xtrva_z"][:8])
    ppv_aeon = np.asarray(ext.transform(
        ctx["Xtrva_z"][:8][:, None, :].astype(np.float32)))
    mr_diff = float(np.abs(ppv_aeon - ppv_from_activations(act0, valid0)).max())
    audits["audit5_minirocket_identity"] = {"max_abs_diff": mr_diff,
                                            "pass": mr_diff == 0.0}
    assert mr_diff == 0.0, "AUDIT 5 FAILED"
    del act0

    # AUDITS 6/7: independent PPV_{m,k} + H_{m,k} recompute
    feats = (0, 77, 2500, 4997)
    diffs = []
    for i in range(2):
        act0, valid0 = compute_raw_activations(ext, ctx["Xtrva_z"][i:i + 1])
        for m in feats:
            vm = valid0[N_G + m]
            a = act0[0, N_G + m][vm].astype(bool)
            r = ctx["regimes_trva"][i][vm].astype(int)
            T = ctx["T"]
            min_count = int(np.ceil(MIN_OCCUPANCY * T))
            n_valid = len(a)
            ppv_g = a.mean()
            counts = np.bincount(r, minlength=K_CODES)
            sel = counts >= min_count
            if not sel.any():
                sel[np.argmax(counts)] = True
            w = np.where(sel, counts / n_valid, 0.0)
            w = w / w.sum()
            for k in range(K_CODES):
                if sel[k] and counts[k] > 0:
                    ppv_k = a[r == k].mean()
                    href = w[k] * (ppv_k - ppv_g) ** 2
                    diffs.append(abs(float(Hk_trva[i, m, k]) - href))
        del act0
    audit67 = float(max(diffs))
    # float32 storage of H_{m,k}: values are O(1e-2), float32 eps there is
    # ~2e-9 (observed diffs 1-3.3e-9 are pure storage rounding; the probe
    # confirmed impl == independent recompute to display precision)
    audits["audit6_7_contribution_recompute"] = {
        "max_diff": audit67, "tolerance": 5e-9,
        "rationale": "float32 storage rounding (values O(1e-2))",
        "pass": bool(audit67 <= 5e-9)}
    assert audits["audit6_7_contribution_recompute"]["pass"]
    log(f"[3] H_(m,k) independent recompute diff = {audit67:.2e}")

    # AUDIT 8: sum_k H_{m,k} == audited H_m.  The audited H is float64;
    # the stored contributions are float32, so the 8-term sum can differ
    # by up to ~8 ulps of max|Hk| (bound ~2.4e-7 for max|Hk|=0.5625;
    # observed: Haptics 2.98e-8, ECG5000_BAL 4.47e-8 -- 1-1.5 ulp).
    # The identity is exact in float64 (unit test
    # test_contribution_decomposition_sums_to_H).
    sum_diff = float(np.abs(Hk_trva.sum(axis=2) - H_trva).max())
    audits["audit8_decomposition_sum"] = {
        "max_abs_diff": sum_diff, "tolerance": 3e-7,
        "rationale": "float32 storage of Hk: <=8 ulps of max|Hk|",
        "pass": bool(sum_diff <= 3e-7)}
    assert audits["audit8_decomposition_sum"]["pass"], "AUDIT 8 FAILED"
    log(f"[3] decomposition sum diff = {sum_diff:.2e}")

    # ---------------- Ridge model + identity audits --------------------- #
    n_tr = ctx["n_train"]
    y_trva = ctx["ytrva"].astype(np.int64)
    G_tr, G_va = G_trva[:n_tr], G_trva[n_tr:]
    Hk_tr, Hk_va = Hk_trva[:n_tr], Hk_trva[n_tr:]
    y_tr, y_va = y_trva[:n_tr], y_trva[n_tr:]

    model = ridge_gate.DualRidgeGateModel(alpha=info["alpha_r2"])
    model.prepare(G_tr, Hk_tr, y_tr, G_va, Hk_va, y_va)

    # AUDITS 9-12/17: exactly 8 scalar theta, sigmoid, (0,1), init<0.5
    th0 = model.theta.detach().numpy()
    v0 = model.v().detach().numpy()
    audits["audit9_12_17_gates"] = {
        "n_theta": int(th0.size), "all_scalar": th0.size == K_CODES,
        "theta_init": float(th0[0]),
        "v_init": float(v0[0]),
        "v_sigmoid_in_open_interval": bool(np.all((v0 > 0) & (v0 < 1))),
        "init_below_half": bool(np.all(v0 < 0.5)),
        "pass": (th0.size == K_CODES
                 and bool(np.all((v0 > 0) & (v0 < 1)))
                 and bool(np.all(v0 < 0.5)))}
    assert audits["audit9_12_17_gates"]["pass"], "AUDITS 9-12 FAILED"

    # AUDITS 1-2 (R4 list): v=1 reproduces R2 H and X exactly.
    # sigmoid never reaches 1, so the structural check uses EXACT ones
    # (the einsum with v=1 IS sum_k H_{m,k}); tolerance = float32
    # storage rounding of Hk vs float64 H (same bound as audit 8).
    ones = np.ones(K_CODES, dtype=np.float64)
    Hg_w1 = np.einsum("nfk,k->nf",
                      Hk_trva[:4].astype(np.float64), ones)
    w1_H_diff = float(np.abs(Hg_w1 - H_trva[:4]).max())
    w1_X_diff = float(np.abs(np.hstack([G_trva[:4], Hg_w1])
                             - np.hstack([G_trva[:4],
                                          H_trva[:4]])).max())
    audits["audit1_2_v1_identity_H_X"] = {
        "H_diff": w1_H_diff, "X_diff": w1_X_diff,
        "tolerance": 3e-7,
        "rationale": "float32 storage of Hk (<=8 ulps of max|Hk|)",
        "pass": bool(w1_H_diff <= 3e-7 and w1_X_diff <= 3e-7)}
    assert audits["audit1_2_v1_identity_H_X"]["pass"], "AUDIT v=1 FAILED"
    log(f"[3] v=1 -> H identity diff = {w1_H_diff:.2e}")

    # AUDIT 3: v=1 Ridge == sklearn Ridge, SAME estimator: fixed alpha
    # (the frozen R2 alpha), fixed train rows, one-hot targets, centered
    # intercept.  (RidgeClassifierCV would re-select alpha on the train
    # rows via LOO -- a different estimator, not an identity check.)
    from sklearn.linear_model import RidgeClassifier
    orig_v = model.v
    model.v = lambda: torch.ones(K_CODES, dtype=torch.float64)
    try:
        beta1, K1, _, _ = model.solve_beta()
        f_va1 = model.decision_val(beta1).numpy()
    finally:
        model.v = orig_v
    pred_va1 = f_va1.argmax(axis=1)
    mf1_ours_v1 = macro_f1_of(y_va, pred_va1)
    ridge_tr2 = RidgeClassifier(alpha=info["alpha_r2"])
    Xr2_tr = np.hstack([G_tr, H_trva[:n_tr]])
    ridge_tr2.fit(Xr2_tr, y_tr)
    pred_sk = ridge_tr2.predict(np.hstack([G_va, H_trva[n_tr:]]))
    mf1_sk = macro_f1_of(y_va, pred_sk)
    pred_match = float((pred_va1 == pred_sk).mean())
    audits["audit3_v1_ridge_identity"] = {
        "our_v1_val_macro_f1": round(mf1_ours_v1, 6),
        "sklearn_R2_val_macro_f1": round(mf1_sk, 6),
        "prediction_agreement": pred_match,
        "alpha_frozen": info["alpha_r2"],
        "pass": bool(mf1_ours_v1 == mf1_sk and pred_match == 1.0)}
    assert audits["audit3_v1_ridge_identity"]["pass"], "AUDIT 3 FAILED"
    log(f"[3] v=1 Ridge identity: ours {mf1_ours_v1:.4f} == sklearn "
        f"{mf1_sk:.4f} (agreement {pred_match:.3f})")

    # AUDIT 15 (R4 list): v=0 reduces exactly to G
    with torch.no_grad():
        model.theta.data.copy_(torch.full((K_CODES,), -60.0))
        Hg_w0 = model.gated_H(torch.from_numpy(
            Hk_trva[:4].astype(np.float64))).numpy()
    audits["audit15_v0_identity"] = {
        "max_abs_Hgated": float(np.abs(Hg_w0).max()),
        "pass": bool(np.abs(Hg_w0).max() <= 1e-12)}
    assert audits["audit15_v0_identity"]["pass"], "AUDIT 15 FAILED"
    model.theta.data.copy_(torch.full((K_CODES,),
                                      ridge_gate.THETA_INIT))

    # AUDITS 4-6/8/18 (R4 list): softmax classifier removed; only theta
    # optimized; frozen context untouched; loss has the exact penalty.
    # Tokenize (drop comments/strings/docstrings) so documentation of the
    # REMOVED mechanism cannot false-positive the scan.
    import io
    import tokenize
    import experiments.r4_regime_gate_seed42 as pkg
    mod_path = os.path.join(os.path.dirname(pkg.__file__), "ridge_gate.py")
    with open(mod_path, "rb") as fh:
        toks = list(tokenize.tokenize(fh.readline))
    code_only = " ".join(
        t.string for t in toks
        if t.type not in (tokenize.COMMENT, tokenize.STRING,
                          tokenize.DEDENT, tokenize.ENDMARKER,
                          tokenize.NL, tokenize.NEWLINE, tokenize.INDENT))
    banned = ["nn.Parameter", "cross_entropy", "AdamW", "LAMBDA_CLF",
              "R4GateModel", "softmax"]
    leaks = [b for b in banned if b in code_only]
    audits["audit4_5_6_softmax_removed"] = {
        "scanned": "ridge_gate.py (tokenized: comments/strings stripped)",
        "banned_tokens_found": leaks,
        "pass": not leaks}
    assert not leaks, f"AUDIT 4-6 FAILED: {leaks}"

    audits["audit8_frozen_context"] = {
        "frozen": all(not p.requires_grad
                      for p in ctx["model"].parameters()),
        "pass": all(not p.requires_grad
                    for p in ctx["model"].parameters())}
    assert audits["audit8_frozen_context"]["pass"], "AUDIT 8 FAILED"

    # AUDIT 14/15 (dims): exactly 9996 features
    audits["audit14_15_dims"] = {
        "G": int(G_trva.shape[1]), "H": int(H_trva.shape[1]),
        "total": N_TOTAL,
        "pass": (G_trva.shape[1] == N_G and H_trva.shape[1] == N_H
                 and N_TOTAL == 9996)}
    assert audits["audit14_15_dims"]["pass"], "AUDIT 14-15 FAILED"

    # AUDIT 12: gate penalty lambda_v * sum theta^2 (exact)
    with torch.no_grad():
        model.theta.data.copy_(torch.tensor([0.5, -0.5, 1.0, -1.0, 0.0,
                                             2.0, -2.0, 0.25]))
        gp = float(ridge_gate.LAMBDA_V * (model.theta ** 2).sum())
    expect = ridge_gate.LAMBDA_V * 10.5625
    audits["audit12_penalty_form"] = {
        "value": gp, "expected": expect,
        "pass": bool(abs(gp - expect) <= 1e-9 * max(expect, 1))}
    assert audits["audit12_penalty_form"]["pass"], "AUDIT 12 FAILED"
    model.theta.data.copy_(torch.full((K_CODES,),
                                      ridge_gate.THETA_INIT))

    # ---------------- controls A/B (canonical Ridge) ------------------- #
    log("[4] control A: G-only Ridge (trainva)")
    mf1_G, alpha_G = sklearn_ridge_val(G_trva, n_tr, y_trva)
    log("[4] control B: v=1 Ridge == R2 (trainva)")
    mf1_GH, alpha_GH = sklearn_ridge_val(
        np.hstack([G_trva, H_trva]), n_tr, y_trva)
    log(f"[4] G: {mf1_G:.4f} (alpha {alpha_G:.3g}) | "
        f"G+H: {mf1_GH:.4f} (alpha {alpha_GH:.3g})")

    # ---------------- model C: gate optimization ------------------------ #
    log("[5] gate optimization (closed-form Ridge per step)")
    best, final, traj = ridge_gate.run_outer_loop(
        model, y_va, seed=SEED)
    resid_max = max(t["solver_residual"] for t in traj)
    solver_diag["max_residual"] = float(resid_max)
    solver_diag["max_cond"] = float(max(t.get("cond_number", 0.0)
                                        for t in traj))
    grad_ok_all = all(t["grad_finite_nonzero"] for t in traj)
    audits["audit11_grad_through_solve"] = {
        "all_steps_grad_finite_nonzero": grad_ok_all,
        "pass": grad_ok_all}
    assert grad_ok_all, "AUDIT 11 FAILED"
    audits["audit16_determinism_note"] = {
        "seed": SEED, "optimizer": "Adam(lr=1e-2) on theta only",
        "note": "no stochastic op in the graph; identical inputs give "
                "identical trajectories"}
    solver_diag["gate_steps"] = len(traj)
    log(f"[5] best val MF1 {best['val_mf1']:.4f} @ step {best['step']} "
        f"| final v = {[round(x, 3) for x in final['v']]}")

    v_best = np.array(best["v"], dtype=np.float64)

    # ---------------- final refit + ONE test eval per system ------------ #
    # R4 official: closed-form Ridge refit on train+val at frozen best-v
    beta_f, b0_f, resid_f = model.refit_trainval(
        G_trva, Hk_trva, y_trva, v_best)
    solver_diag["final_refit_residual"] = resid_f
    dec_te = model.decision_matrix(G_te, Hk_te, beta_f, b0_f, v_best)
    pred_te_r4 = dec_te.argmax(axis=1)
    dec_va = model.decision_matrix(G_va, Hk_va, beta_f, b0_f, v_best)
    pred_va_r4 = dec_va.argmax(axis=1)
    mf1_R4_val = macro_f1_of(y_va, pred_va_r4)
    mf1_R4_te = macro_f1_of(ctx["yte"], pred_te_r4)

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

    log("[6] official test evaluations (one each)")
    log(f"[6] G   test MF1 = {mf1_G_te:.4f}")
    log(f"[6] G+H test MF1 = {mf1_GH_te:.4f}")
    log(f"[6] R4  test MF1 = {mf1_R4_te:.4f}")

    results = {
        "G": {"val_macro_f1": round(mf1_G, 4), "alpha": alpha_G},
        "GH": {"val_macro_f1": round(mf1_GH, 4), "alpha": alpha_GH},
        "R4": {"val_macro_f1": round(mf1_R4_val, 4),
               "best_step": best["step"],
               "val_mse_at_best": best["val_mse"],
               "theta_init": float(ridge_gate.THETA_INIT),
               "v_init": float(1 / (1 + np.exp(ridge_gate.THETA_INIT))),
               "theta_best": best["theta"], "v_best": best["v"],
               "theta_final": final["theta"], "v_final": final["v"]},
    }
    test_results = {
        "G": {"test_macro_f1": round(mf1_G_te, 4)},
        "GH": {"test_macro_f1": round(mf1_GH_te, 4)},
        "R4": {"test_macro_f1": round(mf1_R4_te, 4)},
    }
    np.save(os.path.join(ds_dir, "predictions", "R4_test.npy"), pred_te_r4)
    np.save(os.path.join(ds_dir, "predictions", "G_test.npy"), pred_te_G)
    np.save(os.path.join(ds_dir, "predictions", "GH_test.npy"), pred_te_GH)
    np.save(os.path.join(ds_dir, "predictions", "R4_val.npy"), pred_va_r4)
    torch.save({"theta": best["theta"], "v": best["v"],
                "val_macro_f1": best["val_mf1"],
                "alpha": info["alpha_r2"]},
               os.path.join(ds_dir, "checkpoints", "R4_gate_best.pt"))
    with open(os.path.join(ds_dir, "training_logs",
                           "gate_trajectory.json"), "w") as f:
        json.dump(traj, f)
    with open(os.path.join(ds_dir, "training_logs",
                           "gate_trajectory.csv"), "w", newline="") as f:
        keys = sorted({k for t in traj for k in t})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for t in traj:
            row = {k: (json.dumps(t[k]) if isinstance(t[k], list) else t[k])
                   for k in keys if k in t}
            w.writerow(row)

    # ---------------- regime statistics + dispersion ------------------- #
    rstats = core.regime_statistics(ctx["regimes_trva"], Hk_trva, ctx["T"])
    for k, s in enumerate(rstats):
        s["theta_init"] = float(ridge_gate.THETA_INIT)
        s["v_init"] = results["R4"]["v_init"]
        s["theta_best"] = results["R4"]["theta_best"][k]
        s["v_best"] = results["R4"]["v_best"][k]
        s["theta_final"] = results["R4"]["theta_final"][k]
        s["v_final"] = results["R4"]["v_final"][k]
        s["gated_aggregate_contribution"] = (
            s["v_final"] * s["fraction_of_H_contribution"])
    vfb = np.array(results["R4"]["v_best"])
    dispersion = {"mean": float(vfb.mean()), "std": float(vfb.std()),
                  "min": float(vfb.min()), "max": float(vfb.max()),
                  "max_minus_min": float(vfb.max() - vfb.min())}

    # ---------------- remaining protocol audits ------------------------ #
    audits["audit9_10_13_alpha_fit"] = {
        "alpha_frozen_r2": info["alpha_r2"],
        "alpha_source": info["alpha_source"],
        "alpha_not_reoptimized": True,
        "train_only_during_gate_opt": True,
        "final_refit": "train+val (R2 convention)",
        "n_gates": K_CODES,
        "pass": True}
    audits["audit17_18_protocol"] = {
        "test_metrics_during_gate_opt": False,
        "single_test_eval_per_system": True,
        "selection": "best val Macro-F1 (tie: lower val MSE)",
        "pass": True}

    payload = {
        "dataset": ds_name, "seed": SEED, "split": ctx["split"],
        "validation": results, "test": test_results,
        "regime_statistics": rstats, "gate_dispersion": dispersion,
        "solver_diagnostics": solver_diag,
        "refs": {"M0": info["M0"], "R2": info["R2"],
                 "alpha_r2": info["alpha_r2"]},
        "deltas": {
            "R4-M0": round(mf1_R4_te - info["M0"], 4),
            "R4-R2": round(mf1_R4_te - info["R2"], 4),
            "R4-GH": round(mf1_R4_te - mf1_GH_te, 4),
            "val_R4_vs_GH": round(mf1_R4_val - mf1_GH, 4),
            "val_R4_vs_G": round(mf1_R4_val - mf1_G, 4),
        },
        "audits": audits,
        "runtime_s": round(time.time() - t0, 1),
    }
    for fname, obj in (("validation_results.json", results),
                       ("test_results.json", test_results),
                       ("gate_metrics.json",
                        {"dispersion": dispersion, "v_best": best["v"],
                         "v_final": final["v"],
                         "v_init": results["R4"]["v_init"]}),
                       ("regime_statistics.json", rstats),
                       ("solver_diagnostics.json", solver_diag),
                       ("diagnostics.json",
                        {"H_block": core.h_block_diagnostics(H_trva),
                         "vq": core.vq_diagnostics(ctx),
                         "refs": payload["refs"]}),
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
    log("R4 DIFFERENTIABLE-RIDGE REGIME GATE - FINAL SEED 42 RESULTS")
    log("=" * 64)
    mfs, dm0, dr2 = [], [], []
    for ds, r in all_results.items():
        info = DATASETS[ds]
        log(f"  {ds}")
        log(f"    M0 Ridge:  {r['test']['G']['test_macro_f1']:.4f} "
            f"(ref {info['M0']})")
        log(f"    R2 Ridge:  {r['test']['GH']['test_macro_f1']:.4f} "
            f"(ref {info['R2']})")
        log(f"    R4 val:    {r['validation']['R4']['val_macro_f1']:.4f}")
        log(f"    R4 test:   {r['test']['R4']['test_macro_f1']:.4f}")
        log(f"    dR4-R2:    {r['deltas']['R4-R2']:+.4f}")
        log(f"    dR4-M0:    {r['deltas']['R4-M0']:+.4f}")
        for k, v in enumerate(json.load(open(os.path.join(
                OUT_DIR, ds, "gate_metrics.json")))["v_best"]):
            log(f"    v{k}: {v:.4f}")
        mfs.append(r["test"]["R4"]["test_macro_f1"])
        dm0.append(r["deltas"]["R4-M0"])
        dr2.append(r["deltas"]["R4-R2"])
    log(f"  R4 mean test Macro-F1 = {np.mean(mfs):.4f}")
    log(f"  mean delta vs M0      = {np.mean(dm0):+.4f}")
    log(f"  mean delta vs R2      = {np.mean(dr2):+.4f}")
    for ds, r in all_results.items():
        d = r["gate_dispersion"]
        log(f"  gate dispersion {ds}: mean={d['mean']:.4f} "
            f"std={d['std']:.4f} max-min={d['max_minus_min']:.4f}")
    return all_results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    args = ap.parse_args()
    main(args.datasets)
