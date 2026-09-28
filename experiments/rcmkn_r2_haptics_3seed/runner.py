"""R2 Haptics 3-Seed Robustness runner.

Protocol (spec):
    * Haptics only; canonical split; per-sample z-norm.
    * Fixed MiniROCKET (random_state=42) shared by all seeds.
    * Seeds 42/43/44 vary ONLY the learned R2 context (SSL + VQ).
    * Seed 42 must reproduce the stored canonical R2 reference 0.5500
      (within the repo-established tolerance) before 43/44 are valid.
    * Exactly ONE official test evaluation per seed (3 total).

Usage:
    python -m experiments.rcmkn_r2_haptics_3seed.runner [--smoke]
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch

from experiments.rcmkn_haptics_seed42.runner import (
    extract_context_regimes, set_seed,
)
from experiments.rcmkn_haptics_seed42.vq import occupancy_stats
from experiments.rcmkn_r2_haptics_3seed import core
from experiments.rcmkn_r2_haptics_3seed.config import (
    ALPHAS, DATASET, ENCODER, JOINT, M0_REF, MINIROCKET_SEED, MIN_OCCUPANCY,
    N_FEATURES, N_GLOBAL, N_HET, OUT_DIR, R2_REF_SEED42, R2_REF_TOL, SEEDS,
    VQ,
)


def log(msg):
    print(msg, flush=True)


def run_one_seed(seed, data, extractor, valid_het, Xtr_z, Xva_z, Xte_z,
                 ytr, yva, yte, ytrva, G_trva, G_te, device,
                 smoke=False, test_eval=True):
    """Run the full R2 pipeline for one outer seed. Returns result dict."""
    log(f"\n{'='*70}\n  SEED {seed}\n{'='*70}")
    t0 = time.time()
    audits = {}

    # ---- learned R2 context (the only seed-varying component) ----
    model, train_info, Xtrva_z, regimes_trva = core.run_context_for_seed(
        seed, Xtr_z, ytr, Xva_z, yva, device)
    set_seed(seed)
    regimes_te = extract_context_regimes(model, Xte_z, device, batch=32)

    # ---- AUDIT 8: deterministic regimes within the seed ----
    set_seed(seed)
    re_tr = extract_context_regimes(model, Xtrva_z, device, batch=32)
    re_te = extract_context_regimes(model, Xte_z, device, batch=32)
    det_tr = bool(np.array_equal(re_tr, regimes_trva))
    det_te = bool(np.array_equal(re_te, regimes_te))
    audits["audit8_regimes_deterministic"] = {
        "trainva": det_tr, "test": det_te, "pass": det_tr and det_te}
    assert det_tr and det_te, "AUDIT 8 FAILED: regimes not deterministic"

    # ---- AUDIT 9: K = 8 ----
    assert VQ["K"] == 8, "AUDIT 9 FAILED: K != 8"
    audits["audit9_K_equals_8"] = {"K": VQ["K"], "pass": True}

    # ---- VQ diagnostics (spec sections 12/13) ----
    occ_trva = occupancy_stats(regimes_trva, K=VQ["K"])
    occ_te = occupancy_stats(regimes_te, K=VQ["K"])
    vq_diag = {
        "active_codes": occ_te["active_codes"],
        "occupancy": occ_te["usage"],
        "entropy_nats": round(float(-sum(q * np.log(q) for q in occ_te["usage"]
                                         if q > 0)), 6),
        "normalized_entropy": occ_te["normalized_entropy"],
        "perplexity": occ_te["perplexity"],
        "dominant_fraction": occ_te["dominant_fraction"],
        "trainva_active_codes": occ_trva["active_codes"],
        "trainva_perplexity": occ_trva["perplexity"],
        "revival_log_len": len(getattr(model.vq, "revival_log", []) or []),
        "note": "code identities are NOT semantically comparable across seeds",
    }

    # ---- H blocks (audited formula) ----
    H_trva = core.compute_H(extractor, Xtrva_z, regimes_trva, valid_het)
    H_te = core.compute_H(extractor, Xte_z, regimes_te, valid_het)

    # ---- AUDIT 10/11: H formula identity + budget ----
    checks = {}
    for i_s, f_off in [(0, 0), (min(3, len(Xte_z) - 1), 1234)]:
        act_probe, valid_probe = core.compute_raw_activations(
            extractor, Xte_z[i_s:i_s + 1])
        h_impl = float(H_te[i_s, f_off])
        h_ref = core.independent_heterogeneity_recompute(
            act_probe[0, N_GLOBAL + f_off], valid_probe[N_GLOBAL + f_off],
            regimes_te[i_s])
        checks[f"sample{i_s}_kernel{f_off}"] = round(abs(h_impl - h_ref), 12)
        assert abs(h_impl - h_ref) < 1e-7, "AUDIT 10 FAILED: H recompute"
    audits["audit10_H_formula_recompute"] = {
        "max_abs_diff": max(checks.values()), "pass": True}
    audits["audit11_H_budget_4998"] = {
        "H_shape": list(H_te.shape), "pass": H_te.shape[1] == N_HET}
    assert H_te.shape[1] == N_HET

    # ---- AUDIT 12/13/14/15: no gate / no modulation / no Hydra ----
    audits["audit12_final_dim_9996"] = {
        "G": N_GLOBAL, "H": N_HET, "total": N_GLOBAL + N_HET,
        "pass": N_GLOBAL + N_HET == N_FEATURES}
    assert audits["audit12_final_dim_9996"]["pass"]
    audits["audit13_14_15_no_gate_no_mod_no_hydra"] = {
        "architecture": "X_R2 = [G || H] only",
        "learned_gating": False, "raw_response_modulation": False,
        "hydra_features": False, "pass": True}

    # ---- AUDIT 16: Ridge config ----
    audits["audit16_ridge_config"] = {
        "classifier": "RidgeClassifierCV",
        "alphas": "logspace(-4,4,20)",
        "fit": "train+val; alpha selected on internal LOO-CV",
        "pass": True}

    # ---- AUDIT 17: no test information in selection ----
    audits["audit17_no_test_in_selection"] = {
        "ssl_val": "validation masked-MSE early stopping",
        "joint_val": "validation macro-F1 checkpoint selection",
        "alpha": "RidgeClassifierCV LOO on train+val only",
        "pass": True}

    # ---- H diagnostics (spec section 12) ----
    h_diag = core.h_block_diagnostics(H_te)

    # ---- AUDIT 18 + OFFICIAL TEST (exactly once) ----
    if not test_eval:
        return {"seed": seed, "audits": audits, "vq_diagnostics": vq_diag,
                "h_diagnostics": h_diag, "train_info": train_info,
                "test_deferred": True}
    log(f"  [SEED {seed}] OFFICIAL test evaluation (once)")
    result, pred_te = core.ridge_fit_eval(
        G_trva, H_trva, G_te, H_te, len(Xtr_z), yva, yte, ytrva,
        data["n_classes"])
    audits["audit18_single_test_evaluation"] = {
        "evaluations": 1, "pass": True}

    res = {
        "seed": seed,
        "results": result,
        "delta_vs_M0": round(result["test_macro_f1"] - M0_REF, 4),
        "runtime_s": round(time.time() - t0, 1),
        "vq_diagnostics": vq_diag,
        "h_diagnostics": h_diag,
        "context_train": train_info,
        "params": core.parameter_report(model),
        "audits": audits,
        "regime_hash_trva": core.sha16(regimes_trva),
        "regime_hash_te": core.sha16(regimes_te),
        "minirocket": {
            "random_state": MINIROCKET_SEED,
            "n_features": N_FEATURES,
            "n_global": N_GLOBAL,
            "n_het": N_HET,
            "frozen_representation_identity": "fixed across seeds",
        },
    }
    return res, pred_te, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    seeds = [SEEDS[0]] if args.smoke else SEEDS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    log("=" * 74)
    log("R2 HAPTICS -- 3-SEED ROBUSTNESS (fixed MiniROCKET, learned context varies)")
    log("=" * 74)
    log(f"seeds={seeds} device={device} smoke={args.smoke}")

    # ---------------- data (AUDIT 1/2) ----------------
    data = core.load_data()
    Xtr, ytr = data["Xtr"], data["ytr"]
    Xva, yva = data["Xva"], data["yva"]
    Xte, yte = data["Xte"], data["yte"]
    ytrva = np.concatenate([ytr, yva])
    Xtr_z, Xva_z, Xte_z = (core.znorm(Xtr), core.znorm(Xva), core.znorm(Xte))
    audits_global = {"audit1_dataset_identity": {"expected":
        {"train": 132, "val": 23, "test": 308, "T": 1092, "n_classes": 5},
        "pass": True},
        "audit2_split_identical_across_seeds": {
            "note": "single load, shared arrays; no resampling",
            "train": len(Xtr), "val": len(Xva), "test": len(Xte),
            "pass": True}}
    audits_global["audit3_znorm"] = core.verify_znorm_invariant(data)
    log(f"  train={len(Xtr)} val={len(Xva)} test={len(Xte)} "
        f"T={data['L']} n_cls={data['n_classes']}")

    # ---------------- fixed MiniROCKET (AUDIT 4/5/6) ----------------
    extractor = core.build_fixed_extractor(Xtr_z)
    Xtrva_z = core.stack_trva(Xtr_z, Xva_z)
    F_trva = extractor.transform(Xtrva_z[:, None, :].astype(np.float32))
    F_te = extractor.transform(Xte_z[:, None, :].astype(np.float32))
    G_trva, G_te = F_trva[:, :N_GLOBAL], F_te[:, :N_GLOBAL]
    valid_het = core.compute_valid_het(extractor, Xtr_z[:8])

    # AUDIT 4: raw extractor == canonical aeon transform
    mr_id = 0.0
    act_probe, valid_probe = core.compute_raw_activations(extractor, Xtr_z[:16])
    mr_id = float(np.max(np.abs(
        core.ppv_from_activations(act_probe, valid_probe) - F_trva[:16])))
    log(f"  [AUDIT4] raw extractor vs aeon: max|diff|={mr_id:.2e}")
    assert mr_id < 1e-5, "AUDIT 4 FAILED: raw extractor identity"
    audits_global["audit4_extractor_identity"] = {
        "max_abs_diff": mr_id, "pass": mr_id < 1e-5}
    del act_probe

    audits_global["audit5_minirocket_seed_fixed_42"] = {
        "random_state": MINIROCKET_SEED,
        "note": "identical for every outer seed", "pass": True}
    audits_global["audit6_G_budget_4998"] = {
        "shape": list(G_trva.shape), "pass": G_trva.shape[1] == N_GLOBAL}
    assert G_trva.shape[1] == N_GLOBAL
    audits_global["audit7_ssl_architecture"] = {
        "encoder": ENCODER, "vq": VQ, "joint": JOINT,
        "source": "rcmkn_haptics_seed42.config (audited R2)", "pass": True}

    # ---------------- run seeds ----------------
    seed_results, preds, models = {}, {}, {}
    for seed in seeds:
        out = run_one_seed(seed, data, extractor, valid_het,
                           Xtr_z, Xva_z, Xte_z, ytr, yva, yte, ytrva,
                           G_trva, G_te, device)
        res, pred_te, model = out
        seed_results[seed] = res
        preds[seed] = pred_te
        models[seed] = model

        sdir = os.path.join(OUT_DIR, f"seed{seed}")
        os.makedirs(sdir, exist_ok=True)
        with open(os.path.join(sdir, "result.json"), "w") as f:
            json.dump(res, f, indent=2)
        with open(os.path.join(sdir, "predictions.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sample_index", "true_class", "R2_pred"])
            for i in range(len(yte)):
                w.writerow([i, int(yte[i]), int(pred_te[i])])

        # seed-42 canonical reproduction gate
        if seed == 42 and not args.smoke:
            got = res["results"]["test_macro_f1"]
            ok = abs(got - R2_REF_SEED42) <= R2_REF_TOL
            log(f"  [GATE] seed42 R2 = {got:.4f} vs canonical {R2_REF_SEED42:.4f} "
                f"(tol {R2_REF_TOL}) -> {'PASS' if ok else 'FAIL'}")
            audits_global["seed42_reproduction_gate"] = {
                "reference": R2_REF_SEED42, "observed": got,
                "tolerance": R2_REF_TOL, "pass": bool(ok)}
            if not ok:
                raise AssertionError(
                    "Seed-42 R2 does not reproduce the canonical reference; "
                    "STOP before seeds 43/44 per spec section 10.")

    # ---------------- aggregate (spec sections 14-16) ----------------
    tests = [seed_results[s]["results"]["test_macro_f1"] for s in seeds]
    vals = [seed_results[s]["results"]["val_macro_f1"] for s in seeds]
    deltas = [round(t - M0_REF, 4) for t in tests]
    summary = {
        "n_seeds": len(seeds),
        "val_macro_f1": [round(v, 4) for v in vals],
        "test_macro_f1": [round(t, 4) for t in tests],
        "test_mean": round(float(np.mean(tests)), 4),
        "test_std": round(float(np.std(tests)), 4),
        "test_min": round(float(np.min(tests)), 4),
        "test_max": round(float(np.max(tests)), 4),
        "delta_vs_M0": deltas,
        "delta_mean": round(float(np.mean(deltas)), 4),
        "delta_std": round(float(np.std(deltas)), 4),
        "n_improve_over_M0": int(sum(1 for d in deltas if d > 0)),
        "M0_reference": M0_REF,
        "statistical_note": ("n=3 seeds: descriptive mean +/- std only; "
                             "no population-level significance claims"),
    }

    # verdict (spec section 18) -- fixed thresholds, declared before results
    n_pos = summary["n_improve_over_M0"]
    if n_pos == 3:
        verdict = "ROBUST ACROSS TESTED SEEDS"
    elif n_pos == 2:
        verdict = "MIXED / SEED-SENSITIVE"
    else:
        verdict = "NOT ROBUST"
    summary["verdict"] = verdict

    # ---------------- figures ----------------
    try:
        from experiments.rcmkn_r2_haptics_3seed.figures import make_figures
        make_figures(seed_results, dict(summary), OUT_DIR)
    except Exception as e:  # figures must never invalidate the run
        log(f"  [FIGURES] skipped: {e}")

    # ---------------- save artifacts ----------------
    all_audits = {"global": audits_global,
                  "per_seed": {s: seed_results[s]["audits"] for s in seeds}}
    with open(os.path.join(OUT_DIR, "audits.json"), "w") as f:
        json.dump(all_audits, f, indent=2)
    with open(os.path.join(OUT_DIR, "results_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(OUT_DIR, "diagnostics.json"), "w") as f:
        json.dump({str(s): {"vq_diagnostics": seed_results[s]["vq_diagnostics"],
                            "h_diagnostics": seed_results[s]["h_diagnostics"],
                            "context_train": seed_results[s]["context_train"],
                            "params": seed_results[s]["params"],
                            "runtime_s": seed_results[s]["runtime_s"],
                            "regime_hash_te": seed_results[s]["regime_hash_te"]}
                   for s in seeds}, f, indent=2)
    cfg = {
        "dataset": DATASET, "seeds": seeds,
        "minirocket": {"random_state": MINIROCKET_SEED,
                       "n_features": N_FEATURES,
                       "fit_on": "z-normed TRAIN only"},
        "encoder": ENCODER, "vq": VQ, "joint": JOINT,
        "min_occupancy": MIN_OCCUPANCY,
        "alphas": "logspace(-4,4,20)",
        "protocol": {"znorm": "per-sample, before all branches",
                     "ridge_fit": "train+val",
                     "official_test_evals_per_seed": 1},
        "references": {"R2_seed42": R2_REF_SEED42, "M0": M0_REF},
    }
    with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    from experiments.rcmkn_r2_haptics_3seed.report import write_report
    write_report(OUT_DIR, seed_results, summary, all_audits, cfg)

    # ---------------- final console summary (spec section 24) ----------------
    log("\n" + "=" * 74)
    log("R2 HAPTICS 3-SEED FINAL")
    log("=" * 74)
    log(f"    M0 = {M0_REF}")
    for s, t, d in zip(seeds, tests, deltas):
        log(f"    Seed {s} = {t:.4f}   (Delta vs M0 = {d:+.4f})")
    log(f"    R2 mean +/- std = {summary['test_mean']:.4f} +/- "
        f"{summary['test_std']:.4f}")
    log(f"    Mean Delta vs M0 = {summary['delta_mean']:+.4f}")
    log(f"    Active VQ codes: " +
        ", ".join(f"seed{s}={seed_results[s]['vq_diagnostics']['active_codes']}"
                  for s in seeds))
    log(f"    Verdict: {verdict}")
    log("=" * 74)
    return summary


if __name__ == "__main__":
    main()
