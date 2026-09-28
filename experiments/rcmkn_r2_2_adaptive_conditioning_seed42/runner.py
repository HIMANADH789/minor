"""R2.2 = R2 + adaptive discrete context-to-kernel conditioning (seed 42).

Datasets: Haptics, CWRU_BAL.  Variants: B0/B1/B2/B3A/B3B (5 test
evaluations per dataset, executed only after all audits pass).
"""

import json
import os
import time

import numpy as np
import torch

from experiments.rcmkn_r2_2_adaptive_conditioning_seed42 import conditioning as cond
from experiments.rcmkn_r2_2_adaptive_conditioning_seed42 import core
from experiments.rcmkn_r2_2_adaptive_conditioning_seed42.core import (
    DATASETS, K_CODES, MIN_OCCUPANCY, N_FEATURES, N_GLOBAL, N_HET, OUT_DIR,
    SEED, VARIANT_ORDER, log, regime_hash, run_variant,
)

from experiments.rcmkn_haptics_seed42.runner import set_seed


def run_dataset(ds_name, device, smoke=False):
    """Full R2.2 pipeline for one dataset; returns result dict."""
    t_start = time.time()
    info = DATASETS[ds_name]
    ds_dir = os.path.join(OUT_DIR, ds_name)
    os.makedirs(ds_dir, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "diagnostics"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "predictions"), exist_ok=True)
    audits = {}
    ctx = core.load_frozen_context(ds_name, device)

    # ---------------- AUDIT 1: dataset identity -------------------------
    got = {"train": ctx["split"]["train"], "val": ctx["split"]["val"],
           "test": ctx["split"]["test"], "T": ctx["T"],
           "n_classes": ctx["n_classes"]}
    assert got == info["expected"], f"AUDIT 1 FAILED: {got} != {info['expected']}"
    audits["audit1_config"] = {"expected": info["expected"], "actual": got,
                               "pass": True}
    log(f"=== {ds_name}: train={got['train']} val={got['val']} "
        f"test={got['test']} T={got['T']} classes={got['n_classes']} ===")

    extractor = core.build_extractor(ctx)
    G_tr, G_va, G_te = core.global_blocks(extractor, ctx)
    ctx["_G_trva"] = G_tr                      # for modulation training

    # ---------------- PASS A: identity audit + histograms ---------------
    h_trva, a_tr = core.pass_a_split(extractor, ctx["Xtrva_z"],
                                     ctx["codes_trva"], "trainva", device)
    h_te, a_te = core.pass_a_split(extractor, ctx["Xte_z"],
                                   ctx["codes_te"], "test", device)
    audits["audit2_margin_kernel_identity"] = {
        "trainva": a_tr, "test": a_te,
        "note": "chunk-0 exact bool equality vs audited extractor"}

    # ---------------- modulation training (train-only, val-selected) ----
    log("  [MOD] training modulation table (frozen context)")
    mod, mod_diag, mod_info = core.train_modulation(
        ds_name, ctx, extractor, device, smoke=smoke)
    audits["modulation_training"] = mod_info

    # ---------------- AUDIT 9/10: identity init + bounds ----------------
    delta = mod.delta().detach().cpu().numpy()
    scale_lo, scale_hi = mod.scale_bounds()
    if smoke:
        # a fresh module must be exactly at identity
        fresh = cond.CodeModulation().delta().detach().numpy()
        audits["audit9_identity_init"] = {
            "fresh_delta_abs_max": float(np.abs(fresh).max()),
            "pass": bool(np.abs(fresh).max() == 0.0)}
    audits["audit10_bounds"] = {
        "delta_abs_max": float(np.abs(delta).max()),
        "scale_min": scale_lo, "scale_max": scale_hi,
        "pass": bool(scale_lo > 0.5 - 1e-6 and scale_hi < 1.5 + 1e-6
                     and scale_lo > 0)}
    assert audits["audit10_bounds"]["pass"], "AUDIT 10 FAILED: bounds"

    tau = mod.tau().detach().cpu().numpy()          # (K, M) base thresholds
    bias_sign = np.sign(np.asarray(
        extractor.parameters[4], dtype=np.float32)[N_GLOBAL:])
    tau_signed = cond.fold_bias_sign(tau, bias_sign)
    tau_zero = np.zeros_like(tau_signed)            # B1: exact identity
    np.save(os.path.join(ds_dir, "tau_table.npy"), tau_signed)
    np.save(os.path.join(ds_dir, "bias_sign.npy"), bias_sign)
    torch.save({"mod_state": mod.state_dict(),
                "diag": mod_diag, "info": mod_info},
               os.path.join(ds_dir, "modulation_seed42.pt"))

    # ---------------- control code arrays (audited constructors) --------
    set_seed(SEED)
    c1_trva = core.create_random_regime_control(ctx["codes_trva"], seed=SEED,
                                                K=K_CODES)
    c1_te = core.create_random_regime_control(ctx["codes_te"], seed=SEED,
                                              K=K_CODES)
    set_seed(SEED)
    c2_trva = core.create_shuffled_regime_control(ctx["codes_trva"], seed=SEED)
    c2_te = core.create_shuffled_regime_control(ctx["codes_te"], seed=SEED)

    # AUDIT 13/14/15: occupancy preservation + distinction
    occ_fail = 0
    for ctrl, src in ((c1_trva, ctx["codes_trva"]), (c1_te, ctx["codes_te"]),
                      (c2_trva, ctx["codes_trva"]), (c2_te, ctx["codes_te"])):
        for i in range(len(src)):
            if not np.array_equal(np.bincount(ctrl[i], minlength=K_CODES),
                                  np.bincount(src[i], minlength=K_CODES)):
                occ_fail += 1
    audits["audit13_14_occupancy"] = {"failures": occ_fail, "pass": occ_fail == 0}
    assert occ_fail == 0, "AUDIT 13/14 FAILED: occupancy"
    audits["audit15_distinction"] = {
        "hash_c1_trva": regime_hash(c1_trva), "hash_c2_trva": regime_hash(c2_trva),
        "hash_b2_trva": regime_hash(ctx["codes_trva"]),
        "c1_vs_c2_diff_frac_trva": float(
            (c1_trva != c2_trva).mean()), "c1_vs_c2_diff_frac_te": float(
            (c1_te != c2_te).mean()),
        "b2_vs_c1_diff_frac_trva": float(
            (ctx["codes_trva"] != c1_trva).mean()),
        "shares_memory_c1_c2": bool(np.shares_memory(c1_trva, c2_trva)),
        "pass": bool((c1_trva != c2_trva).mean() > 0.01
                     and not np.shares_memory(c1_trva, c2_trva))}
    assert audits["audit15_distinction"]["pass"], "AUDIT 15 FAILED"

    # ---------------- PASS B: exact conditioned features ----------------
    log("  [PASS B] exact conditioned heterogeneity (all variants)")
    code_arrays_trva = {"B1": ctx["codes_trva"], "B2": ctx["codes_trva"],
                        "B3A": c1_trva, "B3B": c2_trva}
    code_arrays_te = {"B1": ctx["codes_te"], "B2": ctx["codes_te"],
                      "B3A": c1_te, "B3B": c2_te}
    tau_tables_trva = {"B1": tau_zero, "B2": tau_signed,
                       "B3A": tau_signed, "B3B": tau_signed}
    tau_tables_te = {"B1": tau_zero, "B2": tau_signed,
                     "B3A": tau_signed, "B3B": tau_signed}
    H_trva = core.pass_b_split(extractor, ctx["Xtrva_z"], code_arrays_trva,
                               tau_tables_trva, need_unconditioned=True)
    H_te = core.pass_b_split(extractor, ctx["Xte_z"], code_arrays_te,
                             tau_tables_te, need_unconditioned=True)

    # ---------------- AUDIT 12: B1 == B0 (mandatory) --------------------
    b1_diff = float(np.abs(H_trva["B1"] - H_trva["B0"]).max())
    audits["audit12_b1_equals_b0"] = {
        "trainva_max_abs_diff": b1_diff,
        "test_max_abs_diff": float(np.abs(H_te["B1"] - H_te["B0"]).max()),
        "pass": bool(b1_diff == 0.0)}
    log(f"  [AUDIT12] B1 vs B0 max|diff| = {b1_diff:.2e} "
        f"({'PASS' if b1_diff == 0 else 'FAIL'})")
    assert audits["audit12_b1_equals_b0"]["pass"], "AUDIT 12 FAILED: B1 != B0"

    # ---------------- AUDIT 16: conditioning non-placeholder ------------
    b2_changed = float((H_trva["B2"] != H_trva["B0"]).mean())
    audits["audit16_conditioning_active"] = {
        "fraction_H_entries_changed_vs_B0": round(b2_changed, 6),
        "pass": bool(b2_changed > 0.0)}
    log(f"  [AUDIT16] fraction of H entries changed by B2 vs B0: "
        f"{b2_changed:.4f} ({'active' if b2_changed > 0 else 'INACTIVE'})")

    # ---------------- AUDIT 6/7: H formula + valid region ---------------
    act0, u0, valid0 = cond.compute_activations_and_margins_full(
        extractor, ctx["Xtrva_z"][:2])
    act_ref0, valid_ref = core.compute_raw_activations(
        extractor, ctx["Xtrva_z"][:2])
    assert int(np.count_nonzero(act0 != act_ref0)) == 0
    from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (
        heterogeneity_features)
    H_ind = np.array([core.independent_heterogeneity_recompute(
        act0[i, N_GLOBAL + m], valid0[N_GLOBAL + m],
        ctx["codes_trva"][i], K=K_CODES, min_occupancy=MIN_OCCUPANCY)
        for i in range(2) for m in (0, 77, 2500, 4997)])
    H_impl = np.array([heterogeneity_features(
        act0[i:i+1, N_GLOBAL + m:N_GLOBAL + m + 1],
        valid0[N_GLOBAL + m:N_GLOBAL + m + 1],
        ctx["codes_trva"][i:i+1], K=K_CODES, min_occupancy=MIN_OCCUPANCY)[0, 0]
        for i in range(2) for m in (0, 77, 2500, 4997)])
    audits["audit6_H_formula"] = {
        "max_diff": float(np.abs(H_ind - H_impl).max()), "pass": True}
    # valid-region: corrupt invalid positions of one feature; H unchanged
    act_corr = act0.copy()
    inv_mask = ~valid0[N_GLOBAL + 77]
    if inv_mask.any():
        act_corr[:, N_GLOBAL + 77][:, inv_mask] = \
            ~act_corr[:, N_GLOBAL + 77][:, inv_mask]
        H_before = heterogeneity_features(
            act0[:, N_GLOBAL + 77:N_GLOBAL + 78], valid0[N_GLOBAL + 77:N_GLOBAL + 78],
            ctx["codes_trva"][:2], K=K_CODES)
        H_after = heterogeneity_features(
            act_corr[:, N_GLOBAL + 77:N_GLOBAL + 78],
            valid0[N_GLOBAL + 77:N_GLOBAL + 78], ctx["codes_trva"][:2], K=K_CODES)
        vr_ok = bool(np.array_equal(H_before, H_after))
    else:
        vr_ok = True
    audits["audit7_valid_region"] = {"pass": vr_ok}
    log(f"  [AUDIT6/7] H recompute max|diff|="
        f"{audits['audit6_H_formula']['max_diff']:.2e}; valid-region {vr_ok}")
    del act0, u0, act_ref0, act_corr

    # ---------------- assemble variants + Ridge ------------------------
    n_tr = ctx["n_train"]
    yva, yte, ytrva = ctx["yva"], ctx["yte"], ctx["ytrva"]

    def blocks_for(name):
        if name == "B0":
            Ht, Hv, He = (H_trva["B0"], H_trva["B0"][n_tr:], H_te["B0"])
        elif name == "B1":
            Ht, Hv, He = (H_trva["B1"], H_trva["B1"][n_tr:], H_te["B1"])
        elif name == "B2":
            Ht, Hv, He = (H_trva["B2"], H_trva["B2"][n_tr:], H_te["B2"])
        elif name == "B3A":
            Ht, Hv, He = (H_trva["B3A"], H_trva["B3A"][n_tr:], H_te["B3A"])
        else:
            Ht, Hv, He = (H_trva["B3B"], H_trva["B3B"][n_tr:], H_te["B3B"])
        # note: H_trva rows [n_tr:] ARE the validation rows
        return [(G_tr, G_va, G_te), (Ht, Hv, He)]

    results, preds = {}, {}
    for name in VARIANT_ORDER:
        log(f"  [RIDGE:{name}] fitting on train+val")
        results[name], preds[name] = core.run_variant(
            name, blocks_for(name), yva, yte, ytrva, ctx["n_classes"])
        np.save(os.path.join(OUT_DIR, "predictions",
                             f"{ds_name}_{name}_pred_te.npy"), preds[name])

    # ---------------- diagnostics ---------------------------------------
    from experiments.rcmkn_haptics_seed42.vq import occupancy_stats
    vq_diag = {"codes_trva": occupancy_stats(ctx["codes_trva"], K=K_CODES),
               "codes_te": occupancy_stats(ctx["codes_te"], K=K_CODES)}
    mech = {}
    for name in VARIANT_ORDER:
        h = H_trva[name]
        mech[name] = {"mean_H": float(h.mean()), "median_H": float(np.median(h)),
                      "max_H": float(h.max()),
                      "frac_nonzero": float((h > 0).mean())}

    deltas = {f"B2-{k}": round(results["B2"]["test_macro_f1"]
                               - results[k]["test_macro_f1"], 4)
              for k in ("B0", "B1", "B3A", "B3B")}
    deltas["B3A-B3B"] = round(results["B3A"]["test_macro_f1"]
                              - results["B3B"]["test_macro_f1"], 4)
    deltas["B2-M0_ref"] = round(results["B2"]["test_macro_f1"]
                                - info["M0_ref"], 4)

    ds_result = {
        "dataset": ds_name, "split": ctx["split"],
        "results": results, "deltas": deltas,
        "mechanistic_h": mech, "vq_diagnostics": vq_diag,
        "modulation_diagnostics": mod_diag,
        "modulation_training": mod_info,
        "r2_reference": info["R2_ref"], "r0_reference": info["R0_ref"],
        "m0_reference": info["M0_ref"],
        "audits": audits, "runtime_s": round(time.time() - t_start, 1),
    }
    with open(os.path.join(ds_dir, "result.json"), "w") as f:
        json.dump(ds_result, f, indent=2)
    with open(os.path.join(OUT_DIR, "diagnostics",
                           f"{ds_name}_audits.json"), "w") as f:
        json.dump(audits, f, indent=2)
    # free memmaps
    for p in (h_trva, h_te):
        try:
            os.remove(p)
        except OSError:
            pass
    log(f"  [{ds_name}] done in {ds_result['runtime_s']}s: " +
        " ".join(f"{v}={results[v]['test_macro_f1']:.4f}"
                 for v in VARIANT_ORDER))
    return ds_result


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["Haptics", "CWRU_BAL"])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    log("=" * 74)
    log("R2.2 ADAPTIVE DISCRETE CONTEXT-TO-KERNEL CONDITIONING (seed 42) — "
        + ", ".join(args.datasets))
    log("=" * 74)
    log(f"  device={device} smoke={args.smoke} K={K_CODES} "
        f"beta_max={cond.BETA_MAX} hist_bins={cond.HIST_N_INTERIOR} "
        f"soft_w={cond.SOFT_W} lambda_cls={core.TRAIN['lambda_cls']}")

    all_results = {}
    for ds in args.datasets:
        all_results[ds] = core_run_dataset(ds, device, smoke=args.smoke)

    if not args.smoke:
        per_ds = {ds: {v: r["results"][v]["test_macro_f1"]
                       for v in VARIANT_ORDER}
                  for ds, r in all_results.items()}
        verdicts = {}
        for ds, r in all_results.items():
            res = per_ds[ds]
            if res["B2"] > res["B0"]:
                if res["B2"] > res["B3A"] and res["B2"] > res["B3B"]:
                    margin = min(res["B2"] - res["B3A"], res["B2"] - res["B3B"])
                    verdicts[ds] = ("STRONG R2.2 TRANSFER" if margin >= 0.01
                                    else "PARTIAL")
                else:
                    verdicts[ds] = "NO MECHANISTIC BENEFIT (controls match)"
            elif res["B2"] == res["B0"]:
                verdicts[ds] = "NO MECHANISTIC BENEFIT (B2 == B0)"
            else:
                verdicts[ds] = "NEGATIVE"
            if r["modulation_diagnostics"]["fraction_near_zero_code_vec"] > 0.9:
                verdicts[ds] = "COLLAPSED CONDITIONING"

        report = {
            "title": "R2.2 adaptive discrete context-to-kernel conditioning "
                     "(seed 42)",
            "seed": SEED, "datasets": args.datasets,
            "variants": VARIANT_ORDER,
            "per_dataset_test_macro_f1": per_ds,
            "deltas": {ds: r["deltas"] for ds, r in all_results.items()},
            "verdicts": verdicts,
            "modulation_diagnostics": {ds: r["modulation_diagnostics"]
                                       for ds, r in all_results.items()},
            "mechanistic_h": {ds: r["mechanistic_h"]
                              for ds, r in all_results.items()},
            "vq_diagnostics": {ds: r["vq_diagnostics"]
                               for ds, r in all_results.items()},
            "references": {ds: {"M0": DATASETS[ds]["M0_ref"],
                                "R0": DATASETS[ds]["R0_ref"],
                                "R2": DATASETS[ds]["R2_ref"]}
                           for ds in args.datasets},
            "config": {"beta_max": cond.BETA_MAX,
                       "hist_bins": cond.HIST_N_INTERIOR,
                       "soft_w": cond.SOFT_W,
                       "train": core.TRAIN,
                       "note": "no hyperparameter search; frozen R2 context"},
            "audits_all": {ds: r["audits"] for ds, r in all_results.items()},
        }
        with open(os.path.join(OUT_DIR, "report.json"), "w") as f:
            json.dump(report, f, indent=2)
        with open(os.path.join(OUT_DIR, "per_dataset_results.json"), "w") as f:
            json.dump(all_results, f, indent=2)
        with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
            json.dump(report["config"], f, indent=2)

    log("\n" + "=" * 74)
    log("RESULTS (test Macro-F1, seed 42)")
    log("=" * 74)
    for ds, r in all_results.items():
        res = {v: r["results"][v]["test_macro_f1"] for v in VARIANT_ORDER}
        log(f"  {ds}: " + " ".join(f"{v}={res[v]:.4f}" for v in VARIANT_ORDER)
            + f"  deltas={r['deltas']}")
    return all_results


def core_run_dataset(ds, device, smoke=False):
    """Thin wrapper kept for report naming parity with prior experiments."""
    return run_dataset(ds, device, smoke=smoke)


if __name__ == "__main__":
    main()
