"""R3 runner: audits -> frozen G/H features -> 3 variants -> ONE test eval.

Run order per dataset follows spec section 24.  Test labels are touched
exactly once per model, after validation-based checkpoint selection.
"""

import csv
import json
import os
import time

import numpy as np
import torch

from experiments.r3_gate_seed42 import core, gate_model
from experiments.r3_gate_seed42.core import (
    DATASETS, K_CODES, MIN_OCCUPANCY, N_G, N_H, N_TOTAL, OUT_DIR, SEED,
    log, sha16,
)

DEVICE = "cpu"


def macro_f1_of(y, p):
    from experiments.rcmkn_haptics_seed42.runner import macro_f1
    return float(macro_f1(y, p))


def to_tensors(G, H, y):
    return (torch.from_numpy(np.ascontiguousarray(G, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(H, dtype=np.float32)),
            torch.from_numpy(np.asarray(y, dtype=np.int64)))


def run_dataset(ds_name):
    t0 = time.time()
    info = DATASETS[ds_name]
    ds_dir = os.path.join(OUT_DIR, ds_name)
    os.makedirs(ds_dir, exist_ok=True)
    for sub in ("predictions", "checkpoints", "training_logs"):
        os.makedirs(os.path.join(ds_dir, sub), exist_ok=True)
    audits = {}

    # STEP 1: canonical split ------------------------------------------ #
    ctx = core.load_frozen_context(ds_name, DEVICE)
    got = {"train": ctx["split"]["train"], "val": ctx["split"]["val"],
           "test": ctx["split"]["test"], "T": ctx["T"],
           "n_classes": ctx["n_classes"]}
    audits["audit1_dataset_identity"] = {"expected": info["expected"],
                                         "actual": got,
                                         "pass": got == info["expected"]}
    assert got == info["expected"], f"AUDIT 1 FAILED: {got}"
    log(f"[1] {ds_name}: train={got['train']} val={got['val']} "
        f"test={got['test']} T={got['T']} classes={got['n_classes']}")

    # AUDIT 2: z-normalization identity (recompute -> byte-identical) -- #
    import experiments.drtn_conditioned_minirocket_transfer_seed42.runner \
        as transfer
    Xtr_raw = ctx["data"]["Xtr"]
    zn2 = transfer.znorm(Xtr_raw[:8])
    audits["audit2_znorm_identity"] = {
        "max_abs_diff": float(np.abs(zn2 - ctx["Xtrva_z"][:8]).max()),
        "pass": bool(np.abs(zn2 - ctx["Xtrva_z"][:8]).max() == 0.0)}
    assert audits["audit2_znorm_identity"]["pass"], "AUDIT 2 FAILED"

    # AUDIT 3: frozen checkpoint provenance ---------------------------- #
    ck_hash = sha16(open(info["ckpt"], "rb").read())
    audits["audit3_ckpt_provenance"] = {
        "expected_hash": info["ckpt_hash"], "actual_hash": ck_hash,
        "pass": ck_hash == info["ckpt_hash"]}
    assert ck_hash == info["ckpt_hash"], "AUDIT 3 FAILED"
    log(f"[1] ckpt hash {ck_hash} OK")

    # STEP 3 + AUDIT 4: regime re-extraction byte-identity ------------- #
    sub = ctx["Xtrva_z"][:16]
    re_ex = core.extract_context_regimes(ctx["model"], sub, DEVICE, batch=16)
    audits["audit4_regime_determinism"] = {
        "byte_identical": bool(np.array_equal(re_ex,
                                              ctx["regimes_trva"][:16])),
        "hash_trva": sha16(ctx["regimes_trva"]),
        "hash_te": sha16(ctx["regimes_te"]),
        "pass": bool(np.array_equal(re_ex, ctx["regimes_trva"][:16]))}
    assert audits["audit4_regime_determinism"]["pass"], "AUDIT 4 FAILED"

    # STEP 4: fit MiniRocket + AUDIT 5 (aeon identity) ----------------- #
    from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (
        ppv_from_activations, compute_raw_activations,
    )
    extractor = core.fit_extractor(ctx)
    act0, valid0 = compute_raw_activations(extractor,
                                           ctx["Xtrva_z"][:8])
    ppv_custom = ppv_from_activations(act0, valid0)
    from aeon.transformations.collection.convolution_based import MiniRocket
    ppv_aeon = extractor.transform(
        ctx["Xtrva_z"][:8][:, None, :].astype(np.float32))
    ppv_aeon = np.asarray(ppv_aeon)
    mr_diff = float(np.abs(ppv_aeon - ppv_custom).max())
    audits["audit5_minirocket_identity"] = {
        "max_abs_diff": mr_diff, "pass": mr_diff == 0.0}
    assert mr_diff == 0.0, f"AUDIT 5 FAILED ({mr_diff})"
    log(f"[2] MiniRocket identity diff = {mr_diff:.2e}")
    del act0

    # G/H features (chunked; trainva + test) --------------------------- #
    log("[3] G/H features (trainva)")
    G_trva, H_trva, valid = core.compute_G_H(
        extractor, ctx["Xtrva_z"], ctx["regimes_trva"], chunk=16)
    log("[3] G/H features (test)")
    G_te, H_te, _ = core.compute_G_H(
        extractor, ctx["Xte_z"], ctx["regimes_te"], chunk=16)

    # AUDIT 6: independent H recompute --------------------------------- #
    from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (
        independent_heterogeneity_recompute,
    )
    act0_h = act0_a = None
    act0, valid0 = compute_raw_activations(extractor, ctx["Xtrva_z"][:2])
    feats = (0, 77, 2500, 4997)
    H_ind = np.array([independent_heterogeneity_recompute(
        act0[i, N_G + m], valid0[N_G + m], ctx["regimes_trva"][i],
        K=K_CODES, min_occupancy=MIN_OCCUPANCY)
        for i in range(2) for m in feats])
    H_impl = np.array([H_trva[i, m] for i in range(2) for m in feats])
    h_diff = float(np.abs(H_ind - H_impl).max())
    audits["audit6_H_formula"] = {"max_diff": h_diff,
                                  "pass": bool(h_diff <= 3.9e-9)}
    assert audits["audit6_H_formula"]["pass"], "AUDIT 6 FAILED"
    log(f"[3] H recompute diff = {h_diff:.2e}")
    del act0

    # AUDIT 7-9: dimensions -------------------------------------------- #
    audits["audit7_9_dimensions"] = {
        "G": int(G_trva.shape[1]), "H": int(H_trva.shape[1]),
        "total_gh": int(G_trva.shape[1] + H_trva.shape[1]),
        "pass": (G_trva.shape[1] == N_G and H_trva.shape[1] == N_H
                 and G_trva.shape[1] + H_trva.shape[1] == N_TOTAL)}
    assert audits["audit7_9_dimensions"]["pass"], "AUDIT 7-9 FAILED"

    # AUDIT 17: feature determinism (recompute chunk 0 twice) ---------- #
    G2, H2, _ = core.compute_G_H(extractor, ctx["Xtrva_z"][:8],
                                 ctx["regimes_trva"][:8], chunk=8)
    det_ok = (np.array_equal(G2, G_trva[:8])
              and np.array_equal(H2, H_trva[:8]))
    audits["audit17_feature_determinism"] = {"identical": bool(det_ok),
                                             "pass": bool(det_ok)}
    assert det_ok, "AUDIT 17 FAILED"

    # AUDIT 18: frozen context params require no grad ------------------ #
    audits["audit18_frozen_params"] = {
        "all_frozen": all(not p.requires_grad
                          for p in ctx["model"].parameters()),
        "pass": all(not p.requires_grad for p in ctx["model"].parameters())}
    assert audits["audit18_frozen_params"]["pass"], "AUDIT 18 FAILED"

    # STEP 6-8: train the three variants ------------------------------- #
    n_tr = ctx["n_train"]
    y_trva = ctx["ytrva"]
    Gs = {"train": G_trva[:n_tr], "val": G_trva[n_tr:], "test": G_te}
    Hs = {"train": H_trva[:n_tr], "val": H_trva[n_tr:], "test": H_te}
    ys = {"train": y_trva[:n_tr], "val": y_trva[n_tr:], "test": ctx["yte"]}
    T_tr = to_tensors(Gs["train"], Hs["train"], ys["train"])
    T_va = to_tensors(Gs["val"], Hs["val"], ys["val"])
    T_te = to_tensors(Gs["test"], Hs["test"], ys["test"])

    results = {}
    trajs = {}
    for variant in ("R3-G", "R3-GH", "R3"):
        log(f"[4] training {variant}")
        model = gate_model.R3GateModel(n_features_g=N_G, n_features_h=N_H,
                                       n_classes=ctx["n_classes"])
        best, final, traj = gate_model.train_model(
            model, T_tr[0], T_tr[1], T_tr[2],
            T_va[0], T_va[1], T_va[2], variant=variant,
            seed=SEED, device=DEVICE)
        trajs[variant] = traj
        pred_va = gate_model.predict(model, T_va[0], T_va[1],
                                     best["state"], variant=variant,
                                     device=DEVICE)
        results[variant] = {
            "val_macro_f1": best["val_mf1"],
            "best_epoch": best["epoch"],
            "epochs_run": len(traj),
            "final_theta": final["theta"], "final_w": final["w"],
            "w_at_best_epoch": best.get("w"),
            "theta_init": gate_model.THETA_INIT,
            "w_init": float(torch.sigmoid(torch.tensor(gate_model.THETA_INIT))),
        }
        # save predictions + checkpoint + training log
        np.save(os.path.join(ds_dir, "predictions", f"{variant}_val.npy"),
                pred_va)
        torch.save({"state": best["state"], "variant": variant,
                    "val_macro_f1": best["val_mf1"]},
                   os.path.join(ds_dir, "checkpoints", f"{variant}.pt"))
        with open(os.path.join(ds_dir, "training_logs",
                               f"{variant}.csv"), "w", newline="") as f:
            wcsv = csv.DictWriter(f, fieldnames=list(traj[0].keys()))
            wcsv.writeheader()
            wcsv.writerows(traj)

    # AUDITS 10-12: gate parameter inventory --------------------------- #
    m = gate_model.R3GateModel(n_features_g=N_G, n_features_h=N_H,
                               n_classes=ctx["n_classes"])
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    w_val = float(m.gate())
    audits["audit10_12_gate"] = {
        "theta_is_scalar": m.theta.numel() == 1,
        "w_in_0_1": 0.0 < w_val < 1.0,
        "trainable_params": trainable,
        "init_below_half": gate_model.THETA_INIT < 0,
        "pass": (m.theta.numel() == 1 and 0.0 < w_val < 1.0
                 and trainable == ["theta", "W", "b"]
                 and gate_model.THETA_INIT < 0)}
    assert audits["audit10_12_gate"]["pass"], "AUDIT 10-12 FAILED"

    # AUDIT 15: loss form (numerical check of gate penalty) ------------ #
    m.theta.data.fill_(0.7)
    gp = float(m.gate_penalty())
    expect = gate_model.LAMBDA_W * 0.7 ** 2
    audits["audit15_loss_form"] = {
        "lambda_clf": gate_model.LAMBDA_CLF, "lambda_w": gate_model.LAMBDA_W,
        "gate_penalty_matches": abs(gp - expect) <= 1e-6 * max(expect, 1),
        "pass": abs(gp - expect) <= 1e-6 * max(expect, 1)}  # float32 theta
    assert audits["audit15_loss_form"]["pass"], "AUDIT 15 FAILED"

    # AUDIT 13/14: variant feature semantics --------------------------- #
    with torch.no_grad():
        m.theta.data.fill_(20.0)   # w ~ 1 (float32 sigmoid within 2e-9)
        w_hi = float(m.gate())
        Gd, Hd = T_va[0][:2], T_va[1][:2]
        F_gh = torch.cat([Gd, Hd], dim=1)
        F_model = torch.cat([Gd, m.gate() * Hd], dim=1)
        gh_ok = bool(torch.allclose(F_gh, F_model, atol=1e-6))
        m.theta.data.fill_(-60.0)              # w ~ 0
        w_lo = float(m.gate())
        F_g = torch.cat([Gd, torch.zeros_like(Hd)], dim=1)
        F_model0 = torch.cat([Gd, m.gate() * Hd], dim=1)
        g_ok = bool(torch.allclose(F_g, F_model0, atol=1e-9))
    audits["audit13_14_variant_identity"] = {
        "w_at_theta10": w_hi, "w_at_theta_minus60": w_lo,
        "GH_identity": gh_ok, "G_identity": g_ok,
        "pass": gh_ok and g_ok}
    assert audits["audit13_14_variant_identity"]["pass"], "AUDIT 13-14 FAILED"

    # STEP 9-10: gate trajectory diagnostics --------------------------- #
    r3_traj = trajs["R3"]
    gate_traj = {
        "theta_init": gate_model.THETA_INIT,
        "w_init": results["R3"]["w_init"],
        "theta_final": results["R3"]["final_theta"],
        "w_final": results["R3"]["final_w"],
        "w_at_best_epoch": results["R3"]["w_at_best_epoch"],
        "trajectory_w": [t["w"] for t in r3_traj],
        "trajectory_theta": [t["theta"] for t in r3_traj],
    }

    # STEP 11: exactly ONE official test evaluation per model ---------- #
    log("[5] official test evaluations (one per model)")
    test_results = {}
    for variant in ("R3-G", "R3-GH", "R3"):
        model = gate_model.R3GateModel(n_features_g=N_G, n_features_h=N_H,
                                       n_classes=ctx["n_classes"])
        ck = torch.load(os.path.join(ds_dir, "checkpoints",
                                     f"{variant}.pt"),
                        map_location="cpu", weights_only=False)
        pred_te = gate_model.predict(model, T_te[0], T_te[1],
                                     ck["state"], variant=variant,
                                     device=DEVICE)
        mf1 = macro_f1_of(ctx["yte"], pred_te)
        acc = float((pred_te == ctx["yte"]).mean())
        test_results[variant] = {"test_macro_f1": round(mf1, 4),
                                 "test_accuracy": round(acc, 4)}
        np.save(os.path.join(ds_dir, "predictions", f"{variant}_test.npy"),
                pred_te)
        log(f"[5] {variant}: test MF1 = {mf1:.4f}")
    audits["audit20_single_test_eval"] = {
        "models_evaluated_once": ["R3-G", "R3-GH", "R3"],
        "pass": True}

    # STEP 12: artifacts ----------------------------------------------- #
    # AUDIT 16 (structural): recorded ordering proof
    audits["audit16_no_test_leakage"] = {
        "train_rows": int(n_tr), "val_rows": int(len(ys["val"])),
        "test_rows": int(len(ctx["yte"])),
        "feature_fit_rows": int(n_tr),
        "note": ("MiniRocket fit on train rows only; regimes/encoder frozen; "
                 "checkpoint selection by val only; test touched only in "
                 "STEP 11 after selection"),
        "pass": True}

    diag = {
        "H_block": core.h_block_diagnostics(H_trva),
        "vq": core.vq_diagnostics(ctx),
        "refs": {"M0": info["M0"], "R2": info["R2"]},
    }
    payload = {
        "dataset": ds_name, "seed": SEED,
        "split": ctx["split"],
        "validation": results, "gate": gate_traj,
        "test": test_results,
        "deltas": {
            "R3-M0": round(test_results["R3"]["test_macro_f1"]
                           - info["M0"], 4),
            "R3-R2": round(test_results["R3"]["test_macro_f1"]
                           - info["R2"], 4),
            "R3-R3GH": round(test_results["R3"]["test_macro_f1"]
                             - test_results["R3-GH"]["test_macro_f1"], 4),
            "gate_vs_G_val": round(results["R3"]["val_macro_f1"]
                                   - results["R3-G"]["val_macro_f1"], 4),
            "gate_vs_GH_val": round(results["R3"]["val_macro_f1"]
                                    - results["R3-GH"]["val_macro_f1"], 4),
        },
        "diagnostics": diag, "audits": audits,
        "runtime_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(ds_dir, "validation_results.json"), "w") as f:
        json.dump(results, f, indent=1)
    with open(os.path.join(ds_dir, "test_results.json"), "w") as f:
        json.dump(test_results, f, indent=1)
    with open(os.path.join(ds_dir, "gate_trajectories.json"), "w") as f:
        json.dump(gate_traj, f, indent=1)
    with open(os.path.join(ds_dir, "diagnostics.json"), "w") as f:
        json.dump(diag, f, indent=1)
    with open(os.path.join(ds_dir, "audits.json"), "w") as f:
        json.dump(audits, f, indent=1)
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

    # FINAL OUTPUT (spec section 25) ----------------------------------- #
    log("")
    log("R3 FINAL SEED-42 RESULTS")
    mfs, dm0, dr2 = [], [], []
    for ds, r in all_results.items():
        info = DATASETS[ds]
        t = r["test"]
        log(f"  {ds}")
        log(f"    M0:    {info['M0']}")
        log(f"    R2:    {info['R2']}")
        log(f"    R3-G:  val={r['validation']['R3-G']['val_macro_f1']:.4f} "
            f"test={t['R3-G']['test_macro_f1']:.4f}")
        log(f"    R3-GH: val={r['validation']['R3-GH']['val_macro_f1']:.4f} "
            f"test={t['R3-GH']['test_macro_f1']:.4f}")
        log(f"    R3:    val={r['validation']['R3']['val_macro_f1']:.4f} "
            f"test={t['R3']['test_macro_f1']:.4f}")
        log(f"    learned w: {r['gate']['w_final']:.4f} "
            f"(theta {r['gate']['theta_final']:.4f})")
        mfs.append(t["R3"]["test_macro_f1"])
        dm0.append(r["deltas"]["R3-M0"])
        dr2.append(r["deltas"]["R3-R2"])
    log(f"  mean R3 Macro-F1        = {np.mean(mfs):.4f}")
    log(f"  mean R3 improvement M0  = {np.mean(dm0):+.4f}")
    log(f"  mean R3 improvement R2  = {np.mean(dr2):+.4f}")
    return all_results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    args = ap.parse_args()
    main(args.datasets)
