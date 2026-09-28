"""
Corrected negative-dataset re-test (seed 42): DRTN-conditioned MiniROCKET
on ECG5000_UNBAL and CWRU_UNBAL with the AUDITED Stage A implementation.

Reuses, without modification, the audited pieces:
    * M2/M3 controls + valid-region heterogeneity from
      experiments/drtn_conditioned_minirocket_haptics_3seed/runner.py
    * raw activation extractor + canonical PPV from
      experiments/drtn_conditioned_minirocket_transfer_seed42/core.py
    * canonical dataset loaders + DRTN training/protocol from
      experiments/drtn_conditioned_minirocket_transfer_seed42/runner.py
      (z-normalization BEFORE MiniRocket -- the canonical benchmark
      convention that reproduces the baseline_bench M0 values exactly)

Old (pre-audit, INVALID) seed-42 results under re-test:
    ECG5000_UNBAL: M0 0.5938, M1 0.5859, M2 0.5716, M3 0.5565
    CWRU_UNBAL:    M0 0.9917, M1 0.9792, M2 0.9625, M3 0.9583

Canonical M0 seed-42 references (results/baseline_bench):
    ECG5000_UNBAL 0.5938, CWRU_UNBAL 0.9917 (tolerance 0.0011)

Usage:
    python -m experiments.drtn_conditioned_minirocket_corrected_negative_retest.runner \
        [--datasets ECG5000_UNBAL CWRU_UNBAL] [--smoke]
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.drtn_conditioned_minirocket_haptics_3seed.runner import (  # noqa: E402
    create_random_regime_control,     # audited: per-sample occupancy, RNG seed+900001
    create_shuffled_regime_control,   # audited: per-sample permutation, RNG seed+900002
    compute_regime_heterogeneity,     # audited: valid-region H_m
    compute_regime_occupancy_stats,
    M2_RNG_OFFSET,
    M3_RNG_OFFSET,
)
from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (  # noqa: E402
    compute_raw_activations,
    ppv_from_activations,
    independent_heterogeneity_recompute,
)
import experiments.drtn_conditioned_minirocket_transfer_seed42.runner as transfer  # noqa: E402

SEED = 42
N_FEATURES = 9996
N_GLOBAL = N_FEATURES // 2
N_HETEROGENEITY = N_FEATURES - N_GLOBAL
ALPHAS = np.logspace(-4, 4, 20)
K_CODES = 8

CANONICAL_M0_REF = {
    "ECG5000_UNBAL": 0.5938,
    "CWRU_UNBAL": 0.9917,
}
M0_TOLERANCE = 0.0011

OLD_PREAUDIT = {
    "ECG5000_UNBAL": {"M0": 0.5938, "M1": 0.5859, "M2": 0.5716, "M3": 0.5565},
    "CWRU_UNBAL": {"M0": 0.9917, "M1": 0.9792, "M2": 0.9625, "M3": 0.9583},
}

DATASETS = ["ECG5000_UNBAL", "CWRU_UNBAL"]
OUT_DIR = os.path.join(ROOT, "results",
                       "drtn_conditioned_minirocket_corrected_negative_retest")


def arr_hash(a):
    return hashlib.sha1(np.ascontiguousarray(a, dtype=np.int64).tobytes()).hexdigest()[:16]


def macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def run_dataset(ds_name, device, smoke=False):
    print(f"\n{'='*74}\nDATASET {ds_name} (corrected implementation, seed {SEED})\n{'='*74}",
          flush=True)
    ds_dir = os.path.join(OUT_DIR, ds_name)
    diag_dir = os.path.join(ds_dir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)

    # ---------------- canonical data + frozen DRTN (audited protocol) ------
    data = transfer.load_any_dataset(ds_name)
    T = data["L"]
    Xtr, Xva, Xte = data["Xtr"], data["Xva"], data["Xte"]
    ytr, yva, yte = data["ytr"], data["yva"], data["yte"]
    ytrva = np.concatenate([ytr, yva])
    print(f"  train={len(Xtr)} val={len(Xva)} test={len(Xte)} T={T} "
          f"n_cls={data['n_classes']} val_source={data['val_source']}")

    Xtr_z, Xva_z, Xte_z = transfer.znorm(Xtr), transfer.znorm(Xva), transfer.znorm(Xte)

    drtn, drtn_info = transfer.train_or_load_drtn(ds_name, data, device, smoke=smoke)
    ck_path = transfer.drtn_checkpoint_path(ds_name)
    ck_sha = hashlib.sha1(open(ck_path, "rb").read()).hexdigest()[:16] \
        if os.path.exists(ck_path) else None

    regimes_tr = transfer.extract_drtn_regimes(drtn, Xtr_z, device=device)
    regimes_va = transfer.extract_drtn_regimes(drtn, Xva_z, device=device)
    regimes_te = transfer.extract_drtn_regimes(drtn, Xte_z, device=device)
    assert regimes_tr.shape == (len(Xtr), T)
    assert regimes_te.min() >= 0 and regimes_te.max() < K_CODES

    sd1 = {k: v.clone() for k, v in drtn.state_dict().items()}
    _ = transfer.extract_drtn_regimes(drtn, Xte_z[:2], device=device)
    drtn_frozen = all(torch.equal(sd1[k], drtn.state_dict()[k]) for k in sd1)
    assert drtn_frozen, "DRTN parameters changed during regime extraction"

    regime_diag = {
        "train": compute_regime_occupancy_stats(regimes_tr, K=K_CODES),
        "val": compute_regime_occupancy_stats(regimes_va, K=K_CODES),
        "test": compute_regime_occupancy_stats(regimes_te, K=K_CODES),
        "drtn": drtn_info, "drtn_checkpoint_sha1": ck_sha,
        "drtn_frozen_verified": bool(drtn_frozen),
    }

    # ---------------- canonical MiniROCKET (z-normalized input) -------------
    from aeon.transformations.collection.convolution_based import MiniRocket
    extractor = MiniRocket(random_state=SEED, n_jobs=-1)
    extractor.fit(Xtr_z[:, None, :].astype(np.float32))
    Ftr = extractor.transform(Xtr_z[:, None, :].astype(np.float32))
    Fva = extractor.transform(Xva_z[:, None, :].astype(np.float32))
    Fte = extractor.transform(Xte_z[:, None, :].astype(np.float32))
    Ftrva = extractor.transform(
        np.vstack([Xtr_z, Xva_z])[:, None, :].astype(np.float32))
    assert Ftr.shape[1] == N_FEATURES

    # ---------------- raw activations (valid-region convention) -------------
    act_tr, valid = compute_raw_activations(extractor, Xtr_z)
    act_va, _ = compute_raw_activations(extractor, Xva_z)
    act_te, _ = compute_raw_activations(extractor, Xte_z)
    act_trva, _ = compute_raw_activations(extractor, np.vstack([Xtr_z, Xva_z]))

    # ================= AUDIT 1: canonical M0 feature identity ==============
    PPV_trva = ppv_from_activations(act_trva, valid)
    mr_identity = float(np.max(np.abs(PPV_trva - Ftrva)))
    print(f"  [AUDIT1] raw-extractor PPV vs aeon: max|diff|={mr_identity:.2e}")
    assert mr_identity < 1e-5, "raw extractor disagrees with canonical aeon"

    # feature-axis slicing (BUG GUARD 1) -- print actual index ranges
    print(f"  [AUDIT3] global block = features [0, {N_GLOBAL}), "
          f"heterogeneity block = features [{N_GLOBAL}, {N_FEATURES})")
    act_g = lambda a: a[:, :N_GLOBAL, :]      # feature axis (axis 1)
    act_h = lambda a: a[:, N_GLOBAL:, :]
    valid_het = valid[N_GLOBAL:]

    # ================= AUDIT 5: M2/M3 regime arrays =========================
    regimes_trva = np.vstack([regimes_tr, regimes_va])
    m2_trva = create_random_regime_control(regimes_trva, seed=SEED)
    m2_va = create_random_regime_control(regimes_va, seed=SEED)
    m2_te = create_random_regime_control(regimes_te, seed=SEED)
    m3_trva = create_shuffled_regime_control(regimes_trva, seed=SEED)
    m3_va = create_shuffled_regime_control(regimes_va, seed=SEED)
    m3_te = create_shuffled_regime_control(regimes_te, seed=SEED)

    for nm, a, b in [("trva", m2_trva, m3_trva), ("va", m2_va, m3_va),
                     ("te", m2_te, m3_te)]:
        assert not np.array_equal(a, b), f"M2 == M3 arrays on {nm}!"
        assert not np.shares_memory(a, b), f"M2/M3 share memory on {nm}!"
    # ================= AUDIT 4: per-sample occupancy preservation ===========
    occ_fail = 0
    for ctrl, src in [(m2_trva, regimes_trva), (m3_trva, regimes_trva),
                      (m2_va, regimes_va), (m3_va, regimes_va),
                      (m2_te, regimes_te), (m3_te, regimes_te)]:
        for i in range(len(src)):
            if not np.array_equal(np.bincount(ctrl[i], minlength=K_CODES),
                                  np.bincount(src[i], minlength=K_CODES)):
                occ_fail += 1
    assert occ_fail == 0, f"occupancy preservation failed on {occ_fail} samples"
    print(f"  [AUDIT4] per-sample occupancy preserved: all splits, 0 failures")

    m2m3_diff = {
        "m2_hash": arr_hash(m2_te), "m3_hash": arr_hash(m3_te),
        "n_positions": int(m2_te.size),
        "n_different_positions": int((m2_te != m3_te).sum()),
        "fraction_different": round(float((m2_te != m3_te).mean()), 6),
        "shared_memory": bool(np.shares_memory(m2_te, m3_te)),
    }
    print(f"  [AUDIT5] M2 hash {m2m3_diff['m2_hash']} vs M3 hash "
          f"{m2m3_diff['m3_hash']}; differing positions "
          f"{m2m3_diff['n_different_positions']}/{m2m3_diff['n_positions']}")

    # ================= AUDIT 2 + 6 + 7: heterogeneity path ==================
    H1_trva = compute_regime_heterogeneity(
        act_h(act_trva), valid_het, regimes_trva)
    H1_te = compute_regime_heterogeneity(act_h(act_te), valid_het, regimes_te)
    H2_trva = compute_regime_heterogeneity(act_h(act_trva), valid_het, m2_trva)
    H2_te = compute_regime_heterogeneity(act_h(act_te), valid_het, m2_te)
    H3_trva = compute_regime_heterogeneity(act_h(act_trva), valid_het, m3_trva)
    H3_te = compute_regime_heterogeneity(act_h(act_te), valid_het, m3_te)

    recompute_checks = {}
    for i_s, f_off in [(0, 0), (min(5, len(act_te) - 1), 1234)]:
        h_impl = float(H1_te[i_s, f_off])
        h_ref = independent_heterogeneity_recompute(
            act_h(act_te)[i_s, f_off], valid_het[f_off], regimes_te[i_s])
        recompute_checks[f"sample{i_s}_kernel{f_off}"] = {
            "implemented": h_impl, "independent": h_ref,
            "abs_diff": abs(h_impl - h_ref)}
        assert abs(h_impl - h_ref) < 1e-7, f"H_m mismatch at ({i_s},{f_off})"
    print(f"  [AUDIT2/7] independent H_m recompute: "
          f"max diff {max(v['abs_diff'] for v in recompute_checks.values()):.2e}")

    # valid-region invariance (AUDIT 6): per-feature padding. A global
    # position can be padded for one dilation group and VALID for another
    # (full-validity features read all T), so the correct statement is
    # per-feature: flipping activations OUTSIDE each feature's own valid
    # mask must never change H_m.
    act_te_corrupt = act_te.copy()
    het_act = act_te_corrupt[:, N_GLOBAL:, :]    # (N, F_het, T) view
    outside = np.broadcast_to(~valid_het[None, :, :], het_act.shape)
    het_act[outside] = ~het_act[outside]         # flip every out-of-valid cell
    n_flipped = int(outside.sum()) * int(act_te.shape[0])
    if n_flipped:
        H1_corrupt = compute_regime_heterogeneity(
            act_h(act_te_corrupt), valid_het, regimes_te)
        vr_invariance = bool(np.array_equal(H1_te, H1_corrupt))
        assert vr_invariance, "padded positions influenced H_m (AUDIT 6 failure)"
    else:
        vr_invariance = True   # all features fully valid at this T
    print(f"  [AUDIT6] valid-region invariance: flipped {n_flipped} "
          f"out-of-valid activations -> H unchanged: {vr_invariance}")

    # ================= AUDIT 3: feature budget / allocation =================
    M0_tr, M0_va, M0_te = Ftrva, Fva, Fte
    M1_tr = np.hstack([Ftrva[:, :N_GLOBAL], H1_trva])
    M1_va = np.hstack([Fva[:, :N_GLOBAL],
                       compute_regime_heterogeneity(act_h(act_va), valid_het, regimes_va)])
    M1_te = np.hstack([Fte[:, :N_GLOBAL], H1_te])
    M2_tr = np.hstack([Ftrva[:, :N_GLOBAL], H2_trva])
    M2_va = np.hstack([Fva[:, :N_GLOBAL],
                       compute_regime_heterogeneity(act_h(act_va), valid_het, m2_va)])
    M2_te = np.hstack([Fte[:, :N_GLOBAL], H2_te])
    M3_tr = np.hstack([Ftrva[:, :N_GLOBAL], H3_trva])
    M3_va = np.hstack([Fva[:, :N_GLOBAL],
                       compute_regime_heterogeneity(act_h(act_va), valid_het, m3_va)])
    M3_te = np.hstack([Fte[:, :N_GLOBAL], H3_te])

    feature_diffs = {
        f"H{a}vsH{b}": {
            "max_abs_diff": float(np.abs(x - y).max()),
            "mean_abs_diff": float(np.abs(x - y).mean()),
            "n_exactly_equal": int((x == y).sum()), "n_elements": int(x.size)}
        for a, b, x, y in [("1", "2", H1_te, H2_te), ("1", "3", H1_te, H3_te),
                           ("2", "3", H2_te, H3_te)]
    }
    assert not np.array_equal(H2_te, H3_te), "M2/M3 features identical"
    assert (H1_te != 0).mean() > 0, "M1 heterogeneity all zeros"

    for nm, Mt, Mv, Me in [("M0", M0_tr, M0_va, M0_te), ("M1", M1_tr, M1_va, M1_te),
                           ("M2", M2_tr, M2_va, M2_te), ("M3", M3_tr, M3_va, M3_te)]:
        assert Mt.shape[1] == N_FEATURES and Mv.shape[1] == N_FEATURES \
            and Me.shape[1] == N_FEATURES, f"{nm} budget"
        assert np.array_equal(Me[:, :N_GLOBAL], M0_te[:, :N_GLOBAL]), f"{nm} global"
    assert np.allclose(M0_tr[:, :N_GLOBAL], Ftrva[:, :N_GLOBAL])
    print(f"  [AUDIT3] budgets: M0/M1/M2/M3 all 9996 = 4998 + 4998; "
          f"global blocks identical to M0")

    # ================= classifier: train+val fit, test once =================
    results, preds = {}, {}
    for name, F_tr, F_va, F_te_v in [("M0", M0_tr, M0_va, M0_te),
                                     ("M1", M1_tr, M1_va, M1_te),
                                     ("M2", M2_tr, M2_va, M2_te),
                                     ("M3", M3_tr, M3_va, M3_te)]:
        ridge = RidgeClassifierCV(alphas=ALPHAS)
        ridge.fit(F_tr, ytrva)               # AUDIT 8: train + validation
        pred_va = ridge.predict(F_va)
        pred_te = ridge.predict(F_te_v)      # same variant's test features
        results[name] = {
            "val_macro_f1": round(macro_f1(yva, pred_va), 4),
            "test_macro_f1": round(macro_f1(yte, pred_te), 4),
            "accuracy": round(float(accuracy_score(yte, pred_te)), 4),
            "selected_alpha": float(ridge.alpha_),
            "class_f1s": [round(float(x), 4) for x in f1_score(
                yte, pred_te, average=None, zero_division=0,
                labels=list(range(data["n_classes"])))],
        }
        preds[name] = pred_te
        print(f"    {name}: val={results[name]['val_macro_f1']:.4f} "
              f"test={results[name]['test_macro_f1']:.4f} "
              f"alpha={ridge.alpha_:.6f}", flush=True)

    # ---- canonical M0 reproduction gate ----
    m0_ref = CANONICAL_M0_REF[ds_name]
    m0_ok = abs(results["M0"]["test_macro_f1"] - m0_ref) < M0_TOLERANCE
    print(f"  [GATE] M0 = {results['M0']['test_macro_f1']:.4f} vs canonical "
          f"{m0_ref} -> {'PASS' if m0_ok else 'FAIL'}")
    assert m0_ok, f"M0 failed to reproduce canonical reference for {ds_name}"

    deltas = {f"M{a}_M{b}": round(results[f"M{a}"]["test_macro_f1"]
                                  - results[f"M{b}"]["test_macro_f1"], 4)
              for a, b in [(1, 0), (1, 2), (1, 3), (2, 3)]}

    m0c = preds["M0"] == yte
    m1c = preds["M1"] == yte
    complementarity = {
        "both_correct": int((m0c & m1c).sum()),
        "M0_only": int((m0c & ~m1c).sum()),
        "M1_only": int((~m0c & m1c).sum()),
        "both_wrong": int((~m0c & ~m1c).sum()),
        "n_test": int(len(yte)),
    }

    # per-variant feature dims + predictions
    with open(os.path.join(diag_dir, "feature_dimensions.json"), "w") as f:
        json.dump({nm: {"train": list(Mt.shape), "val": list(Mv.shape),
                        "test": list(Me.shape)}
                   for nm, Mt, Mv, Me in [("M0", M0_tr, M0_va, M0_te),
                                          ("M1", M1_tr, M1_va, M1_te),
                                          ("M2", M2_tr, M2_va, M2_te),
                                          ("M3", M3_tr, M3_va, M3_te)]}, f, indent=2)

    os.makedirs(os.path.join(OUT_DIR, "predictions"), exist_ok=True)
    with open(os.path.join(OUT_DIR, "predictions", f"{ds_name}_seed42.csv"),
              "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_index", "true_class", "M0_pred", "M1_pred",
                    "M2_pred", "M3_pred"])
        for i in range(len(yte)):
            w.writerow([i, int(yte[i])] + [int(preds[v][i])
                        for v in ["M0", "M1", "M2", "M3"]])

    result = {
        "dataset": ds_name, "seed": SEED,
        "protocol": {"train": len(Xtr), "val": len(Xva), "test": len(Xte),
                     "T": T, "n_classes": data["n_classes"],
                     "val_source": data["val_source"]},
        "results": results, "deltas": deltas,
        "canonical_M0_reference": m0_ref, "canonical_M0_reproduced": bool(m0_ok),
        "old_preaudit_reference": OLD_PREAUDIT[ds_name],
        "old_vs_corrected_delta": {
            v: round(results[v]["test_macro_f1"] - OLD_PREAUDIT[ds_name][v], 4)
            for v in ["M0", "M1", "M2", "M3"]},
        "regime_diagnostics": regime_diag,
        "audits": {
            "mr_identity_max_diff": mr_identity,
            "occupancy_failures": occ_fail,
            "m2_m3_arrays": m2m3_diff,
            "feature_diffs": feature_diffs,
            "independent_recompute": recompute_checks,
            "valid_region_invariance": vr_invariance,
            "m2_rng_offset": M2_RNG_OFFSET, "m3_rng_offset": M3_RNG_OFFSET,
        },
        "complementarity_M0_M1": complementarity,
        "heterogeneity_stats": {
            v: {"mean": float(h.mean()), "std": float(h.std()),
                "nonzero_fraction": float((h != 0).mean()),
                "max": float(h.max())}
            for v, h in [("M1", H1_te), ("M2", H2_te), ("M3", H3_te)]},
    }
    with open(os.path.join(ds_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(diag_dir, "regime_occupancy.json"), "w") as f:
        json.dump({k: v for k, v in regime_diag.items()
                   if k in ("train", "val", "test")}, f, indent=2)

    print(f"\n  [SUMMARY] {ds_name}: "
          + " ".join(f"{v}={results[v]['test_macro_f1']:.4f}"
                     for v in ["M0", "M1", "M2", "M3"]))
    print(f"            deltas: {deltas}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 74)
    print("CORRECTED DRTN-CONDITIONED MINIROCKET -- NEGATIVE-DATASET RE-TEST (seed 42)")
    print("=" * 74)

    all_results = [run_dataset(ds, device, smoke=args.smoke) for ds in args.datasets]

    print("\n" + "=" * 74)
    print("OLD vs CORRECTED (seed 42, test Macro-F1)")
    print("=" * 74)
    print(f"{'dataset':<15} {'variant':<7} {'OLD':>7} {'CORRECTED':>10} {'delta':>8}")
    for r in all_results:
        for v in ["M0", "M1", "M2", "M3"]:
            print(f"{r['dataset']:<15} {v:<7} {r['old_preaudit_reference'][v]:>7.4f} "
                  f"{r['results'][v]['test_macro_f1']:>10.4f} "
                  f"{r['old_vs_corrected_delta'][v]:>+8.4f}")

    report = {
        "title": "CORRECTED DRTN-CONDITIONED MINIROCKET -- NEGATIVE-DATASET RE-TEST",
        "seed": SEED, "datasets": args.datasets,
        "implementation": "audited Stage A (per-sample-occupancy M2, valid-region H)",
        "results": all_results,
    }
    with open(os.path.join(OUT_DIR, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {os.path.join(OUT_DIR, 'report.json')}")
    return report


if __name__ == "__main__":
    main()
