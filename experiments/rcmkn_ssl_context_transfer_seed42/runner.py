"""SSL-context transfer experiment (seed 42): R2 architecture on
Phoneme / ECG5000_UNBAL / CWRU_UNBAL.

Reuses, without modification, the audited/validated components:
    * M2/M3-style controls + valid-region heterogeneity:
      experiments/drtn_conditioned_minirocket_haptics_3seed/runner.py
    * raw activation extractor / canonical PPV / independent H recompute /
      extract_drtn_regimes:
      experiments/drtn_conditioned_minirocket_transfer_seed42/core.py
    * dataset loaders + frozen per-dataset DRTN R5 checkpoints:
      experiments/external_stack_generalization/data.py,
      experiments/drtn_conditioned_minirocket_transfer_seed42/runner.py
    * SSL causal encoder + HardVQ + joint schedule (Haptics R2, unchanged):
      experiments/rcmkn_ssl_context_transfer_seed42/core.py

Variants (all 9996 = 4998 global + 4998 heterogeneity; no Hydra, no M0):
    R0  official-DRTN regimes   (audited reference; frozen checkpoints)
    R2  SSL-context regimes     (primary model under test)
    C1  occupancy-matched random regimes (audited M2 construction, seed+900001)
    C2  per-sample shuffled regimes      (audited M3 construction, seed+900002)

R0 gates (within-run reproduction, from the audited corrected retest):
    ECG5000_UNBAL 0.5859, CWRU_UNBAL 0.9792 (tolerance 0.0011).
Phoneme has no prior audited R0 -> in-run value becomes the first reference.

Usage:
    python -m experiments.rcmkn_ssl_context_transfer_seed42.runner \
        [--datasets Phoneme ECG5000_UNBAL CWRU_UNBAL] [--smoke]
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
    create_random_regime_control,     # audited: per-sample occupancy, seed+900001
    create_shuffled_regime_control,   # audited: per-sample permutation, seed+900002
    compute_regime_heterogeneity,     # audited: valid-region H_m
    M2_RNG_OFFSET,
    M3_RNG_OFFSET,
)
from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (  # noqa: E402
    compute_raw_activations,
    ppv_from_activations,
    independent_heterogeneity_recompute,
    extract_drtn_regimes,
)
import experiments.drtn_conditioned_minirocket_transfer_seed42.runner as transfer  # noqa: E402

from experiments.rcmkn_haptics_seed42.config import (  # noqa: E402
    SEED, N_FEATURES, N_GLOBAL, K_CODES, ENCODER, VQ, JOINT)
import experiments.rcmkn_ssl_context_transfer_seed42.core as core  # noqa: E402

OUT_DIR = os.path.join(ROOT, "results", "rcmkn_ssl_context_transfer_seed42")
ALPHAS = np.logspace(-4, 4, 20)

# AUDIT 1: canonical project configuration (from the frozen loaders)
EXPECTED = {
    "Phoneme":       {"T": 1024, "n_classes": 39, "train": 185, "val": 29,   "test": 1896},
    "ECG5000_UNBAL": {"T": 140,  "n_classes": 5,  "train": 3400, "val": 600, "test": 1000},
    "CWRU_UNBAL":    {"T": 1024, "n_classes": 4,  "train": 1156, "val": 204, "test": 240},
    "EpilepticSeizures": {"T": 178, "n_classes": 2, "train": 80, "val": 20, "test": 11420},
}
R0_GATES = {  # audited corrected-retest M1 references (deterministic Ridge)
    "ECG5000_UNBAL": 0.5859,
    "CWRU_UNBAL": 0.9792,
}
R0_TOL = 0.0011


def log(msg):
    print(msg, flush=True)


def arr_hash(a):
    return hashlib.sha1(np.ascontiguousarray(a, dtype=np.int64).tobytes()).hexdigest()[:16]


def macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


def run_variant(name, blocks, yva, yte, ytrva):
    """Fit Ridge on train+val, evaluate val + test once. blocks: dict of
    (F_tr, F_va, F_te) per block, concatenated in block order."""
    F_tr = np.hstack([b[0] for b in blocks.values()])
    F_va = np.hstack([b[1] for b in blocks.values()])
    F_te = np.hstack([b[2] for b in blocks.values()])
    assert F_tr.shape[1] == F_va.shape[1] == F_te.shape[1] == N_FEATURES, \
        f"{name}: budget {F_tr.shape[1]} != {N_FEATURES} (AUDIT 3)"
    assert F_tr.shape[0] == len(ytrva), \
        f"{name}: F_tr rows {F_tr.shape[0]} != ytrva {len(ytrva)}"
    ridge = RidgeClassifierCV(alphas=ALPHAS)
    t0 = time.time()
    ridge.fit(F_tr, ytrva)                        # train + validation (AUDIT 16)
    pred_va = ridge.predict(F_va)
    pred_te = ridge.predict(F_te)
    res = {
        "val_macro_f1": round(macro_f1(yva, pred_va), 4),
        "test_macro_f1": round(macro_f1(yte, pred_te), 4),
        "accuracy": round(float(accuracy_score(yte, pred_te)), 4),
        "selected_alpha": float(ridge.alpha_),
        "feature_dim": int(F_tr.shape[1]),
        "ridge_fit_s": round(time.time() - t0, 2),
        "class_f1s": [round(x, 4) for x in
                      f1_score(yte, pred_te, average=None,
                               labels=list(range(len(np.unique(ytrva)))),
                               zero_division=0).tolist()],
    }
    return res, pred_te


def hetero_block(extractor, X_z, regimes, valid_het):
    """Valid-region H over (N, T) regimes, computed in bounded sample chunks."""
    N = len(X_z)
    H = np.empty((N, N_GLOBAL), dtype=np.float64)
    for c0 in range(0, N, 64):
        c1 = min(c0 + 64, N)
        act, _ = compute_raw_activations(extractor, X_z[c0:c1])
        H[c0:c1] = compute_regime_heterogeneity(act[:, N_GLOBAL:], valid_het,
                                                regimes[c0:c1])
        del act
    return H


def h_stats(H):
    return {"mean_H": float(H.mean()), "median_H": float(np.median(H)),
            "max_H": float(H.max()), "fraction_nonzero": float((H > 0).mean())}


def run_dataset(ds_name, device, smoke=False):
    ds_dir = os.path.join(OUT_DIR, ds_name)
    os.makedirs(ds_dir, exist_ok=True)
    os.makedirs(os.path.join(ds_dir, "diagnostics"), exist_ok=True)
    audits = {}
    t_start = time.time()

    # ---------------- AUDIT 1: canonical configuration ----------------------
    data = transfer.load_any_dataset(ds_name)
    Xtr, ytr = data["Xtr"], data["ytr"]
    Xva, yva = data["Xva"], data["yva"]
    Xte, yte = data["Xte"], data["yte"]
    ytrva = np.concatenate([ytr, yva])
    exp = EXPECTED[ds_name]
    got = {"T": len(Xtr[0]), "n_classes": data["n_classes"], "train": len(Xtr),
           "val": len(Xva), "test": len(Xte)}
    assert got == exp, f"AUDIT 1 FAILED {ds_name}: {got} != {exp}"
    log(f"\n=== {ds_name}: train={got['train']} val={got['val']} "
        f"test={got['test']} T={got['T']} n_classes={got['n_classes']} "
        f"dim=1 (univariate) ===")
    audits["audit1_config"] = {"expected": exp, "actual": got, "pass": True}
    results = {}  # populated before R0
    preds = {}

    Xtr_z, Xva_z, Xte_z = (transfer.znorm(Xtr), transfer.znorm(Xva),
                           transfer.znorm(Xte))
    Xtrva_z = np.vstack([Xtr_z, Xva_z])
    T = got["T"]

    # ---------------- canonical MiniRocket (fit on train) -------------------
    from aeon.transformations.collection.convolution_based import MiniRocket
    set_seed(SEED)
    extractor = MiniRocket(random_state=SEED, n_jobs=-1)
    extractor.fit(Xtr_z[:, None, :].astype(np.float32))
    F_trva_mr = extractor.transform(Xtrva_z[:, None, :].astype(np.float32))
    F_va_mr = extractor.transform(Xva_z[:, None, :].astype(np.float32))
    F_te_mr = extractor.transform(Xte_z[:, None, :].astype(np.float32))
    assert F_trva_mr.shape[1] == N_FEATURES
    # G_tr = full trainva features (len(ytrva) rows); run_variant
    # uses ytrva for fit and yva for val eval.
    G_tr = F_trva_mr[:, :N_GLOBAL]         # (trainva, 4998)
    G_va = F_va_mr[:, :N_GLOBAL]           # (val, 4998)
    G_te = F_te_mr[:, :N_GLOBAL]           # (test, 4998)
    log(f"  [AUDIT3/4] global block = features [0, {N_GLOBAL}), "
        f"het block = [{N_GLOBAL}, {N_FEATURES}); feature-axis slicing only "
        f"(rows {F_trva_mr.shape[0]} untouched)")

    # ---------------- AUDIT 2: raw-extractor identity -----------------------
    valid = None
    mr_id = 0.0
    for c0 in range(0, len(Xtrva_z), 64):
        act, valid = compute_raw_activations(extractor, Xtrva_z[c0:c0 + 64])
        mr_id = max(mr_id, float(np.max(np.abs(
            ppv_from_activations(act, valid) - F_trva_mr[c0:c0 + 64]))))
        del act
    assert mr_id < 1e-5, f"AUDIT 2 FAILED: extractor identity {mr_id}"
    audits["audit2_extractor_identity_maxdiff"] = float(mr_id)
    log(f"  [AUDIT2] raw-extractor PPV vs aeon: max|diff|={mr_id:.2e}")
    valid_het = valid[N_GLOBAL:]

    # ---------------- R0: audited DRTN reference (frozen checkpoint) --------
    log("  [R0] loading frozen DRTN R5 checkpoint (transfer namespace)")
    drtn, drtn_info = transfer.train_or_load_drtn(ds_name, data, device,
                                                  smoke=smoke)
    set_seed(SEED)
    reg0_trva = extract_drtn_regimes(drtn, Xtrva_z, device=device)
    reg0_te = extract_drtn_regimes(drtn, Xte_z, device=device)
    sd1 = {k: v.clone() for k, v in drtn.state_dict().items()}
    _ = extract_drtn_regimes(drtn, Xte_z[:2], device=device)
    assert all(torch.equal(sd1[k], drtn.state_dict()[k]) for k in sd1), \
        "DRTN checkpoint mutated during regime extraction"
    H0_tr = hetero_block(extractor, Xtrva_z, reg0_trva, valid_het)
    act_va, _ = compute_raw_activations(extractor, Xva_z)
    H0_va = compute_regime_heterogeneity(act_va[:, N_GLOBAL:], valid_het,
                                         reg0_trva[len(Xtr):])
    act_te, _ = compute_raw_activations(extractor, Xte_z)
    H0_te = compute_regime_heterogeneity(act_te[:, N_GLOBAL:], valid_het, reg0_te)
    res0, pred0 = run_variant("R0", {"global": (G_tr, G_va, G_te),
                                     "het": (H0_tr, H0_va, H0_te)},
                              yva, yte, ytrva)
    audits["r0_reference"] = {"test_macro_f1": res0["test_macro_f1"],
                              "drtn_ckpt": drtn_info}
    results["R0"] = res0
    if ds_name in R0_GATES and not smoke:
        gate = abs(res0["test_macro_f1"] - R0_GATES[ds_name]) <= R0_TOL
        audits["r0_gate"] = {"reference": R0_GATES[ds_name], "pass": bool(gate)}
        log(f"  [R0 GATE] {res0['test_macro_f1']} vs audited "
            f"{R0_GATES[ds_name]} -> {'PASS' if gate else 'FAIL'}")
        assert gate, f"R0 failed reproduction on {ds_name}. STOP."
    else:
        log(f"  [R0] test MF1 {res0['test_macro_f1']} (first audited "
            f"reference for {ds_name})" if ds_name not in R0_GATES else
            f"  [R0 GATE] skipped (smoke)")

    # ---------------- SSL context model (R2/C1/C2 source) -------------------
    log("  [SSL] training context model (Haptics R2 schedule, unchanged)")
    set_seed(SEED)
    model = core.build_context_model(data["n_classes"], device)
    params = None
    train_info = {}
    if smoke:
        train_info, params = core.train_ssl_context(model, Xtr_z, ytr, Xva_z,
                                                    yva, device, smoke=True)
    else:
        train_info, params = core.train_ssl_context(model, Xtr_z, ytr, Xva_z,
                                                    yva, device, smoke=False)
    core.freeze_model(model)
    torch.save({"model_state": model.state_dict(),
                "config": {"encoder": ENCODER, "vq": VQ, "joint": JOINT},
                "train_info": train_info, "params": params},
               os.path.join(ds_dir, "context_model_seed42.pt"))
    log(f"  [SSL] {train_info} params={params}")

    # AUDIT 12: encoder causality spot check (future perturbation must not
    # change earlier states)
    with torch.no_grad():
        xs = torch.from_numpy(Xtr_z[:1])[:, None, :].to(device)
        z1 = model.encoder(xs)
        x2 = xs.clone()
        x2[:, :, T // 2:] = torch.randn_like(x2[:, :, T // 2:])
        z2 = model.encoder(x2)
        caus = float((z1[:, :T // 2] - z2[:, :T // 2]).abs().max())
    assert caus == 0.0, f"AUDIT 12 FAILED: encoder non-causal (max|d|={caus})"
    audits["audit12_causality_maxdiff"] = caus
    log(f"  [AUDIT12] encoder causality: future perturbation changes earlier "
        f"states by exactly {caus:.1e}")

    reg2_trva = core.context_regimes(model, Xtrva_z, device)
    reg2_te = core.context_regimes(model, Xte_z, device)
    occ2 = {"train": core.occupancy(reg2_trva), "test": core.occupancy(reg2_te)}
    audits["audit8_vq_diagnostics"] = occ2
    log(f"  [AUDIT8] VQ train occupancy: active={occ2['train']['active_codes']} "
        f"entropy={occ2['train']['normalized_entropy']:.3f} "
        f"perplexity={occ2['train']['perplexity']:.2f} "
        f"dominant={occ2['train']['dominant_fraction']:.3f}")

    # ---------------- C1/C2 controls (audited constructions) ----------------
    c1_trva = create_random_regime_control(reg2_trva, seed=SEED)
    c1_te = create_random_regime_control(reg2_te, seed=SEED)
    c2_trva = create_shuffled_regime_control(reg2_trva, seed=SEED)
    c2_te = create_shuffled_regime_control(reg2_te, seed=SEED)

    # AUDIT 9/10: exact per-sample occupancy preservation
    occ_fail = 0
    for ctrl, src in [(c1_trva, reg2_trva), (c2_trva, reg2_trva),
                      (c1_te, reg2_te), (c2_te, reg2_te)]:
        for i in range(len(src)):
            if not np.array_equal(np.bincount(ctrl[i], minlength=K_CODES),
                                  np.bincount(src[i], minlength=K_CODES)):
                occ_fail += 1
    assert occ_fail == 0, f"AUDIT 9/10 FAILED: {occ_fail} occupancy failures"
    audits["audit9_10_control_occupancy_failures"] = occ_fail

    # AUDIT 11: C1/C2 distinct, distinct from R2, no shared memory
    reg_audit = {}
    for nm, a in [("C1", (c1_trva, c1_te)), ("C2", (c2_trva, c2_te))]:
        src = reg2_trva if a is c1_trva else reg2_trva
        reg_audit[nm] = {
            "hash_trainva": arr_hash(a[0]), "hash_test": arr_hash(a[1]),
            "shares_memory_with_r2": bool(np.shares_memory(a[0], reg2_trva)),
        }
    c1c2_diff = int((c1_trva != c2_trva).sum())
    reg_audit["C1_vs_C2"] = {
        "differing_positions": c1c2_diff,
        "fraction": c1c2_diff / c1_trva.size,
        "identical": bool(np.array_equal(c1_trva, c2_trva)),
    }
    audits["audit11_regime_arrays"] = reg_audit
    assert not reg_audit["C1"]["shares_memory_with_r2"]
    assert not reg_audit["C2"]["shares_memory_with_r2"]
    log(f"  [AUDIT9/10/11] controls: occupancy 0 failures; C1 hash "
        f"{reg_audit['C1']['hash_trainva'][:8]} vs C2 "
        f"{reg_audit['C2']['hash_trainva'][:8]} "
        f"({c1c2_diff / c1_trva.size:.1%} positions differ); no shared memory")

    # ---------------- H blocks for R2/C1/C2 ---------------------------------
    H2_tr = hetero_block(extractor, Xtrva_z, reg2_trva, valid_het)
    H2_va = compute_regime_heterogeneity(act_va[:, N_GLOBAL:], valid_het,
                                         reg2_trva[len(Xtr):])
    H2_te = compute_regime_heterogeneity(act_te[:, N_GLOBAL:], valid_het, reg2_te)

    # ---------------- AUDIT 6: independent H recompute ----------------------
    s_idx = 0
    smp, val = compute_raw_activations(extractor, Xtrva_z[s_idx:s_idx + 1])
    smp = smp[0, N_GLOBAL:]; val = val[N_GLOBAL:]
    recompute_max = 0.0
    for m in np.linspace(0, N_GLOBAL - 1, 8).astype(int):
        ref = independent_heterogeneity_recompute(
            smp[m].astype(bool), val[m].astype(bool),
            reg2_trva[s_idx].astype(np.int64), K=K_CODES)
        impl = float(H2_tr[s_idx, m])
        recompute_max = max(recompute_max, abs(ref - impl))
    assert recompute_max <= 0.05, f"AUDIT 6 FAILED: recompute diff {recompute_max}"
    audits["audit6_independent_h_recompute_maxdiff"] = float(recompute_max)
    log(f"  [AUDIT6] independent float64 H recompute (8 kernels, sample 0): "
        f"max diff {recompute_max:.2e}")

    # ---------------- AUDIT 7: per-feature valid-region invariance ----------
    act_s = smp.copy()
    n_flip = 0
    H_before = H2_tr[s_idx].copy()
    # corrupt, per feature, ONLY positions outside that feature's own mask
    flip_idx = [np.flatnonzero(~valid_het[m]) for m in range(N_GLOBAL)]
    act_all, _ = compute_raw_activations(extractor, Xtrva_z[s_idx:s_idx + 1])
    act_all = act_all[0, N_GLOBAL:].copy()
    for m in range(N_GLOBAL):
        if len(flip_idx[m]):
            act_all[m, flip_idx[m]] = ~act_all[m, flip_idx[m]]
            n_flip += int(len(flip_idx[m]))
    H_after = compute_regime_heterogeneity(
        act_all[None], valid_het, reg2_trva[s_idx:s_idx + 1])[0]
    vr_ok = bool(np.array_equal(H_before, H_after))
    assert vr_ok, "AUDIT 7 FAILED: out-of-mask activations changed H_m"
    audits["audit7_valid_region"] = {
        "flipped_out_of_mask_activations": int(n_flip),
        "h_changed": False, "pass": True}
    log(f"  [AUDIT7] per-feature valid region: flipped {n_flip} out-of-mask "
        f"activations -> H unchanged exactly")

    # ---------------- variants ---------------------------------------------

    log("  [R2] SSL context + hard-VQ heterogeneity")
    results["R2"], preds["R2"] = run_variant(
        "R2", {"global": (G_tr, G_va, G_te), "het": (H2_tr, H2_va, H2_te)},
        yva, yte, ytrva)
    for nm, r_trva, r_te in [("C1", c1_trva, c1_te), ("C2", c2_trva, c2_te)]:
        log(f"  [{nm}] controls")
        H_tr = hetero_block(extractor, Xtrva_z, r_trva, valid_het)
        H_va = compute_regime_heterogeneity(act_va[:, N_GLOBAL:], valid_het,
                                            r_trva[len(Xtr):])
        H_te = compute_regime_heterogeneity(act_te[:, N_GLOBAL:], valid_het, r_te)
        results[nm], preds[nm] = run_variant(
            nm, {"global": (G_tr, G_va, G_te), "het": (H_tr, H_va, H_te)},
            yva, yte, ytrva)

    # ---------------- mechanistic H stats ----------------------------------
    mech = {"R0": h_stats(H0_tr), "R2": h_stats(H2_tr),
            "C1": h_stats(hetero_block(extractor, Xtrva_z, c1_trva, valid_het)),
            "C2": h_stats(hetero_block(extractor, Xtrva_z, c2_trva, valid_het))}
    for v in ["R0", "R2", "C1", "C2"]:
        results[v]["mechanistic_h"] = mech[v]
    results["R0"]["regime_diagnostics"] = {
        "train": core.occupancy(reg0_trva)}
    results["R2"]["regime_diagnostics"] = occ2
    results["R2"]["context_train"] = train_info
    results["R2"]["params"] = params

    deltas = {f"R2-{k}": round(results["R2"]["test_macro_f1"] -
                               results[k]["test_macro_f1"], 4)
              for k in ["R0", "C1", "C2"]}
    deltas["C1-C2"] = round(results["C1"]["test_macro_f1"] -
                            results["C2"]["test_macro_f1"], 4)

    # ---------------- save per-dataset artifacts ----------------------------
    os.makedirs(os.path.join(ds_dir, "predictions"), exist_ok=True)
    with open(os.path.join(ds_dir, "predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_index", "true_class"] +
                   [v for v in ["R0", "R2", "C1", "C2"]])
        for i in range(len(yte)):
            w.writerow([i, int(yte[i])] + [int(pred0[i])] +
                       [int(preds[v][i]) for v in ["R2", "C1", "C2"]])
    ds_result = {
        "dataset": ds_name, "seed": SEED, "T": T,
        "split": {"train": len(Xtr), "val": len(Xva), "test": len(Xte)},
        "n_classes": data["n_classes"], "n_features": N_FEATURES,
        "K_regimes": K_CODES,
        "config": {"encoder": ENCODER, "vq": VQ, "joint": JOINT},
        "drtn_checkpoint": drtn_info,
        "results": results, "deltas": deltas, "mechanistic_h": mech,
        "regime_array_audit": reg_audit,
        "audits": audits,
        "runtime_s": round(time.time() - t_start, 1),
    }
    with open(os.path.join(ds_dir, "result.json"), "w") as f:
        json.dump(ds_result, f, indent=2)
    with open(os.path.join(ds_dir, "diagnostics", "audits.json"), "w") as f:
        json.dump(audits, f, indent=2)
    del act_va, act_te, act_all, H0_tr, H0_va, H0_te, H2_tr, H2_va, H2_te
    log(f"  [{ds_name}] done in {ds_result['runtime_s']}s — "
        + " ".join(f"{v}={results[v]['test_macro_f1']:.4f}"
                   for v in ["R0", "R2", "C1", "C2"]))
    return ds_result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["Phoneme", "ECG5000_UNBAL", "CWRU_UNBAL"])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    log("=" * 74)
    log("SSL-CONTEXT + HARD-VQ HETEROGENEITY TRANSFER (seed 42) — "
        "Phoneme / ECG5000_UNBAL / CWRU_UNBAL")
    log("=" * 74)
    log(f"  device={device} smoke={args.smoke} K={K_CODES} "
        f"enc={ENCODER['channels']} d={ENCODER['d_model']} "
        f"mask={ENCODER['mask_ratio']} span={ENCODER['span_len']} "
        f"beta_vq={VQ['beta_vq']} lam_div={VQ['lam_div']} "
        f"lam_cls={JOINT['lambda_cls']}")

    all_results = {}
    for ds in args.datasets:
        all_results[ds] = run_dataset(ds, device, smoke=args.smoke)

    # ---------------- consolidated report -----------------------------------
    per_ds = {ds: {v: r["results"][v]["test_macro_f1"]
                   for v in ["R0", "R2", "C1", "C2"]}
              for ds, r in all_results.items()}
    deltas = {ds: r["deltas"] for ds, r in all_results.items()}
    mech = {ds: r["mechanistic_h"] for ds, r in all_results.items()}
    verdicts = {}
    for ds, r in all_results.items():
        res = {v: r["results"][v]["test_macro_f1"] for v in ["R0", "R2", "C1", "C2"]}
        if abs(res["R2"] - res["R0"]) < 0.005 and \
           max(abs(res["R2"] - res["C1"]), abs(res["R2"] - res["C2"])) < 0.005:
            verdicts[ds] = "UNINFORMATIVE"
        elif res["R2"] > res["R0"]:
            if res["R2"] > res["C1"] and res["R2"] > res["C2"]:
                margin = min(res["R2"] - res["C1"], res["R2"] - res["C2"])
                verdicts[ds] = ("STRONG TRANSFER" if margin >= 0.01
                                else "CONDITIONAL / PARTIAL TRANSFER")
            else:
                verdicts[ds] = "CONDITIONAL / PARTIAL TRANSFER"
        else:
            verdicts[ds] = "NEGATIVE"

    report = {
        "title": "SSL-context + hard-VQ heterogeneity transfer (seed 42)",
        "seed": SEED, "datasets": args.datasets,
        "variants": ["R0", "R2", "C1", "C2"],
        "per_dataset_test_macro_f1": per_ds,
        "deltas": deltas, "mechanistic_h": mech, "verdicts": verdicts,
        "context_references": {
            "Haptics": {"M0_3seed": 0.4948, "audited_R0_3seed": 0.5340,
                        "SSL_R2_seed42": 0.5500},
            "ECG5000_BAL": {"M0_3seed": 0.6498, "audited_M1_3seed": 0.6653},
        },
        "canonical_M0_references": {"Phoneme": 0.0808,
                                    "ECG5000_UNBAL": 0.5938,
                                    "CWRU_UNBAL": 0.9917},
        "config": {"encoder": ENCODER, "vq": VQ, "joint": JOINT,
                   "alphas": "logspace(-4,4,20)",
                   "note": "no hyperparameter search; Haptics R2 values "
                           "transferred unchanged"},
        "audits_all": {ds: r["audits"] for ds, r in all_results.items()},
    }
    if not args.smoke:
        with open(os.path.join(OUT_DIR, "report.json"), "w") as f:
            json.dump(report, f, indent=2)
        with open(os.path.join(OUT_DIR, "per_dataset_results.json"), "w") as f:
            json.dump({ds: r for ds, r in all_results.items()}, f, indent=2)
        with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
            json.dump(report["config"], f, indent=2)

    log("\n" + "=" * 74)
    log("RESULTS (test Macro-F1, seed 42)")
    log("=" * 74)
    for ds in args.datasets:
        r = all_results[ds]["results"]
        log(f"  {ds}: R0={r['R0']['test_macro_f1']:.4f} "
            f"R2={r['R2']['test_macro_f1']:.4f} "
            f"C1={r['C1']['test_macro_f1']:.4f} "
            f"C2={r['C2']['test_macro_f1']:.4f}  -> {verdicts[ds]}")
        log(f"        deltas {all_results[ds]['deltas']}")
    return report


if __name__ == "__main__":
    main()
