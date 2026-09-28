"""Runner: R2 architecture on GunPoint (seed 42).

Single dataset, single seed, test touched exactly once per variant.

Pipeline (all audited components reused unchanged):
    1.  Load verified Kaggle GunPoint TSVs via load_kaggle_ucr (the exact
        loader + split the context3 screen used for its M0 reference).
    2.  Per-sample z-normalization (context3 znorm), once, before all branches.
    3.  Canonical aeon MiniRocket(random_state=42) fit on TRAIN rows;
        M0 = full 9996-feature transform -> reproduction gate 0.9933.
    4.  Train the exact Haptics R2 context model (SSL + HardVQ K=8) with
        train_context_model; freeze; extract hard regimes.
    5.  C1/C2 controls via the audited M2/M3 constructions.
    6.  H blocks via compute_regime_heterogeneity (valid-region convention).
    7.  RidgeClassifierCV(logspace(-4,4,20)) fit on train+val; one test pass.
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, f1_score

from experiments.drtn_conditioned_minirocket_haptics_3seed.runner import (
    create_random_regime_control,     # audited: per-sample occupancy, seed+900001
    create_shuffled_regime_control,   # audited: per-sample permutation, seed+900002
    compute_regime_heterogeneity,     # audited: valid-region H_m
    M2_RNG_OFFSET, M3_RNG_OFFSET,
)
from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (
    compute_raw_activations,
    ppv_from_activations,
    independent_heterogeneity_recompute,
)
from experiments.drtn_conditioned_minirocket_context3_seed42.core import (
    DATASET_SPECS, load_kaggle_ucr, znorm,
)
from experiments.rcmkn_haptics_seed42.config import (
    SEED, N_FEATURES, N_GLOBAL, N_HET, K_CODES, ENCODER, VQ, JOINT,
)
from experiments.rcmkn_haptics_seed42.model import parameter_report
from experiments.rcmkn_haptics_seed42.runner import (
    train_context_model, extract_context_regimes, set_seed,
)
from experiments.rcmkn_haptics_seed42.vq import occupancy_stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "results", "rcmkn_r2_gunpoint_seed42")
DATASET = "GunPoint"
ALPHAS = np.logspace(-4, 4, 20)

# AUDIT 1: canonical configuration (context3 spec + repository split rule)
EXPECTED = {"train": 42, "val": 8, "test": 150, "T": 150, "n_classes": 2}
M0_REF = 0.9933                 # context3 canonical aeon MiniRocket (seed 42)
M0_TOL = 0.0011                 # repo canonical M0 reproduction guard
R2_HAPTICS_REF = 0.5500         # for context in the report


def log(msg):
    print(msg, flush=True)


def macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def run_variant(name, blocks, yva, yte, ytrva):
    """Fit Ridge on train+val, evaluate val + test once (audited protocol)."""
    F_tr = np.hstack([b[0] for b in blocks.values()])
    F_va = np.hstack([b[1] for b in blocks.values()])
    F_te = np.hstack([b[2] for b in blocks.values()])
    assert F_tr.shape[1] == F_va.shape[1] == F_te.shape[1] == N_FEATURES, \
        f"{name}: budget {F_tr.shape[1]} != {N_FEATURES}"
    assert F_tr.shape[0] == len(ytrva), \
        f"{name}: rows {F_tr.shape[0]} != labels {len(ytrva)}"
    ridge = RidgeClassifierCV(alphas=ALPHAS)
    t0 = time.time()
    ridge.fit(F_tr, ytrva)                    # train + validation
    pred_va = ridge.predict(F_va)
    pred_te = ridge.predict(F_te)             # OFFICIAL test eval: once
    return {
        "val_macro_f1": round(macro_f1(yva, pred_va), 4),
        "test_macro_f1": round(macro_f1(yte, pred_te), 4),
        "accuracy": round(float(accuracy_score(yte, pred_te)), 4),
        "selected_alpha": float(ridge.alpha_),
        "feature_dim": int(F_tr.shape[1]),
        "ridge_fit_s": round(time.time() - t0, 2),
        "class_f1s": [round(x, 4) for x in f1_score(
            yte, pred_te, average=None, zero_division=0,
            labels=list(range(EXPECTED["n_classes"])))],
    }, pred_te


def hetero_block(extractor, X_z, regimes, valid_het, chunk=64):
    """Valid-region H over (N, T) regimes in bounded sample chunks."""
    N = len(X_z)
    H = np.empty((N, N_HET), dtype=np.float64)
    for c0 in range(0, N, chunk):
        c1 = min(c0 + chunk, N)
        act, _ = compute_raw_activations(extractor, X_z[c0:c1])
        H[c0:c1] = compute_regime_heterogeneity(
            act[:, N_GLOBAL:], valid_het, regimes[c0:c1])
        del act
    return H


def h_stats(H):
    return {"mean_H": float(H.mean()), "median_H": float(np.median(H)),
            "max_H": float(H.max()),
            "fraction_nonzero": float((H > 0).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)

    log("=" * 74)
    log(f"R2 (SSL + HardVQ heterogeneity) ON {DATASET} (seed {SEED})")
    log("=" * 74)
    log(f"  device={device} smoke={args.smoke} K={K_CODES} "
        f"enc={ENCODER['channels']} d={ENCODER['d_model']}")
    audits = {}

    # ---------------- 1. data (AUDIT 1) ----------------
    data = load_kaggle_ucr(DATASET, seed=SEED)
    Xtr, ytr = data["Xtr"], data["ytr"]
    Xva, yva = data["Xva"], data["yva"]
    Xte, yte = data["Xte"], data["yte"]
    ytrva = np.concatenate([ytr, yva])
    got = {"train": len(Xtr), "val": len(Xva), "test": len(Xte),
           "T": data["L"], "n_classes": data["n_classes"]}
    assert got == EXPECTED, f"AUDIT 1 FAILED: {got} != {EXPECTED}"
    audits["audit1_config"] = {"expected": EXPECTED, "actual": got,
                               "pass": True}
    audits["provenance"] = {
        "source_path": data["source_path"],
        "file_sha256": data["file_sha256"],
        "label_map": data["label_map"],
        "val_source": data["val_source"],
    }
    log(f"  train={got['train']} val={got['val']} test={got['test']} "
        f"T={got['T']} n_cls={got['n_classes']} "
        f"val_source={data['val_source']}")

    Xtr_z, Xva_z, Xte_z = znorm(Xtr), znorm(Xva), znorm(Xte)
    Xtrva_z = np.vstack([Xtr_z, Xva_z])

    # ---------------- 2. canonical MiniRocket + M0 (AUDIT 2/3 + gate) ------
    from aeon.transformations.collection.convolution_based import MiniRocket
    set_seed(SEED)
    extractor = MiniRocket(random_state=SEED, n_jobs=-1)
    extractor.fit(Xtr_z[:, None, :].astype(np.float32))
    F_trva = extractor.transform(Xtrva_z[:, None, :].astype(np.float32))
    F_te = extractor.transform(Xte_z[:, None, :].astype(np.float32))
    assert F_trva.shape[1] == N_FEATURES, \
        f"AUDIT 3 FAILED: {F_trva.shape[1]} features"
    G_trva, G_te = F_trva[:, :N_GLOBAL], F_te[:, :N_GLOBAL]
    G_tr, G_va = G_trva[:len(Xtr)], G_trva[len(Xtr):]
    audits["audit3_budget_9996"] = {"pass": True}

    # AUDIT 2: raw-extractor identity vs canonical aeon transform
    valid = None
    mr_id = 0.0
    for c0 in range(0, len(Xtrva_z), 64):
        act, valid = compute_raw_activations(extractor, Xtrva_z[c0:c0 + 64])
        mr_id = max(mr_id, float(np.max(np.abs(
            ppv_from_activations(act, valid) - F_trva[c0:c0 + 64]))))
        del act
    assert mr_id < 1e-5, f"AUDIT 2 FAILED: extractor identity {mr_id}"
    audits["audit2_extractor_identity_maxdiff"] = float(mr_id)
    log(f"  [AUDIT2] raw-extractor PPV vs aeon: max|diff|={mr_id:.2e}")
    valid_het = valid[N_GLOBAL:]

    # M0 reproduction gate (context3 canonical reference)
    results, preds = {}, {}
    results["M0"], preds["M0"] = run_variant(
        "M0", {"all": (F_trva, F_trva[len(Xtr):], F_te)},
        yva, yte, ytrva)
    gate = abs(results["M0"]["test_macro_f1"] - M0_REF) <= M0_TOL
    audits["m0_gate"] = {"reference": M0_REF,
                         "observed": results["M0"]["test_macro_f1"],
                         "tolerance": M0_TOL, "pass": bool(gate)}
    log(f"  [M0 GATE] {results['M0']['test_macro_f1']:.4f} vs context3 "
        f"canonical {M0_REF} -> {'PASS' if gate else 'FAIL'}")
    assert gate, "M0 failed to reproduce the context3 canonical reference."

    # ---------------- 3. R2 context model (exact Haptics schedule) ---------
    log("  [SSL] training context model (Haptics R2 schedule, unchanged)")
    set_seed(SEED)
    from experiments.rcmkn_haptics_seed42.model import RCMKNContextModel
    model = RCMKNContextModel(n_classes=data["n_classes"]).to(device)
    params = parameter_report(model)
    train_info = train_context_model(model, Xtr_z, ytr, Xva_z, yva,
                                     device, smoke=args.smoke)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    torch.save({"model_state": model.state_dict(),
                "config": {"encoder": ENCODER, "vq": VQ, "joint": JOINT},
                "train_info": train_info, "params": params},
               os.path.join(OUT_DIR, "context_model_seed42.pt"))
    log(f"  [SSL] {train_info}")

    # AUDIT 4: causality spot check (future perturbation, earlier states)
    with torch.no_grad():
        xs = torch.from_numpy(Xtr_z[:1])[:, None, :].to(device)
        z1 = model.encoder(xs)
        x2 = xs.clone()
        x2[:, :, got["T"] // 2:] = torch.randn_like(x2[:, :, got["T"] // 2:])
        z2 = model.encoder(x2)
        caus = float((z1[:, :got["T"] // 2] - z2[:, :got["T"] // 2])
                     .abs().max())
    assert caus == 0.0, f"AUDIT 4 FAILED: encoder non-causal ({caus})"
    audits["audit4_causality_maxdiff"] = caus
    log(f"  [AUDIT4] causality: future perturbation changes earlier states "
        f"by exactly {caus:.1e}")

    # regimes
    set_seed(SEED)
    reg_trva = extract_context_regimes(model, Xtrva_z, device, batch=32)
    set_seed(SEED)
    reg_te = extract_context_regimes(model, Xte_z, device, batch=32)
    # AUDIT 5: determinism (re-extract, must be identical)
    set_seed(SEED)
    assert np.array_equal(
        extract_context_regimes(model, Xtrva_z, device, batch=32), reg_trva), \
        "AUDIT 5 FAILED: regimes not deterministic"
    audits["audit5_regimes_deterministic"] = {"pass": True}
    occ = {"train": occupancy_stats(reg_trva, K=K_CODES),
           "test": occupancy_stats(reg_te, K=K_CODES)}
    audits["audit6_vq_diagnostics"] = occ
    log(f"  [AUDIT6] VQ train occupancy: active={occ['train']['active_codes']} "
        f"entropy={occ['train']['normalized_entropy']:.3f} "
        f"perplexity={occ['train']['perplexity']:.2f}")

    # ---------------- 4. controls (audited constructions) ------------------
    set_seed(SEED)
    c1_trva = create_random_regime_control(reg_trva, seed=SEED, K=K_CODES)
    c1_te = create_random_regime_control(reg_te, seed=SEED, K=K_CODES)
    set_seed(SEED)
    c2_trva = create_shuffled_regime_control(reg_trva, seed=SEED)
    c2_te = create_shuffled_regime_control(reg_te, seed=SEED)

    occ_fail = 0
    for ctrl, src in [(c1_trva, reg_trva), (c2_trva, reg_trva),
                      (c1_te, reg_te), (c2_te, reg_te)]:
        for i in range(len(src)):
            if not np.array_equal(
                    np.bincount(ctrl[i], minlength=K_CODES),
                    np.bincount(src[i], minlength=K_CODES)):
                occ_fail += 1
    assert occ_fail == 0, f"AUDIT 7 FAILED: {occ_fail} occupancy failures"
    audits["audit7_control_occupancy_failures"] = occ_fail
    assert not np.array_equal(c1_trva, c2_trva) and \
        not np.shares_memory(c1_trva, c2_trva), "AUDIT 8 FAILED: controls"
    audits["audit8_controls_distinct"] = {"pass": True}
    log("  [AUDIT7/8] controls occupancy-preserving, distinct, no shared memory")

    # ---------------- 5. H blocks ------------------------------------------
    H_tr = hetero_block(extractor, Xtrva_z, reg_trva, valid_het)
    H_te = hetero_block(extractor, Xte_z, reg_te, valid_het)
    act_va, _ = compute_raw_activations(extractor, Xva_z)
    H_va = compute_regime_heterogeneity(act_va[:, N_GLOBAL:], valid_het,
                                        reg_trva[len(Xtr):])
    del act_va

    # AUDIT 9: independent float64 H recompute
    smp, val = compute_raw_activations(extractor, Xtrva_z[0:1])
    smp, val = smp[0, N_GLOBAL:], val[N_GLOBAL:]
    recompute_max = 0.0
    for m in np.linspace(0, N_HET - 1, 8).astype(int):
        ref = independent_heterogeneity_recompute(
            smp[m], val[m], reg_trva[0].astype(np.int64), K=K_CODES)
        recompute_max = max(recompute_max, abs(ref - float(H_tr[0, m])))
    assert recompute_max <= 1e-6, \
        f"AUDIT 9 FAILED: recompute diff {recompute_max}"
    audits["audit9_independent_h_recompute_maxdiff"] = float(recompute_max)
    log(f"  [AUDIT9] independent H recompute (8 kernels, sample 0): "
        f"max diff {recompute_max:.2e}")

    # AUDIT 10: per-feature valid-region invariance
    flip_idx = [np.flatnonzero(~valid_het[m]) for m in range(N_HET)]
    act_all = smp.copy()
    n_flip = 0
    for m in range(N_HET):
        if len(flip_idx[m]):
            act_all[m, flip_idx[m]] = ~act_all[m, flip_idx[m]]
            n_flip += int(len(flip_idx[m]))
    H_after = compute_regime_heterogeneity(
        act_all[None], valid_het, reg_trva[0:1])[0]
    assert np.array_equal(H_tr[0], H_after), \
        "AUDIT 10 FAILED: out-of-mask activations changed H"
    audits["audit10_valid_region"] = {
        "flipped_out_of_mask_activations": int(n_flip), "h_changed": False,
        "pass": True}
    log(f"  [AUDIT10] valid region: flipped {n_flip} out-of-mask "
        f"activations -> H unchanged exactly")

    mech = {"R2": h_stats(H_tr),
            "C1": h_stats(hetero_block(extractor, Xtrva_z, c1_trva, valid_het)),
            "C2": h_stats(hetero_block(extractor, Xtrva_z, c2_trva, valid_het))}

    # ---------------- 6. variants (R2, C1, C2) -----------------------------
    log("  [R2] SSL context + hard-VQ heterogeneity")
    results["R2"], preds["R2"] = run_variant(
        "R2", {"global": (G_trva, G_va, G_te), "het": (H_tr, H_va, H_te)},
        yva, yte, ytrva)
    results["R2"]["regime_diagnostics"] = occ
    results["R2"]["context_train"] = train_info
    results["R2"]["params"] = params

    for nm, r_trva, r_te in [("C1", c1_trva, c1_te),
                             ("C2", c2_trva, c2_te)]:
        log(f"  [{nm}] control")
        Hc_tr = hetero_block(extractor, Xtrva_z, r_trva, valid_het)
        act_va, _ = compute_raw_activations(extractor, Xva_z)
        Hc_va = compute_regime_heterogeneity(act_va[:, N_GLOBAL:], valid_het,
                                             r_trva[len(Xtr):])
        Hc_te = hetero_block(extractor, Xte_z, r_te, valid_het)
        results[nm], preds[nm] = run_variant(
            nm, {"global": (G_trva, G_va, G_te),
                 "het": (Hc_tr, Hc_va, Hc_te)}, yva, yte, ytrva)
        del act_va

    deltas = {f"R2-{k}": round(results["R2"]["test_macro_f1"]
                               - results[k]["test_macro_f1"], 4)
              for k in ("M0", "C1", "C2")}
    deltas["C1-C2"] = round(results["C1"]["test_macro_f1"]
                            - results["C2"]["test_macro_f1"], 4)

    # ---------------- 7. save artifacts ------------------------------------
    os.makedirs(os.path.join(OUT_DIR, "predictions"), exist_ok=True)
    with open(os.path.join(OUT_DIR, "predictions",
                           "gunpoint_seed42.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_index", "true_class"] +
                   ["M0", "R2", "C1", "C2"])
        for i in range(len(yte)):
            w.writerow([i, int(yte[i])] +
                       [int(preds[v][i]) for v in ("M0", "R2", "C1", "C2")])

    res = {
        "dataset": DATASET, "seed": SEED, "T": got["T"],
        "split": {"train": len(Xtr), "val": len(Xva), "test": len(Xte)},
        "n_classes": data["n_classes"], "n_features": N_FEATURES,
        "K_regimes": K_CODES,
        "config": {"encoder": ENCODER, "vq": VQ, "joint": JOINT,
                   "alphas": "logspace(-4,4,20)"},
        "provenance": audits["provenance"],
        "m0_reference": {"context3_canonical": M0_REF,
                         "tolerance": M0_TOL},
        "r2_haptics_reference": R2_HAPTICS_REF,
        "results": results, "deltas": deltas,
        "mechanistic_h": mech,
        "audits": audits,
        "runtime_s": round(time.time() - t_start, 1),
    }
    with open(os.path.join(OUT_DIR, "result.json"), "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(OUT_DIR, "audits.json"), "w") as f:
        json.dump(audits, f, indent=2)

    log(f"\n  [{DATASET}] done in {res['runtime_s']}s — " +
        " ".join(f"{v}={results[v]['test_macro_f1']:.4f}"
                 for v in ("M0", "R2", "C1", "C2")))
    log(f"  deltas: {deltas}")

    # verdict
    res_ = {v: results[v]["test_macro_f1"] for v in ("M0", "R2", "C1", "C2")}
    if res_["M0"] >= 0.99 and res_["R2"] >= 0.99:
        verdict = ("UNINFORMATIVE (ceiling-saturated): M0 and R2 both solve "
                   "the task; no headroom to assess the heterogeneity effect")
    elif res_["R2"] > res_["M0"] and res_["R2"] > res_["C1"] \
            and res_["R2"] > res_["C2"]:
        verdict = "POSITIVE"
    elif res_["R2"] <= res_["M0"]:
        verdict = "NEGATIVE"
    else:
        verdict = "PARTIAL"
    res["verdict"] = verdict
    with open(os.path.join(OUT_DIR, "result.json"), "w") as f:
        json.dump(res, f, indent=2)
    log(f"  VERDICT: {verdict}")

    from experiments.rcmkn_r2_gunpoint_seed42.report import write_report
    write_report(OUT_DIR, res)
    return res


if __name__ == "__main__":
    main()
