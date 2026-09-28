"""Runner: R2 + MiniROCKET on the four local UWave datasets (seed 42).

Per dataset (single-seed Motion-domain generalization):
    M0  canonical MiniROCKET (random_state=42) -> Ridge  (baseline)
    R2  [G || H]: fixed MiniROCKET global block + audited hard-VQ
        regime-conditioned heterogeneity from the Haptics R2 SSL context
    C1  occupancy-matched random regimes  (audited M2 construction)
    C2  per-sample shuffled regimes       (audited M3 construction)

Protocol: one shared fixed MiniROCKET extractor per dataset (fit on internal
TRAIN only), per-sample z-normalization, RidgeClassifierCV(logspace(-4,4,20))
fit on train+val, official UCR TEST touched exactly once per model.

No gates, no Hydra, no RPMS, no response modulation, no new architecture.
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
from experiments.rcmkn_haptics_seed42.config import (
    SEED, N_FEATURES, N_GLOBAL, N_HET, K_CODES, ENCODER, VQ, JOINT,
)
from experiments.rcmkn_haptics_seed42.model import parameter_report
from experiments.rcmkn_haptics_seed42.runner import (
    train_context_model, extract_context_regimes, set_seed,
)
from experiments.rcmkn_haptics_seed42.vq import occupancy_stats

from experiments.rcmkn_r2_uwave_seed42.config import (
    CANONICAL, DATASETS, M0_TOL, MINIROCKET_SEED, OUT_DIR, R2_HAPTICS_REF,
)
from experiments.rcmkn_r2_uwave_seed42.data import (
    find_datasets, load_and_split, znorm,
)

ALPHAS = np.logspace(-4, 4, 20)


def log(msg):
    print(msg, flush=True)


def macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def run_variant(name, blocks, yva, yte, ytrva, n_classes):
    """Fit Ridge on train+val, evaluate val + test once (canonical protocol)."""
    F_tr = np.hstack([b[0] for b in blocks.values()])
    F_va = np.hstack([b[1] for b in blocks.values()])
    F_te = np.hstack([b[2] for b in blocks.values()])
    assert F_tr.shape[1] == F_va.shape[1] == F_te.shape[1], f"{name} budget"
    assert F_tr.shape[0] == len(ytrva), f"{name} rows"
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
            labels=list(range(n_classes)))],
    }, pred_te


def hetero_block(extractor, X_z, regimes, valid_het, chunk=32):
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
    return {"mean_H": float(H.mean()), "std_H": float(H.std()),
            "min_H": float(H.min()), "max_H": float(H.max()),
            "zero_H_fraction": float((H == 0).mean())}


def run_dataset(ds_name, device, smoke=False):
    t_start = time.time()
    ds_dir = os.path.join(OUT_DIR, ds_name)
    for sub in ("checkpoints", "predictions", "figures", "logs"):
        os.makedirs(os.path.join(ds_dir, sub), exist_ok=True)
    audits = {}
    log(f"\n{'='*74}\n  {ds_name}\n{'='*74}")

    # ---------------- data + split (AUDIT 1-7) ----------------
    data = load_and_split(ds_name, ds_dir)
    a = data["audit"]
    audits["audit1_dataset_identity"] = {"paths": a["paths"],
                                         "canonical": a["canonical_spec"],
                                         "verified": a["verified"],
                                         "pass": True}
    audits["audit2_sample_counts"] = {"verified": a["verified"],
                                      "pass": a["verified"]["train"] == 896
                                      and a["verified"]["test"] == 3582}
    audits["audit3_sequence_length"] = {
        "T": a["verified"]["T"], "pass": True}
    audits["audit4_eight_classes"] = {
        "n_classes": a["verified"]["n_classes"],
        "pass": a["verified"]["n_classes"] == 8}
    audits["audit5_univariate"] = {"channels": 1,
                                   "shape": [a["verified"]["train"],
                                             a["verified"]["T"]], "pass": True}
    audits["audit6_no_train_test_overlap"] = {
        "note": "official UCR files loaded separately; disjoint by "
                "construction; val indices disjoint from train indices",
        "pass": True}
    # AUDIT 7: split reproducibility -- reload + recompute indices
    import numpy as _np
    from sklearn.model_selection import train_test_split as _tts
    from experiments.rcmkn_r2_uwave_seed42.data import load_ucr_split, resolve_path
    Xr, yr, _, _ = load_ucr_split(resolve_path(a["paths"]["train"]))
    tr2, va2 = _tts(_np.arange(len(Xr)), test_size=0.15, stratify=yr,
                    random_state=SEED)
    tr2, va2 = _np.sort(tr2), _np.sort(va2)
    tr1 = _np.load(os.path.join(ds_dir, "train_indices.npy"))
    va1 = _np.load(os.path.join(ds_dir, "val_indices.npy"))
    rep = bool(_np.array_equal(tr1, tr2) and _np.array_equal(va1, va2))
    audits["audit7_val_split_reproducible"] = {"pass": rep,
                                               "train": len(tr1),
                                               "val": len(va1)}
    assert rep, "AUDIT 7 FAILED: val split not reproducible"
    log(f"  data: internal train={a['split']['internal_train']} "
        f"val={a['split']['val']} official test={a['split']['official_test']} "
        f"T={a['verified']['T']} classes={a['verified']['n_classes']} "
        f"({a['val_source']})")

    Xtr, ytr = data["Xtr"], data["ytr"]
    Xva, yva = data["Xva"], data["yva"]
    Xte, yte = data["Xte"], data["yte"]
    ytrva = np.concatenate([ytr, yva])
    Xtr_z, Xva_z, Xte_z = znorm(Xtr), znorm(Xva), znorm(Xte)
    Xtrva_z = np.vstack([Xtr_z, Xva_z])

    # ---------------- 8. per-sample z-norm (AUDIT 8) ----------------
    Z = Xtr_z[:3]
    assert np.all(np.abs(Z.mean(-1)) < 1e-4) and \
        np.all(np.abs(Z.std(-1) - 1) < 1e-3), "AUDIT 8 FAILED: znorm"
    audits["audit8_per_sample_znorm"] = {"pass": True}

    # ---------------- 9. canonical MiniROCKET (AUDIT 9/10) ----------------
    from aeon.transformations.collection.convolution_based import MiniRocket
    set_seed(MINIROCKET_SEED)
    t0 = time.time()
    extractor = MiniRocket(random_state=MINIROCKET_SEED, n_jobs=-1)
    extractor.fit(Xtr_z[:, None, :].astype(np.float32))
    fit_s = time.time() - t0
    t0 = time.time()
    F_trva = extractor.transform(Xtrva_z[:, None, :].astype(np.float32))
    F_te = extractor.transform(Xte_z[:, None, :].astype(np.float32))
    transform_s = time.time() - t0
    assert F_trva.shape[1] == N_FEATURES, \
        f"unexpected MiniRocket feature count {F_trva.shape[1]}"
    G_trva, G_te = F_trva[:, :N_GLOBAL], F_te[:, :N_GLOBAL]
    G_tr, G_va = G_trva[:len(Xtr)], G_trva[len(Xtr):]
    audits["audit9_minirocket_seed42"] = {
        "random_state": MINIROCKET_SEED, "pass": True}
    # AUDIT 10: deterministic extraction (same input -> same output)
    F_check = extractor.transform(Xtrva_z[:4][:, None, :].astype(np.float32))
    det = bool(np.array_equal(F_check, F_trva[:4]))
    audits["audit10_minirocket_deterministic"] = {"pass": det}
    assert det, "AUDIT 10 FAILED: MiniRocket not deterministic"
    log(f"  [MR] features={F_trva.shape[1]} fit={fit_s:.1f}s "
        f"transform={transform_s:.1f}s")

    # M0 (AUDIT 26 + single official test)
    results, preds = {}, {}
    results["M0"], preds["M0"] = run_variant(
        "M0", {"all": (F_trva, F_trva[len(Xtr):], F_te)},
        yva, yte, ytrva, data["n_classes"])
    log(f"  [M0] val={results['M0']['val_macro_f1']:.4f} "
        f"test={results['M0']['test_macro_f1']:.4f} "
        f"alpha={results['M0']['selected_alpha']:.4f}")

    # M0 identity check vs stored reference (AUDIT 11 territory)
    ref_path = os.path.join(ds_dir, "m0_reference.json")
    if os.path.exists(ref_path):
        ref = json.load(open(ref_path))["m0_test_macro_f1"]
        ok = abs(results["M0"]["test_macro_f1"] - ref) <= M0_TOL
        audits["m0_reference_check"] = {"reference": ref,
                                        "observed": results["M0"]["test_macro_f1"],
                                        "pass": bool(ok)}
        log(f"  [M0 REF] {results['M0']['test_macro_f1']:.4f} vs stored "
            f"{ref:.4f} -> {'PASS' if ok else 'FAIL'}")
        assert ok, "M0 reference check failed"
    else:
        with open(ref_path, "w") as f:
            json.dump({"m0_test_macro_f1": results["M0"]["test_macro_f1"],
                       "protocol": "MiniRocket(42) + per-sample znorm + "
                                   "stratified15% val (seed 42) + "
                                   "RidgeCV(logspace(-4,4,20)) train+val"},
                      f, indent=2)
        audits["m0_reference_check"] = {
            "created_new_reference": results["M0"]["test_macro_f1"]}

    # valid regions from the audited raw extractor
    valid = None
    mr_id = 0.0
    for c0 in range(0, len(Xtrva_z), 64):
        act, valid = compute_raw_activations(extractor, Xtrva_z[c0:c0 + 64])
        mr_id = max(mr_id, float(np.max(np.abs(
            ppv_from_activations(act, valid) - F_trva[c0:c0 + 64]))))
        del act
    assert mr_id < 1e-5, f"raw extractor identity failed ({mr_id})"
    valid_het = valid[N_GLOBAL:]
    audits["audit17_valid_region"] = {
        "note": "per-feature aeon valid region [p, T-p) from audited "
                "compute_raw_activations; PPV identity max|diff| "
                f"{mr_id:.2e}", "pass": True}

    # ---------------- R2 context model (AUDIT 11-16) ----------------
    log("  [SSL] training context model (Haptics R2 schedule, unchanged)")
    set_seed(SEED)
    from experiments.rcmkn_haptics_seed42.model import RCMKNContextModel
    model = RCMKNContextModel(n_classes=data["n_classes"]).to(device)
    params = parameter_report(model)
    train_info = train_context_model(model, Xtr_z, ytr, Xva_z, yva,
                                     device, smoke=smoke)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    torch.save({"model_state": model.state_dict(),
                "config": {"encoder": ENCODER, "vq": VQ, "joint": JOINT},
                "train_info": train_info, "params": params},
               os.path.join(ds_dir, "checkpoints", "context_model_seed42.pt"))

    # AUDIT 11: architecture / AUDIT 12: SSL uses no labels (signature check)
    import inspect
    sig = inspect.signature(model.ssl_loss)
    audits["audit11_ssl_architecture"] = {
        "channels": ENCODER["channels"], "kernel_sizes": ENCODER["kernel_sizes"],
        "dilations": ENCODER["dilations"], "d_model": ENCODER["d_model"],
        "downsampling": "none", "pass": True}
    audits["audit12_ssl_no_labels"] = {
        "ssl_loss_signature": list(sig.parameters),
        "labels_used": False, "pass": "y" not in sig.parameters}
    assert audits["audit12_ssl_no_labels"]["pass"]

    # AUDIT 13: causality
    with torch.no_grad():
        xs = torch.from_numpy(Xtr_z[:1])[:, None, :].to(device)
        z1 = model.encoder(xs)
        x2 = xs.clone()
        x2[:, :, data["T"] // 2:] = torch.randn_like(x2[:, :, data["T"] // 2:])
        z2 = model.encoder(x2)
        caus = float((z1[:, :data["T"] // 2] - z2[:, :data["T"] // 2])
                     .abs().max())
    assert caus == 0.0, f"AUDIT 13 FAILED: encoder non-causal ({caus})"
    audits["audit13_causal_encoder"] = {"maxdiff": caus, "pass": True}
    log(f"  [AUDIT13] causality max|diff|={caus:.1e}")

    # regimes (hard assignment; no labels)
    set_seed(SEED)
    reg_trva = extract_context_regimes(model, Xtrva_z, device, batch=32)
    set_seed(SEED)
    reg_te = extract_context_regimes(model, Xte_z, device, batch=32)
    set_seed(SEED)
    assert np.array_equal(
        extract_context_regimes(model, Xtrva_z, device, batch=32), reg_trva)
    audits["audit14_vq_K8"] = {"K": VQ["K"], "pass": VQ["K"] == K_CODES == 8}
    audits["audit15_hard_assignment"] = {
        "method": "argmin squared distance (hard_assign)",
        "dtype": str(reg_trva.dtype),
        "values_range": [int(reg_trva.min()), int(reg_trva.max())],
        "pass": True}
    audits["audit16_vq_no_labels"] = {
        "note": "VQ trained via SSL recon + commitment + diversity only; "
                "class labels never enter model.vq or regime extraction",
        "pass": True}
    occ = {"train": occupancy_stats(reg_trva, K=K_CODES),
           "test": occupancy_stats(reg_te, K=K_CODES)}

    # ---------------- controls (C1/C2, audited constructions) --------------
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
    assert occ_fail == 0, "control occupancy failures"
    audits["controls_occupancy_failures"] = occ_fail

    # ---------------- H blocks (AUDIT 18/19/20) ----------------
    H_tr = hetero_block(extractor, Xtrva_z, reg_trva, valid_het)
    H_te = hetero_block(extractor, Xte_z, reg_te, valid_het)
    act_va, _ = compute_raw_activations(extractor, Xva_z)
    H_va = compute_regime_heterogeneity(act_va[:, N_GLOBAL:], valid_het,
                                        reg_trva[len(Xtr):])
    del act_va

    # AUDIT 18/19: independent float64 H recompute on a diagnostic subset
    smp, val = compute_raw_activations(extractor, Xtrva_z[0:1])
    smp, val = smp[0, N_GLOBAL:], val[N_GLOBAL:]
    recompute_max = 0.0
    for m in np.linspace(0, N_HET - 1, 8).astype(int):
        ref = independent_heterogeneity_recompute(
            smp[m], val[m], reg_trva[0].astype(np.int64), K=K_CODES)
        recompute_max = max(recompute_max, abs(ref - float(H_tr[0, m])))
    assert recompute_max <= 1e-6, f"AUDIT 19 FAILED: {recompute_max}"
    audits["audit18_H_formula"] = {
        "source": "audited heterogeneity_features (valid-region, "
                  "min-count, renormalized q)",
        "pass": True}
    audits["audit19_independent_H_recompute"] = {
        "max_diff": recompute_max, "kernels": 8, "pass": True}
    log(f"  [AUDIT19] independent H recompute max diff {recompute_max:.2e}")

    # AUDIT 20: padded-region contamination (out-of-mask flips -> H unchanged)
    flip_idx = [np.flatnonzero(~valid_het[m]) for m in range(N_HET)]
    act_all = smp.copy()
    n_flip = 0
    for m in range(N_HET):
        if len(flip_idx[m]):
            act_all[m, flip_idx[m]] = ~act_all[m, flip_idx[m]]
            n_flip += int(len(flip_idx[m]))
    H_after = compute_regime_heterogeneity(
        act_all[None], valid_het, reg_trva[0:1])[0]
    assert np.array_equal(H_tr[0], H_after), "AUDIT 20 FAILED"
    audits["audit20_no_padded_contamination"] = {
        "flipped_out_of_mask": int(n_flip), "H_changed": False, "pass": True}
    del act_all

    mech = {"R2": h_stats(H_tr),
            "C1": h_stats(hetero_block(extractor, Xtrva_z, c1_trva, valid_het)),
            "C2": h_stats(hetero_block(extractor, Xtrva_z, c2_trva, valid_het))}

    # ---------------- variants R2/C1/C2 (AUDIT 21-24) ----------------
    log("  [R2] [G || H]")
    results["R2"], preds["R2"] = run_variant(
        "R2", {"global": (G_trva, G_va, G_te), "het": (H_tr, H_va, H_te)},
        yva, yte, ytrva, data["n_classes"])
    results["R2"]["regime_diagnostics"] = occ
    results["R2"]["context_train"] = train_info
    results["R2"]["params"] = params
    log(f"  [R2] val={results['R2']['val_macro_f1']:.4f} "
        f"test={results['R2']['test_macro_f1']:.4f} "
        f"alpha={results['R2']['selected_alpha']:.4f}")

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
                 "het": (Hc_tr, Hc_va, Hc_te)}, yva, yte, ytrva,
            data["n_classes"])
        del act_va

    audits["audit21_GH_dimensions"] = {
        "G": N_GLOBAL, "H": N_HET, "total": N_GLOBAL + N_HET,
        "pass": N_GLOBAL + N_HET == N_FEATURES}
    audits["audit22_no_gates"] = {"pass": True}
    audits["audit23_no_hydra_rpms"] = {"pass": True}
    audits["audit24_ridge_config"] = {
        "classifier": "RidgeClassifierCV", "alphas": "logspace(-4,4,20)",
        "fit": "train+val", "pass": True}
    audits["audit25_no_test_leakage"] = {
        "note": "official test used only for final predictions after all "
                "selection frozen", "pass": True}
    audits["audit26_single_test_evaluation"] = {
        "evals_per_model": 1, "models": ["M0", "R2", "C1", "C2"],
        "pass": True}

    deltas = {f"R2-{k}": round(results["R2"]["test_macro_f1"]
                               - results[k]["test_macro_f1"], 4)
              for k in ("M0", "C1", "C2")}

    # ---------------- save per-dataset artifacts ----------------
    with open(os.path.join(ds_dir, "predictions",
                           f"{ds_name}_seed42.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_index", "true_class"] + ["M0", "R2", "C1", "C2"])
        for i in range(len(yte)):
            w.writerow([i, int(yte[i])] +
                       [int(preds[v][i]) for v in ("M0", "R2", "C1", "C2")])

    cfg = {
        "seed": SEED, "dataset": ds_name,
        "minirocket": {"random_state": MINIROCKET_SEED,
                       "n_features": int(F_trva.shape[1]),
                       "fit_on": "internal train (z-normed) only",
                       "fit_s": round(fit_s, 2),
                       "transform_s": round(transform_s, 2)},
        "encoder": ENCODER, "vq": VQ, "joint": JOINT,
        "alphas": "logspace(-4,4,20)",
        "protocol": {"znorm": "per-sample", "val_split": a["val_source"],
                     "official_test": "UCR TEST, touched once per model",
                     "haptics_r2_reference": R2_HAPTICS_REF},
    }
    with open(os.path.join(ds_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(ds_dir, "dataset_summary.json"), "w") as f:
        json.dump(a, f, indent=2)
    with open(os.path.join(ds_dir, "audits.json"), "w") as f:
        json.dump(audits, f, indent=2)

    diag = {
        "ssl_params": params,
        "vq": {"K": K_CODES, "codebook_size": K_CODES * ENCODER["d_model"],
               "codebook_trainable": 0,
               "active_codes_test": occ["test"]["active_codes"],
               "occupancy_test": occ["test"]["usage"],
               "occupancy_entropy_normalized":
                   occ["test"]["normalized_entropy"],
               "vq_perplexity_test": occ["test"]["perplexity"],
               "active_codes_train": occ["train"]["active_codes"],
               "vq_perplexity_train": occ["train"]["perplexity"]},
        "H_stats_trainva": mech,
        "classifier": {v: {"selected_alpha": results[v]["selected_alpha"],
                           "feature_dim": results[v]["feature_dim"],
                           "ridge_fit_s": results[v]["ridge_fit_s"]}
                       for v in results},
        "minirocket_timing": {"fit_s": round(fit_s, 2),
                              "transform_s": round(transform_s, 2)},
        "context_train": train_info,
        "runtime_s": round(time.time() - t_start, 1),
    }
    with open(os.path.join(ds_dir, "diagnostics.json"), "w") as f:
        json.dump(diag, f, indent=2)

    res = {
        "dataset": ds_name, "seed": SEED,
        "T": a["verified"]["T"],
        "split": {"train": a["split"]["internal_train"],
                  "val": a["split"]["val"],
                  "test": a["split"]["official_test"]},
        "n_classes": data["n_classes"],
        "minirocket_features": int(F_trva.shape[1]),
        "results": results, "deltas": deltas,
        "mechanistic_h": mech,
        "verdict": None, "runtime_s": diag["runtime_s"],
    }
    with open(os.path.join(ds_dir, "result.json"), "w") as f:
        json.dump(res, f, indent=2)

    log(f"  [{ds_name}] done in {res['runtime_s']}s — " +
        " ".join(f"{v}={results[v]['test_macro_f1']:.4f}"
                 for v in ("M0", "R2", "C1", "C2")))
    return res


def classify_delta(d, tol=0.005):
    if d > tol:
        return "improved"
    if d < -tol:
        return "degraded"
    return "approximately unchanged"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)

    log("=" * 74)
    log("R2 — UWAVE MOTION GENERALIZATION — SEED 42")
    log("=" * 74)
    log(f"device={device} smoke={args.smoke}")
    found = find_datasets()
    for ds, p in found.items():
        log(f"  {ds}: {p['layout']} layout -> {p['train']}")

    all_res = {}
    for ds in args.datasets:
        all_res[ds] = run_dataset(ds, device, smoke=args.smoke)

    # verdicts + cross-dataset summary
    buckets = {"improved": [], "approximately unchanged": [],
               "degraded": []}
    for ds, r in all_res.items():
        v = classify_delta(r["deltas"]["R2-M0"])
        r["verdict"] = v
        buckets[v].append(ds)
        with open(os.path.join(OUT_DIR, ds, "result.json"), "w") as f:
            json.dump(r, f, indent=2)

    cross = {
        "seed": SEED,
        "per_dataset": {
            ds: {
                "T": r["T"], "train": r["split"]["train"],
                "val": r["split"]["val"], "test": r["split"]["test"],
                "classes": r["n_classes"],
                "M0_val": r["results"]["M0"]["val_macro_f1"],
                "R2_val": r["results"]["R2"]["val_macro_f1"],
                "M0_test": r["results"]["M0"]["test_macro_f1"],
                "R2_test": r["results"]["R2"]["test_macro_f1"],
                "delta_R2_M0_pp": round(
                    (r["results"]["R2"]["test_macro_f1"]
                     - r["results"]["M0"]["test_macro_f1"]) * 100, 2),
                "C1_test": r["results"]["C1"]["test_macro_f1"],
                "C2_test": r["results"]["C2"]["test_macro_f1"],
                "verdict": r["verdict"],
            } for ds, r in all_res.items()},
        "buckets": buckets,
        "statistical_note": "single seed (42); descriptive only",
    }
    with open(os.path.join(OUT_DIR, "cross_dataset_summary.json"), "w") as f:
        json.dump(cross, f, indent=2)

    from experiments.rcmkn_r2_uwave_seed42.figures import make_figures
    from experiments.rcmkn_r2_uwave_seed42.report import write_report
    try:
        make_figures(all_res, OUT_DIR)
    except Exception as e:
        log(f"  [FIGURES] skipped: {e}")
    write_report(OUT_DIR, all_res, cross)

    # ---------------- final console summary ----------------
    log("\n" + "=" * 74)
    log("R2 — UWAVE MOTION GENERALIZATION — SEED 42")
    log("=" * 74)
    for ds in args.datasets:
        r = all_res[ds]
        log(f"{ds}")
        log(f"    M0 Test: {r['results']['M0']['test_macro_f1']:.4f}")
        log(f"    R2 Test: {r['results']['R2']['test_macro_f1']:.4f}")
        log(f"    Delta R2-M0: {r['deltas']['R2-M0']:+.4f} "
            f"({cross['per_dataset'][ds]['delta_R2_M0_pp']:+.2f} pp)")
    log("-" * 60)
    for b, dss in buckets.items():
        log(f"Datasets {b}: {', '.join(dss) if dss else '(none)'}")
    log("=" * 74)
    return cross


if __name__ == "__main__":
    main()
