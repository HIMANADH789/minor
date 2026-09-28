"""SSL-context transfer experiment (seed 42), part 2: ECG5000_BAL + CWRU_BAL.

Identical methodology to experiments/rcmkn_ssl_context_transfer_seed42
(Phoneme / ECG5000_UNBAL / CWRU_UNBAL) — same audited components, same
variants (R0/R2/C1/C2), same audits — applied to the two remaining
canonical NPZ datasets. NO M0, NO Hydra, NO architecture changes.

R0 provenance:
    ECG5000_BAL: frozen seed-42 DRTN checkpoint from the AUDITED 3-seed
        study (results/drtn_ecg5000_bal_3seed/seed42/checkpoint.pt,
        val MF1 0.8638@ep21) -> R0 gate 0.6748 (audited M1, tolerance 0.0011).
    CWRU_BAL: frozen transfer-screen checkpoint (no audited reference
        exists) -> in-run R0 becomes the first audited reference; the
        pre-audit 0.9930 is descriptive only.

Canonical M0 references (NOT evaluated here): ECG5000_BAL 0.6553,
CWRU_BAL 0.9947.

Usage:
    python -m experiments.rcmkn_ssl_context_important2_seed42.runner \
        [--datasets ECG5000_BAL CWRU_BAL] [--smoke]
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

OUT_DIR = os.path.join(ROOT, "results", "rcmkn_ssl_context_important2_seed42")
ALPHAS = np.logspace(-4, 4, 20)

EXPECTED = {
    "ECG5000_BAL": {"T": 140, "n_classes": 5, "train": 5226, "val": 923,
                    "test": 1000},
    "CWRU_BAL": {"T": 1024, "n_classes": 4, "train": 2727, "val": 482,
                 "test": 567},
}
R0_GATES = {"ECG5000_BAL": 0.6748}   # audited 3-seed M1 (seed 42)
R0_TOL = 0.0011
M0_REFS = {"ECG5000_BAL": 0.6553, "CWRU_BAL": 0.9947}  # context only


def log(msg):
    print(msg, flush=True)


def arr_hash(a):
    return hashlib.sha1(np.ascontiguousarray(a, dtype=np.int64).tobytes()).hexdigest()[:16]


def macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_drtn_for_dataset(ds_name, data, device):
    """Frozen DRTN R5 per dataset (audited provenance, see module docstring)."""
    from models.drtn.model import build_model
    if ds_name == "ECG5000_BAL":
        path = os.path.join(ROOT, "results", "drtn_ecg5000_bal_3seed",
                            "seed42", "checkpoint.pt")
        ck = torch.load(path, map_location=device, weights_only=False)
        cfg = ck["config"]
        model = build_model(
            "R5", c_in=1, n_classes=data["n_classes"], d_model=cfg["d_model"],
            n_codes=cfg["n_codes"], tau=cfg["tau"], ema_decay=cfg["ema_decay"],
            beta=cfg["beta_commit"], lam_div=cfg["lam_div"],
            dead_threshold=cfg["dead_threshold"],
            revival_patience=cfg["revival_patience"],
            traj_layers=cfg["trajectory"]["layers"],
            traj_heads=cfg["trajectory"]["heads"],
            traj_ffn=cfg["trajectory"]["ffn"],
            traj_dropout=cfg["trajectory"]["dropout"])
        model.load_state_dict(ck["model_state"])
        model.to(device).eval()
        info = {"best_val_mf1": float(ck["best_val_mf1"]), "epoch": ck["epoch"],
                "source": "results/drtn_ecg5000_bal_3seed/seed42 (audited 3-seed)"}
        log(f"  [DRTN] loaded audited 3-seed checkpoint (val MF1 "
            f"{info['best_val_mf1']:.4f}, epoch {info['epoch']})")
        return model, info
    # CWRU_BAL: frozen transfer-screen checkpoint (train_or_load_drtn loads,
    # never trains, since the checkpoint file exists)
    return transfer.train_or_load_drtn(ds_name, data, device, smoke=False)


def run_variant(name, blocks, yva, yte, ytrva):
    """Fit Ridge on train+val; evaluate val + test once. AUDIT 17."""
    F_tr = np.hstack([b[0] for b in blocks.values()])
    F_va = np.hstack([b[1] for b in blocks.values()])
    F_te = np.hstack([b[2] for b in blocks.values()])
    assert F_tr.shape[1] == F_va.shape[1] == F_te.shape[1] == N_FEATURES, \
        f"{name}: budget {F_tr.shape[1]} != {N_FEATURES} (AUDIT 3)"
    assert F_tr.shape[0] == len(ytrva), f"{name}: F_tr rows != trainva"
    ridge = RidgeClassifierCV(alphas=ALPHAS)
    t0 = time.time()
    ridge.fit(F_tr, ytrva)                    # train + validation
    pred_va = ridge.predict(F_va)
    pred_te = ridge.predict(F_te)
    return {
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
    }, pred_te


def hetero_blocks_multi(extractor, X_z, regimes_list, valid_het, chunk=64):
    """Valid-region H for SEVERAL (N, T) regime arrays in ONE bounded
    activation pass (memory-safe: per-chunk activations, freed immediately).
    Returns list of (N, 4998) float64 arrays, order = regimes_list."""
    N = len(X_z)
    outs = [np.empty((N, N_GLOBAL), dtype=np.float64) for _ in regimes_list]
    for c0 in range(0, N, chunk):
        c1 = min(c0 + chunk, N)
        act, _ = compute_raw_activations(extractor, X_z[c0:c1])
        act_het = act[:, N_GLOBAL:]
        for o, regs in zip(outs, regimes_list):
            o[c0:c1] = compute_regime_heterogeneity(act_het, valid_het,
                                                    regs[c0:c1])
        del act, act_het
    return outs


def h_stats(H):
    return {"mean_H": float(H.mean()), "median_H": float(np.median(H)),
            "max_H": float(H.max()), "fraction_nonzero": float((H > 0).mean())}


def run_dataset(ds_name, device, smoke=False):
    ds_dir = os.path.join(OUT_DIR, ds_name)
    os.makedirs(ds_dir, exist_ok=True)
    os.makedirs(os.path.join(ds_dir, "diagnostics"), exist_ok=True)
    audits = {}
    t_start = time.time()
    results = {}

    # ---------------- AUDIT 1: canonical configuration ----------------------
    data = transfer.load_any_dataset(ds_name)
    Xtr, ytr = data["Xtr"], data["ytr"]
    Xva, yva = data["Xva"], data["yva"]
    Xte, yte = data["Xte"], data["yte"]
    ytrva = np.concatenate([ytr, yva])
    exp = EXPECTED[ds_name]
    got = {"T": int(Xtr.shape[1]), "n_classes": data["n_classes"],
           "train": len(Xtr), "val": len(Xva), "test": len(Xte)}
    assert got == exp, f"AUDIT 1 FAILED {ds_name}: {got} != {exp}"
    log(f"\n=== {ds_name}: train={got['train']} val={got['val']} "
        f"test={got['test']} T={got['T']} n_classes={got['n_classes']} "
        f"dim=1 classes={sorted(set(int(v) for v in np.unique(ytrva)))} ===")
    audits["audit1_config"] = {"expected": exp, "actual": got, "pass": True}

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
    # BUG GUARD 8: validation responses explicitly materialized and used.
    G_tr, G_va, G_te = (F_trva_mr[:, :N_GLOBAL], F_va_mr[:, :N_GLOBAL],
                        F_te_mr[:, :N_GLOBAL])
    assert not np.isnan(F_trva_mr).any() and not np.isinf(F_trva_mr).any()
    log(f"  [AUDIT3/4] global = features [0, {N_GLOBAL}), het = "
        f"[{N_GLOBAL}, {N_FEATURES}); feature-axis slicing (rows untouched: "
        f"{F_trva_mr.shape[0]})")

    # ---------------- AUDIT 2: raw-extractor identity -----------------------
    valid = None
    mr_id, mr_mean = 0.0, 0.0
    for c0 in range(0, len(Xtrva_z), 64):
        act, valid = compute_raw_activations(extractor, Xtrva_z[c0:c0 + 64])
        d = np.abs(ppv_from_activations(act, valid) - F_trva_mr[c0:c0 + 64])
        mr_id = max(mr_id, float(d.max()))
        mr_mean = max(mr_mean, float(d.mean()))
        del act
    assert mr_id < 1e-5, f"AUDIT 2 FAILED: extractor identity {mr_id}"
    audits["audit2_extractor_identity"] = {"max_abs_diff": float(mr_id),
                                           "mean_abs_diff": float(mr_mean)}
    log(f"  [AUDIT2] raw-extractor PPV vs aeon: max|diff|={mr_id:.2e} "
        f"mean|diff|={mr_mean:.2e}")
    valid_het = valid[N_GLOBAL:]

    # ---------------- R0: audited DRTN reference ----------------------------
    log(f"  [R0] frozen DRTN checkpoint for {ds_name}")
    drtn, drtn_info = load_drtn_for_dataset(ds_name, data, device)
    set_seed(SEED)
    reg0_trva = extract_drtn_regimes(drtn, Xtrva_z, device=device)
    reg0_te = extract_drtn_regimes(drtn, Xte_z, device=device)
    sd1 = {k: v.clone() for k, v in drtn.state_dict().items()}
    _ = extract_drtn_regimes(drtn, Xte_z[:2], device=device)
    assert all(torch.equal(sd1[k], drtn.state_dict()[k]) for k in sd1), \
        "DRTN checkpoint mutated during extraction"
    # chunked H for all three splits (memory-safe)
    H0_trva = hetero_blocks_multi(extractor, Xtrva_z, [reg0_trva], valid_het)[0]
    H0_va = hetero_blocks_multi(extractor, Xva_z, [reg0_trva[len(Xtr):]],
                                valid_het)[0]
    H0_te = hetero_blocks_multi(extractor, Xte_z, [reg0_te], valid_het)[0]
    res0, pred0 = run_variant("R0", {"global": (G_tr, G_va, G_te),
                                     "het": (H0_trva, H0_va, H0_te)},
                              yva, yte, ytrva)
    results["R0"] = res0
    audits["r0_reference"] = {"test_macro_f1": res0["test_macro_f1"],
                              "drtn_ckpt": drtn_info}
    if ds_name in R0_GATES and not smoke:
        gate = abs(res0["test_macro_f1"] - R0_GATES[ds_name]) <= R0_TOL
        audits["r0_gate"] = {"reference": R0_GATES[ds_name],
                             "pass": bool(gate)}
        log(f"  [R0 GATE] {res0['test_macro_f1']} vs audited "
            f"{R0_GATES[ds_name]} -> {'PASS' if gate else 'FAIL'}")
        assert gate, f"R0 failed reproduction on {ds_name}. STOP."
    else:
        log(f"  [R0] test MF1 {res0['test_macro_f1']} (first audited "
            f"reference for {ds_name})")

    # ---------------- SSL context model (R2/C1/C2 source) -------------------
    log("  [SSL] training context model (Haptics R2 schedule, unchanged)")
    set_seed(SEED)
    model = core.build_context_model(data["n_classes"], device)
    train_info, params = core.train_ssl_context(model, Xtr_z, ytr, Xva_z,
                                                yva, device, smoke=smoke)
    core.freeze_model(model)
    torch.save({"model_state": model.state_dict(),
                "config": {"encoder": ENCODER, "vq": VQ, "joint": JOINT},
                "train_info": train_info, "params": params},
               os.path.join(ds_dir, "context_model_seed42.pt"))
    log(f"  [SSL] {train_info} params={params}")

    # AUDIT 8: causality
    with torch.no_grad():
        xs = torch.from_numpy(Xtr_z[:1])[:, None, :].to(device)
        z1 = model.encoder(xs)
        x2 = xs.clone()
        x2[:, :, T // 2:] = torch.randn_like(x2[:, :, T // 2:])
        z2 = model.encoder(x2)
        caus = float((z1[:, :T // 2] - z2[:, :T // 2]).abs().max())
    assert caus == 0.0, f"AUDIT 8 FAILED: non-causal ({caus})"
    audits["audit8_causality_maxdiff"] = caus
    log(f"  [AUDIT8] encoder causality: exactly {caus:.1e}")

    reg2_trva = core.context_regimes(model, Xtrva_z, device)
    reg2_te = core.context_regimes(model, Xte_z, device)
    occ2 = {"train": core.occupancy(reg2_trva), "test": core.occupancy(reg2_te)}
    audits["regime_diagnostics"] = occ2
    log(f"  [DIAG] VQ train occupancy: active={occ2['train']['active_codes']} "
        f"entropy={occ2['train']['normalized_entropy']:.3f} "
        f"perplexity={occ2['train']['perplexity']:.2f} "
        f"dominant={occ2['train']['dominant_fraction']:.3f}")

    # ---------------- C1/C2 controls (audited constructions) ----------------
    c1_trva = create_random_regime_control(reg2_trva, seed=SEED)
    c1_te = create_random_regime_control(reg2_te, seed=SEED)
    c2_trva = create_shuffled_regime_control(reg2_trva, seed=SEED)
    c2_te = create_shuffled_regime_control(reg2_te, seed=SEED)

    # AUDIT 13/14: per-sample occupancy preservation
    occ_fail = 0
    for ctrl, src in [(c1_trva, reg2_trva), (c2_trva, reg2_trva),
                      (c1_te, reg2_te), (c2_te, reg2_te)]:
        for i in range(len(src)):
            if not np.array_equal(np.bincount(ctrl[i], minlength=K_CODES),
                                  np.bincount(src[i], minlength=K_CODES)):
                occ_fail += 1
    assert occ_fail == 0, f"AUDIT 13/14 FAILED: {occ_fail} occupancy failures"
    audits["audit13_14_occupancy_failures"] = occ_fail

    # AUDIT 15/16: distinct arrays, hashes, no shared memory
    reg_audit = {
        "C1": {"hash_trainva": arr_hash(c1_trva), "hash_test": arr_hash(c1_te),
               "shares_memory_with_r2": bool(np.shares_memory(c1_trva, reg2_trva))},
        "C2": {"hash_trainva": arr_hash(c2_trva), "hash_test": arr_hash(c2_te),
               "shares_memory_with_r2": bool(np.shares_memory(c2_trva, reg2_trva))},
    }
    c1c2 = int((c1_trva != c2_trva).sum())
    reg_audit["C1_vs_C2"] = {"differing_positions": c1c2,
                             "fraction": c1c2 / c1_trva.size,
                             "identical": bool(np.array_equal(c1_trva, c2_trva))}
    audits["audit15_16_regime_arrays"] = reg_audit
    assert not reg_audit["C1"]["shares_memory_with_r2"]
    assert not reg_audit["C2"]["shares_memory_with_r2"]
    log(f"  [AUDIT13/14/15/16] controls: occupancy 0 failures; C1 "
        f"{reg_audit['C1']['hash_trainva'][:8]} vs C2 "
        f"{reg_audit['C2']['hash_trainva'][:8]} ({c1c2 / c1_trva.size:.1%} "
        f"differ); no shared memory")

    # ---------------- H blocks: ONE activation pass for R2/C1/C2 ------------
    H2_tr, C1_tr, C2_tr = hetero_blocks_multi(
        extractor, Xtrva_z, [reg2_trva, c1_trva, c2_trva], valid_het)
    H2_va, C1_va, C2_va = hetero_blocks_multi(
        extractor, Xva_z, [reg2_trva[len(Xtr):], c1_trva[len(Xtr):],
                           c2_trva[len(Xtr):]], valid_het)
    H2_te, C1_te, C2_te = hetero_blocks_multi(
        extractor, Xte_z, [reg2_te, c1_te, c2_te], valid_het)

    # ---------------- AUDIT 6: independent H recompute ----------------------
    smp, val = compute_raw_activations(extractor, Xtrva_z[:1])
    smp = smp[0, N_GLOBAL:]
    recompute_max = 0.0
    for m in np.linspace(0, N_GLOBAL - 1, 8).astype(int):
        ref = independent_heterogeneity_recompute(
            smp[m].astype(bool), val[N_GLOBAL:][m].astype(bool),
            reg2_trva[0].astype(np.int64), K=K_CODES)
        recompute_max = max(recompute_max, abs(ref - float(H2_tr[0, m])))
    tol = 0.05 if ds_name == "ECG5000_BAL" else 1e-6
    # ECG5000_BAL T=140: the reference recompute applies a min_occupancy
    # filter (ceil(0.01*140)=2) that the production code does not; the
    # tolerance accommodates that structural difference (see transfer run).
    assert recompute_max <= tol, f"AUDIT 6 FAILED: {recompute_max}"
    audits["audit6_independent_h_recompute_maxdiff"] = float(recompute_max)
    log(f"  [AUDIT6] independent float64 H recompute: max diff "
        f"{recompute_max:.2e}")

    # ---------------- AUDIT 7: per-feature valid region ---------------------
    H_before = H2_tr[0].copy()
    act_all, _ = compute_raw_activations(extractor, Xtrva_z[:1])
    act_all = act_all[0, N_GLOBAL:].copy()
    n_flip = 0
    for m in range(N_GLOBAL):
        idx = np.flatnonzero(~val[N_GLOBAL:][m])
        if len(idx):
            act_all[m, idx] = ~act_all[m, idx]
            n_flip += int(len(idx))
    H_after = compute_regime_heterogeneity(
        act_all[None], valid_het, reg2_trva[:1])[0]
    assert np.array_equal(H_before, H_after), "AUDIT 7 FAILED"
    audits["audit7_valid_region"] = {"flipped_out_of_mask": int(n_flip),
                                     "h_changed": False, "pass": True}
    log(f"  [AUDIT7] per-feature valid region: {n_flip} out-of-mask flips "
        f"-> H unchanged exactly")

    # ---------------- variants R2/C1/C2 --------------------------------------
    for nm, blocks in [("R2", {"global": (G_tr, G_va, G_te),
                               "het": (H2_tr, H2_va, H2_te)}),
                       ("C1", {"global": (G_tr, G_va, G_te),
                               "het": (C1_tr, C1_va, C1_te)}),
                       ("C2", {"global": (G_tr, G_va, G_te),
                               "het": (C2_tr, C2_va, C2_te)})]:
        log(f"  [{nm}]")
        results[nm], preds = run_variant(nm, blocks, yva, yte, ytrva)
        if nm == "R2":
            pred_r2 = preds
        elif nm == "C1":
            pred_c1 = preds
        else:
            pred_c2 = preds

    # ---------------- mechanistic diagnostics -------------------------------
    mech = {"R0": h_stats(H0_trva), "R2": h_stats(H2_tr),
            "C1": h_stats(C1_tr), "C2": h_stats(C2_tr)}
    for v in ["R0", "R2", "C1", "C2"]:
        results[v]["mechanistic_h"] = mech[v]
    results["R0"]["regime_diagnostics"] = {"train": core.occupancy(reg0_trva)}
    results["R2"]["regime_diagnostics"] = occ2
    results["R2"]["context_train"] = train_info
    results["R2"]["params"] = params

    deltas = {f"R2-{k}": round(results["R2"]["test_macro_f1"] -
                               results[k]["test_macro_f1"], 4)
              for k in ["R0", "C1", "C2"]}
    deltas["C1-C2"] = round(results["C1"]["test_macro_f1"] -
                            results["C2"]["test_macro_f1"], 4)

    # ---------------- artifacts ---------------------------------------------
    os.makedirs(os.path.join(ds_dir, "predictions"), exist_ok=True)
    with open(os.path.join(ds_dir, "predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_index", "true_class", "R0", "R2", "C1", "C2"])
        for i in range(len(yte)):
            w.writerow([i, int(yte[i]), int(pred0[i]), int(pred_r2[i]),
                        int(pred_c1[i]), int(pred_c2[i])])
    ds_result = {
        "dataset": ds_name, "seed": SEED, "T": T,
        "split": {"train": len(Xtr), "val": len(Xva), "test": len(Xte)},
        "n_classes": data["n_classes"],
        "class_labels": sorted(set(int(v) for v in np.unique(ytrva))),
        "n_features": N_FEATURES, "K_regimes": K_CODES,
        "config": {"encoder": ENCODER, "vq": VQ, "joint": JOINT},
        "drtn_checkpoint": drtn_info, "results": results, "deltas": deltas,
        "mechanistic_h": mech, "regime_array_audit": reg_audit,
        "audits": audits, "runtime_s": round(time.time() - t_start, 1),
    }
    with open(os.path.join(ds_dir, "result.json"), "w") as f:
        json.dump(ds_result, f, indent=2)
    with open(os.path.join(ds_dir, "diagnostics", "audits.json"), "w") as f:
        json.dump(audits, f, indent=2)
    del H0_trva, H0_va, H0_te, H2_tr, H2_va, H2_te, C1_tr, C1_va, C1_te, \
        C2_tr, C2_va, C2_te
    log(f"  [{ds_name}] done in {ds_result['runtime_s']}s — "
        + " ".join(f"{v}={results[v]['test_macro_f1']:.4f}"
                   for v in ["R0", "R2", "C1", "C2"]))
    return ds_result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ECG5000_BAL", "CWRU_BAL"])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    log("=" * 74)
    log("SSL-CONTEXT + HARD-VQ HETEROGENEITY TRANSFER part 2 (seed 42) — "
        "ECG5000_BAL / CWRU_BAL")
    log("=" * 74)
    log(f"  device={device} smoke={args.smoke} K={K_CODES} "
        f"enc={ENCODER['channels']} d={ENCODER['d_model']} "
        f"mask={ENCODER['mask_ratio']} span={ENCODER['span_len']} "
        f"beta_vq={VQ['beta_vq']} lam_div={VQ['lam_div']} "
        f"lam_cls={JOINT['lambda_cls']}")

    all_results = {}
    for ds in args.datasets:
        all_results[ds] = run_dataset(ds, device, smoke=args.smoke)

    per_ds = {ds: {v: r["results"][v]["test_macro_f1"]
                   for v in ["R0", "R2", "C1", "C2"]}
              for ds, r in all_results.items()}
    verdicts = {}
    for ds, r in all_results.items():
        res = {v: r["results"][v]["test_macro_f1"]
               for v in ["R0", "R2", "C1", "C2"]}
        near_ceiling = M0_REFS[ds] >= 0.99
        if near_ceiling and max(abs(res["R2"] - res[v])
                                for v in ["R0", "C1", "C2"]) < 0.005:
            verdicts[ds] = "SATURATED / UNINFORMATIVE"
        elif res["R2"] > res["R0"]:
            if res["R2"] > res["C1"] and res["R2"] > res["C2"]:
                margin = min(res["R2"] - res["C1"], res["R2"] - res["C2"])
                verdicts[ds] = ("STRONG TRANSFER" if margin >= 0.01
                                else "PARTIAL")
            else:
                verdicts[ds] = "PARTIAL"
        else:
            verdicts[ds] = "NEGATIVE"

    report = {
        "title": "SSL-context + hard-VQ heterogeneity transfer part 2 (seed 42)",
        "seed": SEED, "datasets": args.datasets,
        "variants": ["R0", "R2", "C1", "C2"],
        "per_dataset_test_macro_f1": per_ds,
        "deltas": {ds: r["deltas"] for ds, r in all_results.items()},
        "mechanistic_h": {ds: r["mechanistic_h"] for ds, r in all_results.items()},
        "verdicts": verdicts,
        "canonical_M0_references": M0_REFS,
        "r0_gates": R0_GATES,
        "context_references": {
            "Haptics_seed42_RCMKN": {"M0": 0.4974, "R0": 0.5366, "R2": 0.5500,
                                     "C1": 0.4927, "C2": 0.5011},
            "ECG5000_BAL_3seed_audited": {"M0": 0.6553, "M1": 0.6748},
        },
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
            json.dump(all_results, f, indent=2)
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
