"""UWaveY — CAPACITY-CONTROLLED NESTED/HIERARCHICAL REGIME-CONDITIONED H.

Question: does the nested regime hierarchy provide a useful multi-resolution
H representation when total H capacity is FIXED to the flat bank's 4,998
features?  Depth L* comes from the label-free shuffled-regime null; the
budget allocation and carrier selection come from intrinsic structural
energy only.  NO validation/test/labels anywhere in representation design.

Pipeline (spec section 41, exactly):
  UWaveY raw training signals (896 official -> 761/135 internal, seed 42)
    -> per-sample znorm (R2 protocol)
    -> frozen R2 UWaveY context checkpoint -> SSLTemporalEncoder latents
       (T=315, d=32) for TRAIN samples only
    -> ONE nested tree (recursive 2-means, K=1->2->4->8->16, transform-only
       for val/test)
    -> canonical MiniRocket kernels (fit on train z-normed, seed 42) ->
       per-timestep activation indicators (shared transfer core)
    -> exact nested-ANOVA chain: level energies, identity, orthogonality
    -> shuffled-regime null (S=500, seed 52042)
    -> predeclared stopping rule -> L* (FROZEN)
    -> fixed budget 4,998 = largest-remainder energy allocation across
       retained levels -> top-energy label-free carrier selection
       (FROZEN before any classifier)
    -> final [G_full || H_hier_budget] (9,996 features) -> canonical
       RidgeClassifierCV(train+val), single official test evaluation

Identity gates (stored R2 references, checked in-run):
    M0 (MiniROCKET)  test Macro-F1 = 0.7539
    R2 (flat G+H)    test Macro-F1 = 0.7551
Flat H for gate R2 is RECOMPUTED in-run from the same frozen checkpoint via
the unchanged R2 machinery (flat K=8 VQ regimes + audited heterogeneity).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from models.hierarchical_budget.model import (  # noqa: E402
    BUDGET_H, dataset_energy_summary, edge_energies, level_budgets,
    select_carriers)
from models.nested_regimes.model import (  # noqa: E402
    LEVELS, NULL_SEED, SEED, S_PERM, assign_levels, build_latent_tree,
    hierarchy_chain, nesting_errors, save_json, shuffle_null, stopping_rule)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "uwavey_nested_hierarchical", "seed42")
CACHE = "C:/temp/results"
DS = "UWaveGestureLibraryY"
GATE_M0 = 0.7539
GATE_R2 = 0.7551
GATE_TOL = 0.002                      # M0_TOL-style reproduction guard
ALPHAS = np.logspace(-4, 4, 20)       # canonical Ridge grid
N_FEATURES = 9996                     # canonical MiniRocket width
N_GLOBAL = 4998                       # G bank / kernel half


def r2_ckpt_path():
    """Stored R2 UWaveY context checkpoint (repo first, cache fallback)."""
    p = os.path.join(ROOT, "results", "r2_uwave_seed42", DS, "checkpoints",
                     "context_model_seed42.pt")
    if os.path.exists(p):
        return p
    p2 = os.path.join(CACHE, "r2_uwave_seed42", DS, "checkpoints",
                      "context_model_seed42.pt")
    if os.path.exists(p2):
        return p2
    raise FileNotFoundError(
        "R2 UWaveY checkpoint not found in repo or cache")


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# data / frozen models
# ---------------------------------------------------------------------------
def load_data():
    """Canonical UWaveY split via the established R2 loader (indices are
    re-saved into THIS experiment's dir; the R2 dir is never written)."""
    from experiments.rcmkn_r2_uwave_seed42.data import load_and_split
    d = load_and_split(DS, OUT)
    Xtr = np.asarray(d["Xtr"], dtype=np.float64)
    Xva = np.asarray(d["Xva"], dtype=np.float64)
    Xte = np.asarray(d["Xte"], dtype=np.float64)
    ytr, yva, yte = (np.asarray(d["ytr"]), np.asarray(d["yva"]),
                     np.asarray(d["yte"]))
    assert Xtr.shape[0] == 761 and Xva.shape[0] == 135 \
        and Xte.shape[0] == 3582, (Xtr.shape, Xva.shape, Xte.shape)
    assert Xtr.shape[1] == 315, Xtr.shape
    return d, Xtr, Xva, Xte, ytr, yva, yte


def znorm(X):
    """Canonical per-sample z-normalization (project convention)."""
    mu = X.mean(axis=-1, keepdims=True)
    sig = X.std(axis=-1, keepdims=True) + 1e-8
    return ((X - mu) / sig).astype(np.float32)


def load_context_model(device):
    """Frozen R2 UWaveY context checkpoint -> model (eval, params frozen)."""
    import torch
    from experiments.rcmkn_haptics_seed42.model import RCMKNContextModel
    ck = torch.load(r2_ckpt_path(), map_location=device, weights_only=False)
    model = RCMKNContextModel(n_classes=8)
    model.load_state_dict(ck["model_state"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ck


def _encoder_latents(model, X_z, device, batch=64):
    """Frozen encoder -> latents (N, T, 32) float64 (no grad)."""
    import torch
    out = []
    with torch.no_grad():
        for c0 in range(0, len(X_z), batch):
            xb = torch.from_numpy(X_z[c0:c0 + batch])[:, None, :].to(device)
            out.append(model.encoder(xb).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float64)


def fit_minirocket(Xtr_z):
    """Canonical MiniROCKET fit (train z-normed only, seed 42)."""
    from aeon.transformations.collection.convolution_based import MiniRocket
    extractor = MiniRocket(random_state=SEED, n_jobs=-1)
    extractor.fit(Xtr_z[:, None, :].astype(np.float32))
    return extractor


def compute_activations(extractor, X_z):
    from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (
        compute_raw_activations)
    return compute_raw_activations(extractor, X_z)


def ridge_eval(Xtrva, ytrva, Xte, yte):
    """Canonical Ridge protocol: RidgeClassifierCV(ALPHAS) fit on train+val,
    one official test evaluation."""
    from sklearn.linear_model import RidgeClassifierCV
    from sklearn.metrics import accuracy_score, f1_score
    clf = RidgeClassifierCV(alphas=ALPHAS)
    clf.fit(Xtrva, ytrva)
    pred = clf.predict(Xte).astype(np.int64)
    return {
        "macro_f1": round(float(f1_score(yte, pred, average="macro",
                                         zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(yte, pred)), 4),
        "selected_alpha": float(clf.alpha_),
    }, pred


def ppv_all(extractor, X_z, chunk=64):
    """Canonical MiniRocket PPV features (FULL 9996 width), chunked.
    G bank = first 4998 columns (canonical split); the H-bank kernels are
    features 4998:9996."""
    from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (
        ppv_from_activations)
    out = []
    for c0 in range(0, len(X_z), chunk):
        act, valid = compute_activations(extractor, X_z[c0:c0 + chunk])
        out.append(ppv_from_activations(act, valid).astype(np.float64))
        del act
    return np.vstack(out)


def flat_h_banks(model, extractor, Xtrva_z, Xte_z, device, chunk=64):
    """Gate-R2 flat H via the UNCHANGED R2 machinery: frozen flat K=8 VQ
    regimes from the same checkpoint + audited regime heterogeneity."""
    from experiments.drtn_conditioned_minirocket_haptics_3seed.runner import (
        compute_regime_heterogeneity)
    from experiments.rcmkn_haptics_seed42.runner import (
        extract_context_regimes, set_seed)
    set_seed(SEED)
    reg_trva = extract_context_regimes(model, Xtrva_z, device, batch=32)
    reg_te = extract_context_regimes(model, Xte_z, device, batch=32)

    def het(X_z, regimes):
        N = len(X_z)
        H = np.empty((N, N_GLOBAL), dtype=np.float64)
        for c0 in range(0, N, chunk):
            c1 = min(c0 + chunk, N)
            act, valid = compute_activations(extractor, X_z[c0:c1])
            H[c0:c1] = compute_regime_heterogeneity(
                act[:, N_GLOBAL:], valid[N_GLOBAL:], regimes[c0:c1])
            del act
        return H

    return het(Xtrva_z, reg_trva), het(Xte_z, reg_te)


def delta_banks(act_H, valid_het, reg16, kept_K, delta_cols=None,
                sample_chunk=256, dtype=np.float64):
    """Chunked frozen-transform detail banks (memory-safe for the 3,582-row
    test split).  delta_cols=None -> FULL detail bank per kept level
    (column order: level coarse->fine, regimes 0..K-1 within level);
    delta_cols=sel -> exactly the selected (kernel, child) columns per
    level (the hierarchical_budget selection), energies still computed
    for all kernels.  Energies/diagnostics are NOT accumulated here."""
    banks = []
    for c0 in range(0, len(act_H), sample_chunk):
        c1 = min(c0 + sample_chunk, len(act_H))
        ch = hierarchy_chain(act_H[c0:c1], valid_het, reg16[c0:c1],
                             want_deltas=True, want_ips=False,
                             delta_cols=delta_cols)
        if delta_cols is None:
            bank = np.hstack([np.hstack([ch["deltas"][K][:, j, :]
                                         for j in range(K)])
                              for K in kept_K])
        else:
            bank = np.hstack([ch["deltas"][K] for K in kept_K])
        banks.append(bank.astype(dtype))
        del ch
    return np.vstack(banks)


def het_banks_signals(extractor, X_z, reg16, kept_K, delta_cols=None,
                      dtype=np.float64, sig_chunk=32):
    """Frozen-transform detail banks straight from signals, chunked so
    full-split activations never materialize (3,582 x 9,996 x T bool
    would need ~11 GB).  Same frozen tree / frozen selection everywhere."""
    banks = []
    for c0 in range(0, len(X_z), sig_chunk):
        c1 = min(c0 + sig_chunk, len(X_z))
        act, valid = compute_activations(extractor, X_z[c0:c1])
        banks.append(delta_banks(act[:, N_GLOBAL:], valid[N_GLOBAL:],
                                 reg16[c0:c1], kept_K,
                                 delta_cols=delta_cols, dtype=dtype))
        del act
    return np.vstack(banks)


def main(smoke=False):
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "predictions"), exist_ok=True)
    t00 = time.time()
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device: {device}")

    # ---- 1. canonical data --------------------------------------------------
    d, Xtr, Xva, Xte, ytr, yva, yte = load_data()
    n_tr = len(ytr)
    log(f"data: train {Xtr.shape}, val {Xva.shape}, test {Xte.shape}")

    # ---- 2. frozen context model + TRAIN latents ----------------------------
    model, _ = load_context_model(device)
    Xtr_z = znorm(Xtr)
    lat = _encoder_latents(model, Xtr_z, device)
    lat_pool = lat.reshape(-1, lat.shape[-1])
    log(f"train latents: {lat_pool.shape}")

    # ---- 3. ONE nested tree (train only; frozen transform for val/test) -----
    C, tree_meta = build_latent_tree(lat_pool, seed=SEED)
    log("tree levels: " + ", ".join(str(len(c)) for c in C))
    levels_tr = assign_levels(lat_pool, C)
    errs = nesting_errors(levels_tr)
    assert not errs, f"nesting violated: {errs}"
    reg16_tr = levels_tr[4].reshape(n_tr, -1).astype(np.int64)

    # ---- 4. canonical kernels + activations ---------------------------------
    extractor = fit_minirocket(Xtr_z)
    act_tr_full, valid_full = compute_activations(extractor, Xtr_z)
    assert act_tr_full.shape[1] == N_FEATURES, act_tr_full.shape
    valid_het = valid_full[N_GLOBAL:]
    act_tr = act_tr_full[:, N_GLOBAL:, :]
    log(f"activations: {act_tr.shape} (H-bank kernels, features "
        f"{N_GLOBAL}:{N_FEATURES})")
    G_trva = ppv_all(extractor, np.vstack([Xtr_z, znorm(Xva)]))
    G_te = ppv_all(extractor, znorm(Xte))
    assert G_trva.shape == (n_tr + len(yva), N_FEATURES), G_trva.shape
    F_trva, F_te = G_trva, G_te          # full 9996: the M0 representation
    G_trva, G_te = F_trva[:, :N_GLOBAL], F_te[:, :N_GLOBAL]

    # ---- 5. exact nested chain on TRAIN --------------------------------------
    ch = hierarchy_chain(act_tr, valid_het, reg16_tr,
                         want_deltas=True, want_ips=True)
    log(f"identity_rel={ch['identity_max_rel']:.2e} "
        f"zero_sum={ch['zero_sum_max']:.2e} "
        f"cross_ip_max={float(ch['cross_ip'].max()):.2e}")
    E_total = ch["E_total"]
    log("level energies (K=2,4,8,16): " + str(np.round(E_total, 3)))

    # ---- 6. label-free shuffled-regime null ----------------------------------
    log(f"null: S={S_PERM}, seed={NULL_SEED}")
    null = shuffle_null(act_tr, valid_het, reg16_tr, S=S_PERM,
                        seed=NULL_SEED, verbose=100)
    log("null means: " + str(np.round(null.mean(0), 3)))

    # ---- 7. stopping rule -> L* (FROZEN before any classifier) ---------------
    L_star, rows = stopping_rule(E_total, null)
    for r in rows:
        log(f"  K={r['K']}: E={r['real_energy']:.2f} null_mean="
            f"{r['null_mean']:.2f} p95={r['null_p95']:.2f} "
            f"p={r['p_raw_plus_one']:.3f} q={r['q_bh']:.3f} "
            f"keep={r['keep']}")
    log(f"L* = {L_star}")
    retained_K = [r["K"] for r in rows if r["keep"]]

    # ---- 8. fixed-budget allocation + label-free carrier selection -----------
    kept_K, B_l = level_budgets(E_total, retained_K, budget=BUDGET_H)
    log("level budgets: " + str(dict(zip(kept_K, B_l.tolist()))))
    ee = edge_energies(ch, n_tr)
    sel = select_carriers(ee, dict(zip(kept_K, B_l)), kept_K)
    log("selected carriers per level: "
        + str({k: len(v) for k, v in sel.items()}))

    # ---- 9. frozen-transform banks for val/test (chunked) --------------------
    Xva_z = znorm(Xva)
    lat_va = _encoder_latents(model, Xva_z, device)      # (N, T, 32)
    reg16_va = assign_levels(lat_va.reshape(-1, lat_va.shape[-1]),
                             C)[4].reshape(len(yva), -1).astype(np.int64)
    del lat_va
    act_va_full, _ = compute_activations(extractor, Xva_z)
    Xte_z = znorm(Xte)
    lat_te = _encoder_latents(model, Xte_z, device)
    reg16_te = assign_levels(lat_te.reshape(-1, lat_te.shape[-1]),
                             C)[4].reshape(len(yte), -1).astype(np.int64)
    del lat_te

    # PRIMARY budgeted banks: only the selected columns are materialized
    Hh_tr = delta_banks(act_tr, valid_het, reg16_tr, kept_K, delta_cols=sel)
    Hh_va = delta_banks(act_va_full[:, N_GLOBAL:], valid_het, reg16_va,
                        kept_K, delta_cols=sel)
    Hh_te = het_banks_signals(extractor, Xte_z, reg16_te, kept_K,
                              delta_cols=sel)
    Hh_trva = np.vstack([Hh_tr, Hh_va])
    assert Hh_trva.shape == (n_tr + len(yva), BUDGET_H), Hh_trva.shape
    assert Hh_te.shape == (len(yte), BUDGET_H), Hh_te.shape
    log(f"H_hier_budget: {Hh_trva.shape[1]} features, "
        f"trva {Hh_trva.shape}, te {Hh_te.shape}")

    # ---- 10. identity gates (stored R2 references) ---------------------------
    y_dev = np.concatenate([ytr, yva])
    m0_res, m0_pred = ridge_eval(F_trva, y_dev, F_te, yte)
    m0_diff = abs(m0_res["macro_f1"] - GATE_M0)
    log(f"[gate M0] MiniROCKET = {m0_res['macro_f1']} "
        f"(stored {GATE_M0}, diff {m0_diff:.4f})")
    H_trva_c, H_te_c = flat_h_banks(model, extractor,
                                    np.vstack([Xtr_z, Xva_z]), Xte_z, device)
    r2_res, r2_pred = ridge_eval(np.hstack([G_trva, H_trva_c]), y_dev,
                                 np.hstack([G_te, H_te_c]), yte)
    r2_diff = abs(r2_res["macro_f1"] - GATE_R2)
    log(f"[gate R2] flat G+H = {r2_res['macro_f1']} "
        f"(stored {GATE_R2}, diff {r2_diff:.4f})")

    # ---- 11. optional full-detail diagnostic (chunked, float32) --------------
    Bf = int(sum(K * N_GLOBAL for K in kept_K))
    full_res, full_pred = None, None
    if not smoke:
        log(f"[hier full diagnostic] features = {Bf}")
        Hf_tr = delta_banks(act_tr, valid_het, reg16_tr, kept_K,
                            dtype=np.float32)
        Hf_va = delta_banks(act_va_full[:, N_GLOBAL:], valid_het, reg16_va,
                            kept_K, dtype=np.float32)
        Hf_te = het_banks_signals(extractor, Xte_z, reg16_te, kept_K,
                                  dtype=np.float32)
        Hf_trva = np.vstack([Hf_tr, Hf_va])
        full_res, full_pred = ridge_eval(
            np.hstack([G_trva, Hf_trva.astype(np.float64)]), y_dev,
            np.hstack([G_te, Hf_te.astype(np.float64)]), yte)
        log(f"[hier full] = {full_res['macro_f1']}")
        del Hf_tr, Hf_va, Hf_te, Hf_trva
    del act_va_full

    # ---- 12. PRIMARY: [G_full || H_hier_budget] ------------------------------
    hier_res, hier_pred = ridge_eval(np.hstack([G_trva, Hh_trva]), y_dev,
                                     np.hstack([G_te, Hh_te]), yte)
    log(f"[PRIMARY hier-budget] = {hier_res['macro_f1']}")

    # ---- artifacts -----------------------------------------------------------
    save_json({
        "dataset": DS, "seed": SEED,
        "n_train": n_tr, "n_val": int(len(yva)), "n_test": int(len(yte)),
        "T": int(Xtr.shape[1]), "n_classes": int(len(np.unique(ytr))),
        "n_features_G": int(G_trva.shape[1]),
        "tree_meta": tree_meta,
        "identity_max_abs": ch["identity_max_abs"],
        "identity_max_rel": ch["identity_max_rel"],
        "zero_sum_max": ch["zero_sum_max"],
        "cross_ip_max": float(np.abs(ch["cross_ip"]).max()),
        "level_energies": {"K": [int(k) for k in LEVELS[1:]],
                           "real": E_total.tolist()},
        "stopping": {"L_star": int(L_star), "rows": rows,
                     "S": S_PERM, "null_seed": NULL_SEED},
        "budget": {"B_H": BUDGET_H,
                   "allocation": {int(k): int(b) for k, b in
                                  zip(kept_K, B_l)}},
        "selected_counts": {int(k): int(len(v)) for k, v in sel.items()},
        "gates": {"minirocket": m0_res["macro_f1"],
                  "minirocket_ref": GATE_M0, "minirocket_diff": m0_diff,
                  "flat_gh": r2_res["macro_f1"], "flat_gh_ref": GATE_R2,
                  "flat_gh_diff": r2_diff},
        "results": {"minirocket": m0_res, "flat_gh": r2_res,
                    "hierarchical_budget": hier_res,
                    "hierarchical_full": full_res},
        "runtime_s": time.time() - t00,
    }, os.path.join(OUT, "results.json"))
    save_json({"seed": SEED, "S_perm": S_PERM, "null_seed": NULL_SEED,
               "min_occupancy": 0.01, "L_star": int(L_star),
               "decisions": rows}, os.path.join(OUT, "stopping_decision.json"))
    save_json({"tree_meta": tree_meta,
               "level_sizes": [len(c) for c in C],
               "occupied_regimes_level16_train":
                   int((ch["n_active_regimes"].sum(axis=0) > 0).sum())},
              os.path.join(OUT, "hierarchy.json"))
    np.save(os.path.join(OUT, "hierarchy_assignments.npy"), reg16_tr)
    save_json({"identity_max_abs": ch["identity_max_abs"],
               "identity_max_rel": ch["identity_max_rel"],
               "tolerance": "float64; target <= 1e-10 relative",
               "pass": bool(ch["identity_max_rel"] < 1e-10)},
              os.path.join(OUT, "hierarchy_identity.json"))
    save_json({"cross_ip_max": float(np.abs(ch["cross_ip"]).max()),
               "pairs": ["|K%d-K%d|" % p for p in
                         ((2, 4), (2, 8), (2, 16), (4, 8), (4, 16), (8, 16))],
               "method": "exact joint count-weighted inner products; "
                         "no Gram-Schmidt, no CCA",
               "tolerance": "float64 machine precision"},
              os.path.join(OUT, "orthogonality_report.json"))
    summary = dataset_energy_summary(ch, kept_K)
    save_json({"B_H": BUDGET_H, "retained_K": kept_K,
               "level_budgets": {int(k): int(b) for k, b in
                                 zip(kept_K, B_l)},
               "HI_hier_median_sample_total_energy":
                   summary["HI_hier_median_sample_total_energy"],
               "HI_hier_mean_sample_total_energy":
                   summary["HI_hier_mean_sample_total_energy"],
               "energy_fractions": summary["energy_fractions"],
               "rule": "w_l = E_l / sum retained E_l; largest-remainder "
                       "floor allocation, tie-break lower K first",
               "selection": "top e_{m,c,l} = mean_i pi_c Delta^2 per level; "
                            "ties by (kernel, child id); label-free"},
              os.path.join(OUT, "budget_allocation.json"))
    with open(os.path.join(OUT, "selected_features.csv"), "w") as f:
        f.write("level_K,feature_index,kernel,child,structural_energy\n")
        fi0 = 0
        for K in kept_K:
            arr = sel[K]
            e = ee[K]
            for fi, (m, c) in enumerate(arr):
                f.write(f"{K},{fi0 + fi},{m},{c},{e[c, m]:.10f}\n")
            fi0 += len(arr)
    save_json({"poison_checks":
                   "see tests/test_uwavey_nested_hierarchical.py",
               "labels_used_in_intrinsic_stage": False,
               "validation_used_in_intrinsic_stage": False,
               "test_used_in_intrinsic_stage": False,
               "L_star_frozen_before_classifier": True},
              os.path.join(OUT, "leakage_audit.json"))

    with open(os.path.join(OUT, "level_energy.csv"), "w") as f:
        f.write("K,real_energy,null_mean,null_std,null_p95,p_raw,q_bh,keep\n")
        for r in rows:
            f.write(f"{r['K']},{r['real_energy']:.6f},{r['null_mean']:.6f},"
                    f"{r['null_std']:.6f},{r['null_p95']:.6f},"
                    f"{r['p_raw_plus_one']:.6f},{r['q_bh']:.6f},{r['keep']}\n")
    with open(os.path.join(OUT, "sample_energy.csv"), "w") as f:
        f.write("sample," + ",".join(f"E_K{k}" for k in LEVELS[1:])
                + "," + ",".join(f"frac_K{k}" for k in LEVELS[1:])
                + ",n_occupied,occupancy_entropy,n_transitions,"
                  "mean_run_length\n")
        Es = ch["E_sample"]
        tot = np.maximum(Es.sum(axis=1, keepdims=True), 1e-30)
        for i in range(n_tr):
            cnts = np.bincount(reg16_tr[i], minlength=16).astype(float)
            p = cnts[cnts > 0] / cnts.sum()
            ent = float(-(p * np.log(p)).sum())
            tr = int((np.diff(reg16_tr[i]) != 0).sum())
            runs = np.diff(np.flatnonzero(np.concatenate(
                [[True], np.diff(reg16_tr[i]) != 0, [True]])))
            mrl = float(runs.mean()) if len(runs) else float(len(reg16_tr[i]))
            f.write(f"{i}," + ",".join(f"{v:.8f}" for v in Es[i]) + ","
                    + ",".join(f"{v:.8f}" for v in Es[i] / tot[i])
                    + f",{int((cnts > 0).sum())},{ent:.6f},{tr},{mrl:.4f}\n")
    with open(os.path.join(OUT, "null_energy.csv"), "w") as f:
        f.write("perm," + ",".join(f"null_K{k}" for k in LEVELS[1:]) + "\n")
        for s in range(null.shape[0]):
            f.write(f"{s}," + ",".join(f"{v:.8f}" for v in null[s]) + "\n")
    with open(os.path.join(OUT, "final_comparison.csv"), "w") as f:
        f.write("Method,Representation,H_dims,Test_MacroF1,Notes\n")
        f.write(f"MiniROCKET,full MiniROCKET,{N_FEATURES},"
                f"{m0_res['macro_f1']:.4f},"
                f"identity gate (stored {GATE_M0})\n")
        f.write(f"Flat_HERAMBA,[G||H_flat],{N_GLOBAL},"
                f"{r2_res['macro_f1']:.4f},"
                f"identity gate (stored {GATE_R2})\n")
        if full_res is not None:
            f.write(f"Hierarchical_full,[G||H_hier_full],{Bf},"
                    f"{full_res['macro_f1']:.4f},optional diagnostic\n")
        f.write(f"Hierarchical_budget,[G||H_hier_budget],{BUDGET_H},"
                f"{hier_res['macro_f1']:.4f},PRIMARY capacity-controlled\n")
    preds = [("minirocket", m0_pred), ("flat_gh", r2_pred),
             ("hierarchical_budget", hier_pred)]
    if full_pred is not None:
        preds.append(("hierarchical_full", full_pred))
    for nm, pr in preds:
        np.save(os.path.join(OUT, "predictions", f"{nm}.npy"), pr)

    _figures(Xtr, reg16_tr, rows, null, E_total, kept_K, B_l, sel, ee, L_star)
    log(f"done in {time.time() - t00:.0f}s")


def _figures(Xtr, reg16_tr, rows, null, E_total, kept_K, B_l, sel, ee,
             L_star):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from models.nested_regimes.model import LEVELS

    # hierarchy tree
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    depth_y = {0: 3.0, 1: 2.0, 2: 1.0, 3: 0.0}
    for dep in range(4):
        n_nodes = 2 ** dep
        for p in range(n_nodes):
            x = (p + 0.5) / n_nodes
            ax.scatter([x], [depth_y[dep]], s=120, c="C0", zorder=2)
            ax.annotate(str(p), (x, depth_y[dep]), fontsize=6, ha="center",
                        va="center", color="white", zorder=3)
            if dep > 0:
                ax.plot([((p >> 1) + 0.5) / (2 ** (dep - 1)), x],
                        [depth_y[dep - 1], depth_y[dep]], "k-", lw=0.7,
                        zorder=1)
    for dep in range(4):
        ax.text(-0.02, depth_y[dep], f"K={2 ** dep}", fontsize=9, ha="right",
                va="center")
    ax.set_title("Nested regime tree (binary path ids)")
    ax.axis("off")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "figures", f"hierarchy_tree.{ext}"))
    plt.close(fig)

    # level energy vs null
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    Ks = [r["K"] for r in rows]
    mean = [r["null_mean"] for r in rows]
    p95 = [r["null_p95"] for r in rows]
    real = [r["real_energy"] for r in rows]
    ax.errorbar(Ks, mean, yerr=[np.array(p95) - np.array(mean),
                                np.zeros_like(p95)], fmt="o--", color="gray",
                label="null mean / 95th pct", capsize=3)
    ax.plot(Ks, real, "o-", color="C0", label="real energy")
    ax.axvline(L_star, color="C3", ls=":", label=f"L* = {L_star}")
    ax.set(xlabel="K (level)", ylabel="total detail energy",
           title="Level energy vs shuffled-regime null (UWaveY)")
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "figures", f"level_energy.{ext}"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for i, K in enumerate(LEVELS[1:]):
        ax.hist(null[:, i], bins=40, alpha=0.4, label=f"null K={K}")
        ax.axvline(E_total[i], color=f"C{i}", lw=2)
    ax.set(xlabel="total energy", ylabel="permutations",
           title="Real (lines) vs null (histograms)")
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "figures",
                                 f"null_vs_real_energy.{ext}"))
    plt.close(fig)

    # budget allocation vs energy fraction
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    idx = [LEVELS[1:].index(k) for k in kept_K]
    frac = E_total[idx] / max(E_total[idx].sum(), 1e-30)
    x = np.arange(len(kept_K))
    ax.bar(x - 0.2, [b / BUDGET_H for b in B_l], 0.4, label="budget share",
           color="C0")
    ax.bar(x + 0.2, frac, 0.4, label="energy fraction", color="C1")
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in kept_K])
    ax.set(ylabel="share", title="Budget allocation vs structural energy")
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "figures", f"budget_allocation.{ext}"))
    plt.close(fig)

    # regime assignment example (train sample 0, full T)
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    T = Xtr.shape[1]
    ax.plot(Xtr[0], color="k", lw=0.6, zorder=0)
    for l, K in enumerate(LEVELS[1:], start=1):
        rl = reg16_tr[0] >> (4 - l)
        ax.scatter(np.arange(T), np.full(T, -1.0 + 0.55 * l), c=rl,
                   cmap="tab20" if K == 16 else "tab10", s=3, vmin=0,
                   vmax=15)
        ax.text(T + 4, -1.0 + 0.55 * l, f"K={K}", fontsize=8, va="center")
    ax.set(ylabel="normalized signal / level rows", xlabel="timestep",
           title="Nested regime assignments (UWaveY train sample 0)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "figures",
                                 f"regime_assignment_example.{ext}"))
    plt.close(fig)

    # selected vs unselected carrier energy
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    sel_e, unsel_e = [], []
    for K in kept_K:
        Fk = ee[K].shape[1]
        e = ee[K].ravel()
        mask = np.zeros(len(e), dtype=bool)
        arr = sel[K]
        if K == 2:
            mask[arr[:, 0]] = True              # aliases deduped
        else:
            mask[arr[:, 1] * Fk + arr[:, 0]] = True
        sel_e.append(e[mask])
        unsel_e.append(e[~mask])
    ax.hist(np.concatenate(unsel_e), bins=60, alpha=0.5,
            label="unselected candidates")
    ax.hist(np.concatenate(sel_e), bins=60, alpha=0.5,
            label="selected carriers")
    ax.set(xlabel="structural carrier energy e_{m,c,l}", ylabel="count",
           yscale="log", title="Selected vs unselected structural carriers")
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "figures",
                                 f"selected_feature_energy.{ext}"))
    plt.close(fig)


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
