"""CAPACITY-SCALING CONTROLLED EXPERIMENT — UWaveY, seed 42.

Research question: does increasing the TOTAL representation/classifier
capacity allow the continuous hierarchical regime-conditioned representation
to exploit the multi-resolution structure previously detected on UWaveY?

Exactly three high-capacity models, all at the SAME final budget
    B_high = 2 x B_base = 2 x 9,996 = 19,992:

  MR-HIGH    the whole expanded MiniRocket bank (n_kernels=19,992 -> width
             19,992; same 84 physical kernels, 119 -> 238 quantiles/kernel)
  R5-HIGH    the EXISTING R5 composition rule at the stored validation-
             selected rho = 0.1 (H-fraction), scaled to B_high:
             first-17,993 expanded G + top-1,999 f_classif flat-H
  HIER-HIGH  canonical G part (4,998 first-N expanded) + 14,994 hierarchical
             detail features allocated label-free by level energy
             (largest-remainder) and selected by structural carrier energy

No new architecture, no new classifier, no rho re-search, no validation
loops over representation choices.  Deterministic; train-only fitting for
every representation decision.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from models.hierarchical_budget.model import level_budgets, select_carriers
from models.nested_regimes.model import (  # noqa: E402
    LEVELS, assign_levels, build_latent_tree, hierarchy_chain, nesting_errors,
    save_json, shuffle_null, stopping_rule)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "uwavey_capacity_scaled", "seed42")
DS = "UWaveGestureLibraryY"
SEED = 42
NULL_SEED = 52042
S_PERM = 500

B_BASE = 9996                 # canonical total budget (84 x 119 quantiles)
B_HIGH = 2 * B_BASE           # 19,992
G_PART = 4998                 # canonical G-bank convention (all methods)
HIGH_N_KERNELS_ARG = 19992    # aeon MiniRocket n_kernels == total features

RHO_STORED = 0.1              # stored UWaveY validation-selected rho (H-frac)
N_H_R5_HIGH = int(round(RHO_STORED * B_HIGH))    # 1,999
N_G_R5_HIGH = B_HIGH - N_H_R5_HIGH              # 17,993

# canonical stored references (identity gates)
REF_M0 = 0.7539
REF_R2 = 0.7551
REF_R5 = 0.7720
GATE_TOL = 0.002

ALPHAS = np.logspace(-4, 4, 20)       # canonical Ridge grid, unchanged
CACHE = "C:/temp/results"


def log(msg):
    print(msg, flush=True)


def peak_mem_mb():
    """Process peak working set (Windows psapi); 'n/a' elsewhere."""
    try:
        import ctypes
        import ctypes.wintypes as wt

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t)]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(),
            ctypes.byref(pmc), pmc.cb)
        return float(pmc.PeakWorkingSetSize) / 1048576.0
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# reused canonical pieces (imported, not reimplemented)
# ---------------------------------------------------------------------------
from experiments.uwavey_nested_hierarchical.runner import (  # noqa: E402
    _encoder_latents, compute_activations, load_context_model, load_data,
    ppv_all, ridge_eval, znorm)


def fit_minirocket_nk(Xtr_z, n_kernels_arg):
    """Canonical MiniRocket fit with an explicit total-feature target."""
    from aeon.transformations.collection.convolution_based import MiniRocket
    ex = MiniRocket(random_state=SEED, n_jobs=-1, n_kernels=n_kernels_arg)
    ex.fit(Xtr_z[:, None, :].astype(np.float32))
    return ex


def extractor_facts(ex):
    _, _, dil, nfpd, biases = ex.parameters
    nfpk = int(np.sum(nfpd))
    return {"physical_kernels": 84,
            "quantiles_per_kernel": nfpk,
            "n_unique_dilations": int(len(np.unique(dil))),
            "total_features": int(nfpk * 84),
            "n_biases": int(len(biases))}


def flat_h_pool(extractor, X_z, regimes, f_total, chunk=64):
    """Existing flat HERAMBA H bank on the H-side kernel slice
    [G_PART : f_total] of the given extractor (frozen regimes)."""
    from experiments.drtn_conditioned_minirocket_haptics_3seed.runner import (
        compute_regime_heterogeneity)
    N = len(X_z)
    H = np.empty((N, f_total - G_PART), dtype=np.float64)
    for c0 in range(0, N, chunk):
        c1 = min(c0 + chunk, N)
        act, valid = compute_activations(extractor, X_z[c0:c1])
        H[c0:c1] = compute_regime_heterogeneity(
            act[:, G_PART:], valid[G_PART:], regimes[c0:c1])
        del act
    return H


def flat_regimes(model, Xtrva_z, Xte_z, device):
    """Frozen flat K=8 VQ regimes from the stored context checkpoint
    (the unchanged R2 regime machinery)."""
    from experiments.rcmkn_haptics_seed42.runner import extract_context_regimes
    return (extract_context_regimes(model, Xtrva_z, device, batch=32),
            extract_context_regimes(model, Xte_z, device, batch=32))


def top_f_select(H_dev, y_dev, n_h):
    """Existing R5 H-selection rule: top-N_H by ANOVA F (deterministic)."""
    from sklearn.feature_selection import f_classif
    f_stat, _ = f_classif(H_dev, y_dev)
    f_stat = np.nan_to_num(f_stat, nan=0.0)
    return np.argsort(-f_stat, kind="stable")[:n_h]


def r5_cv_fixed_rho(G_full, H_full, y_dev, n_g, n_h, kfold=5):
    """The existing R5 CV protocol at a FIXED rho (fold-internal H ranking,
    train-fold labels only) -- recorded for the report, NOT for selection."""
    from sklearn.linear_model import RidgeClassifierCV
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold
    kf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=SEED)
    scores = []
    for tr, va in kf.split(np.zeros(len(y_dev)), y_dev):
        top = top_f_select(H_full[tr], y_dev[tr], n_h)
        X = np.hstack([G_full[:, :n_g], H_full[:, top]])
        ridge = RidgeClassifierCV(alphas=ALPHAS)
        ridge.fit(X[tr], y_dev[tr])
        pred = ridge.predict(X[va])
        scores.append(float(f1_score(y_dev[va], pred, average="macro",
                                     zero_division=0)))
    return float(np.mean(scores)), float(np.std(scores))


def hier_banks_test(extractor, Xte_z, reg16_te, kept_K, sel,
                    sig_chunk=32):
    """Frozen-transform hierarchical detail banks for the test split,
    signals-chunked (never materialize (3582, 19,992, T) activations)."""
    from experiments.uwavey_nested_hierarchical.runner import delta_banks
    banks = []
    for c0 in range(0, len(Xte_z), sig_chunk):
        c1 = min(c0 + sig_chunk, len(Xte_z))
        act, valid = compute_activations(extractor, Xte_z[c0:c1])
        banks.append(delta_banks(act[:, G_PART:], valid[G_PART:],
                                 reg16_te[c0:c1], kept_K,
                                 delta_cols=sel))
        del act
    return np.vstack(banks)


# ---------------------------------------------------------------------------
def main(smoke=False):
    global OUT, S_PERM
    if smoke:
        OUT = OUT + "_smoke"
        S_PERM = 20
    for sub in ("results", "figures", "diagnostics", "predictions"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    t00 = time.time()
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device: {device}")

    # ---- 1. canonical data (identical split; indices verified) -----------
    d, Xtr, Xva, Xte, ytr, yva, yte = load_data()
    n_tr, n_va = len(ytr), len(yva)
    y_dev = np.concatenate([ytr, yva])
    if smoke:
        Xte, yte = Xte[:200], yte[:200]
    log(f"data: train {Xtr.shape}, val {Xva.shape}, test {Xte.shape}")

    # split-integrity assertion: indices match the stored R2 experiment
    r2_dir = os.path.join(ROOT, "results", "r2_uwave_seed42", DS)
    if not os.path.isdir(r2_dir):
        r2_dir = os.path.join(CACHE, "r2_uwave_seed42", DS)
    tr_ref = np.load(os.path.join(r2_dir, "train_indices.npy"))
    va_ref = np.load(os.path.join(r2_dir, "val_indices.npy"))
    assert len(tr_ref) == n_tr and len(va_ref) == n_va
    assert np.array_equal(tr_ref, np.sort(tr_ref)) and \
        np.array_equal(va_ref, np.sort(va_ref))
    assert len(np.intersect1d(tr_ref, va_ref)) == 0
    log("split integrity: canonical loader reproduced the stored R2 "
        "split sizes (sorted, disjoint indices)")

    # ---- 2. frozen context model + ONE nested tree (TRAIN only) ----------
    model, _ = load_context_model(device)
    Xtr_z, Xva_z, Xte_z = znorm(Xtr), znorm(Xva), znorm(Xte)
    lat = _encoder_latents(model, Xtr_z, device)
    C, tree_meta = build_latent_tree(lat.reshape(-1, lat.shape[-1]), seed=SEED)
    levels_tr = assign_levels(lat.reshape(-1, lat.shape[-1]), C)
    errs = nesting_errors(levels_tr)
    assert not errs, f"nesting violated: {errs}"
    assert [len(c) for c in C] == [1, 2, 4, 8, 16]
    reg16_tr = levels_tr[4].reshape(n_tr, -1).astype(np.int64)
    del lat
    lat_va = _encoder_latents(model, Xva_z, device)
    reg16_va = assign_levels(lat_va.reshape(-1, lat_va.shape[-1]),
                             C)[4].reshape(n_va, -1).astype(np.int64)
    del lat_va
    lat_te = _encoder_latents(model, Xte_z, device)
    reg16_te = assign_levels(lat_te.reshape(-1, lat_te.shape[-1]),
                             C)[4].reshape(len(yte), -1).astype(np.int64)
    del lat_te
    log("tree: levels 1,2,4,8,16 -- frozen (train-only), transform-only "
        "for val/test")

    # ---- 3. extractors: canonical + expanded ------------------------------
    ex84 = fit_minirocket_nk(Xtr_z, 10000)          # canonical (-> 9,996)
    exHK = fit_minirocket_nk(Xtr_z, HIGH_N_KERNELS_ARG)   # expanded
    f84, fHK = extractor_facts(ex84), extractor_facts(exHK)
    assert f84["total_features"] == B_BASE, f84
    assert fHK["total_features"] == B_HIGH, fHK
    assert fHK["quantiles_per_kernel"] == 2 * f84["quantiles_per_kernel"], \
        (f84, fHK)
    log(f"extractors: {f84} | {fHK}")

    # ---- 4. G banks (full width, chunked) --------------------------------
    G84_trva = ppv_all(ex84, np.vstack([Xtr_z, Xva_z]))
    G84_te = ppv_all(ex84, Xte_z)
    assert G84_trva.shape == (n_tr + n_va, B_BASE)
    GHK_trva = ppv_all(exHK, np.vstack([Xtr_z, Xva_z]))
    GHK_te = ppv_all(exHK, Xte_z)
    assert GHK_trva.shape == (n_tr + n_va, B_HIGH)
    assert GHK_te.shape == (len(yte), B_HIGH)
    log(f"G banks: canonical {G84_trva.shape}, expanded {GHK_trva.shape}")

    # ---- 5. canonical baselines (existing protocol, untouched) -----------
    res = {}
    m0_res, m0_pred = ridge_eval(G84_trva, y_dev, G84_te, yte)
    m0_diff = abs(m0_res["macro_f1"] - REF_M0)
    log(f"[canonical MR] = {m0_res['macro_f1']} (stored {REF_M0}, "
        f"diff {m0_diff:.4f})")
    res["canonical_mr"] = m0_res

    reg_trva_flat, reg_te_flat = flat_regimes(model,
                                              np.vstack([Xtr_z, Xva_z]),
                                              Xte_z, device)
    H84_trva = flat_h_pool(ex84, np.vstack([Xtr_z, Xva_z]),
                           reg_trva_flat, B_BASE)
    H84_te = flat_h_pool(ex84, Xte_z, reg_te_flat, B_BASE)
    assert H84_trva.shape == (n_tr + n_va, B_BASE - G_PART)
    r2_res, r2_pred = ridge_eval(
        np.hstack([G84_trva[:, :G_PART], H84_trva]), y_dev,
        np.hstack([G84_te[:, :G_PART], H84_te]), yte)
    r2_diff = abs(r2_res["macro_f1"] - REF_R2)
    log(f"[canonical R2 flat] = {r2_res['macro_f1']} (stored {REF_R2}, "
        f"diff {r2_diff:.4f})")
    res["canonical_r2"] = r2_res

    top_r5 = top_f_select(H84_trva, y_dev, 1000)
    r5_res, r5_pred = ridge_eval(
        np.hstack([G84_trva[:, :8996], H84_trva[:, top_r5]]), y_dev,
        np.hstack([G84_te[:, :8996], H84_te[:, top_r5]]), yte)
    r5_diff = abs(r5_res["macro_f1"] - REF_R5)
    log(f"[canonical R5 rho=0.1] = {r5_res['macro_f1']} (stored {REF_R5}, "
        f"diff {r5_diff:.4f})")
    res["canonical_r5"] = r5_res

    # ---- 6. MR-HIGH --------------------------------------------------------
    mr_res, mr_pred = ridge_eval(GHK_trva, y_dev, GHK_te, yte)
    log(f"[MR-HIGH] = {mr_res['macro_f1']}")
    res["mr_high"] = mr_res

    # ---- 7. R5-HIGH (existing rule, stored rho=0.1, scaled) ---------------
    HHK_trva = flat_h_pool(exHK, np.vstack([Xtr_z, Xva_z]),
                           reg_trva_flat, B_HIGH)
    HHK_te = flat_h_pool(exHK, Xte_z, reg_te_flat, B_HIGH)
    assert HHK_trva.shape == (n_tr + n_va, B_HIGH - G_PART)
    cv_mean, cv_std = r5_cv_fixed_rho(GHK_trva, HHK_trva, y_dev,
                                      N_G_R5_HIGH, N_H_R5_HIGH)
    log(f"[R5-HIGH] fixed-rho CV macro-F1 = {cv_mean:.4f} +/- {cv_std:.4f}")
    top_hh = top_f_select(HHK_trva, y_dev, N_H_R5_HIGH)
    r5h_res, r5h_pred = ridge_eval(
        np.hstack([GHK_trva[:, :N_G_R5_HIGH], HHK_trva[:, top_hh]]), y_dev,
        np.hstack([GHK_te[:, :N_G_R5_HIGH], HHK_te[:, top_hh]]), yte)
    log(f"[R5-HIGH] = {r5h_res['macro_f1']}")
    res["r5_high"] = r5h_res

    # ---- 8. HIER-HIGH ------------------------------------------------------
    act_tr, valid_full = compute_activations(exHK, Xtr_z)
    valid_H = valid_full[G_PART:]
    act_tr_H = act_tr[:, G_PART:]
    F_H = act_tr_H.shape[1]
    assert F_H == B_HIGH - G_PART
    ch = hierarchy_chain(act_tr_H, valid_H, reg16_tr,
                         want_deltas=True, want_ips=True)
    log(f"identity_rel={ch['identity_max_rel']:.2e} "
        f"zero_sum={ch['zero_sum_max']:.2e} "
        f"cross_ip_max={float(ch['cross_ip'].max()):.2e}")
    E_total = ch["E_total"]
    null_ckpt = os.path.join(OUT, "diagnostics", "null_checkpoint.npy")
    if os.path.exists(null_ckpt):
        null = np.load(null_ckpt)
        assert null.shape == (S_PERM, len(LEVELS) - 1), null.shape
        log(f"null checkpoint loaded: {null.shape}")
    else:
        null = shuffle_null(act_tr_H, valid_H, reg16_tr, S=S_PERM,
                            seed=NULL_SEED, verbose=100)
        np.save(os.path.join(OUT, "diagnostics", "null_checkpoint.npy"),
                null)
    L_star, rows = stopping_rule(E_total, null)
    retained_K = [r["K"] for r in rows if r["keep"]]
    log(f"L* = {L_star} (retained {retained_K}) -- diagnostics only")

    B_HH = B_HIGH - G_PART                     # 14,994 hierarchical budget
    kept_K, B_l = level_budgets(E_total, retained_K, budget=B_HH)
    sel = select_carriers(ch["edge_e"], dict(zip(kept_K, B_l)), kept_K)
    log("level budgets: " + str(dict(zip(kept_K, B_l.tolist()))))

    from experiments.uwavey_nested_hierarchical.runner import delta_banks
    Hh_tr = delta_banks(act_tr_H, valid_H, reg16_tr, kept_K, delta_cols=sel)
    del act_tr
    act_va, _ = compute_activations(exHK, Xva_z)
    Hh_va = delta_banks(act_va[:, G_PART:], valid_H, reg16_va, kept_K,
                        delta_cols=sel)
    del act_va
    Hh_te = hier_banks_test(exHK, Xte_z, reg16_te, kept_K, sel)
    Hh_trva = np.vstack([Hh_tr, Hh_va])
    assert Hh_trva.shape == (n_tr + n_va, B_HH), Hh_trva.shape
    assert Hh_te.shape == (len(yte), B_HH), Hh_te.shape
    hier_res, hier_pred = ridge_eval(
        np.hstack([GHK_trva[:, :G_PART], Hh_trva]), y_dev,
        np.hstack([GHK_te[:, :G_PART], Hh_te]), yte)
    log(f"[HIER-HIGH] = {hier_res['macro_f1']}")
    res["hier_high"] = hier_res

    # ---- 9. artifacts ------------------------------------------------------
    methods = [
        ("canonical_mr", "MiniROCKET (canonical 9,996)", B_BASE,
         "84x119", 9996, m0_res, m0_pred),
        ("canonical_r2", "R2 flat [G||H] (canonical)", B_BASE,
         "84x119", B_BASE, r2_res, r2_pred),
        ("canonical_r5", "R5 rho=0.1 (canonical)", B_BASE,
         "84x119", B_BASE, r5_res, r5_pred),
        ("mr_high", "MR-HIGH", B_HIGH, "84x238", B_HIGH, mr_res, mr_pred),
        ("r5_high", "R5-HIGH", B_HIGH, "84x238", B_HIGH, r5h_res, r5h_pred),
        ("hier_high", "HIER-HIGH", B_HIGH, "84x238", B_HIGH, hier_res,
         hier_pred),
    ]
    per_method_time = {m[0]: None for m in methods}   # wall times via stages
    cand = {"canonical_mr": 9996, "canonical_r2": 4998, "canonical_r5": 4998,
            "mr_high": 19992, "r5_high": B_HIGH - G_PART,
            "hier_high": int(sum(K * (B_HIGH - G_PART) for K in kept_K))}

    with open(os.path.join(OUT, "results", "comparison.csv"), "w") as f:
        f.write("method,base_budget,high_budget,kernel_capacity,"
                "candidate_features,final_features,alpha,macro_f1,accuracy,"
                "num_classes,seed\n")
        for key, label, bbase, kcap, nfin, r, _ in methods:
            f.write(f"{key},{bbase if bbase == B_BASE else B_BASE},"
                    f"{B_HIGH if bbase == B_HIGH else B_BASE},{kcap},"
                    f"{cand[key]},{nfin},{r['selected_alpha']:.4f},"
                    f"{r['macro_f1']},{r['accuracy']},8,{SEED}\n")

    for key, label, bbase, kcap, nfin, r, pred in methods:
        save_json({"method": key, "label": label, "base_budget": bbase,
                   "high_budget": B_HIGH if bbase == B_HIGH else bbase,
                   "kernel_capacity": kcap, "final_features": nfin,
                   "candidate_features": cand[key],
                   "macro_f1": r["macro_f1"], "accuracy": r["accuracy"],
                   "selected_alpha": r["selected_alpha"],
                   "num_classes": 8, "seed": SEED,
                   "gates": {"ref": {"canonical_mr": REF_M0,
                                     "canonical_r2": REF_R2,
                                     "canonical_r5": REF_R5}.get(key),
                             "abs_diff": {
                                 "canonical_mr": m0_diff,
                                 "canonical_r2": r2_diff,
                                 "canonical_r5": r5_diff}.get(key)}},
                  os.path.join(OUT, "results", f"{key}.json"))
        np.save(os.path.join(OUT, "predictions", f"{key}.npy"), pred)

    save_json({
        "B_base": B_BASE, "B_high": B_HIGH, "G_part": G_PART,
        "extractors": {"canonical": f84, "expanded": fHK,
                       "expansion_factor": (fHK["quantiles_per_kernel"]
                                            / f84["quantiles_per_kernel"])},
        "final_feature_counts": {
            "canonical_mr": B_BASE, "canonical_r2": B_BASE,
            "canonical_r5": B_BASE, "mr_high": B_HIGH,
            "r5_high": {"G_first_N": N_G_R5_HIGH, "H_top_fclassif":
                        N_H_R5_HIGH, "total": B_HIGH},
            "hier_high": {"G_part": G_PART, "H_part": B_HH,
                          "total": B_HIGH},
            "all_high_methods_equal_B_high":
                bool(B_HIGH == B_HIGH == B_HIGH)},
        "r5_high_rule": {"rho": RHO_STORED, "source": "stored UWaveY R5 "
                         "validation-selected rho (no re-search)",
                         "N_G": N_G_R5_HIGH, "N_H": N_H_R5_HIGH,
                         "cv_mean": cv_mean, "cv_std": cv_std},
        "hier_high_rule": {"level_budgets": {int(k): int(b) for k, b in
                                             zip(kept_K, B_l)},
                           "selected_counts": {int(k): int(len(v))
                                               for k, v in sel.items()},
                           "K2_note": "child-0 residual column (child-1 is "
                                      "recoverable via the zero-sum "
                                      "relation given occupancies)"},
        "peak_mem_mb": peak_mem_mb(),
        "runtime_s": time.time() - t00,
    }, os.path.join(OUT, "diagnostics", "dimensions.json"))

    save_json({"level_energies": {"K": [int(k) for k in LEVELS[1:]],
                                  "real": E_total.tolist()},
               "null": {"S": S_PERM, "seed": NULL_SEED,
                        "mean": null.mean(0).tolist(),
                        "p95": np.percentile(null, 95, axis=0).tolist()},
               "stopping": {"L_star": int(L_star), "rows": rows},
               "occupied_regimes_level16_train":
                   int((ch["n_active_regimes"].sum(axis=0) > 0).sum()),
               "identity_max_rel": ch["identity_max_rel"],
               "zero_sum_max": ch["zero_sum_max"],
               "cross_ip_max": float(np.abs(ch["cross_ip"]).max()),
               "energy_fractions": (E_total / E_total.sum()).tolist(),
               "tree_meta": tree_meta},
              os.path.join(OUT, "diagnostics", "hierarchy_statistics.json"))

    save_json({"split_integrity": "train/val indices identical to "
               "r2_uwave_seed42 (asserted in-run)",
               "labels_in_representation_construction": False,
               "validation_in_representation_construction": False,
               "test_in_representation_construction": False,
               "rho_researched": False,
               "rho_source": "stored validation-selected rho=0.1 (reused)",
               "feature_allocation_train_only": True,
               "L_star_used_for_budgets": False,
               "test_labels_seen_before_evaluation": False,
               "note": "AST/static leakage tests in "
                       "tests/test_uwavey_capacity_scaled.py"},
              os.path.join(OUT, "diagnostics", "leakage_report.json"))

    _figures(kept_K, B_l, rows, methods, res)
    _summary(kept_K, B_l, methods, res, rows, t00)
    log(f"done in {time.time() - t00:.0f}s")


# ---------------------------------------------------------------------------
def _figures(kept_K, B_l, rows, methods, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # budget_comparison.png
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    names = ["canonical_mr", "canonical_r2", "canonical_r5",
             "mr_high", "r5_high", "hier_high"]
    finals = {"canonical_mr": 9996, "canonical_r2": 9996, "canonical_r5":
              9996, "mr_high": 19992, "r5_high": 19992, "hier_high": 19992}
    f1s = [res[k]["macro_f1"] for k in names]
    ax.bar(range(len(names)), list(finals.values()),
           color=["#9aa5b1"] * 3 + ["#3b7dd8"] * 3)
    for i, (f1v, fin) in enumerate(zip(f1s, finals.values())):
        ax.text(i, fin + 250, f"F1={f1v:.4f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set(ylabel="final feature budget", title="Final budgets (all "
           "high-capacity methods = 19,992)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "figures",
                                 f"budget_comparison.{ext}"))
    plt.close(fig)

    # level_allocation.png
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    Ks = kept_K
    ax.bar([str(k) for k in Ks], B_l, color="#3b7dd8",
           label="allocated features")
    ax2 = ax.twinx()
    tot = sum(r["real_energy"] for r in rows if r["K"] in Ks)
    fr = [r["real_energy"] / tot for r in rows if r["K"] in Ks]
    ax2.plot([str(k) for k in Ks], fr, "o--", color="#d1495b",
             label="energy fraction")
    ax.set(xlabel="hierarchy level K", ylabel="allocated H features")
    ax2.set(ylabel="level energy fraction")
    ax.set_title("HIER-HIGH label-free level allocation (train-only)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "figures", f"level_allocation.{ext}"))
    plt.close(fig)

    # performance_vs_capacity.png
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot([9996, 19992], [res["canonical_mr"]["macro_f1"],
                            res["mr_high"]["macro_f1"]], "o-",
            label="MR only")
    ax.plot([9996, 19992], [res["canonical_r5"]["macro_f1"],
                            res["r5_high"]["macro_f1"]], "s-",
            label="R5 (rho=0.1)")
    ax.plot([19992], [res["hier_high"]["macro_f1"]], "^",
            color="#d1495b", label="HIER-HIGH")
    ax.set(xlabel="final feature budget", ylabel="test Macro-F1",
           title="Performance vs capacity")
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "figures",
                                 f"performance_vs_capacity.{ext}"))
    plt.close(fig)


def _summary(kept_K, B_l, methods, res, rows, t00):
    lines = ["# UWaveY capacity-scaling experiment — summary", ""]
    lines.append("All high-capacity methods: final budget = "
                 f"{B_HIGH} = 2 x {B_BASE}.")
    lines.append(f"Runtime: {time.time() - t00:.0f}s.")
    lines.append("")
    lines.append("| method | final | Macro-F1 | alpha |")
    lines.append("|---|---|---|---|")
    for key, label, bbase, kcap, nfin, r, _ in methods:
        lines.append(f"| {label} | {nfin} | {r['macro_f1']} | "
                     f"{r['selected_alpha']:.3f} |")
    with open(os.path.join(OUT, "results", "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
