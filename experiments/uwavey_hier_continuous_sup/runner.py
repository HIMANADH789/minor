"""HIER-CONTINUOUS-SUP - unified-pool controlled ablation (UWaveY, seed 42).

ONE substantive change vs HIER-HIGH-SUP: the protected G/H boundary is
removed.  The full expanded root matrix (19,992 candidates - level-0 of the
hierarchy: PPV_m^(1) = G_m) and the 449,820 hierarchical delta candidates
(telescoping residuals Delta_{m,c}^(l) = PPV_{m,c} - PPV_{m,parent(c)},
l in {2,4,8,16}) form ONE unified pool  C in R^(N x 469,812); a single
top_f_select(C, y_dev, 19,992) ranking picks the final 19,992 features.

No rho, no N_G/N_H, no per-level quota - the final composition EMERGES
from the supervised ranking (reported as diagnostics only).

Everything else is locked and identical to HIER-HIGH-SUP (canonical split,
per-sample z-norm, expanded MiniROCKET, frozen train-only hierarchy,
delta-bank definition, Ridge protocol, seed 42).  All locked components are
imported, not reimplemented.
"""

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DS = "UWaveGestureLibraryY"
SEED = 42

B_BASE = 9996
B_HIGH = 19992
G_PART = 4998
H_KERNELS = B_HIGH - G_PART      # 14,994 base hierarchical kernel carriers
KEPT_K = [2, 4, 8, 16]
EDGES = sum(KEPT_K)              # 30 hierarchy edges (children summed)
H_CAND = EDGES * H_KERNELS       # 449,820
ROOTS = B_HIGH                   # 19,992 full root candidates (level 0)
N_CAND = ROOTS + H_CAND          # 469,812
ALPHAS = np.logspace(-4, 4, 20)

CAP_RESULTS = os.path.join(ROOT, "results", "uwavey_capacity_scaled", "seed42")
NEST_RESULTS = os.path.join(ROOT, "results", "uwavey_nested_hierarchical", "seed42")
REF_MR_HIGH = 0.7477
GATE_TOL = 0.002
REFS = {"canonical_mr": 0.7543, "mr_high": 0.7477, "r5_high": 0.7772,
        "hier_high": 0.7230, "hier_high_sup": 0.7823}

OUT = os.path.join(ROOT, "results", "uwavey_hier_continuous_sup", "seed42")

# reused canonical pieces (imported, not reimplemented)
from experiments.uwavey_nested_hierarchical.runner import (   # noqa: E402
    _encoder_latents, compute_activations, load_context_model, load_data,
    ppv_all, ridge_eval, znorm)
from experiments.uwavey_nested_hierarchical.runner import delta_banks  # noqa: E402,E501
from experiments.uwavey_capacity_scaled.runner import (       # noqa: E402
    extractor_facts, fit_minirocket_nk, hier_banks_test, peak_mem_mb,
    r5_cv_fixed_rho, top_f_select)
from experiments.uwavey_hier_high_sup.runner import col_meta  # noqa: E402
from models.nested_regimes.model import (                     # noqa: E402
    assign_levels, build_latent_tree, nesting_errors)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(smoke=False):
    global OUT
    if smoke:
        OUT = OUT + "_smoke"
    for sub in ("results", "figures", "diagnostics", "predictions"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    t00 = time.time()
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device: {device}  smoke={smoke}")

    # ---- 1. canonical data, split identity ------------------------------
    d, Xtr, Xva, Xte, ytr, yva, yte = load_data()
    n_tr, n_va, n_te = len(ytr), len(yva), len(yte)
    y_dev = np.concatenate([ytr, yva]).astype(np.int64)
    if smoke:
        Xtr, ytr = Xtr[:200], ytr[:200]
        Xva, yva = Xva[:60], yva[:60]
        Xte, yte = Xte[:200], yte[:200]
        n_tr, n_va, n_te = len(ytr), len(yva), len(yte)
        y_dev = np.concatenate([ytr, yva]).astype(np.int64)
    ntr_ref = np.load(os.path.join(NEST_RESULTS, "train_indices.npy"))
    nva_ref = np.load(os.path.join(NEST_RESULTS, "val_indices.npy"))
    nte_ref = np.load(os.path.join(NEST_RESULTS, "test_indices.npy"))
    split_ok = (len(ntr_ref) == 761 and len(nva_ref) == 135 and
                len(nte_ref) == 3582)
    log(f"data: train {Xtr.shape}, val {Xva.shape}, test {Xte.shape}; "
        f"split identity vs stored uwavey_nested indices: {split_ok}")
    assert split_ok or smoke

    # ---- 2. frozen hierarchy (LOCKED; rebuilt deterministically) ---------
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
    if not smoke:
        saved = np.load(os.path.join(NEST_RESULTS,
                                     "hierarchy_assignments.npy"))
        assert np.array_equal(reg16_tr, saved), \
            "hierarchy differs from the stored HIER-HIGH artifact"
        log("hierarchy regression: reg16_tr == stored "
            "hierarchy_assignments.npy (byte-identical)")
    lat_va = _encoder_latents(model, Xva_z, device)
    reg16_va = assign_levels(lat_va.reshape(-1, lat_va.shape[-1]),
                             C)[4].reshape(n_va, -1).astype(np.int64)
    del lat_va
    lat_te = _encoder_latents(model, Xte_z, device)
    reg16_te = assign_levels(lat_te.reshape(-1, lat_te.shape[-1]),
                             C)[4].reshape(n_te, -1).astype(np.int64)
    del lat_te
    del model
    log("tree: levels 1,2,4,8,16 frozen (TRAIN latents only); "
        "val/test transform-only")

    # ---- 3. expanded MiniROCKET (LOCKED) ---------------------------------
    exHK = fit_minirocket_nk(Xtr_z, B_HIGH)
    fHK = extractor_facts(exHK)
    F_H = fHK["total_features"] - G_PART
    assert fHK["physical_kernels"] == 84
    if not smoke:
        stored_dims = json.load(open(os.path.join(
            CAP_RESULTS, "diagnostics", "dimensions.json")))
        assert fHK == stored_dims["extractors"]["expanded"], (fHK,)
        assert fHK["total_features"] == ROOTS and F_H == H_KERNELS \
            and 30 * F_H == H_CAND
        log(f"expanded extractor facts match stored: {fHK}")
    else:
        log(f"expanded extractor facts (smoke): {fHK}")

    # ---- 4. FULL root matrix (NO 17,993 preselection) --------------------
    GHK_trva = ppv_all(exHK, np.vstack([Xtr_z, Xva_z]))
    GHK_te = ppv_all(exHK, Xte_z)
    assert GHK_trva.shape == (n_tr + n_va, ROOTS)
    assert GHK_te.shape == (n_te, ROOTS)
    log(f"root matrix (level 0): {GHK_trva.shape} - full width, "
        f"no G/H cut")

    # ---- 5. MR-HIGH reproduction gate (regression assertion) -------------
    mr_gate, _ = ridge_eval(GHK_trva, y_dev, GHK_te, yte)
    gate_diff = abs(mr_gate["macro_f1"] - REF_MR_HIGH)
    log(f"[gate MR-HIGH reproduction] = {mr_gate['macro_f1']} "
        f"(stored {REF_MR_HIGH}, diff {gate_diff:.4f})")
    assert smoke or gate_diff <= GATE_TOL, gate_diff

    # ---- 6. FULL hierarchical candidate pool, train+val ------------------
    # signals-chunked: never materialize (896, 19,992, 315) activations
    pool_parts = []
    sig_chunk = 32
    for c0 in range(0, n_tr + n_va, sig_chunk):
        c1 = min(c0 + sig_chunk, n_tr + n_va)
        act, valid = compute_activations(
            exHK, np.vstack([Xtr_z, Xva_z])[c0:c1])
        pool_parts.append(delta_banks(act[:, G_PART:], valid[G_PART:],
                                      np.vstack([reg16_tr, reg16_va])[c0:c1],
                                      KEPT_K, delta_cols=None,
                                      sample_chunk=96))
        del act
    Hcand_trva = np.vstack(pool_parts)
    del pool_parts
    assert Hcand_trva.shape == (n_tr + n_va, H_CAND), Hcand_trva.shape
    log(f"hierarchical candidate pool: {Hcand_trva.shape} "
        f"({EDGES} edges x {F_H} kernels)")

    # ---- 7. UNIFIED pool - THE defining change ---------------------------
    C_dev = np.hstack([GHK_trva, Hcand_trva])
    del GHK_trva, Hcand_trva
    assert C_dev.shape == (n_tr + n_va, N_CAND), C_dev.shape
    assert np.isfinite(C_dev).all()
    log(f"unified pool: {C_dev.shape} "
        f"(root {ROOTS} + delta {H_CAND} = {N_CAND})")

    # ---- 8. CV diagnostic (fold-internal selection, reported only) -------
    # the r5_cv_fixed_rho machinery with n_g=0 exposes the FULL pool to
    # fold-internal selection: one ranking per fold over all 469,812
    cv_mean, cv_std = r5_cv_fixed_rho(
        np.zeros((C_dev.shape[0], 0)), C_dev, y_dev, 0, B_HIGH)
    log(f"[CV unified fold-internal] macro-F1 = {cv_mean:.4f} "
        f"+/- {cv_std:.4f}  (diagnostic only)")

    # ---- 9. THE ONE SELECTION: top-19,992 of the unified pool ------------
    # identical selector semantics as R5-HIGH / HIER-HIGH-SUP: f_classif,
    # NaN->0, stable argsort, top-N; single ranking, NO composition quotas.
    from sklearn.feature_selection import f_classif
    f_stat, _ = f_classif(C_dev, y_dev)
    f_stat = np.nan_to_num(f_stat, nan=0.0)
    sel = np.argsort(-f_stat, kind="stable")[:B_HIGH]
    assert np.array_equal(sel, top_f_select(C_dev, y_dev, B_HIGH))
    assert len(sel) == B_HIGH
    log(f"unified supervised selection: top {B_HIGH} of {C_dev.shape[1]} "
        f"(single ranking, no G/H boundary)")

    # ---- 10. selected-feature metadata (diagnostic ONLY) ------------------
    meta = []
    for c in sel:
        if c < ROOTS:
            meta.append((0, -1, -1, int(c)))            # root candidate
        else:
            K, j, p, kg = col_meta(int(c) - ROOTS, F_H)
            meta.append((K, j, p, kg))
    lv_counts = {0: 0, 2: 0, 4: 0, 8: 0, 16: 0}
    for m in meta:
        lv_counts[m[0]] += 1
    assert sum(lv_counts.values()) == B_HIGH
    log(f"selected composition (EMERGED, diagnostic only): {lv_counts} "
        f"(root {lv_counts[0] / B_HIGH:.1%}, "
        f"K2 {lv_counts[2] / B_HIGH:.1%}, K4 {lv_counts[4] / B_HIGH:.1%}, "
        f"K8 {lv_counts[8] / B_HIGH:.1%}, K16 {lv_counts[16] / B_HIGH:.1%})")

    # ONE shared column order for dev and test: roots first, then deltas
    # level-grouped [K2 | K4 | K8 | K16] (rank order preserved within each
    # level) -- exactly the layout hier_banks_test emits, so the dev
    # gather and the frozen-transform test gather are semantically equal.
    # Column ORDER is cosmetic: the selected SET is unchanged and Ridge is
    # permutation-invariant in features (same practice as HIER-HIGH-SUP).
    lv_rank = {0: -1, 2: 0, 4: 1, 8: 2, 16: 3}
    order = np.argsort([lv_rank[m[0]] for m in meta], kind="stable")
    sel = sel[order]
    meta = [meta[i] for i in order]
    lv_seq = [m[0] for m in meta if m[0] != 0]
    assert lv_seq == sorted(lv_seq), "shared order not level-grouped"
    log("final column order: roots first, then K2/K4/K8/K16 "
        "(selection set unchanged)")

    # frozen-transform test gather: per-level (kernel, child) ids in the
    # SAME row order as the selected columns (Ridge is permutation-
    # invariant in features - one shared column order for both splits)
    sel_dict = {K: np.array([(kg - G_PART, j) for (Kk, j, p, kg) in meta
                             if Kk == K], dtype=np.int64)
                for K in KEPT_K}
    assert all(len(v) == lv_counts[K] for K, v in sel_dict.items())
    n_delta_sel = B_HIGH - lv_counts[0]

    with open(os.path.join(OUT, "diagnostics",
                           "selected_counts_by_level.csv"), "w") as f:
        f.write("level,source_type,pool_columns,selected,"
                "share_of_final,share_of_pool\n")
        f.write(f"0,root,{ROOTS},{lv_counts[0]},"
                f"{lv_counts[0] / B_HIGH:.6f},{lv_counts[0] / ROOTS:.6f}\n")
        for K, cbefore in ((2, 0), (4, 2), (8, 6), (16, 14)):
            pool_k = K * F_H
            sel_k = lv_counts[K]
            f.write(f"{K},delta,{pool_k},{sel_k},{sel_k / B_HIGH:.6f},"
                    f"{sel_k / pool_k:.6f}\n")

    with open(os.path.join(OUT, "diagnostics",
                           "selected_feature_metadata.csv"), "w") as f:
        f.write("selected_rank,candidate_index,source_type,level,parent,"
                "child,kernel_id,f_stat\n")
        for rank, (c, m) in enumerate(zip(sel, meta)):
            K, j, p, kg = m
            if K == 0:
                f.write(f"{rank},{c},root,0,NA,NA,{kg},{f_stat[c]:.6g}\n")
            else:
                f.write(f"{rank},{c},delta,{K},{p},{j},{kg},"
                        f"{f_stat[c]:.6g}\n")

    # ---- 11. final 19,992-wide representation for BOTH splits ------------
    # (shared column order established in section 10)
    root_idx = np.array([c for c in sel if c < ROOTS], dtype=np.int64)
    n_root_sel = len(root_idx)
    assert n_root_sel + n_delta_sel == B_HIGH
    Z_dev = C_dev[:, sel]
    del C_dev
    assert Z_dev.shape == (n_tr + n_va, B_HIGH)
    # test: root columns direct + selected delta columns via the frozen
    # transform (signals-chunked; never materialize full test activations)
    Hcand_te = hier_banks_test(exHK, Xte_z, reg16_te, KEPT_K, sel_dict)
    assert Hcand_te.shape == (n_te, n_delta_sel), Hcand_te.shape
    Z_te = np.empty((n_te, B_HIGH), dtype=np.float64)
    Z_te[:, :n_root_sel] = GHK_te[:, root_idx]
    Z_te[:, n_root_sel:] = Hcand_te
    del GHK_te, Hcand_te
    assert np.isfinite(Z_dev).all() and np.isfinite(Z_te).all()
    log(f"final representation: {Z_dev.shape} / test {Z_te.shape} "
        f"(shared order: {n_root_sel} roots + {n_delta_sel} deltas)")

    # ---- 12. canonical Ridge (identical protocol) -------------------------
    res, pred = ridge_eval(Z_dev, y_dev, Z_te, yte)
    log(f"[HIER-CONTINUOUS-SUP] = {res['macro_f1']}  "
        f"(alpha={res['selected_alpha']})")

    # ---- 13. deltas vs saved references (NOT rerun) -----------------------
    deltas = {k: round(res["macro_f1"] - v, 4) for k, v in REFS.items()}
    np.save(os.path.join(OUT, "predictions", "test_predictions.npy"),
            pred.astype(np.int64))

    # ---- 14. artifacts ----------------------------------------------------
    with open(os.path.join(OUT, "results",
                           "hier_continuous_sup.json"), "w") as f:
        json.dump({
            "method": "hier_continuous_sup",
            "label": "HIER-CONTINUOUS-SUP",
            "dataset": DS, "seed": SEED,
            "base_budget": B_BASE, "high_budget": B_HIGH,
            "kernel_capacity": "84x238",
            "final_features": B_HIGH,
            "candidate_pool": {"root": ROOTS, "delta": H_CAND,
                               "unified": N_CAND},
            "selection": "ONE supervised ANOVA-F ranking over the unified "
                         "469,812-candidate pool (train+val labels, "
                         "canonical R5 final-stage protocol); no rho, no "
                         "N_G/N_H, no per-level quota",
            "changed_vs_hier_high_sup":
                "protected G/H boundary removed: G_full (19,992 level-0 "
                "roots) + delta pool (449,820) compete in one ranking",
            "selected_composition": lv_counts,
            "cv_unified_fold_internal": {"mean": cv_mean, "std": cv_std,
                                         "kfold": 5},
            "macro_f1": res["macro_f1"], "accuracy": res["accuracy"],
            "alpha": res["selected_alpha"],
            "deltas_vs": deltas,
            "gate_mr_high": {"ref": REF_MR_HIGH,
                             "reproduced": mr_gate["macro_f1"],
                             "diff": round(gate_diff, 4),
                             "tol": GATE_TOL},
            "runtime_s": round(time.time() - t00, 1),
            "peak_mem_mb": round(peak_mem_mb(), 1),
            "smoke": smoke,
        }, f, indent=1)

    with open(os.path.join(OUT, "diagnostics", "dimensions.json"), "w") as f:
        json.dump({
            "split": {"train": n_tr, "validation": n_va, "test": n_te,
                      "T": int(Xtr.shape[1]), "n_classes":
                      int(len(np.unique(ytr)))},
            "extractors": {"expanded": fHK},
            "pools": {"root_candidates": ROOTS, "delta_candidates": H_CAND,
                      "edges": EDGES, "kernels_per_edge": F_H,
                      "unified_candidates": N_CAND},
            "final": {"selected": B_HIGH,
                      "Z_dev": list(Z_dev.shape),
                      "Z_test": list(Z_te.shape),
                      "composition": lv_counts},
        }, f, indent=1)

    with open(os.path.join(OUT, "diagnostics", "leakage_report.json"),
              "w") as f:
        json.dump({
            "minirocket_fit": "train signals only (seed 42)",
            "hierarchy": "TRAIN latents only (frozen seed-42 SSL encoder, "
                         "recursive 2-means); val/test transform-only; "
                         "reg16_tr byte-identical to the stored "
                         "uwavey_nested artifact",
            "delta_pool_and_root_matrix": "frozen transform, label-free",
            "unified_selection": "train+val labels (canonical R5 "
                                 "final-stage protocol - identical to "
                                 "HIER-HIGH-SUP / R5-HIGH)",
            "cv_diagnostic": "5-fold StratifiedKFold(shuffle=True, "
                             "random_state=42); fold-internal selection "
                             "on fold-train labels ONLY",
            "test_labels": "touched only inside ridge_eval for scoring",
            "no_quota": "composition emerged from the ranking; reported "
                        "as diagnostics only",
        }, f, indent=1)

    with open(os.path.join(OUT, "diagnostics",
                           "selection_protocol.md"), "w") as f:
        f.write(
            "# Selection protocol - HIER-CONTINUOUS-SUP\n\n"
            "## Unified pool (the defining change)\n\n"
            "C = [G_full || Delta_full] with\n"
            f"- G_full: {ROOTS} root candidates (full expanded PPV matrix; "
            "PPV_m^(1) = G_m, level 0 of the hierarchy)\n"
            f"- Delta_full: {H_CAND} hierarchical telescoping residuals "
            f"({EDGES} edges x {F_H} kernels; levels K=2,4,8,16)\n"
            f"- total: {N_CAND} candidates (asserted C_dev.shape == "
            f"(896, {N_CAND}))\n\n"
            "## Stage 1 - CV diagnostic (never selects)\n\n"
            "5-fold StratifiedKFold(shuffle=True, random_state=42) over "
            "train+val (896 rows). Per fold: top_f_select on the "
            "fold-train rows/labels ONLY (fold-internal selection over the "
            "full unified pool), held-out rows transformed with those "
            "identities, RidgeClassifierCV(ALPHAS) fit on fold-train, "
            "macro-F1 on the fold. Diagnostic only.\n\n"
            "## Stage 2 - final selection (canonical R5 protocol)\n\n"
            "top_f_select(C_dev, y_dev, 19,992) on the FULL dev matrix with "
            "train+val labels - exactly the protocol R5-HIGH and "
            "HIER-HIGH-SUP use for their final H selection. Selector "
            "semantics identical (sklearn f_classif, NaN->0, "
            "np.argsort(-f, kind='stable'), top-N). NO separate G stage, "
            "no rho, no N_G/N_H, no per-level quota.\n\n"
            "## Stage 3 - refit + one official test evaluation\n\n"
            "RidgeClassifierCV(ALPHAS=logspace(-4,4,20)) fit on the "
            f"selected {B_HIGH}-column representation over train+val; "
            "single evaluation on the 3,582-row official test split.\n\n"
            "## Resulting composition (emerged, diagnostic only)\n\n"
            f"root {lv_counts[0]}, K2 {lv_counts[2]}, K4 {lv_counts[4]}, "
            f"K8 {lv_counts[8]}, K16 {lv_counts[16]} "
            f"(sum {sum(lv_counts.values())} = {B_HIGH}).\n")

    # comparison.csv + figures + summary ------------------------------------
    rows = [("canonical_mr", "Canonical MR", 9996, "84x119", "NA"),
            ("mr_high", "MR-HIGH", 19992, "84x238", "NA"),
            ("r5_high", "R5-HIGH", 19992, "84x238", "14994"),
            ("hier_high", "HIER-HIGH", 19992, "84x238", "449820"),
            ("hier_high_sup", "HIER-HIGH-SUP", 19992, "84x238", "449820"),
            ("hier_continuous_sup", "HIER-CONTINUOUS-SUP", 19992, "84x238",
             "469812")]
    with open(os.path.join(OUT, "results", "comparison.csv"), "w") as f:
        f.write("method,label,final_features,root_selected,k2_selected,"
                "k4_selected,k8_selected,k16_selected,candidate_pool,"
                "kernel_capacity,macro_f1,accuracy,alpha,runtime_s,"
                "peak_mem_mb\n")
        for key, label, ff, kcap, pool in rows:
            if key == "hier_continuous_sup":
                f.write(f"{key},{label},{ff},{lv_counts[0]},{lv_counts[2]},"
                        f"{lv_counts[4]},{lv_counts[8]},{lv_counts[16]},"
                        f"{pool},{kcap},{res['macro_f1']},{res['accuracy']},"
                        f"{res['selected_alpha']},"
                        f"{round(time.time() - t00, 1)},"
                        f"{round(peak_mem_mb(), 1)}\n")
            else:
                ref = REFS[key]
                f.write(f"{key},{label},{ff},NA,NA,NA,NA,NA,{pool},{kcap},"
                        f"{ref},NA,NA,NA,NA\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # comparison figure
        labels = [r[1] for r in rows]
        vals = [REFS.get(r[0]) if r[0] in REFS else res["macro_f1"]
                for r in rows]
        fig, ax = plt.subplots(figsize=(9, 4.2))
        cols = ["#9e9e9e"] * 5 + ["#1f77b4"]
        ax.bar(labels, vals, color=cols)
        ax.set_ylim(0.70, 0.80)
        ax.set_ylabel("Macro-F1 (official test)")
        ax.set_title("HIER-CONTINUOUS-SUP - unified pool vs incumbents "
                     "(UWaveY, seed 42)")
        for i, v in enumerate(vals):
            ax.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=8)
        plt.xticks(rotation=20, ha="right")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "figures",
                                 "comparison_macro_f1.png"), dpi=150)
        plt.close(fig)

        # selected-features-by-level figure
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        lv = ["root (0)", "K=2", "K=4", "K=8", "K=16"]
        cnt = [lv_counts[0], lv_counts[2], lv_counts[4], lv_counts[8],
               lv_counts[16]]
        ax.bar(lv, cnt, color="#1f77b4")
        ax.set_ylabel("selected features (of 19,992)")
        ax.set_title("Composition EMERGED from the unified ranking "
                     "(no quotas)")
        for i, v in enumerate(cnt):
            ax.text(i, v + 30, f"{v}\n({v / B_HIGH:.1%})", ha="center",
                    fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "figures",
                                 "selected_features_by_level.png"), dpi=150)
        plt.close(fig)

        # candidate-composition figure (pool make-up vs selection)
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        pool_cols = [ROOTS, 2 * F_H, 4 * F_H, 8 * F_H, 16 * F_H]
        sel_cols = cnt
        x = np.arange(5)
        ax.bar(x - 0.2, pool_cols, 0.4, label="pool candidates", log=True)
        ax.bar(x + 0.2, sel_cols, 0.4, label="selected", log=True)
        ax.set_xticks(x)
        ax.set_xticklabels(lv)
        ax.set_ylabel("count (log)")
        ax.set_title("Unified pool vs emerged selection")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "figures",
                                 "candidate_composition.png"), dpi=150)
        plt.close(fig)
        log("figures written")
    except Exception as e:  # figures must never kill the science
        log(f"figure generation skipped: {e!r}")

    rt = round(time.time() - t00, 1)
    mem = round(peak_mem_mb(), 1)
    with open(os.path.join(OUT, "results", "summary.md"), "w") as f:
        f.write(
            "# HIER-CONTINUOUS-SUP - summary (UWaveY, seed 42)\n\n"
            f"**Macro-F1 = {res['macro_f1']}** (accuracy "
            f"{res['accuracy']}, alpha {res['selected_alpha']})\n\n"
            "## The one change\n\n"
            "HIER-HIGH-SUP protects G at 17,993 (rho=0.1) and supervises "
            "only the 449,820-delta pool for 1,999 H slots. "
            "HIER-CONTINUOUS-SUP removes that boundary: the full 19,992 "
            "root matrix (level 0 of the same hierarchy) joins the 449,820 "
            "delta candidates in ONE pool of 469,812, and a single "
            "ANOVA-F ranking (train+val labels, canonical R5 protocol) "
            "picks exactly 19,992 features.\n\n"
            "## Comparison (saved references; only this model was run)\n\n"
            "| Method | Final | Macro-F1 | Delta vs this |\n|---|---|---|---|\n"
            "| Canonical MR | 9,996 | 0.7543 | "
            f"{round(res['macro_f1'] - REFS['canonical_mr'], 4):+} |\n"
            "| MR-HIGH | 19,992 | 0.7477 | "
            f"{round(res['macro_f1'] - REFS['mr_high'], 4):+} |\n"
            "| R5-HIGH | 19,992 | 0.7772 | "
            f"{round(res['macro_f1'] - REFS['r5_high'], 4):+} |\n"
            "| HIER-HIGH | 19,992 | 0.7230 | "
            f"{round(res['macro_f1'] - REFS['hier_high'], 4):+} |\n"
            "| HIER-HIGH-SUP | 19,992 | 0.7823 | "
            f"{round(res['macro_f1'] - REFS['hier_high_sup'], 4):+} |\n"
            f"| **HIER-CONTINUOUS-SUP** | 19,992 | **{res['macro_f1']}** | "
            "NA |\n\n"
            f"## Emerged composition (diagnostic)\n\nroot {lv_counts[0]} "
            f"({lv_counts[0] / B_HIGH:.1%}), K2 {lv_counts[2]} "
            f"({lv_counts[2] / B_HIGH:.1%}), K4 {lv_counts[4]} "
            f"({lv_counts[4] / B_HIGH:.1%}), K8 {lv_counts[8]} "
            f"({lv_counts[8] / B_HIGH:.1%}), K16 {lv_counts[16]} "
            f"({lv_counts[16] / B_HIGH:.1%}).\n\n"
            f"CV (fold-internal over the unified pool): {cv_mean:.4f} "
            f"+/- {cv_std:.4f}.\n\n"
            f"MR-HIGH reproduction gate: {mr_gate['macro_f1']} vs stored "
            f"{REF_MR_HIGH} (diff {gate_diff:.4f}).\n\n"
            f"Runtime {rt:.0f}s; peak process memory {mem} MB.\n")
    log(f"artifacts complete in {rt:.0f}s  OUT={OUT}")
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    r = main(smoke=a.smoke)
    print("DONE", r["macro_f1"])
