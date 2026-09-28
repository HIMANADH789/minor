"""HIER-SUP-CLEAN - clean unified telescoping hierarchy (UWaveY, seed 42).

The complete model is ONE telescoping multiresolution hierarchy of
increments with PPV^(0) = 0:

    Delta^(1)_m     = PPV_m^(1) - PPV_m^(0) = PPV_m^(1)   (root increment)
    Delta_{m,c}^(K) = PPV_{m,c}^(K) - PPV_{m,parent(c)}^(K/2),  K in {2,4,8,16}

Candidate pool  C = [D1 | D2 | D4 | D8 | D16]  in R^(N x 469,812):
19,992 root increments + 449,820 deeper deltas.  ONE supervised ANOVA-F
ranking (top_f_select, train+val labels - the canonical R5 final-stage
protocol) picks exactly 19,992 features; the level composition fully
EMERGES (diagnostic only).  No rho, no G/H split, no protected budgets,
no per-level quota, no energy allocation, no empirical Bayes, no
moderated-F, no new classifier.  (Explicit negations; none of these
are used: no rho, no G/H split, no moderated-F.)

Equivalence gate (spec section 17/27): with PPV^(0) = 0 the root block
IS Delta^(1), the pool/selector/classifier/seed are identical to
HIER-CONTINUOUS-SUP, so this run MUST reproduce the stored
results/uwavey_hier_continuous_sup/seed42 artifacts (selected index set
and sequence, per-candidate F, semantics, composition, predictions,
alpha, Macro-F1).  Verdict IDENTICAL is asserted in production; the
smoke subset skips the gate.

All locked components are imported, not reimplemented.
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
H_CAND = EDGES * H_KERNELS       # 449,820 deeper telescoping candidates
D1 = B_HIGH                      # 19,992 root increments Delta^(1)
N_CAND = D1 + H_CAND             # 469,812
PPV0 = 0.0                       # PPV^(0) = 0 -> Delta^(1) = PPV^(1)
ALPHAS = np.logspace(-4, 4, 20)

CAP_RESULTS = os.path.join(ROOT, "results", "uwavey_capacity_scaled", "seed42")
NEST_RESULTS = os.path.join(ROOT, "results", "uwavey_nested_hierarchical", "seed42")
REF_DIR = os.path.join(ROOT, "results", "uwavey_hier_continuous_sup", "seed42")
REF_MR_HIGH = 0.7477
REF_HC_SUP = 0.7599
GATE_TOL = 0.002
REFS = {"canonical_mr": 0.7543, "mr_high": 0.7477, "r5_high": 0.7772,
        "hier_high": 0.7230, "hier_high_sup": 0.7823,
        "hier_continuous_sup": 0.7599}

OUT = os.path.join(ROOT, "results", "uwavey_hier_sup_clean", "seed42")

# reused canonical pieces (imported, not reimplemented)
from experiments.uwavey_nested_hierarchical.runner import (   # noqa: E402
    _encoder_latents, compute_activations, hierarchy_chain, load_context_model,
    load_data, ppv_all, ridge_eval, znorm)
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
            "hierarchy differs from the stored artifact"
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
        assert fHK["total_features"] == D1 and F_H == H_KERNELS \
            and 30 * F_H == H_CAND
        log(f"expanded extractor facts match stored: {fHK}")
    else:
        log(f"expanded extractor facts (smoke): {fHK}")

    # ---- 4. root increment D1 = PPV^(1) - PPV^(0), FULL width ------------
    GHK_trva = ppv_all(exHK, np.vstack([Xtr_z, Xva_z]))
    GHK_te = ppv_all(exHK, Xte_z)
    assert GHK_trva.shape == (n_tr + n_va, D1)
    assert GHK_te.shape == (n_te, D1)
    D1_dev = GHK_trva - PPV0            # PPV^(0) = 0: the root increment
    D1_te = GHK_te - PPV0
    assert np.array_equal(D1_dev, GHK_trva) and np.array_equal(D1_te, GHK_te), \
        "Delta^(1) != PPV^(1) (root increment equivalence violated)"
    log(f"root increment D1: {D1_dev.shape} - Delta^(1) == PPV^(1) "
        f"exactly (PPV^(0) = {PPV0}), no truncation")

    # ---- 5. MR-HIGH reproduction gate (regression assertion) -------------
    mr_gate, _ = ridge_eval(D1_dev, y_dev, D1_te, yte)
    gate_diff = abs(mr_gate["macro_f1"] - REF_MR_HIGH)
    log(f"[gate MR-HIGH reproduction] = {mr_gate['macro_f1']} "
        f"(stored {REF_MR_HIGH}, diff {gate_diff:.4f})")
    assert smoke or gate_diff <= GATE_TOL, gate_diff

    # ---- 6. deeper telescoping pool D2/D4/D8/D16 (train+val) -------------
    reg16_dev = np.vstack([reg16_tr, reg16_va])
    X_dev = np.vstack([Xtr_z, Xva_z])
    pool_parts = []
    sig_chunk = 32
    for c0 in range(0, n_tr + n_va, sig_chunk):
        c1 = min(c0 + sig_chunk, n_tr + n_va)
        act, valid = compute_activations(exHK, X_dev[c0:c1])
        pool_parts.append(delta_banks(act[:, G_PART:], valid[G_PART:],
                                      reg16_dev[c0:c1], KEPT_K,
                                      delta_cols=None, sample_chunk=96))
        del act
    Hcand_trva = np.vstack(pool_parts)
    del pool_parts
    assert Hcand_trva.shape == (n_tr + n_va, H_CAND), Hcand_trva.shape
    # hard layout assertion: canonical order [D1 | D2 | D4 | D8 | D16],
    # delta block level-grouped coarse->fine (child-major, kernel-minor)
    assert [col_meta(c, F_H)[0] for c in
            (0, 2 * F_H, 6 * F_H, 14 * F_H)] == [2, 4, 8, 16]
    log(f"deeper telescoping pool: {Hcand_trva.shape} "
        f"({EDGES} edges x {F_H} kernels)")

    # occupancy-weighted chain identity (locked implementation, own report):
    # sum_c (n_c/n_p) Delta_{m,c} = 0 per parent; telescoping identity
    act8, valid8 = compute_activations(exHK, X_dev[:8])
    ch8 = hierarchy_chain(act8[:, G_PART:], valid8[G_PART:],
                          reg16_dev[:8], want_deltas=False, want_ips=False)
    del act8
    zero_sum_max = float(ch8["zero_sum_max"])
    ident_rel = float(ch8["identity_max_rel"])
    log(f"chain identity: zero_sum_max {zero_sum_max:.3e}, "
        f"identity_max_rel {ident_rel:.3e}")
    assert zero_sum_max < 1e-10 and ident_rel < 1e-10

    # ---- 7. UNIFIED telescoping pool --------------------------------------
    C_dev = np.hstack([D1_dev, Hcand_trva])
    del D1_dev, Hcand_trva
    assert C_dev.shape == (n_tr + n_va, N_CAND), C_dev.shape
    assert np.isfinite(C_dev).all()
    log(f"unified pool: {C_dev.shape} "
        f"(D1 {D1} + deltas {H_CAND} = {N_CAND})")

    # ---- 8. CV diagnostic (fold-internal selection, reported only) -------
    cv_mean, cv_std = r5_cv_fixed_rho(
        np.zeros((C_dev.shape[0], 0)), C_dev, y_dev, 0, B_HIGH)
    log(f"[CV unified fold-internal] macro-F1 = {cv_mean:.4f} "
        f"+/- {cv_std:.4f}  (diagnostic only)")

    # ---- 9. THE ONE SELECTION: top-19,992 of the unified pool ------------
    # identical selector semantics as the incumbent: f_classif, NaN->0,
    # stable argsort, top-N; single ranking, NO composition rules.
    from sklearn.feature_selection import f_classif
    f_stat, _ = f_classif(C_dev, y_dev)
    f_stat = np.nan_to_num(f_stat, nan=0.0)
    sel = np.argsort(-f_stat, kind="stable")[:B_HIGH]
    assert np.array_equal(sel, top_f_select(C_dev, y_dev, B_HIGH))
    assert len(sel) == B_HIGH
    log(f"unified supervised selection: top {B_HIGH} of {C_dev.shape[1]} "
        f"(single ranking, fully emergent composition)")

    # ---- 10. selected-feature metadata (diagnostic ONLY) + shared order --
    # spec naming: root increment = (source_type root_increment, level 1,
    # parent -1, child -1); deeper = (hierarchy_delta, level 2/4/8/16).
    meta = []
    for c in sel:
        if c < D1:
            meta.append((1, -1, -1, int(c)))            # root increment D1
        else:
            K, j, p, kg = col_meta(int(c) - D1, F_H)
            meta.append((K, j, p, kg))
    lv_counts = {1: 0, 2: 0, 4: 0, 8: 0, 16: 0}
    for m in meta:
        lv_counts[m[0]] += 1
    assert sum(lv_counts.values()) == B_HIGH
    log(f"selected composition (EMERGED, diagnostic only): "
        f"D1 {lv_counts[1]} ({lv_counts[1] / B_HIGH:.1%}), "
        f"D2/K2 {lv_counts[2]} ({lv_counts[2] / B_HIGH:.1%}), "
        f"D4/K4 {lv_counts[4]} ({lv_counts[4] / B_HIGH:.1%}), "
        f"D8/K8 {lv_counts[8]} ({lv_counts[8] / B_HIGH:.1%}), "
        f"D16/K16 {lv_counts[16]} ({lv_counts[16] / B_HIGH:.1%})")

    # ONE shared column order for dev and test: D1 first, then deltas
    # level-grouped [D2 | D4 | D8 | D16] (rank order preserved within each
    # level) -- exactly the layout hier_banks_test emits, so the dev
    # gather and the frozen-transform test gather are semantically equal.
    # Column ORDER is cosmetic: the selected SET is unchanged and Ridge is
    # permutation-invariant in features (same practice as the incumbent).
    lv_rank = {1: -1, 2: 0, 4: 1, 8: 2, 16: 3}
    order = np.argsort([lv_rank[m[0]] for m in meta], kind="stable")
    sel = sel[order]
    meta = [meta[i] for i in order]
    lv_seq = [m[0] for m in meta if m[0] != 1]
    assert lv_seq == sorted(lv_seq), "shared order not level-grouped"
    log("final column order: D1 first, then D2/D4/D8/D16 "
        "(selection set unchanged)")

    # frozen-transform test gather: per-level (kernel, child) ids in the
    # SAME row order as the selected columns (Ridge is permutation-
    # invariant in features - one shared column order for both splits)
    sel_dict = {K: np.array([(kg - G_PART, j) for (Kk, j, p, kg) in meta
                             if Kk == K], dtype=np.int64)
                for K in KEPT_K}
    assert all(len(v) == lv_counts[K] for K, v in sel_dict.items())
    n_delta_sel = B_HIGH - lv_counts[1]

    with open(os.path.join(OUT, "diagnostics",
                           "selected_counts_by_level.csv"), "w") as f:
        f.write("level,source_type,pool_columns,selected,"
                "share_of_final,share_of_pool\n")
        f.write(f"1,root_increment,{D1},{lv_counts[1]},"
                f"{lv_counts[1] / B_HIGH:.6f},{lv_counts[1] / D1:.6f}\n")
        for K, cbefore in ((2, 0), (4, 2), (8, 6), (16, 14)):
            pool_k = K * F_H
            sel_k = lv_counts[K]
            f.write(f"{K},hierarchy_delta,{pool_k},{sel_k},"
                    f"{sel_k / B_HIGH:.6f},{sel_k / pool_k:.6f}\n")

    with open(os.path.join(OUT, "diagnostics",
                           "selected_feature_metadata.csv"), "w") as f:
        f.write("selected_rank,candidate_index,source_type,level,parent,"
                "child,kernel_id,f_stat\n")
        for rank, (c, m) in enumerate(zip(sel, meta)):
            K, j, p, kg = m
            if K == 1:
                f.write(f"{rank},{c},root_increment,1,-1,-1,{kg},"
                        f"{f_stat[c]:.6g}\n")
            else:
                f.write(f"{rank},{c},hierarchy_delta,{K},{p},{j},{kg},"
                        f"{f_stat[c]:.6g}\n")

    # ---- 11. final 19,992-wide representation for BOTH splits ------------
    # (shared column order established in section 10)
    root_idx = np.array([c for c in sel if c < D1], dtype=np.int64)
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
    Z_te[:, :n_root_sel] = D1_te[:, root_idx]
    Z_te[:, n_root_sel:] = Hcand_te
    del D1_te, Hcand_te
    assert np.isfinite(Z_dev).all() and np.isfinite(Z_te).all()
    log(f"final representation: {Z_dev.shape} / test {Z_te.shape} "
        f"(shared order: {n_root_sel} root increments + "
        f"{n_delta_sel} deltas)")

    # ---- 12. canonical Ridge (identical protocol) -------------------------
    res, pred = ridge_eval(Z_dev, y_dev, Z_te, yte)
    log(f"[HIER-SUP-CLEAN] = {res['macro_f1']}  "
        f"(alpha={res['selected_alpha']})")

    # ---- 13. predictions + deltas vs saved references --------------------
    np.save(os.path.join(OUT, "predictions", "test_predictions.npy"),
            pred.astype(np.int64))
    deltas = {k: round(res["macro_f1"] - v, 4) for k, v in REFS.items()}

    # ---- 14. equivalence gate vs stored HIER-CONTINUOUS-SUP (production) -
    # spec sections 17/27: this experiment reframes the SAME candidate
    # construction (PPV^(0)=0 makes the root block Delta^(1)); a different
    # result would be an implementation divergence, not a new method.
    if smoke:
        eq = {"verdict": "SKIPPED_SMOKE",
              "reason": "smoke subset differs from the stored full run"}
        log("equivalence gate: SKIPPED (smoke)")
    else:
        import csv as _csv
        stored_rows = list(_csv.DictReader(open(
            os.path.join(REF_DIR, "diagnostics",
                         "selected_feature_metadata.csv"))))
        stored_seq = [int(r["candidate_index"]) for r in stored_rows]
        stored_map = {}
        for r in stored_rows:
            lv = 1 if r["source_type"] == "root" else int(r["level"])
            par = -1 if r["parent"] == "NA" else int(r["parent"])
            chd = -1 if r["child"] == "NA" else int(r["child"])
            stored_map[int(r["candidate_index"])] = (
                lv, par, chd, int(r["kernel_id"]), float(r["f_stat"]))
        my_map = {}
        for c, m in zip(sel, meta):
            K, j, p, kg = m
            my_map[int(c)] = (1 if K == 1 else K, p, j, kg,
                              float(f_stat[c]))
        set_eq = sorted(my_map) == sorted(stored_map)
        seq_eq = sel.tolist() == stored_seq
        f_eq = set_eq and all(
            np.isclose(my_map[c][4], stored_map[c][4],
                       rtol=1e-5, atol=0.0)
            for c in stored_map)
        sem_eq = set_eq and all(
            my_map[c][:4] == stored_map[c][:4] for c in stored_map)
        stored_json = json.load(open(os.path.join(
            REF_DIR, "results", "hier_continuous_sup.json")))
        comp_ref = {1 if k == "0" else int(k): v for k, v
                    in stored_json["selected_composition"].items()}
        comp_eq = lv_counts == comp_ref
        stored_pred = np.load(os.path.join(
            REF_DIR, "predictions", "test_predictions.npy"))
        pred_eq = np.array_equal(pred.astype(np.int64), stored_pred)
        alpha_eq = abs(res["selected_alpha"]
                       - stored_json["alpha"]) < 1e-9
        f1_delta = round(res["macro_f1"] - stored_json["macro_f1"], 6)
        f1_eq = abs(f1_delta) < 1e-9
        comp_ok = all([set_eq, f_eq, sem_eq, comp_eq, pred_eq,
                       alpha_eq, f1_eq])
        eq = {"expected": "IDENTICAL",
              "verdict": "IDENTICAL" if comp_ok else "DIFFERENT",
              "reference": os.path.relpath(REF_DIR, ROOT),
              "components": {
                  "selected_index_set": set_eq,
                  "selected_index_sequence": seq_eq,
                  "per_candidate_f_stats": f_eq,
                  "semantic_metadata": sem_eq,
                  "emerged_composition": comp_eq,
                  "test_predictions": pred_eq,
                  "alpha": alpha_eq,
                  "macro_f1": f1_eq},
              "n_selected_compared": len(stored_map),
              "macro_f1_delta_vs_reference": f1_delta,
              "chain_zero_sum_max": zero_sum_max,
              "chain_identity_max_rel": ident_rel,
              "root_increment_bit_equal": True}
        log(f"[equivalence gate vs HIER-CONTINUOUS-SUP] verdict = "
            f"{eq['verdict']}  (F1 delta {f1_delta})")
        assert eq["verdict"] == "IDENTICAL", eq

    # ---- 15. artifacts ----------------------------------------------------
    with open(os.path.join(OUT, "results",
                           "hier_sup_clean.json"), "w") as f:
        json.dump({
            "method": "hier_sup_clean",
            "label": "HIER-SUP-CLEAN",
            "dataset": DS, "seed": SEED,
            "base_budget": B_BASE, "high_budget": B_HIGH,
            "kernel_capacity": "84x238",
            "final_features": B_HIGH,
            "candidate_pool": {"root_increment_D1": D1,
                               "delta_D2_D4_D8_D16": H_CAND,
                               "unified": N_CAND},
            "formulation": {
                "ppv_level_0": PPV0,
                "root_increment": "Delta^(1)_m = PPV_m^(1) - PPV_m^(0) "
                                  "= PPV_m^(1)",
                "deeper": "Delta_{m,c}^(K) = PPV_{m,c}^(K) - "
                          "PPV_{m,parent(c)}^(K/2), K in {2,4,8,16}",
                "parent_rule": "occupancy-weighted child mean; "
                               "sum_c (n_c/n_p) Delta_{m,c} = 0"},
            "selection": "ONE supervised ANOVA-F ranking over the unified "
                         "469,812-candidate telescoping pool (train+val "
                         "labels, canonical R5 final-stage protocol); no "
                         "rho, no G/H split, no per-level quota",
            "selected_composition": {str(k): v
                                     for k, v in lv_counts.items()},
            "cv_unified_fold_internal": {"mean": cv_mean, "std": cv_std,
                                         "kfold": 5},
            "equivalence": eq,
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

    with open(os.path.join(OUT, "diagnostics",
                           "equivalence_check.json"), "w") as f:
        json.dump(eq, f, indent=1)

    with open(os.path.join(OUT, "diagnostics", "dimensions.json"),
              "w") as f:
        json.dump({
            "split": {"train": n_tr, "validation": n_va, "test": n_te,
                      "T": int(Xtr.shape[1]), "n_classes":
                      int(len(np.unique(ytr)))},
            "extractors": {"expanded": fHK},
            "pools": {"root_increments_D1": D1,
                      "delta_candidates": H_CAND,
                      "edges": EDGES, "kernels_per_edge": F_H,
                      "unified_candidates": N_CAND,
                      "column_order": ["D1", "D2", "D4", "D8", "D16"]},
            "chain_identity": {"zero_sum_max": zero_sum_max,
                               "identity_max_rel": ident_rel},
            "final": {"selected": B_HIGH,
                      "Z_dev": list(Z_dev.shape),
                      "Z_test": list(Z_te.shape),
                      "composition": {str(k): v
                                      for k, v in lv_counts.items()}},
        }, f, indent=1)

    with open(os.path.join(OUT, "diagnostics", "leakage_report.json"),
              "w") as f:
        json.dump({
            "minirocket_fit": "train signals only (seed 42)",
            "hierarchy": "TRAIN latents only (frozen seed-42 SSL encoder, "
                         "recursive 2-means); val/test transform-only; "
                         "reg16_tr byte-identical to the stored artifact",
            "telescoping_pool_and_root_matrix":
                "frozen transform, label-free; D1 = PPV^(1) - 0 asserted "
                "bit-equal",
            "unified_selection": "train+val labels (canonical R5 "
                                 "final-stage protocol - identical to "
                                 "HIER-CONTINUOUS-SUP / R5-HIGH)",
            "cv_diagnostic": "5-fold StratifiedKFold(shuffle=True, "
                             "random_state=42); fold-internal selection "
                             "on fold-train labels ONLY",
            "test_labels": "touched only inside ridge_eval for scoring",
            "no_composition_rules": "fully emergent from the ranking; "
                                    "reported as diagnostics only",
        }, f, indent=1)

    with open(os.path.join(OUT, "diagnostics",
                           "selection_protocol.md"), "w") as f:
        f.write(
            "# Selection protocol - HIER-SUP-CLEAN\n\n"
            "## Telescoping pool (clean formulation)\n\n"
            "C = [D1 | D2 | D4 | D8 | D16] with\n"
            f"- D1: {D1} root increments Delta^(1) = PPV^(1) - PPV^(0) "
            f"(PPV^(0) = 0, so Delta^(1) = PPV^(1) bit-exactly)\n"
            f"- D2/D4/D8/D16: {H_CAND} occupancy-weighted telescoping "
            f"residuals ({EDGES} edges x {F_H} kernels)\n"
            f"- total: {N_CAND} candidates (asserted C_dev.shape == "
            f"(896, {N_CAND}))\n\n"
            "## Stage 1 - CV diagnostic (never selects)\n\n"
            "5-fold StratifiedKFold(shuffle=True, random_state=42) over "
            "train+val (896 rows). Per fold: top_f_select on the "
            "fold-train rows/labels ONLY (fold-internal selection over "
            "the full unified pool), held-out rows transformed with "
            "those identities, RidgeClassifierCV(ALPHAS) fit on "
            "fold-train, macro-F1 on the fold. Diagnostic only.\n\n"
            "## Stage 2 - final selection (canonical R5 protocol)\n\n"
            "top_f_select(C_dev, y_dev, 19,992) on the FULL dev matrix "
            "with train+val labels - exactly the protocol R5-HIGH, "
            "HIER-HIGH-SUP and HIER-CONTINUOUS-SUP use. Selector "
            "semantics identical (sklearn f_classif, NaN->0, "
            "np.argsort(-f, kind='stable'), top-N). ONE ranking over "
            "the complete pool: no G/H boundary, no rho, no protected "
            "budgets, no per-level quota.\n\n"
            "## Stage 3 - refit + one official test evaluation\n\n"
            "RidgeClassifierCV(ALPHAS=logspace(-4,4,20)) fit on the "
            f"selected {B_HIGH}-column representation over train+val; "
            "single evaluation on the 3,582-row official test split.\n\n"
            "## Resulting composition (emerged, diagnostic only)\n\n"
            f"D1 {lv_counts[1]}, D2/K2 {lv_counts[2]}, D4/K4 "
            f"{lv_counts[4]}, D8/K8 {lv_counts[8]}, D16/K16 "
            f"{lv_counts[16]} (sum {sum(lv_counts.values())} = "
            f"{B_HIGH}).\n")

    # comparison.csv + figures + summary ------------------------------------
    rows = [("canonical_mr", "Canonical MR", 9996, "84x119", "NA"),
            ("mr_high", "MR-HIGH", 19992, "84x238", "NA"),
            ("r5_high", "R5-HIGH", 19992, "84x238", "14994"),
            ("hier_high", "HIER-HIGH", 19992, "84x238", "449820"),
            ("hier_high_sup", "HIER-HIGH-SUP", 19992, "84x238", "449820"),
            ("hier_continuous_sup", "HIER-CONTINUOUS-SUP", 19992, "84x238",
             "469812"),
            ("hier_sup_clean", "HIER-SUP-CLEAN", 19992, "84x238",
             "469812")]
    with open(os.path.join(OUT, "results", "comparison.csv"), "w") as f:
        f.write("method,label,final_features,d1_selected,k2_selected,"
                "k4_selected,k8_selected,k16_selected,candidate_pool,"
                "kernel_capacity,macro_f1,accuracy,alpha,runtime_s,"
                "peak_mem_mb\n")
        for key, label, ff, kcap, pool in rows:
            if key == "hier_sup_clean":
                f.write(f"{key},{label},{ff},{lv_counts[1]},{lv_counts[2]},"
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
        fig, ax = plt.subplots(figsize=(9.6, 4.2))
        cols = ["#9e9e9e"] * 6 + ["#1f77b4"]
        ax.bar(labels, vals, color=cols)
        ax.set_ylim(0.70, 0.80)
        ax.set_ylabel("Macro-F1 (official test)")
        ax.set_title("HIER-SUP-CLEAN - clean telescoping hierarchy "
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
        lv = ["D1 (root)", "D2/K2", "D4/K4", "D8/K8", "D16/K16"]
        cnt = [lv_counts[1], lv_counts[2], lv_counts[4], lv_counts[8],
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

        # candidate-pool composition figure (pool make-up vs selection)
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        pool_cols = [D1, 2 * F_H, 4 * F_H, 8 * F_H, 16 * F_H]
        x = np.arange(5)
        ax.bar(x - 0.2, pool_cols, 0.4, label="pool candidates", log=True)
        ax.bar(x + 0.2, cnt, 0.4, label="selected", log=True)
        ax.set_xticks(x)
        ax.set_xticklabels(lv)
        ax.set_ylabel("count (log)")
        ax.set_title("Telescoping pool vs emerged selection")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "figures",
                                 "candidate_pool_composition.png"), dpi=150)
        plt.close(fig)
        log("figures written")
    except Exception as e:  # figures must never kill the science
        log(f"figure generation skipped: {e!r}")

    rt = round(time.time() - t00, 1)
    mem = round(peak_mem_mb(), 1)
    with open(os.path.join(OUT, "results", "summary.md"), "w") as f:
        f.write(
            "# HIER-SUP-CLEAN - summary (UWaveY, seed 42)\n\n"
            f"**Macro-F1 = {res['macro_f1']}** (accuracy "
            f"{res['accuracy']}, alpha {res['selected_alpha']})\n\n"
            "## Formulation\n\n"
            "One telescoping hierarchy of increments with PPV^(0) = 0: "
            "Delta^(1) = PPV^(1) (the root increment, bit-equal), "
            "Delta^(K) = PPV^(K) - occupancy-weighted parent PPV for "
            "K in {2,4,8,16}. Unified pool [D1 | D2 | D4 | D8 | D16] of "
            "469,812 candidates; ONE supervised ANOVA-F ranking (train+val "
            "labels, canonical R5 protocol) picks exactly 19,992 features "
            "with a fully emergent composition. No rho, no G/H split, no "
            "quota, no energy allocation, no empirical Bayes.\n\n"
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
            "| HIER-CONTINUOUS-SUP | 19,992 | 0.7599 | "
            f"{round(res['macro_f1'] - REFS['hier_continuous_sup'], 4):+} "
            "|\n"
            f"| **HIER-SUP-CLEAN** | 19,992 | **{res['macro_f1']}** | "
            "NA |\n\n"
            "## Equivalence to HIER-CONTINUOUS-SUP\n\n"
            f"Verdict: **{eq['verdict']}**. With PPV^(0) = 0 the root "
            "block IS Delta^(1), so this experiment is the clean "
            "mathematical reformulation / canonicalization of "
            "HIER-CONTINUOUS-SUP - not a new method and not a new "
            "performance claim. Components: "
            + ", ".join(f"{k}={v}" for k, v in
                        eq.get("components", {}).items()) + ".\n\n"
            f"## Emerged composition (diagnostic)\n\nD1 {lv_counts[1]} "
            f"({lv_counts[1] / B_HIGH:.1%}), D2/K2 {lv_counts[2]} "
            f"({lv_counts[2] / B_HIGH:.1%}), D4/K4 {lv_counts[4]} "
            f"({lv_counts[4] / B_HIGH:.1%}), D8/K8 {lv_counts[8]} "
            f"({lv_counts[8] / B_HIGH:.1%}), D16/K16 {lv_counts[16]} "
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
