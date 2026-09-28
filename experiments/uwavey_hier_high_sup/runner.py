"""HIER-HIGH-SUP — one-change controlled ablation on UWaveGestureLibraryY.

ONLY change vs HIER-HIGH (capacity-scaled experiment):
    hierarchical candidate pool (449,820)
        -> label-free energy allocation        (ABLATED)
        -> fold-internal supervised ANOVA-F, top-1,999   (NEW)

Everything else locked: same expanded MiniROCKET (n_kernels=19,992, seed 42),
same frozen K=1->2->4->8->16 hierarchy, same candidate-pool definition,
same G_high = first 17,993 (R5-HIGH convention, rho=0.1), same canonical
RidgeClassifierCV protocol.  Final budget exactly 19,992 for every method.
"""

import ast
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
RHO = 0.1
N_H = int(round(RHO * B_HIGH))          # 1,999
N_G = B_HIGH - N_H                      # 17,993
H_KERNELS = B_HIGH - G_PART             # 14,994
KEPT_K = [2, 4, 8, 16]
H_CAND = sum(KEPT_K) * H_KERNELS        # 449,820
ALPHAS = np.logspace(-4, 4, 20)

CAP_RESULTS = os.path.join(ROOT, "results", "uwavey_capacity_scaled", "seed42")
NEST_RESULTS = os.path.join(ROOT, "results", "uwavey_nested_hierarchical", "seed42")
REF_MR_HIGH = 0.7477
GATE_TOL = 0.002

OUT = os.path.join(ROOT, "results", "uwavey_hier_high_sup", "seed42")

# reused canonical pieces (imported, not reimplemented)
from experiments.uwavey_nested_hierarchical.runner import (   # noqa: E402
    _encoder_latents, compute_activations, load_context_model, load_data,
    ppv_all, ridge_eval, znorm)
from experiments.uwavey_nested_hierarchical.runner import delta_banks  # noqa: E402,E501
from experiments.uwavey_capacity_scaled.runner import (       # noqa: E402
    extractor_facts, fit_minirocket_nk, hier_banks_test, peak_mem_mb,
    r5_cv_fixed_rho, top_f_select)
from models.nested_regimes.model import (                     # noqa: E402
    assign_levels, build_latent_tree, nesting_errors)

CAP_RUNNER = os.path.join(ROOT, "experiments", "uwavey_capacity_scaled",
                          "runner.py")


def log(msg):
    print(msg, flush=True)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def col_meta(c, f_h):
    """(level K, child j, parent p, global kernel id) for full-pool column c.

    Layout (delta_banks, delta_cols=None, kept_K coarse->fine):
    K=2 block (children 0..1), K=4 (0..3), K=8 (0..7), K=16 (0..15);
    within a block: child-major, kernel-minor.
    """
    for K, cbefore in ((2, 0), (4, 2), (8, 6), (16, 14)):
        s = cbefore * f_h
        if c < s + K * f_h:
            j = (c - s) // f_h
            k = (c - s) % f_h
            return K, int(j), int(j >> 1), G_PART + int(k)
    raise IndexError(f"column {c} outside pool of {30 * f_h}")


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
    n_tr, n_va = len(ytr), len(yva)
    y_dev = np.concatenate([ytr, yva]).astype(np.int64)
    if smoke:
        Xtr, ytr = Xtr[:200], ytr[:200]
        Xva, yva = Xva[:60], yva[:60]
        Xte, yte = Xte[:200], yte[:200]
        n_tr, n_va = len(ytr), len(yva)
        y_dev = np.concatenate([ytr, yva]).astype(np.int64)
    log(f"data: train {Xtr.shape}, val {Xva.shape}, test {Xte.shape}")

    ntr_ref = np.load(os.path.join(NEST_RESULTS, "train_indices.npy"))
    nva_ref = np.load(os.path.join(NEST_RESULTS, "val_indices.npy"))
    nte_ref = np.load(os.path.join(NEST_RESULTS, "test_indices.npy"))
    split_ok = (len(ntr_ref) == 761 and len(nva_ref) == 135 and
                len(nte_ref) == 3582)
    log(f"split identity vs stored uwavey_nested indices: {split_ok} "
        f"(sizes {len(ntr_ref)}/{len(nva_ref)}/{len(nte_ref)})")
    assert split_ok or smoke

    # ---- 2. frozen hierarchy (LOCKED; rebuilt deterministically) --------
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
                             C)[4].reshape(len(yte), -1).astype(np.int64)
    del lat_te
    log("tree: levels 1,2,4,8,16 frozen (TRAIN latents only); val/test "
        "transform-only")

    # ---- 3. expanded MiniROCKET (LOCKED) --------------------------------
    exHK = fit_minirocket_nk(Xtr_z, B_HIGH)
    fHK = extractor_facts(exHK)
    F_H = fHK["total_features"] - G_PART
    assert fHK["physical_kernels"] == 84
    if not smoke:
        stored_dims = json.load(open(os.path.join(
            CAP_RESULTS, "diagnostics", "dimensions.json")))
        assert fHK == stored_dims["extractors"]["expanded"], (fHK,)
        assert F_H == H_KERNELS and 30 * F_H == H_CAND
        log(f"expanded extractor facts match stored: {fHK}")
    else:
        log(f"expanded extractor facts (smoke): {fHK}")

    # ---- 4. G banks (full width, chunked) --------------------------------
    GHK_trva = ppv_all(exHK, np.vstack([Xtr_z, Xva_z]))
    GHK_te = ppv_all(exHK, Xte_z)
    assert GHK_trva.shape == (n_tr + n_va, B_HIGH)
    assert GHK_te.shape == (len(yte), B_HIGH)
    log(f"G banks: {GHK_trva.shape}, {GHK_te.shape}")

    # ---- 5. MR-HIGH reproduction gate (regression assertion) ------------
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
        act, valid = compute_activations(exHK,
                                         np.vstack([Xtr_z, Xva_z])[c0:c1])
        pool_parts.append(delta_banks(act[:, G_PART:], valid[G_PART:],
                                      np.vstack([reg16_tr, reg16_va])[c0:c1],
                                      KEPT_K, delta_cols=None,
                                      sample_chunk=96))
        del act
    Hcand_trva = np.vstack(pool_parts)
    del pool_parts
    assert Hcand_trva.shape == (n_tr + n_va, H_CAND), Hcand_trva.shape
    log(f"hierarchical candidate pool: {Hcand_trva.shape} "
        f"({30} edges x {F_H} kernels)")

    # ---- 7. CV diagnostic (fold-internal selection, reported only) -------
    cv_mean, cv_std = r5_cv_fixed_rho(GHK_trva, Hcand_trva, y_dev,
                                      N_G, N_H)
    log(f"[CV fixed-rho fold-internal] macro-F1 = {cv_mean:.4f} "
        f"+/- {cv_std:.4f}  (diagnostic only)")

    # ---- 8. THE ONE CHANGE: supervised top-1,999 of the full pool --------
    from sklearn.feature_selection import f_classif
    f_stat, _ = f_classif(Hcand_trva, y_dev)
    f_stat = np.nan_to_num(f_stat, nan=0.0)
    top = np.argsort(-f_stat, kind="stable")[:N_H]
    assert np.array_equal(top, top_f_select(Hcand_trva, y_dev, N_H))
    # group selected columns by level (rank order preserved within level)
    # so the train/val gather and the frozen-transform test gather share
    # one column order (Ridge is permutation-invariant in features)
    lvl_rank = {2: 0, 4: 1, 8: 2, 16: 3}
    lvl_of = np.array([lvl_rank[col_meta(int(c), F_H)[0]] for c in top])
    top = top[np.argsort(lvl_of, kind="stable")]
    log(f"supervised selection: top {N_H} of {Hcand_trva.shape[1]} "
        f"(f-classif, stable argsort)")

    # ---- 9. selected-feature metadata (diagnostic only) ------------------
    meta = [col_meta(int(c), F_H) for c in top]
    lv_counts = {K: sum(1 for m in meta if m[0] == K) for K in KEPT_K}
    assert sum(lv_counts.values()) == N_H
    log(f"selected by level (diagnostic): {lv_counts}")

    # frozen-transform test gather: the chain consumes per-level
    # (kernel, child) ids in the SAME row order as the columns below
    sel_dict = {K: np.array([(kg - G_PART, j) for (Kk, j, p, kg) in meta
                             if Kk == K], dtype=np.int64)
                for K in KEPT_K}
    assert all(len(v) == lv_counts[K] for K, v in sel_dict.items())

    with open(os.path.join(OUT, "diagnostics",
                           "selected_feature_counts_by_level.csv"), "w") as f:
        f.write("level,edges_in_level,kernels_per_edge,pool_columns,"
                "selected,share_of_H,share_of_pool\n")
        for K, cbefore in ((2, 0), (4, 2), (8, 6), (16, 14)):
            pool_k = K * F_H
            sel_k = lv_counts[K]
            f.write(f"{K},{K},{F_H},{pool_k},{sel_k},"
                    f"{sel_k / N_H:.6f},{sel_k / H_CAND:.6f}\n")

    with open(os.path.join(OUT, "diagnostics",
                           "selected_feature_metadata.csv"), "w") as f:
        f.write("candidate_index,level,child,parent,kernel_global,"
                "kernel_h_slice,f_stat\n")
        for rank, (c, m) in enumerate(zip(top, meta)):
            K, j, p, kg = m
            f.write(f"{rank},{K},{j},{p},{kg},{kg - G_PART},"
                    f"{f_stat[c]:.6g}\n")

    # ---- 10. test bank for the SELECTED columns only ---------------------
    Hcand_te = hier_banks_test(exHK, Xte_z, reg16_te, KEPT_K, sel_dict)
    assert Hcand_te.shape == (len(yte), N_H), Hcand_te.shape
    if not smoke:
        assert Hcand_trva.shape[1] == 449820

    # ---- 11. final representation + canonical Ridge ----------------------
    Z_dev = np.hstack([GHK_trva[:, :N_G], Hcand_trva[:, top]])
    Z_te = np.hstack([GHK_te[:, :N_G], Hcand_te])
    assert Z_dev.shape == (n_tr + n_va, B_HIGH)
    assert Z_te.shape == (len(yte), B_HIGH)
    res, pred = ridge_eval(Z_dev, y_dev, Z_te, yte)
    log(f"[HIER-HIGH-SUP] = {res['macro_f1']}  (alpha={res['selected_alpha']})")

    # ---- 12. saved references (NOT rerun) --------------------------------
    refs = {}
    for mkey in ("canonical_mr", "mr_high", "r5_high", "hier_high"):
        refs[mkey] = json.load(open(os.path.join(
            CAP_RESULTS, "results", f"{mkey}.json")))
    deltas = {k: round(res["macro_f1"] - refs[k]["macro_f1"], 4)
              for k in refs}

    # ---- 13. artifacts ----------------------------------------------------
    with open(os.path.join(OUT, "results", "hier_high_sup.json"), "w") as f:
        json.dump({
            "method": "hier_high_sup",
            "label": "HIER-HIGH-SUP",
            "dataset": DS, "seed": SEED,
            "base_budget": B_BASE, "high_budget": B_HIGH,
            "kernel_capacity": "84x238",
            "g_features": N_G, "h_features": N_H,
            "h_candidate_pool": H_CAND,
            "final_features": B_HIGH,
            "selection": "fold-internal supervised ANOVA-F top-1999 "
                         "(final stage on train+val per canonical R5 protocol)",
            "ablated": "label-free energy allocation "
                       "(level_budgets/select_carriers, K2=4437/K4=3767/"
                       "K8=3128/K16=3662)",
            "cv_fixed_rho_fold_internal": {
                "mean": cv_mean, "std": cv_std, "kfold": 5},
            "selected_by_level": lv_counts,
            "macro_f1": res["macro_f1"], "accuracy": res["accuracy"],
            "alpha": res["selected_alpha"],
            "deltas_vs": deltas,
            "gate_mr_high": {"ref": REF_MR_HIGH,
                             "reproduced": mr_gate["macro_f1"],
                             "abs_diff": gate_diff},
            "runtime_s": round(time.time() - t00, 1),
            "peak_mem_mb": peak_mem_mb(),
        }, f, indent=1)
    np.save(os.path.join(OUT, "predictions", "test_predictions.npy"),
            pred.astype(np.int64))

    rows = [("canonical_mr", refs["canonical_mr"]),
            ("mr_high", refs["mr_high"]),
            ("r5_high", refs["r5_high"]),
            ("hier_high", refs["hier_high"]),
            ("hier_high_sup",
             {"kernel_capacity": "84x238", "final_features": B_HIGH,
              "candidate_features": H_CAND,
              "alpha": res["selected_alpha"], "macro_f1": res["macro_f1"],
              "accuracy": res["accuracy"]})]
    with open(os.path.join(OUT, "results", "comparison.csv"), "w") as f:
        f.write("method,base_budget,high_budget,kernel_capacity,"
                "candidate_features,final_features,g_features,h_features,"
                "h_candidate_pool,alpha,macro_f1,accuracy,num_classes,seed\n")
        for key, r in rows:
            alpha = r.get("selected_alpha", r.get("alpha"))
            if key == "hier_high_sup":
                g, h, hc = N_G, N_H, H_CAND
            elif key in ("canonical_mr", "mr_high"):
                g, h, hc = r["final_features"], 0, r["final_features"]
            elif key == "r5_high":
                g, h, hc = N_G, N_H, r["candidate_features"]
            else:  # hier_high
                g, h, hc = G_PART, B_HIGH - G_PART, r["candidate_features"]
            f.write(f"{key},{B_BASE},{r['final_features']},"
                    f"{r['kernel_capacity']},{hc},{r['final_features']},"
                    f"{g},{h},{hc},{alpha},{r['macro_f1']},"
                    f"{r['accuracy']},8,{SEED}\n")

    save_json({
        "B_base": B_BASE, "B_high": B_HIGH, "G_part": G_PART,
        "N_G": N_G, "N_H": N_H, "rho": RHO,
        "H_kernels": F_H, "edges": 30, "H_candidate_pool": H_CAND,
        "final_dims": {"Z_dev": list(Z_dev.shape), "Z_te": list(Z_te.shape)},
        "all_methods_final_features": B_HIGH,
        "extractor_expanded": fHK,
        "selected_by_level": lv_counts,
        "runtime_s": round(time.time() - t00, 1),
        "peak_mem_mb": peak_mem_mb(),
        "smoke": smoke,
    }, os.path.join(OUT, "diagnostics", "dimensions.json"))
    save_json({
        "dataset": DS, "seed": SEED,
        "split_identity_vs_stored_indices": bool(split_ok),
        "split_sizes": {"train": int(n_tr), "val": int(n_va),
                        "test": int(len(yte))},
        "minirocket_fit": "train-only, random_state=42, n_kernels=19992",
        "hierarchy": "frozen tree rebuilt deterministically from TRAIN "
                     "latents only; regression-equal to stored "
                     "hierarchy_assignments.npy",
        "val_test_regimes": "assign_levels transform-only (no refit)",
        "final_selection_data": "train+val rows (y_dev) -- the canonical "
                                "R5-HIGH protocol; CV diagnostic uses "
                                "fold-train labels only",
        "test_labels": "accessed only inside ridge_eval for scoring",
        "poisoning_tests": "tests/test_uwavey_hier_high_sup.py "
                           "(val/test signal and label poisoning)",
    }, os.path.join(OUT, "diagnostics", "leakage_report.json"))

    _figures(res, refs)
    _level_figure(lv_counts, F_H)
    _summary(rows, lv_counts, deltas, cv_mean, cv_std, res, gate_diff, t00)
    _selection_protocol()
    log(f"done in {time.time() - t00:.0f}s -> {OUT}")


def _figures(res, refs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["Canonical MR", "MR-HIGH", "R5-HIGH", "HIER-HIGH",
              "HIER-HIGH-SUP"]
    vals = [refs["canonical_mr"]["macro_f1"], refs["mr_high"]["macro_f1"],
            refs["r5_high"]["macro_f1"], refs["hier_high"]["macro_f1"],
            res["macro_f1"]]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(labels, vals,
                  color=["#9aa5b1", "#9aa5b1", "#4c78a8", "#9aa5b1",
                         "#e45756"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.4f}",
                ha="center", fontsize=9)
    ax.set_ylim(0.6, 0.85)
    ax.set_ylabel("Test Macro-F1 (UWaveY, seed 42)")
    ax.set_title("Capacity-fixed comparison (all high methods = 19,992)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figures", "comparison_macro_f1.png"),
                dpi=150)
    plt.close(fig)


def _level_figure(lv_counts, f_h):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ks = [2, 4, 8, 16]
    pool = [K * f_h for K in ks]
    sel = [lv_counts[K] for K in ks]
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.bar(x - 0.2, pool, 0.4, label="pool columns", color="#c7cdd4")
    ax.bar(x + 0.2, sel, 0.4, label="selected (supervised)", color="#e45756")
    for i, (p, s) in enumerate(zip(pool, sel)):
        ax.text(i - 0.2, p, f"{p:,}", ha="center", va="bottom", fontsize=8)
        ax.text(i + 0.2, s, f"{s:,}", ha="center", va="bottom", fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(x, [f"K={K}" for K in ks])
    ax.set_ylabel("features (log)")
    ax.set_title("HIER-HIGH-SUP: supervised selections by hierarchy level")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figures",
                             "selected_features_by_level.png"), dpi=150)
    plt.close(fig)


def _summary(rows, lv_counts, deltas, cv_mean, cv_std, res, gate_diff, t00):
    lines = [
        "# HIER-HIGH-SUP — one-change ablation (UWaveY, seed 42)",
        "",
        "Only change vs HIER-HIGH: hierarchical candidate pool (449,820) is",
        "ranked by **supervised ANOVA-F** (top 1,999) instead of the",
        "label-free energy allocator. Hierarchy, expanded MiniROCKET,",
        "G_high (17,993, R5-HIGH convention), candidate definition and the",
        "canonical Ridge protocol are locked and asserted identical.",
        "",
        "| method | final feats | Macro-F1 |",
        "|---|---|---|",
    ]
    for key, r in rows:
        lines.append(f"| {key} | {r['final_features']:,} | "
                     f"{r['macro_f1']:.4f} |")
    lines += [
        "",
        f"Deltas of HIER-HIGH-SUP: " +
        ", ".join(f"{k} {v:+.4f}" for k, v in deltas.items()),
        "",
        f"CV (fold-internal selection, diagnostic): {cv_mean:.4f} +/- "
        f"{cv_std:.4f}",
        f"Selected per level (diagnostic only): "
        f"{ {k: v for k, v in lv_counts.items()} }",
        f"MR-HIGH reproduction gate diff: {gate_diff:.4f} (tol 0.002)",
        f"Runtime: {time.time() - t00:.0f}s",
    ]
    with open(os.path.join(OUT, "results", "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def _selection_protocol():
    txt = """# Selection protocol (audited from the existing R5-HIGH code)

Selector: `experiments/uwavey_capacity_scaled/runner.py::top_f_select` --
sklearn `f_classif` -> `np.nan_to_num(nan=0.0)` -> `np.argsort(-f_stat,
kind="stable")` -> top-N.  Deterministic; identical implementation, tie
handling, NaN/constant-feature handling and top-N semantics as R5-HIGH.

## CV diagnostic (reported, never used for selection)

`r5_cv_fixed_rho(G_full, H_full, y_dev, n_g=17,993, n_h=1,999)`:
5-fold `StratifiedKFold(shuffle=True, random_state=42)`; for EVERY fold:
1. fold-train rows/labels only -> `f_classif` -> top 1,999 hierarchical
   candidates (fold-internal selection),
2. held-out fold rows transformed with THOSE selected feature identities,
3. `RidgeClassifierCV(ALPHAS)` fit on the fold-train, macro-F1 on fold-val.

## Final stage (the canonical R5-HIGH protocol, replicated verbatim)

`top = top_f_select(H_dev, y_dev, 1,999)` on the FULL dev matrix
(train+val = 896 rows) -- i.e. the canonical protocol DOES use train+val
labels for the final selection, exactly as the existing R5-HIGH did
(cap-runner L388-396).  Then `ridge_eval([G_dev[:, :17993] | H_dev[:, top]],
y_dev, [G_te[:, :17993] | H_te[:, top]], yte)`: RidgeClassifierCV
(alphas=np.logspace(-4,4,20)) fit on train+val, ONE official test evaluation.

## Data/label flow

- hierarchy: TRAIN latents only; val/test transform-only.
- candidate pool: frozen transform, label-free.
- final selection: train+val labels (canonical R5 protocol).
- CV diagnostic: fold-train labels only.
- test labels: touched only inside `ridge_eval` for scoring.

## Pool notes (locked definition)

- 30 edges x 14,994 H-side kernels = 449,820 candidates; column order
  K=2,4,8,16 coarse->fine, children 0..K-1 within level, kernels within.
- At K=2 the two child deltas are exact anti-correlated rescalings (zero-sum
  with two children); their F-stats are equal up to float rounding, so both
  columns can enter the top-1,999.  The previous ENERGY allocator deduped
  K=2 child-1; that dedup belongs to the ablated allocator and is NOT used.
- No level quotas, no energy, no null, no largest-remainder budgeting:
  the selector ranks the entire pool and the observed per-level counts are
  diagnostic only.
"""
    with open(os.path.join(OUT, "diagnostics",
                           "selection_protocol.md"), "w") as f:
        f.write(txt)


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
