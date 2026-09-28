"""HIERARCHICAL / NESTED REGIME-CONDITIONED HETEROGENEITY — Haptics seed 42.

Pipeline (train-only until the frozen-depth demonstration):
  1. canonical raw split; per-sample z-normalization (project convention)
  2. frozen RCMKN seed-42 context model -> per-timestep SSLTemporalEncoder
     latents z in R^32 for TRAIN signals only
  3. nested binary tree by recursive 2-means on pooled train latents
     (K = 1 -> 2 -> 4 -> 8 -> 16; binary path ids; never refit)
  4. MiniRocket(random_state=42) fit on TRAIN z-normed signals; per-
     timestep activation indicators via the shared transfer core
  5. train regime sequences (frozen tree) -> exact nested-ANOVA chain:
     per-level energies, identity / zero-sum / cross-level orthogonality
  6. label-free shuffled-regime null (S=500, seed 52042)
  7. predeclared stopping rule -> L*, FROZEN before any classifier
  8. downstream demonstration: gates A/B + [G || H_hier(L*)] (frozen
     tree applied transform-only to test signals), single evaluation
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from models.nested_regimes.model import (  # noqa: E402
    LEVELS, NULL_SEED, SEED, S_PERM, assign_levels, build_latent_tree,
    hierarchy_chain, nesting_errors, save_json, shuffle_null, stopping_rule)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "haptics_nested_regimes", "seed42")
CACHE = "C:/temp/results"
GATE_A = 0.5037
GATE_B = 0.5500


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# data / frozen models
# ---------------------------------------------------------------------------
def load_raw():
    from experiments.external_stack_generalization.data import load_dataset
    d = load_dataset("Haptics")
    Xtr = np.asarray(d["Xtr"], dtype=np.float64)
    Xva = np.asarray(d["Xva"], dtype=np.float64)
    Xte = np.asarray(d["Xte"], dtype=np.float64)
    assert Xtr.shape == (132, 1092) and Xva.shape == (23, 1092) \
        and Xte.shape == (308, 1092), (Xtr.shape, Xva.shape, Xte.shape)
    ytr = np.asarray(d["ytr"])
    yva = np.asarray(d["yva"])
    yte = np.asarray(d["yte"])
    return d, Xtr, Xva, Xte, ytr, yva, yte


def znorm(X):
    """Canonical per-sample z-normalization (project convention)."""
    mu = X.mean(axis=-1, keepdims=True)
    sig = X.std(axis=-1, keepdims=True) + 1e-8
    return ((X - mu) / sig).astype(np.float32)


def load_context_model(device):
    """Rebuild the RCMKN context model exactly and load the frozen
    seed-42 checkpoint (eval mode, params frozen)."""
    import torch
    from experiments.rcmkn_haptics_seed42.model import RCMKNContextModel
    path = os.path.join(ROOT, "results", "rcmkn_haptics_seed42",
                        "context_model_seed42.pt")
    if not os.path.exists(path):
        path = os.path.join(CACHE, "context_model_seed42.pt")
    ck = torch.load(path, map_location=device, weights_only=False)
    model = RCMKNContextModel(n_classes=5)
    model.load_state_dict(ck["model_state"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ck


def _encoder_latents(model, X_z, device, batch=32):
    """Frozen encoder -> latents (N, T, 32) float64 (no grad)."""
    import torch
    out = []
    with torch.no_grad():
        for c0 in range(0, len(X_z), batch):
            xb = torch.from_numpy(X_z[c0:c0 + batch])[:, None, :].to(device)
            z = model.encoder(xb)                             # (B, T, 32)
            out.append(z.cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float64)


def fit_minirocket(Xtr_z):
    """Canonical MiniROCKET fit (train z-normed only)."""
    from aeon.transformations.collection.convolution_based import MiniRocket
    extractor = MiniRocket(random_state=SEED, n_jobs=-1)
    extractor.fit(Xtr_z[:, None, :].astype(np.float32))
    return extractor


def compute_activations(extractor, X_z):
    from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (
        compute_raw_activations)
    return compute_raw_activations(extractor, X_z)


# ---------------------------------------------------------------------------
def main(smoke=False):
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "predictions"), exist_ok=True)
    t00 = time.time()
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device: {device}")

    # ---- 1. canonical data --------------------------------------------------
    d, Xtr, Xva, Xte, ytr, yva, yte = load_raw()
    n_tr = len(ytr)
    log(f"data: train {Xtr.shape}, val {Xva.shape}, test {Xte.shape}")

    # ---- 2. frozen context model + TRAIN latents ----------------------------
    model, ck = load_context_model(device)
    Xtr_z = znorm(Xtr)
    lat = _encoder_latents(model, Xtr_z, device)          # (n, T, 32)
    lat_pool = lat.reshape(-1, lat.shape[-1])
    log(f"train latents: {lat_pool.shape}")

    # ---- 3. nested tree (train only) ----------------------------------------
    C, tree_meta = build_latent_tree(lat_pool, seed=SEED)
    log("tree levels: " + ", ".join(str(len(c)) for c in C))

    # ---- 4. activations (train) ---------------------------------------------
    # Canonical pipeline: MiniRocket(z-normed train fit) has 9996 features;
    # G-bank = first 4998 (kernel PPVs), H-bank kernels = features 4998:9996.
    # The hierarchy chain partitions the H-bank kernel activations.
    extractor = fit_minirocket(Xtr_z)
    act_tr_full, valid_full = compute_activations(extractor, Xtr_z)
    assert act_tr_full.shape[1] == 9996, act_tr_full.shape
    F_H = 4998                       # kernel units in the H bank
    act_tr = act_tr_full[:, 4998:, :]                # (n, 4998, T)
    valid_het = valid_full[4998:]
    log(f"activations: {act_tr.shape} (H-bank kernels, features 4998:9996)")

    # extractor sanity: first-half PPVs vs the frozen G bank
    G_trva = np.load(os.path.join(CACHE,
                                  "haptics_inference_banks_G_trva.npy"))
    from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (
        ppv_from_activations)
    ppv_first = ppv_from_activations(act_tr_full[:, :4998],
                                     valid_full[:4998]).astype(np.float64)
    mr_match = float(np.abs(ppv_first - G_trva[:n_tr].astype(np.float64)).max())
    log(f"extractor-vs-frozen-G agreement: max|diff| = {mr_match:.2e} "
        f"(small aeon-version quantile drift; per-run recompute is the "
        f"repository convention)")
    assert mr_match < 5e-3, mr_match

    # ---- 5. train regime sequences + exact chain ----------------------------
    levels_tr = assign_levels(lat_pool, C)                # frozen transform
    reg16 = levels_tr[4].reshape(n_tr, -1).astype(np.int64)
    errs = nesting_errors(levels_tr)
    assert not errs, f"nesting violated: {errs}"
    ch = hierarchy_chain(act_tr, valid_het, reg16, want_deltas=True,
                         want_ips=True)
    log(f"identity_rel={ch['identity_max_rel']:.2e} "
        f"zero_sum={ch['zero_sum_max']:.2e} "
        f"cross_ip_max={float(ch['cross_ip'].max()):.2e}")
    E_total = ch["E_total"]
    log("level energies (K=2,4,8,16): " + str(np.round(E_total, 3)))

    # ---- 6. label-free shuffled-regime null ---------------------------------
    log(f"null: S={S_PERM}, seed={NULL_SEED}")
    null = shuffle_null(act_tr, valid_het, reg16, S=S_PERM, seed=NULL_SEED,
                        verbose=100)
    log("null means: " + str(np.round(null.mean(0), 3)))

    # ---- 7. stopping rule (FROZEN before any classifier) --------------------
    L_star, rows = stopping_rule(E_total, null)
    for r in rows:
        log(f"  K={r['K']}: E={r['real_energy']:.2f} null_mean="
            f"{r['null_mean']:.2f} p95={r['null_p95']:.2f} p={r['p_raw_plus_one']:.3f} "
            f"q={r['q_bh']:.3f} keep={r['keep']}")
    log(f"L* = {L_star}")

    # ---- 8. downstream demonstration ----------------------------------------
    H_trva = np.load(os.path.join(CACHE, "haptics_inference_banks_H_trva.npy")
                     ).astype(np.float64)
    G_te = np.load(os.path.join(CACHE, "haptics_inference_banks_G_te.npy")
                   ).astype(np.float64)
    H_te = np.load(os.path.join(CACHE, "haptics_inference_banks_H_te.npy")
                   ).astype(np.float64)
    y_dev = np.concatenate([ytr, yva])

    from experiments.heramba_cca_ranked_haptics_seed42.runner import ridge_eval
    m0_res, m0_pred = ridge_eval(G_trva, y_dev, G_te, yte)
    assert abs(m0_res["macro_f1"] - GATE_A) < 5e-5, m0_res
    log(f"[gate A] MiniROCKET = {m0_res['macro_f1']} OK")
    r2_res, r2_pred = ridge_eval(np.hstack([G_trva, H_trva]), y_dev,
                                 np.hstack([G_te, H_te]), yte)
    assert r2_res["macro_f1"] == GATE_B, r2_res
    log(f"[gate B] Raw G+H   = {r2_res['macro_f1']} OK")

    # H-replication check: recompute the flat K=8 bank from MY activations
    # using the STORED frozen flat regimes (same estimator as the bank)
    reg_flat_stored = np.load(os.path.join(
        CACHE, "haptics_inference_banks_reg_trva.npy")).astype(np.int64)
    from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (
        heterogeneity_features)
    H_rep = heterogeneity_features(act_tr, valid_het,
                                   reg_flat_stored[:n_tr], K=8)
    h_rep_diff = float(np.abs(H_rep - H_trva[:n_tr]).max())
    log(f"H-bank replication from my activations: max|diff| = "
        f"{h_rep_diff:.2e}")

    # hierarchical bank at the FROZEN depth, for TRAIN+VAL (155 rows,
    # matching the canonical banks' dev scope) and TEST; the train rows
    # reuse the audit chain, val/test rows use the frozen tree transform-only
    kept_K = [r["K"] for r in rows if r["keep"]]
    Xva_z = znorm(Xva)
    lat_va = _encoder_latents(model, Xva_z, device)
    lat_va_pool = lat_va.reshape(-1, lat_va.shape[-1])
    reg16_va = assign_levels(lat_va_pool, C)[4].reshape(len(yva), -1) \
        .astype(np.int64)
    act_va_full, _ = compute_activations(extractor, Xva_z)
    ch_va = hierarchy_chain(act_va_full[:, 4998:], valid_het, reg16_va,
                            want_deltas=True, want_ips=False)
    Xte_z = znorm(Xte)
    lat_te = _encoder_latents(model, Xte_z, device)
    lat_te_pool = lat_te.reshape(-1, lat_te.shape[-1])
    levels_te = assign_levels(lat_te_pool, C)             # never refits
    reg16_te = levels_te[4].reshape(len(yte), -1).astype(np.int64)
    act_te_full, _ = compute_activations(extractor, Xte_z)
    ch_te = hierarchy_chain(act_te_full[:, 4998:], valid_het, reg16_te,
                            want_deltas=True, want_ips=False)
    Hh_tr = np.hstack([np.hstack([ch["deltas"][K][:, j, :]
                                  for j in range(K)]) for K in kept_K])
    Hh_va = np.hstack([np.hstack([ch_va["deltas"][K][:, j, :]
                                  for j in range(K)]) for K in kept_K])
    Hh_te = np.hstack([np.hstack([ch_te["deltas"][K][:, j, :]
                                  for j in range(K)]) for K in kept_K])
    Hh_trva = np.vstack([Hh_tr, Hh_va])
    log(f"H_hier: K-blocks {kept_K} -> {Hh_trva.shape[1]} features "
        f"(trva {Hh_trva.shape}, te {Hh_te.shape})")
    hier_res, hier_pred = ridge_eval(np.hstack([G_trva, Hh_trva]), y_dev,
                                     np.hstack([G_te, Hh_te]), yte)
    log(f"[hier demo] [G || H_hier(L*={L_star})] = {hier_res['macro_f1']}")

    # ---- artifacts ----------------------------------------------------------
    save_json({
        "dataset": "Haptics", "seed": SEED,
        "n_train": n_tr, "n_val": int(len(yva)), "n_test": int(len(yte)),
        "T": int(Xtr.shape[1]), "n_features_G": int(G_trva.shape[1]),
        "tree_meta": tree_meta,
        "identity_max_abs": ch["identity_max_abs"],
        "identity_max_rel": ch["identity_max_rel"],
        "zero_sum_max": ch["zero_sum_max"],
        "cross_ip_max": float(np.abs(ch["cross_ip"]).max()),
        "level_energies": {"K": [int(k) for k in LEVELS[1:]],
                           "real": E_total.tolist()},
        "stopping": {"L_star": int(L_star), "rows": rows,
                     "S": S_PERM, "null_seed": NULL_SEED},
        "gates": {"minirocket": m0_res["macro_f1"],
                  "raw_gh": r2_res["macro_f1"]},
        "demo": {"H_hier_features": int(Hh_trva.shape[1]),
                 "kept_K": kept_K,
                 "h_replication_maxdiff": h_rep_diff,
                 "macro_f1": hier_res["macro_f1"]},
        "runtime_s": time.time() - t00,
    }, os.path.join(OUT, "results.json"))
    save_json({"seed": SEED, "S_perm": S_PERM, "null_seed": NULL_SEED,
               "min_occupancy": 0.01, "L_star": int(L_star),
               "decisions": rows}, os.path.join(OUT, "stopping_decision.json"))
    save_json({"tree_meta": tree_meta,
               "level_sizes": [len(c) for c in C]},
              os.path.join(OUT, "hierarchy.json"))
    np.save(os.path.join(OUT, "hierarchy_assignments.npy"), reg16)
    save_json({"identity_max_abs": ch["identity_max_abs"],
               "identity_max_rel": ch["identity_max_rel"],
               "tolerance": "float64; target <= 1e-10 relative",
               "pass": bool(ch["identity_max_rel"] < 1e-10)},
              os.path.join(OUT, "hierarchy_energy_identity.json"))
    save_json({"cross_ip_max": float(np.abs(ch["cross_ip"]).max()),
               "pairs": ["|K%d-K%d|" % p for p in
                         ((2, 4), (2, 8), (2, 16), (4, 8), (4, 16), (8, 16))],
               "method": "exact joint count-weighted inner products; "
                         "no Gram-Schmidt, no CCA",
               "tolerance": "float64 machine precision"},
              os.path.join(OUT, "orthogonality_report.json"))
    save_json({"poison_checks": "see tests/test_nested_regimes.py",
               "labels_used_in_intrinsic_stage": False,
               "val_used_in_intrinsic_stage": False,
               "test_used_in_intrinsic_stage": False},
              os.path.join(OUT, "leakage_audit.json"))

    with open(os.path.join(OUT, "level_energy.csv"), "w") as f:
        f.write("K,real_energy,null_mean,null_std,null_p95,p_raw,q_bh,keep\n")
        for r in rows:
            f.write(f"{r['K']},{r['real_energy']:.6f},{r['null_mean']:.6f},"
                    f"{r['null_std']:.6f},{r['null_p95']:.6f},"
                    f"{r['p_raw_plus_one']:.6f},{r['q_bh']:.6f},{r['keep']}\n")
    with open(os.path.join(OUT, "sample_energy.csv"), "w") as f:
        f.write("sample," + ",".join(f"E_K{k}" for k in LEVELS[1:])
                + "," + ",".join(f"frac_K{k}" for k in LEVELS[1:]) + "\n")
        Es = ch["E_sample"]
        tot = np.maximum(Es.sum(axis=1, keepdims=True), 1e-30)
        for i in range(n_tr):
            f.write(f"{i}," + ",".join(f"{v:.8f}" for v in Es[i]) + ","
                    + ",".join(f"{v:.8f}" for v in Es[i] / tot[i]) + "\n")
    with open(os.path.join(OUT, "null_energy.csv"), "w") as f:
        f.write("perm," + ",".join(f"null_K{k}" for k in LEVELS[1:]) + "\n")
        for s in range(null.shape[0]):
            f.write(f"{s}," + ",".join(f"{v:.8f}" for v in null[s]) + "\n")
    with open(os.path.join(OUT, "final_comparison.csv"), "w") as f:
        f.write("Method,Representation,H_dims,Test_MacroF1,Notes\n")
        f.write(f"MiniROCKET,G,{G_trva.shape[1]},{m0_res['macro_f1']:.4f},"
                f"identity gate A\n")
        f.write(f"Raw G+H,[G||H],{H_trva.shape[1]},{r2_res['macro_f1']:.4f},"
                f"identity gate B (flat K=8 bank)\n")
        f.write(f"Hierarchical,[G||H_hier(L*={L_star})],{Hh_tr.shape[1]},"
                f"{hier_res['macro_f1']:.4f},intrinsic-depth demonstration\n")
    for nm, pr in (("minirocket", m0_pred), ("raw_gh", r2_pred),
                   ("hierarchical", hier_pred)):
        np.save(os.path.join(OUT, "predictions", f"{nm}.npy"), pr)

    # ---- figures ------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # energy_by_level + null band
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    Ks = [r["K"] for r in rows]
    real = [r["real_energy"] for r in rows]
    mean = [r["null_mean"] for r in rows]
    p95 = [r["null_p95"] for r in rows]
    ax.errorbar(Ks, mean, yerr=[np.array(p95) - np.array(mean),
                                np.zeros_like(p95)], fmt="o--",
                color="gray", label="null mean / 95th pct", capsize=3)
    ax.plot(Ks, real, "o-", color="C0", label="real energy")
    kept = [k for k, r in zip(Ks, rows) if r["keep"]]
    ax.axvline(L_star, color="C3", ls=":", label=f"L* = {L_star}")
    ax.set(xlabel="K (level)", ylabel="total detail energy",
           title="Level energy vs shuffled-regime null")
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "figures", f"energy_by_level.{ext}"))
    plt.close(fig)

    # energy_fraction
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    frac = ch["E_sample"].mean(axis=0)
    fr = frac / max(frac.sum(), 1e-30)
    ax.bar([str(k) for k in LEVELS[1:]], fr, color="C0")
    ax.set(xlabel="K (level)", ylabel="mean energy fraction (train)",
           title="Normalized level energies")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "figures", f"energy_fraction.{ext}"))
    plt.close(fig)

    # null_vs_real
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for i, K in enumerate(LEVELS[1:]):
        ax.hist(null[:, i], bins=40, alpha=0.4,
                label=f"null K={K}")
        ax.axvline(E_total[i], color=f"C{i}", lw=2)
    ax.set(xlabel="total energy", ylabel="permutations",
           title="Real (lines) vs null (histograms)")
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "figures", f"null_vs_real_energy.{ext}"))
    plt.close(fig)

    # regime assignment example (train sample 0, first 400 timesteps)
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    seg = slice(0, 400)
    ax.plot(Xtr[0, seg], color="k", lw=0.6, zorder=0)
    for l, K in enumerate(LEVELS[1:], start=1):
        rl = reg16[0, seg] >> (4 - l)
        ax.scatter(np.arange(400), np.full(400, -1.0 + 0.55 * l), c=rl,
                   cmap="tab20" if K == 16 else "tab10", s=3, vmin=0,
                   vmax=15)
        ax.text(402, -1.0 + 0.55 * l, f"K={K}", fontsize=8, va="center")
    ax.set(ylabel="normalized signal / level rows", xlabel="timestep",
           title="Nested regime assignments (train sample 0)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "figures",
                                 f"regime_assignment_example.{ext}"))
    plt.close(fig)

    # hierarchy tree diagram
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    depth_y = {0: 3.0, 1: 2.0, 2: 1.0, 3: 0.0}
    xs = {}
    for dep in range(4):
        n_nodes = 2 ** dep
        for p in range(n_nodes):
            xs[(dep, p)] = (p + 0.5) / n_nodes
            ax.scatter([xs[(dep, p)]], [depth_y[dep]], s=120, c="C0", zorder=2)
            ax.annotate(str(p), (xs[(dep, p)], depth_y[dep]), fontsize=6,
                        ha="center", va="center", color="white", zorder=3)
            if dep > 0:
                pp = p >> 1
                ax.plot([xs[(dep - 1, pp)], xs[(dep, p)]],
                        [depth_y[dep - 1], depth_y[dep]], "k-", lw=0.7,
                        zorder=1)
    for dep in range(4):
        ax.text(-0.02, depth_y[dep], f"K={2 ** dep}", fontsize=9,
                ha="right", va="center")
    ax.set_title("Nested regime tree (binary path ids)")
    ax.axis("off")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "figures", f"hierarchy_tree.{ext}"))
    plt.close(fig)

    save_json({
        "experiment": "haptics_nested_regimes",
        "dataset": "Haptics", "seed": SEED,
        "tree": {"levels": [1, 2, 4, 8, 16],
                 "splitter": tree_meta["splitter"],
                 "child_rule": tree_meta["child_rule"],
                 "input": tree_meta["input"],
                 "singleton_fallbacks": tree_meta["singleton_fallbacks"]},
        "min_occupancy": 0.01,
        "null": {"S": S_PERM, "seed": NULL_SEED},
        "stopping": {"rule": "real E_l > null p95 AND q_bh < 0.05, "
                             "coarse->fine, stop at first failure",
                     "alpha": 0.05},
        "gates": {"A_expected": GATE_A, "B_expected": GATE_B},
        "label_free": "hierarchy, energies, null, stopping and depth are "
                      "derived from train signals/latents only",
    }, os.path.join(OUT, "config.json"))
    log(f"total: {time.time() - t00:.0f}s")


if __name__ == "__main__":
    main()
