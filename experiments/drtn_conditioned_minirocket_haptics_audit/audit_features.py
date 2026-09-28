"""
Feature-level audit (A9/A10): does H(M2) == H(M3) under the CURRENT code?

Computes heterogeneity features from the SAME raw responses with the three
regime variants (M1 DRTN, M2 current-iid, M2spec per-sample, M3 shuffled)
on a deterministic sample subset, and reports the checkpoint table:

    C0 original DRTN regimes
    C1 M2 regime arrays (current iid-global + spec per-sample)
    C2 M3 regime arrays
    C3 pre-feature-builder inputs (= C1/C2 arrays, hash-verified)
    C4 heterogeneity features
    C5 classifier input (global block + hetero block)
    C6 predictions (RidgeCV on the subset, diagnostic only)

Usage:
    python -m experiments.drtn_conditioned_minirocket_haptics_audit.audit_features \
        [--n-trainval 40] [--n-test 40]
"""
import argparse
import csv
import hashlib
import json
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from experiments.external_stack_generalization.data import load_dataset  # noqa: E402
from experiments.drtn_conditioned_minirocket_haptics_3seed.runner import (  # noqa: E402
    load_drtn_checkpoint,
    extract_drtn_regimes,
    create_random_regime_control,
    create_shuffled_regime_control,
    compute_regime_heterogeneity,
)
from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (  # noqa: E402
    compute_raw_activations,
    ppv_from_activations,
)
from experiments.drtn_conditioned_minirocket_haptics_audit.audit_regimes import (  # noqa: E402
    arr_hash, occupancy_hist, spec_random_control,
)

OUT_DIR = os.path.join(ROOT, "results", "drtn_conditioned_minirocket_haptics_audit")
K = 8
SEEDS = [42, 43, 44]
ALPHAS = np.logspace(-4, 4, 20)
N_FEATURES = 9996
N_GLOBAL = N_FEATURES // 2
N_HET = N_FEATURES - N_GLOBAL


def feat_stats(a, b):
    d = a - b
    return {
        "max_abs_diff": float(np.max(np.abs(d))),
        "mean_abs_diff": float(np.mean(np.abs(d))),
        "n_exactly_equal": int(np.sum(d == 0)),
        "n_elements": int(d.size),
        "hash_a": arr_hash(a.view(np.int64) if a.dtype == np.int64
                           else (a * 1e9).astype(np.int64)),
        "allclose": bool(np.allclose(a, b, atol=1e-12)),
    }


def audit_seed_features(seed, data, device, n_trva=40, n_te=40):
    print(f"\n{'='*74}\nFEATURE AUDIT SEED {seed}\n{'='*74}", flush=True)
    seed_dir = os.path.join(OUT_DIR, f"seed{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    Xtrva = np.vstack([data["Xtr"], data["Xva"]])
    Xte = data["Xte"]
    ytr = data["ytr"]; yva = data["yva"]; yte = data["yte"]

    # deterministic sample subset (first n of each split; ordering preserved)
    idx_trva = np.arange(min(n_trva, len(Xtrva)))
    idx_te = np.arange(min(n_te, len(Xte)))
    Xa, Xb = Xtrva[idx_trva], Xte[idx_te]

    # regimes (frozen DRTN)
    drtn, _ = load_drtn_checkpoint(seed, device)
    C0_trva = extract_drtn_regimes(drtn, Xa, device=device)
    C0_te = extract_drtn_regimes(drtn, Xb, device=device)

    # variant regime arrays (current implementations)
    C1_trva = create_random_regime_control(C0_trva, seed=seed)
    C1_te = create_random_regime_control(C0_te, seed=seed)
    C1s_trva = spec_random_control(C0_trva, seed=seed)
    C1s_te = spec_random_control(C0_te, seed=seed)
    C2_trva = create_shuffled_regime_control(C0_trva, seed=seed)
    C2_te = create_shuffled_regime_control(C0_te, seed=seed)

    # canonical MiniROCKET + raw activations (SAME extractor/responses for all)
    from aeon.transformations.collection.convolution_based import MiniRocket
    ext = MiniRocket(random_state=seed, n_jobs=-1)
    Xa_z = ((Xa - Xa.mean(-1, keepdims=True)) /
            (Xa.std(-1, keepdims=True) + 1e-8)).astype(np.float32)
    Xb_z = ((Xb - Xb.mean(-1, keepdims=True)) /
            (Xb.std(-1, keepdims=True) + 1e-8)).astype(np.float32)
    ext.fit(Xa_z[:, None, :].astype(np.float32))
    act_a, valid = compute_raw_activations(ext, Xa_z)
    act_b, _ = compute_raw_activations(ext, Xb_z)

    # canonical PPV via the validated valid-region convention; also the
    # 3-seed runner's hstack path uses Ftrva_global from aeon transform,
    # equivalent by construction (verified in transfer tests).
    PPV_a = ppv_from_activations(act_a, valid)
    PPV_b = ppv_from_activations(act_b, valid)
    biases = ext.parameters[-1]

    # heterogeneity for each variant -- SAME responses, different regimes
    H1_trva = compute_regime_heterogeneity(
        act_a[:, N_GLOBAL:], biases[N_GLOBAL:], C0_trva, N_HET)
    H1_te = compute_regime_heterogeneity(
        act_b[:, N_GLOBAL:], biases[N_GLOBAL:], C0_te, N_HET)

    H2_trva = compute_regime_heterogeneity(
        act_a[:, N_GLOBAL:], biases[N_GLOBAL:], C1_trva, N_HET)
    H2_te = compute_regime_heterogeneity(
        act_b[:, N_GLOBAL:], biases[N_GLOBAL:], C1_te, N_HET)

    H2s_trva = compute_regime_heterogeneity(
        act_a[:, N_GLOBAL:], biases[N_GLOBAL:], C1s_trva, N_HET)
    H2s_te = compute_regime_heterogeneity(
        act_b[:, N_GLOBAL:], biases[N_GLOBAL:], C1s_te, N_HET)

    H3_trva = compute_regime_heterogeneity(
        act_a[:, N_GLOBAL:], biases[N_GLOBAL:], C2_trva, N_HET)
    H3_te = compute_regime_heterogeneity(
        act_b[:, N_GLOBAL:], biases[N_GLOBAL:], C2_te, N_HET)

    # ---- checkpoint table ----
    table = {
        "C0": {"name": "DRTN regimes", "hash_test": arr_hash(C0_te),
               "occupancy_test": occupancy_hist(C0_te)},
        "C1": {"name": "M2 current regimes", "hash_test": arr_hash(C1_te),
               "occupancy_test": occupancy_hist(C1_te)},
        "C1s": {"name": "M2 spec regimes", "hash_test": arr_hash(C1s_te),
                "occupancy_test": occupancy_hist(C1s_te)},
        "C2": {"name": "M3 regimes", "hash_test": arr_hash(C2_te),
               "occupancy_test": occupancy_hist(C2_te)},
    }
    array_checks = {
        "C1_vs_C2_arrays_differ": not np.array_equal(C1_te, C2_te),
        "C1s_vs_C2_arrays_differ": not np.array_equal(C1s_te, C2_te),
        "C1_vs_C1s_arrays_differ": not np.array_equal(C1_te, C1s_te),
        "C1_vs_C0_arrays_differ": not np.array_equal(C1_te, C0_te),
        "C2_vs_C0_arrays_differ": not np.array_equal(C2_te, C0_te),
        "no_shared_memory_C1_C2": not np.shares_memory(C1_te, C2_te),
    }

    # ---- feature-level identity ----
    feat_checks = {
        "H2_vs_H3_test": feat_stats(H2_te, H3_te),
        "H2s_vs_H3_test": feat_stats(H2s_te, H3_te),
        "H2_vs_H2s_test": feat_stats(H2_te, H2s_te),
        "H1_vs_H3_test": feat_stats(H1_te, H3_te),
        "H2_vs_H3_trva": feat_stats(H2_trva, H3_trva),
        "H2s_vs_H3_trva": feat_stats(H2s_trva, H3_trva),
    }

    # ---- classifier-level on the subset (diagnostic C5/C6) ----
    ytrva = np.concatenate([ytr, yva])[idx_trva]
    yte_sub = yte[idx_te]
    variants = {
        "M1": (np.hstack([PPV_a[:, :N_GLOBAL], H1_trva]),
               np.hstack([PPV_b[:, :N_GLOBAL], H1_te])),
        "M2": (np.hstack([PPV_a[:, :N_GLOBAL], H2_trva]),
               np.hstack([PPV_b[:, :N_GLOBAL], H2_te])),
        "M2spec": (np.hstack([PPV_a[:, :N_GLOBAL], H2s_trva]),
                   np.hstack([PPV_b[:, :N_GLOBAL], H2s_te])),
        "M3": (np.hstack([PPV_a[:, :N_GLOBAL], H3_trva]),
               np.hstack([PPV_b[:, :N_GLOBAL], H3_te])),
    }
    preds, sel_alpha = {}, {}
    for v, (Ftr, Fte) in variants.items():
        assert Ftr.shape[1] == N_FEATURES and Fte.shape[1] == N_FEATURES
        ridge = RidgeClassifierCV(alphas=ALPHAS).fit(Ftr, ytrva)
        preds[v] = ridge.predict(Fte)
        sel_alpha[v] = float(ridge.alpha_)
    pred_checks = {
        f"{a}_vs_{b}_pred_identical":
            bool(np.array_equal(preds[a], preds[b]))
        for a, b in [("M2", "M3"), ("M2spec", "M3"), ("M1", "M3")]
    }
    classifier_checks = {
        "selected_alpha": sel_alpha,
        "test_mf1_subset": {
            v: round(float(f1_score(yte_sub, preds[v], average="macro",
                                    zero_division=0)), 4) for v in preds},
        **pred_checks,
    }

    # save a representative feature slice for forensic reference
    np.savez_compressed(
        os.path.join(seed_dir, "feature_audit_slice.npz"),
        H1_te=H1_te[:5], H2_te=H2_te[:5], H2s_te=H2s_te[:5], H3_te=H3_te[:5])

    result = {
        "seed": seed,
        "n_subset_trva": int(len(idx_trva)), "n_subset_test": int(len(idx_te)),
        "checkpoint_table": table,
        "array_checks": array_checks,
        "feature_checks": feat_checks,
        "classifier_checks": classifier_checks,
    }
    with open(os.path.join(seed_dir, "feature_audit.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"  arrays differ (M2cur/M3): {array_checks['C1_vs_C2_arrays_differ']}  "
          f"(M2spec/M3): {array_checks['C1s_vs_C2_arrays_differ']}")
    print(f"  H2 vs H3 test: max|d|={feat_checks['H2_vs_H3_test']['max_abs_diff']:.3e} "
          f"n_equal={feat_checks['H2_vs_H3_test']['n_exactly_equal']}/"
          f"{feat_checks['H2_vs_H3_test']['n_elements']}")
    print(f"  H2s vs H3 test: max|d|={feat_checks['H2s_vs_H3_test']['max_abs_diff']:.3e} "
          f"n_equal={feat_checks['H2s_vs_H3_test']['n_exactly_equal']}/"
          f"{feat_checks['H2s_vs_H3_test']['n_elements']}")
    print(f"  subset alphas: {sel_alpha}")
    print(f"  subset preds identical M2==M3: {pred_checks['M2_vs_M3_pred_identical']}  "
          f"M2spec==M3: {pred_checks['M2spec_vs_M3_pred_identical']}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trainval", type=int, default=40)
    ap.add_argument("--n-test", type=int, default=40)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    data = load_dataset("Haptics")
    out = {}
    for seed in SEEDS:
        out[str(seed)] = audit_seed_features(seed, data, device,
                                             args.n_trainval, args.n_test)
    with open(os.path.join(OUT_DIR, "feature_audit_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nFeature audit complete.")


if __name__ == "__main__":
    main()
