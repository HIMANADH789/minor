"""
Runner: DRTN-Conditioned MiniROCKET -- Haptics Seed 42
=====================================================

Tests whether learned discrete temporal regimes from DRTN can improve
MiniROCKET's fixed convolutional representation by conditioning its
temporal pooling on learned regimes, under an equal 9,996-feature budget.

Models:
    M0: Canonical MiniROCKET (9,996 PPV features)
    M1: DRTN-conditioned MiniROCKET (4,998 global PPV + 4,998 heterogeneity)
    M2: Random-regime control (same as M1 but random regimes)
    M3: Shuffled-regime control (same as M1 but temporally shuffled regimes)

Usage:
    python -m experiments.drtn_conditioned_minirocket_haptics_seed42.runner
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from experiments.external_stack_generalization.data import load_dataset  # noqa: E402
from experiments.drtn_conditioned_minirocket_haptics_seed42.core import (  # noqa: E402
    compute_regime_occupancy_stats,
    create_random_regime_control,
    create_shuffled_regime_control,
    extract_drtn_regimes,
)

SEED = 42
DATASET = "Haptics"
OUT_DIR = os.path.join(ROOT, "results", "drtn_conditioned_minirocket_haptics_seed42")
ALPHAS = np.logspace(-4, 4, 20)

# Feature budget: exactly 9,996 features
N_FEATURES = 9996
N_GLOBAL = N_FEATURES // 2  # 4998
N_HETEROGENEITY = N_FEATURES - N_GLOBAL  # 4998


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def load_official_drtn(device):
    """Load the official seed-42 R5 checkpoint (frozen)."""
    from models.drtn.model import build_model
    rdir = os.path.join(ROOT, "results", "drtn_haptics_seed42", "R5")
    with open(os.path.join(rdir, "result.json")) as f:
        official = json.load(f)
    ck = torch.load(os.path.join(rdir, "checkpoint.pt"),
                    map_location=device, weights_only=False)
    cfg = ck["config"]
    model = build_model("R5", c_in=1, n_classes=5, d_model=cfg["d_model"],
                        n_codes=cfg["n_codes"], tau=cfg["tau"],
                        ema_decay=cfg["ema_decay"], beta=cfg["beta_commit"],
                        lam_div=cfg["lam_div"], dead_threshold=cfg["dead_threshold"],
                        revival_patience=cfg["revival_patience"],
                        traj_layers=cfg["trajectory"]["layers"],
                        traj_heads=cfg["trajectory"]["heads"],
                        traj_ffn=cfg["trajectory"]["ffn"],
                        traj_dropout=cfg["trajectory"]["dropout"])
    model.load_state_dict(ck["model_state"])
    model.to(device).eval()
    return model, official


def compute_raw_minirocket_features(extractor, X):
    """
    Compute raw MiniROCKET convolution responses for ALL kernels.

    Reimplements the aeon MiniROCKET univariate transform to extract
    per-kernel per-timestep activation responses (before PPV thresholding).

    Returns
    -------
    responses : np.ndarray, shape (n_samples, n_features, T)
        Raw convolution responses for each kernel/feature.
    biases : np.ndarray, shape (n_features,)
        Activation thresholds (biases) for each kernel/feature.
    """
    from itertools import combinations

    X = X.astype(np.float32)
    n_samples, T = X.shape

    parameters = extractor.parameters
    n_channels_per_comb, channel_idx, dilations, n_features_per_dil, biases_full = parameters

    indices = np.array(list(combinations(range(9), 3)), dtype=np.int32)
    n_kernels = len(indices)
    n_dilations = len(dilations)
    n_features_total = n_kernels * int(np.sum(n_features_per_dil))

    responses = np.zeros((n_samples, n_features_total, T), dtype=np.float32)

    for i in range(n_samples):
        _X = X[i]
        A = -_X
        G = 3.0 * _X

        f_start = 0
        for j in range(n_dilations):
            _padding0 = j % 2
            dilation = dilations[j]
            padding = (8 * dilation) // 2
            n_features = n_features_per_dil[j]

            C_alpha = A.copy()
            C_gamma = np.zeros((9, T), dtype=np.float32)
            C_gamma[4] = G

            start = dilation
            end = T - padding
            for gi in range(4):
                C_alpha[-end:] += A[:end]
                C_gamma[gi, -end:] = G[:end]
                end += dilation
            for gi in range(5, 9):
                C_alpha[:-start] += A[start:]
                C_gamma[gi, :-start] = G[start:]
                start += dilation

            for k_idx in range(n_kernels):
                f_end = f_start + n_features
                _padding1 = (_padding0 + k_idx) % 2
                a, b, c = indices[k_idx]
                C = C_alpha + C_gamma[a] + C_gamma[b] + C_gamma[c]

                if _padding1 == 0:
                    for f in range(n_features):
                        responses[i, f_start + f, :] = C
                else:
                    for f in range(n_features):
                        responses[i, f_start + f, padding:-padding] = C[padding:-padding]

                f_start = f_end

    return responses, biases_full


def compute_regime_heterogeneity(
    responses, biases, regimes, n_kernels, K=8, min_occupancy=0.01
):
    """
    Compute regime-conditioned activation heterogeneity.

    F_m = sum_k  q_k * (p_{m,k} - PPV_m)^2

    where:
        q_k      = fraction of timesteps in regime k
        p_{m,k}  = PPV of kernel m within regime k
        PPV_m    = global PPV of kernel m
    """
    n_samples, _, T = responses.shape
    min_count = int(np.ceil(min_occupancy * T))

    heterogeneity = np.zeros((n_samples, n_kernels), dtype=np.float64)

    for i in range(n_samples):
        regimes_i = regimes[i]
        # binary activation: a_m(t) = 1[r_m(t) > b_m]
        active = (responses[i] > biases[:, None])  # (n_kernels, T)

        global_ppv = active.mean(axis=1)  # (n_kernels,)

        for k in range(K):
            mask = regimes_i == k
            count_k = int(mask.sum())
            if count_k < min_count:
                continue
            q_k = count_k / T
            ppv_k = active[:, mask].mean(axis=1)
            heterogeneity[i] += q_k * (ppv_k - global_ppv) ** 2

    return heterogeneity


def fit_ridge_classifier(X_train, y_train, alphas=ALPHAS):
    """Fit RidgeClassifierCV with canonical alpha grid."""
    ridge = RidgeClassifierCV(alphas=alphas)
    ridge.fit(X_train, y_train)
    return ridge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-m4", action="store_true",
                        help="Skip optional M4 control")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "diagnostics"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "figures"), exist_ok=True)

    print("=" * 70)
    print("DRTN-CONDITIONED MINIROCKET -- HAPTICS SEED 42")
    print("=" * 70)

    # === STEP 1: Load and verify dataset ===
    print("\n[STEP 1] Loading Haptics dataset...")
    d = load_dataset(DATASET)
    assert len(d["Xtr"]) == 132
    assert len(d["Xva"]) == 23
    assert len(d["Xte"]) == 308
    assert d["L"] == 1092
    assert d["n_classes"] == 5
    Xtr, ytr = d["Xtr"], d["ytr"]
    Xva, yva = d["Xva"], d["yva"]
    Xte, yte = d["Xte"], d["yte"]
    print(f"  Train: {len(Xtr)}, Val: {len(Xva)}, Test: {len(Xte)}")
    print(f"  T: {d['L']}, Classes: {d['n_classes']}")

    # === STEP 2: Load frozen DRTN R5 checkpoint ===
    print("\n[STEP 2] Loading frozen DRTN R5 checkpoint...")
    drtn, official = load_official_drtn(device)
    print(f"  Official val MF1: {official['best_val_mf1']}")
    print(f"  Official test MF1: {official['test']['macro_f1']}")

    # === STEP 3: Extract DRTN regime assignments ===
    print("\n[STEP 3] Extracting DRTN regime assignments...")
    t0 = time.time()
    Xtrva = np.vstack([Xtr, Xva])
    ytrva = np.concatenate([ytr, yva])
    regimes_trva = extract_drtn_regimes(drtn, Xtrva, device=device)
    regimes_te = extract_drtn_regimes(drtn, Xte, device=device)
    regimes_tr = regimes_trva[:len(Xtr)]
    regimes_va = regimes_trva[len(Xtr):]
    print(f"  Extracted in {time.time() - t0:.1f}s")
    print(f"  Regime values: {np.unique(regimes_te)}")
    regime_stats = compute_regime_occupancy_stats(regimes_te, K=8)
    print(f"  Active codes: {regime_stats['active_codes']}, "
          f"Entropy: {regime_stats['entropy']:.4f}, "
          f"Perplexity: {regime_stats['perplexity']:.4f}")

    # === STEP 4: Fit canonical MiniRocket extractor ===
    print("\n[STEP 4] Fitting canonical MiniRocket extractor...")
    t0 = time.time()
    from aeon.transformations.collection.convolution_based import MiniRocket
    extractor = MiniRocket(random_state=SEED, n_jobs=-1)
    extractor.fit(Xtr[:, None, :].astype(np.float32))
    print(f"  Fitted in {time.time() - t0:.1f}s")

    def mr_transform(X):
        return extractor.transform(X[:, None, :].astype(np.float32))

    Ftr = mr_transform(Xtr)
    Fva = mr_transform(Xva)
    Fte = mr_transform(Xte)
    Ftrva = mr_transform(Xtrva)
    n_features = Ftr.shape[1]
    print(f"  Features: {n_features}")
    assert n_features == 9996

    # === STEP 5: Compute raw activations for regime-conditioned features ===
    print("\n[STEP 5] Computing raw MiniROCKET activations...")
    t0 = time.time()
    responses_tr, biases = compute_raw_minirocket_features(extractor, Xtr)
    responses_va, _ = compute_raw_minirocket_features(extractor, Xva)
    responses_te, _ = compute_raw_minirocket_features(extractor, Xte)
    responses_trva, _ = compute_raw_minirocket_features(extractor, Xtrva)
    print(f"  Computed in {time.time() - t0:.1f}s")

    # === STEP 6: Split features: global block + heterogeneity block ===
    print("\n[STEP 6] Constructing feature blocks...")

    Ftr_global = Ftr[:, :N_GLOBAL]
    Fva_global = Fva[:, :N_GLOBAL]
    Fte_global = Fte[:, :N_GLOBAL]
    Ftrva_global = Ftrva[:, :N_GLOBAL]

    # Use last N_HETEROGENEITY raw responses for heterogeneity computation
    resp_tr_het = responses_tr[:, N_GLOBAL:]
    resp_va_het = responses_va[:, N_GLOBAL:]
    resp_te_het = responses_te[:, N_GLOBAL:]
    resp_trva_het = responses_trva[:, N_GLOBAL:]
    biases_het = biases[N_GLOBAL:]

    print("\n[STEP 6a] M0: Canonical MiniROCKET...")
    M0_tr = Ftrva
    M0_te = Fte

    print("[STEP 6b] M1: DRTN-conditioned (DRTN regimes)...")
    Ftrva_het_m1 = compute_regime_heterogeneity(resp_trva_het, biases_het, regimes_trva, N_HETEROGENEITY)
    Fte_het_m1 = compute_regime_heterogeneity(resp_te_het, biases_het, regimes_te, N_HETEROGENEITY)
    M1_tr = np.hstack([Ftrva_global, Ftrva_het_m1])
    M1_te = np.hstack([Fte_global, Fte_het_m1])

    print("[STEP 6c] M2: Random-regime control...")
    regimes_trva_random = create_random_regime_control(regimes_trva, seed=SEED)
    regimes_te_random = create_random_regime_control(regimes_te, seed=SEED)
    Ftrva_het_m2 = compute_regime_heterogeneity(resp_trva_het, biases_het, regimes_trva_random, N_HETEROGENEITY)
    Fte_het_m2 = compute_regime_heterogeneity(resp_te_het, biases_het, regimes_te_random, N_HETEROGENEITY)
    M2_tr = np.hstack([Ftrva_global, Ftrva_het_m2])
    M2_te = np.hstack([Fte_global, Fte_het_m2])

    print("[STEP 6d] M3: Shuffled-regime control...")
    regimes_trva_shuf = create_shuffled_regime_control(regimes_trva, seed=SEED)
    regimes_te_shuf = create_shuffled_regime_control(regimes_te, seed=SEED)
    Ftrva_het_m3 = compute_regime_heterogeneity(resp_trva_het, biases_het, regimes_trva_shuf, N_HETEROGENEITY)
    Fte_het_m3 = compute_regime_heterogeneity(resp_te_het, biases_het, regimes_te_shuf, N_HETEROGENEITY)
    M3_tr = np.hstack([Ftrva_global, Ftrva_het_m3])
    M3_te = np.hstack([Fte_global, Fte_het_m3])

    for nm, Mt, Me in [("M0", M0_tr, M0_te), ("M1", M1_tr, M1_te),
                        ("M2", M2_tr, M2_te), ("M3", M3_tr, M3_te)]:
        print(f"  {nm} train: {Mt.shape}, test: {Me.shape}")

    # Verify feature counts
    for nm, Mt, Me in [("M0", M0_tr, M0_te), ("M1", M1_tr, M1_te),
                        ("M2", M2_tr, M2_te), ("M3", M3_tr, M3_te)]:
        assert Mt.shape[1] == 9996, f"{nm} train features: {Mt.shape[1]}"
        assert Me.shape[1] == 9996, f"{nm} test features: {Me.shape[1]}"

    # Verify global block identity with canonical
    assert np.allclose(Ftrva_global, M0_tr[:, :N_GLOBAL]), \
        "Global block differs from canonical MiniROCKET"

    # === STEP 7: Fit classifiers and evaluate ===
    print("\n[STEP 7] Fitting Ridge classifiers...")
    results = {}
    all_preds = {}

    for name, F_tr, F_te in [
        ("M0_canonical", M0_tr, M0_te),
        ("M1_drtn_conditioned", M1_tr, M1_te),
        ("M2_random_regime", M2_tr, M2_te),
        ("M3_shuffled_regime", M3_tr, M3_te),
    ]:
        print(f"\n  [{name}] Fitting RidgeClassifierCV...")
        t0 = time.time()
        ridge = fit_ridge_classifier(F_tr, ytrva)
        pred_te = ridge.predict(F_te)
        all_preds[name] = pred_te

        test_mf1 = macro_f1(yte, pred_te)
        test_acc = float(accuracy_score(yte, pred_te))
        class_f1s = [round(float(x), 4) for x in f1_score(
            yte, pred_te, average=None, zero_division=0,
            labels=list(range(d["n_classes"]))
        )]
        cm = confusion_matrix(yte, pred_te, labels=list(range(d["n_classes"]))).tolist()

        results[name] = {
            "test_macro_f1": round(test_mf1, 4),
            "accuracy": round(test_acc, 4),
            "class_f1s": class_f1s,
            "confusion_matrix": cm,
            "fit_time_s": round(time.time() - t0, 1),
            "selected_alpha": float(ridge.alpha_),
        }
        print(f"    Test MF1: {test_mf1:.4f}, Accuracy: {test_acc:.4f}, Alpha: {ridge.alpha_:.6f}")

    # === STEP 8: Compute diagnostics ===
    print("\n[STEP 8] Computing diagnostics...")
    m0_mf1 = results["M0_canonical"]["test_macro_f1"]
    deltas = {k: round(results[k]["test_macro_f1"] - m0_mf1, 4)
              for k in results if k != "M0_canonical"}

    pred_m0 = all_preds["M0_canonical"]
    pred_m1 = all_preds["M1_drtn_conditioned"]
    m0_correct = pred_m0 == yte
    m1_correct = pred_m1 == yte
    complementarity = {
        "M0_only_correct": int(np.sum(m0_correct & ~m1_correct)),
        "M1_only_correct": int(np.sum(~m0_correct & m1_correct)),
        "both_correct": int(np.sum(m0_correct & m1_correct)),
        "both_wrong": int(np.sum(~m0_correct & ~m1_correct)),
        "n_samples": int(len(yte)),
    }

    # === STEP 9: Save artifacts ===
    print("\n[STEP 9] Saving artifacts...")
    config = {
        "seed": SEED, "dataset": DATASET,
        "split": {"train": len(Xtr), "val": len(Xva), "test": len(Xte)},
        "T": int(d["L"]), "n_classes": d["n_classes"],
        "n_features_total": 9996,
        "n_global_features": N_GLOBAL,
        "n_heterogeneity_features": N_HETEROGENEITY,
        "minirocket": "aeon MiniRocket(random_state=42)",
        "ridge": "RidgeClassifierCV(alphas=np.logspace(-4,4,20))",
        "drtn": "official seed-42 R5 checkpoint (frozen)",
        "K_regimes": 8, "min_regime_occupancy": 0.01,
    }
    with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    with open(os.path.join(OUT_DIR, "predictions", "predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_index", "true_class", "M0_prediction", "M1_prediction",
                     "M2_prediction", "M3_prediction", "M0_correct", "M1_correct",
                     "M2_correct", "M3_correct"])
        for i in range(len(yte)):
            w.writerow([i, int(yte[i]),
                         int(pred_m0[i]), int(pred_m1[i]),
                         int(all_preds["M2_random_regime"][i]),
                         int(all_preds["M3_shuffled_regime"][i]),
                         int(m0_correct[i]), int(m1_correct[i]),
                         int((pred_m0[i] == yte[i])),
                         int((pred_m1[i] == yte[i]))])

    with open(os.path.join(OUT_DIR, "diagnostics", "feature_statistics.json"), "w") as f:
        json.dump({
            "M0_global_mean": float(Fte_global.mean()),
            "M0_global_std": float(Fte_global.std()),
            "M0_global_nonzero": float((Fte_global != 0).mean()),
            "M1_heterogeneity_mean": float(Fte_het_m1.mean()),
            "M1_heterogeneity_std": float(Fte_het_m1.std()),
            "M1_heterogeneity_nonzero": float((Fte_het_m1 != 0).mean()),
        }, f, indent=2)

    with open(os.path.join(OUT_DIR, "diagnostics", "regime_statistics.json"), "w") as f:
        json.dump({
            "train": compute_regime_occupancy_stats(regimes_tr, K=8),
            "val": compute_regime_occupancy_stats(regimes_va, K=8),
            "test": compute_regime_occupancy_stats(regimes_te, K=8),
        }, f, indent=2)

    with open(os.path.join(OUT_DIR, "diagnostics", "complementarity.json"), "w") as f:
        json.dump(complementarity, f, indent=2)

    m1_mf1 = results["M1_drtn_conditioned"]["test_macro_f1"]
    m3_mf1 = results["M3_shuffled_regime"]["test_macro_f1"]
    if m1_mf1 > m0_mf1 and m1_mf1 > m3_mf1:
        conclusion = ("CASE A: M1 > M0 and M1 > M3. Learned DRTN temporal regimes "
                      "provide useful context for ROCKET activations.")
        verdict = "YES"
    elif m1_mf1 > m0_mf1 and abs(m1_mf1 - m3_mf1) < 0.01:
        conclusion = ("CASE B: M1 > M0 but M1 ~ M3. Regime conditioning may help, "
                      "but not specifically attributable to DRTN's learned segmentation.")
        verdict = "INCONCLUSIVE"
    elif abs(m1_mf1 - m0_mf1) < 0.01:
        conclusion = ("CASE C: M1 ~ M0. Learned regime conditioning does not improve "
                      "MiniROCKET under equal feature budget on Haptics.")
        verdict = "NO"
    else:
        conclusion = ("CASE D: M1 < M0. Regime-conditioned representation is harmful "
                      "under the tested configuration.")
        verdict = "NO"

    report = {
        "title": "DRTN-CONDITIONED MINIROCKET -- HAPTICS SEED 42",
        "scientific_question": (
            "Can a learned discrete temporal regime assignment from DRTN make "
            "MiniROCKET's fixed convolutional representation more informative by "
            "conditioning its temporal pooling on learned regimes, under an equal "
            "9,996-feature budget?"),
        "config": config, "results": results,
        "deltas_vs_canonical": deltas, "complementarity": complementarity,
        "regime_statistics": {"test": compute_regime_occupancy_stats(regimes_te, K=8)},
        "conclusion": conclusion, "verdict": verdict,
    }
    with open(os.path.join(OUT_DIR, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # === STEP 10: Console summary ===
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nM0 (Canonical MiniROCKET):      {results['M0_canonical']['test_macro_f1']:.4f}")
    print(f"M1 (DRTN-conditioned):          {results['M1_drtn_conditioned']['test_macro_f1']:.4f}  "
          f"(delta={deltas['M1_drtn_conditioned']:+.4f})")
    print(f"M2 (Random-regime control):     {results['M2_random_regime']['test_macro_f1']:.4f}  "
          f"(delta={deltas['M2_random_regime']:+.4f})")
    print(f"M3 (Shuffled-regime control):   {results['M3_shuffled_regime']['test_macro_f1']:.4f}  "
          f"(delta={deltas['M3_shuffled_regime']:+.4f})")
    print(f"\nScientific interpretation: {conclusion}")
    print(f"Verdict: {verdict}")
    print("=" * 70)

    return results, report


if __name__ == "__main__":
    main()
