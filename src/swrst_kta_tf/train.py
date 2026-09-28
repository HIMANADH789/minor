"""
KTA-TF — Kernel Target Alignment via Transport Field
Main training / evaluation script for ECG5000.
"""
import torch
import numpy as np
import json
import time
import os
import sys
from pathlib import Path
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

# Allow imports from the stages sub-package
sys.path.insert(0, os.path.dirname(__file__))

from stages.scale_decomposition import scale_decompose
from stages.local_measures import local_measures, compute_bin_edges, global_histogram
from stages.transport_field import transport_field
from stages.cross_scale_correlation import cross_scale_corr
from stages.target_kernel import classification_target_kernel
from stages.kta_projection import kta_projection, normalize_train, normalize_test
from stages.kernel import (compute_train_kernel, compute_test_kernel,
                           compute_hellinger_sigma)
from stages.krr import krr_solve, predict, class_conditional_lambda


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = False   # Strict FP32

    # ---- Load data (same split as baselines) ----------------------------
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data',
                             'ecg5000_resplit.npz')
    data = np.load(data_path)
    X_train = torch.tensor(data['X_train'], dtype=torch.float32, device=device)
    y_train = torch.tensor(data['y_train'], dtype=torch.long, device=device)
    X_test  = torch.tensor(data['X_test'],  dtype=torch.float32, device=device)
    y_test  = torch.tensor(data['y_test'],  dtype=torch.long, device=device)
    n_classes = int(y_train.max().item()) + 1
    print(f"Train {X_train.shape}, Test {X_test.shape}, Classes {n_classes}")

    t0 = time.time()

    # ---- Stage 1: Scale decomposition ------------------------------------
    X_scales_tr, X_details_tr = scale_decompose(X_train, device)
    X_scales_te, X_details_te = scale_decompose(X_test, device)
    print(f"Stage 1 (scale decomposition)  : {time.time()-t0:.2f}s")

    # ---- Stage 2: Local measures -----------------------------------------
    bin_edges = compute_bin_edges(X_train)
    MU_tr = local_measures(X_scales_tr, bin_edges)
    MU_te = local_measures(X_scales_te, bin_edges)
    print(f"Stage 2 (local measures)       : {time.time()-t0:.2f}s  "
          f"MU shape {MU_tr.shape}")

    # ---- Stage 3: Exact 1-D OT field ------------------------------------
    R_tr, W_tr = transport_field(MU_tr, bin_edges)
    R_te, W_te = transport_field(MU_te, bin_edges)
    print(f"Stage 3 (transport field)      : {time.time()-t0:.2f}s  "
          f"R shape {R_tr.shape}")

    # ---- Stage 4: Cross-scale correlations --------------------------------
    PHI_tr = cross_scale_corr(X_details_tr, X_scales_tr, max_lag=5)
    PHI_te = cross_scale_corr(X_details_te, X_scales_te, max_lag=5)
    print(f"Stage 4 (cross-scale corr)     : {time.time()-t0:.2f}s  "
          f"PHI shape {PHI_tr.shape}")

    # Global histograms for Hellinger kernel
    NU_tr = global_histogram(X_scales_tr, bin_edges)
    NU_te = global_histogram(X_scales_te, bin_edges)

    # ---- Stage 5: Target kernel ------------------------------------------
    K_y = classification_target_kernel(y_train)
    print(f"Stage 5 (target kernel)        : {time.time()-t0:.2f}s")

    # ---- Stage 6: KTA projection -----------------------------------------
    R_flat_tr = R_tr.reshape(X_train.shape[0], -1)              # (4000, D)
    D = R_flat_tr.shape[1]
    print(f"  R_flat dimension D = {D}")
    W_star = kta_projection(K_y, R_flat_tr, rank=32, n_oversampling=10)
    Z_tr = R_flat_tr @ W_star
    Z_tr_norm, z_mean, z_std = normalize_train(Z_tr)

    R_flat_te = R_te.reshape(X_test.shape[0], -1)
    Z_te_norm = normalize_test(R_flat_te @ W_star, z_mean, z_std)
    print(f"Stage 6 (KTA projection)       : {time.time()-t0:.2f}s  "
          f"Z shape {Z_tr_norm.shape}")

    # ---- Compute bandwidth sigmas from training data ---------------------
    sigma_proj   = torch.pdist(Z_tr_norm).pow(2).median().item()
    sigma_cross  = torch.pdist(PHI_tr).pow(2).median().item()
    sigma_global = compute_hellinger_sigma(NU_tr)
    sigmas = {'proj': sigma_proj, 'cross': sigma_cross, 'global': sigma_global}
    print(f"  Sigmas: proj={sigma_proj:.4f}  cross={sigma_cross:.4f}  "
          f"global={sigma_global:.6f}")

    # ---- Stage 7: Training kernel ----------------------------------------
    K_train = compute_train_kernel(Z_tr_norm, PHI_tr, NU_tr, sigmas)
    print(f"Stage 7 (kernel)               : {time.time()-t0:.2f}s")

    # ---- Stage 8: Kernel Ridge Regression --------------------------------
    Y_onehot = torch.zeros(len(y_train), n_classes, device=device)
    Y_onehot.scatter_(1, y_train.unsqueeze(1), 1.0)
    Lambda = class_conditional_lambda(y_train, device)
    alpha = krr_solve(K_train, Y_onehot, Lambda)
    print(f"Stage 8 (KRR solve)            : {time.time()-t0:.2f}s")

    # ---- Test prediction -------------------------------------------------
    K_test = compute_test_kernel(Z_te_norm, Z_tr_norm,
                                 PHI_te, PHI_tr,
                                 NU_te, NU_tr, sigmas)
    y_pred = predict(K_test, alpha, y_train=y_train).cpu().numpy()
    y_true = y_test.cpu().numpy()
    total_time = time.time() - t0

    # ---- Metrics ---------------------------------------------------------
    acc      = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    per_cls  = f1_score(y_true, y_pred, average=None).tolist()
    cm       = confusion_matrix(y_true, y_pred).tolist()

    # ---- Validation checks -----------------------------------------------
    checks = {}

    eigvals = torch.linalg.eigvalsh(K_train)
    checks['PSD'] = bool(eigvals.min() > -1e-4)
    print(f"  Check PSD: min eigval = {eigvals.min().item():.6e}  "
          f"{'PASS' if checks['PSD'] else 'FAIL'}")

    checks['Symmetric'] = bool(torch.allclose(K_train, K_train.T, atol=1e-5))
    print(f"  Check Symmetry: {'PASS' if checks['Symmetric'] else 'FAIL'}")

    diag_ok = torch.allclose(K_train.diag(),
                             torch.ones(K_train.shape[0], device=device),
                             atol=1e-3)
    checks['Diagonal_Ones'] = bool(diag_ok)
    print(f"  Check Diagonal ~1: {'PASS' if diag_ok else 'FAIL'}  "
          f"(range [{K_train.diag().min():.4f}, {K_train.diag().max():.4f}])")

    WtW = W_star.T @ W_star
    orth_ok = torch.allclose(WtW, torch.eye(32, device=device), atol=1e-4)
    checks['W_star_Orthogonal'] = bool(orth_ok)
    print(f"  Check W* orthogonal: {'PASS' if orth_ok else 'FAIL'}")

    checks['Nonzero_Recall'] = all(r > 0 for r in per_cls)
    print(f"  Check non-zero recall: {'PASS' if checks['Nonzero_Recall'] else 'FAIL'}  "
          f"{per_cls}")

    # ---- Save results ----------------------------------------------------
    results = {
        'Test_Accuracy': float(acc),
        'Test_Macro_F1': float(macro_f1),
        'Per_Class_Recall': per_cls,
        'Confusion_Matrix': cm,
        'Validation_Checks': checks,
        'Training_Time_Seconds': total_time,
    }

    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
    Path(results_dir).mkdir(exist_ok=True)
    out_path = os.path.join(results_dir, 'swrst_kta_tf_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\n{'='*50}")
    print(f"Test Accuracy : {acc:.4f}")
    print(f"Macro F1      : {macro_f1:.4f}")
    print(f"Total time    : {total_time:.1f}s")
    print(f"Results saved : {out_path}")


if __name__ == '__main__':
    main()
