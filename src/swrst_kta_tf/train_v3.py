import torch
import numpy as np
import json
import time
import os
import sys
from pathlib import Path
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

sys.path.insert(0, os.path.dirname(__file__))

from stages.scale_decomposition import scale_decompose
from stages.local_measures import local_measures, compute_bin_edges, global_histogram
from stages.transport_field import transport_field
from stages.cross_scale_correlation import cross_scale_corr
from stages.kta_projection import class_balanced_target_kernel, balanced_kta_projection, normalize_train, normalize_test
from stages.kernel import (rbf_kernel_batched, hellinger_kernel, fuse_kernels_v2,
                           compute_test_kernel_v2, compute_hellinger_sigma, learn_fusion_weights)
from stages.krr import cv_select_lambda, predict
from stages.morphological_features import extract_morphological_features

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'ecg5000_resplit.npz')
    data = np.load(data_path)
    X_train = torch.tensor(data['X_train'], dtype=torch.float32, device=device)
    y_train = torch.tensor(data['y_train'], dtype=torch.long, device=device)
    X_test  = torch.tensor(data['X_test'],  dtype=torch.float32, device=device)
    y_test  = torch.tensor(data['y_test'],  dtype=torch.long,  device=device)
    
    t0 = time.time()
    
    # Stage 1
    X_scales_tr, X_details_tr = scale_decompose(X_train, device)
    X_scales_te, X_details_te = scale_decompose(X_test,  device)
    
    # Upgrade 2: morphological features
    MORPH_tr = extract_morphological_features(X_train)
    MORPH_te = extract_morphological_features(X_test)
    morph_mean = MORPH_tr.mean(0, keepdim=True)
    morph_std  = MORPH_tr.std(0,  keepdim=True) + 1e-8
    MORPH_tr = (MORPH_tr - morph_mean) / morph_std
    MORPH_te = (MORPH_te - morph_mean) / morph_std
    print(f"Stage 1 + Morph done: {time.time()-t0:.1f}s")
    
    # Stage 2
    bin_edges = compute_bin_edges(X_train)
    MU_tr = local_measures(X_scales_tr, bin_edges)
    MU_te = local_measures(X_scales_te, bin_edges)
    print(f"Stage 2 done: {time.time()-t0:.1f}s")
    
    # Stage 3
    R_tr, _ = transport_field(MU_tr, bin_edges)
    R_te, _ = transport_field(MU_te, bin_edges)
    print(f"Stage 3 done: {time.time()-t0:.1f}s")
    
    # Stage 4
    PHI_tr = cross_scale_corr(X_details_tr, X_scales_tr, max_lag=5)
    PHI_te = cross_scale_corr(X_details_te, X_scales_te, max_lag=5)
    NU_tr  = global_histogram(X_scales_tr, bin_edges)
    NU_te  = global_histogram(X_scales_te, bin_edges)
    print(f"Stage 4 done: {time.time()-t0:.1f}s")
    
    # Stage 6: Upgrade A — balanced KTA projection
    R_flat_tr = R_tr.reshape(len(X_train), -1)   # (4000, 8768)
    R_flat_te = R_te.reshape(len(X_test),  -1)
    
    K_y_w = class_balanced_target_kernel(y_train, n_classes=5)
    
    W_star = balanced_kta_projection(
        R_flat_tr, y_train, n_classes=5,
        rank=5, n_oversampling=10
    )  # (8768, 5)
    
    Z_tr = R_flat_tr @ W_star             # (4000, 40)
    z_mean = Z_tr.mean(0, keepdim=True)
    z_std  = Z_tr.std(0,  keepdim=True) + 1e-8
    Z_tr_norm = (Z_tr - z_mean) / z_std
    Z_te_norm = ((R_flat_te @ W_star) - z_mean) / z_std
    print(f"Stage 6 (balanced KTA) done: {time.time()-t0:.1f}s")
    
    # rank sanity check
    eff_rank = torch.linalg.matrix_rank(R_flat_tr[:200] @ W_star).item()
    print(f"Effective rank check on projection: {eff_rank} (should be near {W_star.shape[1]}, not collapsed)")
    
    # Stage 7: four-component kernel with learned fusion
    sigma_proj   = torch.pdist(Z_tr_norm).pow(2).median().item()
    sigma_cross  = torch.pdist(PHI_tr).pow(2).median().item()
    sigma_global = compute_hellinger_sigma(NU_tr)
    sigma_morph  = torch.pdist(MORPH_tr).pow(2).median().item()
    sigmas = {
        'proj': sigma_proj, 'cross': sigma_cross,
        'global': sigma_global, 'morph': sigma_morph
    }
    
    # Parallel CUDA streams for four kernels
    s1 = torch.cuda.Stream(); s2 = torch.cuda.Stream()
    s3 = torch.cuda.Stream(); s4 = torch.cuda.Stream()
    with torch.cuda.stream(s1):
        K_proj   = rbf_kernel_batched(Z_tr_norm, Z_tr_norm, sigma_proj)
    with torch.cuda.stream(s2):
        K_cross  = rbf_kernel_batched(PHI_tr, PHI_tr, sigma_cross)
    with torch.cuda.stream(s3):
        K_global = hellinger_kernel(NU_tr, NU_tr, sigma_global)
    with torch.cuda.stream(s4):
        K_morph  = rbf_kernel_batched(MORPH_tr, MORPH_tr, sigma_morph)
    torch.cuda.synchronize()
    
    fusion_weights = learn_fusion_weights([K_proj, K_cross, K_global, K_morph], K_y_w, n_grid=11)
    print(f"Learned fusion weights: {fusion_weights.tolist()}")

    log_K = sum(w * torch.log(K.clamp(min=1e-30))
                for w, K in zip(fusion_weights, [K_proj, K_cross, K_global, K_morph]))
    K_train = torch.exp(torch.clamp(log_K, min=-30.0))
    K_train = (K_train + K_train.T) / 2.0
    
    # Ensure PSD: add minimal diagonal jitter
    eigvals_min = torch.linalg.eigvalsh(K_train)[0].item()
    if eigvals_min < 0:
        jitter = abs(eigvals_min) + 1e-6
        K_train = K_train + jitter * torch.eye(K_train.shape[0], device=K_train.device)
        # Re-normalise diagonal to ~1.0
        d = torch.sqrt(torch.diag(K_train).clamp(min=1e-8))
        K_train = K_train / (d.unsqueeze(1) * d.unsqueeze(0))
        
    print(f"Stage 7 done: {time.time()-t0:.1f}s")
    
    # Stage 8: Upgrade C — CV selected lambda
    (best_lambda_0, best_beta), cv_score = cv_select_lambda(
        K_train, y_train, n_classes=5,
        lambda_grid=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
        beta_grid=[0.3, 0.5, 0.7, 1.0, 1.5],
    )
    print(f"CV-selected lambda_0={best_lambda_0}, beta={best_beta}, CV macro-F1={cv_score:.3f}")

    n_per_class = torch.bincount(y_train, minlength=5).float()
    lambda_c = best_lambda_0 * (n_per_class / n_per_class.max()).pow(best_beta)
    Lambda_diag = lambda_c[y_train]
    Y_onehot = torch.zeros(len(y_train), 5, device=device)
    Y_onehot.scatter_(1, y_train.unsqueeze(1), 1.0)
    K_reg = K_train + torch.diag(Lambda_diag + 1e-6)
    L = torch.linalg.cholesky(K_reg)
    alpha = torch.cholesky_solve(Y_onehot, L)
    
    print(f"Stage 8 done: {time.time()-t0:.1f}s")
    
    # Test kernel and prediction
    K_test = compute_test_kernel_v2(
        Z_te_norm, Z_tr_norm, PHI_te, PHI_tr,
        NU_te, NU_tr, MORPH_te, MORPH_tr, sigmas
    )
    # the test kernel needs to be fused with the same weights
    K_test_chunks = []
    chunk = 128
    for i in range(0, len(X_test), chunk):
        sl = slice(i, i+chunk)
        Kp = rbf_kernel_batched(Z_te_norm[sl], Z_tr_norm, sigmas['proj'])
        Kc = rbf_kernel_batched(PHI_te[sl], PHI_tr, sigmas['cross'])
        Kg = hellinger_kernel(NU_te[sl], NU_tr, sigmas['global'])
        Km = rbf_kernel_batched(MORPH_te[sl], MORPH_tr, sigmas['morph'])
        
        log_K_test = sum(w * torch.log(K.clamp(min=1e-30))
                for w, K in zip(fusion_weights, [Kp, Kc, Kg, Km]))
        K_test_fused = torch.exp(torch.clamp(log_K_test, min=-30.0))
        K_test_chunks.append(K_test_fused)
    K_test = torch.cat(K_test_chunks, dim=0)

    y_pred = predict(K_test, alpha, y_train=y_train).cpu().numpy()
    y_true = y_test.cpu().numpy()
    
    acc       = accuracy_score(y_true, y_pred)
    macro_f1  = f1_score(y_true, y_pred, average='macro')
    per_class = f1_score(y_true, y_pred, average=None).tolist()
    cm        = confusion_matrix(y_true, y_pred).tolist()
    
    # Validation checks
    checks = {}

    # PSD
    eigvals = torch.linalg.eigvalsh(K_train)
    checks['kernel_psd'] = bool(eigvals.min().item() > -1e-4)

    # Symmetry
    checks['kernel_symmetric'] = bool(
        torch.allclose(K_train, K_train.T, atol=1e-5))

    # Diagonal near 1
    checks['diagonal_near_1'] = bool(
        torch.allclose(K_train.diag(),
                       torch.ones(len(X_train), device=device),
                       atol=1e-3))

    # W_star orthogonality
    WtW = W_star.T @ W_star
    checks['wstar_orthonormal'] = bool(
        torch.allclose(WtW,
                       torch.eye(W_star.shape[1], device=device),
                       atol=1e-3))

    # All classes nonzero recall
    checks['all_classes_nonzero_recall'] = all(v > 0 for v in per_class)
    
    checks['balanced_kernel_rank_ok'] = eff_rank >= 0.8 * W_star.shape[1]
    checks['cv_macro_f1'] = float(cv_score)
    
    results = {
        'Test_Accuracy':       float(acc),
        'Test_Macro_F1':       float(macro_f1),
        'Per_Class_Recall':    per_class,
        'Confusion_Matrix':    cm,
        'Training_Time_Seconds': time.time() - t0,
        'Upgrades_Applied':    [
            'class_balanced_kta_projection',
            'morphological_features',
            'learned_fusion_weights',
            'cv_selected_lambda'
        ],
        'Validation_Checks': checks
    }
    
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
    Path(results_dir).mkdir(exist_ok=True)
    with open(os.path.join(results_dir, 'swrst_kta_tf_v3_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nTest Accuracy : {acc:.4f}")
    print(f"Macro F1      : {macro_f1:.4f}")
    print(f"Per-class F1  : {[f'{v:.3f}' for v in per_class]}")
    print(f"Total time    : {time.time()-t0:.1f}s")

if __name__ == '__main__':
    main()
