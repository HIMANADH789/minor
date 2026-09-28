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
from stages.kta_projection import (class_balanced_target_kernel, per_scale_balanced_kta,
                                   residual_pca_embedding_multiscale, confusion_graded_target_kernel)
from stages.kernel import (rbf_kernel_batched, hellinger_kernel, compute_hellinger_sigma,
                           learn_fusion_weights_grad)
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
    
    MORPH_tr = extract_morphological_features(X_train)
    MORPH_te = extract_morphological_features(X_test)
    morph_mean = MORPH_tr.mean(0, keepdim=True)
    morph_std  = MORPH_tr.std(0,  keepdim=True) + 1e-8
    MORPH_tr = (MORPH_tr - morph_mean) / morph_std
    MORPH_te = (MORPH_te - morph_mean) / morph_std
    print(f"Stage 1 + Morph done: {time.time()-t0:.1f}s")
    
    # Stage 2
    bin_edges = compute_bin_edges(X_train)
    MU_tr, MU_list_tr = local_measures(X_scales_tr, bin_edges)
    MU_te, MU_list_te = local_measures(X_scales_te, bin_edges)
    print(f"Stage 2 done: {time.time()-t0:.1f}s")
    
    # Stage 3
    R_tr, _, R_by_scale_tr = transport_field(MU_tr, MU_list_tr, bin_edges)
    R_te, _, R_by_scale_te = transport_field(MU_te, MU_list_te, bin_edges)
    print(f"Stage 3 done: {time.time()-t0:.1f}s")
    
    # Stage 4
    PHI_tr = cross_scale_corr(X_details_tr, X_scales_tr, max_lag=5)
    PHI_te = cross_scale_corr(X_details_te, X_scales_te, max_lag=5)
    NU_tr  = global_histogram(X_scales_tr, bin_edges)
    NU_te  = global_histogram(X_scales_te, bin_edges)
    print(f"Stage 4 done: {time.time()-t0:.1f}s")

    # Common un-aligned kernels
    sigma_cross  = torch.pdist(PHI_tr).pow(2).median().item()
    sigma_global = compute_hellinger_sigma(NU_tr)
    sigma_morph  = torch.pdist(MORPH_tr).pow(2).median().item()
    
    K_cross_tr = rbf_kernel_batched(PHI_tr, PHI_tr, sigma_cross)
    K_global_tr = hellinger_kernel(NU_tr, NU_tr, sigma_global)
    K_morph_tr = rbf_kernel_batched(MORPH_tr, MORPH_tr, sigma_morph)
    
    # Calculate batched to prevent memory issues with fully expanded distances
    K_cross_te_chunks = []
    chunk = 500
    for i in range(0, len(X_test), chunk):
        sl = slice(i, i+chunk)
        K_cross_te_chunks.append(rbf_kernel_batched(PHI_te[sl], PHI_tr, sigma_cross))
    K_cross_te = torch.cat(K_cross_te_chunks, dim=0)

    K_global_te_chunks = []
    for i in range(0, len(X_test), chunk):
        sl = slice(i, i+chunk)
        K_global_te_chunks.append(hellinger_kernel(NU_te[sl], NU_tr, sigma_global))
    K_global_te = torch.cat(K_global_te_chunks, dim=0)

    K_morph_te_chunks = []
    for i in range(0, len(X_test), chunk):
        sl = slice(i, i+chunk)
        K_morph_te_chunks.append(rbf_kernel_batched(MORPH_te[sl], MORPH_tr, sigma_morph))
    K_morph_te = torch.cat(K_morph_te_chunks, dim=0)

    def compute_all_kernels(K_target):
        # Upgrade 1: Per-Scale KTA
        W_list = per_scale_balanced_kta(R_by_scale_tr, K_target, n_classes=5, n_oversampling=10)
        
        M_s = (K_target - K_target.mean(0, keepdim=True) - K_target.mean(1, keepdim=True) + K_target.mean()) @ R_by_scale_tr[3]
        eff_rank = torch.linalg.matrix_rank(M_s[:200]).item()
        
        # Upgrade 2: Residual PCA
        W_pca = residual_pca_embedding_multiscale(R_by_scale_tr, W_list, rank_residual=16, n_oversampling=10)
        
        def embed_multiscale(R_by_s):
            Z_scales = []
            residuals = []
            for R_s, W_s in zip(R_by_s, W_list):
                Z_s = R_s @ W_s
                Z_scales.append(Z_s)
                R_hat_s = Z_s @ W_s.T
                residuals.append(R_s - R_hat_s)
            Z_scales = torch.stack(Z_scales, dim=1) # (N, S, C)
            R_res = torch.cat(residuals, dim=1)
            Z_pca = R_res @ W_pca
            return Z_scales, Z_pca

        Z_scales_tr, Z_pca_tr = embed_multiscale(R_by_scale_tr)
        Z_scales_te, Z_pca_te = embed_multiscale(R_by_scale_te)
        
        S_num = len(R_by_scale_tr)
        K_list_tr = []
        K_list_te = []
        
        for s in range(S_num):
            Z_s_tr = Z_scales_tr[:, s, :]
            Z_s_te = Z_scales_te[:, s, :]
            mu = Z_s_tr.mean(0, keepdim=True)
            std = Z_s_tr.std(0, keepdim=True) + 1e-8
            Z_s_tr = (Z_s_tr - mu) / std
            Z_s_te = (Z_s_te - mu) / std
            
            sigma_s = torch.pdist(Z_s_tr).pow(2).median().item()
            if sigma_s < 1e-6:
                sigma_s = 1.0  # fallback for empty scales
            
            K_list_tr.append(rbf_kernel_batched(Z_s_tr, Z_s_tr, sigma_s))
            
            # batch for te
            K_te_chunks = []
            for i in range(0, len(X_test), chunk):
                sl = slice(i, i+chunk)
                K_te_chunks.append(rbf_kernel_batched(Z_s_te[sl], Z_s_tr, sigma_s))
            K_list_te.append(torch.cat(K_te_chunks, dim=0))
            
        mu_p = Z_pca_tr.mean(0, keepdim=True)
        std_p = Z_pca_tr.std(0, keepdim=True) + 1e-8
        Z_pca_tr = (Z_pca_tr - mu_p) / std_p
        Z_pca_te = (Z_pca_te - mu_p) / std_p
        sigma_pca = torch.pdist(Z_pca_tr).pow(2).median().item()
        if sigma_pca < 1e-6:
            sigma_pca = 1.0

        
        K_list_tr.append(rbf_kernel_batched(Z_pca_tr, Z_pca_tr, sigma_pca))
        
        K_pca_te_chunks = []
        for i in range(0, len(X_test), chunk):
            sl = slice(i, i+chunk)
            K_pca_te_chunks.append(rbf_kernel_batched(Z_pca_te[sl], Z_pca_tr, sigma_pca))
        K_list_te.append(torch.cat(K_pca_te_chunks, dim=0))
        
        K_list_tr.extend([K_cross_tr, K_global_tr, K_morph_tr])
        K_list_te.extend([K_cross_te, K_global_te, K_morph_te])
        
        return K_list_tr, K_list_te, eff_rank

    def make_psd(K):
        eigvals_min = torch.linalg.eigvalsh(K)[0].item()
        if eigvals_min < 0:
            jitter = abs(eigvals_min) + 1e-6
            K = K + jitter * torch.eye(K.shape[0], device=K.device)
            d = torch.sqrt(torch.diag(K).clamp(min=1e-8))
            K = K / (d.unsqueeze(1) * d.unsqueeze(0))
        return K

    print("\n--- PASS 1: Base Target Kernel ---")
    K_y_w_1 = class_balanced_target_kernel(y_train, n_classes=5)
    K_list_tr_1, K_list_te_1, eff_rank_1 = compute_all_kernels(K_y_w_1)
    print(f"Pass 1 Effective rank check on projection (Scale 3): {eff_rank_1} (should be near 5)")
    
    fusion_weights_1 = learn_fusion_weights_grad(K_list_tr_1, K_y_w_1, n_steps=300, lr=0.5)
    print(f"Pass 1 Learned fusion weights:\n{np.round(fusion_weights_1.cpu().numpy(), 3)}")
    
    log_K_tr_1 = sum(w * torch.log(K.clamp(min=1e-30)) for w, K in zip(fusion_weights_1, K_list_tr_1))
    K_train_1 = torch.exp(torch.clamp(log_K_tr_1, min=-30.0))
    K_train_1 = (K_train_1 + K_train_1.T) / 2.0
    K_train_1 = make_psd(K_train_1)
    
    (best_lambda_0_1, best_beta_1), cv_score_1, cv_cm = cv_select_lambda(
        K_train_1, y_train, n_classes=5,
        lambda_grid=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
        beta_grid=[0.3, 0.5, 0.7, 1.0, 1.5],
    )
    print(f"Pass 1 CV macro-F1: {cv_score_1:.3f}")
    
    print("\n--- PASS 2: Confusion-Graded Target Kernel ---")
    # Upgrade 3: Confusion-Graded Target Kernel
    K_y_soft = confusion_graded_target_kernel(y_train, cv_cm, n_classes=5)
    K_list_tr_2, K_list_te_2, eff_rank_2 = compute_all_kernels(K_y_soft)
    print(f"Pass 2 Effective rank check on projection (Scale 0): {eff_rank_2} (should be near 5)")
    
    fusion_weights_2 = learn_fusion_weights_grad(K_list_tr_2, K_y_soft, n_steps=300, lr=0.5)
    print(f"Pass 2 Learned fusion weights:\n{np.round(fusion_weights_2.cpu().numpy(), 3)}")
    
    log_K_tr_2 = sum(w * torch.log(K.clamp(min=1e-30)) for w, K in zip(fusion_weights_2, K_list_tr_2))
    K_train_2 = torch.exp(torch.clamp(log_K_tr_2, min=-30.0))
    K_train_2 = (K_train_2 + K_train_2.T) / 2.0
    K_train_2 = make_psd(K_train_2)
    
    (best_lambda_0_2, best_beta_2), cv_score_2, _ = cv_select_lambda(
        K_train_2, y_train, n_classes=5,
        lambda_grid=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
        beta_grid=[0.3, 0.5, 0.7, 1.0, 1.5],
    )
    print(f"Pass 2 CV macro-F1: {cv_score_2:.3f}")

    # Final Train
    n_per_class = torch.bincount(y_train, minlength=5).float()
    lambda_c = best_lambda_0_2 * (n_per_class / n_per_class.max()).pow(best_beta_2)
    Lambda_diag = lambda_c[y_train]
    Y_onehot = torch.zeros(len(y_train), 5, device=device)
    Y_onehot.scatter_(1, y_train.unsqueeze(1), 1.0)
    K_reg = K_train_2 + torch.diag(Lambda_diag + 1e-6)
    
    # Ensure PSD for final solve just in case
    try:
        L = torch.linalg.cholesky(K_reg)
        alpha = torch.cholesky_solve(Y_onehot, L)
    except torch.linalg.LinAlgError:
        K_reg = K_reg + torch.eye(K_train_2.shape[0], device=device) * 1e-3
        L = torch.linalg.cholesky(K_reg)
        alpha = torch.cholesky_solve(Y_onehot, L)
    
    log_K_te_2 = sum(w * torch.log(K.clamp(min=1e-30)) for w, K in zip(fusion_weights_2, K_list_te_2))
    K_test = torch.exp(torch.clamp(log_K_te_2, min=-30.0))
    
    y_pred = predict(K_test, alpha, y_train=y_train).cpu().numpy()
    y_true = y_test.cpu().numpy()
    
    acc       = accuracy_score(y_true, y_pred)
    macro_f1  = f1_score(y_true, y_pred, average='macro')
    per_class = f1_score(y_true, y_pred, average=None).tolist()
    cm        = confusion_matrix(y_true, y_pred).tolist()
    
    # Validation checks
    checks = {}
    eigvals = torch.linalg.eigvalsh(K_train_2)
    checks['kernel_psd'] = bool(eigvals.min().item() > -1e-4)
    checks['kernel_symmetric'] = bool(torch.allclose(K_train_2, K_train_2.T, atol=1e-5))
    checks['diagonal_near_1'] = bool(torch.allclose(K_train_2.diag(), torch.ones(len(X_train), device=device), atol=1e-3))
    checks['all_classes_nonzero_recall'] = all(v > 0 for v in per_class)
    checks['balanced_kernel_rank_ok'] = eff_rank_1 <= 6 and eff_rank_2 <= 6
    
    results = {
        'Test_Accuracy':       float(acc),
        'Test_Macro_F1':       float(macro_f1),
        'Per_Class_Recall':    per_class,
        'Confusion_Matrix':    cm,
        'Training_Time_Seconds': time.time() - t0,
        'Upgrades_Applied': [
            'per_scale_balanced_kta',
            'residual_pca',
            'confusion_graded_target_kernel'
        ],
        'Validation_Checks': checks
    }
    
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
    Path(results_dir).mkdir(exist_ok=True)
    with open(os.path.join(results_dir, 'swrst_kta_tf_v4_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nTest Accuracy : {acc:.4f}")
    print(f"Macro F1      : {macro_f1:.4f}")
    print(f"Per-class F1  : {[f'{v:.3f}' for v in per_class]}")
    print(f"Total time    : {time.time()-t0:.1f}s")

if __name__ == '__main__':
    main()
