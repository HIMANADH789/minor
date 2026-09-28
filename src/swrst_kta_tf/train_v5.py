import torch
import numpy as np
import json
import time
import os
import sys
import math
from pathlib import Path
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

sys.path.insert(0, os.path.dirname(__file__))

from stages.scale_decomposition import scale_decompose
from stages.local_measures import local_measures, compute_bin_edges, global_histogram
from stages.transport_field import transport_field
from stages.cross_scale_correlation import cross_scale_corr
from stages.kta_projection import (svd_once, ridge_shrink_and_align, residual_pca_embedding_multiscale)
from stages.kernel import (rbf_kernel_batched, hellinger_kernel, compute_hellinger_sigma)
from stages.krr import loocv_loss_balanced, learn_fusion_and_lambda_joint, predict
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
    
    chunk = 500
    K_cross_te = torch.cat([rbf_kernel_batched(PHI_te[i:i+chunk], PHI_tr, sigma_cross) for i in range(0, len(X_test), chunk)], dim=0)
    K_global_te = torch.cat([hellinger_kernel(NU_te[i:i+chunk], NU_tr, sigma_global) for i in range(0, len(X_test), chunk)], dim=0)
    K_morph_te = torch.cat([rbf_kernel_batched(MORPH_te[i:i+chunk], MORPH_tr, sigma_morph) for i in range(0, len(X_test), chunk)], dim=0)

    # Onehot labels for LOOCV loss
    Y_onehot = torch.zeros(len(y_train), 5, device=device)
    Y_onehot.scatter_(1, y_train.unsqueeze(1), 1.0)

    # Fix 1 & 2: Grid search for lam per scale
    W_list = []
    print("\n--- Per-Scale Ridge-Regularized KTA ---")
    tau_fixed = 5.0  # Initial tau for grid search
    for s, R_s in enumerate(R_by_scale_tr):
        U, Sigma, Vt = svd_once(R_s, q=800)
        
        if len(Sigma) == 0:
            print(f"Scale {s}: completely empty (all singular values 0). Skipping KTA.")
            W_list.append(torch.zeros(R_s.shape[1], 5, device=device))
            continue
            
        top_sq = Sigma[0].item()**2
        fracs = torch.logspace(-4, 2, 7)
        lams = (fracs * top_sq).tolist()
        
        best_lam = None
        best_loss = float('inf')
        best_W_s = None
        
        for lam in lams:
            W_s_cand = ridge_shrink_and_align(U, Sigma, Vt, y_train, 5, lam, tau=tau_fixed, n_oversampling=5)
            Z_s_cand = R_s @ W_s_cand
            mu_cand = Z_s_cand.mean(0, keepdim=True)
            std_cand = Z_s_cand.std(0, keepdim=True) + 1e-8
            Z_s_cand = (Z_s_cand - mu_cand) / std_cand
            sigma_s = torch.pdist(Z_s_cand).pow(2).median().item()
            if sigma_s < 1e-6:
                sigma_s = 1.0
            
            K_s_cand = rbf_kernel_batched(Z_s_cand, Z_s_cand, sigma_s)
            
            # Isolated LOOCV loss
            baseline_ridge = 1e-3 * torch.diag(K_s_cand).mean()
            Lambda_diag = torch.full((len(y_train),), baseline_ridge.item(), device=device)
            try:
                from stages.krr import loocv_loss_balanced
                loss = loocv_loss_balanced(K_s_cand, Y_onehot, Lambda_diag, y_train, 5)
            except torch.linalg.LinAlgError:
                loss = torch.tensor(float('inf'))
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_lam = lam
                best_W_s = W_s_cand
                
        W_list.append(best_W_s)
        print(f"Scale {s}: raw_dim={R_s.shape[1]}, effective_rank={len(Sigma)}, selected_lam={best_lam:.2e} (rel={best_lam/top_sq:.1e})")

    # Fix 3: Residual PCA
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
    
    K_list_tr = []
    K_list_te = []
    
    for s in range(len(R_by_scale_tr)):
        Z_s_tr = Z_scales_tr[:, s, :]
        Z_s_te = Z_scales_te[:, s, :]
        mu = Z_s_tr.mean(0, keepdim=True)
        std = Z_s_tr.std(0, keepdim=True) + 1e-8
        Z_s_tr = (Z_s_tr - mu) / std
        Z_s_te = (Z_s_te - mu) / std
        
        sigma_s = torch.pdist(Z_s_tr).pow(2).median().item()
        if sigma_s < 1e-6:
            sigma_s = 1.0
        
        K_list_tr.append(rbf_kernel_batched(Z_s_tr, Z_s_tr, sigma_s))
        K_list_te.append(torch.cat([rbf_kernel_batched(Z_s_te[i:i+chunk], Z_s_tr, sigma_s) for i in range(0, len(X_test), chunk)], dim=0))
        
    mu_p = Z_pca_tr.mean(0, keepdim=True)
    std_p = Z_pca_tr.std(0, keepdim=True) + 1e-8
    Z_pca_tr = (Z_pca_tr - mu_p) / std_p
    Z_pca_te = (Z_pca_te - mu_p) / std_p
    sigma_pca = torch.pdist(Z_pca_tr).pow(2).median().item()
    if sigma_pca < 1e-6:
        sigma_pca = 1.0
        
    K_list_tr.append(rbf_kernel_batched(Z_pca_tr, Z_pca_tr, sigma_pca))
    K_list_te.append(torch.cat([rbf_kernel_batched(Z_pca_te[i:i+chunk], Z_pca_tr, sigma_pca) for i in range(0, len(X_test), chunk)], dim=0))
    
    K_list_tr.extend([K_cross_tr, K_global_tr, K_morph_tr])
    K_list_te.extend([K_cross_te, K_global_te, K_morph_te])
    
    print("\n--- Joint LOOCV Optimization ---")
    fusion_weights, best_lambda0, best_beta, best_tau = learn_fusion_and_lambda_joint(K_list_tr, Y_onehot, y_train, 5, n_steps=300, lr=0.05)
    print(f"Learned fusion weights:\n{np.round(fusion_weights.cpu().numpy(), 3)}")
    print(f"Learned lambda_0: {best_lambda0:.2e}, beta (softplus): {best_beta:.3f}, tau: {best_tau:.3f}")
    
    log_K_tr = sum(w * torch.log(K.clamp(min=1e-30)) for w, K in zip(fusion_weights, K_list_tr))
    K_train = torch.exp(torch.clamp(log_K_tr, min=-30.0))
    K_train = (K_train + K_train.T) / 2.0
    
    # Final Train
    n_per_class = torch.bincount(y_train, minlength=5).float()
    lambda_c = best_lambda0 * (n_per_class / n_per_class.max()).pow(best_beta)
    Lambda_diag = lambda_c[y_train]
    K_reg = K_train + torch.diag(Lambda_diag + 1e-6)
    
    # Ensure PSD for final solve just in case
    try:
        L = torch.linalg.cholesky(K_reg)
        alpha = torch.cholesky_solve(Y_onehot, L)
    except torch.linalg.LinAlgError:
        K_reg = K_reg + torch.eye(K_train.shape[0], device=device) * 1e-3
        L = torch.linalg.cholesky(K_reg)
        alpha = torch.cholesky_solve(Y_onehot, L)
        
    # Effective DF
    K_inv = torch.cholesky_inverse(L)
    H_diag = (K_train * K_inv).sum(dim=1)
    effective_df = H_diag.sum().item()
    print(f"\nEffective degrees of freedom: {effective_df:.1f} (N={len(y_train)}) "
          f"-- should be well below N")
    
    log_K_te = sum(w * torch.log(K.clamp(min=1e-30)) for w, K in zip(fusion_weights, K_list_te))
    K_test = torch.exp(torch.clamp(log_K_te, min=-30.0))
    
    y_pred = predict(K_test, alpha, y_train=y_train).cpu().numpy()
    y_true = y_test.cpu().numpy()
    
    acc       = accuracy_score(y_true, y_pred)
    macro_f1  = f1_score(y_true, y_pred, average='macro')
    per_class = f1_score(y_true, y_pred, average=None).tolist()
    cm        = confusion_matrix(y_true, y_pred).tolist()
    
    results = {
        'Test_Accuracy':       float(acc),
        'Test_Macro_F1':       float(macro_f1),
        'Per_Class_Recall':    per_class,
        'Confusion_Matrix':    cm,
        'Training_Time_Seconds': time.time() - t0,
        'Upgrades_Applied': [
            'ridge_regularized_discriminant_kta',
            'loocv_joint_fusion_and_lambda',
            'residual_pca',
        ],
        'Diagnostics': {
            'Effective_DF': effective_df
        }
    }
    
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
    Path(results_dir).mkdir(exist_ok=True)
    with open(os.path.join(results_dir, 'swrst_kta_tf_v5_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nTest Accuracy : {acc:.4f}")
    print(f"Macro F1      : {macro_f1:.4f}")
    print(f"Per-class F1  : {[f'{v:.3f}' for v in per_class]}")
    print(f"Total time    : {time.time()-t0:.1f}s")

if __name__ == '__main__':
    main()
