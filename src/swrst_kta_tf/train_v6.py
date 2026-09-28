import torch
import numpy as np
import json
import time
import os
import sys
from pathlib import Path
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))

from stages.scale_decomposition import scale_decompose
from stages.local_measures import local_measures, compute_bin_edges, global_histogram
from stages.transport_field import transport_field
from stages.cross_scale_correlation import cross_scale_corr
from stages.kernel import (rbf_kernel_batched, hellinger_kernel, compute_hellinger_sigma)
from stages.morphological_features import extract_morphological_features

def pool_scale_descriptor(R_s: torch.Tensor) -> torch.Tensor:
    N = R_s.shape[0]
    flat = R_s.reshape(N, R_s.shape[1], -1)          # (N, T_s, K*K)
    mean_feat = flat.mean(dim=1)                       # (N, K*K)
    if R_s.shape[1] > 1:
        std_feat = flat.std(dim=1, unbiased=False)
    else:
        std_feat = torch.zeros_like(mean_feat)
    return torch.cat([mean_feat, std_feat], dim=1)      # (N, 2*K*K)

def rbf_kernel_median(A: torch.Tensor, B: torch.Tensor, sigma=None):
    d2 = torch.cdist(A, B) ** 2
    if sigma is None:
        mask = d2 > 0
        if mask.any():
            sigma = d2[mask].median().item()
        else:
            sigma = 1.0
        if sigma < 1e-6:
            sigma = 1.0
    return torch.exp(-d2 / (sigma + 1e-8)), sigma

def class_balanced_target_kernel(y_train: torch.Tensor, n_classes: int) -> torch.Tensor:
    n_per_class = torch.bincount(y_train, minlength=n_classes).float()
    w = 1.0 / n_per_class[y_train]
    same_class = (y_train.unsqueeze(0) == y_train.unsqueeze(1)).float()
    return same_class * w.unsqueeze(0) * w.unsqueeze(1)

def alignment_score(K: torch.Tensor, K_y_w: torch.Tensor) -> float:
    return ((K * K_y_w).sum() / (K.norm() * K_y_w.norm() + 1e-12)).item()

def fixed_alignment_weights(K_list, K_y_w):
    scores = torch.tensor([alignment_score(K, K_y_w) for K in K_list])
    scores = torch.nan_to_num(scores, nan=0.0)
    scores = scores.clamp(min=0)
    if scores.sum() == 0:
        scores = torch.ones_like(scores)
    return scores / scores.sum()

def fuse(K_list, weights):
    log_K = sum(w * torch.log(K.clamp(min=1e-30)) for w, K in zip(weights, K_list))
    return torch.exp(torch.clamp(log_K, min=-30.0))

def fit_predict(K_train, K_valtest, Y_onehot_train, y_train, lambda_0, beta, n_classes, return_H=False):
    n_per_class = torch.bincount(y_train, minlength=n_classes).float()
    lambda_c = lambda_0 * (n_per_class / n_per_class.max()).pow(beta)
    Lambda_diag = lambda_c[y_train]
    K_reg = K_train + torch.diag(Lambda_diag + 1e-6)
    try:
        L = torch.linalg.cholesky(K_reg)
    except torch.linalg.LinAlgError:
        try:
            K_reg = K_reg + torch.eye(K_train.shape[0], device=K_train.device) * 1e-3
            L = torch.linalg.cholesky(K_reg)
        except torch.linalg.LinAlgError:
            K_reg = K_reg + torch.eye(K_train.shape[0], device=K_train.device) * 1e-1
            L = torch.linalg.cholesky(K_reg)
    alpha = torch.cholesky_solve(Y_onehot_train, L)
    preds = K_valtest @ alpha
    
    if return_H:
        K_inv = torch.cholesky_inverse(L)
        H_diag = (K_train * K_inv).sum(dim=1)
        effective_df = H_diag.sum().item()
        return preds, effective_df
    return preds

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'ecg5000_resplit.npz')
    data = np.load(data_path)
    X_train_full = data['X_train']
    y_train_full = data['y_train']
    X_test  = torch.tensor(data['X_test'],  dtype=torch.float32, device=device)
    y_test  = torch.tensor(data['y_test'],  dtype=torch.long,  device=device)
    
    X_tr_np, X_val_np, y_tr_np, y_val_np = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )
    
    X_train = torch.tensor(X_tr_np, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_tr_np, dtype=torch.long, device=device)
    X_val   = torch.tensor(X_val_np, dtype=torch.float32, device=device)
    y_val   = torch.tensor(y_val_np, dtype=torch.long, device=device)
    
    X_train_all = torch.tensor(X_train_full, dtype=torch.float32, device=device)
    y_train_all = torch.tensor(y_train_full, dtype=torch.long, device=device)
    
    t0 = time.time()
    
    def extract_features(X):
        X_scales, X_details = scale_decompose(X, device)
        MORPH = extract_morphological_features(X)
        return X_scales, X_details, MORPH
        
    X_scales_tr, X_details_tr, MORPH_tr = extract_features(X_train)
    X_scales_val, X_details_val, MORPH_val = extract_features(X_val)
    X_scales_all, X_details_all, MORPH_all = extract_features(X_train_all)
    X_scales_te, X_details_te, MORPH_te = extract_features(X_test)
    
    bin_edges_tr = compute_bin_edges(X_train)
    MU_tr, MU_list_tr = local_measures(X_scales_tr, bin_edges_tr)
    MU_val, MU_list_val = local_measures(X_scales_val, bin_edges_tr)
    
    bin_edges_all = compute_bin_edges(X_train_all)
    MU_all, MU_list_all = local_measures(X_scales_all, bin_edges_all)
    MU_te, MU_list_te = local_measures(X_scales_te, bin_edges_all)
    
    R_tr, _, R_by_scale_tr = transport_field(MU_tr, MU_list_tr, bin_edges_tr)
    R_val, _, R_by_scale_val = transport_field(MU_val, MU_list_val, bin_edges_tr)
    
    R_all, _, R_by_scale_all = transport_field(MU_all, MU_list_all, bin_edges_all)
    R_te, _, R_by_scale_te = transport_field(MU_te, MU_list_te, bin_edges_all)
    
    PHI_tr = cross_scale_corr(X_details_tr, X_scales_tr, max_lag=5)
    PHI_val = cross_scale_corr(X_details_val, X_scales_val, max_lag=5)
    NU_tr  = global_histogram(X_scales_tr, bin_edges_tr)
    NU_val  = global_histogram(X_scales_val, bin_edges_tr)
    
    PHI_all = cross_scale_corr(X_details_all, X_scales_all, max_lag=5)
    PHI_te = cross_scale_corr(X_details_te, X_scales_te, max_lag=5)
    NU_all  = global_histogram(X_scales_all, bin_edges_all)
    NU_te  = global_histogram(X_scales_te, bin_edges_all)
    
    morph_mean_tr = MORPH_tr.mean(0, keepdim=True)
    morph_std_tr  = MORPH_tr.std(0,  keepdim=True) + 1e-8
    MORPH_tr = (MORPH_tr - morph_mean_tr) / morph_std_tr
    MORPH_val = (MORPH_val - morph_mean_tr) / morph_std_tr
    
    morph_mean_all = MORPH_all.mean(0, keepdim=True)
    morph_std_all  = MORPH_all.std(0,  keepdim=True) + 1e-8
    MORPH_all = (MORPH_all - morph_mean_all) / morph_std_all
    MORPH_te = (MORPH_te - morph_mean_all) / morph_std_all

    print(f"Features extracted: {time.time()-t0:.1f}s")
    
    pool_tr = [pool_scale_descriptor(R_s) for R_s in R_by_scale_tr]
    pool_val = [pool_scale_descriptor(R_s) for R_s in R_by_scale_val]
    pool_all = [pool_scale_descriptor(R_s) for R_s in R_by_scale_all]
    pool_te = [pool_scale_descriptor(R_s) for R_s in R_by_scale_te]

    def compute_kernel_list(pool_A, pool_B, PHI_A, PHI_B, NU_A, NU_B, MORPH_A, MORPH_B, sigmas=None):
        K_list = []
        new_sigmas = {}
        for s in range(len(pool_A)):
            sigma_s = sigmas.get(f'pool_{s}') if sigmas else None
            K_s, sig_s = rbf_kernel_median(pool_A[s], pool_B[s], sigma_s)
            K_list.append(K_s)
            if not sigmas: new_sigmas[f'pool_{s}'] = sig_s
            
        sig_cross = sigmas.get('cross') if sigmas else None
        K_cross, sig_cross = rbf_kernel_median(PHI_A, PHI_B, sig_cross)
        K_list.append(K_cross)
        if not sigmas: new_sigmas['cross'] = sig_cross
        
        sig_global = sigmas.get('global') if sigmas else compute_hellinger_sigma(NU_B)
        if not sigmas: new_sigmas['global'] = sig_global
        K_global = hellinger_kernel(NU_A, NU_B, sig_global)
        K_list.append(K_global)
        
        sig_morph = sigmas.get('morph') if sigmas else None
        K_morph, sig_morph = rbf_kernel_median(MORPH_A, MORPH_B, sig_morph)
        K_list.append(K_morph)
        if not sigmas: new_sigmas['morph'] = sig_morph
        
        return K_list, new_sigmas

    K_list_tr, sigmas_tr = compute_kernel_list(pool_tr, pool_tr, PHI_tr, PHI_tr, NU_tr, NU_tr, MORPH_tr, MORPH_tr)
    K_list_val, _ = compute_kernel_list(pool_val, pool_tr, PHI_val, PHI_tr, NU_val, NU_tr, MORPH_val, MORPH_tr, sigmas_tr)

    K_y_w_tr = class_balanced_target_kernel(y_train, 5)
    weights = fixed_alignment_weights(K_list_tr, K_y_w_tr)
    print(f"Computed Alignment Weights: {np.round(weights.cpu().numpy(), 3)}")
    
    K_train_fused = fuse(K_list_tr, weights)
    K_train_fused = (K_train_fused + K_train_fused.T) / 2.0
    K_val_fused = fuse(K_list_val, weights)
    
    Y_onehot_tr = torch.zeros(len(y_train), 5, device=device)
    Y_onehot_tr.scatter_(1, y_train.unsqueeze(1), 1.0)
    
    best_f1, best_params = -1, None
    for lambda_0 in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        for beta in [0.3, 0.5, 0.7, 1.0]:
            try:
                scores_val = fit_predict(K_train_fused, K_val_fused, Y_onehot_tr, y_train, lambda_0, beta, 5)
                f1 = f1_score(y_val.cpu(), scores_val.argmax(-1).cpu(), average='macro')
                if f1 > best_f1:
                    best_f1, best_params = f1, (lambda_0, beta)
            except torch.linalg.LinAlgError:
                pass
                
    best_lambda_0, best_beta = best_params
    print(f"Val F1: {best_f1:.4f} | Selected lambda_0: {best_lambda_0:.2e}, beta: {best_beta}")

    K_list_all, sigmas_all = compute_kernel_list(pool_all, pool_all, PHI_all, PHI_all, NU_all, NU_all, MORPH_all, MORPH_all)
    
    K_list_te = []
    chunk = 500
    for s in range(len(pool_te)):
        K_s = torch.cat([rbf_kernel_median(pool_te[s][i:i+chunk], pool_all[s], sigmas_all[f'pool_{s}'])[0] for i in range(0, len(X_test), chunk)], dim=0)
        K_list_te.append(K_s)
    
    K_cross_te = torch.cat([rbf_kernel_median(PHI_te[i:i+chunk], PHI_all, sigmas_all['cross'])[0] for i in range(0, len(X_test), chunk)], dim=0)
    K_list_te.append(K_cross_te)
    
    K_global_te = torch.cat([hellinger_kernel(NU_te[i:i+chunk], NU_all, sigmas_all['global']) for i in range(0, len(X_test), chunk)], dim=0)
    K_list_te.append(K_global_te)
    
    K_morph_te = torch.cat([rbf_kernel_median(MORPH_te[i:i+chunk], MORPH_all, sigmas_all['morph'])[0] for i in range(0, len(X_test), chunk)], dim=0)
    K_list_te.append(K_morph_te)

    K_y_w_all = class_balanced_target_kernel(y_train_all, 5)
    weights_all = fixed_alignment_weights(K_list_all, K_y_w_all)
    print(f"Final Alignment Weights on Full Train: {np.round(weights_all.cpu().numpy(), 3)}")

    K_train_all_fused = fuse(K_list_all, weights_all)
    K_train_all_fused = (K_train_all_fused + K_train_all_fused.T) / 2.0
    K_test_fused = fuse(K_list_te, weights_all)
    
    Y_onehot_all = torch.zeros(len(y_train_all), 5, device=device)
    Y_onehot_all.scatter_(1, y_train_all.unsqueeze(1), 1.0)
    
    scores_te, effective_df = fit_predict(K_train_all_fused, K_test_fused, Y_onehot_all, y_train_all, best_lambda_0, best_beta, 5, return_H=True)
    y_pred = scores_te.argmax(-1).cpu().numpy()
    y_true = y_test.cpu().numpy()
    
    acc       = accuracy_score(y_true, y_pred)
    macro_f1  = f1_score(y_true, y_pred, average='macro')
    per_class = f1_score(y_true, y_pred, average=None).tolist()
    cm        = confusion_matrix(y_true, y_pred).tolist()
    
    print(f"\nTotal supervised scalar parameters: 2 (lambda_0, beta) "
          f"— alignment weights are closed-form, not fit iteratively.")
    print(f"Effective degrees of freedom: {effective_df:.1f} (N_train={len(y_train_all)})")
    
    results = {
        'Test_Accuracy':       float(acc),
        'Test_Macro_F1':       float(macro_f1),
        'Val_Macro_F1':        float(best_f1),
        'Per_Class_Recall':    per_class,
        'Confusion_Matrix':    cm,
        'Training_Time_Seconds': time.time() - t0,
        'Diagnostics': {
            'Effective_DF': effective_df,
            'Supervised_Params': 2,
            'Best_Lambda_0': best_lambda_0,
            'Best_Beta': best_beta,
            'Fusion_Weights': weights_all.cpu().numpy().tolist()
        }
    }
    
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
    Path(results_dir).mkdir(exist_ok=True)
    with open(os.path.join(results_dir, 'swrst_tgke_v6_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nTest Accuracy : {acc:.4f}")
    print(f"Macro F1      : {macro_f1:.4f}")
    print(f"Per-class F1  : {[f'{v:.3f}' for v in per_class]}")
    print(f"Total time    : {time.time()-t0:.1f}s")

if __name__ == '__main__':
    main()
