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
import itertools

def learn_fusion_weights(K_list, K_y, n_grid=11):
    K_y_c = K_y - K_y.mean(0, keepdim=True) - K_y.mean(1, keepdim=True) + K_y.mean()
    Ky_norm = K_y_c.norm()
    
    logs = torch.stack([torch.log(K.clamp(min=1e-30)) for K in K_list])
    best_score = -float('inf')
    best_w = None
    
    grid = torch.linspace(0, 1, n_grid, device=K_y.device)
    for w in itertools.product(grid, repeat=len(K_list)):
        w_tensor = torch.tensor(w, device=K_y.device)
        if w_tensor.sum() == 0:
            continue
        w_tensor = w_tensor / w_tensor.sum()
        log_K = (w_tensor.view(-1, 1, 1) * logs).sum(0)
        K_fused = torch.exp(torch.clamp(log_K, min=-30.0))
        K_f_c = K_fused - K_fused.mean(0, keepdim=True) - K_fused.mean(1, keepdim=True) + K_fused.mean()
        score = (K_f_c * K_y_c).sum() / (K_f_c.norm() * Ky_norm + 1e-12)
        if score > best_score:
            best_score = score
            best_w = w_tensor
            
    return best_w

def class_balanced_target_kernel(y_train: torch.Tensor, n_classes: int) -> torch.Tensor:
    n_per_class = torch.bincount(y_train, minlength=n_classes).float()
    w = 1.0 / n_per_class[y_train]
    same_class = (y_train.unsqueeze(0) == y_train.unsqueeze(1)).float()
    return same_class * w.unsqueeze(0) * w.unsqueeze(1)

def randomized_svd_right(M: torch.Tensor, rank: int, n_oversampling: int = 10) -> torch.Tensor:
    m, n = M.shape
    k = min(rank + n_oversampling, n, m)
    Omega = torch.randn(n, k, device=M.device, dtype=M.dtype)
    Y = M @ Omega
    Q, _ = torch.linalg.qr(Y)
    B = Q.T @ M
    _, _, Vt = torch.linalg.svd(B, full_matrices=False)
    return Vt[:rank].T

def balanced_kta_projection(R_flat_tr: torch.Tensor, y_train: torch.Tensor, n_classes: int = 5, rank: int = 32, n_oversampling: int = 10) -> torch.Tensor:
    K_y = class_balanced_target_kernel(y_train, n_classes=n_classes)
    K_y = K_y - K_y.mean(0, keepdim=True) - K_y.mean(1, keepdim=True) + K_y.mean()
    N_train, D = R_flat_tr.shape
    M = torch.empty((N_train, D), dtype=torch.float32, device=K_y.device)
    block_size = 1024
    for d in range(0, D, block_size):
        end = min(d + block_size, D)
        M[:, d:end] = K_y @ R_flat_tr[:, d:end]
    W_star = randomized_svd_right(M, rank=rank, n_oversampling=n_oversampling)
    return W_star

def fuse(K_list, weights):
    log_K = sum(w * torch.log(K.clamp(min=1e-30)) for w, K in zip(weights, K_list))
    K_fused = torch.exp(torch.clamp(log_K, min=-30.0))
    if K_fused.shape[0] == K_fused.shape[1]:
        return (K_fused + K_fused.T) / 2.0
    return K_fused

def fit_predict(K_train, K_valtest, y_train, lambda_0, beta, n_classes=5):
    device = K_train.device
    n_per_class = torch.bincount(y_train, minlength=n_classes).float()
    lambda_c = lambda_0 * (n_per_class / n_per_class.max()).pow(beta)
    Lambda_diag = lambda_c[y_train]
    
    Y_onehot = torch.zeros(len(y_train), n_classes, device=device)
    Y_onehot.scatter_(1, y_train.unsqueeze(1), 1.0)
    
    K_reg = K_train + torch.diag(Lambda_diag + 1e-6)
    
    try:
        L = torch.linalg.cholesky(K_reg)
    except torch.linalg.LinAlgError:
        try:
            K_reg = K_reg + torch.eye(K_train.shape[0], device=device) * 1e-3
            L = torch.linalg.cholesky(K_reg)
        except torch.linalg.LinAlgError:
            K_reg = K_reg + torch.eye(K_train.shape[0], device=device) * 1e-1
            L = torch.linalg.cholesky(K_reg)
            
    alpha = torch.cholesky_solve(Y_onehot, L)
    preds = K_valtest @ alpha
    return preds

def extract_all(X, device):
    X_scales, X_details = scale_decompose(X, device)
    MORPH = extract_morphological_features(X)
    return X_scales, X_details, MORPH

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
    
    X_all = torch.tensor(X_train_full, dtype=torch.float32, device=device)
    y_all = torch.tensor(y_train_full, dtype=torch.long, device=device)
    
    t0 = time.time()
    
    # Validation Phase
    X_scales_tr, X_details_tr, MORPH_tr = extract_all(X_train, device)
    X_scales_val, X_details_val, MORPH_val = extract_all(X_val, device)
    
    morph_mean_tr = MORPH_tr.mean(0, keepdim=True)
    morph_std_tr  = MORPH_tr.std(0,  keepdim=True) + 1e-8
    MORPH_tr = (MORPH_tr - morph_mean_tr) / morph_std_tr
    MORPH_val = (MORPH_val - morph_mean_tr) / morph_std_tr
    
    bin_edges_tr = compute_bin_edges(X_train)
    MU_tr, MU_list_tr = local_measures(X_scales_tr, bin_edges_tr)
    MU_val, MU_list_val = local_measures(X_scales_val, bin_edges_tr)
    
    R_tr, _, R_by_scale_tr = transport_field(MU_tr, MU_list_tr, bin_edges_tr)
    R_val, _, R_by_scale_val = transport_field(MU_val, MU_list_val, bin_edges_tr)
    
    PHI_tr = cross_scale_corr(X_details_tr, X_scales_tr, max_lag=5)
    PHI_val = cross_scale_corr(X_details_val, X_scales_val, max_lag=5)
    NU_tr  = global_histogram(X_scales_tr, bin_edges_tr)
    NU_val  = global_histogram(X_scales_val, bin_edges_tr)
    
    R_flat_tr = R_tr.reshape(len(X_train), -1)
    R_flat_val = R_val.reshape(len(X_val), -1)
    
    K_y_w_tr = class_balanced_target_kernel(y_train, n_classes=5)
    W_star_tr = balanced_kta_projection(R_flat_tr, y_train, n_classes=5, rank=5, n_oversampling=10)
    
    Z_tr = R_flat_tr @ W_star_tr
    z_mean_tr = Z_tr.mean(0, keepdim=True)
    z_std_tr  = Z_tr.std(0,  keepdim=True) + 1e-8
    Z_tr_norm = (Z_tr - z_mean_tr) / z_std_tr
    Z_val_norm = ((R_flat_val @ W_star_tr) - z_mean_tr) / z_std_tr
    
    sigma_proj   = torch.pdist(Z_tr_norm).pow(2).median().item()
    sigma_cross  = torch.pdist(PHI_tr).pow(2).median().item()
    sigma_global = compute_hellinger_sigma(NU_tr)
    sigma_morph  = torch.pdist(MORPH_tr).pow(2).median().item()
    
    K_proj_tr   = rbf_kernel_batched(Z_tr_norm, Z_tr_norm, sigma_proj)
    K_cross_tr  = rbf_kernel_batched(PHI_tr, PHI_tr, sigma_cross)
    K_global_tr = hellinger_kernel(NU_tr, NU_tr, sigma_global)
    K_morph_tr  = rbf_kernel_batched(MORPH_tr, MORPH_tr, sigma_morph)
    
    fusion_weights_tr = learn_fusion_weights([K_proj_tr, K_cross_tr, K_global_tr, K_morph_tr], K_y_w_tr, n_grid=11)
    
    K_train_fused = fuse([K_proj_tr, K_cross_tr, K_global_tr, K_morph_tr], fusion_weights_tr)
    
    K_proj_val   = rbf_kernel_batched(Z_val_norm, Z_tr_norm, sigma_proj)
    K_cross_val  = rbf_kernel_batched(PHI_val, PHI_tr, sigma_cross)
    K_global_val = hellinger_kernel(NU_val, NU_tr, sigma_global)
    K_morph_val  = rbf_kernel_batched(MORPH_val, MORPH_tr, sigma_morph)
    
    K_val_fused = fuse([K_proj_val, K_cross_val, K_global_val, K_morph_val], fusion_weights_tr)
    
    best_f1, best_params = -1, None
    for lambda_0 in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
        for beta in [0.3, 0.5, 0.7, 1.0, 1.5]:
            try:
                preds_val = fit_predict(K_train_fused, K_val_fused, y_train, lambda_0, beta, n_classes=5)
                f1 = f1_score(y_val.cpu(), preds_val.argmax(-1).cpu(), average='macro')
                if f1 > best_f1:
                    best_f1 = f1
                    best_params = (lambda_0, beta)
            except torch.linalg.LinAlgError:
                pass
                
    best_lambda_0, best_beta = best_params
    print(f"Val F1: {best_f1:.4f} | Selected lambda_0: {best_lambda_0:.2e}, beta: {best_beta}")
    
    # Testing Phase
    X_scales_all, X_details_all, MORPH_all = extract_all(X_all, device)
    X_scales_te, X_details_te, MORPH_te = extract_all(X_test, device)
    
    morph_mean_all = MORPH_all.mean(0, keepdim=True)
    morph_std_all  = MORPH_all.std(0,  keepdim=True) + 1e-8
    MORPH_all = (MORPH_all - morph_mean_all) / morph_std_all
    MORPH_te = (MORPH_te - morph_mean_all) / morph_std_all
    
    bin_edges_all = compute_bin_edges(X_all)
    MU_all, MU_list_all = local_measures(X_scales_all, bin_edges_all)
    MU_te, MU_list_te = local_measures(X_scales_te, bin_edges_all)
    
    R_all, _, R_by_scale_all = transport_field(MU_all, MU_list_all, bin_edges_all)
    R_te, _, R_by_scale_te = transport_field(MU_te, MU_list_te, bin_edges_all)
    
    PHI_all = cross_scale_corr(X_details_all, X_scales_all, max_lag=5)
    PHI_te = cross_scale_corr(X_details_te, X_scales_te, max_lag=5)
    NU_all  = global_histogram(X_scales_all, bin_edges_all)
    NU_te  = global_histogram(X_scales_te, bin_edges_all)
    
    R_flat_all = R_all.reshape(len(X_all), -1)
    R_flat_te = R_te.reshape(len(X_test), -1)
    
    K_y_w_all = class_balanced_target_kernel(y_all, n_classes=5)
    W_star_all = balanced_kta_projection(R_flat_all, y_all, n_classes=5, rank=5, n_oversampling=10)
    
    Z_all = R_flat_all @ W_star_all
    z_mean_all = Z_all.mean(0, keepdim=True)
    z_std_all  = Z_all.std(0,  keepdim=True) + 1e-8
    Z_all_norm = (Z_all - z_mean_all) / z_std_all
    Z_te_norm = ((R_flat_te @ W_star_all) - z_mean_all) / z_std_all
    
    sigma_proj_all   = torch.pdist(Z_all_norm).pow(2).median().item()
    sigma_cross_all  = torch.pdist(PHI_all).pow(2).median().item()
    sigma_global_all = compute_hellinger_sigma(NU_all)
    sigma_morph_all  = torch.pdist(MORPH_all).pow(2).median().item()
    
    K_proj_all   = rbf_kernel_batched(Z_all_norm, Z_all_norm, sigma_proj_all)
    K_cross_all  = rbf_kernel_batched(PHI_all, PHI_all, sigma_cross_all)
    K_global_all = hellinger_kernel(NU_all, NU_all, sigma_global_all)
    K_morph_all  = rbf_kernel_batched(MORPH_all, MORPH_all, sigma_morph_all)
    
    fusion_weights_all = learn_fusion_weights([K_proj_all, K_cross_all, K_global_all, K_morph_all], K_y_w_all, n_grid=11)
    print(f"Final learned fusion weights on all data: {fusion_weights_all.tolist()}")
    
    K_train_all_fused = fuse([K_proj_all, K_cross_all, K_global_all, K_morph_all], fusion_weights_all)
    
    K_test_chunks = []
    chunk = 256
    for i in range(0, len(X_test), chunk):
        sl = slice(i, i+chunk)
        Kp = rbf_kernel_batched(Z_te_norm[sl], Z_all_norm, sigma_proj_all)
        Kc = rbf_kernel_batched(PHI_te[sl], PHI_all, sigma_cross_all)
        Kg = hellinger_kernel(NU_te[sl], NU_all, sigma_global_all)
        Km = rbf_kernel_batched(MORPH_te[sl], MORPH_all, sigma_morph_all)
        K_test_chunks.append(fuse([Kp, Kc, Kg, Km], fusion_weights_all))
    K_test_fused = torch.cat(K_test_chunks, dim=0)
    
    preds_te = fit_predict(K_train_all_fused, K_test_fused, y_all, best_lambda_0, best_beta, n_classes=5)
    
    y_pred = preds_te.argmax(-1).cpu().numpy()
    y_true = y_test.cpu().numpy()
    
    acc       = accuracy_score(y_true, y_pred)
    macro_f1  = f1_score(y_true, y_pred, average='macro')
    per_class = f1_score(y_true, y_pred, average=None).tolist()
    cm        = confusion_matrix(y_true, y_pred).tolist()
    
    results = {
        'Test_Accuracy':       float(acc),
        'Test_Macro_F1':       float(macro_f1),
        'Val_Macro_F1':        float(best_f1),
        'Per_Class_Recall':    per_class,
        'Confusion_Matrix':    cm,
        'Training_Time_Seconds': time.time() - t0,
        'Diagnostics': {
            'Best_Lambda_0': best_lambda_0,
            'Best_Beta': best_beta,
            'Fusion_Weights': fusion_weights_all.cpu().numpy().tolist()
        }
    }
    
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
    Path(results_dir).mkdir(exist_ok=True)
    with open(os.path.join(results_dir, 'swrst_kta_tf_v3_singlesplit_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nTest Accuracy : {acc:.4f}")
    print(f"Macro F1      : {macro_f1:.4f}")
    print(f"Per-class F1  : {[f'{v:.3f}' for v in per_class]}")
    print(f"Total time    : {time.time()-t0:.1f}s")

if __name__ == '__main__':
    main()
