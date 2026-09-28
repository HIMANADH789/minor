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

from predict import (
    compute_local_sigma,
    self_tuning_kernel,
    target_kernel_classification,
    density_weight_classification,
    fixed_alignment_weights,
    tgnwr_predict,
    fuse
)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'swrst_kta_tf'))
from stages.scale_decomposition import scale_decompose
from stages.local_measures import local_measures, compute_bin_edges, global_histogram
from stages.transport_field import transport_field
from stages.cross_scale_correlation import cross_scale_corr
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

def get_balanced_target_kernel(y_train, n_classes):
    K_y = target_kernel_classification(y_train, n_classes)
    rho = density_weight_classification(y_train, n_classes)
    return K_y / (rho.unsqueeze(0) * rho.unsqueeze(1))

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

    pool_tr = [pool_scale_descriptor(R_s) for R_s in R_by_scale_tr]
    pool_val = [pool_scale_descriptor(R_s) for R_s in R_by_scale_val]
    pool_all = [pool_scale_descriptor(R_s) for R_s in R_by_scale_all]
    pool_te = [pool_scale_descriptor(R_s) for R_s in R_by_scale_te]

    print(f"Features extracted: {time.time()-t0:.1f}s")
    
    def compute_kernel_list(pool_A, pool_B, PHI_A, PHI_B, NU_A, NU_B, MORPH_A, MORPH_B, sigmas=None, k=None):
        K_list = []
        new_sigmas = {}
        for s in range(len(pool_A)):
            if sigmas:
                sig_A, sig_B = sigmas[f'pool_{s}']
            else:
                sig_A = compute_local_sigma(pool_A[s], k)
                sig_B = sig_A
                new_sigmas[f'pool_{s}'] = (sig_A, sig_A)
            K_s = self_tuning_kernel(pool_A[s], pool_B[s], sig_A, sig_B)
            K_list.append(K_s)
            
        if sigmas:
            sig_A, sig_B = sigmas['cross']
        else:
            sig_A = compute_local_sigma(PHI_A, k)
            sig_B = sig_A
            new_sigmas['cross'] = (sig_A, sig_A)
        K_cross = self_tuning_kernel(PHI_A, PHI_B, sig_A, sig_B)
        K_list.append(K_cross)
        
        sqrt_A = torch.sqrt(NU_A.clamp(min=1e-10))
        sqrt_B = torch.sqrt(NU_B.clamp(min=1e-10))
        bc = (sqrt_A @ sqrt_B.T).clamp(max=1.0)
        d2_global = (2.0 - 2.0 * bc).clamp(min=0.0)
        
        if sigmas:
            sig_A, sig_B = sigmas['global']
        else:
            d2_A = (2.0 - 2.0 * (sqrt_A @ sqrt_A.T).clamp(max=1.0)).clamp(min=0.0)
            sig_A = torch.topk(d2_A, min(k+1, d2_A.shape[0]), dim=1, largest=False).values[:, -1].clamp(min=1e-6)
            sig_B = sig_A
            new_sigmas['global'] = (sig_A, sig_A)
            
        sigma_prod = sig_A.unsqueeze(1) * sig_B.unsqueeze(0)
        K_global = torch.exp(-d2_global / sigma_prod)
        K_list.append(K_global)
        
        if sigmas:
            sig_A, sig_B = sigmas['morph']
        else:
            sig_A = compute_local_sigma(MORPH_A, k)
            sig_B = sig_A
            new_sigmas['morph'] = (sig_A, sig_A)
        K_morph = self_tuning_kernel(MORPH_A, MORPH_B, sig_A, sig_B)
        K_list.append(K_morph)
        
        return K_list, new_sigmas

    K_y_w_tr = get_balanced_target_kernel(y_train, 5)

    best_score, best_k = -float('inf'), None
    best_weights = None
    
    for k in [5, 10, 15, 20, 30]:
        K_list_tr, sigmas_tr = compute_kernel_list(pool_tr, pool_tr, PHI_tr, PHI_tr, NU_tr, NU_tr, MORPH_tr, MORPH_tr, k=k)
        
        sigmas_val_eval = {
            key: (sig_val, sig_tr) for (key, (sig_val, _)), (_, (_, sig_tr)) in zip(
                compute_kernel_list(pool_val, pool_val, PHI_val, PHI_val, NU_val, NU_val, MORPH_val, MORPH_val, k=k)[1].items(),
                sigmas_tr.items()
            )
        }
        
        K_list_val, _ = compute_kernel_list(pool_val, pool_tr, PHI_val, PHI_tr, NU_val, NU_tr, MORPH_val, MORPH_tr, sigmas=sigmas_val_eval)
        
        weights = fixed_alignment_weights(K_list_tr, K_y_w_tr)
        
        K_train_fused = fuse(K_list_tr, weights)
        K_val_fused = fuse(K_list_val, weights)
        
        rho_tr = density_weight_classification(y_train, 5)
        
        posterior = tgnwr_predict(K_val_fused, y_train, rho_tr, 'classification', n_classes=5)
        score = f1_score(y_val.cpu(), posterior.argmax(-1).cpu(), average='macro')
        
        print(f"k={k} -> Val F1: {score:.4f}")
        if score > best_score:
            best_score, best_k = score, k
            best_weights = weights

    print(f"\nBest k={best_k} with Val F1={best_score:.4f}")

    K_list_all, sigmas_all = compute_kernel_list(pool_all, pool_all, PHI_all, PHI_all, NU_all, NU_all, MORPH_all, MORPH_all, k=best_k)
    K_y_w_all = get_balanced_target_kernel(y_train_all, 5)
    weights_all = fixed_alignment_weights(K_list_all, K_y_w_all)
    
    sigmas_te_eval = {
        key: (sig_te, sig_all) for (key, (sig_te, _)), (_, (_, sig_all)) in zip(
            compute_kernel_list(pool_te, pool_te, PHI_te, PHI_te, NU_te, NU_te, MORPH_te, MORPH_te, k=best_k)[1].items(),
            sigmas_all.items()
        )
    }
    
    K_list_te, _ = compute_kernel_list(pool_te, pool_all, PHI_te, PHI_all, NU_te, NU_all, MORPH_te, MORPH_all, sigmas=sigmas_te_eval)
    
    K_test_fused = fuse(K_list_te, weights_all)
    rho_all = density_weight_classification(y_train_all, 5)
    
    posterior_te = tgnwr_predict(K_test_fused, y_train_all, rho_all, 'classification', n_classes=5)
    y_pred = posterior_te.argmax(-1).cpu().numpy()
    y_true = y_test.cpu().numpy()
    
    acc       = accuracy_score(y_true, y_pred)
    macro_f1  = f1_score(y_true, y_pred, average='macro')
    per_class = f1_score(y_true, y_pred, average=None).tolist()
    cm        = confusion_matrix(y_true, y_pred).tolist()
    
    w_te = K_test_fused / rho_all.unsqueeze(0)
    w_sum = w_te.sum(dim=1)
    w_sq_sum = (w_te ** 2).sum(dim=1)
    n_eff = (w_sum ** 2) / w_sq_sum.clamp(min=1e-12)
    
    mean_n_eff = n_eff.mean().item()
    
    class_4_mask = (y_test == 4)
    if class_4_mask.any():
        mean_n_eff_c4 = n_eff[class_4_mask].mean().item()
    else:
        mean_n_eff_c4 = 0.0
        
    print(f"\nEffective Sample Size (overall): {mean_n_eff:.1f}")
    print(f"Effective Sample Size (Class 4 queries): {mean_n_eff_c4:.1f}")
    
    results = {
        'Test_Accuracy':       float(acc),
        'Test_Macro_F1':       float(macro_f1),
        'Val_Macro_F1':        float(best_score),
        'Per_Class_Recall':    per_class,
        'Confusion_Matrix':    cm,
        'Training_Time_Seconds': time.time() - t0,
        'Diagnostics': {
            'Best_k': best_k,
            'Fusion_Weights': weights_all.cpu().numpy().tolist(),
            'Mean_N_eff': mean_n_eff,
            'Mean_N_eff_Class4': mean_n_eff_c4
        }
    }
    
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
    Path(results_dir).mkdir(exist_ok=True)
    with open(os.path.join(results_dir, 'swrst_tgnwr_v7_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nTest Accuracy : {acc:.4f}")
    print(f"Macro F1      : {macro_f1:.4f}")
    print(f"Per-class F1  : {[f'{v:.3f}' for v in per_class]}")
    print(f"Total time    : {time.time()-t0:.1f}s")

if __name__ == '__main__':
    main()
