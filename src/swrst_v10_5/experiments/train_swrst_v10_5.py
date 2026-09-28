import torch
import numpy as np
import os
import json
import time
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score

from src.swrst.features.phase_canonicalizer import canonicalize_phase
from src.swrst.features.windowizer import windowize_signal
from src.swrst.transport.deterministic_ot import compute_cost_matrix
from src.swrst.transport.independence import compute_independence_cost
from src.swrst_v4.transport.multi_density import compute_multi_density

from src.swrst_v10_5.resolution.extra_grid import get_sorted_extra_grid
from src.swrst_v10_5.resolution.branch_weights import get_sorted_branch_weights
from src.swrst_v10_5.resolution.multi_transport import compute_multi_transport
from src.swrst_v10_5.resolution.resolution_derivative import compute_resolution_kinematics
from src.swrst_v10_5.resolution.band_matrix import precompute_resolution_band_matrix

from src.swrst_v10_5.kernels.fisher_hellinger import compute_fisher_normalization_cache, compute_fisher_hellinger_kernel
from src.swrst_v10_5.kernels.derivative_hellinger import compute_signed_hellinger_kernel
from src.swrst_v10_5.kernels.curvature_kernel import compute_curvature_kernel
from src.swrst_v10_5.kernels.spectral_kernel import compute_spectral_persistence_kernel
from src.swrst_v10_5.kernels.profile_kernel import compute_profile_kernel
from src.swrst_v10_5.kernels.base_fusion import compute_base_fusion
from src.swrst_v10_5.kernels.mmd_resolution import compute_mmd_resolution_self, compute_mmd_resolution_cross, compute_mmd_squared
from src.swrst_v10_5.kernels.time_weights import compute_time_weights
from src.swrst_v10_5.kernels.final_kernel import compute_final_kernel, compute_global_median_heuristic

from src.swrst_v10_5.regressors.kernel_ridge import ClassConditionalKRR

def process_chunk_mmd(
    S_x, S_y, w_sorted, B_mm, chunk_size=32
):
    """
    Computes pairwise MMD2.
    S = (Xi_packed, D_packed, A_packed, Xi_full, norm_cache, w_t)
    """
    Xi_x_packed, D_x_packed, A_x_packed, Xi_x_full, norm_x, w_t_x, norm_full_x = S_x
    Xi_y_packed, D_y_packed, A_y_packed, Xi_y_full, norm_y, w_t_y, norm_full_y = S_y
    
    Nx, T, M, _ = Xi_x_packed.shape
    Ny = Xi_y_packed.shape[0]
    
    MMD2_t = torch.zeros((Nx, Ny, T), device=Xi_x_packed.device, dtype=torch.float32)
    
    # H7: Cache self-kernels
    # Compute self XX and YY
    XX_t = torch.zeros((Nx, T), device=Xi_x_packed.device, dtype=torch.float32)
    YY_t = torch.zeros((Ny, T), device=Xi_x_packed.device, dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, Nx, chunk_size):
            end_i = min(i + chunk_size, Nx)
            k_P_self_x = compute_profile_kernel(D_x_packed[i:end_i], D_x_packed[i:end_i], w_sorted)
            for t in range(T):
                k_H_self_x = compute_fisher_hellinger_kernel(Xi_x_packed[i:end_i, t], Xi_x_packed[i:end_i, t], norm_x[i:end_i, t], norm_x[i:end_i, t])
                k_D_self_x = compute_signed_hellinger_kernel(D_x_packed[i:end_i, t], D_x_packed[i:end_i, t], norm_x[i:end_i, t], norm_x[i:end_i, t])
                k_A_self_x = compute_curvature_kernel(A_x_packed[i:end_i, t], A_x_packed[i:end_i, t], norm_x[i:end_i, t], norm_x[i:end_i, t])
                k_S_self_x = compute_spectral_persistence_kernel(Xi_x_full[i:end_i, t], Xi_x_full[i:end_i, t], norm_full_x[i:end_i, t], norm_full_x[i:end_i, t])
                
                k_self_x = compute_base_fusion(k_H_self_x, k_D_self_x, k_A_self_x, k_P_self_x, k_S_self_x)
                k_self_diag = torch.diagonal(k_self_x, dim1=0, dim2=1).permute(2, 0, 1)
                XX_t[i:end_i, t] = compute_mmd_resolution_self(k_self_diag, w_sorted)
        
        if S_x is S_y:
            YY_t = XX_t.clone()
        else:
            for j in range(0, Ny, chunk_size):
                end_j = min(j + chunk_size, Ny)
                k_P_self_y = compute_profile_kernel(D_y_packed[j:end_j], D_y_packed[j:end_j], w_sorted)
                for t in range(T):
                    k_H_self_y = compute_fisher_hellinger_kernel(Xi_y_packed[j:end_j, t], Xi_y_packed[j:end_j, t], norm_y[j:end_j, t], norm_y[j:end_j, t])
                    k_D_self_y = compute_signed_hellinger_kernel(D_y_packed[j:end_j, t], D_y_packed[j:end_j, t], norm_y[j:end_j, t], norm_y[j:end_j, t])
                    k_A_self_y = compute_curvature_kernel(A_y_packed[j:end_j, t], A_y_packed[j:end_j, t], norm_y[j:end_j, t], norm_y[j:end_j, t])
                    k_S_self_y = compute_spectral_persistence_kernel(Xi_y_full[j:end_j, t], Xi_y_full[j:end_j, t], norm_full_y[j:end_j, t], norm_full_y[j:end_j, t])
                    
                    k_self_y = compute_base_fusion(k_H_self_y, k_D_self_y, k_A_self_y, k_P_self_y, k_S_self_y)
                    k_self_diag = torch.diagonal(k_self_y, dim1=0, dim2=1).permute(2, 0, 1)
                    YY_t[j:end_j, t] = compute_mmd_resolution_self(k_self_diag, w_sorted)
        
        # Cross XY in chunks
        for i in range(0, Nx, chunk_size):
            end_i = min(i + chunk_size, Nx)
            for j in range(0, Ny, chunk_size):
                end_j = min(j + chunk_size, Ny)
                k_P_cross = compute_profile_kernel(D_x_packed[i:end_i], D_y_packed[j:end_j], w_sorted)
                for t in range(T):
                    k_H_cross = compute_fisher_hellinger_kernel(
                        Xi_x_packed[i:end_i, t], Xi_y_packed[j:end_j, t], 
                        norm_x[i:end_i, t], norm_y[j:end_j, t]
                    )
                    k_D_cross = compute_signed_hellinger_kernel(
                        D_x_packed[i:end_i, t], D_y_packed[j:end_j, t], 
                        norm_x[i:end_i, t], norm_y[j:end_j, t]
                    )
                    k_A_cross = compute_curvature_kernel(
                        A_x_packed[i:end_i, t], A_y_packed[j:end_j, t], 
                        norm_x[i:end_i, t], norm_y[j:end_j, t]
                    )
                    k_S_cross = compute_spectral_persistence_kernel(
                        Xi_x_full[i:end_i, t], Xi_y_full[j:end_j, t], 
                        norm_full_x[i:end_i, t], norm_full_y[j:end_j, t]
                    )
                    
                    k_cross = compute_base_fusion(k_H_cross, k_D_cross, k_A_cross, k_P_cross, k_S_cross)
                    XY_chunk = compute_mmd_resolution_cross(k_cross, w_sorted, B_mm)
                    
                    MMD2_t[i:end_i, j:end_j, t] = compute_mmd_squared(
                        XX_t[i:end_i, t], YY_t[j:end_j, t], XY_chunk
                    )
                
    return MMD2_t

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data = np.load(r'C:\temp\ECG_Benchmark\data\ecg5000_balanced.npz')
    X_train = torch.tensor(data['X_train'], dtype=torch.float32, device=device)
    y_train = torch.tensor(data['y_train'], dtype=torch.long, device=device)
    X_test = torch.tensor(data['X_test'], dtype=torch.float32, device=device)
    y_test = torch.tensor(data['y_test'], dtype=torch.long, device=device)
    
    K_bins = 12
    dim = K_bins * (K_bins + 1) // 2
    
    # Compute median cost for adaptive epsilon
    # We will use the V3 method: compute C over a subset to find median
    N_tr = X_train.size(0)
    indices = torch.randperm(N_tr)[:500]
    sample_c = canonicalize_phase(X_train[indices])
    mu_sample = windowize_signal(sample_c)
    C_sample = compute_independence_cost(mu_sample[:, :-1], K_bins)
    C_median = torch.median(C_sample).item()
    
    eps_sorted, sort_idx, adaptive_idx = get_sorted_extra_grid(C_median, device)
    w_sorted = get_sorted_branch_weights(sort_idx)
    B_mm = precompute_resolution_band_matrix(eps_sorted)
    
    print(f"Adaptive Epsilon Position: {adaptive_idx}")
    print(f"Epsilons: {eps_sorted.tolist()}")
    
    BATCH_SIZE = 64
    
    def process_features(X):
        N = X.size(0)
        Xi_packed_list = []
        D_packed_list = []
        A_packed_list = []
        Xi_full_list = []
        norm_list = []
        w_t_list = []
        norm_full_list = []
        
        with torch.no_grad():
            for i in range(0, N, BATCH_SIZE):
                x_b = X[i:i+BATCH_SIZE]
                
                x_c = canonicalize_phase(x_b)
                mu = windowize_signal(x_c)
                mu_t = mu[:, :-1]
                mu_t_plus_1 = mu[:, 1:]
                
                Q_t = mu_t.unsqueeze(-1) * mu_t_plus_1.unsqueeze(-2)
                
                # Adaptive branch density
                rho_t = compute_multi_density(Q_t, eps_sorted[adaptive_idx].unsqueeze(0).expand(1, 1, 1))
                # compute_multi_density expects E_m which is shape [M, 1, 1], here [1, 1, 1]
                # Actually, compute_multi_density in v4 takes (Q_t, E_m)
                E_m_adaptive = eps_sorted[adaptive_idx].view(1, 1, 1)
                rho_t_adaptive = compute_multi_density(Q_t, E_m_adaptive).squeeze(2) # [B, T, K, K]
                
                C = compute_independence_cost(mu_t, K_bins)
                
                # We need C in [K, K] for Sinkhorn? No, C in independence_cost is [B, T, K, K] in V4!
                # Wait, compute_multi_sinkhorn in v4 takes C which is [B, T, K, K].
                # Let's use the average C over time/batch or just the exact C_t?
                # Actually, cost geometry is pairwise spatial coordinates!
                # compute_cost_matrix is already imported at top level
                C_spatial_np = compute_cost_matrix(K_bins)
                C_spatial = torch.tensor(C_spatial_np, dtype=torch.float32, device=device) # [K, K]
                
                P, Xi = compute_multi_transport(
                    mu_t, mu_t_plus_1, Q_t, rho_t_adaptive, C_spatial, eps_sorted
                )
                
                Xi_packed, D_packed, A_packed = compute_resolution_kinematics(Xi, eps_sorted)
                
                norm_cache = compute_fisher_normalization_cache(Q_t) # [B, T, K, K]
                norm_cache_packed = norm_cache[..., torch.triu_indices(K_bins, K_bins)[0], torch.triu_indices(K_bins, K_bins)[1]] # [B, T, d]
                
                w_t = compute_time_weights(Xi_packed, w_sorted)
                
                Xi_packed_list.append(Xi_packed)
                D_packed_list.append(D_packed)
                A_packed_list.append(A_packed)
                Xi_full_list.append(Xi)
                norm_list.append(norm_cache_packed)
                w_t_list.append(w_t)
                norm_full_list.append(norm_cache)
                
        return (
            torch.cat(Xi_packed_list), torch.cat(D_packed_list), 
            torch.cat(A_packed_list), torch.cat(Xi_full_list), 
            torch.cat(norm_list), torch.cat(w_t_list), torch.cat(norm_full_list)
        )
        
    print("Precomputing Train Features (SWRST-v10.5)...")
    S_tr = process_features(X_train)
    
    print("Precomputing Test Features (SWRST-v10.5)...")
    S_te = process_features(X_test)
    
    print("Computing Train MMD Matrix...")
    start = time.time()
    MMD2_train = process_chunk_mmd(S_tr, S_tr, w_sorted, B_mm)
    print(f"Train MMD computed in {time.time() - start:.2f}s")
    
    print("Computing Test MMD Matrix...")
    MMD2_test = process_chunk_mmd(S_te, S_tr, w_sorted, B_mm)
    
    # H8: Median heuristic for final kernel
    w_t_train = S_tr[5]
    dist2_train = torch.sum(MMD2_train * w_t_train.unsqueeze(1), dim=2)
    sigma = compute_global_median_heuristic(dist2_train)
    print(f"Global Bandwidth Sigma: {sigma:.4f}")
    
    w_t_test = S_te[5]
    K_train, _ = compute_final_kernel(MMD2_train, w_t_train, sigma)
    K_test, _ = compute_final_kernel(MMD2_test, w_t_test, sigma)
    
    Y_train = torch.nn.functional.one_hot(y_train, num_classes=5).float()
    
    print("Solving Class-Conditional KRR...")
    krr = ClassConditionalKRR()
    krr.fit(K_train, Y_train)
    
    print("Evaluating...")
    Y_pred = krr.predict(K_test)
    test_preds = torch.argmax(Y_pred, dim=1)
    
    test_acc = accuracy_score(y_test.cpu(), test_preds.cpu())
    test_f1 = f1_score(y_test.cpu(), test_preds.cpu(), average='macro')
    recalls = recall_score(y_test.cpu(), test_preds.cpu(), average=None).tolist()
    cm = confusion_matrix(y_test.cpu(), test_preds.cpu()).tolist()
    
    print(f"SWRST-v10.5 Test Accuracy: {test_acc:.4f}")
    print(f"SWRST-v10.5 Test Macro F1: {test_f1:.4f}")
    
    results = {
        "Test_Accuracy": float(test_acc),
        "Test_Macro_F1": float(test_f1),
        "Per_Class_Recall": recalls,
        "Confusion_Matrix": cm
    }
    
    out_file = r'C:\temp\ECG_Benchmark\results\balanced\swrst_v10_5_balanced.json'
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=4)
        
if __name__ == '__main__':
    main()
