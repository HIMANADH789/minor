import torch
import numpy as np
import os
import json

from src.swrst.features.phase_canonicalizer import canonicalize_phase
from src.swrst.features.windowizer import windowize_signal
from src.swrst.transport.deterministic_ot import get_cached_ot, compute_cost_matrix

from src.swrst_v4.transport.resolution_grid import get_resolution_grid, precompute_em
from src.swrst_v4.transport.multi_density import compute_multi_density
from src.swrst_v4.transport.multi_sinkhorn import compute_multi_sinkhorn
from src.swrst_v4.transport.weighted_field import compute_weighted_field
from src.swrst_v4.transport.resolution_curvature import compute_resolution_curvature
from src.swrst_v4.transport.resolution_router import compute_resolution_router

from src.swrst_v4.geometry.fisher_field import compute_fisher_field
from src.swrst_v4.geometry.persistent_spectrum import compute_persistent_spectrum
from src.swrst_v4.geometry.energy_trajectory import compute_energy_trajectory

from src.swrst_v4.kernels.hellinger_kernel import compute_hellinger_kernel
from src.swrst_v4.kernels.spectral_persistence_kernel import compute_spectral_persistence_kernel
from src.swrst_v4.kernels.fisher_kernel import compute_fisher_kernel
from src.swrst_v4.kernels.energy_trajectory_kernel import compute_energy_trajectory_kernel
from src.swrst_v4.kernels.order_variance_kernel import compute_order_variance, compute_order_variance_kernel
from src.swrst_v4.kernels.cross_order_kernel import compute_cross_order_interactions, compute_cross_order_kernel
from src.swrst_v4.kernels.route_switch_kernel import compute_route_switch_entropy, compute_route_switch_kernel
from src.swrst_v4.kernels.fisher_residual_kernel import compute_fisher_residual, compute_fisher_residual_kernel
from src.swrst_v4.geometry.log_geometric_field import compute_log_geometric_field
from src.swrst_v4.transport.weighted_field import pack_upper_triangular

from src.swrst_v4.kernels.adaptive_fusion import compute_fisher_dominant_fusion
from src.swrst_v4.regressors.kernel_ridge import solve_kernel_ridge
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score

def cache_features(X, C, E_m, epsilons, BATCH_SIZE=64):
    N = X.size(0)
    device = X.device
    
    x_c = canonicalize_phase(X)
    mu = windowize_signal(x_c)
    
    Xi_packed_list = []
    F_packed_list = []
    Phi_spec_list = []
    delta_lam_list = []
    e_t_list = []
    h_t_list = []
    v_t_list = []
    c_t_list = []
    r_t_list = []
    df_t_list = []
    
    with torch.no_grad():
        for i in range(0, N, BATCH_SIZE):
            mu_t = mu[i:i+BATCH_SIZE, :-1]
            mu_t_plus_1 = mu[i:i+BATCH_SIZE, 1:]
            
            Q_t = mu_t.unsqueeze(-1) * mu_t_plus_1.unsqueeze(-2)
            
            rho_t = compute_multi_density(Q_t, E_m)
            
            # multi_sinkhorn returns P_t, h_t, F_num (which is P_t - Q_t = Omega_t)
            P_t, h_t, F_num = compute_multi_sinkhorn(mu_t, mu_t_plus_1, Q_t, rho_t, C, epsilons)
            
            Xi_t, Xi_packed_unused = compute_weighted_field(rho_t, F_num)
            
            log_G_t, log_G_packed = compute_log_geometric_field(Xi_t)
            
            K_t = compute_resolution_curvature(Xi_t)
            F_t, F_t_packed_unused = compute_fisher_field(F_num, Q_t)
            
            Xi_star, F_star, idx = compute_resolution_router(h_t, K_t, Xi_t, F_t)
            
            Xi_star_M = Xi_star.unsqueeze(2)
            F_star_M = F_star.unsqueeze(2)
            
            log_G_t, log_G_packed = compute_log_geometric_field(Xi_star_M)
            
            F_star_packed = pack_upper_triangular(F_star_M)
            
            Phi_spec, delta_lam = compute_persistent_spectrum(Xi_star_M)
            
            e_t = compute_energy_trajectory(Xi_t) # Full spectrum
            v_t = compute_order_variance(Xi_t) # Full spectrum
            c_t = compute_cross_order_interactions(Xi_t)
            r_t = compute_route_switch_entropy(idx)
            df_t = compute_fisher_residual(F_star)
            
            Xi_packed_list.append(log_G_packed)
            F_packed_list.append(F_star_packed)
            Phi_spec_list.append(Phi_spec)
            delta_lam_list.append(delta_lam)
            e_t_list.append(e_t)
            h_t_list.append(h_t)
            v_t_list.append(v_t)
            c_t_list.append(c_t)
            r_t_list.append(r_t)
            df_t_list.append(df_t)
            
    return (torch.cat(Xi_packed_list), 
            torch.cat(F_packed_list), 
            torch.cat(Phi_spec_list),
            torch.cat(delta_lam_list), 
            torch.cat(e_t_list), 
            torch.cat(h_t_list),
            torch.cat(v_t_list),
            torch.cat(c_t_list),
            torch.cat(r_t_list),
            torch.cat(df_t_list))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data = np.load(r'C:\temp\ECG_Benchmark\data\ecg5000_resplit.npz')
    X_train = torch.tensor(data['X_train'], dtype=torch.float32, device=device)
    y_train = torch.tensor(data['y_train'], dtype=torch.long, device=device)
    X_test = torch.tensor(data['X_test'], dtype=torch.float32, device=device)
    y_test = torch.tensor(data['y_test'], dtype=torch.long, device=device)
    
    K = 12
    C, E_m = precompute_em(K, device)
    epsilons = get_resolution_grid()
    
    print("Precomputing Train Features...")
    Xi_train, F_train, Phi_train, lam_train, e_train, h_train, v_train, c_train, r_train, df_train = cache_features(X_train, C, E_m, epsilons)
    
    print("Precomputing Test Features...")
    Xi_test, F_test, Phi_test, lam_test, e_test, h_test, v_test, c_test, r_test, df_test = cache_features(X_test, C, E_m, epsilons)
    
    def standardize(train_feat, test_feat):
        # Flatten time and batch to compute mean/std per M-channel or feature-channel
        shape = train_feat.shape
        if len(shape) == 3: # [N, T, M]
            mean_f = torch.mean(train_feat, dim=(0, 1), keepdim=True)
            std_f = torch.std(train_feat, dim=(0, 1), keepdim=True) + 1e-8
        elif len(shape) == 4: # [N, T, M, K]
            mean_f = torch.mean(train_feat, dim=(0, 1), keepdim=True)
            std_f = torch.std(train_feat, dim=(0, 1), keepdim=True) + 1e-8
        else:
            mean_f = torch.mean(train_feat, dim=0, keepdim=True)
            std_f = torch.std(train_feat, dim=0, keepdim=True) + 1e-8
        return (train_feat - mean_f) / std_f, (test_feat - mean_f) / std_f
        
    print("Standardizing features to balance resolution spectrum...")
    Phi_train, Phi_test = standardize(Phi_train, Phi_test)
    lam_train, lam_test = standardize(lam_train, lam_test)
    e_train, e_test = standardize(e_train, e_test)
    h_train, h_test = standardize(h_train, h_test)
    v_train, v_test = standardize(v_train, v_test)
    c_train, c_test = standardize(c_train, c_test)
    r_train, r_test = standardize(r_train, r_test)
    df_train, df_test = standardize(df_train, df_test)
    
    print("Computing Train Kernels...")
    K_H_train = compute_hellinger_kernel(Xi_train, Xi_train)
    K_F_train = compute_fisher_kernel(F_train, F_train)
    K_P_train = compute_spectral_persistence_kernel(lam_train, lam_train)
    K_E_train = compute_energy_trajectory_kernel(e_train, e_train)
    K_V_train = compute_order_variance_kernel(v_train, v_train)
    K_C_train = compute_cross_order_kernel(c_train, c_train)
    K_R_train = compute_route_switch_kernel(r_train, r_train)
    K_dF_train = compute_fisher_residual_kernel(df_train, df_train)
    
    print("Fusing Train Kernels (Fisher-dominant)...")
    K_fused_train = compute_fisher_dominant_fusion(K_F_train, K_H_train, K_P_train, K_E_train, K_V_train, K_C_train, K_R_train, K_dF_train)
    
    K_H_test = compute_hellinger_kernel(Xi_test, Xi_train)
    K_F_test = compute_fisher_kernel(F_test, F_train)
    K_P_test = compute_spectral_persistence_kernel(lam_test, lam_train)
    K_E_test = compute_energy_trajectory_kernel(e_test, e_train)
    K_V_test = compute_order_variance_kernel(v_test, v_train)
    K_C_test = compute_cross_order_kernel(c_test, c_train)
    K_R_test = compute_route_switch_kernel(r_test, r_train)
    K_dF_test = compute_fisher_residual_kernel(df_test, df_train)
    
    kernel_names = ["Hellinger", "Fisher", "Persistence", "Energy", "OrderVariance", "CrossOrder", "RouteSwitch", "FisherResidual"]
    train_kernels = [K_H_train, K_F_train, K_P_train, K_E_train, K_V_train, K_C_train, K_R_train, K_dF_train]
    test_kernels = [K_H_test, K_F_test, K_P_test, K_E_test, K_V_test, K_C_test, K_R_test, K_dF_test]
    
    num_classes = 5
    Y_train = torch.nn.functional.one_hot(y_train, num_classes=num_classes).float()
    
    print("Evaluating individual kernels...")
    for name, K_tr, K_te in zip(kernel_names, train_kernels, test_kernels):
        alpha_ind = solve_kernel_ridge(K_tr, Y_train, h_train)
        Y_pred_ind = torch.matmul(K_te, alpha_ind)
        test_preds_ind = torch.argmax(Y_pred_ind, dim=1)
        acc = accuracy_score(y_test.cpu(), test_preds_ind.cpu())
        print(f"Kernel {name} Accuracy: {acc:.4f}")
    
    print("Fusing Test Kernels (Fisher-dominant)...")
    K_fused_test = compute_fisher_dominant_fusion(K_F_test, K_H_test, K_P_test, K_E_test, K_V_test, K_C_test, K_R_test, K_dF_test)
    
    print("Solving KRR for fused model...")
    alpha = solve_kernel_ridge(K_fused_train, Y_train, h_train)
    
    print("Evaluating...")
    Y_pred = torch.matmul(K_fused_test, alpha)
    test_preds = torch.argmax(Y_pred, dim=1)
    
    test_acc = accuracy_score(y_test.cpu(), test_preds.cpu())
    test_f1 = f1_score(y_test.cpu(), test_preds.cpu(), average='macro')
    cm = confusion_matrix(y_test.cpu(), test_preds.cpu())
    recalls = recall_score(y_test.cpu(), test_preds.cpu(), average=None)
    
    print(f"SWRST-v4 Test Accuracy: {test_acc:.4f}")
    print(f"SWRST-v4 Test Macro F1: {test_f1:.4f}")
    
    results = {
        "Test_Accuracy": float(test_acc),
        "Test_Macro_F1": float(test_f1),
        "Confusion_Matrix": cm.tolist(),
        "Per_Class_Recall": recalls.tolist()
    }
    
    out_file = r'C:\temp\ECG_Benchmark\results\swrst_v4_results.json'
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Results saved to {out_file}")

if __name__ == '__main__':
    main()
