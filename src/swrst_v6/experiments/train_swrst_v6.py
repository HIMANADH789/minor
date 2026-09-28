import torch
import numpy as np
import os
import json

# Phase I: Frozen V4 Imports
from src.swrst.features.phase_canonicalizer import canonicalize_phase
from src.swrst.features.windowizer import windowize_signal
from src.swrst_v4.transport.resolution_grid import get_resolution_grid, precompute_em
from src.swrst_v4.transport.multi_density import compute_multi_density
from src.swrst_v4.transport.multi_sinkhorn import compute_multi_sinkhorn
from src.swrst_v4.transport.weighted_field import compute_weighted_field, pack_upper_triangular
from src.swrst_v4.geometry.fisher_field import compute_fisher_field

# Phase I-VII: SWRST-v6 Spectrum Modules
from src.swrst_v6.spectrum.log_resolution import compute_log_resolution
from src.swrst_v6.spectrum.resolution_derivative import compute_resolution_derivative
from src.swrst_v6.spectrum.resolution_acceleration import compute_resolution_acceleration
from src.swrst_v6.spectrum.resolution_profile import compute_resolution_profile
from src.swrst_v6.spectrum.resolution_persistence import compute_resolution_persistence
from src.swrst_v6.spectrum.continuous_barycenter import compute_continuous_barycenter
from src.swrst_v6.spectrum.resolution_anisotropy import compute_resolution_anisotropy
from src.swrst_v6.spectrum.soft_velocity import compute_soft_velocity
from src.swrst_v6.spectrum.soft_acceleration import compute_soft_acceleration

# Phase VIII: SWRST-v6 Kernels
from src.swrst_v6.kernels.transport_kernel import compute_transport_kernel
from src.swrst_v6.kernels.profile_kernel import compute_profile_kernel
from src.swrst_v6.kernels.fisher_profile_kernel import compute_fisher_profile_kernel
from src.swrst_v6.kernels.velocity_kernel import compute_kinematic_kernel
from src.swrst_v6.kernels.anisotropy_kernel import compute_anisotropy_kernel
from src.swrst_v6.kernels.adaptive_fusion import compute_adaptive_fusion
from src.swrst_v6.regressors.kernel_ridge import ClassConditionalKRR

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score

def cache_v6_features(X, C, E_m, epsilons, u_m, w_hat_m, inv_du_m, BATCH_SIZE=64):
    N = X.size(0)
    device = X.device
    M = epsilons.size(0)
    K = 12
    
    x_c = canonicalize_phase(X)
    mu = windowize_signal(x_c)
    
    Xi_tilde_list = []
    p_t_list = []
    V_bar_list = []
    A_bar_list = []
    F_packed_list = []
    eig_list = []
    R_t_list = []
    H_var_list = []
    
    with torch.no_grad():
        for i in range(0, N, BATCH_SIZE):
            mu_t = mu[i:i+BATCH_SIZE, :-1]
            mu_t_plus_1 = mu[i:i+BATCH_SIZE, 1:]
            
            # [B, T-1, K, K]
            Q_t = mu_t.unsqueeze(-1) * mu_t_plus_1.unsqueeze(-2)
            
            rho_t = compute_multi_density(Q_t, E_m)
            
            # V4 Frozen Transport
            P_t, h_t, F_num = compute_multi_sinkhorn(mu_t, mu_t_plus_1, Q_t, rho_t, C, epsilons)
            Xi_t, Xi_packed = compute_weighted_field(rho_t, F_num)
            F_t, _ = compute_fisher_field(F_num, Q_t)
            F_packed = pack_upper_triangular(F_t) # [B, T-1, M, K(K+1)/2]
            
            # V6 Resolution Log-Domain Geometry
            V_packed, norm_V = compute_resolution_derivative(Xi_t, Q_t, inv_du_m)
            A_padded, norm_A = compute_resolution_acceleration(V_packed, u_m)
            
            p_t = compute_resolution_profile(norm_V, norm_A, h_t)
            
            Xi_tilde = compute_continuous_barycenter(Xi_packed, p_t, w_hat_m)
            R_t = compute_resolution_persistence(Xi_packed)
            
            # V_bar and A_bar
            V_bar = compute_soft_velocity(V_packed, p_t)
            A_bar = compute_soft_acceleration(A_padded, p_t)
            
            # Anisotropy
            eig_top4 = compute_resolution_anisotropy(Xi_tilde, K)
            
            # Collect sample-wise H_t variance for KRR
            h_var = torch.var(h_t, dim=1).mean(dim=1) # [B]
            
            Xi_tilde_list.append(Xi_tilde)
            p_t_list.append(p_t)
            V_bar_list.append(V_bar)
            A_bar_list.append(A_bar)
            F_packed_list.append(F_packed)
            eig_list.append(eig_top4)
            R_t_list.append(R_t)
            H_var_list.append(h_var)
            
    return (
        torch.cat(Xi_tilde_list),
        torch.cat(p_t_list),
        torch.cat(V_bar_list),
        torch.cat(A_bar_list),
        torch.cat(F_packed_list),
        torch.cat(eig_list),
        torch.cat(R_t_list),
        torch.cat(H_var_list)
    )

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
    epsilons = torch.tensor(get_resolution_grid(), dtype=torch.float32, device=device)
    M = epsilons.size(0)
    
    # H3, H9: Log-domain cache and static pinned epsilons
    u_m, w_hat_m, inv_du_m = compute_log_resolution(epsilons)
    
    print("Precomputing Train Features (SWRST-v6)...")
    Xi_tilde_tr, p_tr, V_bar_tr, A_bar_tr, F_tr, eig_tr, R_tr, H_var_tr = cache_v6_features(
        X_train, C, E_m, epsilons, u_m, w_hat_m, inv_du_m
    )
    
    print("Precomputing Test Features (SWRST-v6)...")
    Xi_tilde_te, p_te, V_bar_te, A_bar_te, F_te, eig_te, R_te, H_var_te = cache_v6_features(
        X_test, C, E_m, epsilons, u_m, w_hat_m, inv_du_m
    )
    
    print("Computing Train Kernels...")
    K_Xi_tr = compute_transport_kernel(Xi_tilde_tr, Xi_tilde_tr)
    K_p_tr = compute_profile_kernel(p_tr, R_tr, p_tr, R_tr)
    K_V_tr = compute_kinematic_kernel(V_bar_tr, V_bar_tr)
    K_A_tr = compute_kinematic_kernel(A_bar_tr, A_bar_tr)
    K_F_tr = compute_fisher_profile_kernel(F_tr, F_tr)
    K_eig_tr = compute_anisotropy_kernel(eig_tr, eig_tr)
    
    K_train = compute_adaptive_fusion(K_Xi_tr, K_p_tr, K_V_tr, K_A_tr, K_F_tr, K_eig_tr, p_tr, p_tr, M)
    
    print("Computing Test Kernels...")
    K_Xi_te = compute_transport_kernel(Xi_tilde_te, Xi_tilde_tr)
    K_p_te = compute_profile_kernel(p_te, R_te, p_tr, R_tr)
    K_V_te = compute_kinematic_kernel(V_bar_te, V_bar_tr)
    K_A_te = compute_kinematic_kernel(A_bar_te, A_bar_tr)
    K_F_te = compute_fisher_profile_kernel(F_te, F_tr)
    K_eig_te = compute_anisotropy_kernel(eig_te, eig_tr)
    
    K_test = compute_adaptive_fusion(K_Xi_te, K_p_te, K_V_te, K_A_te, K_F_te, K_eig_te, p_te, p_tr, M)
    
    num_classes = 5
    Y_train = torch.nn.functional.one_hot(y_train, num_classes=num_classes).float()
    
    print("Solving KRR (Class Conditional)...")
    krr = ClassConditionalKRR(lambda_c=0.1)
    krr.fit(K_train, Y_train, H_var_tr)
    
    print("Evaluating...")
    Y_pred = krr.predict(K_test)
    test_preds = torch.argmax(Y_pred, dim=1)
    
    test_acc = accuracy_score(y_test.cpu(), test_preds.cpu())
    test_f1 = f1_score(y_test.cpu(), test_preds.cpu(), average='macro')
    
    print(f"SWRST-v6 Test Accuracy: {test_acc:.4f}")
    print(f"SWRST-v6 Test Macro F1: {test_f1:.4f}")
    
    results = {
        "Test_Accuracy": float(test_acc),
        "Test_Macro_F1": float(test_f1)
    }
    
    out_file = r'C:\temp\ECG_Benchmark\results\swrst_v6_results.json'
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=4)
        
if __name__ == '__main__':
    main()
