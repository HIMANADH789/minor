import torch
import numpy as np
import os
import json
import time

# Phase I: Frozen V7 Transport
from src.swrst.features.phase_canonicalizer import canonicalize_phase
from src.swrst.features.windowizer import windowize_signal
from src.swrst_v4.transport.resolution_grid import get_resolution_grid, precompute_em
from src.swrst_v4.transport.multi_density import compute_multi_density
from src.swrst_v4.transport.multi_sinkhorn import compute_multi_sinkhorn

# Statistics Stack
from src.swrst_v8.statistics.log_ratio_centered import compute_log_ratio_fisher
from src.swrst_v8.statistics.nystrom_svd import NystromSVD
from src.swrst_v8.statistics.resolution_kinematics import compute_kinematics
from src.swrst_v8.statistics.spectral_entropy import compute_spectral_entropy
from src.swrst_v8.statistics.cross_resolution_gram import compute_cross_resolution_gram
from src.swrst_v8.statistics.profile_energy import compute_profile_energy_and_state

# Kernels
from src.swrst_v8.kernels.tensor_product_rkhs import compute_sigma2_m, compute_tensor_product_kernel
from src.swrst_v8.regressors.kernel_ridge import ClassConditionalKRR

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

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
    
    BATCH_SIZE = 64
    r = 8
    
    # Initialize Nystrom SVD
    nystrom = NystromSVD(r=r, num_classes=5)
    
    print("Accumulating Nyström Basis Covariance...")
    start = time.time()
    
    N_tr = X_train.size(0)
    with torch.no_grad():
        for i in range(0, N_tr, BATCH_SIZE):
            x_b = X_train[i:i+BATCH_SIZE]
            y_b = y_train[i:i+BATCH_SIZE]
            
            x_c = canonicalize_phase(x_b)
            mu = windowize_signal(x_c)
            mu_t = mu[:, :-1]
            mu_t_plus_1 = mu[:, 1:]
            
            Q_t = mu_t.unsqueeze(-1) * mu_t_plus_1.unsqueeze(-2)
            rho_t = compute_multi_density(Q_t, E_m)
            
            P_sinkhorn, _, _ = compute_multi_sinkhorn(mu_t, mu_t_plus_1, Q_t, rho_t, C, epsilons)
            
            L_F_packed = compute_log_ratio_fisher(P_sinkhorn, Q_t, K)
            
            nystrom.accumulate(L_F_packed, y_b)
            
    nystrom.fit()
    print(f"Nyström SVD Basis Fitted in {time.time() - start:.2f}s")
    
    def process_features(X):
        N = X.size(0)
        Z_full_list = []
        p_list = []
        
        with torch.no_grad():
            for i in range(0, N, BATCH_SIZE):
                x_b = X[i:i+BATCH_SIZE]
                
                x_c = canonicalize_phase(x_b)
                mu = windowize_signal(x_c)
                mu_t = mu[:, :-1]
                mu_t_plus_1 = mu[:, 1:]
                
                Q_t = mu_t.unsqueeze(-1) * mu_t_plus_1.unsqueeze(-2)
                rho_t = compute_multi_density(Q_t, E_m)
                
                P_sinkhorn, _, _ = compute_multi_sinkhorn(mu_t, mu_t_plus_1, Q_t, rho_t, C, epsilons)
                
                L_F_packed = compute_log_ratio_fisher(P_sinkhorn, Q_t, K)
                
                z = nystrom.project(L_F_packed) # [B, T-1, M, r]
                
                D, A, J = compute_kinematics(z)
                
                H_spec = compute_spectral_entropy(z) # [B, M]
                
                G_eigs = compute_cross_resolution_gram(z) # [B, T-1, 4]
                
                Z_full, p_m = compute_profile_energy_and_state(z, D, A, J, H_spec, G_eigs)
                
                Z_full_list.append(Z_full)
                p_list.append(p_m)
                
        return torch.cat(Z_full_list), torch.cat(p_list)
        
    print("Precomputing Train Features (SWRST-v8)...")
    Z_tr, p_tr = process_features(X_train)
    
    print("Precomputing Test Features (SWRST-v8)...")
    Z_te, p_te = process_features(X_test)
    
    print("Caching intrinsic bandwidth sigma_m^2...")
    sigma2_m = compute_sigma2_m(Z_tr)
    
    print("Computing Train Tensor Product Kernel...")
    K_train = compute_tensor_product_kernel(Z_tr, Z_tr, p_tr, p_tr, sigma2_m)
    
    print("Computing Test Tensor Product Kernel...")
    K_test = compute_tensor_product_kernel(Z_te, Z_tr, p_te, p_tr, sigma2_m)
    
    Y_train = torch.nn.functional.one_hot(y_train, num_classes=5).float()
    
    print("Solving Class-Conditional KRR...")
    krr = ClassConditionalKRR()
    krr.fit(K_train, Y_train)
    
    print("Evaluating...")
    Y_pred = krr.predict(K_test)
    test_preds = torch.argmax(Y_pred, dim=1)
    
    test_acc = accuracy_score(y_test.cpu(), test_preds.cpu())
    test_f1 = f1_score(y_test.cpu(), test_preds.cpu(), average='macro')
    
    print(f"SWRST-v8 Test Accuracy: {test_acc:.4f}")
    print(f"SWRST-v8 Test Macro F1: {test_f1:.4f}")
    
    results = {
        "Test_Accuracy": float(test_acc),
        "Test_Macro_F1": float(test_f1)
    }
    
    out_file = r'C:\temp\ECG_Benchmark\results\swrst_v8_results.json'
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=4)
        
if __name__ == '__main__':
    main()
