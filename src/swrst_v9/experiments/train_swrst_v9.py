import torch
import numpy as np
import os
import json
import time

from src.swrst.features.phase_canonicalizer import canonicalize_phase
from src.swrst.features.windowizer import windowize_signal
from src.swrst_v4.transport.resolution_grid import get_resolution_grid, precompute_em
from src.swrst_v4.transport.multi_density import compute_multi_density
from src.swrst_v4.transport.multi_sinkhorn import compute_multi_sinkhorn
from src.swrst_v8.statistics.log_ratio_centered import compute_log_ratio_fisher

from src.swrst_v9.statistics.per_resolution_nystrom import PerResolutionNystrom
from src.swrst_v9.state.spectral_state import compute_spectral_state
from src.swrst_v9.state.discriminative_profile import compute_discriminative_profile
from src.swrst_v9.correspondence.cross_resolution_matrix import compute_cost_matrix_chunk
from src.swrst_v9.correspondence.resolution_ot import compute_resolution_ot
from src.swrst_v9.kernels.adaptive_bandwidth import compute_adaptive_bandwidth
from src.swrst_v9.kernels.correspondence_kernel import compute_sigma2_m, precompute_S_mn, compute_correspondence_kernel_chunk
from src.swrst_v9.kernels.soft_aggregation import compute_soft_aggregation
from src.swrst_v9.regressors.kernel_ridge import ClassConditionalKRR

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score

def process_chunk_kernel(
    Z_x, Z_y, p_x, p_y, sigma2_xy, S_mn, chunk_size=32
):
    """
    Computes soft-aggregated correspondence kernel progressively to save memory.
    """
    Nx = Z_x.size(0)
    Ny = Z_y.size(0)
    
    K = torch.zeros((Nx, Ny), device=Z_x.device, dtype=torch.float32)
    
    for i in range(0, Nx, chunk_size):
        end_i = min(i + chunk_size, Nx)
        Z_x_i = Z_x[i:end_i]
        p_x_i = p_x[i:end_i]
        
        for j in range(0, Ny, chunk_size):
            end_j = min(j + chunk_size, Ny)
            Z_y_j = Z_y[j:end_j]
            p_y_j = p_y[j:end_j]
            
            sigma2_xy_ij = sigma2_xy[i:end_i, j:end_j]
            
            # 1. Cost matrix C_t
            C_t = compute_cost_matrix_chunk(Z_x_i, Z_y_j)
            
            # 2. Resolution OT Gamma_t
            Gamma_t = compute_resolution_ot(C_t, p_x_i, p_y_j)
            
            # 3. Log-domain kernel chunk logK_t
            logK_t = compute_correspondence_kernel_chunk(
                Z_x_i, Z_y_j, Gamma_t, S_mn, sigma2_xy_ij
            )
            
            # 4. Soft Aggregation K_xy
            K_chunk = compute_soft_aggregation(logK_t)
            
            K[i:end_i, j:end_j] = K_chunk
            
    return K

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data = np.load(r'C:\temp\ECG_Benchmark\data\ecg5000_resplit.npz')
    X_train = torch.tensor(data['X_train'], dtype=torch.float32, device=device)
    y_train = torch.tensor(data['y_train'], dtype=torch.long, device=device)
    X_test = torch.tensor(data['X_test'], dtype=torch.float32, device=device)
    y_test = torch.tensor(data['y_test'], dtype=torch.long, device=device)
    
    K_bins = 12
    C, E_m = precompute_em(K_bins, device)
    epsilons = torch.tensor(get_resolution_grid(), dtype=torch.float32, device=device)
    
    BATCH_SIZE = 64
    r = 8
    M = 7
    dim = K_bins * (K_bins + 1) // 2
    
    nystrom = PerResolutionNystrom(M=M, r=r, num_classes=5, dim=dim)
    
    print("Accumulating Per-Resolution Nyström Covariance...")
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
            
            L_F = compute_log_ratio_fisher(P_sinkhorn, Q_t, K_bins)
            nystrom.accumulate(L_F, y_b)
            
    nystrom.fit()
    
    def process_features(X):
        N = X.size(0)
        S_list = []
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
                
                L_F = compute_log_ratio_fisher(P_sinkhorn, Q_t, K_bins)
                z = nystrom.project(L_F)
                
                S = compute_spectral_state(z)
                p_tilde, _ = compute_discriminative_profile(z)
                
                S_list.append(S)
                p_list.append(p_tilde)
                
        return torch.cat(S_list), torch.cat(p_list)
        
    print("Precomputing Train Features (SWRST-v9)...")
    S_tr, p_tr = process_features(X_train)
    
    print("Precomputing Test Features (SWRST-v9)...")
    S_te, p_te = process_features(X_test)
    
    print("Caching intrinsic bandwidth sigma_m^2...")
    sigma2_m = compute_sigma2_m(S_tr)
    S_mn = precompute_S_mn(sigma2_m)
    
    print("Computing Adaptive Bandwidth sigma_xy^2...")
    sigma2_xy_train = compute_adaptive_bandwidth(p_tr)
    sigma2_xy_test = compute_adaptive_bandwidth(p_tr, p_te)
    
    print("Computing Train Correspondence Kernel...")
    start = time.time()
    K_train = process_chunk_kernel(S_tr, S_tr, p_tr, p_tr, sigma2_xy_train, S_mn, chunk_size=32)
    print(f"Train Kernel computed in {time.time() - start:.2f}s")
    
    print("Computing Test Correspondence Kernel...")
    K_test = process_chunk_kernel(S_te, S_tr, p_te, p_tr, sigma2_xy_test, S_mn, chunk_size=32)
    
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
    
    print(f"SWRST-v9 Test Accuracy: {test_acc:.4f}")
    print(f"SWRST-v9 Test Macro F1: {test_f1:.4f}")
    
    results = {
        "Test_Accuracy": float(test_acc),
        "Test_Macro_F1": float(test_f1),
        "Per_Class_Recall": recalls,
        "Confusion_Matrix": cm
    }
    
    out_file = r'C:\temp\ECG_Benchmark\results\swrst_v9_results.json'
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=4)
        
if __name__ == '__main__':
    main()
