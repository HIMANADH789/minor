import torch
import numpy as np
import time
import json

from src.swrst.features.phase_canonicalizer import canonicalize_phase
from src.swrst.features.windowizer import windowize_signal
from src.swrst_v4.transport.resolution_grid import get_resolution_grid, precompute_em
from src.swrst_v4.transport.multi_density import compute_multi_density
from src.swrst_v4.transport.multi_sinkhorn import compute_multi_sinkhorn
from src.swrst_v4.transport.weighted_field import compute_weighted_field
from src.swrst_v4.transport.resolution_curvature import compute_resolution_curvature

from src.swrst_v5.state.unified_state import compute_unified_state
from src.swrst_v5.state.state_barycenter import compute_state_barycenter
from src.swrst_v5.state.state_normalizer import compute_state_normalizer
from src.swrst_v5.state.weight_field import compute_weight_field
from src.swrst_v5.kernels.unified_metric import compute_unified_metric_and_kernel
from src.swrst_v4.regressors.kernel_ridge import solve_kernel_ridge
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score

BATCH_SIZE = 32

def cache_features(X, C, E_m, epsilons):
    N = X.shape[0]
    
    Z_hat_list = []
    W_list = []
    h_list = []
    
    with torch.no_grad():
        x_c = canonicalize_phase(X)
        mu = windowize_signal(x_c)
        
        for i in range(0, N, BATCH_SIZE):
            mu_t = mu[i:i+BATCH_SIZE, :-1]
            mu_t_plus_1 = mu[i:i+BATCH_SIZE, 1:]
            
            Q_t = mu_t.unsqueeze(-1) * mu_t_plus_1.unsqueeze(-2)
            rho_t = compute_multi_density(Q_t, E_m)
            
            P_t, h_t, F_num = compute_multi_sinkhorn(mu_t, mu_t_plus_1, Q_t, rho_t, C, epsilons)
            
            Xi_t, _ = compute_weighted_field(rho_t, F_num)
            
            K_t = compute_resolution_curvature(Xi_t)
            
            Z_t = compute_unified_state(Xi_t, P_t, Q_t, K_t, h_t)
            
            Z_bar_t = compute_state_barycenter(Z_t)
            
            Z_hat_t = compute_state_normalizer(Z_bar_t, Z_t)
            
            W_t = compute_weight_field(Z_bar_t, Z_hat_t)
            
            Z_hat_list.append(Z_hat_t)
            W_list.append(W_t)
            h_list.append(h_t)
            
    return torch.cat(Z_hat_list), torch.cat(W_list), torch.cat(h_list)

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
    
    print("Precomputing Train Features (SWRST-v5 Unified State)...")
    Z_hat_train, W_train, h_train = cache_features(X_train, C, E_m, epsilons)
    
    print("Precomputing Test Features (SWRST-v5 Unified State)...")
    Z_hat_test, W_test, h_test = cache_features(X_test, C, E_m, epsilons)
    
    print("Computing Train Kernel...")
    K_train = compute_unified_metric_and_kernel(Z_hat_train, W_train, h_train, Z_hat_train, W_train, h_train)
    
    print("Computing Test Kernel...")
    K_test = compute_unified_metric_and_kernel(Z_hat_test, W_test, h_test, Z_hat_train, W_train, h_train)
    
    num_classes = 5
    Y_train = torch.nn.functional.one_hot(y_train, num_classes=num_classes).float()
    
    print("Solving KRR for unified kernel...")
    alpha = solve_kernel_ridge(K_train, Y_train, h_train)
    
    print("Evaluating...")
    test_preds = torch.matmul(K_test, alpha)
    test_preds_ind = torch.argmax(test_preds, dim=1)
    
    acc = accuracy_score(y_test.cpu(), test_preds_ind.cpu())
    f1 = f1_score(y_test.cpu(), test_preds_ind.cpu(), average='macro')
    
    print(f"SWRST-v5 Test Accuracy: {acc:.4f}")
    print(f"SWRST-v5 Test Macro F1: {f1:.4f}")
    
    with open('results/swrst_v5_results.json', 'w') as f:
        json.dump({'accuracy': acc, 'macro_f1': f1}, f, indent=4)
    print("Results saved to results/swrst_v5_results.json")

if __name__ == "__main__":
    main()
