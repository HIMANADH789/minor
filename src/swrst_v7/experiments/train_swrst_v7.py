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

# From V6 for H2 cache
from src.swrst_v6.spectrum.log_resolution import compute_log_resolution

# Phase II-III: SWRST-v7 State & Density
from src.swrst_v7.state.canonical_field import compute_canonical_field
from src.swrst_v7.state.spectral_residual import compute_spectral_residual
from src.swrst_v7.state.joint_density import compute_joint_density

# Phase IV-V: Adaptive JAP Metric
from src.swrst_v7.kernels.jap_metric import compute_jap_metric
from src.swrst_v7.kernels.adaptive_bandwidth import compute_adaptive_kernel

# Phase VI: Regressor
from src.swrst_v7.regressors.kernel_ridge import ClassConditionalKRR
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Final Hyperparameters
LAMBDA_A = 0.35
ETA = 0.15
GAMMA = 0.10
BETA = 0.50
LAMBDA_MIN = 1e-6
TAU = 1e-8

def cache_v7_features(X, C, E_m, epsilons, u_m, w_m, inv_du_m, BATCH_SIZE=64):
    N = X.size(0)
    device = X.device
    K = 12
    
    x_c = canonicalize_phase(X)
    mu = windowize_signal(x_c)
    
    Xi_can_list = []
    P_t_list = []
    KL_x_list = []
    
    with torch.no_grad():
        for i in range(0, N, BATCH_SIZE):
            mu_t = mu[i:i+BATCH_SIZE, :-1]
            mu_t_plus_1 = mu[i:i+BATCH_SIZE, 1:]
            
            # [B, T-1, K, K]
            Q_t = mu_t.unsqueeze(-1) * mu_t_plus_1.unsqueeze(-2)
            
            rho_t = compute_multi_density(Q_t, E_m)
            
            # V4 Frozen Transport
            P_sinkhorn, H_t, F_num = compute_multi_sinkhorn(mu_t, mu_t_plus_1, Q_t, rho_t, C, epsilons)
            
            # V7 Canonical Field (H4 packed)
            Xi_can_packed = compute_canonical_field(P_sinkhorn, Q_t, rho_t, tau=TAU)
            
            # V7 Spectral Residual (Upgrade U1)
            R_t = compute_spectral_residual(Xi_can_packed, K)
            
            # V7 Joint Density & KL Divergence (H5 fused)
            P_t, KL_x = compute_joint_density(
                Xi_can_packed, R_t, u_m, w_m, inv_du_m, H_t, lam_A=LAMBDA_A, eta=ETA
            )
            
            Xi_can_list.append(Xi_can_packed)
            P_t_list.append(P_t)
            KL_x_list.append(KL_x)
            
    return (
        torch.cat(Xi_can_list),
        torch.cat(P_t_list),
        torch.cat(KL_x_list)
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
    
    # H8: Static pinned epsilons
    epsilons = torch.tensor(get_resolution_grid(), dtype=torch.float32, device=device)
    
    # H2: Precompute log-grid inverse widths
    u_m, w_m, inv_du_m = compute_log_resolution(epsilons)
    
    print("Precomputing Train Features (SWRST-v7)...")
    Xi_can_tr, P_tr, KL_tr = cache_v7_features(X_train, C, E_m, epsilons, u_m, w_m, inv_du_m)
    
    print("Precomputing Test Features (SWRST-v7)...")
    Xi_can_te, P_te, KL_te = cache_v7_features(X_test, C, E_m, epsilons, u_m, w_m, inv_du_m)
    
    print("Computing Train JAP Metric...")
    d2_train = compute_jap_metric(Xi_can_tr, Xi_can_tr, P_tr, P_tr, w_m, gamma=GAMMA, tau=TAU)
    
    print("Computing Test JAP Metric...")
    d2_test = compute_jap_metric(Xi_can_te, Xi_can_tr, P_te, P_tr, w_m, gamma=GAMMA, tau=TAU)
    
    print("Applying Locally Adaptive Bandwidth...")
    K_train, sigma0_2 = compute_adaptive_kernel(d2_train, KL_tr, KL_tr, beta=BETA)
    K_test, _ = compute_adaptive_kernel(d2_test, KL_te, KL_tr, beta=BETA, sigma0_2=sigma0_2)
    
    num_classes = 5
    Y_train = torch.nn.functional.one_hot(y_train, num_classes=num_classes).float()
    
    print("Solving Profile-aware KRR...")
    krr = ClassConditionalKRR(lambda_c=1.0, lambda_min=LAMBDA_MIN)
    krr.fit(K_train, Y_train, KL_tr)
    
    print("Evaluating...")
    Y_pred = krr.predict(K_test)
    test_preds = torch.argmax(Y_pred, dim=1)
    
    test_acc = accuracy_score(y_test.cpu(), test_preds.cpu())
    test_f1 = f1_score(y_test.cpu(), test_preds.cpu(), average='macro')
    
    print(f"SWRST-v7 Test Accuracy: {test_acc:.4f}")
    print(f"SWRST-v7 Test Macro F1: {test_f1:.4f}")
    
    results = {
        "Test_Accuracy": float(test_acc),
        "Test_Macro_F1": float(test_f1)
    }
    
    out_file = r'C:\temp\ECG_Benchmark\results\swrst_v7_results.json'
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=4)
        
if __name__ == '__main__':
    main()
