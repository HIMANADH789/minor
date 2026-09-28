import torch
import numpy as np
import os
import json
import time
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score

from src.swrst.features.phase_canonicalizer import canonicalize_phase
from src.swrst.features.windowizer import windowize_signal
from src.swrst_v4.transport.resolution_grid import get_resolution_grid, precompute_em
from src.swrst_v4.transport.multi_density import compute_multi_density
from src.swrst_v4.transport.multi_sinkhorn import compute_multi_sinkhorn
from src.swrst_v8.statistics.log_ratio_centered import compute_log_ratio_fisher

from src.swrst_v10.statistics.shared_nystrom import SharedNystrom
from src.swrst_v10.statistics.local_kinematics import compute_local_kinematics
from src.swrst_v10.correspondence.local_ot import precompute_band_masks, compute_local_ot_chunk
from src.swrst_v10.correspondence.sparse_transport import compute_sparse_sharpening
from src.swrst_v10.kernels.local_correspondence import compute_sigma_m, precompute_S_mn, compute_sigma_xy, compute_profile_confidence, compute_local_correspondence_kernel
from src.swrst_v10.kernels.path_aggregation import compute_path_aggregation
from src.swrst_v10.regularizers.locality_score import compute_locality_score_chunk
from src.swrst_v10.regressors.kernel_ridge import ClassConditionalKRR

def process_chunk_kernel(
    Z_x, Z_y, p_x, p_y, sigma_xy, S_mn, M1, M2, soft_prior, c_x, c_y, chunk_size=32
):
    """
    Computes soft-aggregated local correspondence kernel.
    """
    Nx = Z_x.size(0)
    Ny = Z_y.size(0)
    
    K = torch.zeros((Nx, Ny), device=Z_x.device, dtype=torch.float32)
    L_x = torch.zeros((Nx, Ny), device=Z_x.device, dtype=torch.float32)
    
    # We precompute energy-based marginals for the OT radius adaptive check
    # E_x = ||Z_x(m)||^2
    # But wait, Z_x has size [Nx, T, M, d]. E_x is size [Nx, T, M]. We pass Z_x inside.
    # What about p_x and p_y for sigma_xy? We already computed sigma_xy.
    
    # We extract z from S (the first r dims) to compute E_x inside OT.
    # z is S_x[..., :r]
    r = (Z_x.shape[-1] - 4) // 5 # since dim = 5r + 4
    
    for i in range(0, Nx, chunk_size):
        end_i = min(i + chunk_size, Nx)
        Z_x_i = Z_x[i:end_i]
        z_x_i = Z_x_i[:, :, :, :r]
        c_x_i = c_x[i:end_i]
        
        for j in range(0, Ny, chunk_size):
            end_j = min(j + chunk_size, Ny)
            Z_y_j = Z_y[j:end_j]
            z_y_j = Z_y_j[:, :, :, :r]
            c_y_j = c_y[j:end_j]
            
            sigma_xy_ij = sigma_xy[i:end_i, j:end_j]
            
            # 1. Resolution OT Gamma_t
            Gamma_t = compute_local_ot_chunk(
                Z_x_i, Z_y_j, z_x_i, z_y_j, M1, M2, soft_prior
            )
            
            # 2. Sparse transport sharpening
            Gamma_t = compute_sparse_sharpening(Gamma_t)
            
            # 3. Locality score chunk
            L_x[i:end_i, j:end_j] = torch.sum(compute_locality_score_chunk(Gamma_t), dim=-1)
            
            # 4. Local correspondence kernel chunk K_t
            K_t = compute_local_correspondence_kernel(
                Z_x_i, Z_y_j, Gamma_t, S_mn, sigma_xy_ij, c_x_i, c_y_j
            )
            if torch.isnan(K_t).any():
                print(f"NaN in K_t at chunk {i}, {j}")
                print(f"Gamma NaN: {torch.isnan(Gamma_t).any().item()}, d2 denom NaN: {torch.isnan(S_mn).any().item()}")
            
            # 5. Path Aggregation K_xy
            K_chunk = compute_path_aggregation(K_t, Z_x_i)
            if torch.isnan(K_chunk).any():
                print(f"NaN in K_chunk at chunk {i}, {j}")
            
            K[i:end_i, j:end_j] = K_chunk
            
    # L_x is accumulated per x over all y
    L_x_mean = torch.mean(L_x, dim=1) # [Nx]
            
    return K, L_x_mean

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
    
    nystrom = SharedNystrom(r=r, num_classes=5, dim=dim, epsilons=epsilons)
    
    print("Accumulating Shared Nyström Covariance...")
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
                
                S = compute_local_kinematics(z)
                
                # We need p_x for sigma_xy and confidence
                # Let p(m) = sum_t ||z_t(m)||^2
                E_z = torch.sum(z**2, dim=(1, 3)) # [B, M]
                p_tilde = E_z / (torch.sum(E_z, dim=-1, keepdim=True) + 1e-8)
                
                S_list.append(S)
                p_list.append(p_tilde)
                
        return torch.cat(S_list), torch.cat(p_list)
        
    print("Precomputing Train Features (SWRST-v10)...")
    S_tr, p_tr = process_features(X_train)
    
    print("Precomputing Test Features (SWRST-v10)...")
    S_te, p_te = process_features(X_test)
    
    print("Precomputing Masks and Scales...")
    sigma_m = compute_sigma_m(S_tr)
    S_mn = precompute_S_mn(sigma_m)
    
    M1, M2, soft_prior = precompute_band_masks(M, device)
    
    sigma_xy_train = compute_sigma_xy(p_tr, p_tr)
    sigma_xy_test = compute_sigma_xy(p_te, p_tr)
    
    c_tr = compute_profile_confidence(p_tr)
    c_te = compute_profile_confidence(p_te)
    
    print("Computing Train Local Correspondence Kernel...")
    start = time.time()
    K_train, L_x_train = process_chunk_kernel(
        S_tr, S_tr, p_tr, p_tr, sigma_xy_train, S_mn, M1, M2, soft_prior, c_tr, c_tr, chunk_size=32
    )
    print(f"Train Kernel computed in {time.time() - start:.2f}s")
    
    print(f"K_train NaN count: {torch.isnan(K_train).sum().item()}")
    print(f"L_x_train NaN count: {torch.isnan(L_x_train).sum().item()}")
    
    print("Computing Test Local Correspondence Kernel...")
    K_test, _ = process_chunk_kernel(
        S_te, S_tr, p_te, p_tr, sigma_xy_test, S_mn, M1, M2, soft_prior, c_te, c_tr, chunk_size=32
    )
    
    Y_train = torch.nn.functional.one_hot(y_train, num_classes=5).float()
    
    print("Solving Class-Conditional KRR...")
    krr = ClassConditionalKRR()
    krr.fit(K_train, Y_train, L_x_train)
    
    print("Evaluating...")
    Y_pred = krr.predict(K_test)
    test_preds = torch.argmax(Y_pred, dim=1)
    
    test_acc = accuracy_score(y_test.cpu(), test_preds.cpu())
    test_f1 = f1_score(y_test.cpu(), test_preds.cpu(), average='macro')
    recalls = recall_score(y_test.cpu(), test_preds.cpu(), average=None).tolist()
    cm = confusion_matrix(y_test.cpu(), test_preds.cpu()).tolist()
    
    print(f"SWRST-v10 Test Accuracy: {test_acc:.4f}")
    print(f"SWRST-v10 Test Macro F1: {test_f1:.4f}")
    
    results = {
        "Test_Accuracy": float(test_acc),
        "Test_Macro_F1": float(test_f1),
        "Per_Class_Recall": recalls,
        "Confusion_Matrix": cm
    }
    
    out_file = r'C:\temp\ECG_Benchmark\results\swrst_v10_results.json'
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=4)
        
if __name__ == '__main__':
    main()
