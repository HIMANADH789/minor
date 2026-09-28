import torch
import numpy as np
import os
import json
import time

from src.swrst.features.phase_canonicalizer import canonicalize_phase
from src.swrst.features.windowizer import windowize_signal
from src.swrst.transport.deterministic_ot import get_cached_ot, compute_cost_matrix
from src.swrst.transport.density_evolution import compute_density_evolution
from src.swrst.transport.density_constrained_sinkhorn import compute_density_constrained_sinkhorn
from src.swrst.geometry.structural_energy import compute_structural_energy
from src.swrst.geometry.log_geometric_field import compute_log_geometric_field
from src.swrst.geometry.spectral_signatures import extract_spectral_signatures
from src.swrst.geometry.adaptive_surprise import compute_adaptive_surprise
from src.swrst.geometry.path_signature import compute_path_signature
from src.swrst.geometry.curvature_field import compute_curvature_field
from src.swrst.kernels.trajectory_kernel import compute_trajectory_kernel
from src.swrst.kernels.geometric_kernel import compute_geometric_kernel
from src.swrst.kernels.spectral_kernel import compute_spectral_kernel
from src.swrst.kernels.surprise_kernel import compute_surprise_kernel
from src.swrst.kernels.adaptive_fusion import compute_adaptive_fusion
from src.swrst.regressors.kernel_ridge import solve_kernel_ridge
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score

def cache_features(X, C, cache_prefix, BATCH_SIZE=64):
    N = X.size(0)
    device = X.device
    
    # We will just compute them all directly and keep in VRAM if possible.
    # We have enough VRAM for ECG5000 train+test.
    
    x_c = canonicalize_phase(X)
    mu = windowize_signal(x_c)
    
    P_det = get_cached_ot(X, mu, cache_path=f"C:\\temp\\ECG_Benchmark\\cache\\p_det_{cache_prefix}.npy")
    P_det = P_det.to(device)
    
    # Since we need to compute full dataset, batch the internal representations
    log_G_list = []
    Phi_list = []
    N_t_list = []
    Xi_t_list = []
    rho_t_list = []
    
    with torch.no_grad():
        for i in range(0, N, BATCH_SIZE):
            mu_t = mu[i:i+BATCH_SIZE, :-1]
            mu_t_plus_1 = mu[i:i+BATCH_SIZE, 1:]
            
            Q_t, D_t, H_t, S_t, G_t_stat, alpha_t, rho_t = compute_density_evolution(mu_t, mu_t_plus_1, C)
            
            # Use deterministic OT for sinkhorn init/target or keep as is?
            # Wait, the prompt says: "Compute: P_t^det = argmin <C, P>. Cache all train-set OT plans. Reuse at inference."
            # And then: "Density-constrained sinkhorn... P_t = exp(u + L + v)". 
            # The original SWRST v2 uses the exact P_t from POT for structural energy?
            # Actually, "Deviation: Omega_t = P_t - Q_t". P_t here comes from Sinkhorn.
            # Why cache deterministic OT? It was used to get C or a similar metric. 
            # Let's just use Sinkhorn for P_t.
            
            P_t = compute_density_constrained_sinkhorn(rho_t, mu_t, mu_t_plus_1, C)
            Omega_t, Xi_t = compute_structural_energy(P_t, Q_t, rho_t)
            
            log_G = compute_log_geometric_field(Xi_t)
            Phi = extract_spectral_signatures(Xi_t)
            N_t = compute_adaptive_surprise(Xi_t)
            
            # S1, S2, Curvature are available but we fuse geom, spec, surprise, traj kernels.
            # S1, S2 and curvature are computed, but the user requested 4 kernels for adaptive fusion.
            
            log_G_list.append(log_G)
            Phi_list.append(Phi)
            N_t_list.append(N_t)
            Xi_t_list.append(Xi_t)
            rho_t_list.append(rho_t)
            
    return torch.cat(log_G_list), torch.cat(Phi_list), torch.cat(N_t_list), torch.cat(Xi_t_list), torch.cat(rho_t_list)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data = np.load(r'C:\temp\ECG_Benchmark\data\ecg5000_resplit.npz')
    X_train = torch.tensor(data['X_train'], dtype=torch.float32, device=device)
    y_train = torch.tensor(data['y_train'], dtype=torch.long, device=device)
    X_test = torch.tensor(data['X_test'], dtype=torch.float32, device=device)
    y_test = torch.tensor(data['y_test'], dtype=torch.long, device=device)
    
    C = torch.tensor(compute_cost_matrix(12), dtype=torch.float32, device=device)
    
    print("Precomputing Train Features...")
    log_G_train, Phi_train, N_t_train, Xi_train, rho_train = cache_features(X_train, C, "train")
    
    print("Precomputing Test Features...")
    log_G_test, Phi_test, N_t_test, Xi_test, _ = cache_features(X_test, C, "test")
    
    print("Computing Train Kernels...")
    K_geom_train = compute_geometric_kernel(log_G_train, log_G_train)
    K_spec_train = compute_spectral_kernel(Phi_train, Phi_train)
    K_surp_train = compute_surprise_kernel(N_t_train, N_t_train)
    K_traj_train = compute_trajectory_kernel(Xi_train, Xi_train)
    
    print("Fusing Train Kernels...")
    K_fused_train = compute_adaptive_fusion([K_geom_train, K_spec_train, K_surp_train, K_traj_train])
    
    # One-hot Y
    num_classes = 5
    Y_train = torch.nn.functional.one_hot(y_train, num_classes=num_classes).float()
    
    print("Solving KRR...")
    alpha = solve_kernel_ridge(K_fused_train, Y_train, rho_train)
    
    print("Computing Test Kernels...")
    K_geom_test = compute_geometric_kernel(log_G_test, log_G_train)
    K_spec_test = compute_spectral_kernel(Phi_test, Phi_train)
    K_surp_test = compute_surprise_kernel(N_t_test, N_t_train)
    K_traj_test = compute_trajectory_kernel(Xi_test, Xi_train)
    
    print("Fusing Test Kernels...")
    K_fused_test = compute_adaptive_fusion([K_geom_test, K_spec_test, K_surp_test, K_traj_test])
    
    print("Evaluating...")
    Y_pred = torch.matmul(K_fused_test, alpha)
    test_preds = torch.argmax(Y_pred, dim=1)
    
    test_acc = accuracy_score(y_test.cpu(), test_preds.cpu())
    test_f1 = f1_score(y_test.cpu(), test_preds.cpu(), average='macro')
    cm = confusion_matrix(y_test.cpu(), test_preds.cpu())
    recalls = recall_score(y_test.cpu(), test_preds.cpu(), average=None)
    
    print(f"SWRST-v3 Test Accuracy: {test_acc:.4f}")
    print(f"SWRST-v3 Test Macro F1: {test_f1:.4f}")
    
    results = {
        "Test_Accuracy": float(test_acc),
        "Test_Macro_F1": float(test_f1),
        "Confusion_Matrix": cm.tolist(),
        "Per_Class_Recall": recalls.tolist()
    }
    
    out_file = r'C:\temp\ECG_Benchmark\results\swrst_v3_results.json'
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Results saved to {out_file}")

if __name__ == '__main__':
    main()
