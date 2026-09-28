import torch
from typing import List, Dict, Any
from .correspondence import compute_chunked_cck
from ..config import SWRSTConfig
from rich.console import Console
from rich.progress import track

console = Console()

def extract_sequence_chunk(sequences: List[List[Dict[str, Any]]], start: int, end: int, max_L: int, max_F: int):
    """ Converts a chunk of variable-length sequences to padded tensors dynamically """
    chunk_seqs = sequences[start:end]
    chunk_size = len(chunk_seqs)
    
    P = torch.zeros((chunk_size, max_L, max_F, max_F), dtype=torch.float32)
    loc = torch.zeros((chunk_size, max_L), dtype=torch.float32)
    nov = torch.zeros((chunk_size, max_L), dtype=torch.float32)
    conf = torch.zeros((chunk_size, max_L), dtype=torch.float32)
    mask = torch.zeros((chunk_size, max_L), dtype=torch.bool)
    
    for i, seq in enumerate(chunk_seqs):
        L = len(seq)
        if L > 0:
            padded_plans = []
            for obs in seq:
                p = obs['transport_plan'].cpu()
                if p.is_sparse:
                    p = p.to_dense()
                p = p.float()
                f_dim = p.shape[-1]
                if f_dim < max_F:
                    p_pad = torch.zeros((max_F, max_F), dtype=torch.float32)
                    p_pad[:f_dim, :f_dim] = p
                    padded_plans.append(p_pad)
                else:
                    padded_plans.append(p)
                    
            P[i, :L] = torch.stack(padded_plans)
            loc[i, :L] = torch.tensor([obs['locality'] for obs in seq], dtype=torch.float32)
            nov[i, :L] = torch.tensor([obs['information_gain'] for obs in seq], dtype=torch.float32)
            conf[i, :L] = torch.tensor([obs['confidence'] for obs in seq], dtype=torch.float32)
            mask[i, :L] = True
            
    return P, loc, nov, conf, mask

def estimate_gammas(tensors, config: SWRSTConfig, num_samples: int = 512) -> tuple:
    P, loc, nov, conf, mask = tensors
    N = P.shape[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    num_samples = min(num_samples, N)
    idx = torch.randperm(N)[:num_samples]
    
    P_sub = P[idx].to(device)
    
    # We estimate gamma by using gamma=1.0 and computing avg distance from the CCK logic
    # Actually, CCK normalizes the distances, so gamma can just be 1.0.
    # The previous implementation assumed distance wasn't normalized.
    # Since our `compute_chunked_cck` normalizes distance to [0,1], gamma can be fixed!
    # A standard choice for normalized distance is gamma=5.0 or similar.
    # Let's set gamma = 5.0 explicitly and avoid full pairwise estimation.
    return 5.0, 5.0

def build_kernel_matrix(
    sequences: List[List[Dict[str, Any]]], 
    config: SWRSTConfig,
    gamma: float = None
) -> torch.Tensor:
    
    N = len(sequences)
    if N == 0:
        return torch.empty((0, 0))
        
    max_L = max(len(s) for s in sequences)
    max_F = 0
    for seq in sequences:
        for obs in seq:
            p = obs['transport_plan']
            f_dim = p.shape[-1] if not p.is_sparse else p.size(-1)
            max_F = max(max_F, f_dim)
            
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    K = torch.zeros((N, N), dtype=torch.float32).pin_memory() if torch.cuda.is_available() else torch.zeros((N, N), dtype=torch.float32)
    
    chunk_size = config.kernel_chunk_size
    console.print(f"[cyan]Building {N}x{N} Kernel matrix (Streaming Upper Triangle, chunk={chunk_size})...[/cyan]")
    
    stream = torch.cuda.Stream() if torch.cuda.is_available() else None
    
    with torch.no_grad():
        for i in track(range(0, N, chunk_size), description="Kernel rows"):
            i_end = min(i + chunk_size, N)
            
            P_i, loc_i, nov_i, conf_i, mask_i = extract_sequence_chunk(sequences, i, i_end, max_L, max_F)
            
            P_i = P_i.to(device, non_blocking=True)
            loc_i = loc_i.to(device, non_blocking=True)
            nov_i = nov_i.to(device, non_blocking=True)
            conf_i = conf_i.to(device, non_blocking=True)
            mask_i = mask_i.to(device, non_blocking=True)
            
            for j in range(i, N, chunk_size):
                j_end = min(j + chunk_size, N)
                
                P_j, loc_j, nov_j, conf_j, mask_j = extract_sequence_chunk(sequences, j, j_end, max_L, max_F)
                
                if stream:
                    with torch.cuda.stream(stream):
                        P_j = P_j.to(device, non_blocking=True)
                        loc_j = loc_j.to(device, non_blocking=True)
                        nov_j = nov_j.to(device, non_blocking=True)
                        conf_j = conf_j.to(device, non_blocking=True)
                        mask_j = mask_j.to(device, non_blocking=True)
                        
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            K_ij = compute_chunked_cck(
                                P_i, P_j, loc_i, loc_j, conf_i, conf_j, mask_i, mask_j, config
                            )
                        
                        K[i:i_end, j:j_end] = K_ij.cpu()
                else:
                    P_j = P_j.to(device)
                    loc_j = loc_j.to(device)
                    nov_j = nov_j.to(device)
                    conf_j = conf_j.to(device)
                    mask_j = mask_j.to(device)
                    
                    K_ij = compute_chunked_cck(
                        P_i, P_j, loc_i, loc_j, conf_i, conf_j, mask_i, mask_j, config
                    )
                    K[i:i_end, j:j_end] = K_ij
                    
        if stream:
            torch.cuda.synchronize()
            
    # Mirror lower triangle
    for i in range(N):
        for j in range(i+1, N):
            K[j, i] = K[i, j]
                    
    # Normalize Kernel: K(x, y) = K(x, y) / sqrt(K(x, x) * K(y, y))
    d = torch.diag(K)
    d_sqrt = torch.sqrt(d.clamp(min=1e-8))
    
    K = K / (d_sqrt.unsqueeze(1) * d_sqrt.unsqueeze(0))
    
    # Mathematical Projection to PSD cone to fix any floating-point roundoff errors
    L_eig, V_eig = torch.linalg.eigh(K.to(device) if torch.cuda.is_available() else K)
    if L_eig.min() < 0:
        L_eig = L_eig.clamp(min=1e-7)
        K_psd = V_eig @ torch.diag(L_eig) @ V_eig.T
        # Re-normalize diagonal to exactly 1.0
        d_new = torch.diag(K_psd)
        d_sqrt_new = torch.sqrt(d_new.clamp(min=1e-8))
        K = (K_psd / (d_sqrt_new.unsqueeze(1) * d_sqrt_new.unsqueeze(0))).cpu()
        
    # Ensure exact symmetry
    K = (K + K.T) / 2.0
    
    return K.to(device)

def build_cross_kernel_matrix(
    seq_A: List[List[Dict[str, Any]]], 
    seq_B: List[List[Dict[str, Any]]], 
    config: SWRSTConfig,
    gamma: float = None
) -> torch.Tensor:
    M = len(seq_A)
    N = len(seq_B)
    
    if M == 0 or N == 0:
        return torch.empty((0, 0))
        
    max_L = max(max((len(s) for s in seq_A), default=0), max((len(s) for s in seq_B), default=0))
    max_F = 0
    for sequences in (seq_A, seq_B):
        for seq in sequences:
            for obs in seq:
                p = obs['transport_plan']
                f_dim = p.shape[-1] if not p.is_sparse else p.size(-1)
                max_F = max(max_F, f_dim)
                
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    K = torch.zeros((M, N), device=device, dtype=torch.float32)
    
    chunk_size = config.kernel_chunk_size
    console.print(f"[cyan]Building {M}x{N} Cross-Kernel matrix in chunks...[/cyan]")
    
    with torch.no_grad():
        for i in track(range(0, M, chunk_size), description="Cross-Kernel rows"):
            i_end = min(i + chunk_size, M)
            
            P_i, loc_i, nov_i, conf_i, mask_i = extract_sequence_chunk(seq_A, i, i_end, max_L, max_F)
            P_i = P_i.to(device)
            loc_i = loc_i.to(device)
            nov_i = nov_i.to(device)
            conf_i = conf_i.to(device)
            mask_i = mask_i.to(device)
            
            for j in range(0, N, chunk_size):
                j_end = min(j + chunk_size, N)
                
                P_j, loc_j, nov_j, conf_j, mask_j = extract_sequence_chunk(seq_B, j, j_end, max_L, max_F)
                P_j = P_j.to(device)
                loc_j = loc_j.to(device)
                nov_j = nov_j.to(device)
                conf_j = conf_j.to(device)
                mask_j = mask_j.to(device)
                
                K_ij = compute_chunked_cck(
                    P_i, P_j,
                    loc_i, loc_j,
                    conf_i, conf_j,
                    mask_i, mask_j,
                    config
                )
                
                K[i:i_end, j:j_end] = K_ij
                
    return K

