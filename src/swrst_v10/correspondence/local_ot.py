import torch

def precompute_band_masks(M: int, device: torch.device):
    """
    H3: Band mask precomputed for delta=1 and delta=2.
    """
    m_idx = torch.arange(M, device=device).unsqueeze(1)
    n_idx = torch.arange(M, device=device).unsqueeze(0)
    dist = torch.abs(m_idx - n_idx)
    
    M1 = (dist <= 1)
    M2 = (dist <= 2)
    
    # U3: Locality-aware OT prior distance matrix
    # lambda * |m - n|
    soft_prior = 0.15 * dist.float()
    
    return M1, M2, soft_prior

def compute_local_ot_chunk(
    S_x: torch.Tensor, S_y: torch.Tensor,
    z_x: torch.Tensor, z_y: torch.Tensor,
    M1: torch.Tensor, M2: torch.Tensor, soft_prior: torch.Tensor,
    tau: float = 1e-8, epsilon: float = 0.03, num_iters: int = 5
):
    """
    S_x, z_x: [Nx, T, M, d]
    S_y, z_y: [Ny, T, M, d]
    
    Returns:
        Gamma: [Nx, Ny, T, M, M]
    """
    Nx, T, M, d = S_x.shape
    Ny = S_y.shape[0]
    
    # Compute normalized A_t(m,n) = <S_x, S_y>
    norm_x = torch.norm(S_x, dim=-1, keepdim=True) + tau
    norm_y = torch.norm(S_y, dim=-1, keepdim=True) + tau
    S_x_norm = S_x / norm_x
    S_y_norm = S_y / norm_y
    
    # H5: Batched GEMM for A(m,n)
    S_x_flat = S_x_norm.permute(1, 0, 2, 3).reshape(T, Nx*M, d)
    S_y_flat = S_y_norm.permute(1, 0, 2, 3).reshape(T, Ny*M, d)
    
    A_flat = torch.bmm(S_x_flat, S_y_flat.transpose(1, 2)) # [T, Nx*M, Ny*M]
    A = A_flat.view(T, Nx, M, Ny, M).permute(1, 3, 0, 2, 4) # [Nx, Ny, T, M, M]
    
    # U3: Bias diagonal softly before mask
    # A_t(m,n) <- A_t(m,n) - lambda |m - n|
    A = A - soft_prior.view(1, 1, 1, M, M)
    
    # C5: Adaptive locality radius delta_t
    # p_x(m) = ||z_x(m)||^2 / sum_m
    E_x = torch.sum(z_x**2, dim=-1) # [Nx, T, M]
    p_x_t = E_x / (torch.sum(E_x, dim=-1, keepdim=True) + tau) # [Nx, T, M]
    var_x = torch.var(p_x_t, dim=-1, unbiased=False) # [Nx, T]
    
    # delta_x = 1 + 1[Var_m(p_x) > 0.15]
    use_M2 = (var_x > 0.15) # [Nx, T]
    
    # Construct mask per [Nx, Ny, T, M, M]
    # We expand use_M2 to [Nx, Ny, T, 1, 1]
    use_M2_exp = use_M2.view(Nx, 1, T, 1, 1).expand(Nx, Ny, T, 1, 1)
    
    mask = torch.where(
        use_M2_exp,
        M2.view(1, 1, 1, M, M).expand(Nx, Ny, T, M, M),
        M1.view(1, 1, 1, M, M).expand(Nx, Ny, T, M, M)
    )
    
    # Mask outside band with -inf
    A_masked = torch.where(mask, A, torch.tensor(float('-inf'), device=A.device))
    
    # C_t = 1 - A_t (inside band, outside is inf)
    C_t = 1.0 - A_masked
    
    # H4 / H6: Chunked OT, log-domain Sinkhorn
    B = Nx * Ny * T
    C_flat = C_t.reshape(B, M, M)
    
    # Marginals (uniform for simplicity, or based on p_x/p_y)
    # The user didn't specify OT marginals, we use 1/M.
    log_a = torch.log(torch.full((B, M), 1.0/M, device=A.device))
    log_b = torch.log(torch.full((B, M), 1.0/M, device=A.device))
    
    K_log = -C_flat / epsilon
    
    u = torch.zeros_like(log_a)
    v = torch.zeros_like(log_b)
    
    for _ in range(num_iters):
        val = K_log + v.unsqueeze(1)
        u = log_a - torch.logsumexp(val, dim=2)
        val = K_log + u.unsqueeze(2)
        v = log_b - torch.logsumexp(val, dim=1)
        
    log_Gamma = u.unsqueeze(2) + K_log + v.unsqueeze(1)
    Gamma_flat = torch.exp(log_Gamma)
    
    Gamma = Gamma_flat.view(Nx, Ny, T, M, M)
    
    return Gamma
