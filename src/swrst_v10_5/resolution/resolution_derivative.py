import torch

def pack_upper_triangular(X: torch.Tensor):
    """
    H4: Upper triangular packing before Hellinger.
    X: [..., K, K]
    Returns:
        X_packed: [..., K*(K+1)//2]
    """
    K = X.shape[-1]
    row, col = torch.triu_indices(K, K)
    return X[..., row, col]

def compute_resolution_kinematics(Xi: torch.Tensor, eps_sorted: torch.Tensor):
    """
    H3: Fused derivative + curvature pass. One pass, avoid two traversals.
    Xi: [B, T, M, K, K]
    eps_sorted: [M]
    
    Returns:
        Xi_packed: [B, T, M, K*(K+1)//2]
        D_packed:  [B, T, M, K*(K+1)//2]
        A_packed:  [B, T, M, K*(K+1)//2]
    """
    # H2: Precompute log eps
    u = torch.log(eps_sorted) # [M]
    
    D = torch.zeros_like(Xi)
    A = torch.zeros_like(Xi)
    
    # D_eps = Delta Xi / Delta u
    # Centered for interior
    du_centered = u[2:] - u[:-2] # [M-2]
    D[:, :, 1:-1] = (Xi[:, :, 2:] - Xi[:, :, :-2]) / du_centered.view(1, 1, -1, 1, 1)
    
    # Forward/backward for boundaries
    du_forward = u[1] - u[0]
    D[:, :, 0] = (Xi[:, :, 1] - Xi[:, :, 0]) / du_forward
    
    du_backward = u[-1] - u[-2]
    D[:, :, -1] = (Xi[:, :, -1] - Xi[:, :, -2]) / du_backward
    
    # A_eps = (Xi_{m+1} - 2Xi_m + Xi_{m-1}) / (Delta u)^2
    # We use Delta u = (u_{m+1} - u_{m-1}) / 2 for the centered approximation
    du_avg = du_centered / 2.0
    du_sq = du_avg ** 2
    
    A[:, :, 1:-1] = (Xi[:, :, 2:] - 2.0 * Xi[:, :, 1:-1] + Xi[:, :, :-2]) / du_sq.view(1, 1, -1, 1, 1)
    
    # Boundaries (forward/backward second derivative approx)
    # Using simple replication or forward diff of D
    A[:, :, 0] = (D[:, :, 1] - D[:, :, 0]) / du_forward
    A[:, :, -1] = (D[:, :, -1] - D[:, :, -2]) / du_backward
    
    # H4: Upper triangular packing
    Xi_packed = pack_upper_triangular(Xi)
    D_packed = pack_upper_triangular(D)
    A_packed = pack_upper_triangular(A)
    
    return Xi_packed, D_packed, A_packed
