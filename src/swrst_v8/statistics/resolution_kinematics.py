import torch

def compute_kinematics(z: torch.Tensor):
    """
    z: [B, T-1, M, r]
    
    Returns:
        D: [B, T-1, M, r]
        A: [B, T-1, M, r]
        J: [B, T-1, M, r]
    """
    B, T, M, r = z.shape
    
    # D: Centered derivative (Correction C5)
    D = torch.zeros_like(z)
    D[:, :, 1:-1, :] = (z[:, :, 2:, :] - z[:, :, :-2, :]) / 2.0
    D[:, :, 0, :] = z[:, :, 1, :] - z[:, :, 0, :] # Forward
    D[:, :, -1, :] = z[:, :, -1, :] - z[:, :, -2, :] # Backward
    
    # A: Resolution curvature (Upgrade U1)
    A = torch.zeros_like(z)
    A[:, :, 1:-1, :] = z[:, :, 2:, :] - 2.0 * z[:, :, 1:-1, :] + z[:, :, :-2, :]
    A[:, :, 0, :] = A[:, :, 1, :] # Replicate
    A[:, :, -1, :] = A[:, :, -2, :] # Replicate
    
    # J: Resolution jerk (Upgrade U1)
    J = torch.zeros_like(A)
    J[:, :, :-1, :] = A[:, :, 1:, :] - A[:, :, :-1, :]
    J[:, :, -1, :] = J[:, :, -2, :] # Replicate final
    
    return D, A, J
