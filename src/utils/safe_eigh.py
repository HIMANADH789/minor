import torch

class SafeEigh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        L, V = torch.linalg.eigh(A)
        if not torch.isfinite(L).all() or not torch.isfinite(V).all():
            U, S, Vh = torch.linalg.svd(A)
            L = S
            V = U
        
        # Absolute safeguard
        L = torch.nan_to_num(L, nan=1e-6, posinf=1e4, neginf=-1e4)
        V = torch.nan_to_num(V, nan=0.0, posinf=1.0, neginf=-1.0)
        
        ctx.save_for_backward(L, V)
        return L, V

    @staticmethod
    def backward(ctx, grad_L, grad_V):
        L, V = ctx.saved_tensors
        L_i = L.unsqueeze(-2)
        L_j = L.unsqueeze(-1)
        diff = L_j - L_i
        
        mask = torch.abs(diff) > 1e-5
        F = torch.zeros_like(diff)
        F[mask] = 1.0 / diff[mask]
        
        Vt_dV = torch.matmul(V.transpose(-2, -1), grad_V)
        Vt_dV_sym = 0.5 * (Vt_dV - Vt_dV.transpose(-2, -1))
        
        term2 = F * Vt_dV_sym
        term1 = torch.diag_embed(grad_L)
        
        dA = torch.matmul(V, torch.matmul(term1 + term2, V.transpose(-2, -1)))
        dA = torch.nan_to_num(dA, nan=0.0, posinf=1e4, neginf=-1e4)
        dA = torch.clamp(dA, min=-1e4, max=1e4)
        return dA

def safe_eigh(A):
    # Enforce exact symmetry
    A = (A + A.transpose(-2, -1)) * 0.5
    return SafeEigh.apply(A)
