import torch
import torch.nn as nn

class TransportOperatorAlgebra(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, P):
        """
        P: (B, T, N, N)
        Computes Gram matrix G using normalized Hilbert-Schmidt:
        G_ij = Tr(P_i P_j) / (||P_i||_F ||P_j||_F)
        """
        B, T, N, _ = P.shape
        
        # Flatten P_t to vectors of size N*N
        P_flat = P.reshape(B, T, N * N) # (B, T, N*N)
        
        # Norms ||P_t||_F
        norms = torch.norm(P_flat, p=2, dim=-1, keepdim=True) # (B, T, 1)
        
        # Gram matrix: P_flat @ P_flat^T gives (B, T, T)
        unnormalized_G = torch.bmm(P_flat, P_flat.transpose(1, 2))
        
        # Outer product of norms
        norm_matrix = torch.bmm(norms, norms.transpose(1, 2)) + 1e-8
        
        G = unnormalized_G / norm_matrix # (B, T, T)
        
        return G
