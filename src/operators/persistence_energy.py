import torch
import torch.nn as nn

class PersistenceEnergy(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, M_t_seq, P_t_seq=None, E_t_minus_1=None):
        """
        M_t_seq: Commutator sequence across time, shape (B, T, K_dim, K_dim)
        P_t_seq: Transport plan sequence across time, shape (B, T, K_dim, K_dim)
        
        Returns:
        E_t_seq: Energy across time, shape (B, T)
        Pi_t_seq: Persistence across time, shape (B, T)
        """
        # M_t * M_t^T
        M_MT = torch.matmul(M_t_seq, M_t_seq.transpose(-2, -1))
        trace_M = torch.diagonal(M_MT, dim1=-2, dim2=-1).sum(-1)
        
        # P_t * P_t^T
        if P_t_seq is not None:
            P_PT = torch.matmul(P_t_seq, P_t_seq.transpose(-2, -1))
            trace_P = torch.diagonal(P_PT, dim1=-2, dim2=-1).sum(-1)
        else:
            trace_P = 0.0
            
        tau = 0.02
        
        # Energy (PATCH J)
        E_t_seq = torch.log(1.0 + trace_P + tau * trace_M) # (B, T)
        
        # Persistence
        Pi_t_seq = []
        
        for t in range(E_t_seq.size(1)):
            if t == 0:
                if E_t_minus_1 is None:
                    E_prev = torch.zeros_like(E_t_seq[:, 0])
                else:
                    E_prev = E_t_minus_1
            else:
                E_prev = E_t_seq[:, t-1]
                
            Pi_t = E_t_seq[:, t] - E_prev
            Pi_t_seq.append(Pi_t)
            
        Pi_t_seq = torch.stack(Pi_t_seq, dim=1) # (B, T)
        
        return E_t_seq, Pi_t_seq
