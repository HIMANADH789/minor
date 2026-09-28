import torch
import torch.nn as nn
import torch.nn.functional as F

class RGTNLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        
    def forward(self, logits, targets, K_batch=None, Pi_seq=None, eps_seq=None, 
                occ_metric=None, spec_curvature=None, K_t_seq=None, evals_x=None,
                w_geom=0.0, w_persist=0.0, w_eps=0.0, w_occ=0.0, w_spec=0.0, w_sep=0.0):
        """
        L = CE + w_geom L_geom + w_persist L_persist + w_eps L_eps + w_occ L_occ + w_spec L_spec + w_sep L_sep
        """
        loss_ce = self.ce(logits, targets)
        
        # 1. L_geom (Kernel Target Alignment if K_batch is provided)
        loss_geom = 0.0
        if K_batch is not None and w_geom > 0:
            Y = (targets.unsqueeze(0) == targets.unsqueeze(1)).float()
            loss_geom = F.mse_loss(K_batch, Y)
            
        # 2. L_persist (Persistence Energy norm)
        loss_persist = 0.0
        if Pi_seq is not None and w_persist > 0:
            loss_persist = torch.mean(Pi_seq ** 2)
            
        # 3. L_eps (ERF Smoothness)
        loss_eps = 0.0
        if eps_seq is not None and w_eps > 0:
            d_eps = eps_seq[:, 1:] - eps_seq[:, :-1]
            loss_eps = torch.mean(d_eps ** 2)
            
        # 4. L_occ (Occupancy)
        loss_occ = 0.0
        if occ_metric is not None and w_occ > 0:
            loss_occ = torch.mean(occ_metric ** 2)
            
        # 5. L_spec (Spectral Curvature / Resolution Curvature K_t)
        loss_spec = 0.0
        if spec_curvature is not None and w_spec > 0:
            loss_spec = torch.mean(spec_curvature ** 2)
        elif K_t_seq is not None and w_spec > 0:
            loss_spec = torch.mean(K_t_seq ** 2)
            
        # 6. L_sep (Spectral Contrastive Separation)
        loss_sep = 0.0
        if evals_x is not None and w_sep > 0:
            B = evals_x.size(0)
            Y = (targets.unsqueeze(0) == targets.unsqueeze(1)).float()
            # L_sep = sum_{i,j} y_ij ||lambda_i - lambda_j||^2 - (1 - y_ij) m
            diff = evals_x.unsqueeze(1) - evals_x.unsqueeze(0) # (B, B, K)
            dist_sq = torch.sum(diff ** 2, dim=-1) # (B, B)
            m = 1.0 # Margin
            sep = Y * dist_sq - (1 - Y) * F.relu(m - dist_sq)
            loss_sep = torch.mean(sep)
            
        total_loss = loss_ce + w_geom * loss_geom + w_persist * loss_persist + \
                     w_eps * loss_eps + w_occ * loss_occ + w_spec * loss_spec + w_sep * loss_sep
        
        return total_loss, {
            'ce': loss_ce.item(),
            'geom': loss_geom.item() if isinstance(loss_geom, torch.Tensor) else 0.0,
            'persist': loss_persist.item() if isinstance(loss_persist, torch.Tensor) else 0.0,
            'eps': loss_eps.item() if isinstance(loss_eps, torch.Tensor) else 0.0,
            'occ': loss_occ.item() if isinstance(loss_occ, torch.Tensor) else 0.0,
            'spec': loss_spec.item() if isinstance(loss_spec, torch.Tensor) else 0.0,
            'sep': loss_sep.item() if isinstance(loss_sep, torch.Tensor) else 0.0
        }
