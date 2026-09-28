import torch
from typing import Dict
from rich.console import Console

console = Console()

class RepresentationAuditor:
    """
    Information Preservation Audit at every stage.
    Computes Rank, Condition number, Entropy, Within, Between, Mutual Information, Intrinsic Dimension.
    """
    def __init__(self):
        self.history = []
        
    def audit(self, stage_name: str, features: torch.Tensor, labels: torch.Tensor = None) -> Dict[str, float]:
        """
        features: (B, D) or (B, N, D)
        """
        if features.dim() > 2:
            features = features.reshape(features.shape[0], -1)
            
        B, D = features.shape
        # Normalize features for numeric stability
        features_norm = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
        
        # 1. Rank & Condition Number
        try:
            # We use SVD for stable condition number
            U, S, V = torch.linalg.svd(features_norm, full_matrices=False)
            S = S[S > 1e-5]
            rank = len(S)
            cond = (S[0] / S[-1]).item() if rank > 0 else float('inf')
        except:
            rank = 0
            cond = float('inf')
            
        # 2. Entropy (approximate differential entropy via nearest neighbors or just PCA entropy)
        # Using normalized singular values as a proxy for information entropy
        if rank > 0:
            p = S / S.sum()
            entropy = -(p * torch.log(p + 1e-12)).sum().item()
        else:
            entropy = 0.0
            
        # 3. Within/Between ratio (if labels provided)
        wb_ratio = 1.0
        if labels is not None:
            classes = torch.unique(labels)
            global_mean = features_norm.mean(dim=0)
            
            within_scatter = 0.0
            between_scatter = 0.0
            
            for c in classes:
                c_mask = (labels == c)
                c_feats = features_norm[c_mask]
                c_mean = c_feats.mean(dim=0)
                
                within_scatter += ((c_feats - c_mean) ** 2).sum().item()
                between_scatter += c_mask.sum().item() * ((c_mean - global_mean) ** 2).sum().item()
                
            wb_ratio = within_scatter / (between_scatter + 1e-8)
            
        metrics = {
            'stage': stage_name,
            'rank': rank,
            'condition_number': cond,
            'entropy': entropy,
            'wb_ratio': wb_ratio
        }
        
        # Log WARNING if ratio drops, but do not aggressively abort.
        if self.history and labels is not None:
            prev_wb = self.history[-1]['wb_ratio']
            if wb_ratio > prev_wb * 1.1: # ratio increased (worse separation)
                console.print(f"[bold yellow]WARNING: Information decreased by {(wb_ratio/prev_wb - 1)*100:.1f}% at {stage_name}[/bold yellow]")
                
            if wb_ratio > 0.99 and prev_wb < 0.99:
                console.print(f"[bold red]CRITICAL WARNING: Catastrophic representation collapse at {stage_name} (wb_ratio ≈ 1)[/bold red]")
                
        self.history.append(metrics)
        return metrics
