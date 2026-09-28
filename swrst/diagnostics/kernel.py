import torch
import json
import matplotlib.pyplot as plt
import os
from typing import Dict, Any
from rich.console import Console

console = Console()

class KernelAuditor:
    def __init__(self):
        pass
        
    def audit(self, K: torch.Tensor, name: str = "train") -> Dict[str, Any]:
        """
        Audits Kernel for PSD properties and saves histogram.
        """
        # Eigenvalues
        try:
            L_eig = torch.linalg.eigvalsh(K)
            min_eig = L_eig.min().item()
            max_eig = L_eig.max().item()
            is_psd = min_eig >= -1e-5
        except:
            min_eig = float('-inf')
            max_eig = float('inf')
            is_psd = False
            
        metrics = {
            'min_eigenvalue': min_eig,
            'max_eigenvalue': max_eig,
            'is_psd': is_psd
        }
        
        if not is_psd:
            console.print(f"[bold red]CRITICAL: {name} Kernel matrix is not PSD! Min eig: {min_eig}[/bold red]")
            
        # Histogram
        plt.figure(figsize=(8, 6))
        plt.hist(K.flatten().cpu().numpy(), bins=50, color='blue', alpha=0.7)
        plt.title(f"Kernel Values Histogram ({name})")
        plt.savefig(f"kernel_histogram_{name}.png")
        plt.close()
        
        with open(f"kernel_statistics_{name}.json", "w") as f:
            json.dump(metrics, f, indent=4)
            
        return metrics
