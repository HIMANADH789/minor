import torch
import matplotlib.pyplot as plt
import os
from typing import Dict, Any, List

class TransportVisualizer:
    def __init__(self, output_dir: str = "transport_examples"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def visualize(self, P_A: torch.Tensor, P_B: torch.Tensor, label_A: int, label_B: int, sample_idx: int):
        """
        Visualizes Transport Plans and their differences.
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # A
        im1 = axes[0].imshow(P_A.cpu().numpy(), cmap='viridis', aspect='auto')
        axes[0].set_title(f"Class {label_A} Transport")
        fig.colorbar(im1, ax=axes[0])
        
        # B
        im2 = axes[1].imshow(P_B.cpu().numpy(), cmap='viridis', aspect='auto')
        axes[1].set_title(f"Class {label_B} Transport")
        fig.colorbar(im2, ax=axes[1])
        
        # Difference
        diff = torch.abs(P_A - P_B)
        im3 = axes[2].imshow(diff.cpu().numpy(), cmap='hot', aspect='auto')
        match_str = "Same" if label_A == label_B else "Diff"
        axes[2].set_title(f"Abs Difference ({match_str} Class)")
        fig.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"transport_diff_{match_str}_sample_{sample_idx}.png"))
        plt.close(fig)
