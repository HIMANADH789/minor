import os
import json
import torch
import numpy as np
from typing import Dict, Any, List
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from swrst.config import SWRSTConfig
from swrst.model import SWRSTModel
from swrst.kernel.builder import build_kernel_matrix
from swrst.organization.locality import LocalityContext
from rich.console import Console

console = Console()

def fisher_score(X: np.ndarray, y: np.ndarray) -> float:
    classes = np.unique(y)
    if len(classes) < 2: return 0.0
    
    mean_all = np.mean(X, axis=0)
    S_b = 0.0
    S_w = 0.0
    
    for c in classes:
        X_c = X[y == c]
        n_c = X_c.shape[0]
        mean_c = np.mean(X_c, axis=0)
        
        S_b += n_c * np.sum((mean_c - mean_all)**2)
        S_w += np.sum((X_c - mean_c)**2)
        
    if S_w == 0: return 0.0
    return float(S_b / S_w)

def within_between_dist(X: np.ndarray, y: np.ndarray):
    # Use a small random subset if X is large
    n_samples = X.shape[0]
    if n_samples > 1000:
        idx = np.random.choice(n_samples, 1000, replace=False)
        X = X[idx]
        y = y[idx]
        
    dist_matrix = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    same_class = (y[:, None] == y[None, :])
    
    # Exclude diagonal
    np.fill_diagonal(same_class, False)
    
    diff_class = ~same_class
    np.fill_diagonal(diff_class, False)
    
    within = dist_matrix[same_class].mean() if same_class.any() else 0.0
    between = dist_matrix[diff_class].mean() if diff_class.any() else 0.0
    return float(within), float(between)

def compute_mi(X: np.ndarray, y: np.ndarray) -> float:
    # Digitize X to compute MI
    X_dig = np.digitize(X, bins=np.histogram_bin_edges(X, bins=10))
    mi_scores = [mutual_info_score(X_dig[:, i], y) for i in range(X.shape[1])]
    return float(np.mean(mi_scores))

def compute_entropy(X: np.ndarray) -> float:
    # X shape: (B, D). Compute mean entropy of normalized features per sample
    X_pos = np.abs(X)
    X_sum = X_pos.sum(axis=1, keepdims=True) + 1e-8
    X_prob = X_pos / X_sum
    ent = entropy(X_prob.T)
    return float(np.mean(ent))

def evaluate_stage(name: str, X: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    console.print(f"Evaluating Stage: [bold cyan]{name}[/bold cyan]")
    
    X_np = X.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    
    if X_np.ndim > 2:
        X_np = X_np.reshape(X_np.shape[0], -1)
        
    within, between = within_between_dist(X_np, y_np)
    fs = fisher_score(X_np, y_np)
    mi = compute_mi(X_np, y_np)
    ent = compute_entropy(X_np)
    
    passed = (between > within * 1.05) and (fs > 0.01)
    
    return {
        "Stage": name,
        "Within": within,
        "Between": between,
        "Fisher": fs,
        "MI": mi,
        "Entropy": ent,
        "PASS": passed
    }

def audit_representation():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_file = os.path.join(base_dir, "data", "ecg5000_balanced.npz")
    
    data = np.load(data_file)
    # Use train set for audit
    X_train = torch.tensor(data['X_train'][:2000], dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(data['y_train'][:2000], dtype=torch.long)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    
    config = SWRSTConfig()
    model = SWRSTModel(config).to(device)
    
    results = []
    
    # Stage 1: Raw Signal
    results.append(evaluate_stage("1. Raw Signal", X_train.squeeze(1), y_train))
    
    # Stage 3: STFT
    shared_stft = LocalityContext.compute_shared_stft(X_train)
    results.append(evaluate_stage("3. STFT (Time-Freq)", shared_stft, y_train))
    
    # Run model to get sequences
    with torch.no_grad():
        sequences = model(X_train)
        
    # We will pick the final observation for each sequence to evaluate downstream stages
    # Stage 5: Averaged spectrum / Normalized marginals
    # In SWRST-C Rep A, the marginal is the averaged spectrum
    # We can reconstruct it from the shared STFT and the final locality
    B, F, T = shared_stft.shape
    C_w_list = []
    P_list = []
    for b in range(B):
        seq = sequences[b]
        loc = seq[-1]['locality'] if seq else 1
        w_idx = min(loc, T)
        C_w = shared_stft[b, :, :w_idx].sum(dim=-1) / max(1, w_idx)
        # Normalize
        C_w = C_w / (C_w.sum() + 1e-8)
        C_w_list.append(C_w)
        
        P = seq[-1]['transport_plan']
        P_list.append(P)
        
    C_w_tensor = torch.stack(C_w_list)
    P_tensor = torch.stack(P_list)
    
    results.append(evaluate_stage("5. Averaged Spectrum Marginal", C_w_tensor, y_train))
    results.append(evaluate_stage("7. Transport Plans", P_tensor, y_train))
    
    # Stage 8: Kernel Input
    # We build the kernel matrix for this subset
    K = build_kernel_matrix(sequences, config)
    results.append(evaluate_stage("8. Kernel Matrix", K, y_train))
    
    # Export Report
    report_path = os.path.join(base_dir, "results", "balanced", "representation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# SWRST-C Representation Audit (Phase 1)\n\n")
        f.write("| Stage | Within | Between | Fisher | MI | Entropy | PASS/FAIL |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        
        for r in results:
            pass_str = "✅ PASS" if r["PASS"] else "❌ FAIL"
            f.write(f"| {r['Stage']} | {r['Within']:.4f} | {r['Between']:.4f} | {r['Fisher']:.4f} | {r['MI']:.4f} | {r['Entropy']:.4f} | {pass_str} |\n")
            
    console.print(f"[green]Saved audit report to {report_path}[/green]")
    
    # Export 10 random samples per class for visual inspection
    console.print("[cyan]Exporting 10 random transport plans per class...[/cyan]")
    visual_samples = {}
    classes = torch.unique(y_train)
    for c in classes:
        c_val = c.item()
        idx_c = torch.where(y_train == c)[0]
        n_samples = min(10, len(idx_c))
        sampled_idx = idx_c[torch.randperm(len(idx_c))[:n_samples]]
        
        visual_samples[f"Class_{c_val}"] = [
            P_tensor[i].cpu().numpy().tolist() for i in sampled_idx
        ]
        
    flow_path = os.path.join(base_dir, "results", "balanced", "representation_flow.json")
    with open(flow_path, "w") as f:
        json.dump({"metrics": results, "visual_samples": visual_samples}, f, indent=4)
    console.print(f"[green]Saved flow data to {flow_path}[/green]")

if __name__ == "__main__":
    audit_representation()
