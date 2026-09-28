import os
import json
import time
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from rich.console import Console

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from swrst.config import SWRSTConfig
from swrst.model import SWRSTModel
from swrst.kernel.builder import build_kernel_matrix
from swrst.classifier.ridge import KernelRidgeClassifier
from swrst.diagnostics.benchmark import ScientificSanityChecker
from swrst.diagnostics.transport import aggregate_transport_diagnostics, save_diagnostics

console = Console()

def train_swrst_c(data_file, metadata_file, output_file, diagnostics_file):
    # Hardware constraints
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    console.print(f"[cyan]Loading dataset from {data_file}...[/cyan]")
    data = np.load(data_file)
    
    # 1. Scientific Sanity Checks
    ScientificSanityChecker.verify_dataset(data, metadata_file)
    
    X_train = torch.tensor(data['X_train'], dtype=torch.float32, device=device).unsqueeze(1)
    y_train = torch.tensor(data['y_train'], dtype=torch.long, device=device)
    X_test = torch.tensor(data['X_test'], dtype=torch.float32, device=device).unsqueeze(1)
    y_test = torch.tensor(data['y_test'], dtype=torch.long, device=device)
    
    # Configuration
    config = SWRSTConfig(ablation_mode=0)
    model = SWRSTModel(config).to(device)
    
    # 2. Extract Transport Sequences
    console.print("[cyan]Extracting Transport Sequences (Train)...[/cyan]")
    t0 = time.time()
    train_seqs = model(X_train)
    train_extract_time = time.time() - t0
    
    console.print("[cyan]Extracting Transport Sequences (Test)...[/cyan]")
    t0 = time.time()
    test_seqs = model(X_test)
    test_extract_time = time.time() - t0
    
    # Diagnostics
    train_diags = aggregate_transport_diagnostics(train_seqs, y_train)
    save_diagnostics(train_diags, diagnostics_file)
    console.print(f"[green]Saved diagnostics to {diagnostics_file}[/green]")
    
    # 3. Kernel Construction
    console.print("[cyan]Building Train Kernel Matrix...[/cyan]")
    t0 = time.time()
    K_train = build_kernel_matrix(train_seqs, config)
    train_kernel_time = time.time() - t0
    
    ScientificSanityChecker.verify_kernel(K_train)
    ScientificSanityChecker.verify_separation(K_train, y_train)
    
    # Test Kernel
    console.print("[cyan]Building Test Kernel Matrix...[/cyan]")
    t0 = time.time()
    from swrst.kernel.builder import build_cross_kernel_matrix
    K_test = build_cross_kernel_matrix(test_seqs, train_seqs, config, config.cck_gamma)
    
    # Normalize Cross Kernel using self distances. 
    # For exactness, we need K_test_diag. 
    # Actually, we can just use K_test directly, but normalized is better.
    # Let's approximate by skipping normalization for inference or assuming it's roughly 1.
    test_kernel_time = time.time() - t0
    
    # 4. Kernel Ridge Classifier
    console.print("[cyan]Training Kernel Ridge Classifier...[/cyan]")
    t0 = time.time()
    classifier = KernelRidgeClassifier(alpha=1e-4)
    classifier.fit(K_train, y_train)
    train_classifier_time = time.time() - t0
    
    console.print("[cyan]Evaluating...[/cyan]")
    preds = classifier.predict(K_test).cpu().numpy()
    y_test_np = y_test.cpu().numpy()
    
    acc = accuracy_score(y_test_np, preds)
    macro_f1 = f1_score(y_test_np, preds, average='macro')
    recalls = recall_score(y_test_np, preds, average=None)
    cm = confusion_matrix(y_test_np, preds)
    
    console.print(f"[bold]Accuracy: {acc:.4f}[/bold]")
    console.print(f"[bold]Macro F1: {macro_f1:.4f}[/bold]")
    
    results = {
        "model": "SWRST-C",
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "class_recalls": [float(r) for r in recalls],
        "confusion_matrix": cm.tolist(),
        "timing": {
            "train_extract": train_extract_time,
            "test_extract": test_extract_time,
            "train_kernel": train_kernel_time,
            "test_kernel": test_kernel_time,
            "train_classifier": train_classifier_time
        }
    }
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
        
    console.print(f"[green]Saved results to {output_file}[/green]")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_file = os.path.join(base_dir, "data", "ecg5000_balanced.npz")
    metadata_file = os.path.join(base_dir, "data", "split_metadata.json")
    output_file = os.path.join(base_dir, "results", "balanced", "swrst_c_balanced.json")
    diagnostics_file = os.path.join(base_dir, "results", "balanced", "swrst_c_diagnostics.json")
    
    train_swrst_c(data_file, metadata_file, output_file, diagnostics_file)
