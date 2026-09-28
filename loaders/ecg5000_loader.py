import os
import json
import hashlib
import numpy as np
from aeon.datasets import load_from_ts_file
from sklearn.model_selection import StratifiedShuffleSplit
from rich.console import Console

console = Console()

def load_and_resplit(data_dir, output_file, metadata_file):
    train_file = os.path.join(data_dir, "ECG5000_TRAIN.ts")
    test_file = os.path.join(data_dir, "ECG5000_TEST.ts")
    
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        raise FileNotFoundError(f"Missing .ts files in {data_dir}. Run download script first.")
        
    console.print("[cyan]Loading train and test files...[/cyan]")
    X_train, y_train = load_from_ts_file(train_file)
    X_test, y_test = load_from_ts_file(test_file)
    
    # Flatten from (N, 1, 140) to (N, 140)
    X_train = X_train.squeeze(axis=1)
    X_test = X_test.squeeze(axis=1)
    
    # Concatenate
    X_full = np.concatenate([X_train, X_test])
    y_full = np.concatenate([y_train, y_test])
    
    # Convert string labels to ints, then normalize: 1,2,3,4,5 -> 0,1,2,3,4
    y_full = y_full.astype(int) - 1
    
    # Pooled stratified split
    console.print("[cyan]Performing stratified split (4000 train / 1000 test)...[/cyan]")
    sss = StratifiedShuffleSplit(n_splits=1, train_size=4000, test_size=1000, random_state=42)
    train_idx, test_idx = next(sss.split(X_full, y_full))
    
    X_train_new, y_train_new = X_full[train_idx], y_full[train_idx]
    X_test_new, y_test_new = X_full[test_idx], y_full[test_idx]
    
    # Save split
    np.savez_compressed(output_file, X_train=X_train_new, y_train=y_train_new, X_test=X_test_new, y_test=y_test_new)
    console.print(f"[green]Saved split to {output_file}[/green]")
    
    # Compute counts
    train_counts = {int(k): int(v) for k, v in zip(*np.unique(y_train_new, return_counts=True))}
    test_counts = {int(k): int(v) for k, v in zip(*np.unique(y_test_new, return_counts=True))}
    
    # Simple hash of original data (for reproducibility verification)
    source_hash = hashlib.md5(X_full.tobytes()).hexdigest()
    
    metadata = {
        "seed": 42,
        "class_counts": {
            "train": train_counts,
            "test": test_counts
        },
        "source_hash": source_hash,
        "normalization_mode": "y = y - 1 (labels 0,1,2,3,4)"
    }
    
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)
        
    console.print(f"[green]Saved metadata to {metadata_file}[/green]")
    
    # Print summary
    console.print("\n[bold]Split Summary:[/bold]")
    console.print(f"Train shapes: {X_train_new.shape}, {y_train_new.shape}")
    console.print(f"Test shapes: {X_test_new.shape}, {y_test_new.shape}")
    console.print(f"Train class counts: {train_counts}")
    console.print(f"Test class counts: {test_counts}")
    
    return metadata

import torch
from torch.utils.data import TensorDataset, DataLoader

def get_dataloaders(batch_size=32, train_size=4000, val_size=1000, augment=False, balanced=False):
    base_dir = os.path.dirname(os.path.dirname(__file__))
    split_file = os.path.join(base_dir, "data", "ecg5000_resplit.npz")
    
    if not os.path.exists(split_file):
        data_dir = os.path.join(base_dir, "data", "ECG5000")
        metadata_file = os.path.join(base_dir, "data", "split_metadata.json")
        load_and_resplit(data_dir, split_file, metadata_file)
        
    data = np.load(split_file)
    X_train = data['X_train'].astype(np.float32)
    y_train = data['y_train'].astype(np.int64)
    X_test = data['X_test'].astype(np.float32)
    y_test = data['y_test'].astype(np.int64)
    
    # RGTN expects (B, 140) or (B, 1, 140)
    # The loaded data is (N, 140). We can expand dims to (N, 1, 140)
    X_train = np.expand_dims(X_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)
    
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    return train_loader, val_loader, None

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, "data", "ECG5000")
    output_file = os.path.join(base_dir, "data", "ecg5000_resplit.npz")
    metadata_file = os.path.join(base_dir, "data", "split_metadata.json")
    
    load_and_resplit(data_dir, output_file, metadata_file)
