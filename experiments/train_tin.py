"""
Train Transport-Inception Network (TIN) on ECG5000.

Novel architecture: Inception convolutions + differentiable transport + cross-modal attention.
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from rich.console import Console

console = Console()

# Hardware opts
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True


class ECGDataset:
    def __init__(self, X, y, augment=False, aug_target=200):
        eps = 1e-8
        mu = np.mean(X, axis=-1, keepdims=True)
        sigma = np.std(X, axis=-1, keepdims=True)
        X = (X - mu) / (sigma + eps)

        # Geodesic augmentation for minority classes
        if augment:
            X, y = self._augment(X, y, aug_target)

        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N, 1, 140)
        self.y = torch.tensor(y, dtype=torch.long)

    def _augment(self, X, y, target):
        classes, counts = np.unique(y, return_counts=True)
        to_augment = [c for c, n in zip(classes, counts) if n < target]
        if not to_augment:
            return X, y

        rng = np.random.RandomState(42)
        X_synth, y_synth = [], []

        for c in to_augment:
            idx = np.where(y == c)[0]
            n_needed = target - len(idx)
            for _ in range(n_needed):
                j, k = rng.choice(idx, 2, replace=(len(idx) < 2))
                t = rng.uniform(0, 1)
                synth = (1 - t) * X[j] + t * X[k]
                X_synth.append(synth)
                y_synth.append(c)

        if X_synth:
            X_synth = np.array(X_synth)
            y_synth = np.array(y_synth)
            X = np.concatenate([X, X_synth])
            y = np.concatenate([y, y_synth])
            perm = rng.permutation(len(X))
            return X[perm], y[perm]
        return X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_tin(data_file, output_file, checkpoint_dir):
    from models.transport_inception import TransportInceptionNetwork

    console.print("[bold cyan]Transport-Inception Network (TIN)[/bold cyan]")

    # Load data
    data = np.load(data_file)
    X_train, y_train = data["X_train"], data["y_train"].astype(np.int64)
    X_test, y_test = data["X_test"], data["y_test"].astype(np.int64)

    # Datasets with augmentation
    train_ds = ECGDataset(X_train, y_train, augment=True, aug_target=200)
    test_ds = ECGDataset(X_test, y_test, augment=False)

    batch_size = 64
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Device: {device}, Train: {len(train_ds)}, Test: {len(test_ds)}")

    # Model
    model = TransportInceptionNetwork(
        num_classes=5, input_length=140, K=32,
        conv_hidden=64, transport_hidden=64, num_heads=4,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"Parameters: {n_params:,}")

    # Optimizer with differential learning rates
    conv_params = list(model.conv_branch.parameters())
    transport_params = list(model.transport_branch.parameters())
    other_params = [p for n, p in model.named_parameters()
                    if not any(x in n for x in ['conv_branch', 'transport_branch'])]

    optimizer = torch.optim.AdamW([
        {"params": conv_params, "lr": 1e-3},
        {"params": transport_params, "lr": 5e-4},
        {"params": other_params, "lr": 3e-4},
    ], weight_decay=1e-2)

    # Class weights for imbalance
    class_counts = np.bincount(y_train, minlength=5)
    class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32, device=device)
    class_weights = class_weights / class_weights.sum() * 5
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, epochs=150, steps_per_epoch=len(train_loader)
    )

    # Training
    best_loss = float('inf')
    patience = 25
    patience_counter = 0
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_path = os.path.join(checkpoint_dir, "best_tin.pt")

    console.print("[cyan]Starting training...[/cyan]")
    total_time = 0

    for epoch in range(150):
        model.train()
        epoch_loss = 0
        t0 = time.time()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        epoch_time = time.time() - t0
        total_time += epoch_time
        avg_loss = epoch_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                val_loss += criterion(logits, y).item()
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_targets.extend(y.cpu().numpy())

        avg_val_loss = val_loss / len(test_loader)
        acc = accuracy_score(all_targets, all_preds)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            console.print(f"  Epoch {epoch+1:03d} | Train: {avg_loss:.4f} | Val: {avg_val_loss:.4f} | "
                          f"Acc: {acc:.4f} | {epoch_time:.1f}s")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                console.print(f"  [yellow]Early stopping at epoch {epoch+1}[/yellow]")
                break

    console.print(f"[green]Training complete: {total_time:.1f}s[/green]")

    # Final evaluation
    console.print("[cyan]Evaluating best model...[/cyan]")
    model.load_state_dict(torch.load(best_path, weights_only=True))
    model.eval()

    all_preds, all_targets = [], []
    all_quantiles = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits, info = model(x)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            all_quantiles.append(info['quantile_fn'].cpu())

    y_pred = np.array(all_preds)
    y_true = np.array(all_targets)

    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average='macro')
    recalls = recall_score(y_true, y_pred, average=None)
    cm = confusion_matrix(y_true, y_pred)

    console.print(f"\n[bold green]TIN Results:[/bold green]")
    console.print(f"  Accuracy:  {acc:.4f}")
    console.print(f"  Macro F1:  {mf1:.4f}")
    console.print(f"  Recalls:   {', '.join(f'{r:.4f}' for r in recalls)}")

    results = {
        "model": "Transport-Inception-Network",
        "accuracy": float(acc),
        "macro_f1": float(mf1),
        "class_recalls": [float(r) for r in recalls],
        "confusion_matrix": cm.tolist(),
        "n_params": n_params,
        "training_time_sec": total_time,
        "config": {
            "K": 32, "conv_hidden": 64, "transport_hidden": 64,
            "num_heads": 4, "epochs": 150, "patience": 25,
            "augmentation": "geodesic (target=200)",
        },
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    console.print(f"[green]Results saved to {output_file}[/green]")

    return results


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    base_dir = os.path.dirname(os.path.dirname(__file__))
    train_tin(
        os.path.join(base_dir, "data", "ecg5000_resplit.npz"),
        os.path.join(base_dir, "results", "tin_results.json"),
        os.path.join(base_dir, "checkpoints"),
    )
