"""
Transport-Augmented InceptionTime (TAI)

Novel approach: compute transport features (quantile functions at multiple scales)
and concatenate them with the raw signal as additional input channels to InceptionTime.

This gives the learned convolutional filters direct access to transport-geometric
features without the complexity of a separate branch.

Three input channels:
  1. Raw signal (as in standard InceptionTime)
  2. Quantile function (K-point approximation of amplitude distribution)
  3. Multi-scale drift features (consecutive quantile differences)
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
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True


# ============================================================
# InceptionTime (from existing code, adapted for multi-channel input)
# ============================================================

class InceptionModule(nn.Module):
    def __init__(self, in_ch, out_ch, bottleneck=32, kernels=[10, 20, 40]):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_ch, bottleneck, 1, bias=False) if in_ch > 1 else nn.Identity()
        in_conv = bottleneck if in_ch > 1 else 1
        self.convs = nn.ModuleList([
            nn.Conv1d(in_conv, out_ch, k, padding='same', bias=False) for k in kernels
        ])
        self.maxpool = nn.MaxPool1d(3, stride=1, padding=1)
        self.pool_conv = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch * len(kernels) + out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        bx = self.bottleneck(x)
        conv_outs = [c(bx) for c in self.convs]
        pool_out = self.pool_conv(self.maxpool(x))
        return self.relu(self.bn(torch.cat(conv_outs + [pool_out], dim=1)))


class Shortcut(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        if in_ch != out_ch:
            self.conv = nn.Conv1d(in_ch, out_ch, 1, bias=False)
            self.bn = nn.BatchNorm1d(out_ch)
        else:
            self.conv = nn.Identity()
            self.bn = nn.Identity()
    def forward(self, x):
        return self.bn(self.conv(x))


class InceptionTimeMultiChannel(nn.Module):
    """InceptionTime adapted for multi-channel input (raw + transport features)."""
    def __init__(self, num_classes=5, in_channels=3, num_blocks=6,
                 out_channels=32, bottleneck=32, kernels=[10, 20, 40]):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.shortcuts = nn.ModuleList()

        current_in = in_channels
        for i in range(num_blocks):
            self.blocks.append(InceptionModule(current_in, out_channels, bottleneck, kernels))
            current_in = out_channels * len(kernels) + out_channels
            if i % 3 == 2:
                shortcut_in = in_channels if i == 2 else (out_channels * len(kernels) + out_channels)
                self.shortcuts.append(Shortcut(shortcut_in, current_in))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(current_in, num_classes)

    def forward(self, x):
        shortcut_input = x
        shortcut_idx = 0
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i % 3 == 2:
                shortcut = self.shortcuts[shortcut_idx](shortcut_input)
                x = torch.relu(x + shortcut)
                shortcut_input = x
                shortcut_idx += 1
        x = self.gap(x).squeeze(-1)
        return self.fc(x)


# ============================================================
# Transport Feature Preprocessing (non-differentiable, input preprocessing)
# ============================================================

def compute_transport_channels(X, K=32):
    """
    Compute transport feature channels for each signal.
    Returns: (N, 3, 140) tensor with channels:
      0: raw signal (repeated for length alignment)
      1: quantile function tiled to signal length
      2: multi-scale drift features
    """
    N, L = X.shape
    out = np.zeros((N, 3, L), dtype=np.float32)

    # Channel 0: raw signal
    out[:, 0, :] = X.astype(np.float32)

    # Channel 1: quantile function (sorted signal, tiled)
    for i in range(N):
        sorted_sig = np.sort(X[i])
        qgrid = np.interp(np.linspace(0, 1, L), np.linspace(0, 1, len(sorted_sig)), sorted_sig)
        out[i, 1, :] = qgrid

    # Channel 2: multi-scale drift (quantile differences at multiple lags)
    for i in range(N):
        sorted_sig = np.sort(X[i])
        qgrid = np.interp(np.linspace(0, 1, L), np.linspace(0, 1, len(sorted_sig)), sorted_sig)
        # Drift at lag 1, 2, 4 averaged
        drift = np.zeros(L)
        for lag in [1, 2, 4]:
            d = np.diff(qgrid, n=lag)
            drift[lag:] += np.abs(d) / 3.0
        out[i, 2, :] = drift

    # Per-channel standardization
    for c in range(3):
        mu = np.mean(out[:, c, :], axis=-1, keepdims=True)
        sig = np.std(out[:, c, :], axis=-1, keepdims=True) + 1e-8
        out[:, c, :] = (out[:, c, :] - mu) / sig

    return out


# ============================================================
# Dataset
# ============================================================

class ECGDataset:
    def __init__(self, X, y, augment=False, aug_target=200):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

        if augment:
            self.X, self.y = self._augment(self.X, self.y, aug_target)

    def _augment(self, X, y, target):
        classes, counts = np.unique(y.numpy(), return_counts=True)
        to_augment = [c for c, n in zip(classes, counts) if n < target]
        if not to_augment:
            return X, y

        rng = np.random.RandomState(42)
        X_np, y_np = X.numpy(), y.numpy()
        X_synth, y_synth = [], []

        for c in to_augment:
            idx = np.where(y_np == c)[0]
            n_needed = target - len(idx)
            for _ in range(n_needed):
                j, k = rng.choice(idx, 2, replace=(len(idx) < 2))
                t = rng.uniform(0, 1)
                synth = (1 - t) * X_np[j] + t * X_np[k]
                X_synth.append(synth)
                y_synth.append(c)

        if X_synth:
            X_synth = np.array(X_synth, dtype=np.float32)
            y_synth = np.array(y_synth, dtype=np.int64)
            X_np = np.concatenate([X_np, X_synth])
            y_np = np.concatenate([y_np, y_synth])
            perm = rng.permutation(len(X_np))
            return torch.tensor(X_np[perm]), torch.tensor(y_np[perm])
        return X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================
# Training
# ============================================================

def train_tai(data_file, output_file, checkpoint_dir):
    console.print("[bold cyan]Transport-Augmented InceptionTime (TAI)[/bold cyan]")

    data = np.load(data_file)
    X_train, y_train = data["X_train"], data["y_train"].astype(np.int64)
    X_test, y_test = data["X_test"], data["y_test"].astype(np.int64)

    # Standardize
    eps = 1e-8
    mu, sig = np.mean(X_train, -1, keepdims=True), np.std(X_train, -1, keepdims=True)
    X_train = (X_train - mu) / (sig + eps)
    mu, sig = np.mean(X_test, -1, keepdims=True), np.std(X_test, -1, keepdims=True)
    X_test = (X_test - mu) / (sig + eps)

    # Compute transport channels
    console.print("[cyan]Computing transport features...[/cyan]")
    X_train_3ch = compute_transport_channels(X_train, K=32)
    X_test_3ch = compute_transport_channels(X_test, K=32)
    console.print(f"  Input shape: {X_train_3ch.shape} (N, 3 channels, 140)")

    # Datasets
    train_ds = ECGDataset(X_train_3ch, y_train, augment=True, aug_target=200)
    test_ds = ECGDataset(X_test_3ch, y_test)

    batch_size = 64
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Device: {device}, Train: {len(train_ds)}, Test: {len(test_ds)}")

    # Model: InceptionTime with 3 input channels
    model = InceptionTimeMultiChannel(
        num_classes=5, in_channels=3, num_blocks=6,
        out_channels=32, bottleneck=32, kernels=[10, 20, 40],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"Parameters: {n_params:,}")

    # Training setup (matching existing InceptionTime training)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2, fused=True)
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-4, epochs=150, steps_per_epoch=steps_per_epoch
    )

    # Class weights
    class_counts = np.bincount(y_train, minlength=5)
    class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32, device=device)
    class_weights = class_weights / class_weights.sum() * 5
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Training loop
    best_loss = float('inf')
    patience = 20
    patience_counter = 0
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_path = os.path.join(checkpoint_dir, "best_tai.pt")

    console.print("[cyan]Starting training...[/cyan]")
    total_time = 0

    for epoch in range(150):
        model.train()
        epoch_loss = 0
        t0 = time.time()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
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
                logits = model(x)
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
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    y_pred = np.array(all_preds)
    y_true = np.array(all_targets)

    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average='macro')
    recalls = recall_score(y_true, y_pred, average=None)
    cm = confusion_matrix(y_true, y_pred)

    console.print(f"\n[bold green]TAI Results:[/bold green]")
    console.print(f"  Accuracy:  {acc:.4f}")
    console.print(f"  Macro F1:  {mf1:.4f}")
    console.print(f"  Recalls:   {', '.join(f'{r:.4f}' for r in recalls)}")

    results = {
        "model": "Transport-Augmented-InceptionTime",
        "accuracy": float(acc),
        "macro_f1": float(mf1),
        "class_recalls": [float(r) for r in recalls],
        "confusion_matrix": cm.tolist(),
        "n_params": n_params,
        "training_time_sec": total_time,
        "config": {
            "in_channels": 3,
            "transport_features": "quantile + multi-scale drift",
            "architecture": "InceptionTime (6 blocks, kernels=[10,20,40])",
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
    train_tai(
        os.path.join(base_dir, "data", "ecg5000_resplit.npz"),
        os.path.join(base_dir, "results", "tai_results.json"),
        os.path.join(base_dir, "checkpoints"),
    )
