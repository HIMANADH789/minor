"""
Enhanced Transport-Augmented InceptionTime (eTAI)

Key changes from TAI to match InceptionTime's training:
  1. batch_size=32, grad_accum=2 (InceptionTime's exact recipe)
  2. Mixed precision (autocast + GradScaler)
  3. No augmentation (augmentation was hurting)
  4. No class weights (plain CE like InceptionTime)
  5. 6 Inception blocks (matching InceptionTime)
  6. workers=4, prefetch_factor=4
  7. Transport features as 3 input channels to InceptionTime
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from rich.console import Console

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

console = Console()


# ============================================================
# Transport feature preprocessing (fixed, non-differentiable)
# ============================================================

def compute_transport_channels(X, K=32):
    """
    3-channel input:
      Ch0: raw signal
      Ch1: sorted signal (amplitude distribution)
      Ch2: multi-scale drift (|diff(Q, lag=1)| + |diff(Q, lag=2)| + |diff(Q, lag=4)|)
    """
    N, L = X.shape
    out = np.zeros((N, 3, L), dtype=np.float32)

    out[:, 0, :] = X.astype(np.float32)

    for i in range(N):
        sorted_sig = np.sort(X[i])
        # Channel 1: sorted signal tiled to length L
        out[i, 1, :] = np.interp(np.linspace(0, 1, L),
                                   np.linspace(0, 1, len(sorted_sig)), sorted_sig)
        # Channel 2: multi-scale drift magnitude
        drift = np.zeros(L)
        for lag in [1, 2, 4]:
            d = np.diff(sorted_sig, n=lag)
            drift[lag:] += np.abs(d) / 3.0
        out[i, 2, :] = drift

    # Per-channel standardization
    for c in range(3):
        mu = np.mean(out[:, c, :], axis=-1, keepdims=True)
        sig = np.std(out[:, c, :], axis=-1, keepdims=True) + 1e-8
        out[:, c, :] = (out[:, c, :] - mu) / sig

    return out


# ============================================================
# InceptionTime with 3 input channels (matching original exactly)
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


class InceptionTimeMultiCh(nn.Module):
    """InceptionTime with configurable input channels (3 for transport-augmented)."""
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
        return self.fc(self.gap(x).squeeze(-1))


# ============================================================
# Dataset (NO augmentation, matching InceptionTime)
# ============================================================

class ECGDataset(Dataset):
    def __init__(self, X, y):
        eps = 1e-8
        mu = np.mean(X, axis=(-1, -2) if X.ndim == 3 else -1, keepdims=True)
        sigma = np.std(X, axis=(-1, -2) if X.ndim == 3 else -1, keepdims=True)
        X = (X - mu) / (sigma + eps)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================
# Training (EXACT InceptionTime recipe)
# ============================================================

def train_eTAI(data_file, output_file, checkpoint_dir):
    console.print("[bold cyan]Enhanced TAI (eTAI): InceptionTime + Transport Channels[/bold cyan]")

    data = np.load(data_file)
    X_train_raw = data['X_train'].astype(np.float64)
    y_train = data['y_train'].astype(np.int64)
    X_test_raw = data['X_test'].astype(np.float64)
    y_test = data['y_test'].astype(np.int64)

    # Standardize
    eps = 1e-8
    mu, sig = np.mean(X_train_raw, -1, keepdims=True), np.std(X_train_raw, -1, keepdims=True)
    X_train_raw = (X_train_raw - mu) / (sig + eps)
    mu, sig = np.mean(X_test_raw, -1, keepdims=True), np.std(X_test_raw, -1, keepdims=True)
    X_test_raw = (X_test_raw - mu) / (sig + eps)

    # Compute transport channels
    console.print("[cyan]Computing transport features...[/cyan]")
    X_train_3ch = compute_transport_channels(X_train_raw)
    X_test_3ch = compute_transport_channels(X_test_raw)
    console.print(f"  Input shape: {X_train_3ch.shape}")

    # Datasets (NO augmentation)
    train_dataset = ECGDataset(X_train_3ch, y_train)
    test_dataset = ECGDataset(X_test_3ch, y_test)

    # EXACT InceptionTime DataLoader config
    batch_size = 32
    grad_accum = 2

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=4
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model: InceptionTime with 3 input channels
    model = InceptionTimeMultiCh(
        num_classes=5, in_channels=3, num_blocks=6,
        out_channels=32, bottleneck=32, kernels=[10, 20, 40],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"Parameters: {n_params:,}")

    # EXACT InceptionTime training recipe
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2, fused=True)

    epochs = 150
    steps_per_epoch = len(train_loader) // grad_accum
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-4, epochs=epochs,
        steps_per_epoch=steps_per_epoch if steps_per_epoch > 0 else 1
    )

    criterion = nn.CrossEntropyLoss()  # NO class weights (matching InceptionTime)
    scaler = torch.cuda.amp.GradScaler()

    best_loss = float('inf')
    patience = 20
    patience_counter = 0
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_path = os.path.join(checkpoint_dir, "best_eTAI.pt")

    console.print("[cyan]Starting Training (InceptionTime recipe)...[/cyan]")
    total_time = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        t0 = time.time()

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(x)
                loss = criterion(outputs, y) / grad_accum

            scaler.scale(loss).backward()

            if (i + 1) % grad_accum == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            epoch_loss += loss.item() * grad_accum

        epoch_time = time.time() - t0
        total_time += epoch_time
        avg_loss = epoch_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(x)
                    val_loss += criterion(outputs, y).item()
                all_preds.extend(outputs.argmax(1).cpu().numpy())
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
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(x)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    y_pred = np.array(all_preds)
    y_true = np.array(all_targets)

    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average='macro')
    recalls = recall_score(y_true, y_pred, average=None)
    cm = confusion_matrix(y_true, y_pred)

    console.print(f"\n[bold green]eTAI Results:[/bold green]")
    console.print(f"  Accuracy:  {acc:.4f}")
    console.print(f"  Macro F1:  {mf1:.4f}")
    console.print(f"  Recalls:   {', '.join(f'{r:.4f}' for r in recalls)}")

    console.print(f"\n[bold]Comparison:[/bold]")
    console.print(f"  {'Model':<35} {'Acc':>8} {'MacroF1':>8}")
    console.print(f"  {'-'*55}")
    console.print(f"  {'InceptionTime':<35} {'0.9610':>8} {'0.7697':>8}")
    console.print(f"  {'MiniRocket':<35} {'0.9560':>8} {'0.6011':>8}")
    console.print(f"  {'KTA-TF-Drift v2.1':<35} {'0.9500':>8} {'0.5891':>8}")
    console.print(f"  {'TAI (old)':<35} {'0.8860':>8} {'0.5552':>8}")
    console.print(f"  [bold]{'eTAI (this run)':<35} {acc:>8.4f} {mf1:>8.4f}[/bold]")

    results = {
        "model": "eTAI-Transport-Augmented-InceptionTime",
        "accuracy": float(acc), "macro_f1": float(mf1),
        "class_recalls": [float(r) for r in recalls],
        "confusion_matrix": cm.tolist(), "n_params": n_params,
        "training_time_sec": total_time,
        "changes_from_TAI": [
            "batch_size=32, grad_accum=2 (was 64, no accum)",
            "mixed precision (was none)",
            "no augmentation (was geodesic aug)",
            "no class weights (was weighted CE)",
            "6 Inception blocks (was 3)",
            "workers=4, prefetch=4 (was 0)",
        ],
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    console.print(f"\n[green]Results saved to {output_file}[/green]")
    return results


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    base_dir = os.path.dirname(os.path.dirname(__file__))
    train_eTAI(
        os.path.join(base_dir, "data", "ecg5000_resplit.npz"),
        os.path.join(base_dir, "results", "eTAI_results.json"),
        os.path.join(base_dir, "checkpoints"),
    )
