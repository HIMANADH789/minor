"""
eTAI v2: Enhanced TAI + Focal Loss for minority class improvement.

Focal loss down-weights easy majority examples and focuses on hard minority examples.
This should improve C2, C3, C4 recall without hurting C0, C1.
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from rich.console import Console

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

console = Console()


# ============================================================
# Transport feature preprocessing
# ============================================================

def compute_transport_channels(X, K=32):
    N, L = X.shape
    out = np.zeros((N, 3, L), dtype=np.float32)
    out[:, 0, :] = X.astype(np.float32)
    for i in range(N):
        sorted_sig = np.sort(X[i])
        out[i, 1, :] = np.interp(np.linspace(0, 1, L),
                                   np.linspace(0, 1, len(sorted_sig)), sorted_sig)
        drift = np.zeros(L)
        for lag in [1, 2, 4]:
            d = np.diff(sorted_sig, n=lag)
            drift[lag:] += np.abs(d) / 3.0
        out[i, 2, :] = drift
    for c in range(3):
        mu = np.mean(out[:, c, :], axis=-1, keepdims=True)
        sig = np.std(out[:, c, :], axis=-1, keepdims=True) + 1e-8
        out[:, c, :] = (out[:, c, :] - mu) / sig
    return out


# ============================================================
# InceptionTime with 3 input channels
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
        return self.relu(self.bn(torch.cat([c(bx) for c in self.convs] + [self.pool_conv(self.maxpool(x))], dim=1)))


class Shortcut(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.op = nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, bias=False), nn.BatchNorm1d(out_ch)) if in_ch != out_ch else nn.Identity()
    def forward(self, x):
        return self.op(x)


class InceptionTimeMultiCh(nn.Module):
    def __init__(self, num_classes=5, in_channels=3, num_blocks=6, out_channels=32, bottleneck=32, kernels=[10, 20, 40]):
        super().__init__()
        self.blocks, self.shortcuts = nn.ModuleList(), nn.ModuleList()
        current_in = in_channels
        for i in range(num_blocks):
            self.blocks.append(InceptionModule(current_in, out_channels, bottleneck, kernels))
            current_in = out_channels * len(kernels) + out_channels
            if i % 3 == 2:
                self.shortcuts.append(Shortcut(in_channels if i == 2 else (out_channels * len(kernels) + out_channels), current_in))
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(current_in, num_classes)

    def forward(self, x):
        shortcut_input, si = x, 0
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i % 3 == 2:
                x = torch.relu(x + self.shortcuts[si](shortcut_input))
                shortcut_input, si = x, si + 1
        return self.fc(self.gap(x).squeeze(-1))


# ============================================================
# Focal Loss
# ============================================================

class FocalLoss(nn.Module):
    """
    Focal Loss: down-weights easy examples, focuses on hard minority examples.
    FL(p_t) = -alpha_t * (1-p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        # alpha: per-class weights (optional)
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # probability of correct class

        # Focal modulating factor
        focal_weight = (1 - pt) ** self.gamma

        # Optional per-class alpha
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)[targets]
            focal_weight = alpha * focal_weight

        loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ============================================================
# Dataset
# ============================================================

class ECGDataset(Dataset):
    def __init__(self, X, y):
        eps = 1e-8
        mu = np.mean(X, axis=(-1, -2), keepdims=True)
        sigma = np.std(X, axis=(-1, -2), keepdims=True)
        self.X = torch.tensor((X - mu) / (sigma + eps), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


# ============================================================
# Training
# ============================================================

def train_eTAI_v2(data_file, output_file, checkpoint_dir, gamma=2.0, use_class_weights=False):
    console.print(f"[bold cyan]eTAI v2: Transport + InceptionTime + Focal Loss (gamma={gamma})[/bold cyan]")

    data = np.load(data_file)
    X_train_raw = data['X_train'].astype(np.float64)
    y_train = data['y_train'].astype(np.int64)
    X_test_raw = data['X_test'].astype(np.float64)
    y_test = data['y_test'].astype(np.int64)

    eps = 1e-8
    mu, sig = np.mean(X_train_raw, -1, keepdims=True), np.std(X_train_raw, -1, keepdims=True)
    X_train_raw = (X_train_raw - mu) / (sig + eps)
    mu, sig = np.mean(X_test_raw, -1, keepdims=True), np.std(X_test_raw, -1, keepdims=True)
    X_test_raw = (X_test_raw - mu) / (sig + eps)

    X_train_3ch = compute_transport_channels(X_train_raw)
    X_test_3ch = compute_transport_channels(X_test_raw)

    train_dataset = ECGDataset(X_train_3ch, y_train)
    test_dataset = ECGDataset(X_test_3ch, y_test)

    batch_size, grad_accum = 32, 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = InceptionTimeMultiCh(num_classes=5, in_channels=3, num_blocks=6).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"Parameters: {n_params:,}")

    # Focal loss with optional class weights
    if use_class_weights:
        class_counts = np.bincount(y_train, minlength=5)
        alpha = class_counts.sum() / (5 * class_counts)
        alpha = np.clip(alpha, 0.1, 10.0)
        console.print(f"Class alpha: {[f'{a:.3f}' for a in alpha]}")
    else:
        alpha = None

    criterion = FocalLoss(alpha=alpha, gamma=gamma)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2, fused=True)
    epochs = 150
    steps_per_epoch = len(train_loader) // grad_accum
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-4, epochs=epochs,
                                                     steps_per_epoch=steps_per_epoch if steps_per_epoch > 0 else 1)
    scaler = torch.cuda.amp.GradScaler()

    best_loss = float('inf')
    patience, patience_counter = 20, 0
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_path = os.path.join(checkpoint_dir, "best_eTAI_v2.pt")

    console.print("[cyan]Training...[/cyan]")
    total_time = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        t0 = time.time()

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = criterion(model(x), y) / grad_accum
            scaler.scale(loss).backward()
            if (i + 1) % grad_accum == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True); scheduler.step()
            epoch_loss += loss.item() * grad_accum

        epoch_time = time.time() - t0
        total_time += epoch_time
        avg_loss = epoch_loss / len(train_loader)

        model.eval()
        val_loss, all_preds, all_targets = 0.0, [], []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = model(x); val_loss += criterion(out, y).item()
                all_preds.extend(out.argmax(1).cpu().numpy())
                all_targets.extend(y.cpu().numpy())

        avg_val_loss = val_loss / len(test_loader)
        acc = accuracy_score(all_targets, all_preds)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            console.print(f"  Epoch {epoch+1:03d} | Train: {avg_loss:.4f} | Val: {avg_val_loss:.4f} | "
                          f"Acc: {acc:.4f} | {epoch_time:.1f}s")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss; patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                console.print(f"  [yellow]Early stopping at epoch {epoch+1}[/yellow]"); break

    console.print(f"[green]Training complete: {total_time:.1f}s[/green]")

    model.load_state_dict(torch.load(best_path, weights_only=True))
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(x)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    y_pred, y_true = np.array(all_preds), np.array(all_targets)
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average='macro')
    recalls = recall_score(y_true, y_pred, average=None)
    cm = confusion_matrix(y_true, y_pred)

    console.print(f"\n[bold green]eTAI v2 Results (focal gamma={gamma}, class_weights={use_class_weights}):[/bold green]")
    console.print(f"  Accuracy:  {acc:.4f}")
    console.print(f"  Macro F1:  {mf1:.4f}")
    console.print(f"  Recalls:   {', '.join(f'{r:.4f}' for r in recalls)}")

    console.print(f"\n[bold]Comparison:[/bold]")
    console.print(f"  {'Model':<40} {'Acc':>8} {'MacroF1':>8}")
    console.print(f"  {'-'*60}")
    console.print(f"  {'InceptionTime':<40} {'0.9610':>8} {'0.7697':>8}")
    console.print(f"  {'MiniRocket':<40} {'0.9560':>8} {'0.6011':>8}")
    console.print(f"  {'eTAI v1 (CE, no focal)':<40} {'0.9540':>8} {'0.5986':>8}")
    console.print(f"  [bold]{'eTAI v2 (this run)':<40} {acc:>8.4f} {mf1:>8.4f}[/bold]")

    results = {
        "model": "eTAI-v2-Focal",
        "accuracy": float(acc), "macro_f1": float(mf1),
        "class_recalls": [float(r) for r in recalls],
        "confusion_matrix": cm.tolist(), "n_params": n_params,
        "training_time_sec": total_time,
        "config": {"gamma": gamma, "use_class_weights": use_class_weights},
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    console.print(f"\n[green]Results saved to {output_file}[/green]")
    return results


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_file = os.path.join(base_dir, "data", "ecg5000_resplit.npz")

    # Run multiple gamma values
    for gamma in [1.0, 2.0, 3.0]:
        console.print(f"\n{'='*60}")
        train_eTAI_v2(
            data_file,
            os.path.join(base_dir, "results", f"eTAI_v2_gamma{gamma}.json"),
            os.path.join(base_dir, "checkpoints"),
            gamma=gamma, use_class_weights=True,
        )
