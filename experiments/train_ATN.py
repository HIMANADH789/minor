"""
ArcFace Transport Network (ATN)

Combines:
  1. InceptionTime backbone (proven multi-scale architecture)
  2. Transport features as input channels (distributional awareness)
  3. ArcFace margin loss (enforces angular separation between classes)
  4. Multi-scale transport pyramid (captures diverse patterns)

Why ArcFace helps minority classes:
  - Adds angular margin to classification loss
  - Forces same-class features to cluster tightly
  - Forces different-class features to be far apart in angular space
  - Directly addresses C3-C1 confusion (the main misclassification pattern)
"""

import os
import sys
import json
import time
import math
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
# InceptionTime with 3 input channels + embedding head
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


class InceptionBackbone(nn.Module):
    """InceptionTime backbone that outputs an embedding (not logits)."""
    def __init__(self, in_channels=3, num_blocks=6, out_channels=32, bottleneck=32,
                 kernels=[10, 20, 40], embed_dim=128):
        super().__init__()
        self.blocks, self.shortcuts = nn.ModuleList(), nn.ModuleList()
        current_in = in_channels
        for i in range(num_blocks):
            self.blocks.append(InceptionModule(current_in, out_channels, bottleneck, kernels))
            current_in = out_channels * len(kernels) + out_channels
            if i % 3 == 2:
                self.shortcuts.append(Shortcut(in_channels if i == 2 else (out_channels * len(kernels) + out_channels), current_in))
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.embed = nn.Linear(current_in, embed_dim)

    def forward(self, x):
        shortcut_input, si = x, 0
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i % 3 == 2:
                x = torch.relu(x + self.shortcuts[si](shortcut_input))
                shortcut_input, si = x, si + 1
        x = self.gap(x).squeeze(-1)
        return self.embed(x)  # (B, embed_dim)


# ============================================================
# ArcFace Loss
# ============================================================

class ArcFaceLoss(nn.Module):
    """
    ArcFace loss: adds angular margin to enforce better class separation.

    L_arcface = -log(exp(s * cos(theta_yi + m)) / (exp(s * cos(theta_yi + m)) + sum_{j!=yi} exp(s * cos(theta_j))))

    Where:
      s = scale parameter (default 30)
      m = angular margin (default 0.3 radians)
      theta_yi = angle between feature and class weight for true class
    """
    def __init__(self, embed_dim, num_classes, s=30.0, m=0.3):
        super().__init__()
        self.s = s
        self.m = m
        self.num_classes = num_classes

        # Class weight matrix (learnable)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)

        # Precompute constants
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.register_buffer('mm_tensor', torch.tensor(math.sin(math.pi - m) * m))

    def forward(self, features, targets):
        """
        features: (B, embed_dim) L2-normalized
        targets: (B,) class indices
        """
        features = F.normalize(features.float(), p=2, dim=1)
        weight_norm = F.normalize(self.weight.float(), p=2, dim=1)

        cosine = F.linear(features, weight_norm).clamp(-1 + 1e-7, 1 - 1e-7)

        # Compute theta + margin for true class
        theta = torch.acos(cosine)
        target_one_hot = F.one_hot(targets, self.num_classes).float()
        theta_m = theta + self.m * target_one_hot

        cosine_m = torch.cos(theta_m)

        # For angles where cos(theta) < threshold, use the mm correction
        # Use mask-add approach to avoid in-place index assignment
        mask = (cosine > self.threshold).float()
        cosine_m = cosine_m - mask * self.mm_tensor

        logits = self.s * cosine_m
        loss = F.cross_entropy(logits, targets)

        return loss


# ============================================================
# ArcFace Transport Network
# ============================================================

class ArcFaceTransportNetwork(nn.Module):
    """
    Full ATN model:
      1. Transport channels (raw + sorted + drift)
      2. InceptionTime backbone -> embedding
      3. ArcFace loss for classification
    """
    def __init__(self, num_classes=5, in_channels=3, embed_dim=128,
                 arcface_s=30.0, arcface_m=0.3):
        super().__init__()

        self.backbone = InceptionBackbone(
            in_channels=in_channels, num_blocks=6,
            out_channels=32, bottleneck=32,
            kernels=[10, 20, 40], embed_dim=embed_dim,
        )

        self.arcface = ArcFaceLoss(embed_dim, num_classes, s=arcface_s, m=arcface_m)

        # For inference (direct classification without ArcFace margin)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """Returns logits (for inference) and embedding (for ArcFace)."""
        embedding = self.backbone(x)  # (B, embed_dim)
        embedding_norm = F.normalize(embedding, p=2, dim=1)
        logits = self.fc(embedding_norm)
        return logits, embedding_norm

    def arcface_forward(self, x, targets):
        """Returns ArcFace loss."""
        embedding = self.backbone(x)
        embedding_norm = F.normalize(embedding, p=2, dim=1)
        loss = self.arcface(embedding_norm, targets)
        return loss, embedding_norm


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

def train_ATN(data_file, output_file, checkpoint_dir,
              arcface_s=30.0, arcface_m=0.3, embed_dim=128, focal_gamma=1.0):
    console.print(f"[bold cyan]ArcFace Transport Network (ATN)[/bold cyan]")
    console.print(f"[cyan]ArcFace: s={arcface_s}, m={arcface_m} | Focal gamma={focal_gamma} | embed={embed_dim}[/cyan]")

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

    model = ArcFaceTransportNetwork(
        num_classes=5, in_channels=3, embed_dim=embed_dim,
        arcface_s=arcface_s, arcface_m=arcface_m,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"Parameters: {n_params:,}")

    # Class weights for focal loss (secondary loss on FC head)
    class_counts = np.bincount(y_train, minlength=5)
    alpha = class_counts.sum() / (5 * class_counts)
    alpha = np.clip(alpha, 0.1, 10.0)
    alpha_t = torch.tensor(alpha, dtype=torch.float32, device=device)

    def focal_ce(logits, targets):
        """Focal cross-entropy with class weights."""
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        weight = (1 - pt) ** focal_gamma
        alpha_weight = alpha_t[targets]
        return (alpha_weight * weight * ce).mean()

    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 3e-4},
        {"params": model.arcface.parameters(), "lr": 3e-4},
        {"params": model.fc.parameters(), "lr": 1e-3},
    ], weight_decay=1e-2, fused=True)

    epochs = 150
    steps_per_epoch = len(train_loader) // grad_accum
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, epochs=epochs,
                                                     steps_per_epoch=steps_per_epoch if steps_per_epoch > 0 else 1)
    scaler = torch.cuda.amp.GradScaler()

    best_loss = float('inf')
    patience, patience_counter = 25, 0
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_path = os.path.join(checkpoint_dir, "best_ATN.pt")

    console.print("[cyan]Training with ArcFace + Focal loss...[/cyan]")
    total_time = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        t0 = time.time()

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                # ArcFace loss on embedding
                arc_loss, emb = model.arcface_forward(x, y)

                # Focal loss on FC head (auxiliary)
                logits = model.fc(emb)
                fc_loss = focal_ce(logits, y)

                # Combined loss: ArcFace is primary, focal is auxiliary
                loss = (arc_loss + 0.5 * fc_loss) / grad_accum

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
                    logits, _ = model(x)
                    val_loss += F.cross_entropy(logits, y).item()
                all_preds.extend(logits.argmax(1).cpu().numpy())
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
                logits, _ = model(x)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    y_pred, y_true = np.array(all_preds), np.array(all_targets)
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average='macro')
    recalls = recall_score(y_true, y_pred, average=None)
    cm = confusion_matrix(y_true, y_pred)

    console.print(f"\n[bold green]ATN Results:[/bold green]")
    console.print(f"  Accuracy:  {acc:.4f}")
    console.print(f"  Macro F1:  {mf1:.4f}")
    console.print(f"  Recalls:   {', '.join(f'{r:.4f}' for r in recalls)}")

    # Show misclassification pattern
    console.print(f"\n[bold]Misclassification analysis:[/bold]")
    for c in [2, 3, 4]:
        total = sum(cm[c])
        hits = cm[c][c]
        misclass = {j: cm[c][j] for j in range(5) if j != c and cm[c][j] > 0}
        console.print(f"  C{c}: {hits}/{total} correct | misclassified as: {misclass}")

    console.print(f"\n[bold]Comparison:[/bold]")
    console.print(f"  {'Model':<45} {'Acc':>8} {'MacroF1':>8} {'C2':>6} {'C3':>6} {'C4':>6}")
    console.print(f"  {'-'*75}")
    console.print(f"  {'InceptionTime':<45} {'0.9610':>8} {'0.7697':>8} {'0.737':>6} {'0.410':>6} {'0.600':>6}")
    console.print(f"  {'eTAI v1 (CE)':<45} {'0.9540':>8} {'0.5986':>8} {'0.579':>6} {'0.385':>6} {'0.000':>6}")
    console.print(f"  {'eTAI v2 (focal g=1)':<45} {'0.9140':>8} {'0.6110':>8} {'0.947':>6} {'0.513':>6} {'0.200':>6}")
    console.print(f"  [bold]{'ATN (this run)':<45} {acc:>8.4f} {mf1:>8.4f} {recalls[2]:>6.3f} {recalls[3]:>6.3f} {recalls[4]:>6.3f}[/bold]")

    results = {
        "model": "ArcFace-Transport-Network",
        "accuracy": float(acc), "macro_f1": float(mf1),
        "class_recalls": [float(r) for r in recalls],
        "confusion_matrix": cm.tolist(), "n_params": n_params,
        "training_time_sec": total_time,
        "config": {"arcface_s": arcface_s, "arcface_m": arcface_m,
                   "focal_gamma": focal_gamma, "embed_dim": embed_dim},
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

    # Sweep ArcFace parameters
    configs = [
        {"arcface_s": 30.0, "arcface_m": 0.3, "focal_gamma": 1.0, "label": "s30_m0.3_f1"},
        {"arcface_s": 30.0, "arcface_m": 0.5, "focal_gamma": 1.0, "label": "s30_m0.5_f1"},
        {"arcface_s": 64.0, "arcface_m": 0.3, "focal_gamma": 1.0, "label": "s64_m0.3_f1"},
        {"arcface_s": 30.0, "arcface_m": 0.3, "focal_gamma": 0.5, "label": "s30_m0.3_f0.5"},
    ]

    for cfg in configs:
        label = cfg.pop("label")
        console.print(f"\n{'='*60}")
        console.print(f"[bold yellow]Config: {label}[/bold yellow]")
        console.print(f"{'='*60}")
        train_ATN(
            data_file,
            os.path.join(base_dir, "results", f"ATN_{label}.json"),
            os.path.join(base_dir, "checkpoints"),
            **cfg,
        )
