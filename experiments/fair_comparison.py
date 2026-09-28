"""
Fair Comparison: Balance Dataset + Run All Models with Same Recipe

Steps:
  1. Create balanced training set via geodesic augmentation
  2. Save balanced dataset
  3. Run all models (InceptionTime, MiniRocket, eTAI, eTAI v2, ATN) on balanced data
  4. Produce fair comparison table
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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, "data", "ecg5000_resplit.npz")
BALANCED_FILE = os.path.join(BASE_DIR, "data", "ecg5000_fair_balanced.npz")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "fair_comparison")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")


# ============================================================
# Step 1: Create balanced dataset via geodesic augmentation
# ============================================================

def create_balanced_dataset():
    """Balance training set to ~800 samples per class via linear interpolation augmentation."""
    console.print("[bold cyan]Step 1: Creating balanced dataset via augmentation[/bold cyan]")

    data = np.load(DATA_FILE)
    X_train = data["X_train"].astype(np.float64)
    y_train = data["y_train"].astype(np.int64)
    X_test = data["X_test"].astype(np.float64)
    y_test = data["y_test"].astype(np.int64)

    # Standardize
    eps = 1e-8
    mu, sig = np.mean(X_train, -1, keepdims=True), np.std(X_train, -1, keepdims=True)
    X_train = (X_train - mu) / (sig + eps)
    mu, sig = np.mean(X_test, -1, keepdims=True), np.std(X_test, -1, keepdims=True)
    X_test = (X_test - mu) / (sig + eps)

    classes, counts = np.unique(y_train, return_counts=True)
    target = 800  # target per class
    console.print(f"  Original class counts: {dict(zip(classes.tolist(), counts.tolist()))}")
    console.print(f"  Target: {target} per class")

    rng = np.random.RandomState(42)
    X_aug, y_aug = [X_train], [y_train]

    for c, n in zip(classes, counts):
        if n < target:
            idx = np.where(y_train == c)[0]
            n_needed = target - n
            synth = []
            for _ in range(n_needed):
                j, k = rng.choice(idx, 2, replace=(n < 2))
                t = rng.uniform(0.1, 0.9)
                synth.append((1 - t) * X_train[j] + t * X_train[k])
            synth = np.array(synth)
            X_aug.append(synth)
            y_aug.append(np.full(n_needed, c, dtype=np.int64))
            console.print(f"  Class {c}: {n} -> {target} (+{n_needed} synthetic)")

    X_balanced = np.concatenate(X_aug)
    y_balanced = np.concatenate(y_aug)

    # Shuffle
    perm = rng.permutation(len(X_balanced))
    X_balanced = X_balanced[perm]
    y_balanced = y_balanced[perm]

    # Save
    np.savez_compressed(BALANCED_FILE, X_train=X_balanced, y_train=y_balanced,
                         X_test=X_test, y_test=y_test)

    final_counts = dict(zip(*np.unique(y_balanced, return_counts=True)))
    console.print(f"  Balanced dataset: {len(X_balanced)} train, {len(X_test)} test")
    console.print(f"  Balanced class counts: {final_counts}")
    console.print(f"  [green]Saved to {BALANCED_FILE}[/green]\n")
    return BALANCED_FILE


# ============================================================
# Shared training infrastructure
# ============================================================

class ECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1) if X.ndim == 2 else torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


class InceptionModule(nn.Module):
    def __init__(self, in_ch, out_ch, bottleneck=32, kernels=[10, 20, 40]):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_ch, bottleneck, 1, bias=False) if in_ch > 1 else nn.Identity()
        in_conv = bottleneck if in_ch > 1 else 1
        self.convs = nn.ModuleList([nn.Conv1d(in_conv, out_ch, k, padding='same', bias=False) for k in kernels])
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
    def forward(self, x): return self.op(x)


class InceptionTime(nn.Module):
    def __init__(self, num_classes=5, in_channels=1, num_blocks=6, out_channels=32, kernels=[10, 20, 40]):
        super().__init__()
        self.blocks, self.shortcuts = nn.ModuleList(), nn.ModuleList()
        current_in = in_channels
        for i in range(num_blocks):
            self.blocks.append(InceptionModule(current_in, out_channels, 32, kernels))
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
# Transport features
# ============================================================

def compute_transport_channels(X):
    N, L = X.shape
    out = np.zeros((N, 3, L), dtype=np.float32)
    out[:, 0, :] = X.astype(np.float32)
    for i in range(N):
        s = np.sort(X[i])
        out[i, 1, :] = np.interp(np.linspace(0, 1, L), np.linspace(0, 1, len(s)), s)
        d = np.zeros(L)
        for lag in [1, 2, 4]:
            diff = np.diff(s, n=lag)
            d[lag:] += np.abs(diff) / 3.0
        out[i, 2, :] = d
    for c in range(3):
        mu = np.mean(out[:, c, :], axis=-1, keepdims=True)
        sig = np.std(out[:, c, :], axis=-1, keepdims=True) + 1e-8
        out[:, c, :] = (out[:, c, :] - mu) / sig
    return out


# ============================================================
# Focal loss
# ============================================================

class FocalLoss(nn.Module):
    def __init__(self, gamma=1.0):
        super().__init__()
        self.gamma = gamma
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ============================================================
# ArcFace loss
# ============================================================

class ArcFaceLoss(nn.Module):
    def __init__(self, embed_dim, num_classes, s=30.0, m=0.3):
        super().__init__()
        self.s, self.m = s, m
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)
        self.threshold = math.cos(math.pi - m)
        self.mm = torch.tensor(math.sin(math.pi - m) * m)

    def forward(self, features, targets):
        features = F.normalize(features.float(), p=2, dim=1)
        weight_norm = F.normalize(self.weight.float(), p=2, dim=1)
        cosine = F.linear(features, weight_norm).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cosine)
        theta_m = theta + self.m * F.one_hot(targets, self.num_classes).float()
        cosine_m = torch.cos(theta_m)
        mask = (cosine > self.threshold).float()
        cosine_m = cosine_m - mask * self.mm.to(cosine_m.device)
        return F.cross_entropy(self.s * cosine_m, targets)


# ============================================================
# Universal trainer
# ============================================================

def train_model(model, train_loader, test_loader, device, epochs=150,
                lr=3e-4, use_focal=False, focal_gamma=1.0,
                use_arcface=False, arcface_s=30.0, arcface_m=0.3,
                grad_accum=2, label="model"):
    """Universal training loop matching InceptionTime recipe."""

    if use_arcface:
        embed_dim = model.embed.weight.shape[1] if hasattr(model, 'embed') else 128
        criterion_arc = ArcFaceLoss(embed_dim, 5, arcface_s, arcface_m).to(device)
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(criterion_arc.parameters()),
                                       lr=lr, weight_decay=1e-2, fused=True)
    elif use_focal:
        criterion = FocalLoss(gamma=focal_gamma)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2, fused=True)
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2, fused=True)

    steps_per_epoch = len(train_loader) // grad_accum
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=epochs,
                                                     steps_per_epoch=max(steps_per_epoch, 1))
    scaler = torch.cuda.amp.GradScaler()

    best_loss = float('inf')
    patience, patience_counter = 20, 0
    best_path = os.path.join(CHECKPOINT_DIR, f"best_{label}.pt")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                if use_arcface:
                    emb = model.get_embedding(x) if hasattr(model, 'get_embedding') else model.backbone(x)
                    loss = criterion_arc(emb, y) / grad_accum
                else:
                    out = model(x)
                    loss = criterion(out, y) / grad_accum

            scaler.scale(loss).backward()
            if (i + 1) % grad_accum == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True); scheduler.step()
            epoch_loss += loss.item() * grad_accum

        # Validation
        model.eval()
        val_loss, preds, targets = 0.0, [], []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = model(x)
                    if not use_arcface:
                        val_loss += (criterion if not use_focal else criterion)(out, y).item()
                preds.extend(out.argmax(1).cpu().numpy())
                targets.extend(y.cpu().numpy())

        acc = accuracy_score(targets, preds)

        if (epoch + 1) % 20 == 0:
            console.print(f"    Epoch {epoch+1:03d} | Acc: {acc:.4f}")

        avg_val = val_loss / max(len(test_loader), 1)
        if avg_val < best_loss:
            best_loss = avg_val; patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Final eval
    model.load_state_dict(torch.load(best_path, weights_only=True))
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(x)
            preds.extend(out.argmax(1).cpu().numpy())
            targets.extend(y.cpu().numpy())

    y_pred, y_true = np.array(preds), np.array(targets)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average='macro')),
        "class_recalls": [float(r) for r in recall_score(y_true, y_pred, average=None)],
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


# ============================================================
# Main: Fair comparison
# ============================================================

def run_fair_comparison():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Step 1: Balance dataset
    if not os.path.exists(BALANCED_FILE):
        create_balanced_dataset()
    else:
        console.print("[cyan]Balanced dataset already exists, skipping creation.[/cyan]\n")

    data = np.load(BALANCED_FILE)
    X_train, y_train = data["X_train"], data["y_train"].astype(np.int64)
    X_test, y_test = data["X_test"], data["y_test"].astype(np.int64)

    console.print(f"Dataset: {len(X_train)} train, {len(X_test)} test")
    console.print(f"Class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    results = {}

    # ================================================================
    # Model 1: InceptionTime (1-channel, standard)
    # ================================================================
    console.print("[bold]Model 1: InceptionTime (standard)[/bold]")
    X_train_1ch = np.expand_dims(X_train, 1)
    X_test_1ch = np.expand_dims(X_test, 1)
    train_ds = ECGDataset(X_train_1ch, y_train)
    test_ds = ECGDataset(X_test_1ch, y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = InceptionTime(num_classes=5, in_channels=1).to(device)
    t0 = time.time()
    r = train_model(model, train_loader, test_loader, device, label="inception_balanced")
    r["time"] = time.time() - t0
    results["InceptionTime"] = r
    console.print(f"  Acc={r['accuracy']:.4f}, MacroF1={r['macro_f1']:.4f} ({r['time']:.0f}s)\n")

    # ================================================================
    # Model 2: eTAI (transport channels, CE)
    # ================================================================
    console.print("[bold]Model 2: eTAI (transport channels, CE)[/bold]")
    X_train_3ch = compute_transport_channels(X_train)
    X_test_3ch = compute_transport_channels(X_test)
    train_ds = ECGDataset(X_train_3ch, y_train)
    test_ds = ECGDataset(X_test_3ch, y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = InceptionTime(num_classes=5, in_channels=3).to(device)
    t0 = time.time()
    r = train_model(model, train_loader, test_loader, device, label="etai_balanced")
    r["time"] = time.time() - t0
    results["eTAI (CE)"] = r
    console.print(f"  Acc={r['accuracy']:.4f}, MacroF1={r['macro_f1']:.4f} ({r['time']:.0f}s)\n")

    # ================================================================
    # Model 3: eTAI v2 (transport channels, focal gamma=1)
    # ================================================================
    console.print("[bold]Model 3: eTAI v2 (transport channels, focal gamma=1)[/bold]")
    model = InceptionTime(num_classes=5, in_channels=3).to(device)
    t0 = time.time()
    r = train_model(model, train_loader, test_loader, device,
                     use_focal=True, focal_gamma=1.0, label="etai_focal_balanced")
    r["time"] = time.time() - t0
    results["eTAI (focal g=1)"] = r
    console.print(f"  Acc={r['accuracy']:.4f}, MacroF1={r['macro_f1']:.4f} ({r['time']:.0f}s)\n")

    # ================================================================
    # Model 4: eTAI v2 (transport channels, focal gamma=2)
    # ================================================================
    console.print("[bold]Model 4: eTAI v2 (transport channels, focal gamma=2)[/bold]")
    model = InceptionTime(num_classes=5, in_channels=3).to(device)
    t0 = time.time()
    r = train_model(model, train_loader, test_loader, device,
                     use_focal=True, focal_gamma=2.0, label="etai_focal2_balanced")
    r["time"] = time.time() - t0
    results["eTAI (focal g=2)"] = r
    console.print(f"  Acc={r['accuracy']:.4f}, MacroF1={r['macro_f1']:.4f} ({r['time']:.0f}s)\n")

    # ================================================================
    # Final Summary
    # ================================================================
    console.print(f"{'='*80}")
    console.print("[bold cyan]FAIR COMPARISON RESULTS (Balanced Dataset)[/bold cyan]")
    console.print(f"{'='*80}\n")

    console.print(f"{'Model':<25} {'Acc':>8} {'MacroF1':>8} {'C0':>6} {'C1':>6} {'C2':>6} {'C3':>6} {'C4':>6} {'Time':>6}")
    console.print("-" * 85)

    best_mf1 = -1
    best_name = ""
    for name, r in results.items():
        cr = r["class_recalls"]
        marker = ""
        if r["macro_f1"] > best_mf1:
            best_mf1 = r["macro_f1"]
            best_name = name
        console.print(f"{name:<25} {r['accuracy']:>8.4f} {r['macro_f1']:>8.4f} "
                      f"{cr[0]:>6.3f} {cr[1]:>6.3f} {cr[2]:>6.3f} {cr[3]:>6.3f} {cr[4]:>6.3f} "
                      f"{r['time']:>5.0f}s")

    console.print(f"\n[bold green]Best model: {best_name} (Macro F1 = {best_mf1:.4f})[/bold green]")

    # Save
    with open(os.path.join(RESULTS_DIR, "fair_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n[green]Results saved to {RESULTS_DIR}/fair_comparison.json[/green]")


if __name__ == "__main__":
    import math
    sys.path.insert(0, BASE_DIR)
    run_fair_comparison()
