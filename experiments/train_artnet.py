"""
ARTNet Training Script — Fair Protocol

Exactly matches the conditions used for InceptionTime, eTAI-Focal, HCRMN-Lite:
  - Seed: 42
  - Split: 70/15/15 stratified
  - Optimizer: Adam, lr=1e-3
  - Loss: Cross-entropy (CE only for first experiment)
  - Early stopping: patience=10 on val MF1
  - Max epochs: 30

Usage:
  python train_artnet.py ECG5000_UNBAL          # Run one dataset
  python train_artnet.py all                     # Run all 4 datasets
  python train_artnet.py ECG5000_UNBAL focal     # Use focal loss (gamma=1)
"""

import os, sys, json, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.artnet import ARTNet, artnet_loss, count_parameters

# ============================================================
# Configuration
# ============================================================

SEED = 42
MAX_EPOCHS = 30
PATIENCE = 10
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
VAL_RATIO = 0.15  # 15% of training data for validation

DATASETS = {
    "ECG5000_UNBAL": {"num_classes": 5, "file": "data/ecg5000_resplit.npz"},
    "ECG5000_BAL": {"num_classes": 5, "file": "data/ecg5000_fair_balanced.npz"},
    "BEARING_UNBAL": {"num_classes": 4, "file": "data/bearing_unbalanced.npz"},
    "BEARING_BAL": {"num_classes": 4, "file": "data/bearing_balanced.npz"},
}

# Loss weights (conservative, as spec recommends)
LAMBDA_SMOOTH = 0.01
LAMBDA_LOCAL = 0.01
LAMBDA_DYN = 0.02
LAMBDA_VAR = 0.01
LAMBDA_GATE = 0.001


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(dataset_name):
    """Load pre-split data and carve 15% of train for validation."""
    cfg = DATASETS[dataset_name]
    data_path = os.path.join(ROOT, cfg["file"])
    data = np.load(data_path)
    
    X_train_full = data["X_train"]  # [N_train, L]
    y_train_full = data["y_train"].astype(int)
    X_test = data["X_test"]  # [N_test, L]
    y_test = data["y_test"].astype(int)
    
    N_train, L = X_train_full.shape
    nc = cfg["num_classes"]
    
    # Stratified split of train into train+val (85%/15%)
    set_seed(SEED)
    indices = np.arange(N_train)
    perm = np.random.permutation(N_train)
    indices = indices[perm]
    
    train_idx, val_idx = [], []
    for c in range(nc):
        class_idx = indices[y_train_full[indices] == c]
        n = len(class_idx)
        n_val = max(1, int(0.15 * n))
        val_idx.extend(class_idx[:n_val])
        train_idx.extend(class_idx[n_val:])
    
    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    
    # Z-normalization (fit on train split only)
    mu = X_train_full[train_idx].mean(axis=(0, 1), keepdims=True)
    sigma = X_train_full[train_idx].std(axis=(0, 1), keepdims=True) + 1e-8
    
    X_train = (X_train_full[train_idx] - mu) / sigma
    X_val = (X_train_full[val_idx] - mu) / sigma
    X_test_n = (X_test - mu) / sigma
    
    # Add channel dim
    X_train = X_train[:, np.newaxis, :]
    X_val = X_val[:, np.newaxis, :]
    X_test_n = X_test_n[:, np.newaxis, :]
    
    return {
        "X_train": X_train, "y_train": y_train_full[train_idx],
        "X_val": X_val, "y_val": y_train_full[val_idx],
        "X_test": X_test_n, "y_test": y_test,
        "num_classes": nc, "seq_len": L, "n_channels": 1,
    }


def make_loader(X, y, batch_size, shuffle=True):
    """Create a simple DataLoader-like generator."""
    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X), torch.LongTensor(y)
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False
    )


class FocalLoss(nn.Module):
    def __init__(self, gamma=1.0):
        super().__init__()
        self.gamma = gamma
    
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


def train_and_evaluate(dataset_name, use_focal=False, gamma=1.0):
    """Train ARTNet on one dataset and evaluate."""
    print(f"\n{'='*60}")
    print(f"  ARTNet on {dataset_name}" + (" + FOCAL" if use_focal else " + CE"))
    print(f"{'='*60}")
    
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load data
    data = load_data(dataset_name)
    nc = data["num_classes"]
    sl = data["seq_len"]
    
    print(f"Train: {len(data['y_train'])}, Val: {len(data['y_val'])}, Test: {len(data['y_test'])}")
    print(f"Classes: {nc}, Seq len: {sl}")
    
    train_loader = make_loader(data["X_train"], data["y_train"], BATCH_SIZE, shuffle=True)
    val_loader = make_loader(data["X_val"], data["y_val"], BATCH_SIZE, shuffle=False)
    test_loader = make_loader(data["X_test"], data["y_test"], BATCH_SIZE, shuffle=False)
    
    # Model
    model = ARTNet(in_channels=1, num_classes=nc, seq_len=sl).to(device)
    params = count_parameters(model)
    print(f"Parameters: {params:,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, epochs=MAX_EPOCHS,
        steps_per_epoch=len(train_loader)
    )
    
    # Loss
    if use_focal:
        criterion = FocalLoss(gamma=gamma)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Training
    best_val_mf1 = -1
    best_state = None
    patience_counter = 0
    best_epoch = 0
    
    start_time = time.time()
    
    for epoch in range(MAX_EPOCHS):
        model.train()
        total_loss = 0
        n_batches = 0
        
        prev_regime = None
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            
            out = model(xb)
            
            # Classification loss
            loss = criterion(out["logits"], yb)
            
            # Auxiliary losses (lightweight)
            aux_loss = torch.tensor(0.0, device=device)
            
            # Smoothness
            if prev_regime is not None and prev_regime.shape[0] == out["regime"].shape[0]:
                aux_loss = aux_loss + LAMBDA_SMOOTH * F.mse_loss(
                    out["regime"], prev_regime.detach()
                )
            
            # Variance regularization
            regime_std = out["regime_std"].mean(dim=0)
            aux_loss = aux_loss + LAMBDA_VAR * F.relu(0.1 - regime_std).pow(2).mean()
            
            # Gate sparsity
            aux_loss = aux_loss + LAMBDA_GATE * out["gate"].abs().mean()
            
            # Dynamics (use current regime as "next" target for within-batch)
            if n_batches > 0:
                aux_loss = aux_loss + LAMBDA_DYN * F.mse_loss(
                    out["predicted_next_regime"], out["regime"].detach()
                )
            
            total_batch_loss = loss + aux_loss
            
            optimizer.zero_grad()
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += total_batch_loss.item()
            n_batches += 1
            prev_regime = out["regime"].detach()
        
        avg_loss = total_loss / max(n_batches, 1)
        
        # Validate
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                out = model(xb)
                preds = out["logits"].argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(yb.numpy())
        
        val_mf1 = f1_score(val_true, val_preds, average="macro", zero_division=0)
        val_acc = accuracy_score(val_true, val_preds)
        
        elapsed = time.time() - start_time
        print(f"  Epoch {epoch+1:2d}: loss={avg_loss:.4f}  val_MF1={val_mf1:.4f}  val_Acc={val_acc:.4f}  ({elapsed:.0f}s)")
        
        # Early stopping
        if val_mf1 > best_val_mf1:
            best_val_mf1 = val_mf1
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    # Evaluate on test set with best model
    total_time = time.time() - start_time
    model.load_state_dict(best_state)
    model.eval()
    
    test_preds, test_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            out = model(xb)
            preds = out["logits"].argmax(dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_true.extend(yb.numpy())
    
    test_preds = np.array(test_preds)
    test_true = np.array(test_true)
    
    acc = accuracy_score(test_true, test_preds)
    mf1 = f1_score(test_true, test_preds, average="macro", zero_division=0)
    wf1 = f1_score(test_true, test_preds, average="weighted", zero_division=0)
    
    # Per-class metrics
    class_f1s = f1_score(test_true, test_preds, average=None, zero_division=0).tolist()
    class_recalls = []
    cm = confusion_matrix(test_true, test_preds, labels=list(range(nc)))
    for c in range(nc):
        row_sum = cm[c].sum()
        class_recalls.append(cm[c, c] / row_sum if row_sum > 0 else 0)
    
    support = {str(c): int(cm[c].sum()) for c in range(nc)}
    
    print(f"\n  TEST: Acc={acc:.4f}  MF1={mf1:.4f}  WtF1={wf1:.4f}  (best_epoch={best_epoch})")
    print(f"  Per-class F1: {['C%d=%.3f' % (c, class_f1s[c]) for c in range(nc)]}")
    
    # Save results
    result = {
        "model": f"ARTNet{'-Focal' if use_focal else '-CE'}",
        "dataset": dataset_name,
        "accuracy": acc,
        "macro_f1": mf1,
        "weighted_f1": wf1,
        "class_f1s": class_f1s,
        "class_recalls": class_recalls,
        "confusion_matrix": cm.tolist(),
        "support": support,
        "params": params,
        "best_epoch": best_epoch,
        "total_epochs": epoch + 1,
        "time_s": total_time,
        "use_focal": use_focal,
        "gamma": gamma if use_focal else 0,
    }
    
    # Save checkpoint
    ckpt_dir = os.path.join(ROOT, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    focal_tag = f"_FOC{gamma}" if use_focal else ""
    ckpt_path = os.path.join(ckpt_dir, f"{dataset_name}_ARTNET{focal_tag}.pt")
    torch.save(best_state, ckpt_path)
    print(f"  Saved: {ckpt_path}")
    
    # Save result JSON
    result_dir = os.path.join(ROOT, "results", "artnet")
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, f"{dataset_name}.json")
    
    # Merge with existing results
    existing = {}
    if os.path.exists(result_path):
        with open(result_path) as f:
            existing = json.load(f)
    
    key = f"ARTNet{'-Focal' if use_focal else '-CE'}"
    if use_focal:
        key += f" g={gamma}"
    existing[key] = result
    
    with open(result_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"  Saved: {result_path}")
    
    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python train_artnet.py <DATASET|all> [focal]")
        sys.exit(1)
    
    dataset_arg = sys.argv[1]
    use_focal = len(sys.argv) > 2 and sys.argv[2] == "focal"
    gamma = 1.0
    if use_focal and len(sys.argv) > 3:
        gamma = float(sys.argv[3])
    
    if dataset_arg == "all":
        datasets_to_run = list(DATASETS.keys())
    else:
        datasets_to_run = [dataset_arg]
    
    results = []
    for ds in datasets_to_run:
        if ds not in DATASETS:
            print(f"Unknown dataset: {ds}")
            continue
        r = train_and_evaluate(ds, use_focal=use_focal, gamma=gamma)
        results.append(r)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        tag = f"+Focal(g={r['gamma']})" if r['use_focal'] else "+CE"
        print(f"  {r['dataset']:20s} {tag:15s} MF1={r['macro_f1']:.4f}  Acc={r['accuracy']:.4f}  p={r['params']:,}")


if __name__ == "__main__":
    main()
