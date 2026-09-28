"""
USTR-Net Training Script — Fair Protocol

Usage:
  python train_ustrnet.py ECG5000_UNBAL          # Run one dataset, CE loss
  python train_ustrnet.py all                     # Run all 4 datasets, CE loss
  python train_ustrnet.py ECG5000_UNBAL focal     # Use focal loss (gamma=1)
  python train_ustrnet.py all focal               # All datasets, focal loss
"""

import os, sys, json, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.ustrnet import USTRNet, count_parameters

SEED = 42
MAX_EPOCHS = 30
PATIENCE = 10
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4

DATASETS = {
    "ECG5000_UNBAL": {"num_classes": 5, "file": "data/ecg5000_resplit.npz"},
    "ECG5000_BAL": {"num_classes": 5, "file": "data/ecg5000_fair_balanced.npz"},
    "BEARING_UNBAL": {"num_classes": 4, "file": "data/bearing_unbalanced.npz"},
    "BEARING_BAL": {"num_classes": 4, "file": "data/bearing_balanced.npz"},
}

# Regime regularization weights
LAMBDA_SMOOTH = 0.01
LAMBDA_VEL = 0.005
LAMBDA_UNC = 0.02


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(dataset_name):
    cfg = DATASETS[dataset_name]
    data_path = os.path.join(ROOT, cfg["file"])
    data = np.load(data_path)
    
    X_train_full = data["X_train"]
    y_train_full = data["y_train"].astype(int)
    X_test = data["X_test"]
    y_test = data["y_test"].astype(int)
    
    N_train, L = X_train_full.shape
    nc = cfg["num_classes"]
    
    # Stratified train/val split
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
    
    # Z-normalization
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


class FocalLoss(nn.Module):
    def __init__(self, gamma=1.0):
        super().__init__()
        self.gamma = gamma
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def train_and_evaluate(dataset_name, use_focal=False, gamma=1.0):
    print(f"\n{'='*60}")
    print(f"  USTR-Net on {dataset_name}" + (" + FOCAL" if use_focal else " + CE"))
    print(f"{'='*60}")
    
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    data = load_data(dataset_name)
    nc = data["num_classes"]
    sl = data["seq_len"]
    
    print(f"Train: {len(data['y_train'])}, Val: {len(data['y_val'])}, Test: {len(data['y_test'])}")
    
    def make_loader(X, y, bs=64, shuffle=True):
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.FloatTensor(X), torch.LongTensor(y)),
            batch_size=bs, shuffle=shuffle)
    
    train_loader = make_loader(data["X_train"], data["y_train"], BATCH_SIZE, True)
    val_loader = make_loader(data["X_val"], data["y_val"], BATCH_SIZE, False)
    test_loader = make_loader(data["X_test"], data["y_test"], BATCH_SIZE, False)
    
    model = USTRNet(in_channels=1, num_classes=nc, regime_dim=8, base_ch=32).to(device)
    params = count_parameters(model)
    print(f"Parameters: {params:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, epochs=MAX_EPOCHS, steps_per_epoch=len(train_loader))
    
    criterion = FocalLoss(gamma=gamma) if use_focal else nn.CrossEntropyLoss()
    
    best_val_mf1, best_state, best_epoch, patience_ct = -1, None, 0, 0
    start_time = time.time()
    
    for epoch in range(MAX_EPOCHS):
        model.train()
        total_loss = 0
        n_batches = 0
        prev_z = None
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb, prev_z=prev_z)
            
            # Task loss
            task_loss = criterion(out["logits"], yb)
            
            # Regime smoothness (penalize large regime jumps)
            smooth_loss = torch.tensor(0.0, device=device)
            if prev_z is not None and prev_z.shape[0] == out["regime"].shape[0]:
                smooth_loss = F.mse_loss(out["regime"], prev_z.detach())
            
            # Regime velocity consistency
            vel_loss = out["regime_velocity"].pow(2).mean()
            
            # Uncertainty calibration: u should predict 1 - p_max
            with torch.no_grad():
                p_max = F.softmax(out["logits"], dim=1).max(dim=1)[0]
                target_unc = (1 - p_max).unsqueeze(1)  # [B, 1]
            unc_loss = F.mse_loss(out["uncertainty"], target_unc)
            
            loss = task_loss + LAMBDA_SMOOTH * smooth_loss + LAMBDA_VEL * vel_loss + LAMBDA_UNC * unc_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            n_batches += 1
            prev_z = out["prev_z"]
        
        # Validate
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                out = model(xb.to(device))
                val_preds.extend(out["logits"].argmax(1).cpu().numpy())
                val_true.extend(yb.numpy())
        
        val_mf1 = f1_score(val_true, val_preds, average="macro", zero_division=0)
        elapsed = time.time() - start_time
        print(f"  Epoch {epoch+1:2d}: loss={total_loss/max(n_batches,1):.4f}  val_MF1={val_mf1:.4f}  ({elapsed:.0f}s)")
        
        if val_mf1 > best_val_mf1:
            best_val_mf1, best_epoch, best_state = val_mf1, epoch+1, copy.deepcopy(model.state_dict())
            patience_ct = 0
        else:
            patience_ct += 1
            if patience_ct >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    # Test
    total_time = time.time() - start_time
    model.load_state_dict(best_state)
    model.eval()
    
    test_preds, test_true = [], []
    all_uncertainties = []
    with torch.no_grad():
        for xb, yb in test_loader:
            out = model(xb.to(device))
            test_preds.extend(out["logits"].argmax(1).cpu().numpy())
            test_true.extend(yb.numpy())
            all_uncertainties.extend(out["uncertainty"].squeeze().cpu().numpy())
    
    test_preds = np.array(test_preds)
    test_true = np.array(test_true)
    
    acc = accuracy_score(test_true, test_preds)
    mf1 = f1_score(test_true, test_preds, average="macro", zero_division=0)
    wf1 = f1_score(test_true, test_preds, average="weighted", zero_division=0)
    class_f1s = f1_score(test_true, test_preds, average=None, zero_division=0).tolist()
    cm = confusion_matrix(test_true, test_preds, labels=list(range(nc)))
    class_recalls = [cm[c, c] / cm[c].sum() if cm[c].sum() > 0 else 0 for c in range(nc)]
    support = {str(c): int(cm[c].sum()) for c in range(nc)}
    avg_uncertainty = float(np.mean(all_uncertainties))
    
    print(f"\n  TEST: Acc={acc:.4f}  MF1={mf1:.4f}  WtF1={wf1:.4f}")
    print(f"  Per-class F1: {['C%d=%.3f' % (c, class_f1s[c]) for c in range(nc)]}")
    print(f"  Avg uncertainty: {avg_uncertainty:.4f}")
    
    # Save checkpoint
    ckpt_dir = os.path.join(ROOT, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    focal_tag = f"_FOC{gamma}" if use_focal else ""
    ckpt_path = os.path.join(ckpt_dir, f"{dataset_name}_USTRNET{focal_tag}.pt")
    torch.save(best_state, ckpt_path)
    
    # Save result
    result = {
        "model": f"USTRNet{'-Focal' if use_focal else '-CE'}",
        "dataset": dataset_name, "accuracy": acc, "macro_f1": mf1,
        "weighted_f1": wf1, "class_f1s": class_f1s, "class_recalls": class_recalls,
        "confusion_matrix": cm.tolist(), "support": support,
        "params": params, "best_epoch": best_epoch, "total_epochs": epoch+1,
        "time_s": total_time, "avg_uncertainty": avg_uncertainty,
        "use_focal": use_focal, "gamma": gamma if use_focal else 0,
    }
    
    result_dir = os.path.join(ROOT, "results", "ustrnet")
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, f"{dataset_name}.json")
    
    existing = {}
    if os.path.exists(result_path):
        with open(result_path) as f:
            existing = json.load(f)
    
    key = f"USTRNet{'-Focal' if use_focal else '-CE'}"
    if use_focal:
        key += f" g={gamma}"
    existing[key] = result
    with open(result_path, "w") as f:
        json.dump(existing, f, indent=2)
    
    print(f"  Saved: {result_path}")
    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python train_ustrnet.py <DATASET|all> [focal [gamma]]")
        sys.exit(1)
    
    ds_arg = sys.argv[1]
    use_focal = len(sys.argv) > 2 and sys.argv[2] == "focal"
    gamma = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    
    datasets_to_run = list(DATASETS.keys()) if ds_arg == "all" else [ds_arg]
    
    for ds in datasets_to_run:
        if ds not in DATASETS:
            print(f"Unknown dataset: {ds}")
            continue
        train_and_evaluate(ds, use_focal=use_focal, gamma=gamma)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for ds in datasets_to_run:
        fp = os.path.join(ROOT, "results", "ustrnet", f"{ds}.json")
        if os.path.exists(fp):
            with open(fp) as f:
                d = json.load(f)
            for k, r in d.items():
                tag = f"+Focal(g={r['gamma']})" if r.get('use_focal') else "+CE"
                print(f"  {r['dataset']:20s} {tag:15s} MF1={r['macro_f1']:.4f}  Acc={r['accuracy']:.4f}  p={r['params']:,}")


if __name__ == "__main__":
    main()
