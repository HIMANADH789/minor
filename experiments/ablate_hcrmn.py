"""
HCRMN Ablation Study + Corrected Comparison Report

Ablations:
  1. NoFiLM      — bypass FiLM modulation (h_modulated = h)
  2. NoCrossAttn  — bypass cross-attention (concat instead)
  3. NoGraph      — bypass graph propagation (raw regimes only)
  4. NoDynamics   — lambda_dyn=0

Also reports support (n) for each class.
"""
import os, sys, json, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, recall_score, accuracy_score

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
sys.path.insert(0, ROOT)

from models.hcrmn import HCRMN, HCRMNLoss, compute_transport_channels


# ============================================================
# Ablated HCRMN variants
# ============================================================
class HCRMN_NoFiLM(nn.Module):
    """HCRMN with FiLM modulation bypassed — h_modulated = h"""
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
    
    def forward(self, x):
        logits, info = self.base(x)
        # Override: skip FiLM
        info['h_modulated'] = info['h']
        # Recompute MoE logits with unmodulated h
        prototypes = info['prototypes']
        h = info['h']
        z_f, z_m, z_c = info['z_fine'], info['z_mid'], info['z_coarse']
        # Run moe with original h
        logits2, moe_info2 = self.base.moe(h, z_f, z_m, z_c, prototypes)
        info['moe_info'] = moe_info2
        return logits2, info


class HCRMN_NoCrossAttn(nn.Module):
    """HCRMN with cross-attention replaced by simple concat+proj"""
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        # Replace cross-attention with simple concat
        hidden_ch = base_model.raw_path.hidden_ch
        self.simple_fuse = nn.Sequential(
            nn.Conv1d(hidden_ch + 64, hidden_ch, 1, bias=False),
            nn.BatchNorm1d(hidden_ch),
            nn.GELU(),
        )
    
    def forward(self, x):
        B = x.shape[0]
        raw = x[:, :1, :]
        x_tr = x[:, 1:4, :]
        
        H_raw, h_seq = self.base.raw_path(raw)
        H_tr = self.base.transport_encoder(x_tr)
        
        # Simple concat instead of cross-attention
        H_fused = self.simple_fuse(torch.cat([h_seq, H_tr], dim=1))
        H_fused_pooled = H_fused.mean(dim=-1)
        H_fused_proj = self.base.fused_norm(self.base.fused_proj(H_fused_pooled))
        h = H_raw + H_fused_proj
        
        z_fine, z_mid, z_coarse = self.base.hier_regime(h)
        z_f, z_m, z_c, adj_info = self.base.regime_graph(z_fine, z_mid, z_coarse)
        h_modulated = self.base.film(z_f, z_m, z_c, h)
        
        prototypes = [self.base.regime_graph.prototypes_fine,
                      self.base.regime_graph.prototypes_mid,
                      self.base.regime_graph.prototypes_coarse]
        logits, moe_info = self.base.moe(h_modulated, z_f, z_m, z_c, prototypes)
        
        z_next, vel, accel, cg = self.base.dynamics(z_f, h_modulated)
        
        info = {'h': h, 'h_modulated': h_modulated, 'H_raw': H_raw,
                'z_fine': z_f, 'z_mid': z_m, 'z_coarse': z_c,
                'z_fine_raw': z_fine, 'z_mid_raw': z_mid, 'z_coarse_raw': z_coarse,
                'z_next': z_next, 'velocity': vel, 'acceleration': accel,
                'change_gate': cg, 'prototypes': prototypes,
                'adj_info': adj_info, 'moe_info': moe_info, 'H_reg': moe_info['H_reg']}
        return logits, info


class HCRMN_NoGraph(nn.Module):
    """HCRMN with graph propagation bypassed — uses raw regime outputs"""
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
    
    def forward(self, x):
        B = x.shape[0]
        raw = x[:, :1, :]
        x_tr = x[:, 1:4, :]
        
        H_raw, h_seq = self.base.raw_path(raw)
        H_tr = self.base.transport_encoder(x_tr)
        H_fused = self.base.cross_attn(h_seq, H_tr)
        H_fused_pooled = H_fused.mean(dim=-1)
        H_fused_proj = self.base.fused_norm(self.base.fused_proj(H_fused_pooled))
        h = H_raw + H_fused_proj
        
        z_fine, z_mid, z_coarse = self.base.hier_regime(h)
        # SKIP graph — use raw regimes directly
        z_f, z_m, z_c = z_fine, z_mid, z_coarse
        
        h_modulated = self.base.film(z_f, z_m, z_c, h)
        
        prototypes = [self.base.regime_graph.prototypes_fine,
                      self.base.regime_graph.prototypes_mid,
                      self.base.regime_graph.prototypes_coarse]
        logits, moe_info = self.base.moe(h_modulated, z_f, z_m, z_c, prototypes)
        
        z_next, vel, accel, cg = self.base.dynamics(z_f, h_modulated)
        
        K = self.base.regime_graph.K
        adj_info = {'adj_fine': torch.eye(K, device=x.device), 
                     'adj_mid': torch.eye(K, device=x.device),
                     'adj_coarse': torch.eye(K, device=x.device),
                     'adj_f2m': torch.ones(K, K, device=x.device) / K,
                     'adj_m2c': torch.ones(K, K, device=x.device) / K}
        
        info = {'h': h, 'h_modulated': h_modulated, 'H_raw': H_raw,
                'z_fine': z_f, 'z_mid': z_m, 'z_coarse': z_c,
                'z_fine_raw': z_fine, 'z_mid_raw': z_mid, 'z_coarse_raw': z_coarse,
                'z_next': z_next, 'velocity': vel, 'acceleration': accel,
                'change_gate': cg, 'prototypes': prototypes,
                'adj_info': adj_info, 'moe_info': moe_info, 'H_reg': moe_info['H_reg']}
        return logits, info


# ============================================================
# Training and evaluation
# ============================================================
def load_data(dataset_key, val_frac=0.15):
    """Load dataset and return train/val/test loaders + support counts."""
    data_dir = os.path.join(ROOT, 'data')
    if 'ECG5000' in dataset_key:
        if 'UNBAL' in dataset_key:
            d = np.load(os.path.join(data_dir, 'ecg5000_resplit.npz'))
        else:
            d = np.load(os.path.join(data_dir, 'ecg5000_fair_balanced.npz'))
        X_all, y_all = d['X_train'], d['y_train'].astype(int)
        X_test, y_test = d['X_test'], d['y_test'].astype(int)
        n_cls = 5
    else:
        if 'UNBAL' in dataset_key:
            d = np.load(os.path.join(data_dir, 'bearing_unbalanced.npz'))
        else:
            d = np.load(os.path.join(data_dir, 'bearing_balanced.npz'))
        X_all, y_all = d['X_train'], d['y_train'].astype(int)
        X_test, y_test = d['X_test'], d['y_test'].astype(int)
        n_cls = 4

    # Stratified train/val split from train set
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=val_frac, stratify=y_all, random_state=42
    )

    # Compute transport channels for all splits
    X_tr_train = compute_transport_channels(X_train)
    X_tr_val = compute_transport_channels(X_val)
    X_tr_test = compute_transport_channels(X_test)
    
    # Stack: (N, 1+3, L) = (N, 4, L)
    Xtrn = np.concatenate([X_train[:, None, :], X_tr_train], axis=1).astype(np.float32)
    Xval = np.concatenate([X_val[:, None, :], X_tr_val], axis=1).astype(np.float32)
    Xtst = np.concatenate([X_test[:, None, :], X_tr_test], axis=1).astype(np.float32)
    
    # Support per class in test set
    support = {}
    for c in range(n_cls):
        support[c] = int((y_test == c).sum())
    
    # z-normalize per channel
    mu = Xtrn.mean(axis=(0, 2), keepdims=True)
    sig = Xtrn.std(axis=(0, 2), keepdims=True) + 1e-8
    Xtrn = (Xtrn - mu) / sig
    Xval = (Xval - mu) / sig
    Xtst = (Xtst - mu) / sig
    
    def make_loader(X, y, batch, shuffle):
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).long())
        return DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=False)
    
    train_loader = make_loader(Xtrn, y_train, 64, True)
    val_loader = make_loader(Xval, y_val, 256, False)
    test_loader = make_loader(Xtst, y_test, 256, False)
    
    return train_loader, val_loader, test_loader, n_cls, support


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    n = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits, info = model(xb)
        loss, _ = criterion(logits, info, yb)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * xb.shape[0]
        n += xb.shape[0]
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, criterion, device, n_cls):
    model.eval()
    all_preds, all_targets = [], []
    total_loss = 0
    n = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits, info = model(xb)
        loss, _ = criterion(logits, info, yb)
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(yb.cpu().numpy())
        total_loss += loss.item() * xb.shape[0]
        n += xb.shape[0]
    
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    
    acc = accuracy_score(targets, preds)
    mf1 = f1_score(targets, preds, average='macro', zero_division=0)
    wf1 = f1_score(targets, preds, average='weighted', zero_division=0)
    recalls = recall_score(targets, preds, average=None, zero_division=0, labels=list(range(n_cls)))
    f1s_per = f1_score(targets, preds, average=None, zero_division=0, labels=list(range(n_cls)))
    
    # Confusion matrix
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(targets, preds):
        cm[t][p] += 1
    
    return {
        'accuracy': round(acc, 4),
        'macro_f1': round(mf1, 4),
        'weighted_f1': round(wf1, 4),
        'class_recalls': [round(r, 4) for r in recalls],
        'class_f1s': [round(f, 4) for f in f1s_per],
        'confusion_matrix': cm.tolist(),
        'loss': total_loss / n,
    }


def train_and_eval(dataset_key, variant, model, device, n_cls, 
                   train_loader, val_loader, test_loader, 
                   focal_gamma=1.0, max_epochs=25, patience=12,
                   extra_kwargs=None):
    """Train model and evaluate on test set."""
    save_dir = os.path.join(ROOT, 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    tag = f"{dataset_key}_{variant}"
    ckpt_path = os.path.join(save_dir, f"HCRMN_ABL_{tag}.pt")
    
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"  Loaded {tag} from checkpoint")
    else:
        criterion = HCRMNLoss(num_classes=n_cls, feat_dim=model.feat_dim if hasattr(model, 'feat_dim') else 128,
                              focal_gamma=focal_gamma, **(extra_kwargs or {}))
        criterion = criterion.to(device)
        
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(criterion.parameters()),
            lr=3e-4, weight_decay=1e-2
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=3e-4, 
            steps_per_epoch=len(train_loader),
            epochs=max_epochs, pct_start=0.3
        )
        
        best_val_loss = float('inf')
        best_state = None
        no_improve = 0
        t0 = time.time()
        
        for ep in range(max_epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_res = evaluate(model, val_loader, criterion, device, n_cls)
            scheduler.step()
            
            if val_res['loss'] < best_val_loss:
                best_val_loss = val_res['loss']
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve >= patience:
                print(f"  Epoch {ep+1}: early stop (best_val={best_val_loss:.4f})")
                break
            
            if (ep+1) % 5 == 0:
                print(f"  Epoch {ep+1}: train_loss={train_loss:.4f} val_mf1={val_res['macro_f1']:.4f}")
        
        elapsed = time.time() - t0
        if best_state is not None:
            model.load_state_dict(best_state)
        torch.save({
            'model_state_dict': model.state_dict(),
            'tag': tag,
            'time': elapsed,
        }, ckpt_path)
        print(f"  Trained {tag} in {elapsed:.1f}s")
    
    # Evaluate on test
    criterion = HCRMNLoss(num_classes=n_cls, feat_dim=model.feat_dim if hasattr(model, 'feat_dim') else 128,
                          focal_gamma=focal_gamma)
    criterion = criterion.to(device)
    
    test_res = evaluate(model, test_loader, criterion, device, n_cls)
    test_res['params'] = sum(p.numel() for p in model.parameters())
    
    return test_res


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    os.makedirs(os.path.join(ROOT, 'results', 'hcrmn_ablation'), exist_ok=True)
    
    # Ablation configs
    configs = [
        ('NoFiLM', 'HCRMN_NoFiLM'),
        ('NoCrossAttn', 'HCRMN_NoCrossAttn'),
        ('NoGraph', 'HCRMN_NoGraph'),
        ('NoDynamics', 'HCRMN_NoDynamics'),
    ]
    
    datasets = ['ECG5000_UNBAL', 'ECG5000_BAL', 'BEARING_UNBAL', 'BEARING_BAL']
    
    all_results = {}
    
    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds}")
        print(f"{'='*60}")
        
        train_loader, val_loader, test_loader, n_cls, support = load_data(ds)
        print(f"  Classes: {n_cls}, Test support: {support}")
        
        ds_results = {'support': support, 'configs': {}}
        
        for cfg_name, cls_name in configs:
            print(f"\n  --- {cfg_name} ---")
            
            # Create base model
            base_model = HCRMN(in_channels=4, num_classes=n_cls).to(device)
            
            if cls_name == 'HCRMN_NoFiLM':
                model = HCRMN_NoFiLM(base_model).to(device)
            elif cls_name == 'HCRMN_NoCrossAttn':
                model = HCRMN_NoCrossAttn(base_model).to(device)
            elif cls_name == 'HCRMN_NoGraph':
                model = HCRMN_NoGraph(base_model).to(device)
            elif cls_name == 'HCRMN_NoDynamics':
                model = base_model  # Use base with modified loss
                # Train with lambda_dyn=0
                save_dir = os.path.join(ROOT, 'checkpoints')
                tag = f"{ds}_NoDynamics"
                ckpt_path = os.path.join(save_dir, f"HCRMN_ABL_{tag}.pt")
                
                if os.path.exists(ckpt_path):
                    ckpt = torch.load(ckpt_path, map_location=device)
                    model.load_state_dict(ckpt['model_state_dict'])
                    print(f"  Loaded {tag} from checkpoint")
                else:
                    criterion = HCRMNLoss(num_classes=n_cls, feat_dim=128,
                                          focal_gamma=1.0, lambda_dyn=0.0).to(device)
                    optimizer = torch.optim.AdamW(
                        list(model.parameters()) + list(criterion.parameters()),
                        lr=3e-4, weight_decay=1e-2
                    )
                    scheduler = torch.optim.lr_scheduler.OneCycleLR(
                        optimizer, max_lr=3e-4,
                        steps_per_epoch=len(train_loader),
                        epochs=25, pct_start=0.3
                    )
                    
                    best_val_loss = float('inf')
                    best_state = None
                    no_improve = 0
                    t0 = time.time()
                    
                    for ep in range(25):
                        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
                        val_res = evaluate(model, val_loader, criterion, device, n_cls)
                        scheduler.step()
                        
                        if val_res['loss'] < best_val_loss:
                            best_val_loss = val_res['loss']
                            best_state = copy.deepcopy(model.state_dict())
                            no_improve = 0
                        else:
                            no_improve += 1
                        
                        if no_improve >= 12:
                            break
                        if (ep+1) % 5 == 0:
                            print(f"  Epoch {ep+1}: train={train_loss:.4f} val_mf1={val_res['macro_f1']:.4f}")
                    
                    elapsed = time.time() - t0
                    if best_state is not None:
                        model.load_state_dict(best_state)
                    torch.save({'model_state_dict': model.state_dict(), 'tag': tag, 'time': elapsed},
                               ckpt_path)
                    print(f"  Trained {tag} in {elapsed:.1f}s")
                
                # Evaluate
                criterion_eval = HCRMNLoss(num_classes=n_cls, feat_dim=128, focal_gamma=1.0)
                criterion_eval = criterion_eval.to(device)
                test_res = evaluate(model, test_loader, criterion_eval, device, n_cls)
                test_res['params'] = sum(p.numel() for p in model.parameters())
                ds_results['configs'][cfg_name] = test_res
                print(f"  {cfg_name}: MF1={test_res['macro_f1']:.4f}, Acc={test_res['accuracy']:.4f}")
                continue
            
            # For NoFiLM, NoCrossAttn, NoGraph — train from scratch
            save_dir = os.path.join(ROOT, 'checkpoints')
            tag = f"{ds}_{cfg_name}"
            ckpt_path = os.path.join(save_dir, f"HCRMN_ABL_{tag}.pt")
            
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(ckpt['model_state_dict'])
                print(f"  Loaded {tag} from checkpoint")
            else:
                criterion = HCRMNLoss(num_classes=n_cls, feat_dim=128,
                                      focal_gamma=1.0).to(device)
                optimizer = torch.optim.AdamW(
                    list(model.parameters()) + list(criterion.parameters()),
                    lr=3e-4, weight_decay=1e-2
                )
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer, max_lr=3e-4,
                    steps_per_epoch=len(train_loader),
                    epochs=25, pct_start=0.3
                )
                
                best_val_loss = float('inf')
                best_state = None
                no_improve = 0
                t0 = time.time()
                
                for ep in range(25):
                    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
                    val_res = evaluate(model, val_loader, criterion, device, n_cls)
                    scheduler.step()
                    
                    if val_res['loss'] < best_val_loss:
                        best_val_loss = val_res['loss']
                        best_state = copy.deepcopy(model.state_dict())
                        no_improve = 0
                    else:
                        no_improve += 1
                    
                    if no_improve >= 12:
                        break
                    if (ep+1) % 5 == 0:
                        print(f"  Epoch {ep+1}: train={train_loss:.4f} val_mf1={val_res['macro_f1']:.4f}")
                
                elapsed = time.time() - t0
                if best_state is not None:
                    model.load_state_dict(best_state)
                torch.save({'model_state_dict': model.state_dict(), 'tag': tag, 'time': elapsed},
                           ckpt_path)
                print(f"  Trained {tag} in {elapsed:.1f}s")
            
            # Evaluate
            criterion_eval = HCRMNLoss(num_classes=n_cls, feat_dim=128, focal_gamma=1.0)
            criterion_eval = criterion_eval.to(device)
            test_res = evaluate(model, test_loader, criterion_eval, device, n_cls)
            test_res['params'] = sum(p.numel() for p in model.parameters())
            ds_results['configs'][cfg_name] = test_res
            print(f"  {cfg_name}: MF1={test_res['macro_f1']:.4f}, Acc={test_res['accuracy']:.4f}")
        
        all_results[ds] = ds_results
        
        # Save
        out_path = os.path.join(ROOT, 'results', 'hcrmn_ablation', f'{ds}.json')
        with open(out_path, 'w') as f:
            json.dump(ds_results, f, indent=2)
        print(f"  Saved to {out_path}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("ABLATION SUMMARY")
    print(f"{'='*80}")
    for ds in datasets:
        print(f"\n{ds}:")
        print(f"  {'Variant':<20} {'Acc':>6} {'MF1':>6} {'Params':>10}")
        print(f"  {'-'*45}")
        for cfg_name in all_results[ds]['configs']:
            r = all_results[ds]['configs'][cfg_name]
            print(f"  {cfg_name:<20} {r['accuracy']:>6.4f} {r['macro_f1']:>6.4f} {r['params']:>10,}")
    
    # Save full summary
    out = os.path.join(ROOT, 'results', 'hcrmn_ablation', 'ABLATION_SUMMARY.json')
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved summary to {out}")


if __name__ == '__main__':
    main()
