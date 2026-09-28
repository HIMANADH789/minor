"""
Ultra-fast HCRMN ablation: subsampled data, fewer epochs, periodic checkpoint saves.
Runs ONE variant per invocation.
Usage: python ablate_fast.py <variant>
"""
import os, sys, json, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, accuracy_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

VARIANT = sys.argv[1] if len(sys.argv) > 1 else 'CE'
DATASET = 'ECG5000_UNBAL'

from models.hcrmn import HCRMN, HCRMNLoss, compute_transport_channels

device = torch.device('cpu')  # Force CPU — GPU driver is broken

# ---- Load + subsample ----
d = np.load(os.path.join(ROOT, 'data/ecg5000_resplit.npz'))
X_all, y_all = d['X_train'], d['y_train'].astype(int)
X_test, y_test = d['X_test'], d['y_test'].astype(int)
n_cls = 5
support = {c: int((y_test == c).sum()) for c in range(n_cls)}

# Subsample to 2000 train (stratified)
from sklearn.model_selection import train_test_split
X_sub, _, y_sub, _ = train_test_split(X_all, y_all, train_size=2000, stratify=y_all, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_sub, y_sub, test_size=0.15, stratify=y_sub, random_state=42)

# Transport channels
Xtr_tr = compute_transport_channels(X_train)
Xva_tr = compute_transport_channels(X_val)
Xte_tr = compute_transport_channels(X_test)

Xtrn = np.concatenate([X_train[:, None, :], Xtr_tr], axis=1).astype(np.float32)
Xval = np.concatenate([X_val[:, None, :], Xva_tr], axis=1).astype(np.float32)
Xtst = np.concatenate([X_test[:, None, :], Xte_tr], axis=1).astype(np.float32)

mu = Xtrn.mean(axis=(0, 2), keepdims=True)
sig = Xtrn.std(axis=(0, 2), keepdims=True) + 1e-8
Xtrn = (Xtrn - mu) / sig; Xval = (Xval - mu) / sig; Xtst = (Xtst - mu) / sig

train_loader = DataLoader(TensorDataset(torch.from_numpy(Xtrn), torch.from_numpy(y_train).long()),
                          batch_size=128, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.from_numpy(Xval), torch.from_numpy(y_val).long()), batch_size=256)
test_loader = DataLoader(TensorDataset(torch.from_numpy(Xtst), torch.from_numpy(y_test).long()), batch_size=256)

print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
print(f"Support: {support}")

# ---- Build model ----
tag = f"{DATASET}_{VARIANT}"
os.makedirs(os.path.join(ROOT, 'checkpoints'), exist_ok=True)
ckpt = os.path.join(ROOT, f'checkpoints/HCRMN_ABL_{tag}.pt')

model = HCRMN(in_channels=4, num_classes=n_cls, n_blocks=4, feat_dim=96, regime_dim=24, n_experts=6).to(device)
nparams = sum(p.numel() for p in model.parameters())
print(f"Params: {nparams:,}")

if VARIANT == 'NoFiLM':
    sys.path.insert(0, os.path.join(ROOT, 'experiments'))
    from ablate_hcrmn import HCRMN_NoFiLM
    model = HCRMN_NoFiLM(model).to(device)
elif VARIANT == 'NoCrossAttn':
    sys.path.insert(0, os.path.join(ROOT, 'experiments'))
    from ablate_hcrmn import HCRMN_NoCrossAttn
    model = HCRMN_NoCrossAttn(model).to(device)
elif VARIANT == 'NoGraph':
    sys.path.insert(0, os.path.join(ROOT, 'experiments'))
    from ablate_hcrmn import HCRMN_NoGraph
    model = HCRMN_NoGraph(model).to(device)

# ---- Train with periodic saves ----
extra_kw = {}
if VARIANT == 'NoDynamics':
    extra_kw['lambda_dyn'] = 0.0

if os.path.exists(ckpt):
    ckpt_data = torch.load(ckpt, map_location=device)
    model.load_state_dict(ckpt_data['model_state_dict'])
    print(f"Loaded from checkpoint (epoch {ckpt_data.get('epoch', '?')})")
else:
    feat_dim = model.feat_dim if hasattr(model, 'feat_dim') else 96
    # Handle wrapped models
    if hasattr(model, 'base') and hasattr(model.base, 'feat_dim'):
        feat_dim = model.base.feat_dim
    
    criterion = HCRMNLoss(num_classes=n_cls, feat_dim=feat_dim, focal_gamma=0.0, **extra_kw).to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(criterion.parameters()), lr=3e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, steps_per_epoch=len(train_loader), epochs=15, pct_start=0.3)

    best_vl = float('inf'); best_st = None; ni = 0; t0 = time.time()
    for ep in range(15):
        model.train(); tl = 0; n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits, info = model(xb)
            loss, _ = criterion(logits, info, yb)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tl += loss.item() * xb.shape[0]; n += xb.shape[0]
        tl /= n

        model.eval(); vp = []; vt = []; vl = 0; vn = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits, info = model(xb)
                loss, _ = criterion(logits, info, yb)
                vp.append(logits.argmax(-1).cpu().numpy()); vt.append(yb.cpu().numpy())
                vl += loss.item() * xb.shape[0]; vn += xb.shape[0]
        vl /= vn
        vmf1 = f1_score(np.concatenate(vt), np.concatenate(vp), average='macro', zero_division=0)

        if vl < best_vl:
            best_vl = vl; best_st = copy.deepcopy(model.state_dict()); ni = 0
        else:
            ni += 1
        
        ep_time = time.time() - t0
        print(f"[{tag}] Ep {ep+1}: loss={tl:.4f} val_mf1={vmf1:.4f} ({ep_time:.1f}s)")
        
        # Save intermediate every 3 epochs
        if (ep + 1) % 3 == 0 and best_st:
            torch.save({'model_state_dict': best_st, 'epoch': ep+1, 'time': time.time()-t0}, ckpt)
        
        if ni >= 5:
            print(f"[{tag}] Early stop ep {ep+1}")
            break

    elapsed = time.time() - t0
    if best_st:
        model.load_state_dict(best_st)
    torch.save({'model_state_dict': model.state_dict(), 'epoch': ep+1, 'time': elapsed}, ckpt)
    print(f"[{tag}] Trained in {elapsed:.1f}s")

# ---- Evaluate ----
feat_dim = model.feat_dim if hasattr(model, 'feat_dim') else 96
if hasattr(model, 'base') and hasattr(model.base, 'feat_dim'):
    feat_dim = model.base.feat_dim
criterion = HCRMNLoss(num_classes=n_cls, feat_dim=feat_dim, focal_gamma=0.0).to(device)
model.eval(); preds = []; true = []
with torch.no_grad():
    for xb, yb in test_loader:
        logits, _ = model(xb.to(device))
        preds.append(logits.argmax(-1).cpu().numpy()); true.append(yb.numpy())
preds = np.concatenate(preds); true = np.concatenate(true)
acc = accuracy_score(true, preds)
mf1 = f1_score(true, preds, average='macro', zero_division=0)
recalls = recall_score(true, preds, average=None, zero_division=0, labels=list(range(n_cls)))
f1s = f1_score(true, preds, average=None, zero_division=0, labels=list(range(n_cls)))

print(f"\n{'='*50}")
print(f"RESULT: {tag}")
print(f"{'='*50}")
print(f"  Accuracy:    {acc:.4f}")
print(f"  Macro F1:    {mf1:.4f}")
print(f"  Params:      {nparams:,}")
for c in range(n_cls):
    print(f"  C{c}: F1={f1s[c]:.4f}  Recall={recalls[c]:.4f}  (n={support[c]})")

result = {
    'variant': VARIANT, 'dataset': DATASET,
    'accuracy': round(float(acc), 4), 'macro_f1': round(float(mf1), 4),
    'class_f1s': [round(float(f), 4) for f in f1s],
    'class_recalls': [round(float(r), 4) for r in recalls],
    'support': support, 'params': nparams,
}

os.makedirs(os.path.join(ROOT, 'results', 'hcrmn_ablation'), exist_ok=True)
with open(os.path.join(ROOT, f'results/hcrmn_ablation/{tag}.json'), 'w') as f:
    json.dump(result, f, indent=2)
print(f"Saved to results/hcrmn_ablation/{tag}.json")
