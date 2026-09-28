"""
Ablate HCRMN one variant at a time on ECG5000_UNBAL.
Usage: python ablate_one.py <variant> <dataset>
Variants: CE, NoFiLM, NoCrossAttn, NoGraph, NoDynamics
"""
import os, sys, json, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hcrmn import HCRMN, HCRMNLoss, compute_transport_channels

VARIANT = sys.argv[1] if len(sys.argv) > 1 else 'CE'
DATASET = sys.argv[2] if len(sys.argv) > 2 else 'ECG5000_UNBAL'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- Load data ----
data_map = {
    'ECG5000_UNBAL': ('data/ecg5000_resplit.npz', 5),
    'ECG5000_BAL': ('data/ecg5000_fair_balanced.npz', 5),
    'BEARING_UNBAL': ('data/bearing_unbalanced.npz', 4),
    'BEARING_BAL': ('data/bearing_balanced.npz', 4),
}
fpath, n_cls = data_map[DATASET]
d = np.load(fpath)
X_all, y_all = d['X_train'], d['y_train'].astype(int)
X_test, y_test = d['X_test'], d['y_test'].astype(int)
support = {c: int((y_test == c).sum()) for c in range(n_cls)}

X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.15, stratify=y_all, random_state=42)

Xtr_tr = compute_transport_channels(X_train)
Xva_tr = compute_transport_channels(X_val)
Xte_tr = compute_transport_channels(X_test)

Xtrn = np.concatenate([X_train[:, None, :], Xtr_tr], axis=1).astype(np.float32)
Xval = np.concatenate([X_val[:, None, :], Xva_tr], axis=1).astype(np.float32)
Xtst = np.concatenate([X_test[:, None, :], Xte_tr], axis=1).astype(np.float32)

mu = Xtrn.mean(axis=(0, 2), keepdims=True)
sig = Xtrn.std(axis=(0, 2), keepdims=True) + 1e-8
Xtrn = (Xtrn - mu) / sig; Xval = (Xval - mu) / sig; Xtst = (Xtst - mu) / sig

train_loader = DataLoader(TensorDataset(torch.from_numpy(Xtrn), torch.from_numpy(y_train).long()), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.from_numpy(Xval), torch.from_numpy(y_val).long()), batch_size=256)
test_loader = DataLoader(TensorDataset(torch.from_numpy(Xtst), torch.from_numpy(y_test).long()), batch_size=256)

# ---- Build model ----
tag = f"{DATASET}_{VARIANT}"
ckpt = f"checkpoints/HCRMN_ABL_{tag}.pt"
os.makedirs('checkpoints', exist_ok=True)

model = HCRMN(in_channels=4, num_classes=n_cls).to(device)
nparams = sum(p.numel() for p in model.parameters())

if VARIANT == 'NoFiLM':
    from experiments.ablate_hcrmn import HCRMN_NoFiLM
    model = HCRMN_NoFiLM(model).to(device)
elif VARIANT == 'NoCrossAttn':
    from experiments.ablate_hcrmn import HCRMN_NoCrossAttn
    model = HCRMN_NoCrossAttn(model).to(device)
elif VARIANT == 'NoGraph':
    from experiments.ablate_hcrmn import HCRMN_NoGraph
    model = HCRMN_NoGraph(model).to(device)

# ---- Train ----
extra_kw = {}
if VARIANT == 'NoDynamics':
    extra_kw['lambda_dyn'] = 0.0

if os.path.exists(ckpt):
    model.load_state_dict(torch.load(ckpt, map_location=device)['model_state_dict'])
    print(f"[{tag}] Loaded from checkpoint")
else:
    criterion = HCRMNLoss(num_classes=n_cls, feat_dim=128, focal_gamma=0.0, **extra_kw).to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(criterion.parameters()), lr=3e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, steps_per_epoch=len(train_loader), epochs=12, pct_start=0.3)
    
    best_vl = float('inf'); best_st = None; ni = 0; t0 = time.time()
    for ep in range(12):
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
        if ni >= 5:
            print(f"[{tag}] Early stop ep {ep+1}")
            break
        print(f"[{tag}] Ep {ep+1}: loss={tl:.4f} val_mf1={vmf1:.4f}")
    
    elapsed = time.time() - t0
    if best_st:
        model.load_state_dict(best_st)
    torch.save({'model_state_dict': model.state_dict(), 'time': elapsed}, ckpt)
    print(f"[{tag}] Trained in {elapsed:.1f}s")

# ---- Evaluate ----
criterion = HCRMNLoss(num_classes=n_cls, feat_dim=128, focal_gamma=0.0).to(device)
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

result = {
    'variant': VARIANT, 'dataset': DATASET,
    'accuracy': round(float(acc), 4),
    'macro_f1': round(float(mf1), 4),
    'class_f1s': [round(float(f), 4) for f in f1s],
    'class_recalls': [round(float(r), 4) for r in recalls],
    'support': {str(k): v for k, v in support.items()},
    'params': nparams,
}

print(f"\n{'='*50}")
print(f"RESULT: {tag}")
print(f"{'='*50}")
print(f"  Accuracy:    {acc:.4f}")
print(f"  Macro F1:    {mf1:.4f}")
print(f"  Params:      {nparams:,}")
for c in range(n_cls):
    print(f"  C{c}: F1={f1s[c]:.4f}  Recall={recalls[c]:.4f}  (n={support[c]})")

os.makedirs('results/hcrmn_ablation', exist_ok=True)
out = f"results/hcrmn_ablation/{tag}.json"
with open(out, 'w') as f:
    json.dump(result, f, indent=2)
print(f"\nSaved to {out}")
