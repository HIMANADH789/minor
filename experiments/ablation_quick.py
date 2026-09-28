"""
Minimal ablation: 3 epochs, 500 train samples, 2 blocks, 48 dim.
Just enough to get a signal.
"""
import os, sys, json, time, copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, recall_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.hcrmn import HCRMN, HCRMNLoss, compute_transport_channels
from experiments.ablate_hcrmn import HCRMN_NoFiLM, HCRMN_NoCrossAttn, HCRMN_NoGraph

# Load data
d = np.load(os.path.join(ROOT, 'data/ecg5000_resplit.npz'))
X_all, y_all = d['X_train'], d['y_train'].astype(int)
X_test, y_test = d['X_test'], d['y_test'].astype(int)
n_cls = 5
support = {c: int((y_test == c).sum()) for c in range(n_cls)}

X_sub, _, y_sub, _ = train_test_split(X_all, y_all, train_size=500, stratify=y_all, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_sub, y_sub, test_size=0.2, stratify=y_sub, random_state=42)

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

print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

def make_model(variant, full=False):
    if full:
        m = HCRMN(in_channels=4, num_classes=n_cls, n_blocks=6, feat_dim=128, regime_dim=32, n_experts=8)
    else:
        m = HCRMN(in_channels=4, num_classes=n_cls, n_blocks=3, feat_dim=64, regime_dim=16, n_experts=4)
    if variant == 'NoFiLM': m = HCRMN_NoFiLM(m)
    elif variant == 'NoCrossAttn': m = HCRMN_NoCrossAttn(m)
    elif variant == 'NoGraph': m = HCRMN_NoGraph(m)
    return m

def get_feat_dim(model):
    if hasattr(model, 'feat_dim'): return model.feat_dim
    if hasattr(model, 'base') and hasattr(model.base, 'feat_dim'): return model.base.feat_dim
    return 64

results = {}
for variant in ['CE', 'NoFiLM', 'NoCrossAttn', 'NoGraph']:
    print(f"\n--- {variant} ---")
    model = make_model(variant)
    nparams = sum(p.numel() for p in model.parameters())
    
    extra_kw = {'lambda_dyn': 0.0} if variant == 'NoDynamics' else {}
    criterion = HCRMNLoss(num_classes=n_cls, feat_dim=get_feat_dim(model), focal_gamma=0.0, **extra_kw)
    opt = torch.optim.AdamW(list(model.parameters()) + list(criterion.parameters()), lr=3e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, steps_per_epoch=len(train_loader), epochs=3, pct_start=0.5)
    
    best_vl = float('inf'); best_st = None
    t0 = time.time()
    for ep in range(3):
        model.train()
        for xb, yb in train_loader:
            logits, info = model(xb)
            loss, _ = criterion(logits, info, yb)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        
        model.eval(); vp = []; vt = []
        with torch.no_grad():
            for xb, yb in val_loader:
                logits, _ = model(xb)
                vp.append(logits.argmax(-1).numpy()); vt.append(yb.numpy())
        vmf1 = f1_score(np.concatenate(vt), np.concatenate(vp), average='macro', zero_division=0)
        print(f"  Ep {ep+1}: val_mf1={vmf1:.4f} ({time.time()-t0:.0f}s)")
        if vmf1 > best_vl:
            best_vl = vmf1; best_st = copy.deepcopy(model.state_dict())
    
    if best_st: model.load_state_dict(best_st)
    
    model.eval(); preds = []; true = []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits, _ = model(xb)
            preds.append(logits.argmax(-1).numpy()); true.append(yb.numpy())
    preds = np.concatenate(preds); true = np.concatenate(true)
    acc = accuracy_score(true, preds)
    mf1 = f1_score(true, preds, average='macro', zero_division=0)
    f1s = f1_score(true, preds, average=None, zero_division=0, labels=list(range(n_cls)))
    recalls = recall_score(true, preds, average=None, zero_division=0, labels=list(range(n_cls)))
    results[variant] = {
        'accuracy': round(float(acc), 4), 'macro_f1': round(float(mf1), 4),
        'class_f1s': [round(float(f), 4) for f in f1s],
        'class_recalls': [round(float(r), 4) for r in recalls],
        'support': support, 'params': nparams,
    }
    print(f"  RESULT: Acc={acc:.4f} MF1={mf1:.4f} Params={nparams:,}")
    for c in range(n_cls):
        print(f"    C{c}: F1={f1s[c]:.4f} Recall={recalls[c]:.4f} (n={support[c]})")

# NoDynamics
print(f"\n--- NoDynamics ---")
model = make_model('NoDynamics')
nparams = sum(p.numel() for p in model.parameters())
criterion = HCRMNLoss(num_classes=n_cls, feat_dim=get_feat_dim(model), focal_gamma=0.0, lambda_dyn=0.0)
opt = torch.optim.AdamW(list(model.parameters()) + list(criterion.parameters()), lr=3e-4, weight_decay=1e-2)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, steps_per_epoch=len(train_loader), epochs=3, pct_start=0.5)
best_vl = float('inf'); best_st = None; t0 = time.time()
for ep in range(3):
    model.train()
    for xb, yb in train_loader:
        logits, info = model(xb)
        loss, _ = criterion(logits, info, yb)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
    model.eval(); vp = []; vt = []
    with torch.no_grad():
        for xb, yb in val_loader:
            logits, _ = model(xb)
            vp.append(logits.argmax(-1).numpy()); vt.append(yb.numpy())
    vmf1 = f1_score(np.concatenate(vt), np.concatenate(vp), average='macro', zero_division=0)
    print(f"  Ep {ep+1}: val_mf1={vmf1:.4f} ({time.time()-t0:.0f}s)")
    if vmf1 > best_vl: best_vl = vmf1; best_st = copy.deepcopy(model.state_dict())
if best_st: model.load_state_dict(best_st)
model.eval(); preds = []; true = []
with torch.no_grad():
    for xb, yb in test_loader:
        logits, _ = model(xb); preds.append(logits.argmax(-1).numpy()); true.append(yb.numpy())
preds = np.concatenate(preds); true = np.concatenate(true)
results['NoDynamics'] = {
    'accuracy': round(float(accuracy_score(true, preds)), 4),
    'macro_f1': round(float(f1_score(true, preds, average='macro', zero_division=0)), 4),
    'class_f1s': [round(float(f), 4) for f in f1_score(true, preds, average=None, zero_division=0, labels=list(range(n_cls)))],
    'class_recalls': [round(float(r), 4) for r in recall_score(true, preds, average=None, zero_division=0, labels=list(range(n_cls)))],
    'support': support, 'params': nparams,
}
print(f"  RESULT: MF1={results['NoDynamics']['macro_f1']:.4f}")

print(f"\n{'='*60}")
print("ABLATION SUMMARY — ECG5000 Unbalanced (compact model)")
print(f"{'='*60}")
for v, r in results.items():
    print(f"{v:<16} Acc={r['accuracy']:.4f}  MF1={r['macro_f1']:.4f}  Params={r['params']:,}")
    for c in range(n_cls):
        print(f"  C{c} (n={support[c]:>4}): F1={r['class_f1s'][c]:.4f}  Recall={r['class_recalls'][c]:.4f}")

with open(os.path.join(ROOT, 'results', 'hcrmn_ablation', 'ECG5000_UNBAL.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved.")
