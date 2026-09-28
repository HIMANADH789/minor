"""
Ultra-compact ablation: 8 epochs, patience 3, 2000 train samples, 4 blocks.
All 5 variants in one script, saving intermediate results.
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

from models.hcrmn import HCRMN, HCRMNLoss, compute_transport_channels
from experiments.ablate_hcrmn import HCRMN_NoFiLM, HCRMN_NoCrossAttn, HCRMN_NoGraph

device = torch.device('cpu')
print(f"Device: {device}")

# ---- Load data ----
d = np.load(os.path.join(ROOT, 'data/ecg5000_resplit.npz'))
X_all, y_all = d['X_train'], d['y_train'].astype(int)
X_test, y_test = d['X_test'], d['y_test'].astype(int)
n_cls = 5
support = {c: int((y_test == c).sum()) for c in range(n_cls)}

X_sub, _, y_sub, _ = train_test_split(X_all, y_all, train_size=2000, stratify=y_all, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_sub, y_sub, test_size=0.15, stratify=y_sub, random_state=42)

Xtr_tr = compute_transport_channels(X_train)
Xva_tr = compute_transport_channels(X_val)
Xte_tr = compute_transport_channels(X_test)

Xtrn = np.concatenate([X_train[:, None, :], Xtr_tr], axis=1).astype(np.float32)
Xval = np.concatenate([X_val[:, None, :], Xva_tr], axis=1).astype(np.float32)
Xtst = np.concatenate([X_test[:, None, :], Xte_tr], axis=1).astype(np.float32)
mu = Xtrn.mean(axis=(0, 2), keepdims=True)
sig = Xtrn.std(axis=(0, 2), keepdims=True) + 1e-8
Xtrn = (Xtrn - mu) / sig; Xval = (Xval - mu) / sig; Xtst = (Xtst - mu) / sig

train_loader = DataLoader(TensorDataset(torch.from_numpy(Xtrn), torch.from_numpy(y_train).long()), batch_size=128, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.from_numpy(Xval), torch.from_numpy(y_val).long()), batch_size=256)
test_loader = DataLoader(TensorDataset(torch.from_numpy(Xtst), torch.from_numpy(y_test).long()), batch_size=256)

print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
print(f"Support: {support}")


def make_model(variant):
    m = HCRMN(in_channels=4, num_classes=n_cls, n_blocks=4, feat_dim=96, regime_dim=24, n_experts=6)
    if variant == 'NoFiLM':
        m = HCRMN_NoFiLM(m)
    elif variant == 'NoCrossAttn':
        m = HCRMN_NoCrossAttn(m)
    elif variant == 'NoGraph':
        m = HCRMN_NoGraph(m)
    return m


def train_variant(variant, max_epochs=8, patience=3):
    tag = f"ECG5000_UNBAL_{variant}"
    ckpt = os.path.join(ROOT, f'checkpoints/HCRMN_ABL_{tag}.pt')
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)

    if os.path.exists(ckpt):
        ckpt_data = torch.load(ckpt, map_location='cpu')
        model = make_model(variant)
        model.load_state_dict(ckpt_data['model_state_dict'])
        print(f"[{variant}] Loaded from checkpoint")
        return model, ckpt_data.get('time', 0)

    model = make_model(variant)
    nparams = sum(p.numel() for p in model.parameters())
    
    # Find feat_dim for loss
    feat_dim = 96
    if hasattr(model, 'feat_dim'):
        feat_dim = model.feat_dim
    elif hasattr(model, 'base') and hasattr(model.base, 'feat_dim'):
        feat_dim = model.base.feat_dim
    
    extra_kw = {'lambda_dyn': 0.0} if variant == 'NoDynamics' else {}
    criterion = HCRMNLoss(num_classes=n_cls, feat_dim=feat_dim, focal_gamma=0.0, **extra_kw)
    opt = torch.optim.AdamW(list(model.parameters()) + list(criterion.parameters()), lr=3e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, steps_per_epoch=len(train_loader), epochs=max_epochs, pct_start=0.3)

    best_vl = float('inf'); best_st = None; ni = 0; t0 = time.time()
    for ep in range(max_epochs):
        model.train(); tl = 0; n = 0
        for xb, yb in train_loader:
            xb, yb = xb, yb
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
                logits, info = model(xb)
                loss, _ = criterion(logits, info, yb)
                vp.append(logits.argmax(-1).numpy()); vt.append(yb.numpy())
                vl += loss.item() * xb.shape[0]; vn += xb.shape[0]
        vl /= vn
        vmf1 = f1_score(np.concatenate(vt), np.concatenate(vp), average='macro', zero_division=0)

        if vl < best_vl:
            best_vl = vl; best_st = copy.deepcopy(model.state_dict()); ni = 0
        else:
            ni += 1
        
        elapsed = time.time() - t0
        print(f"  [{variant}] Ep {ep+1}: loss={tl:.4f} val_mf1={vmf1:.4f} ({elapsed:.0f}s)")
        
        if ni >= patience:
            print(f"  [{variant}] Early stop ep {ep+1}")
            break

    elapsed = time.time() - t0
    if best_st:
        model.load_state_dict(best_st)
    torch.save({'model_state_dict': model.state_dict(), 'epoch': ep+1, 'time': elapsed, 'nparams': nparams}, ckpt)
    print(f"  [{variant}] Trained in {elapsed:.1f}s ({nparams:,} params)")
    return model, elapsed


def evaluate(model, variant):
    feat_dim = 96
    if hasattr(model, 'feat_dim'):
        feat_dim = model.feat_dim
    elif hasattr(model, 'base') and hasattr(model.base, 'feat_dim'):
        feat_dim = model.base.feat_dim
    criterion = HCRMNLoss(num_classes=n_cls, feat_dim=feat_dim, focal_gamma=0.0)
    model.eval(); preds = []; true = []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits, _ = model(xb)
            preds.append(logits.argmax(-1).numpy()); true.append(yb.numpy())
    preds = np.concatenate(preds); true = np.concatenate(true)
    acc = accuracy_score(true, preds)
    mf1 = f1_score(true, preds, average='macro', zero_division=0)
    wf1 = f1_score(true, preds, average='weighted', zero_division=0)
    recalls = recall_score(true, preds, average=None, zero_division=0, labels=list(range(n_cls)))
    f1s = f1_score(true, preds, average=None, zero_division=0, labels=list(range(n_cls)))
    nparams = sum(p.numel() for p in model.parameters())
    return {
        'accuracy': round(float(acc), 4), 'macro_f1': round(float(mf1), 4),
        'weighted_f1': round(float(wf1), 4),
        'class_f1s': [round(float(f), 4) for f in f1s],
        'class_recalls': [round(float(r), 4) for r in recalls],
        'support': support, 'params': nparams,
    }


# ---- Run all variants ----
variants = ['CE', 'NoFiLM', 'NoCrossAttn', 'NoGraph', 'NoDynamics']
all_results = {}

for v in variants:
    print(f"\n{'='*50}")
    print(f"Training variant: {v}")
    print(f"{'='*50}")
    model, t = train_variant(v, max_epochs=8, patience=3)
    result = evaluate(model, v)
    result['train_time'] = round(t, 1)
    all_results[v] = result
    print(f"  RESULT: Acc={result['accuracy']:.4f} MF1={result['macro_f1']:.4f}")
    for c in range(n_cls):
        print(f"    C{c}: F1={result['class_f1s'][c]:.4f} Recall={result['class_recalls'][c]:.4f} (n={support[c]})")

# ---- Full-size HCRMN-CE for comparison (6 blocks, 128 dim) ----
print(f"\n{'='*50}")
print("Training full HCRMN-CE (6 blocks, 128 dim)")
print(f"{'='*50}")

full_model = HCRMN(in_channels=4, num_classes=n_cls, n_blocks=6, feat_dim=128, regime_dim=32, n_experts=8)
full_nparams = sum(p.numel() for p in full_model.parameters())
criterion = HCRMNLoss(num_classes=n_cls, feat_dim=128, focal_gamma=0.0)
opt = torch.optim.AdamW(list(full_model.parameters()) + list(criterion.parameters()), lr=3e-4, weight_decay=1e-2)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, steps_per_epoch=len(train_loader), epochs=8, pct_start=0.3)

best_vl = float('inf'); best_st = None; ni = 0; t0 = time.time()
for ep in range(8):
    full_model.train(); tl = 0; n = 0
    for xb, yb in train_loader:
        logits, info = full_model(xb)
        loss, _ = criterion(logits, info, yb)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(full_model.parameters(), 1.0)
        opt.step(); sched.step()
        tl += loss.item() * xb.shape[0]; n += xb.shape[0]
    tl /= n
    full_model.eval(); vp = []; vt = []; vl = 0; vn = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            logits, info = full_model(xb)
            loss, _ = criterion(logits, info, yb)
            vp.append(logits.argmax(-1).numpy()); vt.append(yb.numpy())
            vl += loss.item() * xb.shape[0]; vn += xb.shape[0]
    vl /= vn
    vmf1 = f1_score(np.concatenate(vt), np.concatenate(vp), average='macro', zero_division=0)
    if vl < best_vl:
        best_vl = vl; best_st = copy.deepcopy(full_model.state_dict()); ni = 0
    else:
        ni += 1
    elapsed = time.time() - t0
    print(f"  Full Ep {ep+1}: loss={tl:.4f} val_mf1={vmf1:.4f} ({elapsed:.0f}s)")
    if ni >= 3:
        break

elapsed_full = time.time() - t0
if best_st:
    full_model.load_state_dict(best_st)

criterion = HCRMNLoss(num_classes=n_cls, feat_dim=128, focal_gamma=0.0)
full_model.eval(); preds = []; true = []
with torch.no_grad():
    for xb, yb in test_loader:
        logits, _ = full_model(xb)
        preds.append(logits.argmax(-1).numpy()); true.append(yb.numpy())
preds = np.concatenate(preds); true = np.concatenate(true)
acc = accuracy_score(true, preds)
mf1 = f1_score(true, preds, average='macro', zero_division=0)
recalls = recall_score(true, preds, average=None, zero_division=0, labels=list(range(n_cls)))
f1s = f1_score(true, preds, average=None, zero_division=0, labels=list(range(n_cls)))
all_results['HCRMN-CE-full'] = {
    'accuracy': round(float(acc), 4), 'macro_f1': round(float(mf1), 4),
    'class_f1s': [round(float(f), 4) for f in f1s],
    'class_recalls': [round(float(r), 4) for r in recalls],
    'support': support, 'params': full_nparams, 'train_time': round(elapsed_full, 1),
}

# ---- Summary ----
print(f"\n{'='*70}")
print("ABLATION SUMMARY — ECG5000 Unbalanced")
print(f"{'='*70}")
print(f"{'Variant':<20} {'Acc':>6} {'MF1':>6} {'Params':>10} {'Time':>6}")
print(f"{'-'*50}")
for v, r in all_results.items():
    print(f"{v:<20} {r['accuracy']:>6.4f} {r['macro_f1']:>6.4f} {r['params']:>10,} {r.get('train_time', 0):>5.0f}s")
    for c in range(n_cls):
        print(f"  C{c} (n={support[c]:>4}): F1={r['class_f1s'][c]:.4f}  Recall={r['class_recalls'][c]:.4f}")

os.makedirs(os.path.join(ROOT, 'results', 'hcrmn_ablation'), exist_ok=True)
with open(os.path.join(ROOT, 'results', 'hcrmn_ablation', 'ECG5000_UNBAL.json'), 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to results/hcrmn_ablation/ECG5000_UNBAL.json")
