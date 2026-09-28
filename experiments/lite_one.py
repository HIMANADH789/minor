"""
Train + evaluate a single model on a single dataset.
Usage: python lite_one.py <DATASET> <VARIANT>
Variants: IT, IT_FOC, ETAI, ETAI_FOC, HLITE, HLITE_FOC
"""
import os, sys, json, time, copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, accuracy_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATASET = sys.argv[1] if len(sys.argv) > 1 else 'ECG5000_UNBAL'
VARIANT = sys.argv[2] if len(sys.argv) > 2 else 'HLITE'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_map = {
    'ECG5000_UNBAL': ('data/ecg5000_resplit.npz', 5),
    'ECG5000_BAL': ('data/ecg5000_fair_balanced.npz', 5),
    'BEARING_UNBAL': ('data/bearing_unbalanced.npz', 4),
    'BEARING_BAL': ('data/bearing_balanced.npz', 4),
}
fpath, n_cls = data_map[DATASET]
d = np.load(os.path.join(ROOT, fpath))
X_all, y_all = d['X_train'], d['y_train'].astype(int)
X_test, y_test = d['X_test'], d['y_test'].astype(int)
support = {c: int((y_test == c).sum()) for c in range(n_cls)}

X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.15, stratify=y_all, random_state=42)

# Prepare loaders based on variant
if VARIANT.startswith('IT'):
    Xtr = X_train[:, None, :].astype(np.float32)
    Xva = X_val[:, None, :].astype(np.float32)
    Xte = X_test[:, None, :].astype(np.float32)
elif VARIANT.startswith('ETAI'):
    from models.hcrmn_lite import compute_transport_channels_lite
    Xtr_raw = X_train[:, None, :].astype(np.float32)
    Xva_raw = X_val[:, None, :].astype(np.float32)
    Xte_raw = X_test[:, None, :].astype(np.float32)
    Xtr_tr = compute_transport_channels_lite(X_train)
    Xva_tr = compute_transport_channels_lite(X_val)
    Xte_tr = compute_transport_channels_lite(X_test)
    Xtr = Xtr_tr.astype(np.float32)  # 3ch: Q, D, M
    Xva = Xva_tr.astype(np.float32)
    Xte = Xte_tr.astype(np.float32)
else:  # HLITE
    from models.hcrmn_lite import compute_transport_channels_lite
    Xtr_raw = X_train[:, None, :].astype(np.float32)
    Xva_raw = X_val[:, None, :].astype(np.float32)
    Xte_raw = X_test[:, None, :].astype(np.float32)
    Xtr_tr = compute_transport_channels_lite(X_train)
    Xva_tr = compute_transport_channels_lite(X_val)
    Xte_tr = compute_transport_channels_lite(X_test)
    Xtr = np.concatenate([Xtr_raw, Xtr_tr], axis=1)
    Xva = np.concatenate([Xva_raw, Xva_tr], axis=1)
    Xte = np.concatenate([Xte_raw, Xte_tr], axis=1)

# Z-normalize
mu = Xtr.mean(axis=(0, 2), keepdims=True)
sig = Xtr.std(axis=(0, 2), keepdims=True) + 1e-8
Xtr = (Xtr - mu) / sig; Xva = (Xva - mu) / sig; Xte = (Xte - mu) / sig

train_l = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(y_train).long()), batch_size=64, shuffle=True)
val_l = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(y_val).long()), batch_size=256)
test_l = DataLoader(TensorDataset(torch.from_numpy(Xte), torch.from_numpy(y_test).long()), batch_size=256)

ic = Xtr.shape[1]
tag = f"{DATASET}_{VARIANT}"
ckpt = os.path.join(ROOT, f'checkpoints/{tag}.pt')
os.makedirs(os.path.dirname(ckpt), exist_ok=True)

# Build model
focal_gamma = 0.0
if VARIANT.endswith('_FOC'):
    focal_gamma = 1.0

if VARIANT.startswith('IT'):
    from experiments.bearing_fair_full import ITNet
    model = ITNet(ic=ic, nc=32, nb=6, ks=[5, 10, 20]).to(device)
elif VARIANT.startswith('ETAI'):
    from experiments.bearing_fair_full import ITNet
    model = ITNet(ic=ic, nc=32, nb=6, ks=[5, 10, 20]).to(device)
else:
    from models.hcrmn_lite import HCRMNLite, HCRMNLiteLoss
    model = HCRMNLite(in_channels=4, num_classes=n_cls, feat_dim=64, regime_dim=16, K=4, n_blocks=4).to(device)

nparams = sum(p.numel() for p in model.parameters())

# Load or train
if os.path.exists(ckpt):
    model.load_state_dict(torch.load(ckpt, map_location=device)['model_state_dict'])
    print(f"[{tag}] Loaded checkpoint ({nparams:,} params)")
else:
    if VARIANT.startswith('HLITE'):
        from models.hcrmn_lite import HCRMNLiteLoss
        criterion = HCRMNLiteLoss(num_classes=n_cls, feat_dim=64, regime_dim=16,
                                  focal_gamma=focal_gamma,
                                  lambda_dyn=0.01, lambda_hier=0.01,
                                  lambda_proto=0.01, lambda_smooth=0.01).to(device)
        params = list(model.parameters()) + list(criterion.parameters())
    else:
        if focal_gamma > 0:
            def criterion(logits, targets):
                ce = nn.functional.cross_entropy(logits, targets, reduction='none')
                pt = torch.exp(-ce)
                return ((1 - pt) ** focal_gamma * ce).mean()
        else:
            criterion = nn.CrossEntropyLoss()
        params = model.parameters()

    opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4,
                                                steps_per_epoch=len(train_l),
                                                epochs=25, pct_start=0.3)

    best_vl = -1; best_st = None; ni = 0; t0 = time.time()
    for ep in range(25):
        model.train()
        for xb, yb in train_l:
            xb, yb = xb.to(device), yb.to(device)
            if VARIANT.startswith('HLITE'):
                logits, info = model(xb)
                loss, _ = criterion(logits, info, yb)
            else:
                logits = model(xb)
                loss = criterion(logits, yb)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()

        # Eval
        model.eval(); vp = []; vt = []
        with torch.no_grad():
            for xb, yb in val_l:
                if VARIANT.startswith('HLITE'):
                    logits, _ = model(xb.to(device))
                else:
                    logits = model(xb.to(device))
                vp.append(logits.argmax(-1).cpu().numpy()); vt.append(yb.numpy())
        vmf1 = f1_score(np.concatenate(vt), np.concatenate(vp), average='macro', zero_division=0)
        if vmf1 > best_vl:
            best_vl = vmf1; best_st = copy.deepcopy(model.state_dict()); ni = 0
        else:
            ni += 1

        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  [{tag}] Ep {ep+1}: val_mf1={vmf1:.4f} ({time.time()-t0:.0f}s)")

        if ni >= 10:
            print(f"  [{tag}] Early stop ep {ep+1}")
            break

    elapsed = time.time() - t0
    if best_st:
        model.load_state_dict(best_st)
    torch.save({'model_state_dict': model.state_dict(), 'time': elapsed}, ckpt)
    print(f"  [{tag}] Trained in {elapsed:.1f}s ({nparams:,} params)")

# Final test eval
model.eval(); preds = []; true = []
with torch.no_grad():
    for xb, yb in test_l:
        if VARIANT.startswith('HLITE'):
            logits, _ = model(xb.to(device))
        else:
            logits = model(xb.to(device))
        preds.append(logits.argmax(-1).cpu().numpy()); true.append(yb.numpy())
preds = np.concatenate(preds); true = np.concatenate(true)
acc = accuracy_score(true, preds)
mf1 = f1_score(true, preds, average='macro', zero_division=0)
f1s = f1_score(true, preds, average=None, zero_division=0, labels=list(range(n_cls)))
recalls = recall_score(true, preds, average=None, zero_division=0, labels=list(range(n_cls)))
cm = np.zeros((n_cls, n_cls), dtype=int)
for t, p in zip(true, preds):
    cm[t][p] += 1

result = {
    'accuracy': round(float(acc), 4), 'macro_f1': round(float(mf1), 4),
    'class_f1s': [round(float(f), 4) for f in f1s],
    'class_recalls': [round(float(r), 4) for r in recalls],
    'confusion_matrix': cm.tolist(),
    'support': support, 'params': nparams,
}

print(f"\n{'='*50}")
print(f"RESULT: {tag}")
print(f"{'='*50}")
print(f"  Accuracy:    {acc:.4f}")
print(f"  Macro F1:    {mf1:.4f}")
print(f"  Params:      {nparams:,}")
for c in range(n_cls):
    print(f"  C{c} (n={support[c]:>4}): F1={f1s[c]:.4f}  Recall={recalls[c]:.4f}")

os.makedirs(os.path.join(ROOT, 'results', 'hcrmn_lite'), exist_ok=True)
out = os.path.join(ROOT, f'results/hcrmn_lite/{tag}.json')
with open(out, 'w') as f:
    json.dump(result, f, indent=2)
print(f"Saved to {out}")
