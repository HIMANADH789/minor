"""
Train all 6 model variants on one dataset.
Usage: python lite_all.py <DATASET>
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

from models.hcrmn_lite import compute_transport_channels_lite
Xtr_raw = X_train[:, None, :].astype(np.float32)
Xva_raw = X_val[:, None, :].astype(np.float32)
Xte_raw = X_test[:, None, :].astype(np.float32)
Xtr_tr = compute_transport_channels_lite(X_train)
Xva_tr = compute_transport_channels_lite(X_val)
Xte_tr = compute_transport_channels_lite(X_test)

# 1ch loaders (for IT)
mu1 = Xtr_raw.mean(axis=(0,2), keepdims=True); sig1 = Xtr_raw.std(axis=(0,2), keepdims=True)+1e-8
Xtr_1 = (Xtr_raw-mu1)/sig1; Xva_1 = (Xva_raw-mu1)/sig1; Xte_1 = (Xte_raw-mu1)/sig1
dl_tr_1 = DataLoader(TensorDataset(torch.from_numpy(Xtr_1), torch.from_numpy(y_train).long()), 64, shuffle=True)
dl_va_1 = DataLoader(TensorDataset(torch.from_numpy(Xva_1), torch.from_numpy(y_val).long()), 256)
dl_te_1 = DataLoader(TensorDataset(torch.from_numpy(Xte_1), torch.from_numpy(y_test).long()), 256)

# 3ch loaders (for eTAI)
mu3 = Xtr_tr.mean(axis=(0,2), keepdims=True); sig3 = Xtr_tr.std(axis=(0,2), keepdims=True)+1e-8
Xtr_3 = ((Xtr_tr-mu3)/sig3).astype(np.float32)
Xva_3 = ((Xva_tr-mu3)/sig3).astype(np.float32)
Xte_3 = ((Xte_tr-mu3)/sig3).astype(np.float32)
dl_tr_3 = DataLoader(TensorDataset(torch.from_numpy(Xtr_3), torch.from_numpy(y_train).long()), 64, shuffle=True)
dl_va_3 = DataLoader(TensorDataset(torch.from_numpy(Xva_3), torch.from_numpy(y_val).long()), 256)
dl_te_3 = DataLoader(TensorDataset(torch.from_numpy(Xte_3), torch.from_numpy(y_test).long()), 256)

# 4ch loaders (for HCRMN-Lite)
Xtr_4 = np.concatenate([Xtr_1, (Xtr_tr-mu3)/sig3], axis=1).astype(np.float32)
Xva_4 = np.concatenate([Xva_1, (Xva_tr-mu3)/sig3], axis=1).astype(np.float32)
Xte_4 = np.concatenate([Xte_1, (Xte_tr-mu3)/sig3], axis=1).astype(np.float32)
dl_tr_4 = DataLoader(TensorDataset(torch.from_numpy(Xtr_4), torch.from_numpy(y_train).long()), 64, shuffle=True)
dl_va_4 = DataLoader(TensorDataset(torch.from_numpy(Xva_4), torch.from_numpy(y_val).long()), 256)
dl_te_4 = DataLoader(TensorDataset(torch.from_numpy(Xte_4), torch.from_numpy(y_test).long()), 256)

print(f"Dataset: {DATASET}, Support: {support}, Train: {len(X_train)}, Test: {len(X_test)}")

def eval_model(model, loader, is_lite=False):
    model.eval(); preds = []; true = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            if is_lite:
                logits, _ = model(xb)
            else:
                logits = model(xb)
            preds.append(logits.argmax(-1).cpu().numpy()); true.append(yb.numpy())
    preds = np.concatenate(preds); true = np.concatenate(true)
    return accuracy_score(true, preds), f1_score(true, preds, average='macro', zero_division=0)

def train_it(tag, ic, tr_l, va_l, te_l, focal=0.0, nb=6, nc=32):
    """Train InceptionTime variant."""
    from experiments.bearing_fair_full import ITNet
    ckpt = os.path.join(ROOT, f'checkpoints/{tag}.pt')
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    model = ITNet(ic=ic, nc=nc, nb=nb, ks=[5,10,20]).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    
    if os.path.exists(ckpt):
        try:
            ckpt_data = torch.load(ckpt, map_location=device)
            if isinstance(ckpt_data, dict) and 'model_state_dict' in ckpt_data:
                sd = ckpt_data['model_state_dict']
            else:
                sd = ckpt_data
            model.load_state_dict(sd)
            print(f"  [{tag}] Loaded ({nparams:,} params)")
            acc, mf1 = eval_model(model, te_l)
            return model, nparams, acc, mf1
        except Exception as e:
            print(f"  [{tag}] Checkpoint incompatible: {e}, retraining")

    if focal > 0:
        def criterion(l, t):
            ce = nn.functional.cross_entropy(l, t, reduction='none')
            return ((1 - torch.exp(-ce))**focal * ce).mean()
    else:
        criterion = nn.CrossEntropyLoss()

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, steps_per_epoch=len(tr_l), epochs=15, pct_start=0.3)
    best_vl = -1; best_st = None; ni = 0; t0 = time.time()
    for ep in range(15):
        model.train()
        for xb, yb in tr_l:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        vmf1 = eval_model(model, va_l)[1]
        if vmf1 > best_vl:
            best_vl = vmf1; best_st = copy.deepcopy(model.state_dict()); ni = 0
        else:
            ni += 1
        if (ep+1)%5==0: print(f"  [{tag}] Ep {ep+1}: val_mf1={vmf1:.4f} ({time.time()-t0:.0f}s)")
        if ni >= 8: break
    if best_st: model.load_state_dict(best_st)
    torch.save({'model_state_dict': model.state_dict()}, ckpt)
    elapsed = time.time()-t0
    acc, mf1 = eval_model(model, te_l)
    print(f"  [{tag}] Done in {elapsed:.0f}s, MF1={mf1:.4f}")
    return model, nparams, acc, mf1

def train_lite(tag, tr_l, va_l, te_l, focal=0.0):
    """Train HCRMN-Lite variant."""
    from models.hcrmn_lite import HCRMNLite, HCRMNLiteLoss
    ckpt = os.path.join(ROOT, f'checkpoints/{tag}.pt')
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    model = HCRMNLite(in_channels=4, num_classes=n_cls, feat_dim=64, regime_dim=16, K=4, n_blocks=4).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device)['model_state_dict'])
        print(f"  [{tag}] Loaded ({nparams:,} params)")
        acc, mf1 = eval_model(model, te_l, is_lite=True)
        return model, nparams, acc, mf1

    criterion = HCRMNLiteLoss(num_classes=n_cls, feat_dim=64, regime_dim=16,
                              focal_gamma=focal, lambda_dyn=0.01, lambda_hier=0.01,
                              lambda_proto=0.01, lambda_smooth=0.01).to(device)
    opt = torch.optim.AdamW(list(model.parameters())+list(criterion.parameters()), lr=3e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, steps_per_epoch=len(tr_l), epochs=15, pct_start=0.3)
    best_vl = -1; best_st = None; ni = 0; t0 = time.time()
    for ep in range(15):
        model.train()
        for xb, yb in tr_l:
            xb, yb = xb.to(device), yb.to(device)
            logits, info = model(xb)
            loss, _ = criterion(logits, info, yb)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        vmf1 = eval_model(model, va_l, is_lite=True)[1]
        if vmf1 > best_vl:
            best_vl = vmf1; best_st = copy.deepcopy(model.state_dict()); ni = 0
        else:
            ni += 1
        if (ep+1)%5==0: print(f"  [{tag}] Ep {ep+1}: val_mf1={vmf1:.4f} ({time.time()-t0:.0f}s)")
        if ni >= 8: break
    if best_st: model.load_state_dict(best_st)
    torch.save({'model_state_dict': model.state_dict()}, ckpt)
    elapsed = time.time()-t0
    acc, mf1 = eval_model(model, te_l, is_lite=True)
    print(f"  [{tag}] Done in {elapsed:.0f}s, MF1={mf1:.4f}")
    return model, nparams, acc, mf1

# ---- Train all 6 models ----
results = {}
configs = [
    (f'{DATASET}_IT_CE', 'IT', False, 0.0),
    (f'{DATASET}_IT_FOC1', 'IT', False, 1.0),
    (f'{DATASET}_eTAI_CE', 'ETAI', False, 0.0),
    (f'{DATASET}_eTAI_FOC1', 'ETAI', False, 1.0),
    (f'{DATASET}_HLITE_CE', 'HLITE', True, 0.0),
    (f'{DATASET}_HLITE_FOC1', 'HLITE', True, 1.0),
]

for tag, variant, is_lite, focal in configs:
    print(f"\n--- {tag} ---")
    if variant == 'IT':
        _, np_, acc, mf1 = train_it(tag, 1, dl_tr_1, dl_va_1, dl_te_1, focal=focal)
    elif variant == 'ETAI':
        _, np_, acc, mf1 = train_it(tag, 3, dl_tr_3, dl_va_3, dl_te_3, focal=focal)
    else:
        _, np_, acc, mf1 = train_lite(tag, dl_tr_4, dl_va_4, dl_te_4, focal=focal)
    results[tag] = {'accuracy': round(acc, 4), 'macro_f1': round(mf1, 4), 'params': np_}

# ---- Per-class for best model ----
best_name = max(results, key=lambda k: results[k]['macro_f1'])
print(f"\n{'='*60}")
print(f"  RESULTS: {DATASET}")
print(f"{'='*60}")
print(f"{'Model':<30} {'Acc':>6} {'MF1':>6} {'Params':>10}")
print(f"{'-'*55}")
for name, r in sorted(results.items(), key=lambda x: -x[1]['macro_f1']):
    print(f"{name:<30} {r['accuracy']:>6.4f} {r['macro_f1']:>6.4f} {r['params']:>10,}")

os.makedirs(os.path.join(ROOT, 'results', 'hcrmn_lite'), exist_ok=True)
with open(os.path.join(ROOT, f'results/hcrmn_lite/{DATASET}.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to results/hcrmn_lite/{DATASET}.json")
