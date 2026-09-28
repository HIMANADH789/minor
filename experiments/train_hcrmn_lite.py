"""
Train HCRMN-Lite on all 4 datasets and compare against baselines.
Usage: python train_hcrmn_lite.py <DATASET>
Datasets: ECG5000_UNBAL, ECG5000_BAL, BEARING_UNBAL, BEARING_BAL
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

from models.hcrmn_lite import HCRMNLite, HCRMNLiteLoss, compute_transport_channels_lite
from experiments.bearing_fair_full import ITNet

DATASET = sys.argv[1] if len(sys.argv) > 1 else 'ECG5000_UNBAL'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}, Dataset: {DATASET}")

# ---- Load data ----
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
print(f"Support: {support}")

X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.15, stratify=y_all, random_state=42)

# Transport channels
Xtr_tr = compute_transport_channels_lite(X_train)
Xva_tr = compute_transport_channels_lite(X_val)
Xte_tr = compute_transport_channels_lite(X_test)

Xtrn = np.concatenate([X_train[:, None, :], Xtr_tr], axis=1).astype(np.float32)
Xval = np.concatenate([X_val[:, None, :], Xva_tr], axis=1).astype(np.float32)
Xtst = np.concatenate([X_test[:, None, :], Xte_tr], axis=1).astype(np.float32)

mu = Xtrn.mean(axis=(0, 2), keepdims=True)
sig = Xtrn.std(axis=(0, 2), keepdims=True) + 1e-8
Xtrn = (Xtrn - mu) / sig; Xval = (Xval - mu) / sig; Xtst = (Xtst - mu) / sig

print(f"Train: {len(Xtrn)}, Val: {len(Xval)}, Test: {len(Xtst)}")

# HCRMN-Lite loaders (4ch: raw+Q+D+M)
train_loader = DataLoader(TensorDataset(torch.from_numpy(Xtrn), torch.from_numpy(y_train).long()),
                          batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.from_numpy(Xval), torch.from_numpy(y_val).long()), batch_size=256)
test_loader = DataLoader(TensorDataset(torch.from_numpy(Xtst), torch.from_numpy(y_test).long()), batch_size=256)

# ITNet loaders (1ch: raw only)
Xtrn_1ch = X_train[:, None, :].astype(np.float32)
Xval_1ch = X_val[:, None, :].astype(np.float32)
Xtst_1ch = X_test[:, None, :].astype(np.float32)
mu1 = Xtrn_1ch.mean(axis=(0, 2), keepdims=True)
sig1 = Xtrn_1ch.std(axis=(0, 2), keepdims=True) + 1e-8
Xtrn_1ch = (Xtrn_1ch - mu1) / sig1; Xval_1ch = (Xval_1ch - mu1) / sig1; Xtst_1ch = (Xtst_1ch - mu1) / sig1

train_loader_1ch = DataLoader(TensorDataset(torch.from_numpy(Xtrn_1ch), torch.from_numpy(y_train).long()), batch_size=64, shuffle=True)
val_loader_1ch = DataLoader(TensorDataset(torch.from_numpy(Xval_1ch), torch.from_numpy(y_val).long()), batch_size=256)
test_loader_1ch = DataLoader(TensorDataset(torch.from_numpy(Xtst_1ch), torch.from_numpy(y_test).long()), batch_size=256)

# eTAI loaders (3ch: Q+D+M)
Xtrn_3ch = Xtrn[:, 1:4, :]
Xval_3ch = Xval[:, 1:4, :]
Xtst_3ch = Xtst[:, 1:4, :]

train_loader_3ch = DataLoader(TensorDataset(torch.from_numpy(Xtrn_3ch), torch.from_numpy(y_train).long()), batch_size=64, shuffle=True)
val_loader_3ch = DataLoader(TensorDataset(torch.from_numpy(Xval_3ch), torch.from_numpy(y_val).long()), batch_size=256)
test_loader_3ch = DataLoader(TensorDataset(torch.from_numpy(Xtst_3ch), torch.from_numpy(y_test).long()), batch_size=256)


# ---- Helper functions ----
def make_loader(X, y, bs=64, sh=False):
    return DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y).long()), batch_size=bs, shuffle=sh)


def evaluate(model, loader, n_cls, focal_gamma=0.0):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits, _ = model(xb.to(device))
            preds.append(logits.argmax(-1).cpu().numpy())
            targets.append(yb.numpy())
    preds = np.concatenate(preds); targets = np.concatenate(targets)
    acc = accuracy_score(targets, preds)
    mf1 = f1_score(targets, preds, average='macro', zero_division=0)
    wf1 = f1_score(targets, preds, average='weighted', zero_division=0)
    recalls = recall_score(targets, preds, average=None, zero_division=0, labels=list(range(n_cls)))
    f1s = f1_score(targets, preds, average=None, zero_division=0, labels=list(range(n_cls)))
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(targets, preds):
        cm[t][p] += 1
    return {
        'accuracy': round(float(acc), 4), 'macro_f1': round(float(mf1), 4),
        'weighted_f1': round(float(wf1), 4),
        'class_f1s': [round(float(f), 4) for f in f1s],
        'class_recalls': [round(float(r), 4) for r in recalls],
        'confusion_matrix': cm.tolist(),
    }


def train_hcrmn_lite(tag, max_epochs=25, patience=12):
    ckpt = os.path.join(ROOT, f'checkpoints/{tag}.pt')
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)

    model = HCRMNLite(in_channels=4, num_classes=n_cls, feat_dim=64, regime_dim=16, K=4, n_blocks=4).to(device)
    nparams = sum(p.numel() for p in model.parameters())

    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device)['model_state_dict'])
        print(f"  [{tag}] Loaded from checkpoint")
    else:
        criterion = HCRMNLiteLoss(num_classes=n_cls, feat_dim=64, regime_dim=16, focal_gamma=0.0,
                                  lambda_dyn=0.01, lambda_hier=0.01, lambda_proto=0.01, lambda_smooth=0.01).to(device)
        opt = torch.optim.AdamW(list(model.parameters()) + list(criterion.parameters()),
                                lr=3e-4, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4,
                                                    steps_per_epoch=len(train_loader),
                                                    epochs=max_epochs, pct_start=0.3)

        best_vl = float('inf'); best_st = None; ni = 0; t0 = time.time()
        for ep in range(max_epochs):
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

            val_res = evaluate(model, val_loader, n_cls)
            if val_res['accuracy'] > best_vl:
                # Use val macro_f1 as selection criterion (or accuracy for majority-heavy)
                pass
            # Use accuracy for early stopping (more stable for imbalanced)
            val_acc = val_res['accuracy']
            # Actually use negative validation loss equivalent — track macro_f1
            vmf1 = val_res['macro_f1']
            if vmf1 > best_vl or (vmf1 == best_vl and val_res['accuracy'] > 0):
                best_vl = vmf1
                best_st = copy.deepcopy(model.state_dict())
                ni = 0
            else:
                ni += 1

            elapsed = time.time() - t0
            if (ep + 1) % 5 == 0 or ep == 0:
                print(f"  [{tag}] Ep {ep+1}: loss={tl:.4f} val_mf1={vmf1:.4f} ({elapsed:.0f}s)")

            if ni >= patience:
                print(f"  [{tag}] Early stop ep {ep+1}")
                break

        elapsed = time.time() - t0
        if best_st:
            model.load_state_dict(best_st)
        torch.save({'model_state_dict': model.state_dict(), 'epoch': ep + 1, 'time': elapsed}, ckpt)
        print(f"  [{tag}] Trained in {elapsed:.1f}s ({nparams:,} params)")

    return model, nparams


def train_baseline(tag, model, train_l, val_l, test_l, max_epochs=25, patience=12, focal_gamma=0.0):
    """Train InceptionTime or eTAI baseline."""
    ckpt = os.path.join(ROOT, f'checkpoints/{tag}.pt')
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)

    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device)['model_state_dict'])
        print(f"  [{tag}] Loaded from checkpoint")
    else:
        if focal_gamma > 0:
            def criterion(logits, targets):
                ce = nn.functional.cross_entropy(logits, targets, reduction='none')
                pt = torch.exp(-ce)
                return ((1 - pt) ** focal_gamma * ce).mean()
        else:
            criterion = nn.CrossEntropyLoss()

        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4,
                                                    steps_per_epoch=len(train_l),
                                                    epochs=max_epochs, pct_start=0.3)

        best_vl = float('inf'); best_st = None; ni = 0; t0 = time.time()
        for ep in range(max_epochs):
            model.train(); tl = 0; n = 0
            for xb, yb in train_l:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step()
                tl += loss.item() * xb.shape[0]; n += xb.shape[0]
            tl /= n

            model.eval(); vp = []; vt = []
            with torch.no_grad():
                for xb, yb in val_l:
                    logits = model(xb.to(device))
                    vp.append(logits.argmax(-1).cpu().numpy()); vt.append(yb.numpy())
            vmf1 = f1_score(np.concatenate(vt), np.concatenate(vp), average='macro', zero_division=0)
            if vmf1 > best_vl:
                best_vl = vmf1; best_st = copy.deepcopy(model.state_dict()); ni = 0
            else:
                ni += 1

            if (ep + 1) % 5 == 0 or ep == 0:
                elapsed = time.time() - t0
                print(f"  [{tag}] Ep {ep+1}: loss={tl:.4f} val_mf1={vmf1:.4f} ({elapsed:.0f}s)")

            if ni >= patience:
                print(f"  [{tag}] Early stop ep {ep+1}")
                break
            sched.step()

        elapsed = time.time() - t0
        if best_st:
            model.load_state_dict(best_st)
        torch.save({'model_state_dict': model.state_dict(), 'epoch': ep + 1, 'time': elapsed}, ckpt)
        print(f"  [{tag}] Trained in {elapsed:.1f}s")

    # Evaluate on the right test loader
    model.eval(); preds = []; targets = []
    with torch.no_grad():
        for xb, yb in test_l:
            logits = model(xb.to(device))
            preds.append(logits.argmax(-1).cpu().numpy()); targets.append(yb.numpy())
    preds = np.concatenate(preds); targets = np.concatenate(targets)
    return model, accuracy_score(targets, preds), f1_score(targets, preds, average='macro', zero_division=0)


# ============================================================
# Run all models
# ============================================================
all_results = {}

print(f"\n{'='*60}")
print(f"  Training all models on {DATASET}")
print(f"{'='*60}")

# 1. InceptionTime (1ch)
print("\n--- InceptionTime (1ch) ---")
it_model = ITNet(ic=1, nc=32, nb=6, ks=[5, 10, 20]).to(device)
it_p = sum(p.numel() for p in it_model.parameters())
it_model, acc, mf1 = train_baseline(f'{DATASET}_IT_1ch', it_model, train_loader_1ch, val_loader_1ch, test_loader_1ch)
res = evaluate(it_model, test_loader_1ch, n_cls)
res['params'] = it_p
all_results['InceptionTime_1ch'] = res
print(f"  MF1={res['macro_f1']:.4f} Params={it_p:,}")

# 2. InceptionTime + Focal (1ch)
print("\n--- InceptionTime + Focal g=1 (1ch) ---")
it_f1 = ITNet(ic=1, nc=32, nb=6, ks=[5, 10, 20]).to(device)
it_f1, acc, mf1 = train_baseline(f'{DATASET}_IT_FOC1', it_f1, train_loader_1ch, val_loader_1ch, test_loader_1ch, focal_gamma=1.0)
res = evaluate(it_f1, test_loader_1ch, n_cls)
res['params'] = it_p
all_results['InceptionTime_focal_g1'] = res
print(f"  MF1={res['macro_f1']:.4f}")

# 3. eTAI CE (3ch)
print("\n--- eTAI CE (3ch) ---")
et_model = ITNet(ic=3, nc=32, nb=6, ks=[5, 10, 20]).to(device)
et_p = sum(p.numel() for p in et_model.parameters())
et_model, acc, mf1 = train_baseline(f'{DATASET}_eTAI_CE', et_model, train_loader_3ch, val_loader_3ch, test_loader_3ch)
res = evaluate(et_model, test_loader_3ch, n_cls)
res['params'] = et_p
all_results['eTAI_CE'] = res
print(f"  MF1={res['macro_f1']:.4f} Params={et_p:,}")

# 4. eTAI focal g=1 (3ch)
print("\n--- eTAI focal g=1 (3ch) ---")
et_f1 = ITNet(ic=3, nc=32, nb=6, ks=[5, 10, 20]).to(device)
et_f1, acc, mf1 = train_baseline(f'{DATASET}_eTAI_FOC1', et_f1, train_loader_3ch, val_loader_3ch, test_loader_3ch, focal_gamma=1.0)
res = evaluate(et_f1, test_loader_3ch, n_cls)
res['params'] = et_p
all_results['eTAI_focal_g1'] = res
print(f"  MF1={res['macro_f1']:.4f}")

# 5. HCRMN-Lite CE
print("\n--- HCRMN-Lite CE ---")
hl_model, hl_p = train_hcrmn_lite(f'{DATASET}_HLITE_CE', max_epochs=25, patience=12)
res = evaluate(hl_model, test_loader, n_cls)
res['params'] = hl_p
all_results['HCRMN-Lite_CE'] = res
print(f"  MF1={res['macro_f1']:.4f} Params={hl_p:,}")

# 6. HCRMN-Lite focal g=1
print("\n--- HCRMN-Lite focal g=1 ---")
hl_f1_model, hl_f1_p = train_hcrmn_lite(f'{DATASET}_HLITE_FOC1', max_epochs=25, patience=12)
res = evaluate(hl_f1_model, test_loader, n_cls)
res['params'] = hl_f1_p
all_results['HCRMN-Lite_focal_g1'] = res
print(f"  MF1={res['macro_f1']:.4f} Params={hl_f1_p:,}")


# ---- Summary ----
print(f"\n{'='*70}")
print(f"  RESULTS: {DATASET}")
print(f"{'='*70}")
print(f"{'Model':<30} {'Acc':>6} {'MF1':>6} {'Params':>10}")
print(f"{'-'*55}")
for name, r in sorted(all_results.items(), key=lambda x: -x[1]['macro_f1']):
    print(f"{name:<30} {r['accuracy']:>6.4f} {r['macro_f1']:>6.4f} {r['params']:>10,}")
    for c in range(n_cls):
        print(f"  C{c} (n={support[c]:>4}): F1={r['class_f1s'][c]:.4f}  Recall={r['class_recalls'][c]:.4f}")

# Save results
os.makedirs(os.path.join(ROOT, 'results', 'hcrmn_lite'), exist_ok=True)
out = os.path.join(ROOT, f'results/hcrmn_lite/{DATASET}.json')
with open(out, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to {out}")
