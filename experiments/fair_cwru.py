"""
Fair Comparison on Real CWRU Bearing Fault Data
================================================
Runs all models under identical conditions on the REAL CWRU vibration data.

Models:
  A. InceptionTime (1ch raw)
  B. eTAI-Focal (3ch: raw+quantile+drift, focal gamma=1)
  C. USTR-Net-CE (regime modulation, CE loss)
  D. USTR-Net-Focal (regime modulation, focal gamma=1)

Protocol (identical for all):
  - Seed=42, 70/15/15 stratified split
  - AdamW lr=3e-4, weight_decay=1e-2
  - OneCycleLR scheduler
  - Early stopping on val MF1, patience=8
  - Max 30 epochs
  - Same ITNet backbone: nb=4, nc=32, ks=[5,10,20]
  - Same z-normalization (per-sample)
"""

import os, sys, json, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (f1_score, recall_score, accuracy_score,
                             confusion_matrix, precision_score)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ============================================================
# CONSTANTS
# ============================================================
SEED = 42
VAL_FRAC = 0.15
MAX_EPOCHS = 30
PATIENCE = 8
LR = 3e-4
WD = 1e-2
BATCH_SIZE = 64

DATASETS = [
    ("CWRU_UNBAL", "data/cwru_unbalanced.npz", 4),
    ("CWRU_BAL",   "data/cwru_balanced.npz", 4),
]

# Filter by arg
if len(sys.argv) > 1:
    DATASETS = [d for d in DATASETS if d[0] in sys.argv[1:]]
    if not DATASETS:
        print(f"No matching datasets in args: {sys.argv[1:]}"); sys.exit(1)

# ============================================================
# Transport channels
# ============================================================
def compute_transport(X):
    """X: (N, L) raw -> (N, 3, L) [raw, quantile, drift]"""
    N, L = X.shape
    out = np.zeros((N, 3, L), dtype=np.float32)
    out[:, 0, :] = X.astype(np.float32)
    for i in range(N):
        s = np.sort(X[i])
        out[i, 1, :] = np.interp(np.linspace(0, 1, L),
                                  np.linspace(0, 1, len(s)), s)
        d = np.zeros(L)
        for lag in [1, 2, 4]:
            diff = np.diff(s, n=lag)
            d[lag:] += np.abs(diff) / 3.0
        out[i, 2, :] = d
    for c in range(3):
        mu = out[:, c, :].mean(axis=-1, keepdims=True)
        sig = out[:, c, :].std(axis=-1, keepdims=True) + 1e-8
        out[:, c, :] = (out[:, c, :] - mu) / sig
    return out


# ============================================================
# ITNet backbone
# ============================================================
class InceptionMod(nn.Module):
    def __init__(self, ic, oc, b=32, ks=[5, 10, 20]):
        super().__init__()
        self.bn_layer = nn.Conv1d(ic, b, 1, bias=False) if ic > 1 else nn.Identity()
        ic2 = b if ic > 1 else 1
        self.convs = nn.ModuleList([nn.Conv1d(ic2, oc, k, padding='same', bias=False) for k in ks])
        self.mp = nn.MaxPool1d(3, 1, 1)
        self.pc = nn.Conv1d(ic, oc, 1, bias=False)
        self.norm = nn.BatchNorm1d(oc * len(ks) + oc)
        self.relu = nn.ReLU()
    def forward(self, x):
        bx = self.bn_layer(x)
        return self.relu(self.norm(torch.cat([c(bx) for c in self.convs] + [self.pc(self.mp(x))], 1)))

class ITNet(nn.Module):
    def __init__(self, ic=1, nc=5, nb=4, oc=32, ks=[5, 10, 20]):
        super().__init__()
        self.blks = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        ci = ic
        for i in range(nb):
            self.blks.append(InceptionMod(ci, oc, 32, ks))
            co = oc * len(ks) + oc
            if i % 3 == 2:
                sc_ic = ic if i == 2 else (oc * len(ks) + oc)
                self.shortcuts.append(nn.Sequential(
                    nn.Conv1d(sc_ic, co, 1, bias=False),
                    nn.BatchNorm1d(co)
                ) if sc_ic != co else nn.Identity())
            ci = co
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(ci, nc)
    def forward(self, x):
        si_i, si = x, 0
        for i, b in enumerate(self.blks):
            x = b(x)
            if i % 3 == 2:
                x = torch.relu(x + self.shortcuts[si](si_i))
                si_i, si = x, si + 1
        return self.fc(self.gap(x).squeeze(-1))


# ============================================================
# Focal loss
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=1.0):
        super().__init__()
        self.gamma = gamma
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ============================================================
# USTR-Net (import from models/)
# ============================================================
def make_ustrnet(n_cls):
    from models.ustrnet import USTRNet
    return USTRNet(in_channels=1, num_classes=n_cls, regime_dim=8, base_ch=32)


# ============================================================
# Unified training loop
# ============================================================
def train_and_evaluate(model, X_tr, y_tr, X_va, y_va, X_te, y_te,
                       n_cls, tag, device, use_focal=False, focal_gamma=1.0):
    ckpt_dir = os.path.join(ROOT, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f'{tag}.pt')

    tr_dl = DataLoader(TensorDataset(torch.from_numpy(X_tr).float(),
                                      torch.from_numpy(y_tr).long()),
                       batch_size=BATCH_SIZE, shuffle=True)
    va_dl = DataLoader(TensorDataset(torch.from_numpy(X_va).float(),
                                      torch.from_numpy(y_va).long()),
                       batch_size=256)
    te_dl = DataLoader(TensorDataset(torch.from_numpy(X_te).float(),
                                      torch.from_numpy(y_te).long()),
                       batch_size=256)

    nparams = sum(p.numel() for p in model.parameters())

    criterion = FocalLoss(gamma=focal_gamma) if use_focal else FocalLoss(gamma=1.0)
    # For USTR-Net, the model returns a dict; we need logits
    is_ustr = hasattr(model, 'transport_builder')

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR,
                                                 steps_per_epoch=len(tr_dl),
                                                 epochs=MAX_EPOCHS)

    best_val_mf1 = -1
    best_state = None
    no_improve = 0
    t0 = time.time()
    best_ep = 0
    prev_z = None

    for ep in range(MAX_EPOCHS):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            if is_ustr:
                out = model(xb, prev_z=prev_z)
                logits = out['logits']
                prev_z = out['prev_z'].detach()
            else:
                logits = model(xb)
            loss = criterion(logits, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

        # Validation
        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for xb, yb in va_dl:
                xb = xb.to(device)
                if is_ustr:
                    out = model(xb)
                    logits = out['logits']
                else:
                    logits = model(xb)
                vp.append(logits.argmax(-1).cpu().numpy())
                vt.append(yb.numpy())
        val_preds = np.concatenate(vp)
        val_true = np.concatenate(vt)
        val_mf1 = f1_score(val_true, val_preds, average='macro', zero_division=0)

        if val_mf1 > best_val_mf1:
            best_val_mf1 = val_mf1
            best_state = copy.deepcopy(model.state_dict())
            best_ep = ep + 1
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"    [{tag}] Ep {ep+1:2d}: val_mf1={val_mf1:.4f} (best={best_val_mf1:.4f})")

        if no_improve >= PATIENCE:
            print(f"    [{tag}] Early stop at ep {ep+1}")
            break

    elapsed = time.time() - t0

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({'model_state_dict': model.state_dict(), 'time': elapsed, 'best_epoch': best_ep}, ckpt_path)

    # Test evaluation
    model.eval()
    tp, tt = [], []
    with torch.no_grad():
        for xb, yb in te_dl:
            xb = xb.to(device)
            if is_ustr:
                out = model(xb)
                logits = out['logits']
            else:
                logits = model(xb)
            tp.append(logits.argmax(-1).cpu().numpy())
            tt.append(yb.numpy())
    test_preds = np.concatenate(tp)
    test_true = np.concatenate(tt)

    acc = accuracy_score(test_true, test_preds)
    mf1 = f1_score(test_true, test_preds, average='macro', zero_division=0)
    wf1 = f1_score(test_true, test_preds, average='weighted', zero_division=0)
    prec = precision_score(test_true, test_preds, average='macro', zero_division=0)
    f1s = f1_score(test_true, test_preds, average=None, zero_division=0, labels=list(range(n_cls)))
    recalls = recall_score(test_true, test_preds, average=None, zero_division=0, labels=list(range(n_cls)))
    cm = confusion_matrix(test_true, test_preds, labels=list(range(n_cls)))
    support = {c: int((test_true == c).sum()) for c in range(n_cls)}

    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(mf1), 4),
        "weighted_f1": round(float(wf1), 4),
        "macro_precision": round(float(prec), 4),
        "class_f1s": [round(float(f), 4) for f in f1s],
        "class_recalls": [round(float(r), 4) for r in recalls],
        "confusion_matrix": cm.tolist(),
        "support": support,
        "params": nparams,
        "best_epoch": best_ep,
        "total_epochs": ep + 1,
        "time_s": round(elapsed, 1),
    }


# ============================================================
# Main
# ============================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Protocol: seed={SEED}, max_ep={MAX_EPOCHS}, patience={PATIENCE}, lr={LR}")

    out_dir = os.path.join(ROOT, 'results', 'cwru_real')
    os.makedirs(out_dir, exist_ok=True)

    all_results = {}

    for ds_name, ds_file, n_cls in DATASETS:
        print(f"\n{'='*70}")
        print(f"  {ds_name} ({n_cls} classes)")
        print(f"{'='*70}")

        data = np.load(os.path.join(ROOT, ds_file))
        X_all, y_all = data['X'], data['y'].astype(int)

        # Same split for all models
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.15, stratify=y_all, random_state=SEED)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.15, stratify=y_train, random_state=SEED)

        print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
        print(f"  Signal length: {X_train.shape[1]}")

        # z-normalize per sample
        def znorm(X):
            mu = X.mean(axis=-1, keepdims=True)
            sig = X.std(axis=-1, keepdims=True) + 1e-8
            return ((X - mu) / sig).astype(np.float32)

        X_train_n = znorm(X_train)
        X_val_n = znorm(X_val)
        X_test_n = znorm(X_test)

        # Transport channels
        Xtr_3 = compute_transport(X_train_n)
        Xva_3 = compute_transport(X_val_n)
        Xte_3 = compute_transport(X_test_n)

        # 1ch for InceptionTime
        Xtr_1 = X_train_n[:, None, :]
        Xva_1 = X_val_n[:, None, :]
        Xte_1 = X_test_n[:, None, :]

        # Check existing results
        ds_json = os.path.join(out_dir, f'{ds_name}.json')
        if os.path.exists(ds_json):
            with open(ds_json) as f:
                ds_results = json.load(f)
            print(f"  Loaded {len(ds_results)} existing results")
        else:
            ds_results = {}

        # Model configs: (label, model_key, Xtr, Xva, Xte, ic, model_fn, use_focal, focal_gamma)
        configs = [
            ("InceptionTime", "IT", Xtr_1, Xva_1, Xte_1, 1,
             lambda ic, nc: ITNet(ic=ic, nc=nc, nb=4, oc=32, ks=[5,10,20]),
             False, 1.0),
            ("eTAI-Focal", "eTAI_foc", Xtr_3, Xva_3, Xte_3, 3,
             lambda ic, nc: ITNet(ic=ic, nc=nc, nb=4, oc=32, ks=[5,10,20]),
             True, 1.0),
            ("USTR-Net-CE", "USTRCe", Xtr_1, Xva_1, Xte_1, 1,
             lambda ic, nc: make_ustrnet(nc),
             False, 1.0),
            ("USTR-Net-Focal", "USTRFoc", Xtr_1, Xva_1, Xte_1, 1,
             lambda ic, nc: make_ustrnet(nc),
             True, 1.0),
        ]

        for label, mkey, Xtr_c, Xva_c, Xte_c, ic, model_fn, use_focal, fg in configs:
            if label in ds_results and 'macro_f1' in ds_results[label]:
                print(f"\n  {label}: already done (MF1={ds_results[label]['macro_f1']:.4f})")
                continue

            print(f"\n  Training: {label}")
            ckpt_tag = f"CWRU_{ds_name}_{mkey}"

            model = model_fn(ic, n_cls).to(device)
            nparams = sum(p.numel() for p in model.parameters())
            print(f"    Params: {nparams:,}")

            r = train_and_evaluate(model, Xtr_c, y_train, Xva_c, y_val, Xte_c, y_test,
                                   n_cls, ckpt_tag, device, use_focal=use_focal, focal_gamma=fg)

            ds_results[label] = r
            print(f"    >> Acc={r['accuracy']:.4f} MF1={r['macro_f1']:.4f} "
                  f"WF1={r['weighted_f1']:.4f} F1s={r['class_f1s']} "
                  f"({r['time_s']:.0f}s, ep={r['best_epoch']})")

            with open(ds_json, 'w') as f:
                json.dump(ds_results, f, indent=2)

        all_results[ds_name] = ds_results

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n\n{'='*90}")
    print(f"  REAL CWRU BEARING FAULT — FAIR COMPARISON")
    print(f"  Same split, same optimizer, same early stopping")
    print(f"{'='*90}")

    for ds_name, _, n_cls in DATASETS:
        res = all_results[ds_name]
        nc = n_cls
        print(f"\n  {ds_name}:")
        print(f"  {'Model':<20} {'Acc':>6} {'MF1':>6} {'WF1':>6} {'Prec':>6}", end="")
        for c in range(nc):
            print(f"  F1-C{c}", end="")
        print(f"  {'Ep':>3} {'Time':>6} {'Params':>8}")
        print(f"  {'-'*70}")

        for name, r in res.items():
            print(f"  {name:<20} {r['accuracy']:>6.4f} {r['macro_f1']:>6.4f} "
                  f"{r['weighted_f1']:>6.4f} {r['macro_precision']:>6.4f}", end="")
            for c in range(nc):
                print(f"  {r['class_f1s'][c]:>5.3f}", end="")
            print(f"  {r['best_epoch']:>3} {r['time_s']:>5.0f}s {r['params']:>8,}")

    # Winners
    print(f"\n  WINNERS:")
    for ds_name, _, n_cls in DATASETS:
        res = all_results[ds_name]
        best_name = max(res, key=lambda k: res[k]['macro_f1'])
        best_mf1 = res[best_name]['macro_f1']
        print(f"    {ds_name}: {best_name} (MF1={best_mf1:.4f})")

    # Confusion matrices
    print(f"\n  CONFUSION MATRICES:")
    for ds_name, _, n_cls in DATASETS:
        res = all_results[ds_name]
        print(f"\n  {ds_name}:")
        for name, r in res.items():
            cm = np.array(r['confusion_matrix'])
            print(f"\n    {name}:")
            for row in cm:
                print(f"      {row}")

    # Save full results
    full_path = os.path.join(out_dir, 'full_comparison.json')
    with open(full_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Full results saved to {full_path}")


if __name__ == '__main__':
    main()
