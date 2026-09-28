"""
Sequential TURS-CS benchmark runner for the four repository datasets.
Follows the validated biomedical protocol used by the repository's fair TURS-Lite benchmark:
  seed=42, 70/15/15 stratified split, AdamW lr=3e-4 wd=1e-2,
  OneCycleLR, early stopping on val MF1 (patience=8), max 30 epochs,
  per-sample z-normalization, batch_size=64.

Writes a per-dataset JSON result in results/turs_cs/<DATASET>.json.
"""

import os
import sys
import time
import json
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

torch.autograd.set_detect_anomaly(True)

SEED = 42
VAL_FRAC = 0.15
MAX_EPOCHS = 30
PATIENCE = 8
LR = 3e-4
WD = 1e-2
BATCH_SIZE = 64

ALL_DATASETS = [
    ("ECG5000_UNBAL", "data/ecg5000_resplit.npz", 5),
    ("ECG5000_BAL", "data/ecg5000_fair_balanced.npz", 5),
    ("CWRU_UNBAL", "data/cwru_unbalanced.npz", 4),
    ("CWRU_BAL", "data/cwru_balanced.npz", 4),
]


def compute_transport_3ch(X):
    """Compute the same 3-channel transport representation used by the fair benchmark."""
    N, L = X.shape
    out = np.zeros((N, 3, L), dtype=np.float32)
    out[:, 0, :] = X
    for i in range(N):
        s = np.sort(X[i])
        out[i, 1, :] = np.interp(np.linspace(0, 1, L), np.linspace(0, 1, len(s)), s)
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


def znorm(X):
    mu = X.mean(axis=-1, keepdims=True)
    sig = X.std(axis=-1, keepdims=True) + 1e-8
    return ((X - mu) / sig).astype(np.float32)


def train_and_evaluate(model, X_tr, y_tr, X_va, y_va, X_te, y_te, n_cls, tag, device):
    ckpt_dir = os.path.join(ROOT, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f'{tag}_TURSCS.pt')

    tr_dl = DataLoader(
        TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).long()),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    va_dl = DataLoader(
        TensorDataset(torch.from_numpy(X_va).float(), torch.from_numpy(y_va).long()),
        batch_size=256,
    )
    te_dl = DataLoader(
        TensorDataset(torch.from_numpy(X_te).float(), torch.from_numpy(y_te).long()),
        batch_size=256,
    )

    nparams = sum(p.numel() for p in model.parameters())
    criterion = nn.CrossEntropyLoss()

    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, steps_per_epoch=len(tr_dl), epochs=MAX_EPOCHS)

    best_val_mf1 = -1
    best_state = None
    no_improve = 0
    t0 = time.time()
    best_ep = 0

    for ep in range(MAX_EPOCHS):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            logits, aux = model(xb, return_aux=True)
            loss = criterion(logits, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for xb, yb in va_dl:
                xb = xb.to(device)
                logits, _ = model(xb, return_aux=True)
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

    model.eval()
    tp, tt = [], []
    all_alpha, all_beta, all_u = [], [], []

    with torch.no_grad():
        for xb, yb in te_dl:
            xb = xb.to(device)
            logits, aux = model(xb, return_aux=True)
            if aux.get('alpha') is not None:
                all_alpha.append(aux['alpha'].cpu().numpy())
            if aux.get('beta') is not None:
                all_beta.append(aux['beta'].cpu().numpy())
            if aux.get('uncertainty') is not None:
                all_u.append(aux['uncertainty'].cpu().numpy())
            tp.append(logits.argmax(-1).cpu().numpy())
            tt.append(yb.numpy())

    test_preds = np.concatenate(tp)
    test_true = np.concatenate(tt)

    acc = accuracy_score(test_true, test_preds)
    mf1 = f1_score(test_true, test_preds, average='macro', zero_division=0)
    wf1 = f1_score(test_true, test_preds, average='weighted', zero_division=0)
    f1s = f1_score(test_true, test_preds, average=None, zero_division=0, labels=list(range(n_cls)))
    recalls = recall_score(test_true, test_preds, average=None, zero_division=0, labels=list(range(n_cls)))
    cm = confusion_matrix(test_true, test_preds, labels=list(range(n_cls)))
    support = {c: int((test_true == c).sum()) for c in range(n_cls)}

    result = {
        'accuracy': round(float(acc), 4),
        'macro_f1': round(float(mf1), 4),
        'weighted_f1': round(float(wf1), 4),
        'class_f1s': [round(float(f), 4) for f in f1s],
        'class_recalls': [round(float(r), 4) for r in recalls],
        'confusion_matrix': cm.tolist(),
        'support': support,
        'params': nparams,
        'best_epoch': best_ep,
        'total_epochs': ep + 1,
        'time_s': round(elapsed, 1),
    }

    # Lightweight diagnostics aligned with repository conventions
    if all_alpha:
        alpha_np = np.concatenate(all_alpha)
        result['gate_stats'] = {
            'alpha_mean': round(float(alpha_np.mean()), 4),
            'alpha_std': round(float(alpha_np.std()), 4),
        }
    if all_u:
        u_np = np.concatenate(all_u)
        result['uncertainty_mean'] = round(float(u_np.mean()), 4)
    if all_beta:
        beta_np = np.concatenate(all_beta)
        result['beta_mean'] = round(float(beta_np.mean()), 4)

    return result


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Protocol: seed={SEED}, max_ep={MAX_EPOCHS}, patience={PATIENCE}")

    out_dir = os.path.join(ROOT, 'results', 'turs_cs')
    os.makedirs(out_dir, exist_ok=True)

    for ds_name, ds_file, n_cls in ALL_DATASETS:
        print(f"\n{'='*80}")
        print(f"  {ds_name} ({n_cls} classes)")
        print(f"{'='*80}")

        data = np.load(os.path.join(ROOT, ds_file))

        # Same data handling as fair_turs
        if 'X_train' in data:
            X_all, y_all = data['X_train'], data['y_train'].astype(int)
            X_test, y_test = data['X_test'], data['y_test'].astype(int)
            X_train, X_val, y_train, y_val = train_test_split(
                X_all, y_all, test_size=VAL_FRAC, stratify=y_all, random_state=SEED,
            )
        else:
            X_all, y_all = data['X'], data['y'].astype(int)
            X_train, X_test, y_train, y_test = train_test_split(
                X_all, y_all, test_size=0.15, stratify=y_all, random_state=SEED,
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=VAL_FRAC, stratify=y_train, random_state=SEED,
            )

        print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
        print(f"  Signal length: {X_train.shape[1]}")

        X_train_n = znorm(X_train)
        X_val_n = znorm(X_val)
        X_test_n = znorm(X_test)

        Xtr = X_train_n[:, None, :]
        Xva = X_val_n[:, None, :]
        Xte = X_test_n[:, None, :]

        # Canonical TURS-CS model from existing implementation
        from models.turs_cs.model import TURSCS
        model = TURSCS(in_channels=1, num_classes=n_cls, regime_dim=16, branch_dim=16, sigma_init=5.0, rho_init=(0.9, 0.95, 0.98)).to(device)

        nparams = sum(p.numel() for p in model.parameters())
        print(f"    Params: {nparams:,}")

        result = train_and_evaluate(
            model,
            Xtr,
            y_train,
            Xva,
            y_val,
            Xte,
            y_test,
            n_cls,
            f'{ds_name}_TURSCS',
            device,
        )

        ds_json = os.path.join(out_dir, f'{ds_name}.json')
        with open(ds_json, 'w') as f:
            json.dump({ds_name: result}, f, indent=2)

        print(f"    >> Acc={result['accuracy']:.4f} MF1={result['macro_f1']:.4f} WF1={result['weighted_f1']:.4f} "
              f"F1s={result['class_f1s']} (time={result['time_s']:.0f}s, ep={result['best_epoch']})")

    print("\nTURS-CS sequential biomedical benchmark process finished.")


if __name__ == '__main__':
    main()
