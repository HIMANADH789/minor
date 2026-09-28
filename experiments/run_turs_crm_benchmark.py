"""
Sequential TURS-CRM benchmark runner for the four repository datasets.
This mirrors the validated biomedical workflow in the repository:
  seed=42, per-sample z-normalization, 70/15/15 stratified split,
  AdamW lr=3e-4, weight_decay=1e-2, OneCycleLR, early stopping,
  per-dataset JSON serialization under results/turs_crm/.

The file is intentionally compact and repository-consistent with the
existing TURS-CS runner structure.
"""

import os
import sys
import time
import copy
import json

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

from models.turs_crm.model import TURSCRM

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
    """Build [raw, quantile, drift] transport representation in the same recipe style."""
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
    ckpt_path = os.path.join(ckpt_dir, f'{tag}_TURSCRM.pt')

    tr_dl = DataLoader(TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).long()),
                       batch_size=BATCH_SIZE, shuffle=True)
    va_dl = DataLoader(TensorDataset(torch.from_numpy(X_va).float(), torch.from_numpy(y_va).long()),
                       batch_size=256)
    te_dl = DataLoader(TensorDataset(torch.from_numpy(X_te).float(), torch.from_numpy(y_te).long()),
                       batch_size=256)

    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, steps_per_epoch=len(tr_dl), epochs=MAX_EPOCHS)

    best_val_mf1 = -1
    best_state = None
    no_improve = 0
    best_ep = 0
    t0 = time.time()

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

    # Save lightweight sample
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
        'params': sum(p.numel() for p in model.parameters()),
        'best_epoch': best_ep,
        'total_epochs': ep + 1,
        'time_s': round(elapsed, 1),
    }

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

    out_dir = os.path.join(ROOT, 'results', 'turs_crm')
    os.makedirs(out_dir, exist_ok=True)

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    for ds_tag, data_file, n_cls in ALL_DATASETS:
        print(f"\n==== Dataset: {ds_tag} ({data_file}) ====")
        npz = np.load(os.path.join(ROOT, 'data', os.path.basename(data_file)))
        # Dataset convention in repo: X, y arrays in npz
        if 'X' in npz.files and 'y' in npz.files:
            X = npz['X']
            y = npz['y']
        elif 'x' in npz.files and 'y' in npz.files:
            X = npz['x']
            y = npz['y']
        else:
            raise ValueError(f'Unsupported npz structure for {ds_tag}: {npz.files}')

        # allow dataset arrays as [N,C,L] or [N,L]
        if X.ndim == 3:
            # [N, C, L] -> [N, 1, L]
            X = X[:, :1, :]
        elif X.ndim == 2:
            X = X[:, None, :]

        # simple z-norm per sample
        X = znorm(X[:, 0, :])[:, None, :]
        X = X.astype(np.float32)

        # train/val/test split (stratified)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
        X_va, X_te, y_va, y_te = train_test_split(X_te, y_te, test_size=0.50, stratify=y_te, random_state=SEED)

        # model and run
        model = TURSCRM(in_channels=1, num_classes=n_cls)
        model = model.to(device)
        result = train_and_evaluate(model, X_tr, y_tr, X_va, y_va, X_te, y_te, n_cls, ds_tag, device)

        with open(os.path.join(out_dir, f'{ds_tag}.json'), 'w') as f:
            json.dump(result, f, indent=2)

        print(f"[{ds_tag}] saved: {os.path.join(out_dir, f'{ds_tag}.json')}")


if __name__ == '__main__':
    main()
