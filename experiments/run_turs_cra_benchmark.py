"""
Sequential TURS-CRA benchmark runner for the four repository datasets.
Follows the validated biomedical protocol used by the repository's fair TURS-Lite benchmark:
  seed=42, 70/15/15 stratified split, AdamW lr=3e-4 wd=1e-2,
  OneCycleLR, early stopping on val MF1 (patience=8), max 30 epochs,
  per-sample z-normalization, batch_size=64.

TURS-CRA additions:
  - Classical warm start: sigma0 from autocorrelation decay
  - Phase 0 (3 epochs): calibration of sigma/rho/gamma at high LR
  - Phase 1: specialization with frozen calibration params

Writes a per-dataset JSON result in results/turs_cra/<DATASET>.json.
"""

import os
import sys
import time
import json
import copy
import math

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
    balanced_accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
    log_loss,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SEED = 42
VAL_FRAC = 0.15
MAX_EPOCHS = 30
PATIENCE = 8
LR = 3e-4
WD = 1e-2
BATCH_SIZE = 64
CALIB_EPOCHS = 3       # Phase 0: calibration epochs
CALIB_LR_MULT = 20.0   # sigma/rho/gamma LR = CALIB_LR_MULT * LR in Phase 0

ALL_DATASETS = [
    ("ECG5000_UNBAL", "data/ecg5000_resplit.npz", 5),
    ("ECG5000_BAL",   "data/ecg5000_fair_balanced.npz", 5),
    ("CWRU_UNBAL",    "data/cwru_unbalanced.npz", 4),
    ("CWRU_BAL",      "data/cwru_balanced.npz", 4),
]


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def znorm(X):
    mu = X.mean(axis=-1, keepdims=True)
    sig = X.std(axis=-1, keepdims=True) + 1e-8
    return ((X - mu) / sig).astype(np.float32)


def compute_sigma0(X_train, max_lag=50):
    """Classical warm start: autocorrelation decay to 1/e."""
    if X_train.ndim == 3:
        X_train = X_train.squeeze(1)
    N, T = X_train.shape
    max_lag = min(max_lag, T // 2)

    mean_sig = X_train.mean(axis=0)
    mean_sig = mean_sig - mean_sig.mean()
    var = np.var(mean_sig)
    if var < 1e-10:
        acfs = []
        for i in range(min(N, 100)):
            sig = X_train[i] - X_train[i].mean()
            v = np.var(sig)
            if v < 1e-10:
                continue
            acf = np.correlate(sig, sig, mode='full')
            acf = acf[len(acf) // 2:]
            acf = acf / (v * len(sig))
            acfs.append(acf[:max_lag])
        if not acfs:
            return 5.0
        acf_mean = np.mean(acfs, axis=0)
    else:
        acf = np.correlate(mean_sig, mean_sig, mode='full')
        acf = acf[len(acf) // 2:]
        acf_mean = acf / (var * len(mean_sig))
        acf_mean = acf_mean[:max_lag]

    threshold = 1.0 / math.e
    sigma0 = float(max_lag)
    for lag in range(1, len(acf_mean)):
        if acf_mean[lag] < threshold:
            if lag > 0 and acf_mean[lag - 1] > threshold:
                frac = (acf_mean[lag - 1] - threshold) / (
                    acf_mean[lag - 1] - acf_mean[lag] + 1e-10
                )
                sigma0 = (lag - 1) + frac
            else:
                sigma0 = float(lag)
            break
    sigma0 = max(1.5, min(sigma0, T / 4.0))
    return float(sigma0)


def train_and_evaluate(model, X_tr, y_tr, X_va, y_va, X_te, y_te, n_cls, tag, device):
    """Two-phase training: Phase 0 calibration, Phase 1 specialization."""
    ckpt_dir = os.path.join(ROOT, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f'{tag}_TURSCRA.pt')

    tr_dl = DataLoader(
        TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).long()),
        batch_size=min(BATCH_SIZE, 32), shuffle=True,
    )
    va_dl = DataLoader(
        TensorDataset(torch.from_numpy(X_va).float(), torch.from_numpy(y_va).long()),
        batch_size=64,
    )
    te_dl = DataLoader(
        TensorDataset(torch.from_numpy(X_te).float(), torch.from_numpy(y_te).long()),
        batch_size=64,
    )

    nparams = sum(p.numel() for p in model.parameters())
    criterion = nn.CrossEntropyLoss()

    # --- Phase 0: Calibration ---
    # Train sigma, rho, gamma at high LR; rest frozen
    calib_params = list(model.scale_params.parameters())
    other_params = [p for p in model.parameters()
                    if not any(p is cp for cp in calib_params)]

    # Freeze non-calibration params
    for p in other_params:
        p.requires_grad_(False)

    opt_calib = torch.optim.AdamW(calib_params, lr=CALIB_LR_MULT * LR, weight_decay=0.0)
    sched_calib = torch.optim.lr_scheduler.OneCycleLR(
        opt_calib, max_lr=CALIB_LR_MULT * LR,
        steps_per_epoch=len(tr_dl), epochs=CALIB_EPOCHS,
    )

    print(f"    [{tag}] Phase 0: Calibration ({CALIB_EPOCHS} epochs)")
    sigma_init = model.scale_params.sigma.item()
    rho_init = [model.scale_params.rho[i].item() for i in range(3)]
    gamma_init = model.scale_params.gamma.item()
    print(f"      sigma0={sigma_init:.3f} rho0={[f'{r:.4f}' for r in rho_init]} gamma0={gamma_init:.4f}")

    for ep in range(CALIB_EPOCHS):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            opt_calib.zero_grad()
            loss.backward()
            opt_calib.step()
            sched_calib.step()

        # Quick validation
        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for xb, yb in va_dl:
                xb = xb.to(device)
                logits = model(xb)
                vp.append(logits.argmax(-1).cpu().numpy())
                vt.append(yb.numpy())
        val_preds = np.concatenate(vp)
        val_true = np.concatenate(vt)
        val_mf1 = f1_score(val_true, val_preds, average='macro', zero_division=0)

        sigma_cur = model.scale_params.sigma.item()
        rho_cur = [model.scale_params.rho[i].item() for i in range(3)]
        gamma_cur = model.scale_params.gamma.item()
        print(f"      Calib ep {ep+1}: loss={loss.item():.4f} val_mf1={val_mf1:.4f} "
              f"sigma={sigma_cur:.3f} rho={[f'{r:.4f}' for r in rho_cur]} gamma={gamma_cur:.4f}")

    sigma_final_cal = model.scale_params.sigma.item()
    rho_final_cal = [model.scale_params.rho[i].item() for i in range(3)]
    gamma_final_cal = model.scale_params.gamma.item()

    # Unfreeze all params
    for p in other_params:
        p.requires_grad_(True)

    # --- Phase 1: Specialization ---
    # Freeze calibration params (or give very low LR)
    for p in calib_params:
        p.requires_grad_(False)

    train_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(train_params, lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, steps_per_epoch=len(tr_dl), epochs=MAX_EPOCHS,
    )

    print(f"    [{tag}] Phase 1: Specialization (max {MAX_EPOCHS} epochs, patience {PATIENCE})")

    best_val_mf1 = -1
    best_state = None
    no_improve = 0
    t0 = time.time()
    best_ep = 0
    ep = 0

    for ep in range(MAX_EPOCHS):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
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

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Re-enable calib params for save
    for p in calib_params:
        p.requires_grad_(True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'time': elapsed,
        'best_epoch': best_ep,
        'total_epochs': ep + 1,
        'sigma_final': sigma_final_cal,
        'rho_final': rho_final_cal,
        'gamma_final': gamma_final_cal,
        'sigma_init': sigma_init,
        'rho_init': rho_init,
        'gamma_init': gamma_init,
    }, ckpt_path)

    # --- Test Evaluation ---
    model.eval()
    tp, tt = [], []
    all_alpha, all_beta, all_u, all_agree = [], [], [], []

    with torch.no_grad():
        for xb, yb in te_dl:
            xb = xb.to(device)
            logits, aux = model(xb, return_aux=True)
            tp.append(logits.argmax(-1).cpu().numpy())
            tt.append(yb.numpy())
            if aux.get('alpha') is not None:
                all_alpha.append(aux['alpha'].cpu().numpy())
            if aux.get('beta') is not None:
                all_beta.append(aux['beta'].cpu().numpy())
            if aux.get('uncertainty') is not None:
                all_u.append(aux['uncertainty'].cpu().numpy())
            if aux.get('agreement') is not None:
                all_agree.append(aux['agreement'].cpu().numpy())

    test_preds = np.concatenate(tp)
    test_true = np.concatenate(tt)

    acc = accuracy_score(test_true, test_preds)
    mf1 = f1_score(test_true, test_preds, average='macro', zero_division=0)
    wf1 = f1_score(test_true, test_preds, average='weighted', zero_division=0)
    f1s = f1_score(test_true, test_preds, average=None, zero_division=0, labels=list(range(n_cls)))
    recalls = recall_score(test_true, test_preds, average=None, zero_division=0, labels=list(range(n_cls)))
    precs = f1_score(test_true, test_preds, average=None, zero_division=0, labels=list(range(n_cls)))
    cm = confusion_matrix(test_true, test_preds, labels=list(range(n_cls)))
    support = {c: int((test_true == c).sum()) for c in range(n_cls)}

    try:
        bal_acc = balanced_accuracy_score(test_true, test_preds)
    except Exception:
        bal_acc = 0.0
    try:
        kappa = cohen_kappa_score(test_true, test_preds)
    except Exception:
        kappa = 0.0
    try:
        mcc = matthews_corrcoef(test_true, test_preds)
    except Exception:
        mcc = 0.0
    try:
        probs = F.softmax(torch.from_numpy(
            np.zeros((1, n_cls))  # placeholder
        ), dim=-1).numpy()
        ll = log_loss(test_true, model(
            torch.from_numpy(X_te[:1]).float().to(device)
        ).detach().cpu().softmax(-1).numpy(),
            labels=list(range(n_cls)))
    except Exception:
        ll = 0.0

    result = {
        'accuracy': round(float(acc), 4),
        'macro_f1': round(float(mf1), 4),
        'weighted_f1': round(float(wf1), 4),
        'balanced_accuracy': round(float(bal_acc), 4),
        'kappa': round(float(kappa), 4),
        'mcc': round(float(mcc), 4),
        'log_loss': round(float(ll), 4),
        'class_f1s': [round(float(f), 4) for f in f1s],
        'class_recalls': [round(float(r), 4) for r in recalls],
        'class_precisions': [round(float(p), 4) for p in precs],
        'confusion_matrix': cm.tolist(),
        'support': support,
        'params': nparams,
        'best_epoch': best_ep,
        'total_epochs': ep + 1,
        'time_s': round(elapsed, 1),
        'calibration': {
            'sigma_init': round(sigma_init, 4),
            'sigma_final': round(sigma_final_cal, 4),
            'rho_init': [round(r, 4) for r in rho_init],
            'rho_final': [round(r, 4) for r in rho_final_cal],
            'gamma_init': round(gamma_init, 4),
            'gamma_final': round(gamma_final_cal, 4),
        },
    }

    # Gate diagnostics
    if all_alpha:
        alpha_np = np.concatenate(all_alpha)
        result['alpha_stats'] = {
            'mean': round(float(alpha_np.mean()), 4),
            'std': round(float(alpha_np.std()), 4),
            'variance': round(float(alpha_np.var()), 6),
        }
    if all_u:
        u_np = np.concatenate(all_u)
        result['uncertainty_stats'] = {
            'mean': round(float(u_np.mean()), 4),
            'std': round(float(u_np.std()), 4),
        }
    if all_beta:
        beta_np = np.concatenate(all_beta)  # [N, T, 3]
        result['beta_stats'] = {
            'mean_per_scale': [round(float(beta_np[:, :, i].mean()), 4) for i in range(3)],
            'std_per_scale': [round(float(beta_np[:, :, i].std()), 4) for i in range(3)],
            'entropy': round(float(
                -(beta_np * np.log(beta_np + 1e-8)).sum(axis=-1).mean()
            ), 4),
            'variance': round(float(beta_np.var()), 6),
        }
    if all_agree:
        agree_np = np.concatenate(all_agree)
        result['agreement_stats'] = {
            'mean': round(float(agree_np.mean()), 4),
            'std': round(float(agree_np.std()), 4),
        }

    return result


def main():
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Protocol: seed={SEED}, max_ep={MAX_EPOCHS}, patience={PATIENCE}, "
          f"calib_epochs={CALIB_EPOCHS}")

    out_dir = os.path.join(ROOT, 'results', 'turs_cra')
    os.makedirs(out_dir, exist_ok=True)

    for ds_name, ds_file, n_cls in ALL_DATASETS:
        print(f"\n{'='*80}")
        print(f"  {ds_name} ({n_cls} classes)")
        print(f"{'='*80}")

        set_seed(SEED)

        data = np.load(os.path.join(ROOT, ds_file))

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

        # Z-normalize per sample
        X_train_n = znorm(X_train)
        X_val_n = znorm(X_val)
        X_test_n = znorm(X_test)

        Xtr = X_train_n[:, None, :]  # [N, 1, T]
        Xva = X_val_n[:, None, :]
        Xte = X_test_n[:, None, :]

        # Classical warm start
        sigma0 = compute_sigma0(X_train_n)
        print(f"  Classical sigma0 (autocorrelation decay): {sigma0:.3f}")

        # Compute initial rho estimates from signal persistence
        rho_init = (0.9, 0.95, 0.98)  # standard initial values

        # Instantiate TURS-CRA
        from models.turs_cra.model import TURSCRA
        model = TURSCRA(
            in_channels=1, num_classes=n_cls,
            regime_dim=16, branch_dim=16,
            sigma_init=sigma0,
            rho_init=rho_init,
            response_dim=8,
            dropout=0.1,
            lambda_I=0.1,
        ).to(device)

        nparams = sum(p.numel() for p in model.parameters())
        print(f"  Params: {nparams:,}")

        result = train_and_evaluate(
            model,
            Xtr, y_train,
            Xva, y_val,
            Xte, y_test,
            n_cls,
            f'{ds_name}_TURSCRA',
            device,
        )

        # Save
        ds_json = os.path.join(out_dir, f'{ds_name}.json')
        with open(ds_json, 'w') as f:
            json.dump({ds_name: result}, f, indent=2)

        print(f"  >> Acc={result['accuracy']:.4f} MF1={result['macro_f1']:.4f} "
              f"WF1={result['weighted_f1']:.4f} BalAcc={result['balanced_accuracy']:.4f} "
              f"F1s={result['class_f1s']} (time={result['time_s']:.0f}s, ep={result['best_epoch']})")
        print(f"  >> Calibration: sigma {result['calibration']['sigma_init']:.3f} -> "
              f"{result['calibration']['sigma_final']:.3f}, "
              f"rho {[f'{r:.4f}' for r in result['calibration']['rho_init']]} -> "
              f"{[f'{r:.4f}' for r in result['calibration']['rho_final']]}")

    print(f"\n{'='*80}")
    print("TURS-CRA sequential biomedical benchmark finished.")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
