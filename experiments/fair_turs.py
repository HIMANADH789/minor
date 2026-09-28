"""
Full Benchmark: All 6 Models x All 4 Datasets
==============================================
Models:
  A. InceptionTime (1ch raw, ~135K params)
  B. eTAI-Focal (3ch raw+Q+drift, focal gamma=1, ~170K params)
  C. USTR-Net-CE (1ch, regime modulation, CE, ~100K params)
  D. USTR-Net-Focal (1ch, regime modulation, focal gamma=1, ~100K params)
  E. TURS-Lite (1ch, adaptive T-R fusion, CE, ~137K params)
  F. TURS-Strong (1ch, multi-scale T-R fusion, CE, ~185K params)

Datasets:
  ECG5000_UNBAL, ECG5000_BAL, CWRU_UNBAL, CWRU_BAL

Protocol (identical for all):
  seed=42, 70/15/15 stratified split, AdamW lr=3e-4 wd=1e-2,
  OneCycleLR, early stopping on val MF1 (patience=8), max 30 epochs,
  per-sample z-normalization, batch_size=64
"""

import os, sys, json, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (f1_score, recall_score, accuracy_score,
                             confusion_matrix)

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

ALL_DATASETS = [
    ("ECG5000_UNBAL", "data/ecg5000_resplit.npz", 5),
    ("ECG5000_BAL",   "data/ecg5000_fair_balanced.npz", 5),
    ("CWRU_UNBAL",    "data/cwru_unbalanced.npz", 4),
    ("CWRU_BAL",      "data/cwru_balanced.npz", 4),
]

# Filter by dataset arg
if len(sys.argv) > 1:
    ALL_DATASETS = [d for d in ALL_DATASETS if d[0] in sys.argv[1:]]
    if not ALL_DATASETS:
        print(f"No match. Options: {[d[0] for d in ALL_DATASETS]}")
        sys.exit(1)

# Filter by model arg
RUN_MODELS = None
if len(sys.argv) > 2:
    RUN_MODELS = [m.lower() for m in sys.argv[2:]]


# ============================================================
# Transport channels
# ============================================================
def compute_transport_3ch(X):
    """(N, L) -> (N, 3, L) [raw, quantile, drift]"""
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
# Model factories
# ============================================================
def make_ustrnet(n_cls):
    from models.ustrnet import USTRNet
    return USTRNet(in_channels=1, num_classes=n_cls, regime_dim=8, base_ch=32)

def make_turs_lite(n_cls):
    from models.tursnet import TURSNet
    return TURSNet(in_channels=1, num_classes=n_cls, regime_dim=16,
                   variant="lite", multi_scale_fusion=False)

def make_turs_strong(n_cls):
    from models.tursnet import TURSNet
    return TURSNet(in_channels=1, num_classes=n_cls, regime_dim=24,
                   variant="strong", multi_scale_fusion=True)


# ============================================================
# Training loop
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

    # Loss
    if use_focal:
        criterion = FocalLoss(gamma=focal_gamma)
    else:
        criterion = nn.CrossEntropyLoss()    # Check model output type
    is_turs = hasattr(model, 'use_transport') and hasattr(model, 'use_regime')
    is_dict_model = is_turs or hasattr(model, 'transport_builder')  # TURS or USTR-Net

    # For TURS-Net, also include TURSLoss params if any
    turs_loss = None
    if is_turs:
        from models.tursnet import TURSLoss
        turs_loss = TURSLoss(num_classes=n_cls, focal_gamma=focal_gamma,
                             use_focal=use_focal,
                             lambda_smooth=0.01, lambda_vel=0.005,
                             lambda_unc=0.01, lambda_inter=0.005).to(device)

    params = list(model.parameters())
    if turs_loss is not None and hasattr(turs_loss, 'parameters'):
        params += list(turs_loss.parameters())

    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR,
                                                 steps_per_epoch=len(tr_dl),
                                                 epochs=MAX_EPOCHS)

    best_val_mf1 = -1
    best_state = None
    no_improve = 0
    t0 = time.time()
    best_ep = 0

    for ep in range(MAX_EPOCHS):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            if is_turs:
                logits, aux = model(xb, return_aux=True)
                if turs_loss is not None:
                    loss, _ = turs_loss(logits, yb, aux)
                else:
                    loss = criterion(logits, yb)
            elif is_dict_model:
                out = model(xb)
                logits = out['logits'] if isinstance(out, dict) else out
                loss = criterion(logits, yb)
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
                if is_turs:
                    logits, _ = model(xb, return_aux=True)
                elif is_dict_model:
                    out = model(xb)
                    logits = out['logits'] if isinstance(out, dict) else out
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
    torch.save({'model_state_dict': model.state_dict(), 'time': elapsed,
                'best_epoch': best_ep}, ckpt_path)

    # Test
    model.eval()
    tp, tt = [], []
    # Also collect gate stats for TURS
    all_alpha, all_u, all_gt, all_gr = [], [], [], []

    with torch.no_grad():
        for xb, yb in te_dl:
            xb = xb.to(device)
            if is_turs:
                logits, aux = model(xb, return_aux=True)
                if aux.get('alpha') is not None:
                    all_alpha.append(aux['alpha'].cpu().numpy())
                if aux.get('uncertainty') is not None:
                    all_u.append(aux['uncertainty'].cpu().numpy())
                if aux.get('transport_gate') is not None:
                    all_gt.append(aux['transport_gate'].cpu().numpy())
                if aux.get('regime_gate') is not None:
                    all_gr.append(aux['regime_gate'].cpu().numpy())
            elif is_dict_model:
                out = model(xb)
                if isinstance(out, dict):
                    logits = out['logits']
                    if 'uncertainty' in out:
                        all_u.append(out['uncertainty'].cpu().numpy())
                else:
                    logits = out
            else:
                logits = model(xb)
            tp.append(logits.argmax(-1).cpu().numpy())
            tt.append(yb.numpy())

    test_preds = np.concatenate(tp)
    test_true = np.concatenate(tt)

    acc = accuracy_score(test_true, test_preds)
    mf1 = f1_score(test_true, test_preds, average='macro', zero_division=0)
    wf1 = f1_score(test_true, test_preds, average='weighted', zero_division=0)
    f1s = f1_score(test_true, test_preds, average=None, zero_division=0,
                   labels=list(range(n_cls)))
    recalls = recall_score(test_true, test_preds, average=None, zero_division=0,
                           labels=list(range(n_cls)))
    cm = confusion_matrix(test_true, test_preds, labels=list(range(n_cls)))
    support = {c: int((test_true == c).sum()) for c in range(n_cls)}

    result = {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(mf1), 4),
        "weighted_f1": round(float(wf1), 4),
        "class_f1s": [round(float(f), 4) for f in f1s],
        "class_recalls": [round(float(r), 4) for r in recalls],
        "confusion_matrix": cm.tolist(),
        "support": support,
        "params": nparams,
        "best_epoch": best_ep,
        "total_epochs": ep + 1,
        "time_s": round(elapsed, 1),
    }

    # Gate statistics for TURS
    if all_alpha:
        alpha_np = np.concatenate(all_alpha)
        result["gate_stats"] = {
            "alpha_mean": round(float(alpha_np.mean()), 4),
            "alpha_std": round(float(alpha_np.std()), 4),
        }
        if all_u:
            u_np = np.concatenate(all_u)
            result["gate_stats"]["uncertainty_mean"] = round(float(u_np.mean()), 4)
        if all_gt:
            gt_np = np.concatenate(all_gt)
            result["gate_stats"]["transport_gate_mean"] = round(float(gt_np.mean()), 4)
        if all_gr:
            gr_np = np.concatenate(all_gr)
            result["gate_stats"]["regime_gate_mean"] = round(float(gr_np.mean()), 4)

    return result


# ============================================================
# Main
# ============================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Protocol: seed={SEED}, max_ep={MAX_EPOCHS}, patience={PATIENCE}")

    out_dir = os.path.join(ROOT, 'results', 'turs_benchmark')
    os.makedirs(out_dir, exist_ok=True)

    all_results = {}

    for ds_name, ds_file, n_cls in ALL_DATASETS:
        print(f"\n{'='*80}")
        print(f"  {ds_name} ({n_cls} classes)")
        print(f"{'='*80}")

        # Load data
        data = np.load(os.path.join(ROOT, ds_file))

        # Handle different data formats
        if 'X_train' in data:
            X_all, y_all = data['X_train'], data['y_train'].astype(int)
            X_test, y_test = data['X_test'], data['y_test'].astype(int)
            # Split train into train+val
            X_train, X_val, y_train, y_val = train_test_split(
                X_all, y_all, test_size=VAL_FRAC, stratify=y_all, random_state=SEED)
        else:
            X_all, y_all = data['X'], data['y'].astype(int)
            X_train, X_test, y_train, y_test = train_test_split(
                X_all, y_all, test_size=0.15, stratify=y_all, random_state=SEED)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=VAL_FRAC, stratify=y_train, random_state=SEED)

        print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
        print(f"  Signal length: {X_train.shape[1]}")

        # z-normalize
        def znorm(X):
            mu = X.mean(axis=-1, keepdims=True)
            sig = X.std(axis=-1, keepdims=True) + 1e-8
            return ((X - mu) / sig).astype(np.float32)

        X_train_n = znorm(X_train)
        X_val_n = znorm(X_val)
        X_test_n = znorm(X_test)

        # 1ch
        Xtr_1 = X_train_n[:, None, :]
        Xva_1 = X_val_n[:, None, :]
        Xte_1 = X_test_n[:, None, :]

        # 3ch transport
        Xtr_3 = compute_transport_3ch(X_train_n)
        Xva_3 = compute_transport_3ch(X_val_n)
        Xte_3 = compute_transport_3ch(X_test_n)

        # Load existing
        ds_json = os.path.join(out_dir, f'{ds_name}.json')
        if os.path.exists(ds_json):
            with open(ds_json) as f:
                ds_results = json.load(f)
            print(f"  Loaded {len(ds_results)} existing results")
        else:
            ds_results = {}

        # Model configs: (label, key, factory, Xtr, Xva, Xte, use_focal, fg)
        all_configs = [
            ("InceptionTime", "IT",
             lambda nc: ITNet(ic=1, nc=nc, nb=4, oc=32, ks=[5,10,20]),
             Xtr_1, Xva_1, Xte_1, False, 1.0),
            ("eTAI-Focal", "eTAI_foc",
             lambda nc: ITNet(ic=3, nc=nc, nb=4, oc=32, ks=[5,10,20]),
             Xtr_3, Xva_3, Xte_3, True, 1.0),
            ("USTR-Net-CE", "USTRCe",
             make_ustrnet, Xtr_1, Xva_1, Xte_1, False, 1.0),
            ("USTR-Net-Focal", "USTRFoc",
             make_ustrnet, Xtr_1, Xva_1, Xte_1, True, 1.0),
            ("TURS-Lite", "TURS_Lite",
             make_turs_lite, Xtr_1, Xva_1, Xte_1, False, 1.0),
            ("TURS-Strong", "TURS_Strong",
             make_turs_strong, Xtr_1, Xva_1, Xte_1, False, 1.0),
        ]

        for label, mkey, factory, Xtr_c, Xva_c, Xte_c, use_focal, fg in all_configs:
            if RUN_MODELS:
                match = any(m.lower() in label.lower().replace('-', '').replace(' ', '')
                           for m in RUN_MODELS)
                if not match:
                    continue
            if label in ds_results and 'macro_f1' in ds_results[label]:
                print(f"\n  {label}: already done (MF1={ds_results[label]['macro_f1']:.4f})")
                continue

            print(f"\n  Training: {label}")
            ckpt_tag = f"{ds_name}_{mkey}"
            model = factory(n_cls).to(device)
            nparams = sum(p.numel() for p in model.parameters())
            print(f"    Params: {nparams:,}")

            r = train_and_evaluate(model, Xtr_c, y_train, Xva_c, y_val, Xte_c, y_test,
                                   n_cls, ckpt_tag, device,
                                   use_focal=use_focal, focal_gamma=fg)

            ds_results[label] = r
            print(f"    >> Acc={r['accuracy']:.4f} MF1={r['macro_f1']:.4f} "
                  f"WF1={r['weighted_f1']:.4f} F1s={r['class_f1s']} "
                  f"({r['time_s']:.0f}s, ep={r['best_epoch']})")
            if 'gate_stats' in r:
                print(f"    Gate stats: {r['gate_stats']}")

            with open(ds_json, 'w') as f:
                json.dump(ds_results, f, indent=2)

        all_results[ds_name] = ds_results

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n\n{'='*110}")
    print(f"  FULL BENCHMARK: All Models x All Datasets")
    print(f"{'='*110}")

    # Build summary table
    model_names = ["InceptionTime", "eTAI-Focal", "USTR-Net-CE", "USTR-Net-Focal",
                   "TURS-Lite", "TURS-Strong"]

    for ds_name, _, n_cls in ALL_DATASETS:
        res = all_results.get(ds_name, {})
        print(f"\n  {ds_name}:")
        print(f"  {'Model':<20} {'Acc':>6} {'MF1':>6} {'WF1':>6}", end="")
        for c in range(n_cls):
            print(f"  F1-C{c}", end="")
        print(f"  {'Ep':>3} {'Time':>6} {'Params':>8}")
        print(f"  {'-'*80}")
        for name in model_names:
            if name in res:
                r = res[name]
                print(f"  {name:<20} {r['accuracy']:>6.4f} {r['macro_f1']:>6.4f} "
                      f"{r['weighted_f1']:>6.4f}", end="")
                for c in range(n_cls):
                    print(f"  {r['class_f1s'][c]:>5.3f}", end="")
                print(f"  {r['best_epoch']:>3} {r['time_s']:>5.0f}s {r['params']:>8,}")
                if 'gate_stats' in r:
                    gs = r['gate_stats']
                    print(f"  {'':>20} alpha={gs.get('alpha_mean','?')} "
                          f"unc={gs.get('uncertainty_mean','?')} "
                          f"gT={gs.get('transport_gate_mean','?')} "
                          f"gR={gs.get('regime_gate_mean','?')}")

    # Average MF1 table
    print(f"\n  AVERAGE MF1:")
    print(f"  {'Model':<20}", end="")
    for ds_name, _, _ in ALL_DATASETS:
        print(f"  {ds_name[:10]:>10}", end="")
    print(f"  {'Avg':>6}")
    print(f"  {'-'*70}")
    for name in model_names:
        vals = []
        print(f"  {name:<20}", end="")
        for ds_name, _, _ in ALL_DATASETS:
            res = all_results.get(ds_name, {})
            if name in res and 'macro_f1' in res[name]:
                v = res[name]['macro_f1']
                vals.append(v)
                print(f"  {v:>10.4f}", end="")
            else:
                print(f"  {'---':>10}", end="")
        if vals:
            print(f"  {np.mean(vals):>6.4f}")
        else:
            print()

    # Confusion matrices for CWRU
    for ds_name in ["CWRU_UNBAL", "CWRU_BAL"]:
        if ds_name in all_results:
            res = all_results[ds_name]
            n_cls = 4
            print(f"\n  {ds_name} Confusion Matrices:")
            for name in model_names:
                if name in res and 'confusion_matrix' in res[name]:
                    cm = np.array(res[name]['confusion_matrix'])
                    print(f"\n    {name}:")
                    for row in cm:
                        print(f"      {row}")

    # Save full results
    full_path = os.path.join(out_dir, 'full_benchmark.json')
    with open(full_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Full results saved to {full_path}")


if __name__ == '__main__':
    main()
