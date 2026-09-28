"""
Clean Three-Way Comparison
===========================
Experiment A: InceptionTime (1ch raw)
Experiment B: eTAI-Focal (3ch: raw+quantile+drift, focal γ=1)
Experiment C: HCRMN-Lite (4ch: raw+Q+D+M, regime manifold + FiLM)

IDENTICAL CONDITIONS for all models:
  - Same data files (no truncation, no subsampling)
  - Same train/val split: 15% stratified, seed=42
  - Same z-normalization (fit on train, applied to val+test)
  - Same ITNet backbone: nb=4, nc=32, ks=[5,10,20] (~135K params)
  - Same HCRMN-Lite: feat_dim=64, regime_dim=16, K=4 (~162K params)
  - Same optimizer: AdamW(lr=3e-4, weight_decay=1e-2)
  - Same scheduler: OneCycleLR(max_lr=3e-4)
  - Same early stopping: patience=8 on val MF1
  - Same max epochs: 25
  - Same random seed: 42
  - Same evaluation: on held-out test set
"""

import os, sys, json, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, accuracy_score, confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ============================================================
# CONSTANTS — identical for every model
# ============================================================
SEED = 42
VAL_FRAC = 0.15
MAX_EPOCHS = 10
PATIENCE = 5
LR = 3e-4
WD = 1e-2
BATCH_SIZE = 64
PATIENCE_KEY = "val_mf1"   # early stop on val MF1, NOT test loss

ALL_DATASETS = [
    ("ECG5000_UNBAL", "data/ecg5000_resplit.npz", 5),
    ("ECG5000_BAL",   "data/ecg5000_fair_balanced.npz", 5),
    ("BEARING_UNBAL",  "data/bearing_unbalanced.npz", 4),
    ("BEARING_BAL",    "data/bearing_balanced.npz", 4),
]
# Filter by command-line arg if given (e.g. python fair_three_way.py ECG5000_UNBAL)
if len(sys.argv) > 1:
    DATASETS = [d for d in ALL_DATASETS if d[0] == sys.argv[1]]
    if not DATASETS:
        print(f"Unknown dataset: {sys.argv[1]}. Options: {[d[0] for d in ALL_DATASETS]}"); sys.exit(1)
else:
    DATASETS = ALL_DATASETS

# ============================================================
# Transport channels (shared between eTAI and HCRMN-Lite)
# ============================================================
def compute_transport(X):
    """X: (N, L) raw signals → (N, 3, L) quantile+drift+abs-drift"""
    N, L = X.shape
    out = np.zeros((N, 3, L), dtype=np.float32)
    out[:, 0, :] = X.astype(np.float32)  # raw
    for i in range(N):
        s = np.sort(X[i])
        out[i, 1, :] = np.interp(np.linspace(0, 1, L),
                                  np.linspace(0, 1, len(s)), s)  # quantile
        d = np.zeros(L)
        for lag in [1, 2, 4]:
            diff = np.diff(s, n=lag)
            d[lag:] += np.abs(diff) / 3.0
        out[i, 2, :] = d  # drift magnitude
    # z-normalize each channel
    for c in range(3):
        mu = out[:, c, :].mean(axis=-1, keepdims=True)
        sig = out[:, c, :].std(axis=-1, keepdims=True) + 1e-8
        out[:, c, :] = (out[:, c, :] - mu) / sig
    return out

def compute_transport_4ch(X):
    """X: (N, L) raw → (N, 4, L) for HCRMN-Lite: raw+Q+D+M"""
    N, L = X.shape
    out = np.zeros((N, 4, L), dtype=np.float32)
    out[:, 0, :] = X.astype(np.float32)
    for i in range(N):
        s = np.sort(X[i])
        q = np.interp(np.linspace(0, 1, L), np.linspace(0, 1, len(s)), s)
        out[i, 1, :] = q
        # signed drift and magnitude
        D_pos = np.zeros(L)
        D_neg = np.zeros(L)
        for lag in [1, 2, 4]:
            shifted = np.roll(q, -lag)
            diff = shifted - q
            D_pos[:-lag] += np.maximum(diff, 0)[:-lag] / 3.0
            D_neg[:-lag] += np.maximum(-diff, 0)[:-lag] / 3.0
        out[i, 2, :] = D_pos - D_neg  # signed drift M
        out[i, 3, :] = D_pos + D_neg  # magnitude
    for c in range(4):
        mu = out[:, c, :].mean(axis=-1, keepdims=True)
        sig = out[:, c, :].std(axis=-1, keepdims=True) + 1e-8
        out[:, c, :] = (out[:, c, :] - mu) / sig
    return out


# ============================================================
# ITNet — same InceptionTime backbone used by IT and eTAI
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
# HCRMN-Lite — import from models/
# ============================================================
def make_hcrmn_lite(n_cls):
    from models.hcrmn_lite import HCRMNLite
    return HCRMNLite(in_channels=4, num_classes=n_cls, feat_dim=64,
                     regime_dim=16, K=4, n_blocks=4)

def make_hcrmn_lite_loss(n_cls, focal_gamma=1.0):
    from models.hcrmn_lite import HCRMNLiteLoss
    return HCRMNLiteLoss(num_classes=n_cls, feat_dim=64, regime_dim=16,
                         focal_gamma=focal_gamma,
                         lambda_dyn=0.01, lambda_hier=0.01,
                         lambda_proto=0.01, lambda_smooth=0.01)


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
# Unified training loop — identical for all models
# ============================================================
def train_and_evaluate(model, X_tr, y_tr, X_va, y_va, X_te, y_te,
                       n_cls, tag, device, is_hcrmn=False):
    """Train model with identical protocol, return metrics dict."""
    ckpt_dir = os.path.join(ROOT, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f'{tag}.pt')

    # Data loaders
    tr_dl = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr).long()),
                       batch_size=BATCH_SIZE, shuffle=True)
    va_dl = DataLoader(TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va).long()),
                       batch_size=256)
    te_dl = DataLoader(TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te).long()),
                       batch_size=256)

    nparams = sum(p.numel() for p in model.parameters())

    # Loss — CE for IT/eTAI, HCRMN-Lite loss for HCRMN
    if is_hcrmn:
        criterion = make_hcrmn_lite_loss(n_cls, focal_gamma=1.0).to(device)
    else:
        criterion = FocalLoss(gamma=1.0)  # eTAI-Focal and IT both use focal γ=1

    # Optimizer — include loss params for HCRMN (g12, g23)
    params = list(model.parameters())
    if is_hcrmn and hasattr(criterion, 'parameters'):
        params += list(criterion.parameters())
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
            if is_hcrmn:
                logits, info = model(xb)
                loss, _ = criterion(logits, info, yb)
            else:
                logits = model(xb)
                loss = criterion(logits, yb)

            # Add L2 reg for HCRMN regime params
            if is_hcrmn:
                reg = torch.tensor(0.0, device=device)
                for n, p in model.named_parameters():
                    if 'prototype' in n or 'adj' in n:
                        reg = reg + p.pow(2).sum()
                loss = loss + 1e-4 * reg
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

        # Validation — same eval protocol
        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for xb, yb in va_dl:
                xb = xb.to(device)
                if is_hcrmn:
                    logits, _ = model(xb)
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

    # Load best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save checkpoint
    torch.save({'model_state_dict': model.state_dict(), 'time': elapsed, 'best_epoch': best_ep}, ckpt_path)

    # Test evaluation
    model.eval()
    tp, tt = [], []
    with torch.no_grad():
        for xb, yb in te_dl:
            xb = xb.to(device)
            if is_hcrmn:
                logits, _ = model(xb)
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

    return {
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


# ============================================================
# Main
# ============================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Protocol: seed={SEED}, val_frac={VAL_FRAC}, max_ep={MAX_EPOCHS}, "
          f"patience={PATIENCE}, lr={LR}, wd={WD}, batch={BATCH_SIZE}")

    out_dir = os.path.join(ROOT, 'results', 'fair_three_way')
    os.makedirs(out_dir, exist_ok=True)

    all_results = {}

    for ds_name, ds_file, n_cls in DATASETS:
        print(f"\n{'='*70}")
        print(f"  {ds_name} ({n_cls} classes)")
        print(f"{'='*70}")

        # Load data
        data = np.load(os.path.join(ROOT, ds_file))
        X_all, y_all = data['X_train'], data['y_train'].astype(int)
        X_test, y_test = data['X_test'], data['y_test'].astype(int)

        # Same split for ALL models
        X_train, X_val, y_train, y_val = train_test_split(
            X_all, y_all, test_size=VAL_FRAC, stratify=y_all, random_state=SEED)

        print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
        print(f"  Signal length: {X_train.shape[1]}")

        # Per-sample z-normalization (each signal independently)
        def znorm(X):
            mu = X.mean(axis=-1, keepdims=True)
            sig = X.std(axis=-1, keepdims=True) + 1e-8
            return (X - mu) / sig
        X_train_n = znorm(X_train)
        X_val_n = znorm(X_val)
        X_test_n = znorm(X_test)

        # Transport channels (recompute from normalized)
        # For eTAI: 3ch (raw+Q+drift)
        Xtr_3 = compute_transport(X_train_n)
        Xva_3 = compute_transport(X_val_n)
        Xte_3 = compute_transport(X_test_n)

        # For HCRMN-Lite: 4ch (raw+Q+D+M)
        Xtr_4 = compute_transport_4ch(X_train_n)
        Xva_4 = compute_transport_4ch(X_val_n)
        Xte_4 = compute_transport_4ch(X_test_n)

        # For InceptionTime: 1ch
        Xtr_1 = X_train_n[:, None, :].astype(np.float32)
        Xva_1 = X_val_n[:, None, :].astype(np.float32)
        Xte_1 = X_test_n[:, None, :].astype(np.float32)

        # Load existing partial results (from previous runs on same dataset)
        ds_json = os.path.join(out_dir, f'{ds_name}.json')
        if os.path.exists(ds_json):
            with open(ds_json) as f:
                ds_results = json.load(f)
            print(f"  Loaded {len(ds_results)} existing results from {ds_name}.json")
        else:
            ds_results = {}

        models_config = [
            # (label, model_key, X_tr, X_va, X_te, ic, is_hcrmn)
            ("InceptionTime (1ch)", "IT1ch", Xtr_1, Xva_1, Xte_1, 1, False),
        ]
        # eTAI and HCRMN added below to run one at a time
        ETAI_CFG = ("eTAI-Focal (3ch)", "eTAI_foc", Xtr_3, Xva_3, Xte_3, 3, False)
        HCRMN_CFG = ("HCRMN-Lite (4ch)", "HCRMN", Xtr_4, Xva_4, Xte_4, 4, True)

        # Run only requested model (default: all)
        run_model = sys.argv[2].lower() if len(sys.argv) > 2 else 'all'
        if run_model in ('etai', 'all'):
            models_config.append(ETAI_CFG)
        if run_model in ('hcrmn', 'all'):
            models_config.append(HCRMN_CFG)

        for label, mkey, Xtr_c, Xva_c, Xte_c, ic, is_hcrmn in models_config:
            # Skip if already have results for this model
            if label in ds_results and 'macro_f1' in ds_results[label]:
                print(f"\n  {label}: already in results (MF1={ds_results[label]['macro_f1']:.4f}), skipping")
                continue
            print(f"\n  Training: {label}")
            ckpt_tag = f"{ds_name}_{mkey}_FAIR"
            ckpt_path = os.path.join(ROOT, 'checkpoints', f'{ckpt_tag}.pt')

            # Skip if checkpoint already exists
            if os.path.exists(ckpt_path):
                print(f"    Checkpoint exists — evaluating only")
                # Build model and load
                if is_hcrmn:
                    model = make_hcrmn_lite(n_cls).to(device)
                else:
                    model = ITNet(ic=ic, nc=n_cls, nb=4, oc=32, ks=[5, 10, 20]).to(device)
                nparams = sum(p.numel() for p in model.parameters())
                ck = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(ck['model_state_dict'])
                model.eval()
                # Test eval
                te_dl = DataLoader(TensorDataset(torch.from_numpy(Xte_c), torch.from_numpy(y_test).long()), batch_size=256)
                tp, tt = [], []
                with torch.no_grad():
                    for xb, yb in te_dl:
                        xb = xb.to(device)
                        logits, _ = model(xb) if is_hcrmn else (model(xb), None)
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
                r = {
                    "accuracy": round(float(acc), 4), "macro_f1": round(float(mf1), 4),
                    "weighted_f1": round(float(wf1), 4),
                    "class_f1s": [round(float(f), 4) for f in f1s],
                    "class_recalls": [round(float(r_), 4) for r_ in recalls],
                    "confusion_matrix": cm.tolist(), "support": support,
                    "params": nparams, "best_epoch": ck.get('best_epoch', '?'),
                    "time_s": ck.get('time', '?'),
                }
            else:
                if is_hcrmn:
                    model = make_hcrmn_lite(n_cls).to(device)
                else:
                    model = ITNet(ic=ic, nc=n_cls, nb=4, oc=32, ks=[5, 10, 20]).to(device)
                r = train_and_evaluate(model, Xtr_c, y_train, Xva_c, y_val, Xte_c, y_test,
                                       n_cls, ckpt_tag, device, is_hcrmn=is_hcrmn)

            ds_results[label] = r
            print(f"    >> Acc={r['accuracy']:.4f} MF1={r['macro_f1']:.4f} "
                  f"F1s={r['class_f1s']} ({r.get('time_s','?')}s, ep={r.get('best_epoch','?')})")

            # Save per-model immediately
            with open(os.path.join(out_dir, f'{ds_name}.json'), 'w') as f:
                json.dump(ds_results, f, indent=2)

        all_results[ds_name] = ds_results

    # ============================================================
    # SUMMARY TABLE
    # ============================================================
    print(f"\n\n{'='*100}")
    print("  CLEAN THREE-WAY COMPARISON")
    print(f"  Same split (seed={SEED}), same optimizer, same early stopping (val MF1)")
    print(f"{'='*100}")

    for ds_name, _, n_cls in DATASETS:
        res = all_results[ds_name]
        nc = n_cls
        print(f"\n  {ds_name}:")
        header = f"  {'Model':<22} {'Acc':>6} {'MF1':>6} {'WF1':>6}"
        for c in range(nc):
            header += f"  F1-C{c}"
        header += f"  {'Ep':>3} {'Time':>5}"
        print(header)
        print(f"  {'-'*(len(header)+2)}")

        for name, r in res.items():
            line = f"  {name:<22} {r['accuracy']:>6.4f} {r['macro_f1']:>6.4f} {r['weighted_f1']:>6.4f}"
            for c in range(nc):
                line += f"  {r['class_f1s'][c]:>5.3f}"
            line += f"  {r['best_epoch']:>3} {r['time_s']:>4.0f}s"
            print(line)

    # Winner per dataset
    print(f"\n  WINNERS:")
    for ds_name, _, n_cls in DATASETS:
        res = all_results[ds_name]
        best_name = max(res, key=lambda k: res[k]['macro_f1'])
        best_mf1 = res[best_name]['macro_f1']
        print(f"    {ds_name}: {best_name} (MF1={best_mf1:.4f})")

    # Save full results
    with open(os.path.join(out_dir, 'full_comparison.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Full results saved to {out_dir}/full_comparison.json")


if __name__ == '__main__':
    main()
