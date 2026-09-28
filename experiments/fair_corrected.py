"""
Corrected Fair Benchmark — All 6 Models × All 4 Datasets
========================================================
Uses the CORRECT InceptionTime backbone (135K/170K params).

Models:
  A. InceptionTime   — 1ch raw, correct backbone ~135K
  B. eTAI-Focal      — 3ch raw+Q+drift, focal γ=1 ~170K
  C. USTR-Net-CE     — 1ch, regime modulation, CE ~100K
  D. USTR-Net-Focal  — 1ch, regime modulation, focal γ=1 ~100K
  E. TURS-Lite       — 1ch, adaptive T-R fusion, CE ~138K
  F. TURS-Strong     — 1ch, multi-scale T-R fusion, CE ~185K

Usage: python fair_corrected.py DATASET_NAME
"""
import os, sys, json, time, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SEED = 42; LR = 3e-4; WD = 1e-2; BS = 64; MAX_EP = 20; PAT = 6

DS = {
    "ECG5000_UNBAL": ("data/ecg5000_resplit.npz", 5),
    "ECG5000_BAL":   ("data/ecg5000_fair_balanced.npz", 5),
    "CWRU_UNBAL":    ("data/cwru_unbalanced.npz", 4),
    "CWRU_BAL":      ("data/cwru_balanced.npz", 4),
}

# ============================================================
# CORRECT ITNet backbone (135K for 1ch, 170K for 3ch)
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
    """CORRECT InceptionTime backbone — 135K (1ch) / 170K (3ch) params."""
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
                    nn.Conv1d(sc_ic, co, 1, bias=False), nn.BatchNorm1d(co)
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

class FocalLoss(nn.Module):
    def __init__(self, gamma=1.0):
        super().__init__(); self.g = gamma
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.g * ce).mean()


# ============================================================
# Main
# ============================================================
def main():
    ds_name = sys.argv[1] if len(sys.argv) > 1 else None
    if ds_name is None:
        print("Usage: python fair_corrected.py DATASET_NAME")
        print(f"Options: {list(DS.keys())}")
        sys.exit(1)
    assert ds_name in DS, f"Unknown: {ds_name}. Options: {list(DS.keys())}"
    ds_file, n_cls = DS[ds_name]

    # Load data
    data = np.load(os.path.join(ROOT, ds_file))
    if 'X_train' in data:
        Xa, ya = data['X_train'], data['y_train'].astype(int)
        Xte, yte = data['X_test'], data['y_test'].astype(int)
        Xtr, Xva, ytr, yva = train_test_split(Xa, ya, test_size=0.15, stratify=ya, random_state=SEED)
    else:
        Xa, ya = data['X'], data['y'].astype(int)
        Xtr, Xte, ytr, yte = train_test_split(Xa, ya, test_size=0.15, stratify=ya, random_state=SEED)
        Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=0.15, stratify=ytr, random_state=SEED)

    def znorm(X):
        mu = X.mean(axis=-1, keepdims=True)
        sig = X.std(axis=-1, keepdims=True) + 1e-8
        return ((X - mu) / sig).astype(np.float32)

    Xn_tr, Xn_va, Xn_te = znorm(Xtr), znorm(Xva), znorm(Xte)

    def transport_3ch(X):
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

    Xtr1 = Xn_tr[:, None, :]; Xva1 = Xn_va[:, None, :]; Xte1 = Xn_te[:, None, :]
    Xtr3 = transport_3ch(Xn_tr); Xva3 = transport_3ch(Xn_va); Xte3 = transport_3ch(Xn_te)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | {ds_name} ({n_cls}cls) train={len(Xtr)} test={len(Xte)}")

    # Import novel models
    from models.tursnet import TURSNet, TURSLoss
    from models.ustrnet import USTRNet

    configs = [
        ("InceptionTime",   lambda: ITNet(ic=1, nc=n_cls),             Xtr1, Xva1, Xte1, False, False),
        ("eTAI-Focal",      lambda: ITNet(ic=3, nc=n_cls),             Xtr3, Xva3, Xte3, True,  False),
        ("USTR-Net-CE",     lambda: USTRNet(in_channels=1, num_classes=n_cls, regime_dim=8, base_ch=32), Xtr1, Xva1, Xte1, False, True),
        ("USTR-Net-Focal",  lambda: USTRNet(in_channels=1, num_classes=n_cls, regime_dim=8, base_ch=32), Xtr1, Xva1, Xte1, True,  True),
        ("TURS-Lite",       lambda: TURSNet(in_channels=1, num_classes=n_cls, regime_dim=16, variant="lite"), Xtr1, Xva1, Xte1, False, False),
        ("TURS-Strong",     lambda: TURSNet(in_channels=1, num_classes=n_cls, regime_dim=24, variant="strong", multi_scale_fusion=True), Xtr1, Xva1, Xte1, False, False),
    ]

    out_dir = os.path.join(ROOT, 'results', 'corrected_bench')
    os.makedirs(out_dir, exist_ok=True)
    ds_json = os.path.join(out_dir, f'{ds_name}.json')

    if os.path.exists(ds_json):
        with open(ds_json) as f:
            results = json.load(f)
    else:
        results = {}

    for label, factory, Xc_tr, Xc_va, Xc_te, use_focal, is_dict in configs:
        if label in results and 'macro_f1' in results[label]:
            print(f"{label}: done MF1={results[label]['macro_f1']:.4f} params={results[label]['params']}")
            continue

        print(f"Training {label}...", end=" ", flush=True)
        t0 = time.time()
        model = factory().to(device)
        np_ = sum(p.numel() for p in model.parameters())
        print(f"({np_:,} params)", end=" ", flush=True)

        tr_dl = DataLoader(TensorDataset(torch.from_numpy(Xc_tr).float(), torch.from_numpy(ytr).long()), batch_size=BS, shuffle=True)
        va_dl = DataLoader(TensorDataset(torch.from_numpy(Xc_va).float(), torch.from_numpy(yva).long()), batch_size=256)
        te_dl = DataLoader(TensorDataset(torch.from_numpy(Xc_te).float(), torch.from_numpy(yte).long()), batch_size=256)

        is_turs = hasattr(model, 'use_transport') and hasattr(model, 'use_regime')
        turs_loss_fn = None
        if is_turs:
            turs_loss_fn = TURSLoss(num_classes=n_cls, focal_gamma=1.0, use_focal=False).to(device)

        criterion = FocalLoss(gamma=1.0) if use_focal else nn.CrossEntropyLoss()
        params = list(model.parameters())
        if turs_loss_fn is not None:
            params += list(turs_loss_fn.parameters())

        opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, steps_per_epoch=len(tr_dl), epochs=MAX_EP)

        best_mf1 = -1; best_s = None; no_imp = 0; best_ep = 0

        for ep in range(MAX_EP):
            model.train()
            for xb, yb in tr_dl:
                xb, yb = xb.to(device), yb.to(device)
                if is_turs:
                    logits, aux = model(xb, return_aux=True)
                    loss, _ = turs_loss_fn(logits, yb, aux)
                elif is_dict:
                    out = model(xb)
                    logits = out['logits'] if isinstance(out, dict) else out
                    loss = criterion(logits, yb)
                else:
                    logits = model(xb)
                    loss = criterion(logits, yb)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step()

            model.eval(); vp, vt = [], []
            with torch.no_grad():
                for xb, yb in va_dl:
                    xb = xb.to(device)
                    if is_turs: logits, _ = model(xb, return_aux=True)
                    elif is_dict:
                        out = model(xb)
                        logits = out['logits'] if isinstance(out, dict) else out
                    else: logits = model(xb)
                    vp.append(logits.argmax(-1).cpu().numpy()); vt.append(yb.numpy())
            vmf1 = f1_score(np.concatenate(vt), np.concatenate(vp), average='macro', zero_division=0)
            if vmf1 > best_mf1:
                best_mf1 = vmf1; best_s = copy.deepcopy(model.state_dict()); no_imp = 0; best_ep = ep + 1
            else:
                no_imp += 1
            if no_imp >= PAT:
                print(f"(stop@{ep+1})", end=" ", flush=True)
                break
            if ep % 5 == 0 or vmf1 >= best_mf1:
                print(f"ep{ep+1}:{vmf1:.3f}", end=" ", flush=True)

        elapsed = time.time() - t0
        if best_s: model.load_state_dict(best_s)

        # Test
        model.eval(); tp, tt = [], []
        with torch.no_grad():
            for xb, yb in te_dl:
                xb = xb.to(device)
                if is_turs: logits, _ = model(xb, return_aux=True)
                elif is_dict:
                    out = model(xb)
                    logits = out['logits'] if isinstance(out, dict) else out
                else: logits = model(xb)
                tp.append(logits.argmax(-1).cpu().numpy()); tt.append(yb.numpy())
        preds, true = np.concatenate(tp), np.concatenate(tt)

        r = {
            "accuracy": round(float(accuracy_score(true, preds)), 4),
            "macro_f1": round(float(f1_score(true, preds, average='macro', zero_division=0)), 4),
            "weighted_f1": round(float(f1_score(true, preds, average='weighted', zero_division=0)), 4),
            "class_f1s": [round(float(f), 4) for f in f1_score(true, preds, average=None, zero_division=0, labels=list(range(n_cls)))],
            "confusion_matrix": confusion_matrix(true, preds, labels=list(range(n_cls))).tolist(),
            "params": np_, "best_epoch": best_ep, "time_s": round(elapsed, 1),
        }
        results[label] = r
        with open(ds_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"=> MF1={r['macro_f1']:.4f} Acc={r['accuracy']:.4f} ({r['time_s']:.0f}s)")

    # Summary
    print(f"\n{'='*80}\n  {ds_name} SUMMARY (CORRECTED)\n{'='*80}")
    print(f"  {'Model':<20} {'Params':>8} {'MF1':>6} {'Acc':>6}", end="")
    for c in range(n_cls): print(f"  F1-C{c}", end="")
    print(f"  {'Ep':>3} {'Time':>6}")
    print(f"  {'-'*80}")
    for label, _, _, _, _, _, _ in configs:
        if label in results:
            r = results[label]
            print(f"  {label:<20} {r['params']:>8,} {r['macro_f1']:>6.4f} {r['accuracy']:>6.4f}", end="")
            for c in range(n_cls): print(f"  {r['class_f1s'][c]:>5.3f}", end="")
            print(f"  {r['best_epoch']:>3} {r['time_s']:>5.0f}s")

if __name__ == '__main__':
    main()
