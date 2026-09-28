"""
Bearing Fault: Fair Comparison — Balanced vs Unbalanced
Runs InceptionTime (1ch) + eTAI CE (3ch) + eTAI focal (3ch) on both dataset versions.
Optimized for CPU: subsampled training, reduced epochs, patience-based early stopping.
"""

import os, sys, json, time, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

# ============================================================
# Transport channels
# ============================================================
def compute_transport(X):
    N, L = X.shape
    out = np.zeros((N, 3, L), dtype=np.float32)
    out[:, 0, :] = X.astype(np.float32)
    for i in range(N):
        s = np.sort(X[i])
        out[i, 1, :] = np.interp(np.linspace(0, 1, L), np.linspace(0, 1, len(s)), s)
        d = np.zeros(L)
        for lag in [1, 2, 4]:
            diff = np.diff(s, n=lag)
            d[lag:] += np.abs(diff) / 3.0
        out[i, 2, :] = d
    for c in range(3):
        mu = np.mean(out[:, c, :], axis=-1, keepdims=True)
        sig = np.std(out[:, c, :], axis=-1, keepdims=True) + 1e-8
        out[:, c, :] = (out[:, c, :] - mu) / sig
    return out


# ============================================================
# Model (same InceptionTime, 4 classes)
# ============================================================
class InceptionMod(nn.Module):
    def __init__(self, ic, oc, b=32, ks=[10, 20, 40]):
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


class Short(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.op = nn.Sequential(nn.Conv1d(ic, oc, 1, bias=False), nn.BatchNorm1d(oc)) if ic != oc else nn.Identity()
    def forward(self, x): return self.op(x)


class ITNet(nn.Module):
    def __init__(self, nc=4, ic=1, nb=6, oc=32, ks=[10, 20, 40]):
        super().__init__()
        self.blks, self.shrts = nn.ModuleList(), nn.ModuleList()
        ci = ic
        for i in range(nb):
            self.blks.append(InceptionMod(ci, oc, 32, ks))
            ci = oc * len(ks) + oc
            if i % 3 == 2:
                self.shrts.append(Short(ic if i == 2 else (oc * len(ks) + oc), ci))
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(ci, nc)
    def forward(self, x):
        si_i, si = x, 0
        for i, b in enumerate(self.blks):
            x = b(x)
            if i % 3 == 2:
                x = torch.relu(x + self.shrts[si](si_i))
                si_i, si = x, si + 1
        return self.fc(self.gap(x).squeeze(-1))


# ============================================================
# Dataset & Loss
# ============================================================
class DS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1) if X.ndim == 2 else torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


class FocalLoss(nn.Module):
    def __init__(self, g=1.0):
        super().__init__()
        self.g = g
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.g * ce).mean()


# ============================================================
# Trainer
# ============================================================
def train_eval(model, tr_loader, te_loader, dev, epochs=50, lr=3e-4,
               focal_g=0, label="m", patience=12):
    crit = FocalLoss(focal_g) if focal_g > 0 else nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=epochs,
                                                  steps_per_epoch=max(len(tr_loader), 1))
    best_vl, pat, pci = float('inf'), 0, 0
    bp = os.path.join(BASE, "checkpoints", f"bear_{label}.pt")
    os.makedirs(os.path.dirname(bp), exist_ok=True)

    for ep in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(tr_loader):
            x, y = x.to(dev), y.to(dev)
            loss = crit(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True); sched.step()

        model.eval(); vl, pr, tg = 0.0, [], []
        with torch.no_grad():
            for x, y in te_loader:
                x, y = x.to(dev), y.to(dev)
                o = model(x); vl += crit(o, y).item()
                pr.extend(o.argmax(1).cpu().numpy()); tg.extend(y.cpu().numpy())
        vl /= max(len(te_loader), 1)
        if vl < best_vl:
            best_vl, pat, pci = vl, 0, ep
            torch.save(model.state_dict(), bp)
        else:
            pat += 1
        if pat >= patience:
            break

    model.load_state_dict(torch.load(bp, weights_only=True)); model.eval()
    pr, tg = [], []
    with torch.no_grad():
        for x, y in te_loader:
            x, y = x.to(dev), y.to(dev)
            o = model(x)
            pr.extend(o.argmax(1).cpu().numpy()); tg.extend(y.cpu().numpy())
    yp, yt = np.array(pr), np.array(tg)
    n_classes = len(np.unique(yt))
    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average='macro', zero_division=0)),
        "class_recalls": [float(r) for r in recall_score(yt, yp, average=None, zero_division=0)],
        "n_classes": int(n_classes),
        "confusion_matrix": confusion_matrix(yt, yp).tolist(),
        "epochs_trained": pci + 1,
    }


# ============================================================
# Main
# ============================================================
def main():
    out_dir = os.path.join(BASE, "results", "bearing_fair")
    os.makedirs(out_dir, exist_ok=True)
    dev = torch.device("cpu")
    MAX_TRAIN = 200   # cap per class for CPU speed
    SIG_LEN = 256     # truncate signals for CPU speed

    versions = [
        ("UNBALANCED", "bearing_unbalanced.npz"),
        ("BALANCED", "bearing_balanced.npz"),
    ]

    all_results = {}

    for vname, vfile in versions:
        print(f"\n{'='*70}")
        print(f"  {vname} DATASET")
        print(f"{'='*70}")

        data = np.load(os.path.join(BASE, "data", vfile))
        Xtr, ytr = data["X_train"], data["y_train"].astype(np.int64)
        Xte, yte = data["X_test"], data["y_test"].astype(np.int64)

        # Subsample per class for speed
        rng = np.random.RandomState(42)
        sel = []
        for c in np.unique(ytr):
            idx = np.where(ytr == c)[0]
            if len(idx) > MAX_TRAIN:
                idx = rng.choice(idx, MAX_TRAIN, replace=False)
            sel.extend(idx)
        sel = np.array(sel)
        Xtr, ytr = Xtr[sel], ytr[sel]

        nc = len(np.unique(yte))
        cc = dict(zip(*np.unique(ytr, return_counts=True)))
        print(f"  Train: {len(Xtr)} (subsampled) | Test: {len(Xte)} | Classes: {nc} | Dist: {cc}")

        # Truncate signals for CPU speed
        Xtr = Xtr[:, :SIG_LEN]
        Xte = Xte[:, :SIG_LEN]
        print(f"  Truncated signals to {SIG_LEN} samples")

        # Standardize
        eps = 1e-8
        mu, sig = Xtr.mean(-1, keepdims=True), Xtr.std(-1, keepdims=True)
        Xtr_s = (Xtr - mu) / (sig + eps)
        mu, sig = Xte.mean(-1, keepdims=True), Xte.std(-1, keepdims=True)
        Xte_s = (Xte - mu) / (sig + eps)

        # Transport channels
        Xtr_3 = compute_transport(Xtr_s)
        Xte_3 = compute_transport(Xte_s)

        results = {}

        configs = [
            (f"InceptionTime (1ch)", 1, Xtr_s, Xte_s, 0, 0.0),
            (f"eTAI CE (3ch)", 3, Xtr_3, Xte_3, 0, 0.0),
            (f"eTAI focal g=1 (3ch)", 3, Xtr_3, Xte_3, 1, 1.0),
            (f"eTAI focal g=2 (3ch)", 3, Xtr_3, Xte_3, 2, 2.0),
        ]

        for name, ic, Xtr_c, Xte_c, _, fg in configs:
            print(f"\n  Training: {name}")
            tr_ds = DS(Xtr_c, ytr); te_ds = DS(Xte_c, yte)
            tr_dl = DataLoader(tr_ds, batch_size=64, shuffle=True, num_workers=0)
            te_dl = DataLoader(te_ds, batch_size=64, shuffle=False, num_workers=0)

            model = ITNet(nc=nc, ic=ic, nb=4, oc=32, ks=[5, 10, 20]).to(dev)  # smaller/fewer for 256-length
            np_ = sum(p.numel() for p in model.parameters())
            print(f"    Params: {np_:,}")

            t0 = time.time()
            r = train_eval(model, tr_dl, te_dl, dev, epochs=25, patience=6,
                           focal_g=fg,
                           label=f"bear_{vname}_{name.replace(' ','_').replace('(','').replace(')','')}")
            r["time"] = time.time() - t0
            cr = r["class_recalls"]
            print(f"    Acc={r['accuracy']:.4f} | MF1={r['macro_f1']:.4f} | "
                  f"Recalls={[f'{x:.3f}' for x in cr]} | {r['time']:.0f}s")
            results[name] = r

        all_results[vname] = results

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*90}")
    print("  COMPREHENSIVE BEARING FAULT COMPARISON")
    print(f"{'='*90}")

    for vname in ["UNBALANCED", "BALANCED"]:
        print(f"\n  {vname} Dataset:")
        results = all_results[vname]
        nc = results[list(results.keys())[0]]["n_classes"]
        header = f"  {'Model':<30} {'Acc':>8} {'MF1':>8}"
        for c in range(nc):
            header += f" {'C'+str(c):>6}"
        header += f" {'Time':>6}"
        print(header)
        print(f"  {'-'*len(header)}")
        for name, r in results.items():
            cr = r["class_recalls"]
            line = f"  {name:<30} {r['accuracy']:>8.4f} {r['macro_f1']:>8.4f}"
            for c in range(nc):
                line += f" {cr[c]:>6.3f}"
            line += f" {r['time']:>5.0f}s"
            print(line)

    # Cross-version
    print(f"\n  Cross-Version Comparison:")
    print(f"  {'Model':<30} {'Unbal MF1':>10} {'Bal MF1':>10} {'dF1':>8}")
    print(f"  {'-'*68}")
    for name in all_results["UNBALANCED"]:
        ru = all_results["UNBALANCED"][name]
        rb = all_results["BALANCED"][name]
        print(f"  {name:<30} {ru['macro_f1']:>10.4f} {rb['macro_f1']:>10.4f} {rb['macro_f1']-ru['macro_f1']:>+8.4f}")

    # Save
    with open(os.path.join(out_dir, "comparison.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {out_dir}/comparison.json")


if __name__ == "__main__":
    main()
