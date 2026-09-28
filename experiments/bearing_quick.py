"""Quick bearing fault comparison - inline everything, short signals."""
import os, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score
from rich.console import Console

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
console = Console()
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def gen_signal(n=256, ft='normal', sev=0, sv=0, rng=None):
    if rng is None: rng = np.random.RandomState()
    t = np.arange(n) / 12000.0
    f = 30 + sv * 5
    s = 0.1 * np.sin(2*np.pi*f*t) + rng.randn(n)*0.05
    if ft == 'inner_race':
        p = max(1, int(12000 / (f*3.58)))
        a = 0.3 + sev*0.4
        for i in range(0, n, p):
            il = min(16, n-i)
            s[i:i+il] += a * np.exp(-np.arange(il)/4.0) * rng.randn(il)
    elif ft == 'ball':
        p = max(1, int(12000 / (f*2.36)))
        a = 0.2 + sev*0.3
        for i in range(0, n, p):
            il = min(12, n-i)
            s[i:i+il] += a * np.exp(-np.arange(il)/3.0) * rng.randn(il)
    elif ft == 'outer_race':
        p = max(1, int(12000 / (f*1.43)))
        a = 0.4 + sev*0.5
        for i in range(0, n, p):
            il = min(20, n-i)
            s[i:i+il] += a * np.exp(-np.arange(il)/5.0) * rng.randn(il)
    return (s + rng.randn(n)*0.02).astype(np.float32)


def gen_data(n_tr, n_te, n=256):
    rng = np.random.RandomState(42)
    Xtr, ytr, Xte, yte = [], [], [], []
    faults = ['normal', 'inner_race', 'ball', 'outer_race']
    for fi, ft in enumerate(faults):
        for _ in range(n_tr[fi]):
            sv = rng.uniform(0, 2)
            sev = rng.randint(0, 3) if ft != 'normal' else 0
            Xtr.append(gen_signal(n, ft, sev, sv, rng))
            ytr.append(fi)
        for _ in range(n_te[fi]):
            sv = rng.uniform(0, 2)
            sev = rng.randint(0, 3) if ft != 'normal' else 0
            Xte.append(gen_signal(n, ft, sev, sv, rng))
            yte.append(fi)
    Xtr, ytr = np.array(Xtr), np.array(ytr)
    Xte, yte = np.array(Xte), np.array(yte)
    p = rng.permutation(len(Xtr)); Xtr, ytr = Xtr[p], ytr[p]
    p = rng.permutation(len(Xte)); Xte, yte = Xte[p], yte[p]
    return Xtr, ytr, Xte, yte


def compute_transport(X):
    N, L = X.shape
    out = np.zeros((N, 3, L), dtype=np.float32)
    out[:, 0, :] = X
    for i in range(N):
        s = np.sort(X[i])
        out[i, 1, :] = np.interp(np.linspace(0, 1, L), np.linspace(0, 1, len(s)), s)
        d = np.zeros(L)
        for lag in [1, 2, 4]:
            diff = np.diff(s, n=lag); d[lag:] += np.abs(diff) / 3.0
        out[i, 2, :] = d
    for c in range(3):
        mu = np.mean(out[:, c, :], axis=-1, keepdims=True)
        sig = np.std(out[:, c, :], axis=-1, keepdims=True) + 1e-8
        out[:, c, :] = (out[:, c, :] - mu) / sig
    return out


class InceptionMod(nn.Module):
    def __init__(self, ic, oc, b=32, ks=[10, 20, 40]):
        super().__init__()
        self.bl = nn.Conv1d(ic, b, 1, bias=False) if ic > 1 else nn.Identity()
        ic2 = b if ic > 1 else 1
        self.convs = nn.ModuleList([nn.Conv1d(ic2, oc, k, padding='same', bias=False) for k in ks])
        self.mp = nn.MaxPool1d(3, 1, 1)
        self.pc = nn.Conv1d(ic, oc, 1, bias=False)
        self.norm = nn.BatchNorm1d(oc * len(ks) + oc)
        self.relu = nn.ReLU()
    def forward(self, x):
        bx = self.bl(x)
        return self.relu(self.norm(torch.cat([c(bx) for c in self.convs] + [self.pc(self.mp(x))], 1)))


class Short(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.op = nn.Sequential(nn.Conv1d(ic, oc, 1, bias=False), nn.BatchNorm1d(oc)) if ic != oc else nn.Identity()
    def forward(self, x): return self.op(x)


class ITNet(nn.Module):
    def __init__(self, nc=5, ic=1, nb=6, oc=32, ks=[10, 20, 40]):
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


class DS(Dataset):
    def __init__(self, X, y):
        X_t = torch.tensor(X, dtype=torch.float32)
        if X_t.ndim == 2:
            X_t = X_t.unsqueeze(1)
        self.X = X_t
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


class FocalLoss(nn.Module):
    def __init__(self, g=1.0): super().__init__(); self.g = g
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        return ((1 - torch.exp(-ce)) ** self.g * ce).mean()


def train_eval(model, tr_dl, te_dl, dev, epochs=40, fg=0, label="q"):
    crit = FocalLoss(fg) if fg > 0 else nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2, fused=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, epochs=epochs,
                                                  steps_per_epoch=max(len(tr_dl) // 2, 1))
    scaler = torch.cuda.amp.GradScaler()
    bvl, pat = float('inf'), 12
    bp = os.path.join(BASE, "checkpoints", f"{label}.pt")
    os.makedirs(os.path.dirname(bp), exist_ok=True)

    for ep in range(epochs):
        model.train(); opt.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(tr_dl):
            x, y = x.to(dev), y.to(dev)
            with torch.autocast("cuda", torch.float16):
                loss = crit(model(x), y) / 2
            scaler.scale(loss).backward()
            if (i + 1) % 2 == 0 or i + 1 == len(tr_dl):
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True); sched.step()

        model.eval(); vl = 0; pr, tg = [], []
        with torch.no_grad():
            for x, y in te_dl:
                x, y = x.to(dev), y.to(dev)
                with torch.autocast("cuda", torch.float16):
                    o = model(x); vl += crit(o, y).item()
                pr.extend(o.argmax(1).cpu().numpy()); tg.extend(y.cpu().numpy())
        vl /= max(len(te_dl), 1)
        if vl < bvl:
            bvl = pat = 0; torch.save(model.state_dict(), bp)
        else:
            pat += 1
        if pat >= 12: break

    model.load_state_dict(torch.load(bp, weights_only=True)); model.eval()
    pr, tg = [], []
    with torch.no_grad():
        for x, y in te_dl:
            x, y = x.to(dev), y.to(dev)
            with torch.autocast("cuda", torch.float16): o = model(x)
            pr.extend(o.argmax(1).cpu().numpy()); tg.extend(y.cpu().numpy())
    yp, yt = np.array(pr), np.array(tg)
    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average='macro')),
        "class_recalls": [float(r) for r in recall_score(yt, yp, average=None)],
    }


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Device: {dev}\n")

    all_r = {}

    for vname, n_tr in [("UNBALANCED", [1000, 100, 80, 100]),
                         ("BALANCED", [1000, 1000, 1000, 1000])]:
        console.print(f"[bold cyan]{'='*60}[/bold cyan]")
        console.print(f"[bold cyan]{vname} DATASET[/bold cyan]")
        console.print(f"[bold cyan]{'='*60}[/bold cyan]")

        Xtr, ytr, Xte, yte = gen_data(n_tr, [200, 200, 200, 200])
        eps = 1e-8
        mu, sig = Xtr.mean(-1, keepdims=True), Xtr.std(-1, keepdims=True)
        Xtr = (Xtr - mu) / (sig + eps)
        mu, sig = Xte.mean(-1, keepdims=True), Xte.std(-1, keepdims=True)
        Xte = (Xte - mu) / (sig + eps)

        Xtr3, Xte3 = compute_transport(Xtr), compute_transport(Xte)
        cc = dict(zip(*np.unique(ytr, return_counts=True)))
        console.print(f"  Train: {len(Xtr)} | Classes: {cc}\n")

        vr = {}
        for name, Xtrc, Xtec, ic, fg in [
            ("InceptionTime", Xtr, Xte, 1, 0),
            ("eTAI CE", Xtr3, Xte3, 3, 0),
            ("eTAI focal g=1", Xtr3, Xte3, 3, 1.0),
        ]:
            console.print(f"  [bold]{name}[/bold]")
            tr_dl = DataLoader(DS(Xtrc, ytr), batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
            te_dl = DataLoader(DS(Xtec, yte), batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

            model = ITNet(nc=5, ic=ic).to(dev)
            t0 = time.time()
            r = train_eval(model, tr_dl, te_dl, dev, fg=fg, label=f"{vname}_{name.replace(' ','')}")
            r["time"] = time.time() - t0
            vr[name] = r
            cr = r["class_recalls"]
            console.print(f"    Acc={r['accuracy']:.4f} | F1={r['macro_f1']:.4f} | "
                          f"C0={cr[0]:.3f} C1={cr[1]:.3f} C2={cr[2]:.3f} C3={cr[3]:.3f} | {r['time']:.0f}s\n")
        all_r[vname] = vr

    # Summary
    console.print(f"\n{'='*80}")
    console.print("[bold cyan]BEARING FAULT DATA - FAIR COMPARISON SUMMARY[/bold cyan]")
    console.print(f"{'='*80}")

    for vn in ["UNBALANCED", "BALANCED"]:
        console.print(f"\n[bold]{vn}:[/bold]")
        console.print(f"  {'Model':<20} {'Acc':>8} {'MacroF1':>8} {'C0':>6} {'C1':>6} {'C2':>6} {'C3':>6}")
        console.print(f"  {'-'*62}")
        for n, r in all_r[vn].items():
            cr = r["class_recalls"]
            console.print(f"  {n:<20} {r['accuracy']:>8.4f} {r['macro_f1']:>8.4f} "
                          f"{cr[0]:>6.3f} {cr[1]:>6.3f} {cr[2]:>6.3f} {cr[3]:>6.3f}")

    console.print(f"\n[bold]Impact of Dataset Balancing:[/bold]")
    console.print(f"  {'Model':<20} {'Unbal F1':>10} {'Bal F1':>10} {'Delta':>8}")
    console.print(f"  {'-'*50}")
    for n in all_r["UNBALANCED"]:
        ru = all_r["UNBALANCED"][n]; rb = all_r["BALANCED"][n]
        console.print(f"  {n:<20} {ru['macro_f1']:>10.4f} {rb['macro_f1']:>10.4f} {rb['macro_f1']-ru['macro_f1']:>+8.4f}")

    # Save
    os.makedirs(os.path.join(BASE, "results", "bearing_fair"), exist_ok=True)
    with open(os.path.join(BASE, "results", "bearing_fair", "comparison.json"), "w") as f:
        json.dump(all_r, f, indent=2)
    import json
    with open(os.path.join(BASE, "results", "bearing_fair", "comparison.json"), "w") as f:
        json.dump(all_r, f, indent=2)
    console.print(f"\n[green]Results saved to results/bearing_fair/comparison.json[/green]")
