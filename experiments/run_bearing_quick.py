"""Quick bearing fault comparison - both balanced and unbalanced."""
import os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score
from rich.console import Console

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
console = Console()

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class InceptionMod(nn.Module):
    def __init__(self, ic, oc, b=32, ks=[10,20,40]):
        super().__init__()
        self.bl = nn.Conv1d(ic,b,1,bias=False) if ic>1 else nn.Identity()
        ic2 = b if ic>1 else 1
        self.convs = nn.ModuleList([nn.Conv1d(ic2,oc,k,padding='same',bias=False) for k in ks])
        self.mp = nn.MaxPool1d(3,1,1); self.pc = nn.Conv1d(ic,oc,1,bias=False)
        self.norm = nn.BatchNorm1d(oc*len(ks)+oc); self.relu = nn.ReLU()
    def forward(self,x):
        bx=self.bl(x)
        return self.relu(self.norm(torch.cat([c(bx) for c in self.convs]+[self.pc(self.mp(x))],1)))

class Short(nn.Module):
    def __init__(self,ic,oc):
        super().__init__()
        self.op = nn.Sequential(nn.Conv1d(ic,oc,1,bias=False),nn.BatchNorm1d(oc)) if ic!=oc else nn.Identity()
    def forward(self,x): return self.op(x)

class ITNet(nn.Module):
    def __init__(self, nc=5, ic=1, nb=6, oc=32, ks=[10,20,40]):
        super().__init__()
        self.blks, self.shrts = nn.ModuleList(), nn.ModuleList()
        ci = ic
        for i in range(nb):
            self.blks.append(InceptionMod(ci, oc, 32, ks))
            ci = oc*len(ks)+oc
            if i%3==2:
                self.shrts.append(Short(ic if i==2 else (oc*len(ks)+oc), ci))
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(ci, nc)
    def forward(self, x):
        si_i, si = x, 0
        for i, b in enumerate(self.blks):
            x = b(x)
            if i%3==2:
                x = torch.relu(x + self.shrts[si](si_i))
                si_i, si = x, si+1
        return self.fc(self.gap(x).squeeze(-1))


class DS(Dataset):
    def __init__(self, X, y):
        X_t = torch.tensor(X, dtype=torch.float32)
        if X_t.ndim == 2:
            X_t = X_t.unsqueeze(1)  # (N, L) -> (N, 1, L)
        self.X = X_t
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


class FocalLoss(nn.Module):
    def __init__(self, g=1.0):
        super().__init__(); self.g = g
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        return ((1 - torch.exp(-ce))**self.g * ce).mean()


def compute_transport(X):
    N, L = X.shape
    out = np.zeros((N, 3, L), dtype=np.float32)
    out[:, 0, :] = X.astype(np.float32)
    for i in range(N):
        s = np.sort(X[i])
        out[i, 1, :] = np.interp(np.linspace(0,1,L), np.linspace(0,1,len(s)), s)
        d = np.zeros(L)
        for lag in [1,2,4]:
            diff = np.diff(s, n=lag); d[lag:] += np.abs(diff)/3.0
        out[i, 2, :] = d
    for c in range(3):
        mu = np.mean(out[:,c,:], axis=-1, keepdims=True)
        sig = np.std(out[:,c,:], axis=-1, keepdims=True) + 1e-8
        out[:,c,:] = (out[:,c,:] - mu) / sig
    return out


def train_eval(model, tr_dl, te_dl, dev, epochs=50, lr=3e-4, fg=0, label="m"):
    crit = FocalLoss(fg) if fg > 0 else nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2, fused=True)
    steps = max(len(tr_dl)//2, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=epochs, steps_per_epoch=steps)
    scaler = torch.cuda.amp.GradScaler()
    best_vl, pat, bp = float('inf'), 15, os.path.join(BASE, "checkpoints", f"{label}.pt")
    os.makedirs(os.path.dirname(bp), exist_ok=True)

    for ep in range(epochs):
        model.train(); opt.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(tr_dl):
            x, y = x.to(dev), y.to(dev)
            with torch.autocast("cuda", torch.float16):
                loss = crit(model(x), y) / 2
            scaler.scale(loss).backward()
            if (i+1)%2==0 or i+1==len(tr_dl):
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
        if vl < best_vl: best_vl = pat = 0; torch.save(model.state_dict(), bp)
        else: pat += 1
        if pat >= 15: break

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


def run_version(vname, vfile):
    console.print(f"\n{'='*70}")
    console.print(f"[bold cyan]{vname} DATASET[/bold cyan]")
    console.print(f"{'='*70}")

    data = np.load(os.path.join(BASE, "data", vfile))
    Xtr, ytr = data["X_train"], data["y_train"].astype(np.int64)
    Xte, yte = data["X_test"], data["y_test"].astype(np.int64)

    cc = dict(zip(*np.unique(ytr, return_counts=True)))
    console.print(f"  Train: {len(Xtr)} | Test: {len(Xte)} | Classes: {cc}\n")

    eps = 1e-8
    mu, sig = Xtr.mean(-1, keepdims=True), Xtr.std(-1, keepdims=True)
    Xtr_s = (Xtr - mu) / (sig + eps)
    mu, sig = Xte.mean(-1, keepdims=True), Xte.std(-1, keepdims=True)
    Xte_s = (Xte - mu) / (sig + eps)

    Xtr_3 = compute_transport(Xtr_s)
    Xte_3 = compute_transport(Xte_s)

    results = {}
    for name, Xtr_c, Xte_c, ic, fg in [
        ("InceptionTime (1ch)", Xtr_s, Xte_s, 1, 0),
        ("eTAI CE (3ch)", Xtr_3, Xte_3, 3, 0),
        ("eTAI focal g=1 (3ch)", Xtr_3, Xte_3, 3, 1.0),
    ]:
        console.print(f"  [bold]{name}[/bold]")
        tr_dl = DataLoader(DS(Xtr_c, ytr), batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
        te_dl = DataLoader(DS(Xte_c, yte), batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

        model = ITNet(nc=5, ic=ic).to(dev)
        t0 = time.time()
        r = train_eval(model, tr_dl, te_dl, dev, fg=fg, label=f"{vname}_{name.replace(' ','').replace('(','').replace(')','')}")
        r["time"] = time.time() - t0
        results[name] = r
        cr = r["class_recalls"]
        console.print(f"    Acc={r['accuracy']:.4f} | F1={r['macro_f1']:.4f} | "
                      f"C0={cr[0]:.3f} C1={cr[1]:.3f} C2={cr[2]:.3f} C3={cr[3]:.3f} | {r['time']:.0f}s")

    return results


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Device: {dev}\n")

    all_r = {}
    for vname, vfile in [("UNBALANCED", "bearing_unbalanced.npz"), ("BALANCED", "bearing_balanced.npz")]:
        all_r[vname] = run_version(vname, vfile)

    # Summary
    console.print(f"\n{'='*90}")
    console.print("[bold cyan]BEARING FAULT DATA - COMPREHENSIVE COMPARISON[/bold cyan]")
    console.print(f"{'='*90}")

    for vname in ["UNBALANCED", "BALANCED"]:
        console.print(f"\n[bold]{vname}:[/bold]")
        console.print(f"  {'Model':<30} {'Acc':>8} {'MacroF1':>8} {'C0':>6} {'C1':>6} {'C2':>6} {'C3':>6}")
        console.print(f"  {'-'*70}")
        for name, r in all_r[vname].items():
            cr = r["class_recalls"]
            console.print(f"  {name:<30} {r['accuracy']:>8.4f} {r['macro_f1']:>8.4f} "
                          f"{cr[0]:>6.3f} {cr[1]:>6.3f} {cr[2]:>6.3f} {cr[3]:>6.3f}")

    console.print(f"\n[bold]Impact of Balancing:[/bold]")
    console.print(f"  {'Model':<30} {'Unbal F1':>10} {'Bal F1':>10} {'dF1':>8}")
    console.print(f"  {'-'*60}")
    for name in all_r["UNBALANCED"]:
        ru = all_r["UNBALANCED"][name]
        rb = all_r["BALANCED"][name]
        console.print(f"  {name:<30} {ru['macro_f1']:>10.4f} {rb['macro_f1']:>10.4f} {rb['macro_f1']-ru['macro_f1']:>+8.4f}")

    with open(os.path.join(BASE, "results", "bearing_fair", "comparison.json"), "w") as f:
        json.dump(all_r, f, indent=2)
    console.print(f"\n[green]Saved to results/bearing_fair/comparison.json[/green]")
