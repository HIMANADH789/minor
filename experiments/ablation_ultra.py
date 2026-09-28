"""
Ultra-fast channel ablation: 2 seeds, 10 epochs, nb=3, 256-length
Runs ONE dataset at a time via command-line arg.
"""
import os, sys, json, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
os.chdir(BASE)

from experiments.bearing_fair_full import ITNet, FocalLoss

class DS(Dataset):
    def __init__(self, X, y):
        if X.ndim == 2:
            self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        else:
            self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

def standardize(X):
    mu = X.mean(-1, keepdims=True); sig = X.std(-1, keepdims=True) + 1e-8
    return (X - mu) / sig

def compute_quantile(X):
    N, L = X.shape; out = np.zeros((N, L), np.float32)
    for i in range(N):
        s = np.sort(X[i])
        out[i] = np.interp(np.linspace(0, 1, L), np.linspace(0, 1, len(s)), s)
    return out

def compute_drift(X):
    N, L = X.shape; out = np.zeros((N, L), np.float32)
    for i in range(N):
        s = np.sort(X[i]); d = np.zeros(L)
        for lag in [1, 2, 4]:
            diff = np.diff(s, n=lag); d[lag:] += np.abs(diff) / 3.0
        out[i] = d
    return out

def build_channels(X_raw, cfg):
    Xs = standardize(X_raw)
    if cfg == "1ch": return Xs[:, np.newaxis, :]
    if cfg == "2ch_q": return np.stack([Xs, standardize(compute_quantile(Xs))], 1)
    if cfg == "2ch_d": return np.stack([Xs, standardize(compute_drift(Xs))], 1)
    if cfg == "3ch": return np.stack([Xs, standardize(compute_quantile(Xs)), standardize(compute_drift(Xs))], 1)

def eval_f1(model, dl):
    model.eval(); pr, tg = [], []
    with torch.no_grad():
        for xb, yb in dl:
            pr.extend(model(xb).argmax(1).cpu().numpy()); tg.extend(yb.numpy())
    return float(accuracy_score(np.array(tg), np.array(pr))), \
           float(f1_score(np.array(tg), np.array(pr), average='macro', zero_division=0))

def run_one(Xtr, ytr, Xval, yval, Xte, yte, ic, nc, dev, seed, g_val, max_ep=12, pat=6):
    torch.manual_seed(seed); np.random.seed(seed)
    tr_dl = DataLoader(DS(Xtr, ytr), 64, shuffle=True, num_workers=0)
    val_dl = DataLoader(DS(Xval, yval), 64, shuffle=False, num_workers=0)
    te_dl = DataLoader(DS(Xte, yte), 64, shuffle=False, num_workers=0)

    model = ITNet(nc=nc, ic=ic, nb=3, oc=32, ks=[5, 10, 20]).to(dev)
    crit = FocalLoss(g_val) if g_val > 0 else nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, epochs=max_ep,
                                                  steps_per_epoch=max(len(tr_dl), 1))
    best_vl, pat_cnt = float('inf'), 0
    ckpt = os.path.join(BASE, "checkpoints", f"_fast_s{seed}_g{g_val}_ic{ic}.pt")

    for ep in range(max_ep):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(dev), yb.to(dev)
            loss = crit(model(xb), yb); opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        vl = 0.0
        model.eval()
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(dev), yb.to(dev)
                vl += crit(model(xb), yb).item()
        vl /= max(len(val_dl), 1)
        if vl < best_vl: best_vl, pat_cnt = vl, 0; torch.save(model.state_dict(), ckpt)
        else: pat_cnt += 1
        if pat_cnt >= pat: break

    model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True))
    _, val_f1 = eval_f1(model, val_dl)
    test_acc, test_f1 = eval_f1(model, te_dl)
    return val_f1, test_acc, test_f1


if __name__ == "__main__":
    ds_key = sys.argv[1] if len(sys.argv) > 1 else "ECG5000_UNBAL"
    dev = torch.device("cpu")
    SEEDS = [42, 123]
    GAMMAS = [0.0, 1.0]
    CONFIGS = [("1ch", "raw only", 1), ("2ch_q", "raw+quantile", 2),
               ("2ch_d", "raw+drift", 2), ("3ch", "raw+quantile+drift", 3)]

    DATASETS = {
        "ECG5000_UNBAL": ("ecg5000_resplit.npz", 5, False),
        "ECG5000_BAL":   ("ecg5000_fair_balanced.npz", 5, False),
        "BEARING_UNBAL": ("bearing_unbalanced.npz", 4, True),
        "BEARING_BAL":   ("bearing_balanced.npz", 4, True),
    }

    ds_file, nc, is_bearing = DATASETS[ds_key]
    print(f"=== {ds_key} ===")
    t0 = time.time()
    data = np.load(os.path.join(BASE, "data", ds_file))
    Xtr_full, ytr_full = data["X_train"].astype(np.float32), data["y_train"].astype(np.int64)
    Xte, yte = data["X_test"].astype(np.float32), data["y_test"].astype(np.int64)
    if Xtr_full.shape[-1] > 256:
        Xtr_full, Xte = Xtr_full[..., :256], Xte[..., :256]

    rng = np.random.RandomState(42)
    # Subsample to max 500 train + 150 val per class
    MAX_TR = 500 if not is_bearing else 200
    MAX_VAL = 150 if not is_bearing else 60
    perm = rng.permutation(len(ytr_full))
    n_tr = int(0.75 * len(perm))
    Xtr_raw, ytr = Xtr_full[perm[:n_tr]], ytr_full[perm[:n_tr]]
    Xval_raw, yval = Xtr_full[perm[n_tr:]], ytr_full[perm[n_tr:]]

    # Subsample per class
    for cap, (Xs, ys) in [(MAX_TR, (Xtr_raw, ytr)), (MAX_VAL, (Xval_raw, yval))]:
        sel = []
        for c in np.unique(ys):
            idx = np.where(ys == c)[0]
            if len(idx) > cap: idx = rng.choice(idx, cap, replace=False)
            sel.extend(idx)
        s = np.array(sel)
        if Xs is Xtr_raw: Xtr_raw, ytr = Xs[s], ys[s]
        else: Xval_raw, yval = Xs[s], ys[s]

    print(f"  Train: {len(ytr)} | Val: {len(yval)} | Test: {len(yte)}")

    results = {}
    for cfg, desc, ic in CONFIGS:
        print(f"\n  [{desc}] ({ic}ch)")
        Xtr_c = build_channels(Xtr_raw, cfg)
        Xval_c = build_channels(Xval_raw, cfg)
        Xte_c = build_channels(Xte, cfg)

        seed_f1s, seed_accs, best_gammas = [], [], []
        for seed in SEEDS:
            best_g, best_vf1 = 0.0, -1.0
            for g in GAMMAS:
                vf1, _, _ = run_one(Xtr_c, ytr, Xval_c, yval, Xte_c, yte, ic, nc, dev, seed, g)
                if vf1 > best_vf1: best_vf1, best_g = vf1, g
            # Retrain on train+val with best gamma
            Xfull = np.concatenate([Xtr_c, Xval_c], 0)
            yfull = np.concatenate([ytr, yval], 0)
            _, acc, f1 = run_one(Xfull, yfull, Xval_c, yval, Xte_c, yte, ic, nc, dev, seed, best_g)
            seed_f1s.append(f1); seed_accs.append(acc); best_gammas.append(best_g)
            print(f"    seed={seed}: best_g={best_g:.1f} f1={f1:.4f} acc={acc:.4f}")

        results[cfg] = {"desc": desc, "n_ch": ic,
                         "f1_mean": round(float(np.mean(seed_f1s)), 4),
                         "f1_std": round(float(np.std(seed_f1s)), 4),
                         "acc_mean": round(float(np.mean(seed_accs)), 4),
                         "acc_std": round(float(np.std(seed_accs)), 4),
                         "gammas": best_gammas}
        print(f"    >>> f1={np.mean(seed_f1s):.4f}+/-{np.std(seed_f1s):.4f} "
              f"acc={np.mean(seed_accs):.4f}+/-{np.std(seed_accs):.4f}")

    out_dir = os.path.join(BASE, "results", "ablation_channel")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{ds_key}.json"), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  {ds_key} RESULTS")
    print(f"{'='*70}")
    for cfg in ["1ch", "2ch_q", "2ch_d", "3ch"]:
        r = results[cfg]
        print(f"  {r['desc']:<25} {r['n_ch']}ch  F1={r['f1_mean']:.4f}+/-{r['f1_std']:.4f}  "
              f"Acc={r['acc_mean']:.4f}+/-{r['acc_std']:.4f}  gamma={r['gammas']}")
    print(f"  Time: {time.time()-t0:.0f}s")
