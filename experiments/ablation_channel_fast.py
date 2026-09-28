"""
Channel-level ablation — optimized for speed:
- 4 configs × 3 seeds × 2 gamma candidates
- Proper validation-based gamma selection
- Mean ± std reporting
"""
import os, sys, json, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score

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
    mu = X.mean(-1, keepdims=True)
    sig = X.std(-1, keepdims=True) + 1e-8
    return (X - mu) / sig

def compute_quantile(X):
    N, L = X.shape
    out = np.zeros((N, L), dtype=np.float32)
    for i in range(N):
        s = np.sort(X[i])
        out[i] = np.interp(np.linspace(0, 1, L), np.linspace(0, 1, len(s)), s)
    return out

def compute_drift(X):
    N, L = X.shape
    out = np.zeros((N, L), dtype=np.float32)
    for i in range(N):
        s = np.sort(X[i])
        d = np.zeros(L)
        for lag in [1, 2, 4]:
            diff = np.diff(s, n=lag)
            d[lag:] += np.abs(diff) / 3.0
        out[i] = d
    return out

def build_channels(X_raw, cfg):
    Xs = standardize(X_raw)
    if cfg == "1ch":
        return Xs[:, np.newaxis, :]
    if cfg == "2ch_q":
        return np.stack([Xs, standardize(compute_quantile(Xs))], 1)
    if cfg == "2ch_d":
        return np.stack([Xs, standardize(compute_drift(Xs))], 1)
    if cfg == "3ch":
        return np.stack([Xs, standardize(compute_quantile(Xs)), standardize(compute_drift(Xs))], 1)

def subsample(X, y, mx, rng):
    sel = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        if len(idx) > mx:
            idx = rng.choice(idx, mx, replace=False)
        sel.extend(idx)
    s = np.array(sel)
    return X[s], y[s]

def eval_f1(model, dl):
    model.eval()
    pr, tg = [], []
    with torch.no_grad():
        for xb, yb in dl:
            pr.extend(model(xb).argmax(1).cpu().numpy())
            tg.extend(yb.numpy())
    yp, yt = np.array(pr), np.array(tg)
    return float(accuracy_score(yt, yp)), float(f1_score(yt, yp, average='macro', zero_division=0))


def run_one(Xtr, ytr, Xval, yval, Xte, yte, ic, nc, nb, ks, dev, seed, g_val, max_ep=20, pat=8):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr_dl = DataLoader(DS(Xtr, ytr), 64, shuffle=True, num_workers=0)
    val_dl = DataLoader(DS(Xval, yval), 64, shuffle=False, num_workers=0)
    te_dl = DataLoader(DS(Xte, yte), 64, shuffle=False, num_workers=0)

    model = ITNet(nc=nc, ic=ic, nb=nb, oc=32, ks=ks).to(dev)
    crit = FocalLoss(g_val) if g_val > 0 else nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, epochs=max_ep,
                                                  steps_per_epoch=max(len(tr_dl), 1))
    best_vl, pat_cnt = float('inf'), 0
    ckpt = os.path.join(BASE, "checkpoints", f"_abl_f_s{seed}_g{g_val}_ic{ic}.pt")

    for ep in range(max_ep):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(dev), yb.to(dev)
            loss = crit(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        vl = 0.0
        model.eval()
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(dev), yb.to(dev)
                vl += crit(model(xb), yb).item()
        vl /= max(len(val_dl), 1)
        if vl < best_vl:
            best_vl, pat_cnt = vl, 0
            torch.save(model.state_dict(), ckpt)
        else:
            pat_cnt += 1
        if pat_cnt >= pat:
            break

    model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True))
    _, val_f1 = eval_f1(model, val_dl)
    test_acc, test_f1 = eval_f1(model, te_dl)

    model.eval()
    pr, tg = [], []
    with torch.no_grad():
        for xb, yb in te_dl:
            pr.extend(model(xb).argmax(1).cpu().numpy())
            tg.extend(yb.numpy())
    recalls = [round(float(r), 4) for r in recall_score(np.array(tg), np.array(pr), average=None, zero_division=0)]

    return val_f1, test_acc, test_f1, recalls


if __name__ == "__main__":
    dev = torch.device("cpu")
    SEEDS = [42, 123, 456]
    GAMMAS = [0.0, 1.0]
    CONFIGS = [
        ("1ch", "raw only", 1),
        ("2ch_q", "raw + quantile", 2),
        ("2ch_d", "raw + drift", 2),
        ("3ch", "raw + quantile + drift", 3),
    ]

    DATASETS = [
        ("ECG5000_UNBAL", "ecg5000_resplit.npz", 5, False),
        ("ECG5000_BAL", "ecg5000_fair_balanced.npz", 5, False),
        ("BEARING_UNBAL", "bearing_unbalanced.npz", 4, True),
        ("BEARING_BAL", "bearing_balanced.npz", 4, True),
    ]

    all_results = {}
    grand_start = time.time()

    for ds_name, ds_file, nc, is_bearing in DATASETS:
        print(f"\n{'='*80}\n  {ds_name}\n{'='*80}")
        data = np.load(os.path.join(BASE, "data", ds_file))
        Xtr_full, ytr_full = data["X_train"].astype(np.float32), data["y_train"].astype(np.int64)
        Xte, yte = data["X_test"].astype(np.float32), data["y_test"].astype(np.int64)
        if Xte.shape[-1] > 512:
            Xtr_full, Xte = Xtr_full[..., :512], Xte[..., :512]

        rng = np.random.RandomState(42)
        perm = rng.permutation(len(ytr_full))
        n_tr = int(0.75 * len(perm))
        tr_idx, val_idx = perm[:n_tr], perm[n_tr:]
        Xtr_raw, ytr = Xtr_full[tr_idx], ytr_full[tr_idx]
        Xval_raw, yval = Xtr_full[val_idx], ytr_full[val_idx]

        MAX_PC = 200 if is_bearing else 9999
        Xtr_raw, ytr = subsample(Xtr_raw, ytr, MAX_PC, rng)
        Xval_raw, yval = subsample(Xval_raw, yval, MAX_PC, rng)

        nb, ks = 4, [5, 10, 20]
        print(f"  Train: {len(ytr)} | Val: {len(yval)} | Test: {len(yte)}")

        ds_results = {}
        for cfg, cfg_desc, ic in CONFIGS:
            print(f"\n  [{cfg_desc}] ({cfg}, {ic}ch)")
            seed_f1s, seed_accs, seed_gammas = [], [], []
            t_start = time.time()

            for seed in SEEDS:
                Xtr_c = build_channels(Xtr_raw, cfg)
                Xval_c = build_channels(Xval_raw, cfg)
                Xte_c = build_channels(Xte, cfg)

                # Phase 1: find best gamma on validation
                best_g, best_vf1 = 0.0, -1.0
                gamma_vf1s = {}
                for g in GAMMAS:
                    vf1, _, _, _ = run_one(Xtr_c, ytr, Xval_c, yval, Xte_c, yte,
                                            ic, nc, nb, ks, dev, seed, g)
                    gamma_vf1s[g] = round(vf1, 4)
                    if vf1 > best_vf1:
                        best_vf1, best_g = vf1, g

                # Phase 2: retrain best gamma on train+val, evaluate on test
                Xtr_full_c = np.concatenate([Xtr_c, Xval_c], axis=0)
                ytr_full_c = np.concatenate([ytr, yval], axis=0)
                _, test_acc, test_f1, recalls = run_one(
                    Xtr_full_c, ytr_full_c, Xval_c, yval, Xte_c, yte,
                    ic, nc, nb, ks, dev, seed, best_g)

                seed_f1s.append(test_f1)
                seed_accs.append(test_acc)
                seed_gammas.append(best_g)
                print(f"    seed={seed}: g_vf1s={gamma_vf1s} best_g={best_g:.1f} "
                      f"test_f1={test_f1:.4f} test_acc={test_acc:.4f}")

            dt = time.time() - t_start
            ds_results[cfg] = {
                "desc": cfg_desc, "n_ch": ic,
                "f1_mean": round(float(np.mean(seed_f1s)), 4),
                "f1_std": round(float(np.std(seed_f1s)), 4),
                "acc_mean": round(float(np.mean(seed_accs)), 4),
                "acc_std": round(float(np.std(seed_accs)), 4),
                "gammas": seed_gammas,
                "per_seed_f1": seed_f1s,
                "per_seed_acc": seed_accs,
                "time": round(dt, 1),
            }
            print(f"    >>> f1={np.mean(seed_f1s):.4f}+/-{np.std(seed_f1s):.4f} "
                  f"acc={np.mean(seed_accs):.4f}+/-{np.std(seed_accs):.4f} ({dt:.0f}s)")

        all_results[ds_name] = ds_results

    # Save
    out_dir = os.path.join(BASE, "results", "ablation_channel")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results_final.json"), 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print final table
    print("\n" + "=" * 110)
    print("  FINAL TABLE: Channel Ablation × Dataset (3 seeds, val-based gamma)")
    print("=" * 110)
    for ds_name in all_results:
        print(f"\n  {ds_name}:")
        print(f"  {'Config':<30} {'Ch':>3} {'Macro F1 (mean±std)':>24} {'Accuracy (mean±std)':>24} {'gamma picks'}")
        print(f"  {'-'*30} {'-'*3} {'-'*24} {'-'*24} {'-'*20}")
        for cfg in ["1ch", "2ch_q", "2ch_d", "3ch"]:
            r = all_results[ds_name][cfg]
            print(f"  {r['desc']:<30} {r['n_ch']:>3} "
                  f"{r['f1_mean']:.4f} +/- {r['f1_std']:.4f}      "
                  f"{r['acc_mean']:.4f} +/- {r['acc_std']:.4f}  "
                  f"{r['gammas']}")
    total = time.time() - grand_start
    print(f"\nTotal time: {total/60:.1f} minutes")
    print(f"Saved to {out_dir}/results_final.json")
