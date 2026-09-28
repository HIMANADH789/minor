"""
Channel-level ablation with proper statistical methodology:
1. Channel configs: 1ch (raw), 2ch (raw+quantile), 2ch (raw+drift), 3ch (full)
2. 5 random seeds for mean ± std
3. Validation-based γ selection (sweep γ ∈ {0, 0.5, 1.0, 1.5, 2.0, 3.0} on val set, pick best val F1, report test F1)
4. Reports mean ± std across seeds
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
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1) if X.ndim == 2 else torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ── Transport channel computation ──────────────────────────────────
def compute_quantile_channel(X):
    """Quantile function of sorted signal — the OT amplitude distribution."""
    N, L = X.shape
    out = np.zeros((N, L), dtype=np.float32)
    for i in range(N):
        s = np.sort(X[i])
        out[i] = np.interp(np.linspace(0, 1, L), np.linspace(0, 1, len(s)), s)
    return out


def compute_drift_channel(X):
    """Multi-lag distributional drift — how the amplitude distribution deforms."""
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


def standardize(X):
    mu = X.mean(-1, keepdims=True)
    sig = X.std(-1, keepdims=True) + 1e-8
    return (X - mu) / sig


def build_channels(X_raw, config):
    """Build input channels based on ablation config."""
    X_raw = standardize(X_raw)
    if config == "1ch":
        return X_raw[:, np.newaxis, :]            # (N, 1, L)
    elif config == "2ch_quantile":
        q = standardize(compute_quantile_channel(X_raw))
        return np.stack([X_raw, q], axis=1)       # (N, 2, L)
    elif config == "2ch_drift":
        d = standardize(compute_drift_channel(X_raw))
        return np.stack([X_raw, d], axis=1)       # (N, 2, L)
    elif config == "3ch_full":
        q = standardize(compute_quantile_channel(X_raw))
        d = standardize(compute_drift_channel(X_raw))
        return np.stack([X_raw, q, d], axis=1)    # (N, 3, L)
    else:
        raise ValueError(f"Unknown config: {config}")


# ── Training with validation-based γ selection ────────────────────
def train_one_epoch(model, dl, crit, opt, sched, dev):
    model.train()
    for xb, yb in dl:
        xb, yb = xb.to(dev), yb.to(dev)
        loss = crit(model(xb), yb)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()


def evaluate_model(model, dl, dev):
    model.eval()
    pr, tg = [], []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(dev)
            pr.extend(model(xb).argmax(1).cpu().numpy())
            tg.extend(yb.numpy())
    yp, yt = np.array(pr), np.array(tg)
    return float(accuracy_score(yt, yp)), float(f1_score(yt, yp, average='macro', zero_division=0))


def train_and_select_gamma(Xtr, ytr, Xval, yval, Xte, yte, ic, nc, nb, ks,
                           dev, seed, max_epochs=25, patience=10):
    """
    For each γ in {0, 0.5, 1.0, 1.5, 2.0, 3.0}:
      train on (Xtr, ytr), pick best epoch by val loss, record val F1.
    Pick best γ by val F1, retrain on (Xtr ∪ Xval), report test F1.
    """
    gamma_candidates = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    tr_dl = DataLoader(DS(Xtr, ytr), 64, shuffle=True, num_workers=0)
    val_dl = DataLoader(DS(Xval, yval), 64, shuffle=False, num_workers=0)
    te_dl = DataLoader(DS(Xte, yte), 64, shuffle=False, num_workers=0)

    best_gamma = 0.0
    best_val_f1 = -1.0
    gamma_val_results = {}

    for g in gamma_candidates:
        torch.manual_seed(seed)
        model = ITNet(nc=nc, ic=ic, nb=nb, oc=32, ks=ks).to(dev)
        crit = FocalLoss(g) if g > 0 else nn.CrossEntropyLoss()
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=3e-4, epochs=max_epochs, steps_per_epoch=max(len(tr_dl), 1))

        best_vl, pat, best_ep = float('inf'), 0, 0
        ckpt = os.path.join(BASE, "checkpoints", f"ablation_tmp_seed{seed}_g{g}_ic{ic}.pt")

        for ep in range(max_epochs):
            train_one_epoch(model, tr_dl, crit, opt, sched, dev)
            model.eval(); vl = 0.0
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(dev), yb.to(dev)
                    vl += crit(model(xb), yb).item()
            vl /= max(len(val_dl), 1)
            if vl < best_vl:
                best_vl, pat, best_ep = vl, 0, ep
                torch.save(model.state_dict(), ckpt)
            else:
                pat += 1
            if pat >= patience:
                break

        # Evaluate best epoch on validation
        model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True))
        _, val_f1 = evaluate_model(model, val_dl, dev)
        gamma_val_results[g] = round(val_f1, 4)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_gamma = g

    # Retrain best γ on train+val combined
    Xtr_full = np.concatenate([Xtr, Xval], axis=0)
    ytr_full = np.concatenate([ytr, yval], axis=0)
    tr_full_dl = DataLoader(DS(Xtr_full, ytr_full), 64, shuffle=True, num_workers=0)

    torch.manual_seed(seed)
    model = ITNet(nc=nc, ic=ic, nb=nb, oc=32, ks=ks).to(dev)
    crit = FocalLoss(best_gamma) if best_gamma > 0 else nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=3e-4, epochs=max_epochs, steps_per_epoch=max(len(tr_full_dl), 1))

    best_vl, pat = float('inf'), 0
    ckpt_final = os.path.join(BASE, "checkpoints", f"ablation_final_seed{seed}_ic{ic}_g{best_gamma}.pt")

    for ep in range(max_epochs):
        train_one_epoch(model, tr_full_dl, crit, opt, sched, dev)
        model.eval(); vl = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(dev), yb.to(dev)
                vl += crit(model(xb), yb).item()
        vl /= max(len(val_dl), 1)
        if vl < best_vl:
            best_vl, pat = vl, 0
            torch.save(model.state_dict(), ckpt_final)
        else:
            pat += 1
        if pat >= patience:
            break

    model.load_state_dict(torch.load(ckpt_final, map_location='cpu', weights_only=True))
    test_acc, test_f1 = evaluate_model(model, te_dl, dev)

    # Also get per-class recalls
    model.eval(); pr, tg = [], []
    with torch.no_grad():
        for xb, yb in te_dl:
            pr.extend(model(xb).argmax(1).cpu().numpy()); tg.extend(yb.numpy())
    yp, yt = np.array(pr), np.array(tg)
    recalls = [round(float(r), 4) for r in recall_score(yt, yp, average=None, zero_division=0)]

    return {
        "best_gamma": best_gamma,
        "gamma_val_f1s": gamma_val_results,
        "test_accuracy": round(test_acc, 4),
        "test_macro_f1": round(test_f1, 4),
        "test_recalls": recalls,
    }


# ── Main ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    dev = torch.device("cpu")
    SEEDS = [42, 123, 456, 789, 2024]
    GAMMA_CANDIDATES = [0.0, 1.0, 2.0]

    channel_configs = {
        "1ch":           ("raw only (baseline)", 1),
        "2ch_quantile":  ("raw + quantile", 2),
        "2ch_drift":     ("raw + drift", 2),
        "3ch_full":      ("raw + quantile + drift (full eTAI)", 3),
    }

    datasets = [
        ("ECG5000_UNBAL", "ecg5000_resplit.npz", 5, False),
        ("ECG5000_BAL",   "ecg5000_fair_balanced.npz", 5, False),
        ("BEARING_UNBAL", "bearing_unbalanced.npz", 4, True),
        ("BEARING_BAL",   "bearing_balanced.npz", 4, True),
    ]

    all_results = {}

    for ds_name, ds_file, nc, is_bearing in datasets:
        print(f"\n{'='*80}")
        print(f"  {ds_name}")
        print(f"{'='*80}")

        data = np.load(os.path.join(BASE, "data", ds_file))
        Xtr_full = data["X_train"].astype(np.float32)
        ytr_full = data["y_train"].astype(np.int64)
        Xte = data["X_test"].astype(np.float32)
        yte = data["y_test"].astype(np.int64)

        sig_len = Xte.shape[-1]
        if sig_len > 512:
            Xtr_full = Xtr_full[..., :512]
            Xte = Xte[..., :512]

        # 60/20/20 split from train (to get proper val set)
        rng = np.random.RandomState(42)
        n = len(ytr_full)
        perm = rng.permutation(n)
        n_tr = int(0.75 * n)  # 75% train, 25% val from original train
        tr_idx, val_idx = perm[:n_tr], perm[n_tr:]

        Xtr_raw, ytr = Xtr_full[tr_idx], ytr_full[tr_idx]
        Xval_raw, yval = Xtr_full[val_idx], ytr_full[val_idx]

        # Subsample for bearing speed
        MAX_PER_CLASS = 200 if is_bearing else 9999
        def subsample(X, y, max_pc, rng_local):
            sel = []
            for c in np.unique(y):
                idx = np.where(y == c)[0]
                if len(idx) > max_pc:
                    idx = rng_local.choice(idx, max_pc, replace=False)
                sel.extend(idx)
            return X[np.array(sel)], y[np.array(sel)]

        Xtr_raw, ytr = subsample(Xtr_raw, ytr, MAX_PER_CLASS, rng)
        Xval_raw, yval = subsample(Xval_raw, yval, MAX_PER_CLASS, rng)

        nb = 4
        ks = [5, 10, 20] if is_bearing else [5, 10, 20]

        print(f"  Train: {len(ytr)} | Val: {len(yval)} | Test: {len(yte)} | nc={nc}")

        ds_results = {}
        for cfg_name, (cfg_desc, ic) in channel_configs.items():
            print(f"\n  --- {cfg_desc} ({cfg_name}) ---")
            seed_results = []

            for si, seed in enumerate(SEEDS):
                t0 = time.time()
                # Build channels
                Xtr_c = build_channels(Xtr_raw, cfg_name)
                Xval_c = build_channels(Xval_raw, cfg_name)
                Xte_c = build_channels(Xte, cfg_name)

                res = train_and_select_gamma(
                    Xtr_c, ytr, Xval_c, yval, Xte_c, yte,
                    ic=ic, nc=nc, nb=nb, ks=ks,
                    dev=dev, seed=seed, max_epochs=15, patience=8
                )
                res["time"] = round(time.time() - t0, 1)
                seed_results.append(res)
                print(f"    seed={seed}: γ={res['best_gamma']:.1f} "
                      f"val_f1s={res['gamma_val_f1s']} "
                      f"test_f1={res['test_macro_f1']:.4f} ({res['time']:.0f}s)")

            # Aggregate across seeds
            f1s = [r["test_macro_f1"] for r in seed_results]
            accs = [r["test_accuracy"] for r in seed_results]
            best_gammas = [r["best_gamma"] for r in seed_results]

            ds_results[cfg_name] = {
                "description": cfg_desc,
                "n_channels": ic,
                "macro_f1_mean": round(float(np.mean(f1s)), 4),
                "macro_f1_std": round(float(np.std(f1s)), 4),
                "accuracy_mean": round(float(np.mean(accs)), 4),
                "accuracy_std": round(float(np.std(accs)), 4),
                "best_gammas": best_gammas,
                "per_seed": seed_results,
            }
            print(f"    >>> MEAN: f1={np.mean(f1s):.4f}±{np.std(f1s):.4f}  "
                  f"acc={np.mean(accs):.4f}±{np.std(accs):.4f}  "
                  f"gammas={best_gammas}")

        all_results[ds_name] = ds_results

    # ── Save ────────────────────────────────────────────────────────
    out_dir = os.path.join(BASE, "results", "ablation_channel")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), 'w') as f:
        json.dump(all_results, f, indent=2)

    # ── Print final table ───────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  FINAL ABLATION TABLE: Channel Config × Dataset (mean ± std over 5 seeds)")
    print("=" * 90)
    for ds_name in all_results:
        print(f"\n  {ds_name}:")
        print(f"  {'Config':<40} {'#Ch':>3} {'Macro F1':>14} {'Accuracy':>14} {'γ selected'}")
        print(f"  {'-'*40} {'-'*3} {'-'*14} {'-'*14} {'-'*10}")
        for cfg_name in ["1ch", "2ch_quantile", "2ch_drift", "3ch_full"]:
            r = all_results[ds_name][cfg_name]
            print(f"  {r['description']:<40} {r['n_channels']:>3} "
                  f"{r['macro_f1_mean']:.4f}±{r['macro_f1_std']:.4f} "
                  f"{r['accuracy_mean']:.4f}±{r['accuracy_std']:.4f}  "
                  f"{r['best_gammas']}")

    print(f"\nSaved to {out_dir}/results.json")
