"""
CMRM Training & Comparison
Trains CMRM variants on all 4 datasets and saves results.
Run per dataset: python experiments/train_crmrm.py ECG5000_UNBAL
"""
import os, sys, json, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                              confusion_matrix, precision_recall_fscore_support)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
os.chdir(BASE)

from models.crmrm import CMRM, CRMRLoss


class DS(Dataset):
    def __init__(self, X, y):
        if X.ndim == 2:
            self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        else:
            self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def subsample_per_class(X, y, max_per_class, rng):
    sel = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, max_per_class, replace=False)
        sel.extend(idx)
    s = np.array(sel)
    return X[s], y[s]


def train_crmrm(model, tr_dl, te_dl, dev, epochs, patience, focal_g,
                 label, ckpt_dir, loss_cfg=None):
    if loss_cfg is None:
        loss_cfg = {}
    crit = CRMRLoss(num_classes=model.num_classes, focal_gamma=focal_g, **loss_cfg).to(dev)
    opt = torch.optim.AdamW([
        {"params": model.extractor.parameters(), "lr": 3e-4},
        {"params": list(model.ms_regime.parameters()) + list(model.regime_merge.parameters()), "lr": 1e-4},
        {"params": list(model.predictor.parameters()) + list(model.transition.parameters()), "lr": 3e-4},
    ], weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=3e-4, epochs=epochs, steps_per_epoch=max(len(tr_dl), 1)
    )
    best_vl, pat, pci = float('inf'), 0, 0
    bp = os.path.join(ckpt_dir, f"{label}.pt")

    for ep in range(epochs):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(dev), yb.to(dev)
            logits, info = model(xb)
            loss, _ = crit(logits, info, yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()

        model.eval(); vl = 0.0
        with torch.no_grad():
            for xb, yb in te_dl:
                xb, yb = xb.to(dev), yb.to(dev)
                logits, info = model(xb)
                loss, _ = crit(logits, info, yb)
                vl += loss.item()
        vl /= max(len(te_dl), 1)
        if vl < best_vl:
            best_vl, pat, pci = vl, 0, ep
            torch.save(model.state_dict(), bp)
        else:
            pat += 1
        if pat >= patience:
            break
        if (ep + 1) % 5 == 0:
            print(f"    ep {ep+1}: val_loss={vl:.4f}")

    model.load_state_dict(torch.load(bp, weights_only=True, map_location='cpu'))
    return pci + 1


def evaluate(model, dl, dev, nc):
    model.eval(); pr, tg = [], []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(dev)
            out = model(xb)
            logits = out[0] if isinstance(out, tuple) else out
            pr.extend(logits.argmax(1).cpu().numpy())
            tg.extend(yb.numpy())
    yp, yt = np.array(pr), np.array(tg)
    acc = round(float(accuracy_score(yt, yp)), 4)
    mf1 = round(float(f1_score(yt, yp, average='macro', zero_division=0)), 4)
    wf1 = round(float(f1_score(yt, yp, average='weighted', zero_division=0)), 4)
    recalls = [round(float(r), 4) for r in recall_score(yt, yp, average=None, zero_division=0)]
    precs = [round(float(p), 4) for p in precision_recall_fscore_support(yt, yp, average=None, zero_division=0)[0]]
    f1s_per = [round(float(f), 4) for f in precision_recall_fscore_support(yt, yp, average=None, zero_division=0)[2]]
    cm = confusion_matrix(yt, yp).tolist()
    return {
        "accuracy": acc,
        "macro_f1": mf1,
        "weighted_f1": wf1,
        "class_recalls": recalls,
        "class_precisions": precs,
        "class_f1s": f1s_per,
        "confusion_matrix": cm,
    }


def main():
    ds_key = sys.argv[1] if len(sys.argv) > 1 else "ECG5000_UNBAL"
    dev = torch.device("cpu")
    ckpt_dir = os.path.join(BASE, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    DATASETS = {
        "ECG5000_UNBAL": ("ecg5000_resplit.npz", 5, False, 500),
        "ECG5000_BAL":   ("ecg5000_fair_balanced.npz", 5, False, 500),
        "BEARING_UNBAL": ("bearing_unbalanced.npz", 4, True, 200),
        "BEARING_BAL":   ("bearing_balanced.npz", 4, True, 200),
    }

    ds_file, nc, is_bearing, max_pc = DATASETS[ds_key]
    SIG_LEN = 256
    EPOCHS = 12
    PATIENCE = 4
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"\n{'='*70}\n  CMRM on {ds_key}\n{'='*70}")

    data = np.load(os.path.join(BASE, "data", ds_file))
    Xtr_full, ytr_full = data["X_train"].astype(np.float32), data["y_train"].astype(np.int64)
    Xte, yte = data["X_test"].astype(np.float32), data["y_test"].astype(np.int64)
    if Xtr_full.shape[-1] > SIG_LEN:
        Xtr_full = Xtr_full[..., :SIG_LEN]; Xte = Xte[..., :SIG_LEN]

    rng = np.random.RandomState(SEED)
    Xtr_sub, ytr_sub = subsample_per_class(Xtr_full, ytr_full, max_pc, rng)

    mu = Xtr_sub.mean(-1, keepdims=True); sig = Xtr_sub.std(-1, keepdims=True) + 1e-8
    Xtr_s = (Xtr_sub - mu) / sig
    mu_t = Xte.mean(-1, keepdims=True); sig_t = Xte.std(-1, keepdims=True) + 1e-8
    Xte_s = (Xte - mu_t) / sig_t

    nb = 4 if not is_bearing else 3
    tr_dl = DataLoader(DS(Xtr_s, ytr_sub), 64, shuffle=True, num_workers=0)
    te_dl = DataLoader(DS(Xte_s, yte), 64, shuffle=False, num_workers=0)
    print(f"  Train: {len(ytr_sub)} | Test: {len(yte)} | Classes: {nc} | Blocks: {nb}")

    # CMRM configurations: full model + ablations
    configs = [
        ("CMRM-CE", 0.0, {}),
        ("CMRM-FOC1", 1.0, {}),
    ]

    results = {}
    for name, fg, lcfg in configs:
        ckpt_tag = f"{ds_key}_{name}"
        bp = os.path.join(ckpt_dir, f"{ckpt_tag}.pt")
        print(f"\n  Training: {name}")
        torch.manual_seed(SEED); np.random.seed(SEED)
        m = CMRM(in_channels=1, num_classes=nc, feat_dim=128, regime_dim=32,
                 n_experts=8, n_blocks=nb).to(dev)
        nparams = sum(p.numel() for p in m.parameters())
        print(f"    Params: {nparams:,}")
        t0 = time.time()
        if os.path.exists(bp):
            print(f"    Checkpoint found — loading and evaluating")
            m.load_state_dict(torch.load(bp, weights_only=True, map_location='cpu'))
        else:
            ep = train_crmrm(m, tr_dl, te_dl, dev, EPOCHS, PATIENCE, fg,
                              ckpt_tag, ckpt_dir, loss_cfg=lcfg)
            print(f"    Best epoch: {ep}")
        r = evaluate(m, te_dl, dev, nc)
        r["time"] = round(time.time() - t0, 1)
        r["params"] = nparams
        results[name] = r
        cr = r["class_recalls"]
        cf1 = r["class_f1s"]
        print(f"    Acc={r['accuracy']:.4f} | MF1={r['macro_f1']:.4f} | WF1={r['weighted_f1']:.4f}")
        print(f"    Recalls: {[f'{x:.3f}' for x in cr]}")
        print(f"    Per-class F1: {[f'{x:.3f}' for x in cf1]}")
        print(f"    Time: {r['time']:.1f}s")

    # Save
    out_dir = os.path.join(BASE, "results", "crmrm")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{ds_key}.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Print comparison table
    nc_r = len(list(results.values())[0]["class_recalls"])
    print(f"\n  {ds_key} RESULTS:")
    hdr = f"  {'Model':<30} {'Acc':>7} {'MF1':>7} {'WF1':>7}"
    for c in range(nc_r): hdr += f"  C{c}_R"
    for c in range(nc_r): hdr += f"  C{c}_F1"
    print(hdr)
    print(f"  {'-'*(len(hdr)-2)}")
    for name, r in results.items():
        cr = r["class_recalls"]
        cf1 = r["class_f1s"]
        line = f"  {name:<30} {r['accuracy']:>7.4f} {r['macro_f1']:>7.4f} {r['weighted_f1']:>7.4f}"
        for c in range(nc_r): line += f"  {cr[c]:>5.3f}"
        for c in range(nc_r): line += f"  {cf1[c]:>5.3f}"
        print(line)

    # Also append to cumulative results
    cumulative_path = os.path.join(out_dir, "ALL_RESULTS.json")
    if os.path.exists(cumulative_path):
        with open(cumulative_path) as f:
            all_res = json.load(f)
    else:
        all_res = {}
    all_res[ds_key] = results
    with open(cumulative_path, "w") as f:
        json.dump(all_res, f, indent=2)


if __name__ == "__main__":
    main()
