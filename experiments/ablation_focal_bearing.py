"""Ablation: IT + Focal on BEARING only (aggressive CPU speed)."""
import os, sys, json, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
os.chdir(BASE)

from experiments.bearing_fair_full import ITNet, DS, FocalLoss

def train_eval(model, tr_dl, te_dl, epochs=20, focal_g=0, ckpt="tmp.pt"):
    crit = FocalLoss(focal_g) if focal_g > 0 else nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, epochs=epochs,
                                                  steps_per_epoch=max(len(tr_dl), 1))
    best, pat, pci = float('inf'), 0, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in tr_dl:
            loss = crit(model(xb), yb); opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        model.eval(); vl, pr, tg = 0.0, [], []
        with torch.no_grad():
            for xb, yb in te_dl:
                o = model(xb); vl += crit(o, yb).item()
                pr.extend(o.argmax(1).cpu().numpy()); tg.extend(yb.cpu().numpy())
        vl /= max(len(te_dl), 1)
        if vl < best: best, pat, pci = vl, 0, ep; torch.save(model.state_dict(), ckpt)
        else: pat += 1
        if pat >= 8: break
    model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True))
    model.eval(); pr, tg = [], []
    with torch.no_grad():
        for xb, yb in te_dl:
            pr.extend(model(xb).argmax(1).cpu().numpy()); tg.extend(yb.numpy())
    yp, yt = np.array(pr), np.array(tg)
    return {
        "accuracy": round(float(accuracy_score(yt, yp)), 4),
        "macro_f1": round(float(f1_score(yt, yp, average='macro', zero_division=0)), 4),
        "recalls": [round(float(r), 4) for r in recall_score(yt, yp, average=None, zero_division=0)],
        "epochs": pci + 1,
    }

results = {}
for split, fname in [("UNBALANCED", "bearing_unbalanced.npz"), ("BALANCED", "bearing_balanced.npz")]:
    print(f"\n=== BEARING {split} ===")
    data = np.load(os.path.join(BASE, "data", fname))
    Xtr, ytr = data["X_train"].astype(np.float32), data["y_train"].astype(np.int64)
    Xte, yte = data["X_test"].astype(np.float32), data["y_test"].astype(np.int64)
    Xtr, Xte = Xtr[..., :256], Xte[..., :256]  # truncate for speed
    eps = 1e-8
    mu, sig = Xtr.mean(-1, keepdims=True), Xtr.std(-1, keepdims=True); Xtr = (Xtr - mu) / (sig + eps)
    mu, sig = Xte.mean(-1, keepdims=True), Xte.std(-1, keepdims=True); Xte = (Xte - mu) / (sig + eps)
    # Subsample
    rng = np.random.RandomState(42); sel = []
    for c in np.unique(ytr):
        idx = np.where(ytr == c)[0]
        if len(idx) > 200: idx = rng.choice(idx, 200, replace=False)
        sel.extend(idx)
    sel = np.array(sel); Xtr, ytr = Xtr[sel], ytr[sel]
    nc = len(np.unique(yte))
    print(f"  train={len(Xtr)} test={len(Xte)} nc={nc}")

    results[split] = {}
    for fg, lbl in [(0.0, "CE"), (1.0, "focal_g1"), (2.0, "focal_g2")]:
        name = f"IT {lbl} (1ch)"
        tr_dl = DataLoader(DS(Xtr, ytr), 64, shuffle=True)
        te_dl = DataLoader(DS(Xte, yte), 64, shuffle=False)
        model = ITNet(nc=nc, ic=1, nb=4, oc=32, ks=[5,10,20])
        ckpt = os.path.join(BASE, "checkpoints", f"ablation_BEARING_{split}_{lbl}_1ch.pt")
        t0 = time.time()
        res = train_eval(model, tr_dl, te_dl, epochs=20, focal_g=fg, ckpt=ckpt)
        dt = round(time.time()-t0, 1)
        results[split][name] = res
        print(f"  {name}: acc={res['accuracy']:.4f} f1={res['macro_f1']:.4f} recalls={res['recalls']} ({dt}s)")

# Save
os.makedirs(os.path.join(BASE, "results", "ablation_focal_only"), exist_ok=True)
with open(os.path.join(BASE, "results", "ablation_focal_only", "bearing.json"), 'w') as f:
    json.dump(results, f, indent=2)
print("\nDone. Saved to results/ablation_focal_only/bearing.json")
