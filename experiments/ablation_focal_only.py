"""
Ablation: Plain InceptionTime + Focal Loss (NO transport channels)
Tests whether focal loss alone — without transport augmentation — can explain
the eTAI gains. If InceptionTime+focal ≈ eTAI+focal, transport features aren't
the cause. If eTAI+focal > InceptionTime+focal, transport features are the cause.
"""
import os, sys, json, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from experiments.bearing_fair_full import ITNet, DS, FocalLoss, compute_transport


def train_eval(model, tr_loader, te_loader, dev, epochs=50, lr=3e-4,
               focal_g=0, patience=12, ckpt_path="tmp.pt"):
    crit = FocalLoss(focal_g) if focal_g > 0 else nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=epochs, steps_per_epoch=max(len(tr_loader), 1)
    )
    best_vl, pat, pci = float('inf'), 0, 0

    for ep in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            loss = crit(model(xb), yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad(); sched.step()

        model.eval(); vl, pr, tg = 0.0, [], []
        with torch.no_grad():
            for xb, yb in te_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                o = model(xb); vl += crit(o, yb).item()
                pr.extend(o.argmax(1).cpu().numpy()); tg.extend(yb.cpu().numpy())
        vl /= max(len(te_loader), 1)
        if vl < best_vl:
            best_vl, pat, pci = vl, 0, ep
            torch.save(model.state_dict(), ckpt_path)
        else:
            pat += 1
        if pat >= patience:
            break

    model.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
    model.eval()
    pr, tg = [], []
    with torch.no_grad():
        for xb, yb in te_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            o = model(xb)
            pr.extend(o.argmax(1).cpu().numpy()); tg.extend(yb.cpu().numpy())
    yp, yt = np.array(pr), np.array(tg)
    n_classes = len(np.unique(yt))
    return {
        "accuracy": round(float(accuracy_score(yt, yp)), 4),
        "macro_f1": round(float(f1_score(yt, yp, average='macro', zero_division=0)), 4),
        "recalls": [round(float(r), 4) for r in recall_score(yt, yp, average=None, zero_division=0)],
        "confusion_matrix": confusion_matrix(yt, yp).tolist(),
        "epochs_trained": pci + 1,
    }


def run_dataset(dataset_name, data_file, nc, is_bearing, dev):
    print(f"\n{'='*70}")
    print(f"  {dataset_name}")
    print(f"{'='*70}")

    data = np.load(os.path.join(BASE, "data", data_file))
    Xtr, ytr = data["X_train"].astype(np.float32), data["y_train"].astype(np.int64)
    Xte, yte = data["X_test"].astype(np.float32), data["y_test"].astype(np.int64)

    sig_len = Xte.shape[-1]
    if sig_len > 512:
        Xtr, Xte = Xtr[..., :512], Xte[..., :512]

    # Standardize
    eps = 1e-8
    mu, sig = Xtr.mean(-1, keepdims=True), Xtr.std(-1, keepdims=True)
    Xtr_s = (Xtr - mu) / (sig + eps)
    mu, sig = Xte.mean(-1, keepdims=True), Xte.std(-1, keepdims=True)
    Xte_s = (Xte - mu) / (sig + eps)

    # Subsample bearing for speed
    MAX_PER_CLASS = 200 if is_bearing else 9999
    rng = np.random.RandomState(42)
    sel = []
    for c in np.unique(ytr):
        idx = np.where(ytr == c)[0]
        if len(idx) > MAX_PER_CLASS:
            idx = rng.choice(idx, MAX_PER_CLASS, replace=False)
        sel.extend(idx)
    sel = np.array(sel)
    Xtr_s = Xtr_s[sel]; ytr_sub = ytr[sel]

    # Model params
    nb = 4 if is_bearing else 4
    ks = [5, 10, 20] if is_bearing else [5, 10, 20]

    results = {}
    focal_gammas = [0.0, 1.0, 2.0]
    loss_names = {0.0: "CE", 1.0: "focal_g1", 2.0: "focal_g2"}

    for fg in focal_gammas:
        label = loss_names[fg]
        name = f"IT {label} (1ch)"

        tr_dl = DataLoader(DS(Xtr_s, ytr_sub), 64, shuffle=True, num_workers=0)
        te_dl = DataLoader(DS(Xte_s, yte), 64, shuffle=False, num_workers=0)

        model = ITNet(nc=nc, ic=1, nb=nb, oc=32, ks=ks).to(dev)
        ckpt = os.path.join(BASE, "checkpoints", f"ablation_{dataset_name.replace(' ', '_')}_{label}_1ch.pt")
        os.makedirs(os.path.dirname(ckpt), exist_ok=True)

        t0 = time.time()
        res = train_eval(model, tr_dl, te_dl, dev, epochs=20 if is_bearing else 15,
                         focal_g=fg, patience=8, ckpt_path=ckpt)
        res["time"] = round(time.time() - t0, 1)
        results[name] = res
        print(f"  {name}: acc={res['accuracy']:.4f} f1={res['macro_f1']:.4f} "
              f"recalls={res['recalls']} ({res['time']:.0f}s)")

    return results


if __name__ == "__main__":
    dev = torch.device("cpu")
    all_results = {}

    for ds_name, ds_file, nc, is_bear in [
        ("ECG5000_UNBALANCED", "ecg5000_resplit.npz", 5, False),
        ("ECG5000_BALANCED", "ecg5000_fair_balanced.npz", 5, False),
        ("BEARING_UNBALANCED", "bearing_unbalanced.npz", 4, True),
        ("BEARING_BALANCED", "bearing_balanced.npz", 4, True),
    ]:
        all_results[ds_name] = run_dataset(ds_name, ds_file, nc, is_bear, dev)

    out = os.path.join(BASE, "results", "ablation_focal_only", "results.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out}")

    # ── Summary table ───────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  ABLATION SUMMARY: InceptionTime (1ch) — CE vs Focal — NO Transport")
    print("=" * 90)

    # Also load existing eTAI results for comparison
    eTAI_path = os.path.join(BASE, "results", "bearing_fair", "comparison.json")
    eTAI_bearing = {}
    if os.path.exists(eTAI_path):
        with open(eTAI_path) as f:
            eTAI_bearing = json.load(f)

    eTAI_ecg_path = os.path.join(BASE, "results", "fair_full", "all_results.json")
    eTAI_ecg = {}
    if os.path.exists(eTAI_ecg_path):
        with open(eTAI_ecg_path) as f:
            eTAI_ecg = json.load(f)

    for ds_key, ds_label in [
        ("ECG5000_UNBALANCED", "ECG5000 Unbalanced"),
        ("ECG5000_BALANCED", "ECG5000 Balanced"),
        ("BEARING_UNBALANCED", "Bearing Unbalanced"),
        ("BEARING_BALANCED", "Bearing Balanced"),
    ]:
        print(f"\n--- {ds_label} ---")
        print(f"  {'Model':<30} {'Acc':>6} {'MacroF1':>8}  {'Recalls'}")
        print(f"  {'-'*30} {'-'*6} {'-'*8}  {'-'*40}")

        # IT results from this ablation
        for name, res in all_results[ds_key].items():
            print(f"  {name:<30} {res['accuracy']:>6.4f} {res['macro_f1']:>8.4f}  {res['recalls']}")

        # eTAI results for comparison
        if "BEARING" in ds_key:
            split = "UNBALANCED" if "UNBALANCED" in ds_key else "BALANCED"
            if split in eTAI_bearing:
                for mname, mres in eTAI_bearing[split].items():
                    print(f"  {mname + ' (eTAI)':<30} {mres['accuracy']:>6.4f} {mres['macro_f1']:>8.4f}  {mres['class_recalls']}")
        elif "ECG" in ds_key:
            split = "UNBALANCED" if "UNBALANCED" in ds_key else "BALANCED"
            if split in eTAI_ecg:
                for mname, mres in eTAI_ecg[split].items():
                    print(f"  {mname:<30} {mres['accuracy']:>6.4f} {mres['macro_f1']:>8.4f}  {mres['recalls']}")
