"""
ECG5000 Fair Comparison: Unbalanced vs Balanced
Runs InceptionTime (1ch) + eTAI CE (3ch) + eTAI focal (3ch) on both dataset versions.
Optimized for speed: 140-sample signals are fast.
"""

import os, sys, json, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from experiments.bearing_fair_full import ITNet, DS, FocalLoss, compute_transport

def main():
    out_dir = os.path.join(BASE, "results", "ecg_fair")
    os.makedirs(out_dir, exist_ok=True)
    dev = torch.device("cpu")

    versions = [
        ("UNBALANCED", "ecg5000_resplit.npz"),
        ("BALANCED", "ecg5000_fair_balanced.npz"),
    ]

    all_results = {}

    for vname, vfile in versions:
        print(f"\n{'='*70}")
        print(f"  ECG5000 {vname}")
        print(f"{'='*70}")

        data = np.load(os.path.join(BASE, "data", vfile))
        Xtr, ytr = data["X_train"], data["y_train"].astype(np.int64)
        Xte, yte = data["X_test"], data["y_test"].astype(np.int64)

        nc = len(np.unique(yte))
        cc = dict(zip(*np.unique(ytr, return_counts=True)))
        print(f"  Train: {len(Xtr)} | Test: {len(Xte)} | Classes: {nc} | Dist: {cc}")

        # Subsample per class for CPU speed (cap at 500 per class)
        MAX_PER_CLASS = 500
        rng = np.random.RandomState(42)
        sel = []
        for c in np.unique(ytr):
            idx = np.where(ytr == c)[0]
            if len(idx) > MAX_PER_CLASS:
                idx = rng.choice(idx, MAX_PER_CLASS, replace=False)
            sel.extend(idx)
        sel = np.array(sel)
        Xtr, ytr = Xtr[sel], ytr[sel]
        cc2 = dict(zip(*np.unique(ytr, return_counts=True)))
        print(f"  After subsample: {len(Xtr)} | Dist: {cc2}")

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
            ("InceptionTime (1ch)", 1, Xtr_s, Xte_s, 0.0),
            ("eTAI CE (3ch)", 3, Xtr_3, Xte_3, 0.0),
            ("eTAI focal g=1 (3ch)", 3, Xtr_3, Xte_3, 1.0),
            ("eTAI focal g=2 (3ch)", 3, Xtr_3, Xte_3, 2.0),
        ]

        for name, ic, Xtr_c, Xte_c, fg in configs:
            print(f"\n  Training: {name}")
            tr_dl = DataLoader(DS(Xtr_c, ytr), 64, shuffle=True, num_workers=0)
            te_dl = DataLoader(DS(Xte_c, yte), 64, shuffle=False, num_workers=0)

            model = ITNet(nc=nc, ic=ic, nb=4, oc=32, ks=[5, 10, 20]).to(dev)
            np_ = sum(p.numel() for p in model.parameters())
            print(f"    Params: {np_:,}")

            t0 = time.time()
            # Inline training for ECG5000 (fast, 140 samples)
            crit = FocalLoss(fg) if fg > 0 else nn.CrossEntropyLoss()
            opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, epochs=25,
                                                          steps_per_epoch=max(len(tr_dl), 1))
            best_vl, pat, pci = float('inf'), 0, 0
            bp = os.path.join(BASE, "checkpoints", f"ecg_{vname}_{name.replace(' ','_').replace('(','').replace(')','')}.pt")
            os.makedirs(os.path.dirname(bp), exist_ok=True)

            for ep in range(25):
                model.train()
                for xb, yb in tr_dl:
                    xb, yb = xb.to(dev), yb.to(dev)
                    loss = crit(model(xb), yb)
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step(); sched.step()

                model.eval(); vl, pr, tg = 0.0, [], []
                with torch.no_grad():
                    for xb, yb in te_dl:
                        xb, yb = xb.to(dev), yb.to(dev)
                        o = model(xb); vl += crit(o, yb).item()
                        pr.extend(o.argmax(1).cpu().numpy()); tg.extend(yb.cpu().numpy())
                vl /= max(len(te_dl), 1)
                if vl < best_vl:
                    best_vl, pat, pci = vl, 0, ep
                    torch.save(model.state_dict(), bp)
                else:
                    pat += 1
                if pat >= 8:
                    break

            model.load_state_dict(torch.load(bp, weights_only=True)); model.eval()
            pr, tg = [], []
            with torch.no_grad():
                for xb, yb in te_dl:
                    xb, yb = xb.to(dev), yb.to(dev)
                    o = model(xb)
                    pr.extend(o.argmax(1).cpu().numpy()); tg.extend(yb.cpu().numpy())
            yp, yt = np.array(pr), np.array(tg)

            r = {
                "accuracy": float(accuracy_score(yt, yp)),
                "macro_f1": float(f1_score(yt, yp, average='macro', zero_division=0)),
                "class_recalls": [float(x) for x in recall_score(yt, yp, average=None, zero_division=0)],
                "n_classes": int(nc),
                "confusion_matrix": confusion_matrix(yt, yp).tolist(),
                "epochs_trained": pci + 1,
            }
            r["time"] = time.time() - t0
            cr = r["class_recalls"]
            print(f"    Acc={r['accuracy']:.4f} | MF1={r['macro_f1']:.4f} | "
                  f"Recalls={[f'{x:.3f}' for x in cr]} | {r['time']:.0f}s | ep={pci+1}")
            results[name] = r

        all_results[vname] = results

    # Summary
    print(f"\n{'='*90}")
    print("  ECG5000 FAIR COMPARISON")
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
    print(f"\n  Cross-Version:")
    print(f"  {'Model':<30} {'Unbal MF1':>10} {'Bal MF1':>10} {'dF1':>8}")
    print(f"  {'-'*68}")
    for name in all_results["UNBALANCED"]:
        ru = all_results["UNBALANCED"][name]
        rb = all_results["BALANCED"][name]
        print(f"  {name:<30} {ru['macro_f1']:>10.4f} {rb['macro_f1']:>10.4f} {rb['macro_f1']-ru['macro_f1']:>+8.4f}")

    with open(os.path.join(out_dir, "comparison.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {out_dir}/comparison.json")


if __name__ == "__main__":
    main()
