"""
CRMN (Continuous Regime Manifold Network) vs Baselines
Runs on ECG5000 (unbalanced + balanced) and Bearing Fault (unbalanced + balanced).

Baselines: InceptionTime, InceptionTime+focal, eTAI CE, eTAI focal
Novel: CRMN CE, CRMN focal
Ablations: CRMN w/o dynamics, CRMN w/o prototype separation
"""
import os, sys, json, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
os.chdir(BASE)

from experiments.bearing_fair_full import ITNet, FocalLoss, compute_transport
from models.crmn import CRMN, CRMNLoss


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


def train_it(model, tr_dl, te_dl, dev, epochs, patience, focal_g, label, ckpt_dir):
    """Train InceptionTime (or eTAI) with standard CE or focal loss."""
    crit = FocalLoss(focal_g) if focal_g > 0 else nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=3e-4, epochs=epochs, steps_per_epoch=max(len(tr_dl), 1)
    )
    best_vl, pat, pci = float('inf'), 0, 0
    bp = os.path.join(ckpt_dir, f"{label}.pt")

    for ep in range(epochs):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(dev), yb.to(dev)
            loss = crit(model(xb), yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()

        model.eval(); vl = 0.0
        with torch.no_grad():
            for xb, yb in te_dl:
                xb, yb = xb.to(dev), yb.to(dev)
                vl += crit(model(xb), yb).item()
        vl /= max(len(te_dl), 1)
        if vl < best_vl: best_vl, pat, pci = vl, 0, ep; torch.save(model.state_dict(), bp)
        else: pat += 1
        if pat >= patience: break

    model.load_state_dict(torch.load(bp, weights_only=True, map_location='cpu'))
    return pci + 1


def train_crmn(model, tr_dl, te_dl, dev, epochs, patience, focal_g,
                label, ckpt_dir, loss_cfg=None):
    """Train CRMN with joint regime losses."""
    if loss_cfg is None:
        loss_cfg = {}
    crit = CRMNLoss(
        num_classes=model.num_classes,
        focal_gamma=focal_g,
        **loss_cfg
    ).to(dev)
    # Separate optimizer params: backbone faster, regime slower
    opt = torch.optim.AdamW([
        {"params": model.extractor.parameters(), "lr": 3e-4},
        {"params": list(model.ms_regime.parameters()) + list(model.regime_merge.parameters()),
         "lr": 1e-4},
        {"params": list(model.predictor.parameters()) + list(model.transition.parameters()),
         "lr": 3e-4},
    ], weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=3e-4, epochs=epochs, steps_per_epoch=max(len(tr_dl), 1)
    )
    best_vl, pat, pci = float('inf'), 0, 0
    bp = os.path.join(ckpt_dir, f"{label}.pt")

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in tr_dl:
            xb, yb = xb.to(dev), yb.to(dev)
            logits, info = model(xb)
            loss, loss_dict = crit(logits, info, yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            total_loss += loss.item()

        # Validate
        model.eval(); vl = 0.0
        with torch.no_grad():
            for xb, yb in te_dl:
                xb, yb = xb.to(dev), yb.to(dev)
                logits, info = model(xb)
                loss, _ = crit(logits, info, yb)
                vl += loss.item()
        vl /= max(len(te_dl), 1)
        if vl < best_vl: best_vl, pat, pci = vl, 0, ep; torch.save(model.state_dict(), bp)
        else: pat += 1
        if pat >= patience: break

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
    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average='macro', zero_division=0)),
        "class_recalls": [round(float(r), 4) for r in recall_score(yt, yp, average=None, zero_division=0)],
        "confusion_matrix": confusion_matrix(yt, yp).tolist(),
    }


def main():
    dev = torch.device("cpu")
    ckpt_dir = os.path.join(BASE, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    out_dir = os.path.join(BASE, "results", "crmn")
    os.makedirs(out_dir, exist_ok=True)

    DATASETS = [
        ("ECG5000_UNBAL", "ecg5000_resplit.npz", 5, False, 500),
        ("ECG5000_BAL",   "ecg5000_fair_balanced.npz", 5, False, 500),
        ("BEARING_UNBAL", "bearing_unbalanced.npz", 4, True, 200),
        ("BEARING_BAL",   "bearing_balanced.npz", 4, True, 200),
    ]

    SIG_LEN = 256
    EPOCHS = 30
    PATIENCE = 10
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    all_results = {}

    for ds_name, ds_file, nc, is_bearing, max_pc in DATASETS:
        print(f"\n{'='*80}\n  {ds_name}\n{'='*80}")

        data = np.load(os.path.join(BASE, "data", ds_file))
        Xtr_full, ytr_full = data["X_train"].astype(np.float32), data["y_train"].astype(np.int64)
        Xte, yte = data["X_test"].astype(np.float32), data["y_test"].astype(np.int64)

        if Xtr_full.shape[-1] > SIG_LEN:
            Xtr_full = Xtr_full[..., :SIG_LEN]
            Xte = Xte[..., :SIG_LEN]

        rng = np.random.RandomState(SEED)
        Xtr_sub, ytr_sub = subsample_per_class(Xtr_full, ytr_full, max_pc, rng)

        # Standardize
        mu = Xtr_sub.mean(-1, keepdims=True); sig = Xtr_sub.std(-1, keepdims=True) + 1e-8
        Xtr_s = (Xtr_sub - mu) / sig
        mu_t = Xte.mean(-1, keepdims=True); sig_t = Xte.std(-1, keepdims=True) + 1e-8
        Xte_s = (Xte - mu_t) / sig_t

        # Transport channels for eTAI
        Xtr_3 = compute_transport(Xtr_s)
        Xte_3 = compute_transport(Xte_s)

        nb = 4 if not is_bearing else 3
        ks = [5, 10, 20]
        ic_3ch = 3

        print(f"  Train: {len(ytr_sub)} | Test: {len(yte)} | Classes: {nc}")
        print(f"  Signal length: {SIG_LEN} | Blocks: {nb}")

        ds_results = {}

        # ── 1. InceptionTime (1ch, CE) ──
        tag = f"{ds_name}_IT_CE"
        tr_dl = DataLoader(DS(Xtr_s, ytr_sub), 64, shuffle=True, num_workers=0)
        te_dl = DataLoader(DS(Xte_s, yte), 64, shuffle=False, num_workers=0)
        m = ITNet(nc=nc, ic=1, nb=nb, oc=32, ks=ks).to(dev)
        t0 = time.time()
        ep = train_it(m, tr_dl, te_dl, dev, EPOCHS, PATIENCE, 0, tag, ckpt_dir)
        r = evaluate(m, te_dl, dev, nc); r["time"] = time.time()-t0; r["epochs"] = ep
        ds_results["IT CE (1ch)"] = r
        print(f"  IT CE:     F1={r['macro_f1']:.4f} Acc={r['accuracy']:.4f} ep={ep} {r['time']:.0f}s")

        # ── 2. InceptionTime (1ch, focal g=1) ──
        tag = f"{ds_name}_IT_FOCAL"
        m = ITNet(nc=nc, ic=1, nb=nb, oc=32, ks=ks).to(dev)
        t0 = time.time()
        ep = train_it(m, tr_dl, te_dl, dev, EPOCHS, PATIENCE, 1.0, tag, ckpt_dir)
        r = evaluate(m, te_dl, dev, nc); r["time"] = time.time()-t0; r["epochs"] = ep
        ds_results["IT focal g=1 (1ch)"] = r
        print(f"  IT focal:  F1={r['macro_f1']:.4f} Acc={r['accuracy']:.4f} ep={ep} {r['time']:.0f}s")

        # ── 3. eTAI CE (3ch) ──
        tag = f"{ds_name}_eTAI_CE"
        tr_dl3 = DataLoader(DS(Xtr_3, ytr_sub), 64, shuffle=True, num_workers=0)
        te_dl3 = DataLoader(DS(Xte_3, yte), 64, shuffle=False, num_workers=0)
        m = ITNet(nc=nc, ic=ic_3ch, nb=nb, oc=32, ks=ks).to(dev)
        t0 = time.time()
        ep = train_it(m, tr_dl3, te_dl3, dev, EPOCHS, PATIENCE, 0, tag, ckpt_dir)
        r = evaluate(m, te_dl3, dev, nc); r["time"] = time.time()-t0; r["epochs"] = ep
        ds_results["eTAI CE (3ch)"] = r
        print(f"  eTAI CE:   F1={r['macro_f1']:.4f} Acc={r['accuracy']:.4f} ep={ep} {r['time']:.0f}s")

        # ── 4. eTAI focal g=1 (3ch) ──
        tag = f"{ds_name}_eTAI_FOCAL"
        m = ITNet(nc=nc, ic=ic_3ch, nb=nb, oc=32, ks=ks).to(dev)
        t0 = time.time()
        ep = train_it(m, tr_dl3, te_dl3, dev, EPOCHS, PATIENCE, 1.0, tag, ckpt_dir)
        r = evaluate(m, te_dl3, dev, nc); r["time"] = time.time()-t0; r["epochs"] = ep
        ds_results["eTAI focal g=1 (3ch)"] = r
        print(f"  eTAI foc:  F1={r['macro_f1']:.4f} Acc={r['accuracy']:.4f} ep={ep} {r['time']:.0f}s")

        # ── 5. CRMN CE (1ch input) ──
        tag = f"{ds_name}_CRMN_CE"
        tr_dl1 = DataLoader(DS(Xtr_s, ytr_sub), 64, shuffle=True, num_workers=0)
        te_dl1 = DataLoader(DS(Xte_s, yte), 64, shuffle=False, num_workers=0)
        m = CRMN(in_channels=1, num_classes=nc, feat_dim=128, regime_dim=32,
                 n_experts=8, n_blocks=nb).to(dev)
        nparams = sum(p.numel() for p in m.parameters())
        t0 = time.time()
        ep = train_crmn(m, tr_dl1, te_dl1, dev, EPOCHS, PATIENCE, 0, tag, ckpt_dir)
        r = evaluate(m, te_dl1, dev, nc); r["time"] = time.time()-t0; r["epochs"] = ep
        r["params"] = nparams
        ds_results["CRMN CE (1ch)"] = r
        print(f"  CRMN CE:   F1={r['macro_f1']:.4f} Acc={r['accuracy']:.4f} ep={ep} params={nparams:,} {r['time']:.0f}s")

        # ── 6. CRMN focal g=1 (1ch input) ──
        tag = f"{ds_name}_CRMN_FOCAL"
        m = CRMN(in_channels=1, num_classes=nc, feat_dim=128, regime_dim=32,
                 n_experts=8, n_blocks=nb).to(dev)
        nparams = sum(p.numel() for p in m.parameters())
        t0 = time.time()
        ep = train_crmn(m, tr_dl1, te_dl1, dev, EPOCHS, PATIENCE, 1.0, tag, ckpt_dir)
        r = evaluate(m, te_dl1, dev, nc); r["time"] = time.time()-t0; r["epochs"] = ep
        r["params"] = nparams
        ds_results["CRMN focal g=1 (1ch)"] = r
        print(f"  CRMN foc:  F1={r['macro_f1']:.4f} Acc={r['accuracy']:.4f} ep={ep} params={nparams:,} {r['time']:.0f}s")

        # ── 7. CRMN w/o dynamics (ablation) ──
        tag = f"{ds_name}_CRMN_NODYN"
        m = CRMN(in_channels=1, num_classes=nc, feat_dim=128, regime_dim=32,
                 n_experts=8, n_blocks=nb).to(dev)
        t0 = time.time()
        ep = train_crmn(m, tr_dl1, te_dl1, dev, EPOCHS, PATIENCE, 1.0, tag, ckpt_dir,
                         loss_cfg={"lambda_dyn": 0.0})
        r = evaluate(m, te_dl1, dev, nc); r["time"] = time.time()-t0; r["epochs"] = ep
        ds_results["CRMN focal w/o dyn (1ch)"] = r
        print(f"  CRMN -dyn: F1={r['macro_f1']:.4f} Acc={r['accuracy']:.4f} ep={ep} {r['time']:.0f}s")

        # ── 8. CRMN w/o prototype (ablation) ──
        tag = f"{ds_name}_CRMN_NOPROTO"
        m = CRMN(in_channels=1, num_classes=nc, feat_dim=128, regime_dim=32,
                 n_experts=8, n_blocks=nb).to(dev)
        t0 = time.time()
        ep = train_crmn(m, tr_dl1, te_dl1, dev, EPOCHS, PATIENCE, 1.0, tag, ckpt_dir,
                         loss_cfg={"lambda_proto": 0.0})
        r = evaluate(m, te_dl1, dev, nc); r["time"] = time.time()-t0; r["epochs"] = ep
        ds_results["CRMN focal w/o proto (1ch)"] = r
        print(f"  CRMN -pr:  F1={r['macro_f1']:.4f} Acc={r['accuracy']:.4f} ep={ep} {r['time']:.0f}s")

        all_results[ds_name] = ds_results

    # ── SUMMARY TABLE ──
    print(f"\n{'='*110}")
    print("  FULL COMPARISON: ALL MODELS × ALL DATASETS")
    print(f"{'='*110}")

    for ds_name in all_results:
        print(f"\n  {ds_name}:")
        results = all_results[ds_name]
        nc = len(list(results.values())[0]["class_recalls"])
        hdr = f"  {'Model':<35} {'Acc':>7} {'MF1':>7}"
        for c in range(nc):
            hdr += f"  C{c:>4}"
        hdr += f"  {'Time':>5}"
        print(hdr)
        print(f"  {'-'*(len(hdr)-2)}")
        for name, r in results.items():
            cr = r["class_recalls"]
            line = f"  {name:<35} {r['accuracy']:>7.4f} {r['macro_f1']:>7.4f}"
            for c in range(nc):
                line += f"  {cr[c]:>5.3f}"
            line += f"  {r['time']:>4.0f}s"
            print(line)

    # Winner per dataset
    print(f"\n  WINNERS:")
    for ds_name in all_results:
        best_name = max(all_results[ds_name], key=lambda k: all_results[ds_name][k]["macro_f1"])
        best_r = all_results[ds_name][best_name]
        print(f"  {ds_name}: {best_name} (MF1={best_r['macro_f1']:.4f})")

    with open(os.path.join(out_dir, "comparison.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved to {out_dir}/comparison.json")


if __name__ == "__main__":
    main()
