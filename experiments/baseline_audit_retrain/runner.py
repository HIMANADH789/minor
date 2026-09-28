"""BASELINE AUDIT Phase 4/5: retrain the neural baselines on Haptics with an
adequate, legitimate training protocol.

Diagnosis being fixed (see AUDIT_REPORT.md):
    canonical protocol (LR 3e-4, WD 1e-2, BS 64, max 15 epochs, patience 6)
    gives 132/64 = 2-3 optimizer steps per epoch and <= 45 total steps;
    several runs select their "best" checkpoint at epoch 1 with collapsed
    single-class predictions.  Models themselves are correct.

Fix (protocol only, architectures unchanged):
    - batch 16 (more steps per epoch on 132 samples)
    - max 100 epochs, patience 25
    - small search on VALIDATION only (seed 42): LR in {3e-4, 1e-3} x
      WD in {1e-2, 1e-4}; OneCycleLR; grad clip 1.0; CE loss
    - selected config frozen, then seeds 43/44 run with it (no re-search)
    - full training history (train/val loss + macro-F1, lr, grad-norm)
    - test evaluated once per run with the best-val checkpoint

No architecture is modified; canonical implementations are imported
verbatim from models/.  Test set is never used for any decision.
"""
import copy
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from experiments.external_stack_generalization.baselines import (  # noqa: E402
    full_metrics, set_seed)
from experiments.external_stack_generalization.data import load_dataset  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "baseline_audit", "retrained_haptics")

SELECTION_SEED = 42
SEEDS = [42, 43, 44, 45, 46]
GRID = [{"lr": lr, "wd": wd} for lr in (3e-4, 1e-3) for wd in (1e-2, 1e-4)]
BS = 16
MAX_EP = 100
PAT = 25
CLIP = 1.0


def log(m):
    print(m, flush=True)


def factory(name, n_classes):
    from models.external_baselines import FCN, ResNet1D, PatchTSTCls
    from models.inceptiontime import InceptionTime
    return {
        "InceptionTime": lambda: InceptionTime(num_classes=n_classes,
                                               in_channels=1),
        "ResNet1D": lambda: ResNet1D(1, n_classes),
        "FCN": lambda: FCN(1, n_classes),
        "PatchTST": lambda: PatchTSTCls(1, n_classes),
    }[name]()


def train_one(model_name, cfg, d, device, seed, run_dir):
    """Train one run; save config/history/metrics/predictions. Returns res."""
    os.makedirs(run_dir, exist_ok=True)
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = factory(model_name, d["n_classes"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["wd"])
    from torch.utils.data import DataLoader, TensorDataset
    tr_dl = DataLoader(TensorDataset(
                       torch.from_numpy(d["Xtr_z"][:, None, :]).float(),
                       torch.from_numpy(d["ytr"]).long()),
                       batch_size=BS, shuffle=True,
                       generator=torch.Generator().manual_seed(seed))
    va_dl = DataLoader(TensorDataset(
                       torch.from_numpy(d["Xva_z"][:, None, :]).float(),
                       torch.from_numpy(d["yva"]).long()),
                       batch_size=256)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg["lr"], steps_per_epoch=len(tr_dl), epochs=MAX_EP)

    hist = {k: [] for k in ["train_loss", "train_mf1", "val_loss",
                            "val_mf1", "lr", "grad_norm"]}
    best_mf1, best_state, best_ep, no_imp = -1.0, None, 0, 0
    t0 = time.time()
    stopped = MAX_EP
    for ep in range(MAX_EP):
        model.train()
        tl, tp, ty, gn = 0.0, [], [], []
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            gn.append(float(gnorm))
            opt.step()
            sched.step()
            tl += float(loss) * len(xb)
            tp.append(model(xb).argmax(-1).cpu().numpy())
            ty.append(yb.cpu().numpy())
        from sklearn.metrics import f1_score
        tr_mf1 = f1_score(np.concatenate(ty), np.concatenate(tp),
                          average="macro", zero_division=0)
        model.eval()
        vl, vp, vt = 0.0, [], []
        with torch.no_grad():
            for xb, yb in va_dl:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                vl += float(criterion(out, yb)) * len(xb)
                vp.append(out.argmax(-1).cpu().numpy())
                vt.append(yb.cpu().numpy())
        vmf1 = f1_score(np.concatenate(vt), np.concatenate(vp),
                        average="macro", zero_division=0)
        hist["train_loss"].append(round(tl / len(tr_dl.dataset), 5))
        hist["train_mf1"].append(round(float(tr_mf1), 4))
        hist["val_loss"].append(round(vl / len(va_dl.dataset), 5))
        hist["val_mf1"].append(round(float(vmf1), 4))
        hist["lr"].append(round(sched.get_last_lr()[0], 7))
        hist["grad_norm"].append(round(float(np.mean(gn)), 4))
        if vmf1 > best_mf1:
            best_mf1, best_ep, no_imp = vmf1, ep + 1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            no_imp += 1
        log(f"    ep{ep+1}: trMF1={tr_mf1:.3f} valMF1={vmf1:.4f} "
            f"(best={best_mf1:.4f})")
        if no_imp >= PAT:
            stopped = ep + 1
            break
    train_t = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    # train-set metrics at the frozen best checkpoint (diagnostic)
    tr_dl_eval = DataLoader(TensorDataset(
        torch.from_numpy(d["Xtr_z"][:, None, :]).float(),
        torch.from_numpy(d["ytr"]).long()), batch_size=256)
    model.eval()
    tp, ty = [], []
    with torch.no_grad():
        for xb, yb in tr_dl_eval:
            tp.append(model(xb.to(device)).argmax(-1).cpu().numpy())
            ty.append(yb.numpy())
    train_res = full_metrics(np.concatenate(ty), np.concatenate(tp),
                             d["n_classes"])

    # single official test evaluation
    te_dl = DataLoader(TensorDataset(
        torch.from_numpy(d["Xte_z"][:, None, :]).float()), batch_size=256)
    preds = []
    with torch.no_grad():
        for (xb,) in te_dl:
            preds.append(model(xb.to(device)).argmax(-1).cpu().numpy())
    preds = np.concatenate(preds).astype(np.int64)
    res = full_metrics(d["yte"], preds, d["n_classes"])
    res.update({
        "model": model_name, "seed": seed, "config": cfg, "batch_size": BS,
        "max_epochs": MAX_EP, "patience": PAT, "stopped_epoch": stopped,
        "best_epoch": int(best_ep), "val_macro_f1": round(float(best_mf1), 4),
        "train_macro_f1_at_best": train_res["macro_f1"],
        "params_trainable": int(n_params),
        "time_train_s": round(train_t, 2), "device": str(device),
    })

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"model": model_name, "seed": seed, **cfg, "BS": BS,
                   "MAX_EP": MAX_EP, "PAT": PAT, "CLIP": CLIP,
                   "optimizer": "AdamW", "scheduler": "OneCycleLR",
                   "loss": "CrossEntropyLoss"}, f, indent=1)
    with open(os.path.join(run_dir, "training_history.json"), "w") as f:
        json.dump(hist, f, indent=1)
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(res, f, indent=1)
    np.save(os.path.join(run_dir, "predictions.npy"), preds)
    with open(os.path.join(run_dir, "predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "y_true", "y_pred"])
        for i, (t, p) in enumerate(zip(d["yte"], preds)):
            w.writerow([i, int(t), int(p)])
    log(f"  [{model_name} cfg={cfg} seed={seed}] val={best_mf1:.4f} "
        f"@ep{best_ep} test={res['macro_f1']:.4f} "
        f"(trainMF1={train_res['macro_f1']:.3f}, {train_t:.0f}s)")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return res


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d0 = load_dataset("Haptics")
    d = {"Xtr_z": d0["Xtr"] - d0["Xtr"].mean(-1, keepdims=True),
         "Xva_z": d0["Xva"] - d0["Xva"].mean(-1, keepdims=True),
         "Xte_z": d0["Xte"] - d0["Xte"].mean(-1, keepdims=True)}
    for k in d:
        sd = d[k].std(-1, keepdims=True) + 1e-8
        d[k] = (d[k] / sd).astype(np.float32)
    d.update({"ytr": d0["ytr"], "yva": d0["yva"], "yte": d0["yte"],
              "n_classes": d0["n_classes"]})

    all_res = []
    # ---- Phase 1: config selection on seed 42 validation only ----
    selected = {}
    for name in ["InceptionTime", "FCN", "ResNet1D", "PatchTST"]:
        best_cfg, best_val = None, -1.0
        for cfg in GRID:
            run_dir = os.path.join(OUT, name, "selection",
                                   f"lr{cfg['lr']}_wd{cfg['wd']}")
            res = train_one(name, cfg, d, device, SELECTION_SEED, run_dir)
            all_res.append(res)
            if res["val_macro_f1"] > best_val:
                best_val, best_cfg = res["val_macro_f1"], cfg
        selected[name] = best_cfg
        log(f"[SELECTED] {name}: {best_cfg} (val {best_val:.4f})")
        with open(os.path.join(OUT, f"{name}_selected_config.json"),
                  "w") as f:
            json.dump({"model": name, "selected_config": best_cfg,
                       "selection_rule": "max val macro-F1 over GRID "
                                         "(seed 42, validation only)",
                       "selection_val_macro_f1": best_val}, f, indent=1)
    with open(os.path.join(OUT, "selected_configs.json"), "w") as f:
        json.dump(selected, f, indent=1)

    # ---- Phase 2: frozen selected config, seeds 43/44 ----
    for name, cfg in selected.items():
        for seed in [s for s in SEEDS if s != SELECTION_SEED]:
            run_dir = os.path.join(OUT, name, f"seed{seed}")
            res = train_one(name, cfg, d, device, seed, run_dir)
            all_res.append(res)

    with open(os.path.join(OUT, "all_runs.json"), "w") as f:
        json.dump(all_res, f, indent=1)
    log("BASELINE AUDIT RETRAIN COMPLETE")


if __name__ == "__main__":
    main()
