"""
Benchmark: ResNet, FCN, PatchTST-Cls, MiniROCKET on all 4 datasets.
Same protocol as corrected_bench (seed=42, CPU, identical splits).
Saves incrementally per model per dataset.
"""
import os, sys, json, time, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SEED = 42; LR = 3e-4; WD = 1e-2; BS = 64; MAX_EP = 15; PAT = 6

DS = [
    ("ECG5000_UNBAL", "data/ecg5000_resplit.npz", 5),
    ("ECG5000_BAL",   "data/ecg5000_fair_balanced.npz", 5),
    ("CWRU_UNBAL",    "data/cwru_unbalanced.npz", 4),
    ("CWRU_BAL",      "data/cwru_balanced.npz", 4),
]

out_dir = os.path.join(ROOT, 'results', 'baseline_bench')
os.makedirs(out_dir, exist_ok=True)

device = torch.device('cpu')
print(f"Device: {device}", flush=True)

# ---- Data helpers (identical to run_all_corrected.py) ----
def znorm(X):
    mu = X.mean(axis=-1, keepdims=True)
    sig = X.std(axis=-1, keepdims=True) + 1e-8
    return ((X - mu) / sig).astype(np.float32)

# ---- Model imports ----
from models.external_baselines import ResNet1D, FCN, PatchTSTCls

# ---- FocalLoss ----
class FocalLoss(nn.Module):
    def __init__(self, gamma=1.0):
        super().__init__(); self.g = gamma
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        return ((1 - torch.exp(-ce)) ** self.g * ce).mean()

# ---- Train/eval (same as corrected_bench) ----
def train_one(model, Xtr, ytr, Xva, yva, n_cls, use_focal, label):
    criterion = FocalLoss(gamma=1.0) if use_focal else nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    tr_dl = DataLoader(TensorDataset(torch.from_numpy(Xtr).float(), torch.from_numpy(ytr).long()),
                       batch_size=BS, shuffle=True)
    va_dl = DataLoader(TensorDataset(torch.from_numpy(Xva).float(), torch.from_numpy(yva).long()),
                       batch_size=256)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, steps_per_epoch=len(tr_dl), epochs=MAX_EP)

    best_mf1 = -1; best_s = None; no_imp = 0; best_ep = 0; t0 = time.time()
    for ep in range(MAX_EP):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()

        model.eval(); vp, vt = [], []
        with torch.no_grad():
            for xb, yb in va_dl:
                xb = xb.to(device)
                logits = model(xb)
                vp.append(logits.argmax(-1).cpu().numpy()); vt.append(yb.numpy())
        vmf1 = f1_score(np.concatenate(vt), np.concatenate(vp), average='macro', zero_division=0)
        if vmf1 > best_mf1:
            best_mf1 = vmf1; best_s = copy.deepcopy(model.state_dict()); no_imp = 0; best_ep = ep + 1
        else:
            no_imp += 1
        print(f"    {label} ep{ep+1}: MF1={vmf1:.4f} (best={best_mf1:.4f}) [{time.time()-t0:.0f}s]", flush=True)
        if no_imp >= PAT:
            break
    if best_s:
        model.load_state_dict(best_s)
    return model, best_ep, time.time() - t0


def eval_one(model, Xte, yte, n_cls):
    te_dl = DataLoader(TensorDataset(torch.from_numpy(Xte).float(), torch.from_numpy(yte).long()),
                       batch_size=256)
    model.eval(); tp, tt = [], []
    with torch.no_grad():
        for xb, yb in te_dl:
            xb = xb.to(device)
            logits = model(xb)
            tp.append(logits.argmax(-1).cpu().numpy()); tt.append(yb.numpy())
    preds, true = np.concatenate(tp), np.concatenate(tt)
    per_class_f1 = f1_score(true, preds, average=None, zero_division=0, labels=list(range(n_cls)))
    per_class_prec = []
    per_class_rec = []
    for c in range(n_cls):
        tp_c = ((preds == c) & (true == c)).sum()
        fp_c = ((preds == c) & (true != c)).sum()
        fn_c = ((preds != c) & (true == c)).sum()
        prec = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
        rec = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
        per_class_prec.append(round(float(prec), 4))
        per_class_rec.append(round(float(rec), 4))
    return {
        "accuracy": round(float(accuracy_score(true, preds)), 4),
        "macro_f1": round(float(f1_score(true, preds, average='macro', zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(true, preds, average='weighted', zero_division=0)), 4),
        "class_f1s": [round(float(f), 4) for f in per_class_f1],
        "class_precision": per_class_prec,
        "class_recall": per_class_rec,
        "confusion_matrix": confusion_matrix(true, preds, labels=list(range(n_cls))).tolist(),
    }


# ---- MiniROCKET ----
def run_minirocket(Xtr, ytr, Xva, yva, Xte, yte, n_cls, label="MiniROCKET"):
    """Run MiniROCKET with RidgeClassifierCV, using validation for alpha selection."""
    from aeon.transformations.collection.convolution_based import MiniRocket
    from sklearn.linear_model import RidgeClassifierCV

    t0 = time.time()

    # Fit transformer on TRAIN only
    Xtr_3d = Xtr[:, None, :].astype(np.float32)
    Xva_3d = Xva[:, None, :].astype(np.float32)
    Xte_3d = Xte[:, None, :].astype(np.float32)

    print(f"  {label}: fitting transformer...", flush=True)
    transformer = MiniRocket(random_state=42, n_jobs=-1)
    Xtr_t = transformer.fit_transform(Xtr_3d)
    t_transform_train = time.time() - t0

    print(f"  {label}: transforming val/test...", flush=True)
    Xva_t = transformer.transform(Xva_3d)
    Xte_t = transformer.transform(Xte_3d)

    # Combine train+val for final classifier fit (validation was used for alpha in RidgeCV)
    Xtrva_t = np.concatenate([Xtr_t, Xva_t], axis=0)
    ytrva = np.concatenate([ytr, yva], axis=0)

    # RidgeCV selects alpha via CV internally
    print(f"  {label}: training classifier ({Xtr_t.shape[1]} features)...", flush=True)
    clf = RidgeClassifierCV(alphas=np.logspace(-4, 4, 20))
    clf.fit(Xtrva_t, ytrva)
    t_train = time.time() - t0

    # Evaluate
    preds = clf.predict(Xte_t)
    t_total = time.time() - t0

    acc = accuracy_score(yte, preds)
    mf1 = f1_score(yte, preds, average='macro', zero_division=0)
    wf1 = f1_score(yte, preds, average='weighted', zero_division=0)
    per_class_f1 = f1_score(yte, preds, average=None, zero_division=0, labels=list(range(n_cls)))
    per_class_prec, per_class_rec = [], []
    for c in range(n_cls):
        tp_c = ((preds == c) & (yte == c)).sum()
        fp_c = ((preds == c) & (yte != c)).sum()
        fn_c = ((preds != c) & (yte == c)).sum()
        per_class_prec.append(round(float(tp_c / (tp_c + fp_c)) if (tp_c + fp_c) > 0 else 0.0, 4))
        per_class_rec.append(round(float(tp_c / (tp_c + fn_c)) if (tp_c + fn_c) > 0 else 0.0, 4))
    cm = confusion_matrix(yte, preds, labels=list(range(n_cls))).tolist()

    print(f"  {label}: MF1={mf1:.4f} Acc={acc:.4f} ({t_total:.1f}s, {Xtr_t.shape[1]} features)", flush=True)

    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(mf1), 4),
        "weighted_f1": round(float(wf1), 4),
        "class_f1s": [round(float(f), 4) for f in per_class_f1],
        "class_precision": per_class_prec,
        "class_recall": per_class_rec,
        "confusion_matrix": cm,
        "n_features": int(Xtr_t.shape[1]),
        "time_s": round(t_total, 1),
        "transform_time_s": round(t_transform_train, 1),
    }


# ---- Configs for neural baselines ----
def get_configs(n_cls):
    return [
        ("ResNet",       lambda: ResNet1D(1, n_cls),     False, "1ch"),
        ("FCN",          lambda: FCN(1, n_cls),           False, "1ch"),
        ("PatchTST-Cls", lambda: PatchTSTCls(1, n_cls),  False, "1ch"),
    ]


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    grand_t0 = time.time()

    for ds_name, ds_file, n_cls in DS:
        print(f"\n{'='*70}\n  {ds_name} ({n_cls}cls)\n{'='*70}", flush=True)
        ds_json = os.path.join(out_dir, f'{ds_name}.json')
        if os.path.exists(ds_json):
            with open(ds_json) as f:
                results = json.load(f)
        else:
            results = {}

        # Load data (same splits as corrected_bench)
        data = np.load(os.path.join(ROOT, ds_file))
        if 'X_train' in data:
            Xa, ya = data['X_train'], data['y_train'].astype(int)
            Xte, yte = data['X_test'], data['y_test'].astype(int)
            Xtr, Xva, ytr, yva = train_test_split(Xa, ya, test_size=0.15, stratify=ya, random_state=SEED)
        else:
            Xa, ya = data['X'], data['y'].astype(int)
            Xtr, Xte, ytr, yte = train_test_split(Xa, ya, test_size=0.15, stratify=ya, random_state=SEED)
            Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=0.15, stratify=ytr, random_state=SEED)

        Xn = znorm(Xtr); Xv = znorm(Xva); Xt = znorm(Xte)
        X1tr = Xn[:, None, :]; X1va = Xv[:, None, :]; X1te = Xt[:, None, :]
        print(f"  Data ready: train={len(Xtr)} val={len(Xva)} test={len(Xte)}", flush=True)

        # Neural baselines
        configs = get_configs(n_cls)
        for label, factory, use_focal, ch_type in configs:
            if label in results and 'macro_f1' in results[label]:
                print(f"  {label}: SKIP (MF1={results[label]['macro_f1']:.4f})", flush=True)
                continue

            model = factory().to(device)
            np_ = sum(p.numel() for p in model.parameters())
            print(f"  Training {label} ({np_:,} params)...", flush=True)
            model, best_ep, ttime = train_one(model, X1tr, ytr, X1va, yva, n_cls, use_focal, label)

            r = eval_one(model, X1te, yte, n_cls)
            r["params"] = np_
            r["best_epoch"] = best_ep
            r["time_s"] = round(ttime, 1)
            results[label] = r
            with open(ds_json, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"  => {label}: MF1={r['macro_f1']:.4f} Acc={r['accuracy']:.4f} ({ttime:.0f}s)", flush=True)
            del model

        # MiniROCKET
        if "MiniROCKET" in results and 'macro_f1' in results["MiniROCKET"]:
            print(f"  MiniROCKET: SKIP (MF1={results['MiniROCKET']['macro_f1']:.4f})", flush=True)
        else:
            print(f"  Running MiniROCKET...", flush=True)
            r = run_minirocket(Xn, ytr, Xv, yva, Xt, yte, n_cls)
            results["MiniROCKET"] = r
            with open(ds_json, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"  => MiniROCKET: MF1={r['macro_f1']:.4f} Acc={r['accuracy']:.4f} ({r['time_s']:.0f}s)", flush=True)

        # Dataset summary
        print(f"\n  {ds_name} SUMMARY:", flush=True)
        print(f"  {'Model':<20} {'Params':>10} {'MF1':>6} {'Acc':>6}", flush=True)
        all_labels = [c[0] for c in configs] + ["MiniROCKET"]
        for label in all_labels:
            if label in results:
                r = results[label]
                p = r.get('params', r.get('n_features', 'N/A'))
                print(f"  {label:<20} {str(p):>10} {r['macro_f1']:>6.4f} {r['accuracy']:>6.4f}", flush=True)

    total_time = time.time() - grand_t0
    print(f"\n{'='*70}\n  TOTAL TIME: {total_time/60:.1f} min\n{'='*70}", flush=True)

    # Final summary across all datasets
    all_labels = ["ResNet", "FCN", "PatchTST-Cls", "MiniROCKET"]
    print(f"\n{'='*70}\n  FINAL BASELINE COMPARISON\n{'='*70}", flush=True)
    header = f"  {'Model':<20}"
    for ds_name, _, _ in DS:
        header += f"  {ds_name[:12]:>12}"
    header += f"  {'Avg':>6}"
    print(header, flush=True)
    print(f"  {'-'*len(header)}", flush=True)

    for label in all_labels:
        vals = []
        line = f"  {label:<20}"
        for ds_name, _, _ in DS:
            dj = os.path.join(out_dir, f'{ds_name}.json')
            if os.path.exists(dj):
                with open(dj) as f:
                    dr = json.load(f)
            else:
                dr = {}
            if label in dr:
                v = dr[label]['macro_f1']
                vals.append(v)
                line += f"  {v:>12.4f}"
            else:
                line += f"  {'---':>12}"
        if vals:
            line += f"  {np.mean(vals):>6.4f}"
        print(line, flush=True)
