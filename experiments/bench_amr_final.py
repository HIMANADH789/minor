"""
Benchmark: FINAL frozen TURS-AMR on all 4 datasets.

Same protocol as corrected_bench (seed=42, CPU, MAX_EP=15, PAT=6, LR=3e-4, WD=1e-2, BS=64).
Saves incrementally per dataset; skips completed models.

Outputs:
  results/amr_final/<DS>.json   metrics for TURS-AMR (final) on that dataset
"""
import os, sys, json, time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SEED = 42; LR = 3e-4; WD = 1e-2; BS = 64; MAX_EP = 15; PAT = 6

DS = [
    ("ECG5000_UNBAL", "data/ecg5000_resplit.npz", 5),
    ("ECG5000_BAL",   "data/ecg5000_fair_balanced.npz", 5),
    ("CWRU_UNBAL",    "data/cwru_unbalanced.npz", 4),
    ("CWRU_BAL",      "data/cwru_balanced.npz", 4),
]

out_dir = os.path.join(ROOT, 'results', 'amr_final')
os.makedirs(out_dir, exist_ok=True)

device = torch.device('cpu')
torch.manual_seed(SEED)
np.random.seed(SEED)


def znorm(X):
    mu = X.mean(axis=-1, keepdims=True)
    sig = X.std(axis=-1, keepdims=True) + 1e-8
    return ((X - mu) / sig).astype(np.float32)


def load_split(ds_file):
    data = np.load(os.path.join(ROOT, ds_file))
    if 'X_train' in data:
        Xa, ya = data['X_train'], data['y_train'].astype(int)
        Xte, yte = data['X_test'], data['y_test'].astype(int)
        Xtr, Xva, ytr, yva = train_test_split(Xa, ya, test_size=0.15,
                                               stratify=ya, random_state=SEED)
    else:
        Xa, ya = data['X'], data['y'].astype(int)
        Xtr, Xte, ytr, yte = train_test_split(Xa, ya, test_size=0.15,
                                               stratify=ya, random_state=SEED)
        Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=0.15,
                                               stratify=ytr, random_state=SEED)
    return (znorm(Xtr), ytr), (znorm(Xva), yva), (znorm(Xte), yte)


def train_final(model, Xtr, ytr, Xva, yva, n_cls, label):
    from models.tursnet import TURSLoss
    loss_fn = TURSLoss(num_classes=n_cls, focal_gamma=1.0, use_focal=False).to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(loss_fn.parameters()),
                            lr=LR, weight_decay=WD)
    tr_dl = DataLoader(TensorDataset(torch.from_numpy(Xtr).float(),
                                     torch.from_numpy(ytr).long()),
                       batch_size=BS, shuffle=True)
    va_dl = DataLoader(TensorDataset(torch.from_numpy(Xva).float(),
                                     torch.from_numpy(yva).long()),
                       batch_size=256)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR,
                                                steps_per_epoch=len(tr_dl),
                                                epochs=MAX_EP)
    best_mf1 = -1; best_s = None; no_imp = 0; best_ep = 0; t0 = time.time()
    for ep in range(MAX_EP):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits, aux = model(xb, return_aux=True)
            loss, _ = loss_fn(logits, yb, aux)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
        # validate
        model.eval()
        all_p, all_y = [], []
        with torch.no_grad():
            for xb, yb in va_dl:
                p = torch.softmax(model(xb), dim=1)
                all_p.append(p); all_y.append(yb)
        P = torch.cat(all_p).numpy(); Y = torch.cat(all_y).numpy()
        mf1 = f1_score(Y, P.argmax(1), average='macro')
        print(f"    {label} ep{ep+1}: MF1={mf1:.4f} [{time.time()-t0:.0f}s]", flush=True)
        if mf1 > best_mf1:
            best_mf1 = mf1; best_s = copy.deepcopy(model.state_dict()); best_ep = ep + 1
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= PAT:
                break
    if best_s is not None:
        model.load_state_dict(best_s)
    return model, best_ep, time.time() - t0


def evaluate(model, Xte, yte, n_cls):
    model.eval()
    with torch.no_grad():
        P = torch.softmax(model(torch.from_numpy(Xte).float()), dim=1).numpy()
    yhat = P.argmax(1)
    return {
        "macro_f1": round(float(f1_score(yte, yhat, average='macro')), 4),
        "accuracy": round(float(accuracy_score(yte, yhat)), 4),
        "weighted_f1": round(float(f1_score(yte, yhat, average='weighted')), 4),
        "class_f1s": [round(float(x), 4) for x in f1_score(yte, yhat, average=None)],
        "class_recall": [round(float(x), 4) for x in
                         f1_score(yte, yhat, average=None, zero_division=0)],
        "confusion_matrix": confusion_matrix(yte, yhat).tolist(),
        "probs": P.tolist(),
    }, P


if __name__ == '__main__':
    import argparse
    import copy
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', default='all')
    args = ap.parse_args()
    ds_list = [d for d in DS if args.ds == 'all' or d[0] == args.ds]

    grand_t0 = time.time()
    for ds_name, ds_file, n_cls in ds_list:
        print(f"\n{'='*70}\n  {ds_name} ({n_cls}cls)\n{'='*70}", flush=True)
        ds_json = os.path.join(out_dir, f'{ds_name}.json')
        results = json.load(open(ds_json)) if os.path.exists(ds_json) else {}

        if 'TURS-AMR-Final' in results and 'macro_f1' in results['TURS-AMR-Final']:
            print(f"  TURS-AMR-Final: SKIP (MF1={results['TURS-AMR-Final']['macro_f1']:.4f})",
                  flush=True)
            continue

        (Xtr, ytr), (Xva, yva), (Xte, yte) = load_split(ds_file)
        X1tr, X1va, X1te = Xtr[:, None, :], Xva[:, None, :], Xte[:, None, :]
        print(f"  Data: train={len(Xtr)} val={len(Xva)} test={len(Xte)}", flush=True)

        from models.turs_amr_final import TURSAMRFinal
        model = TURSAMRFinal(in_channels=1, num_classes=n_cls, regime_dim=16).to(device)
        npar = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Training TURS-AMR-Final ({npar:,} params)...", flush=True)
        model, best_ep, ttime = train_final(model, X1tr, ytr, X1va, yva, n_cls,
                                             "TURS-AMR-Final")
        r, P = evaluate(model, X1te, yte, n_cls)
        r["params"] = npar; r["best_epoch"] = best_ep; r["time_s"] = round(ttime, 1)
        results['TURS-AMR-Final'] = r
        with open(ds_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  => TURS-AMR-Final: MF1={r['macro_f1']:.4f} Acc={r['accuracy']:.4f} "
              f"({ttime:.0f}s)", flush=True)
        np.savez(os.path.join(out_dir, f'probs_{ds_name}.npz'),
                 probs=P, y=yte)
        del model

    print(f"\nTotal: {time.time()-grand_t0:.0f}s", flush=True)
