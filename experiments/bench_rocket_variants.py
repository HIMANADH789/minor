"""
Benchmark: TURS-Rocket variants (rr, rtr, 3f, rcf, rs, rv, full) + late-fusion
control (TURS-Lite + MiniROCKET probability fusion).

Same protocol as corrected_bench / benchmark_baselines (seed=42, CPU, MAX_EP=15,
PAT=6, LR=3e-4, WD=1e-2, BS=64). Saves incrementally per dataset.

Outputs:
  results/rocket_variants/<DS>.json      per-variant metrics
  results/rocket_variants/probs_<DS>.npz late-fusion probabilities
  results/rocket_variants/aux_<DS>.json  test aux (alpha/uncertainty/gates)
"""
import os, sys, json, time, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
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

VARIANT_NAMES = ['rr', 'rtr', '3f', 'rcf', 'rs', 'rv', 'full']

out_dir = os.path.join(ROOT, 'results', 'rocket_variants')
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


def train_turs(model, Xtr, ytr, Xva, yva, n_cls, label):
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
            logits, aux = model(xb, return_aux=True)
            loss, _ = loss_fn(logits, yb, aux)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        model.eval(); vp, vt = [], []
        with torch.no_grad():
            for xb, yb in va_dl:
                xb = xb.to(device)
                logits, _ = model(xb, return_aux=True)
                vp.append(logits.argmax(-1).cpu().numpy()); vt.append(yb.numpy())
        vmf1 = f1_score(np.concatenate(vt), np.concatenate(vp),
                        average='macro', zero_division=0)
        if vmf1 > best_mf1:
            best_mf1 = vmf1; best_s = copy.deepcopy(model.state_dict())
            no_imp = 0; best_ep = ep + 1
        else:
            no_imp += 1
        print(f"    {label} ep{ep+1}: MF1={vmf1:.4f} (best={best_mf1:.4f}) [{time.time()-t0:.0f}s]", flush=True)
        if no_imp >= PAT:
            break
    if best_s:
        model.load_state_dict(best_s)
    return model, best_ep, time.time() - t0


def predict_probs(model, X):
    dl = DataLoader(TensorDataset(torch.from_numpy(X).float()), batch_size=256)
    model.eval(); probs = []
    with torch.no_grad():
        for (xb,) in dl:
            xb = xb.to(device)
            logits, _ = model(xb, return_aux=True)
            probs.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0)


def eval_with_aux(model, Xte, yte, n_cls):
    dl = DataLoader(TensorDataset(torch.from_numpy(Xte).float(),
                                  torch.from_numpy(yte).long()), batch_size=256)
    model.eval(); tp, tt = [], []
    aux_acc = {k: [] for k in ['alpha', 'uncertainty', 'transport_gate',
                               'regime_gate', 'rocket_gate']}
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            logits, aux = model(xb, return_aux=True)
            tp.append(logits.argmax(-1).cpu().numpy()); tt.append(yb.numpy())
            for k in aux_acc:
                v = aux.get(k)
                if v is None:
                    continue
                v = v.detach().cpu().numpy()
                if k == 'alpha':
                    aux_acc[k].append(v.mean(axis=1))  # per-sample scalar(s)
                elif k == 'uncertainty':
                    aux_acc[k].append(v.mean(axis=1))
                else:
                    aux_acc[k].append(v.reshape(v.shape[0], -1).mean(axis=1))
    preds, true = np.concatenate(tp), np.concatenate(tt)
    per_class_f1 = f1_score(true, preds, average=None, zero_division=0,
                            labels=list(range(n_cls)))
    prec, rec = [], []
    for c in range(n_cls):
        tp_c = ((preds == c) & (true == c)).sum()
        fp_c = ((preds == c) & (true != c)).sum()
        fn_c = ((preds != c) & (true == c)).sum()
        prec.append(round(float(tp_c / (tp_c + fp_c)) if (tp_c + fp_c) > 0 else 0.0, 4))
        rec.append(round(float(tp_c / (tp_c + fn_c)) if (tp_c + fn_c) > 0 else 0.0, 4))
    return {
        "accuracy": round(float(accuracy_score(true, preds)), 4),
        "macro_f1": round(float(f1_score(true, preds, average='macro', zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(true, preds, average='weighted', zero_division=0)), 4),
        "class_f1s": [round(float(f), 4) for f in per_class_f1],
        "class_precision": prec,
        "class_recall": rec,
        "confusion_matrix": confusion_matrix(true, preds, labels=list(range(n_cls))).tolist(),
    }, {k: np.concatenate(v) if v else None for k, v in aux_acc.items()}, true, preds


def run_minirocket_probs(Xtr, ytr, Xva, yva, Xte, yte):
    """Fit on train only (keeps val clean for lambda selection)."""
    from aeon.transformations.collection.convolution_based import MiniRocket
    from sklearn.linear_model import RidgeClassifierCV
    t0 = time.time()
    tr = MiniRocket(random_state=42, n_jobs=-1)
    Xtr_t = tr.fit_transform(Xtr[:, None, :].astype(np.float32))
    Xva_t = tr.transform(Xva[:, None, :].astype(np.float32))
    Xte_t = tr.transform(Xte[:, None, :].astype(np.float32))
    clf = RidgeClassifierCV(alphas=np.logspace(-4, 4, 20))
    clf.fit(Xtr_t, ytr)
    def dec2prob(D):
        D = D - D.max(axis=1, keepdims=True)
        e = np.exp(D)
        return e / e.sum(axis=1, keepdims=True)
    return (dec2prob(clf.decision_function(Xva_t)),
            dec2prob(clf.decision_function(Xte_t)),
            time.time() - t0)


def mf1_from_probs(P, y):
    return f1_score(y, P.argmax(1), average='macro', zero_division=0)


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', default='all')
    ap.add_argument('--only', default='all',
                    help="comma list: rr,rtr,3f,rcf,rs,rv,full,lite,minirocket,fusion")
    args = ap.parse_args()
    only = set(args.only.split(',')) if args.only != 'all' else None

    grand_t0 = time.time()
    ds_list = [d for d in DS if args.ds == 'all' or d[0] == args.ds]

    for ds_name, ds_file, n_cls in ds_list:
        print(f"\n{'='*70}\n  {ds_name} ({n_cls}cls)\n{'='*70}", flush=True)
        ds_json = os.path.join(out_dir, f'{ds_name}.json')
        results = json.load(open(ds_json)) if os.path.exists(ds_json) else {}

        (Xtr, ytr), (Xva, yva), (Xte, yte) = load_split(ds_file)
        X1tr, X1va, X1te = Xtr[:, None, :], Xva[:, None, :], Xte[:, None, :]
        print(f"  Data: train={len(Xtr)} val={len(Xva)} test={len(Xte)}", flush=True)

        # ---- Neural variants ----
        from models.turs_rocket import TURSRocket
        for variant in VARIANT_NAMES:
            if only is not None and variant not in only:
                continue
            key = f'TURS-{variant.upper()}'
            if key in results and 'macro_f1' in results[key]:
                print(f"  {key}: SKIP (MF1={results[key]['macro_f1']:.4f})", flush=True)
                continue
            model = TURSRocket(in_channels=1, num_classes=n_cls, regime_dim=16,
                               variant=variant).to(device)
            npar = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Training {key} ({npar:,} params)...", flush=True)
            model, best_ep, ttime = train_turs(model, X1tr, ytr, X1va, yva, n_cls, key)
            r, aux, _, _ = eval_with_aux(model, X1te, yte, n_cls)
            r["params"] = npar; r["best_epoch"] = best_ep; r["time_s"] = round(ttime, 1)
            results[key] = r
            with open(ds_json, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"  => {key}: MF1={r['macro_f1']:.4f} Acc={r['accuracy']:.4f} ({ttime:.0f}s)", flush=True)
            del model

        # ---- TURS-Lite reference (for fusion control + comparison) ----
        if only is None or 'lite' in only:
            key = 'TURS-Lite'
            if key in results and 'macro_f1' in results[key] and \
               os.path.exists(os.path.join(out_dir, f'probs_{ds_name}.npz')):
                print(f"  {key}: SKIP", flush=True)
            else:
                from models.tursnet import TURSNet
                model = TURSNet(in_channels=1, num_classes=n_cls, regime_dim=16,
                                variant='lite').to(device)
                npar = sum(p.numel() for p in model.parameters())
                print(f"  Training {key} ({npar:,} params)...", flush=True)
                model, best_ep, ttime = train_turs(model, X1tr, ytr, X1va, yva, n_cls, key)
                r, _, _, _ = eval_with_aux(model, X1te, yte, n_cls)
                r["params"] = npar; r["best_epoch"] = best_ep; r["time_s"] = round(ttime, 1)
                results[key] = r
                with open(ds_json, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"  => {key}: MF1={r['macro_f1']:.4f} Acc={r['accuracy']:.4f} ({ttime:.0f}s)", flush=True)

                # probabilities for fusion control
                p_lite_val = predict_probs(model, X1va)
                p_lite_te = predict_probs(model, X1te)
                np.savez(os.path.join(out_dir, f'probs_{ds_name}.npz'),
                         p_lite_val=p_lite_val, p_lite_te=p_lite_te,
                         y_val=yva, y_te=yte)
                del model

        # ---- MiniROCKET (for fusion control) ----
        if only is None or 'minirocket' in only or 'fusion' in only:
            pf = os.path.join(out_dir, f'probs_{ds_name}.npz')
            if os.path.exists(pf):
                z = np.load(pf)
                if 'p_mr_val' in z:
                    print("  MiniROCKET: SKIP (probs present)", flush=True)
                else:
                    p_mr_val, p_mr_te, t_mr = run_minirocket_probs(Xtr, ytr, Xva, yva, Xte, yte)
                    np.savez(pf, **dict(z), p_mr_val=p_mr_val, p_mr_te=p_mr_te)
                    print(f"  => MiniROCKET probs ({t_mr:.0f}s): valMF1={mf1_from_probs(p_mr_val, yva):.4f} "
                          f"teMF1={mf1_from_probs(p_mr_te, yte):.4f}", flush=True)

        # ---- Late fusion control (lambda on validation) ----
        if only is None or 'fusion' in only:
            pf = os.path.join(out_dir, f'probs_{ds_name}.npz')
            if os.path.exists(pf):
                z = np.load(pf)
                if 'p_mr_val' in z:
                    pv_t, pv_m = z['p_lite_val'], z['p_mr_val']
                    pt_t, pt_m = z['p_lite_te'], z['p_mr_te']
                    yv, yt = z['y_val'], z['y_te']
                    best_l, best_v, best_t = None, -1, None
                    for l in np.arange(0.0, 1.001, 0.05):
                        vmf1 = mf1_from_probs(l * pv_t + (1 - l) * pv_m, yv)
                        if vmf1 > best_v:
                            best_v, best_l = vmf1, l
                            best_t = mf1_from_probs(l * pt_t + (1 - l) * pt_m, yt)
                    print(f"  FUSION: best lambda={best_l:.2f} (val {best_v:.4f}) -> "
                          f"test MF1={best_t:.4f}", flush=True)
                    fkey = "LateFusion-TURS+MR"
                    results[fkey] = {
                        "macro_f1": round(float(best_t), 4),
                        "lambda": round(float(best_l), 3),
                        "val_mf1": round(float(best_v), 4),
                        "turs_test_mf1": round(float(mf1_from_probs(pt_t, yt)), 4),
                        "mr_test_mf1": round(float(mf1_from_probs(pt_m, yt)), 4),
                    }
                    with open(ds_json, 'w') as f:
                        json.dump(results, f, indent=2)

        # Summary
        print(f"\n  {ds_name} SUMMARY:", flush=True)
        for key in ['TURS-RR', 'TURS-RTR', 'TURS-3F', 'TURS-RCF', 'TURS-RS',
                    'TURS-RV', 'TURS-FULL', 'TURS-Lite']:
            if key in results:
                r = results[key]
                print(f"    {key:<14} MF1={r['macro_f1']:.4f} acc={r['accuracy']:.4f} "
                      f"({r['time_s']:.0f}s)", flush=True)

    print(f"\nTotal: {time.time()-grand_t0:.0f}s", flush=True)