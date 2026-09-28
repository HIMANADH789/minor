"""
Run all 4 datasets sequentially, saving results incrementally.
Uses CPU for small models (faster than GPU on RTX 4050 Laptop).
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

out_dir = os.path.join(ROOT, 'results', 'corrected_bench')
os.makedirs(out_dir, exist_ok=True)

device = torch.device('cpu')  # CPU is faster for small models on RTX 4050 Laptop
print(f"Device: {device}", flush=True)

# ---- Data helpers ----
def znorm(X):
    mu = X.mean(axis=-1, keepdims=True)
    sig = X.std(axis=-1, keepdims=True) + 1e-8
    return ((X - mu) / sig).astype(np.float32)

def transport_3ch(X):
    N, L = X.shape
    out = np.zeros((N, 3, L), dtype=np.float32)
    out[:, 0, :] = X
    for i in range(N):
        s = np.sort(X[i])
        out[i, 1, :] = np.interp(np.linspace(0, 1, L), np.linspace(0, 1, len(s)), s)
        d = np.zeros(L)
        for lag in [1, 2, 4]:
            diff = np.diff(s, n=lag)
            d[lag:] += np.abs(diff) / 3.0
        out[i, 2, :] = d
    for c in range(3):
        mu = out[:, c, :].mean(axis=-1, keepdims=True)
        sig = out[:, c, :].std(axis=-1, keepdims=True) + 1e-8
        out[:, c, :] = (out[:, c, :] - mu) / sig
    return out

# ---- Model defs (CORRECT backbone: 135K/170K) ----
class InceptionMod(nn.Module):
    def __init__(self, ic, oc, b=32, ks=[5, 10, 20]):
        super().__init__()
        self.bn_layer = nn.Conv1d(ic, b, 1, bias=False) if ic > 1 else nn.Identity()
        ic2 = b if ic > 1 else 1
        self.convs = nn.ModuleList([nn.Conv1d(ic2, oc, k, padding='same', bias=False) for k in ks])
        self.mp = nn.MaxPool1d(3, 1, 1)
        self.pc = nn.Conv1d(ic, oc, 1, bias=False)
        self.norm = nn.BatchNorm1d(oc * len(ks) + oc)
        self.relu = nn.ReLU()
    def forward(self, x):
        bx = self.bn_layer(x)
        return self.relu(self.norm(torch.cat([c(bx) for c in self.convs] + [self.pc(self.mp(x))], 1)))

class ITNet(nn.Module):
    def __init__(self, ic=1, nc=5, nb=4, oc=32):
        super().__init__()
        self.blks = nn.ModuleList(); self.shortcuts = nn.ModuleList(); ci = ic
        for i in range(nb):
            self.blks.append(InceptionMod(ci, oc, 32, [5,10,20]))
            co = oc * 3 + oc
            if i % 3 == 2:
                sc_ic = ic if i == 2 else co
                self.shortcuts.append(nn.Sequential(nn.Conv1d(sc_ic, co, 1, bias=False), nn.BatchNorm1d(co)) if sc_ic != co else nn.Identity())
            ci = co
        self.gap = nn.AdaptiveAvgPool1d(1); self.fc = nn.Linear(ci, nc)
    def forward(self, x):
        si_i, si = x, 0
        for i, b in enumerate(self.blks):
            x = b(x)
            if i % 3 == 2:
                x = torch.relu(x + self.shortcuts[si](si_i))
                si_i, si = x, si + 1
        return self.fc(self.gap(x).squeeze(-1))

class FocalLoss(nn.Module):
    def __init__(self, gamma=1.0):
        super().__init__(); self.g = gamma
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        return ((1 - torch.exp(-ce)) ** self.g * ce).mean()

def get_configs(n_cls):
    from models.tursnet import TURSNet
    from models.ustrnet import USTRNet
    return [
        ("InceptionTime",   lambda: ITNet(ic=1, nc=n_cls), False, False, "1ch"),
        ("eTAI-Focal",      lambda: ITNet(ic=3, nc=n_cls), True,  False, "3ch"),
        ("USTR-Net-CE",     lambda: USTRNet(in_channels=1, num_classes=n_cls), False, True,  "1ch"),
        ("USTR-Net-Focal",  lambda: USTRNet(in_channels=1, num_classes=n_cls), True,  True,  "1ch"),
        ("TURS-Lite",       lambda: TURSNet(in_channels=1, num_classes=n_cls, regime_dim=16, variant="lite"), False, False, "1ch"),
        ("TURS-Strong",     lambda: TURSNet(in_channels=1, num_classes=n_cls, regime_dim=24, variant="strong", multi_scale_fusion=True), False, False, "1ch"),
    ]

# ---- Train/eval ----
def train_one(model, Xtr, ytr, Xva, yva, n_cls, use_focal, is_dict, label):
    is_turs = hasattr(model, 'use_transport') and hasattr(model, 'use_regime')
    turs_loss_fn = None
    if is_turs:
        from models.tursnet import TURSLoss
        turs_loss_fn = TURSLoss(num_classes=n_cls, focal_gamma=1.0, use_focal=False).to(device)

    criterion = FocalLoss(gamma=1.0) if use_focal else nn.CrossEntropyLoss()
    params = list(model.parameters())
    if turs_loss_fn: params += list(turs_loss_fn.parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    tr_dl = DataLoader(TensorDataset(torch.from_numpy(Xtr).float(), torch.from_numpy(ytr).long()), batch_size=BS, shuffle=True)
    va_dl = DataLoader(TensorDataset(torch.from_numpy(Xva).float(), torch.from_numpy(yva).long()), batch_size=256)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, steps_per_epoch=len(tr_dl), epochs=MAX_EP)

    best_mf1=-1; best_s=None; no_imp=0; best_ep=0; t0=time.time()
    for ep in range(MAX_EP):
        model.train()
        for xb,yb in tr_dl:
            xb,yb=xb.to(device),yb.to(device)
            if is_turs:
                logits,aux=model(xb,return_aux=True)
                loss,_=turs_loss_fn(logits,yb,aux)
            elif is_dict:
                out=model(xb); logits=out['logits'] if isinstance(out,dict) else out
                loss=criterion(logits,yb)
            else:
                logits=model(xb); loss=criterion(logits,yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        model.eval(); vp,vt=[],[]
        with torch.no_grad():
            for xb,yb in va_dl:
                xb=xb.to(device)
                if is_turs: logits,_=model(xb,return_aux=True)
                elif is_dict:
                    out=model(xb); logits=out['logits'] if isinstance(out,dict) else out
                else: logits=model(xb)
                vp.append(logits.argmax(-1).cpu().numpy()); vt.append(yb.numpy())
        vmf1=f1_score(np.concatenate(vt),np.concatenate(vp),average='macro',zero_division=0)
        if vmf1>best_mf1: best_mf1=vmf1; best_s=copy.deepcopy(model.state_dict()); no_imp=0; best_ep=ep+1
        else: no_imp+=1
        print(f"    {label} ep{ep+1}: MF1={vmf1:.4f} (best={best_mf1:.4f}) [{time.time()-t0:.0f}s]", flush=True)
        if no_imp>=PAT: break
    if best_s: model.load_state_dict(best_s)
    return model, best_ep, time.time()-t0

def eval_one(model, Xte, yte, n_cls, is_dict, is_turs):
    te_dl = DataLoader(TensorDataset(torch.from_numpy(Xte).float(), torch.from_numpy(yte).long()), batch_size=256)
    model.eval(); tp,tt=[],[]
    with torch.no_grad():
        for xb,yb in te_dl:
            xb=xb.to(device)
            if is_turs: logits,_=model(xb,return_aux=True)
            elif is_dict:
                out=model(xb); logits=out['logits'] if isinstance(out,dict) else out
            else: logits=model(xb)
            tp.append(logits.argmax(-1).cpu().numpy()); tt.append(yb.numpy())
    preds,true=np.concatenate(tp),np.concatenate(tt)
    return {
        "accuracy": round(float(accuracy_score(true,preds)),4),
        "macro_f1": round(float(f1_score(true,preds,average='macro',zero_division=0)),4),
        "weighted_f1": round(float(f1_score(true,preds,average='weighted',zero_division=0)),4),
        "class_f1s": [round(float(f),4) for f in f1_score(true,preds,average=None,zero_division=0,labels=list(range(n_cls)))],
        "confusion_matrix": confusion_matrix(true,preds,labels=list(range(n_cls))).tolist(),
    }

# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    grand_t0 = time.time()

    for ds_name, ds_file, n_cls in DS:
        print(f"\n{'='*70}\n  {ds_name} ({n_cls}cls)\n{'='*70}", flush=True)
        ds_json = os.path.join(out_dir, f'{ds_name}.json')
        if os.path.exists(ds_json):
            with open(ds_json) as f: results = json.load(f)
        else: results = {}

        data = np.load(os.path.join(ROOT, ds_file))
        if 'X_train' in data:
            Xa,ya = data['X_train'], data['y_train'].astype(int)
            Xte,yte = data['X_test'], data['y_test'].astype(int)
            Xtr,Xva,ytr,yva = train_test_split(Xa, ya, test_size=0.15, stratify=ya, random_state=SEED)
        else:
            Xa,ya = data['X'], data['y'].astype(int)
            Xtr,Xte,ytr,yte = train_test_split(Xa, ya, test_size=0.15, stratify=ya, random_state=SEED)
            Xtr,Xva,ytr,yva = train_test_split(Xtr, ytr, test_size=0.15, stratify=ytr, random_state=SEED)

        Xn = znorm(Xtr); Xv = znorm(Xva); Xt = znorm(Xte)
        X1tr=Xn[:,None,:]; X1va=Xv[:,None,:]; X1te=Xt[:,None,:]
        print(f"  Computing transport channels...", flush=True)
        X3tr=transport_3ch(Xn); X3va=transport_3ch(Xv); X3te=transport_3ch(Xt)
        print(f"  Data ready: train={len(Xtr)} val={len(Xva)} test={len(Xte)}", flush=True)

        configs = get_configs(n_cls)
        for label, factory, use_focal, is_dict, ch_type in configs:
            if label in results and 'macro_f1' in results[label]:
                print(f"  {label}: SKIP (MF1={results[label]['macro_f1']:.4f})", flush=True)
                continue

            model = factory().to(device)
            np_ = sum(p.numel() for p in model.parameters())
            is_turs = hasattr(model, 'use_transport') and hasattr(model, 'use_regime')

            if ch_type == "1ch": Xtr_c, Xva_c, Xte_c = X1tr, X1va, X1te
            else: Xtr_c, Xva_c, Xte_c = X3tr, X3va, X3te

            print(f"  Training {label} ({np_:,} params)...", flush=True)
            model, best_ep, ttime = train_one(model, Xtr_c, ytr, Xva_c, yva, n_cls, use_focal, is_dict, label)

            r = eval_one(model, Xte_c, yte, n_cls, is_dict, is_turs)
            r["params"] = np_; r["best_epoch"] = best_ep; r["time_s"] = round(ttime,1)
            results[label] = r
            with open(ds_json, 'w') as f: json.dump(results, f, indent=2)
            print(f"  => {label}: MF1={r['macro_f1']:.4f} Acc={r['accuracy']:.4f} ({ttime:.0f}s)", flush=True)
            del model

        # Dataset summary
        print(f"\n  {ds_name} SUMMARY:", flush=True)
        print(f"  {'Model':<20} {'Params':>8} {'MF1':>6} {'Acc':>6}", flush=True)
        for label,_,_,_,_ in configs:
            if label in results:
                r = results[label]
                print(f"  {label:<20} {r['params']:>8,} {r['macro_f1']:>6.4f} {r['accuracy']:>6.4f}", flush=True)

    total_time = time.time() - grand_t0
    print(f"\n{'='*70}\n  TOTAL TIME: {total_time/60:.1f} min\n{'='*70}", flush=True)

    # Final summary
    print(f"\n{'='*70}\n  FINAL CORRECTED COMPARISON\n{'='*70}", flush=True)
    header = f"  {'Model':<20} {'Params':>8}"
    for ds_name,_,_ in DS:
        header += f"  {ds_name[:10]:>10}"
    header += f"  {'Avg':>6}"
    print(header, flush=True)
    print(f"  {'-'*len(header)}", flush=True)
    for label in ["InceptionTime","eTAI-Focal","USTR-Net-CE","USTR-Net-Focal","TURS-Lite","TURS-Strong"]:
        vals = []
        line = f"  {label:<20}"
        # get params from first available dataset
        for ds_name,_,_ in DS:
            dj = os.path.join(out_dir, f'{ds_name}.json')
            if os.path.exists(dj):
                with open(dj) as f: dr = json.load(f)
                if label in dr:
                    line += f"  {dr[label]['params']:>10,}"
                    break
        else:
            line += f"  {'?':>10}"
        for ds_name,_,_ in DS:
            dj = os.path.join(out_dir, f'{ds_name}.json')
            if os.path.exists(dj):
                with open(dj) as f: dr = json.load(f)
            else: dr = {}
            if label in dr:
                v = dr[label]['macro_f1']; vals.append(v)
                line += f"  {v:>10.4f}"
            else:
                line += f"  {'---':>10}"
        if vals: line += f"  {np.mean(vals):>6.4f}"
        print(line, flush=True)
