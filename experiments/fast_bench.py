"""
Fast single-dataset benchmark. Run per-dataset:
  python fast_bench.py ECG5000_UNBAL
  python fast_bench.py ECG5000_BAL
  python fast_bench.py CWRU_UNBAL
  python fast_bench.py CWRU_BAL
"""
import os, sys, json, time, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SEED=42; LR=3e-4; WD=1e-2; BS=64; MAX_EP=20; PAT=6

DS = {
    "ECG5000_UNBAL": ("data/ecg5000_resplit.npz", 5),
    "ECG5000_BAL":   ("data/ecg5000_fair_balanced.npz", 5),
    "CWRU_UNBAL":    ("data/cwru_unbalanced.npz", 4),
    "CWRU_BAL":      ("data/cwru_balanced.npz", 4),
}

ds_name = sys.argv[1]
assert ds_name in DS, f"Unknown: {ds_name}. Options: {list(DS.keys())}"
ds_file, n_cls = DS[ds_name]

# --- Load ---
data = np.load(os.path.join(ROOT, ds_file))
if 'X_train' in data:
    Xa, ya = data['X_train'], data['y_train'].astype(int)
    Xte, yte = data['X_test'], data['y_test'].astype(int)
    Xtr, Xva, ytr, yva = train_test_split(Xa, ya, test_size=0.15, stratify=ya, random_state=SEED)
else:
    Xa, ya = data['X'], data['y'].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(Xa, ya, test_size=0.15, stratify=ya, random_state=SEED)
    Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=0.15, stratify=ytr, random_state=SEED)

def znorm(X):
    mu=X.mean(axis=-1,keepdims=True); sig=X.std(axis=-1,keepdims=True)+1e-8
    return ((X-mu)/sig).astype(np.float32)
Xn_tr, Xn_va, Xn_te = znorm(Xtr), znorm(Xva), znorm(Xte)

def transport_3ch(X):
    N,L=X.shape; out=np.zeros((N,3,L),dtype=np.float32)
    out[:,0,:]=X
    for i in range(N):
        s=np.sort(X[i]); out[i,1,:]=np.interp(np.linspace(0,1,L),np.linspace(0,1,len(s)),s)
        d=np.zeros(L)
        for lag in [1,2,4]:
            diff=np.diff(s,n=lag); d[lag:]+=np.abs(diff)/3.0
        out[i,2,:]=d
    for c in range(3):
        mu=out[:,c,:].mean(axis=-1,keepdims=True); sig=out[:,c,:].std(axis=-1,keepdims=True)+1e-8
        out[:,c,:]=(out[:,c,:]-mu)/sig
    return out

Xtr1=Xn_tr[:,None,:]; Xva1=Xn_va[:,None,:]; Xte1=Xn_te[:,None,:]
Xtr3=transport_3ch(Xn_tr); Xva3=transport_3ch(Xn_va); Xte3=transport_3ch(Xn_te)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device} | {ds_name} ({n_cls}cls) train={len(Xtr)} test={len(Xte)}")

# --- Efficient ITNet using same InceptionBlock as TURS-Net ---
class InceptionBlock(nn.Module):
    def __init__(s, in_ch, out_ch):
        super().__init__()
        bc = in_ch // 2
        s.bn = nn.Sequential(nn.Conv1d(in_ch,bc,1,bias=False),nn.BatchNorm1d(bc),nn.ReLU(inplace=True))
        br = out_ch // 4; rem = out_ch - 3*br
        s.convs = nn.ModuleList([nn.Conv1d(bc, br, k, padding=k//2, bias=False) for k in [9,19,39]])
        s.pool = nn.Sequential(nn.MaxPool1d(3,1,1), nn.Conv1d(in_ch,rem,1,bias=False), nn.BatchNorm1d(rem), nn.ReLU(inplace=True))
        s.obn = nn.BatchNorm1d(out_ch)
        s.res = nn.Sequential(nn.Conv1d(in_ch,out_ch,1,bias=False),nn.BatchNorm1d(out_ch)) if in_ch!=out_ch else nn.Identity()
        s.relu = nn.ReLU(inplace=True)
    def forward(s,x):
        b=s.bn(x); return s.relu(s.obn(torch.cat([c(b) for c in s.convs]+[s.pool(x)],1))+s.res(x))

class ITNet(nn.Module):
    def __init__(s,ic=1,nc=5,nb=4,oc=32,ks=None):
        super().__init__()
        s.proj = nn.Sequential(nn.Conv1d(ic,oc,1,bias=False),nn.BatchNorm1d(oc),nn.ReLU(inplace=True))
        s.blks = nn.ModuleList()
        chs = [oc, oc, oc*2, oc*2]
        for i in range(nb): s.blks.append(InceptionBlock(chs[i-1] if i>0 else oc, chs[i]))
        s.gap = nn.AdaptiveAvgPool1d(1); s.fc = nn.Linear(chs[-1],nc)
    def forward(s,x):
        x=s.proj(x)
        for b in s.blks: x=b(x)
        return s.fc(s.gap(x).squeeze(-1))

def train_eval(model, Xtr_c, Xva_c, Xte_c, tag, use_focal=False, is_dict=False):
    tr_dl=DataLoader(TensorDataset(torch.from_numpy(Xtr_c).float(),torch.from_numpy(ytr).long()),batch_size=BS,shuffle=True)
    va_dl=DataLoader(TensorDataset(torch.from_numpy(Xva_c).float(),torch.from_numpy(yva).long()),batch_size=256)
    te_dl=DataLoader(TensorDataset(torch.from_numpy(Xte_c).float(),torch.from_numpy(yte).long()),batch_size=256)
    np_=sum(p.numel() for p in model.parameters())

    # Loss
    turs_loss_fn = None
    is_turs = hasattr(model, 'use_transport')
    if is_turs:
        from models.tursnet import TURSLoss
        turs_loss_fn = TURSLoss(num_classes=n_cls, focal_gamma=1.0, use_focal=False).to(device)
        params = list(model.parameters()) + list(turs_loss_fn.parameters())
    else:
        criterion = FocalLoss(gamma=1.0) if use_focal else nn.CrossEntropyLoss()
        params = model.parameters()

    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, steps_per_epoch=len(tr_dl), epochs=MAX_EP)
    best_mf1=-1; best_s=None; no_imp=0; t0=time.time(); best_ep=0

    for ep in range(MAX_EP):
        model.train()
        for xb,yb in tr_dl:
            xb,yb=xb.to(device),yb.to(device)
            if is_turs:
                logits,aux=model(xb,return_aux=True)
                loss,_=turs_loss_fn(logits,yb,aux)
            elif is_dict:
                out=model(xb); logits=out['logits']; loss=criterion(logits,yb)
            else:
                logits=model(xb); loss=criterion(logits,yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sched.step()
        model.eval(); vp,vt=[],[]
        with torch.no_grad():
            for xb,yb in va_dl:
                xb=xb.to(device)
                if is_turs: logits,_=model(xb,return_aux=True)
                elif is_dict: out=model(xb); logits=out['logits']
                else: logits=model(xb)
                vp.append(logits.argmax(-1).cpu().numpy()); vt.append(yb.numpy())
        vmf1=f1_score(np.concatenate(vt),np.concatenate(vp),average='macro',zero_division=0)
        if vmf1>best_mf1: best_mf1=vmf1; best_s=copy.deepcopy(model.state_dict()); no_imp=0; best_ep=ep+1
        else: no_imp+=1
        if no_imp>=PAT: break
    elapsed=time.time()-t0
    if best_s: model.load_state_dict(best_s)
    model.eval(); tp,tt=[],[]; all_alpha=[]; all_u=[]
    with torch.no_grad():
        for xb,yb in te_dl:
            xb=xb.to(device)
            if is_turs:
                logits,aux=model(xb,return_aux=True)
                if aux.get('alpha') is not None: all_alpha.append(aux['alpha'].cpu().numpy())
                if aux.get('uncertainty') is not None: all_u.append(aux['uncertainty'].cpu().numpy())
            elif is_dict: out=model(xb); logits=out['logits']
            else: logits=model(xb)
            tp.append(logits.argmax(-1).cpu().numpy()); tt.append(yb.numpy())
    preds,true=np.concatenate(tp),np.concatenate(tt)
    r={
        "accuracy":round(float(accuracy_score(true,preds)),4),
        "macro_f1":round(float(f1_score(true,preds,average='macro',zero_division=0)),4),
        "weighted_f1":round(float(f1_score(true,preds,average='weighted',zero_division=0)),4),
        "class_f1s":[round(float(f),4) for f in f1_score(true,preds,average=None,zero_division=0,labels=list(range(n_cls)))],
        "confusion_matrix":confusion_matrix(true,preds,labels=list(range(n_cls))).tolist(),
        "params":np_, "best_epoch":best_ep, "time_s":round(elapsed,1),
    }
    if all_alpha:
        a=np.concatenate(all_alpha)
        r["gate_stats"]={"alpha_mean":round(float(a.mean()),4),"uncertainty_mean":round(float(np.concatenate(all_u).mean()),4) if all_u else None}
    return r

# FocalLoss
class FocalLoss(nn.Module):
    def __init__(s,gamma=1.0): super().__init__(); s.g=gamma
    def forward(s,l,t):
        ce=F.cross_entropy(l,t,reduction='none'); pt=torch.exp(-ce)
        return ((1-pt)**s.g*ce).mean()

# --- Run all models ---
out_dir=os.path.join(ROOT,'results','turs_benchmark')
os.makedirs(out_dir,exist_ok=True)
ds_json=os.path.join(out_dir,f'{ds_name}.json')
if os.path.exists(ds_json):
    with open(ds_json) as f: results=json.load(f)
else: results={}

from models.tursnet import TURSNet
from models.ustrnet import USTRNet

configs=[
    ("InceptionTime", lambda: ITNet(ic=1,nc=n_cls,nb=4,oc=32,ks=[5,10,20]), Xtr1,Xva1,Xte1,False,False),
    ("eTAI-Focal", lambda: ITNet(ic=3,nc=n_cls,nb=4,oc=32,ks=[5,10,20]), Xtr3,Xva3,Xte3,True,False),
    ("USTR-Net-CE", lambda: USTRNet(in_channels=1,num_classes=n_cls,regime_dim=8,base_ch=32), Xtr1,Xva1,Xte1,False,True),
    ("USTR-Net-Focal", lambda: USTRNet(in_channels=1,num_classes=n_cls,regime_dim=8,base_ch=32), Xtr1,Xva1,Xte1,True,True),
    ("TURS-Lite", lambda: TURSNet(in_channels=1,num_classes=n_cls,regime_dim=16,variant="lite"), Xtr1,Xva1,Xte1,False,False),
    ("TURS-Strong", lambda: TURSNet(in_channels=1,num_classes=n_cls,regime_dim=24,variant="strong",multi_scale_fusion=True), Xtr1,Xva1,Xte1,False,False),
]

for label,factory,Xc_tr,Xc_va,Xc_te,use_foc,is_dict in configs:
    if label in results and 'macro_f1' in results[label]:
        print(f"{label}: done MF1={results[label]['macro_f1']:.4f}")
        continue
    print(f"Training {label}...", end=" ", flush=True)
    model=factory().to(device)
    r=train_eval(model,Xc_tr,Xc_va,Xc_te,label,use_focal=use_foc,is_dict=is_dict)
    results[label]=r
    print(f"MF1={r['macro_f1']:.4f} Acc={r['accuracy']:.4f} F1s={r['class_f1s']} ({r['time_s']:.0f}s)")
    if 'gate_stats' in r: print(f"  gates: {r['gate_stats']}")
    with open(ds_json,'w') as f: json.dump(results,f,indent=2)

# Print summary
print(f"\n{'='*80}\n  {ds_name} SUMMARY\n{'='*80}")
print(f"  {'Model':<20} {'MF1':>6} {'Acc':>6} {'WF1':>6}",end="")
for c in range(n_cls): print(f"  F1-C{c}",end="")
print(f"  {'Ep':>3} {'Time':>6}")
print(f"  {'-'*70}")
for label,_,_,_,_,_,_ in configs:
    if label in results:
        r=results[label]
        print(f"  {label:<20} {r['macro_f1']:>6.4f} {r['accuracy']:>6.4f} {r['weighted_f1']:>6.4f}",end="")
        for c in range(n_cls): print(f"  {r['class_f1s'][c]:>5.3f}",end="")
        print(f"  {r['best_epoch']:>3} {r['time_s']:>5.0f}s")
        if 'gate_stats' in r: print(f"  {'':>20} alpha={r['gate_stats'].get('alpha_mean','?')} unc={r['gate_stats'].get('uncertainty_mean','?')}")
# Confusion matrices
for label,_,_,_,_,_,_ in configs:
    if label in results and 'confusion_matrix' in results[label]:
        cm=np.array(results[label]['confusion_matrix'])
        print(f"\n  {label} confusion:"); [print(f"    {row}") for row in cm]
