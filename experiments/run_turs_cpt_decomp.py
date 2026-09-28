"""CPT decomposition: concentration-only vs phase-only on 10K."""
import os, sys, json, time
import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.features.cpt import (
    estimate_period, cpt_transform,
    rank_kernels_by_discriminative_power, select_top_k_transformers,
)
from src.features.ktm import block_stats
from aeon.transformations.collection.convolution_based import MiniRocket

SEED = 42
K_TOP = 1024
DATASETS = [
    ("ECG5000_UNBAL", "data/ecg5000_resplit.npz", 5),
    ("ECG5000_BAL",   "data/ecg5000_fair_balanced.npz", 5),
    ("CWRU_UNBAL",    "data/cwru_unbalanced.npz", 4),
    ("CWRU_BAL",      "data/cwru_balanced.npz", 4),
]

def znorm(X):
    mu = X.mean(axis=-1, keepdims=True)
    sig = X.std(axis=-1, keepdims=True) + 1e-8
    return ((X - mu) / sig).astype(np.float32)

def load_split(ds_file):
    data = np.load(os.path.join(ROOT, ds_file))
    if "X_train" in data:
        Xa, ya = data["X_train"], data["y_train"].astype(int)
        Xte, yte = data["X_test"], data["y_test"].astype(int)
        Xtr, Xva, ytr, yva = train_test_split(Xa, ya, test_size=0.15, stratify=ya, random_state=SEED)
    else:
        Xa, ya = data["X"], data["y"].astype(int)
        Xtr, Xte, ytr, yte = train_test_split(Xa, ya, test_size=0.15, stratify=ya, random_state=SEED)
        Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=0.15, stratify=ytr, random_state=SEED)
    return Xtr, Xva, Xte, ytr, yva, yte

def run_ridge(X_trva, y_trva, X_te, y_te):
    clf = RidgeClassifierCV(alphas=np.logspace(-4, 4, 20))
    clf.fit(X_trva, y_trva)
    preds = clf.predict(X_te)
    return round(float(f1_score(y_te, preds, average="macro", zero_division=0)), 4)

def std_fit(train, val, test):
    trva = np.concatenate([train, val])
    m, s = trva.mean(0, keepdims=True), trva.std(0, keepdims=True)
    s = np.where(s < 1e-8, 1.0, s)
    return (train - m) / s, (val - m) / s, (test - m) / s

results = {}
for ds_name, ds_file, n_cls in DATASETS:
    print(f"\n--- {ds_name} ---", flush=True)
    Xtr, Xva, Xte, ytr, yva, yte = load_split(ds_file)
    Xn_tr, Xn_va, Xn_te = znorm(Xtr), znorm(Xva), znorm(Xte)
    y_trva = np.concatenate([ytr, yva])

    mr = MiniRocket(n_kernels=10000, random_state=SEED, n_jobs=-1)
    mr.fit_transform(Xn_tr[:, None, :])
    Z_tr = mr.transform(Xn_tr[:, None, :])
    Z_va = mr.transform(Xn_va[:, None, :])
    Z_te = mr.transform(Xn_te[:, None, :])

    X_all = np.concatenate([Xn_tr, Xn_va, Xn_te])
    periods_all = np.array([estimate_period(X_all[i])[0] for i in range(len(X_all))])
    periods_tr = periods_all[:len(Xn_tr)]
    periods_va = periods_all[len(Xn_tr):len(Xn_tr)+len(Xn_va)]
    periods_te = periods_all[len(Xn_tr)+len(Xn_va):]

    ranks, _ = rank_kernels_by_discriminative_power(Z_tr, ytr)
    _, feat_mask = select_top_k_transformers(mr, ranks, K=K_TOP)

    conc_tr, phase_tr, _ = cpt_transform(Xn_tr, mr, periods_tr, return_counts=True)
    conc_va, phase_va, _ = cpt_transform(Xn_va, mr, periods_va, return_counts=True)
    conc_te, phase_te, _ = cpt_transform(Xn_te, mr, periods_te, return_counts=True)

    conc_tr_k, conc_va_k, conc_te_k = conc_tr[:, feat_mask], conc_va[:, feat_mask], conc_te[:, feat_mask]
    phase_tr_k, phase_va_k, phase_te_k = phase_tr[:, feat_mask], phase_va[:, feat_mask], phase_te[:, feat_mask]

    conc_tr_s, conc_va_s, conc_te_s = std_fit(conc_tr_k, conc_va_k, conc_te_k)
    phase_tr_s, phase_va_s, phase_te_s = std_fit(phase_tr_k, phase_va_k, phase_te_k)

    variants = {
        "MR": (np.concatenate([Z_tr, Z_va]), np.concatenate([Z_te])),
        "MR+conc": (np.concatenate([Z_tr, conc_tr_s], axis=1), np.concatenate([Z_va, conc_va_s], axis=1), np.concatenate([Z_te, conc_te_s], axis=1)),
        "MR+phase": (np.concatenate([Z_tr, phase_tr_s], axis=1), np.concatenate([Z_va, phase_va_s], axis=1), np.concatenate([Z_te, phase_te_s], axis=1)),
        "MR+conc+phase": (np.concatenate([Z_tr, conc_tr_s, phase_tr_s], axis=1), np.concatenate([Z_va, conc_va_s, phase_va_s], axis=1), np.concatenate([Z_te, conc_te_s, phase_te_s], axis=1)),
    }

    results[ds_name] = {}
    for vname, blocks in variants.items():
        if vname == "MR":
            trva, te = blocks
        else:
            trva, va, te = blocks[0], blocks[1], blocks[2]
            trva = np.concatenate([trva, va])
        r = run_ridge(trva, y_trva, te, yte)
        results[ds_name][vname] = r
        print(f"  {vname:>16}: MF1={r:.4f}")

print("\n" + "="*60)
print("  DECOMPOSITION (10K)")
print("="*60)
header = f"{'Dataset':<18} {'MR':>8} {'MR+conc':>9} {'MR+phase':>10} {'MR+conc+ph':>12}"
print(header)
print("-"*len(header))
for ds_name, _, _ in DATASETS:
    r = results[ds_name]
    print(f"{ds_name:<18} {r['MR']:>8.4f} {r['MR+conc']:>9.4f} {r['MR+phase']:>10.4f} {r['MR+conc+phase']:>12.4f}")

OUT = os.path.join(ROOT, "results", "turs_cpt")
with open(os.path.join(OUT, "decomposition.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUT}/")
