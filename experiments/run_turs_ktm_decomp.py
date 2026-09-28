"""Section 19 decomposition + diagnostics for TURS-KTM on 10K configuration."""
import os, sys, json, time
import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.features.ktm import ktm_transform, block_stats, feature_owners
from aeon.transformations.collection.convolution_based import MiniRocket

SEED = 42
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
    mf1 = float(np.mean(y_te == preds))  # accuracy as proxy
    from sklearn.metrics import f1_score
    return {
        "macro_f1": round(float(f1_score(y_te, preds, average="macro", zero_division=0)), 4),
        "alpha": round(float(clf.alpha_), 6),
        "n_features": int(X_trva.shape[1]),
    }

def std_fit(train, val, test):
    trva = np.concatenate([train, val])
    m, s = trva.mean(0, keepdims=True), trva.std(0, keepdims=True)
    s = np.where(s < 1e-8, 1.0, s)
    return (train - m) / s, (val - m) / s, (test - m) / s

results = {}
diag_all = {}

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

    ppv_tr, mu_tr, W_tr, ct_tr = ktm_transform(Xn_tr, mr, return_counts=True)
    ppv_va, mu_va, W_va, ct_va = ktm_transform(Xn_va, mr, return_counts=True)
    ppv_te, mu_te, W_te, ct_te = ktm_transform(Xn_te, mr, return_counts=True)

    mu_tr_s, mu_va_s, mu_te_s = std_fit(mu_tr, mu_va, mu_te)
    W_tr_s, W_va_s, W_te_s = std_fit(W_tr, W_va, W_te)

    Z_trva = np.concatenate([Z_tr, Z_va])
    Z_te_cat = Z_te

    variants = {
        "MR": (Z_trva, Z_te_cat),
        "MR+mu": (np.concatenate([Z_trva, np.concatenate([mu_tr_s, mu_va_s])], axis=1),
                   np.concatenate([Z_te_cat, mu_te_s], axis=1)),
        "MR+W": (np.concatenate([Z_trva, np.concatenate([W_tr_s, W_va_s])], axis=1),
                  np.concatenate([Z_te_cat, W_te_s], axis=1)),
        "MR+mu+W": (np.concatenate([Z_trva, np.concatenate([mu_tr_s, mu_va_s]),
                                     np.concatenate([W_tr_s, W_va_s])], axis=1),
                     np.concatenate([Z_te_cat, mu_te_s, W_te_s], axis=1)),
    }
    results[ds_name] = {}
    for vname, (X_trva, X_te_v) in variants.items():
        r = run_ridge(X_trva, y_trva, X_te_v, yte)
        results[ds_name][vname] = r
        print(f"  {vname:>12}: MF1={r['macro_f1']:.4f} (F={r['n_features']})")

    # Block stats
    F_mr = np.concatenate([Z_tr, Z_va])
    F_mu = np.concatenate([mu_tr, mu_va])
    F_W = np.concatenate([W_tr, W_va])
    diag_all[ds_name] = {
        "mr_block": block_stats(F_mr.astype(np.float64)),
        "mu_block": block_stats(F_mu.astype(np.float64)),
        "W_block": block_stats(F_W.astype(np.float64)),
        "counts": {"zero_frac": float((ct_tr == 0).mean()),
                   "mean_count": float(ct_tr.mean()),
                   "max_count": int(ct_tr.max())},
    }

# Print decomposition table
print("\n" + "=" * 60)
print("  SECTION 19: DECOMPOSITION (10K)")
print("=" * 60)
header = f"{'Dataset':<18} {'MR':>8} {'MR+mu':>8} {'MR+W':>8} {'MR+mu+W':>10}"
print(header)
print("-" * len(header))
for ds_name, _, _ in DATASETS:
    r = results[ds_name]
    print(f"{ds_name:<18} {r['MR']['macro_f1']:>8.4f} {r['MR+mu']['macro_f1']:>8.4f}"
          f" {r['MR+W']['macro_f1']:>8.4f} {r['MR+mu+W']['macro_f1']:>10.4f}")

OUT = os.path.join(ROOT, "results", "turs_ktm")
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "decomposition.json"), "w") as f:
    json.dump(results, f, indent=2)
with open(os.path.join(OUT, "diagnostics_detail.json"), "w") as f:
    json.dump(diag_all, f, indent=2)
print(f"\nSaved to {OUT}/")
