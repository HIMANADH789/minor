"""Evaluate all saved checkpoints on ECG5000 and bearing fault datasets."""
import os, sys, json
import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
os.chdir(BASE)

from experiments.bearing_fair_full import ITNet, DS, compute_transport

def evaluate(model, X_test, y_test):
    model.eval()
    ds = DS(X_test, y_test)
    dl = torch.utils.data.DataLoader(ds, 256, shuffle=False, num_workers=0)
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in dl:
            logits = model(xb)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_true.extend(yb.numpy())
    yp, yt = np.array(all_preds), np.array(all_true)
    from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
    return {
        "accuracy": round(float(accuracy_score(yt, yp)), 4),
        "macro_f1": round(float(f1_score(yt, yp, average='macro', zero_division=0)), 4),
        "recalls": [round(float(r), 4) for r in recall_score(yt, yp, average=None, zero_division=0)],
        "confusion_matrix": confusion_matrix(yt, yp).tolist(),
    }


# Map (model_name, in_channels) -> checkpoint suffix
# ECG:    ecg_{SPLIT}_{CKPT_SUFFIX}.pt
# Bearing: bear_bear_{SPLIT}_{CKPT_SUFFIX}.pt
CKPT_MAP = {
    ("InceptionTime", 1):       "InceptionTime_1ch",
    ("eTAI CE", 3):             "eTAI_CE_3ch",
    ("eTAI focal g=1", 3):      "eTAI_focal_g=1_3ch",
    ("eTAI focal g=2", 3):      "eTAI_focal_g=2_3ch",
}

def run_split(split_name, data_file, nc, results_dict, is_bearing=False):
    print(f"\n{'='*70}")
    print(f"  {split_name} ({'BEARING' if is_bearing else 'ECG5000'})")
    print(f"{'='*70}")

    data = np.load(os.path.join(BASE, "data", data_file))
    Xtr, ytr = data["X_train"].astype(np.float32), data["y_train"].astype(np.int64)
    Xte, yte = data["X_test"].astype(np.float32), data["y_test"].astype(np.int64)

    sig_len = Xte.shape[-1]
    trunc = min(sig_len, 512)
    if sig_len > 512:
        Xtr, Xte = Xtr[..., :trunc], Xte[..., :trunc]

    # Standardize
    eps = 1e-8
    mu, sig = Xtr.mean(-1, keepdims=True), Xtr.std(-1, keepdims=True)
    Xtr_s = (Xtr - mu) / (sig + eps)
    mu, sig = Xte.mean(-1, keepdims=True), Xte.std(-1, keepdims=True)
    Xte_s = (Xte - mu) / (sig + eps)

    Xtr_3 = compute_transport(Xtr_s)
    Xte_3 = compute_transport(Xte_s)

    # Subsample for bearing speed
    MAX_PER_CLASS = 200 if is_bearing else 9999
    rng = np.random.RandomState(42)
    sel = []
    for c in np.unique(ytr):
        idx = np.where(ytr == c)[0]
        if len(idx) > MAX_PER_CLASS:
            idx = rng.choice(idx, MAX_PER_CLASS, replace=False)
        sel.extend(idx)
    sel = np.array(sel)
    Xtr_s_sub, Xtr_3_sub, ytr_sub = Xtr_s[sel], Xtr_3[sel], ytr[sel]

    # Model hyperparams — both were trained with nb=4, ks=[5,10,20]
    nb = 4
    ks = [5, 10, 20]

    prefix = "bear_bear" if is_bearing else "ecg"

    results_dict[split_name] = {}

    for (name, ic), Xtr_c, Xte_c in [
        (("InceptionTime", 1), Xtr_s_sub, Xte_s),
        (("eTAI CE", 3), Xtr_3_sub, Xte_3),
        (("eTAI focal g=1", 3), Xtr_3_sub, Xte_3),
        (("eTAI focal g=2", 3), Xtr_3_sub, Xte_3),
    ]:
        suffix = CKPT_MAP[(name, ic)]
        ckpt = os.path.join(BASE, "checkpoints", f"{prefix}_{split_name}_{suffix}.pt")

        if not os.path.exists(ckpt):
            print(f"  {name}: NOT FOUND at {ckpt}")
            continue

        model = ITNet(nc=nc, ic=ic, nb=nb, oc=32, ks=ks)
        state = torch.load(ckpt, map_location='cpu', weights_only=True)
        model.load_state_dict(state)

        res = evaluate(model, Xte_c, yte)
        results_dict[split_name][name] = res
        print(f"  {name}: acc={res['accuracy']:.4f} f1={res['macro_f1']:.4f} recalls={res['recalls']}")


if __name__ == "__main__":
    all_results = {}

    for split, f in [("UNBALANCED", "ecg5000_resplit.npz"), ("BALANCED", "ecg5000_fair_balanced.npz")]:
        run_split(split, f, nc=5, results_dict=all_results, is_bearing=False)

    for split, f in [("UNBALANCED", "bearing_unbalanced.npz"), ("BALANCED", "bearing_balanced.npz")]:
        run_split(split, f, nc=4, results_dict=all_results, is_bearing=True)

    out = os.path.join(BASE, "results", "fair_full", "all_results.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out}")
