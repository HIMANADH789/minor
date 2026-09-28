"""Dataset loading + DRTN soft-assignment support for the context3 screen.

Reuses the audited pieces unchanged:
    * hard-regime controls + valid-region heterogeneity:
      experiments/drtn_conditioned_minirocket_haptics_3seed/runner.py
    * raw activation extractor / canonical PPV / heterogeneity core:
      experiments/drtn_conditioned_minirocket_transfer_seed42/core.py
    * DRTN training protocol (R5 K=8, train-only fitting, val checkpoint
      selection, freezing): experiments/drtn_conditioned_minirocket_transfer_seed42/runner.py

This module adds only:
    * deterministic TSV loading of the verified Kaggle UCRArchive_2018 copies
      (data/kaggle/<DS>/<DS>_{TRAIN,TEST}.tsv) with the repository's
      stratified 15%-of-train validation split (seed 42) -- the same split
      rule the benchmark uses for NPZ datasets
    * DRTN soft-assignment extraction for the A_SOFT ablation
      (softmax(-d2/tau) over the FROZEN VQ codebook, the exact SoftCodebook
      semantics with the same config tau; no architecture change)
    * the soft conditional-rate heterogeneity H_soft (valid-mask restricted)
"""
import hashlib
import os

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KAGGLE_DATA = os.path.join(ROOT, "data", "kaggle")

DATASET_SPECS = {
    "GunPoint": {"train": 50, "test": 150, "T": 150, "n_classes": 2},
    "ItalyPowerDemand": {"train": 67, "test": 1029, "T": 24, "n_classes": 2},
    "FordA": {"train": 3601, "test": 1320, "T": 500, "n_classes": 2},
}


def load_kaggle_ucr(ds_name, seed=42, val_fraction=0.15):
    """Load a verified Kaggle UCR TSV dataset and build the canonical split.

    Train/test come from the canonical UCR files (never re-split). The
    validation set is a stratified 15% of the TRAIN split with
    random_state=seed -- the repository's established split rule.
    Returns float32 raw (unnormalized) X; z-normalization happens later,
    once, before both MiniROCKET and DRTN.
    """
    from sklearn.model_selection import train_test_split

    d_dir = os.path.join(KAGGLE_DATA, ds_name)
    tr = np.loadtxt(os.path.join(d_dir, f"{ds_name}_TRAIN.tsv"),
                    delimiter="\t", ndmin=2)
    te = np.loadtxt(os.path.join(d_dir, f"{ds_name}_TEST.tsv"),
                    delimiter="\t", ndmin=2)
    ytr, Xtr = tr[:, 0].astype(int), tr[:, 1:].astype(np.float32)
    yte, Xte = te[:, 0].astype(int), te[:, 1:].astype(np.float32)

    spec = DATASET_SPECS[ds_name]
    # structural gate against the canonical UCR shapes recorded at install time
    assert Xtr.shape == (spec["train"], spec["T"]), \
        f"{ds_name}: train {Xtr.shape} != canonical {(spec['train'], spec['T'])}"
    assert Xte.shape == (spec["test"], spec["T"]), \
        f"{ds_name}: test {Xte.shape} != canonical {(spec['test'], spec['T'])}"

    labels = sorted(set(ytr.tolist()) | set(yte.tolist()))
    assert len(labels) == spec["n_classes"], f"{ds_name}: label set {labels}"

    Xtr, Xva, ytr, yva = train_test_split(
        Xtr, ytr, test_size=val_fraction, stratify=ytr, random_state=seed)

    # remap labels to contiguous ints while keeping the original mapping
    uniq = sorted(labels)
    lut = {lab: i for i, lab in enumerate(uniq)}
    map_fn = np.vectorize(lut.get)
    return {
        "name": ds_name, "source_path": os.path.relpath(d_dir, ROOT),
        "Xtr": Xtr, "ytr": map_fn(ytr), "Xva": Xva, "yva": map_fn(yva),
        "Xte": Xte, "yte": map_fn(yte),
        "n_classes": len(uniq), "L": spec["T"],
        "label_set_original": labels,
        "label_map": {str(k): v for k, v in lut.items()},
        "val_source": f"stratified_{int(val_fraction*100)}pct_of_train_seed{seed}",
        "file_sha256": {
            f"{ds_name}_TRAIN.tsv": _sha(os.path.join(d_dir, f"{ds_name}_TRAIN.tsv")),
            f"{ds_name}_TEST.tsv": _sha(os.path.join(d_dir, f"{ds_name}_TEST.tsv")),
        },
    }


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def znorm(X):
    return ((X - X.mean(-1, keepdims=True)) /
            (X.std(-1, keepdims=True) + 1e-8)).astype(np.float32)


@torch.no_grad()
def extract_soft_assignments(model, X, device, tau, batch_size=64):
    """Soft assignments q_tk = softmax(-d2(z_t, c_k) / tau) from the FROZEN
    codebook -- exactly the SoftCodebook semantics (models/drtn/model.py,
    class SoftCodebook). The R5 HardVQ module stores no tau; the value is
    taken from the frozen DRTN config (tau=0.5, the same constant the R2
    SoftCodebook uses). Read-only use of the frozen codebook; no
    architecture or parameter change, tau never tuned."""
    model.eval()
    out = []
    zn = znorm(X)
    for s in range(0, len(X), batch_size):
        xb = torch.from_numpy(zn[s:s + batch_size])[:, None, :].to(device)
        z = model.encoder(xb)                                   # (B, T, D)
        z_flat = z.reshape(-1, z.shape[-1])
        d2 = (z_flat.pow(2).sum(1, keepdim=True)
              - 2.0 * (z_flat @ model.vq.codes.T)
              + model.vq.codes.pow(2).sum(1).unsqueeze(0))
        a = torch.softmax(-d2 / tau, dim=1)                     # (B*T, K)
        out.append(a.reshape(xb.shape[0], z.shape[1], -1).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float64)


def soft_heterogeneity_features(act, valid, soft, eps=1e-12):
    """Soft-regime heterogeneity, valid-mask restricted.

    act: (N, F, T) bool; valid: (F, T) bool; soft: (N, T, K) float64
    For each feature m (valid positions only):
        q_k   = mean_t(q_tk)
        PPV_{m,k} = sum_t q_tk * a_mt / sum_t q_tk
        H_soft,m  = sum_k q_k * (PPV_{m,k} - PPV_m)^2
    The weights q_tk are probabilities (non-negative, sum over k = 1), so
    this is a proper weighted conditional rate, NOT an average that
    collapses to PPV_m: H_soft = 0 iff the weighted rates are all equal to
    the global rate.
    """
    n_samples, n_features, T = act.shape
    K = soft.shape[2]
    H = np.zeros((n_samples, n_features), dtype=np.float64)
    groups = {}
    for f in range(n_features):
        row = valid[f]
        p = 0 if row.all() else int(np.argmax(row))
        groups.setdefault(p, []).append(f)
    for i in range(n_samples):
        for p, feats in groups.items():
            hi = T - p if p > 0 else T
            Av = act[i, feats, p:hi].astype(np.float64)      # (Fp, n_valid)
            S = soft[i, p:hi, :]                             # (n_valid, K)
            q_k = S.mean(axis=0)                             # (K,)
            denom = S.sum(axis=0)                            # (K,) = n_valid*q_k
            ppv_k = (S.T @ Av.T) / np.maximum(denom, eps)[:, None]  # (K, Fp)
            ppv_g = Av.mean(axis=1)                          # (Fp,)
            dev = ppv_k - ppv_g[None, :]
            H[i, feats] = (q_k[:, None] * (dev ** 2)).sum(axis=0)
    return H


def independent_soft_recompute(act_single, valid_single, soft_single, eps=1e-12):
    """Float64 loop reference for one sample/feature: returns H_soft."""
    vm = np.asarray(valid_single, dtype=bool)
    a = np.asarray(act_single, dtype=np.float64)[vm]
    S = np.asarray(soft_single, dtype=np.float64)[vm]            # (n_valid, K)
    ppv_g = a.mean()
    q_k = S.mean(axis=0)
    H = 0.0
    for k in range(S.shape[1]):
        w = S[:, k].sum()
        if w <= eps:
            continue
        ppv_k = (S[:, k] * a).sum() / w
        H += q_k[k] * (ppv_k - ppv_g) ** 2
    return float(H)
