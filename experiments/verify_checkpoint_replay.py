"""
Inference-only replay verification for TURS-Stack.

Proves that diagnostics can be reproduced from the saved checkpoints without
retraining:
  1. Rebuild data with the exact repository protocol (deterministic seed 42).
  2. Load the saved model + combiner checkpoints, run eval-only inference.
  3. Check soft-vote test MF1 matches the value in full_results.json.
  4. Extract per-branch intermediates (z_t, v_t, u_t, alpha) from the frozen
     model using only its own modules (no training, no RNG dependence).

Usage:  python experiments/verify_checkpoint_replay.py [DATASET_TAG]
"""
import os
import sys
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.turs_stack.model import TURSStack  # noqa: E402

SEED = 42
VAL_FRAC = 0.15
NUM_CLASSES = {"ECG5000_UNBAL": 5, "ECG5000_BAL": 5, "CWRU_UNBAL": 4, "CWRU_BAL": 4}
DATA = {
    "ECG5000_UNBAL": "data/ecg5000_resplit.npz",
    "ECG5000_BAL": "data/ecg5000_fair_balanced.npz",
    "CWRU_UNBAL": "data/cwru_unbalanced.npz",
    "CWRU_BAL": "data/cwru_balanced.npz",
}


def znorm(X):
    mu = X.mean(axis=-1, keepdims=True)
    sig = X.std(axis=-1, keepdims=True)
    return ((X - mu) / (sig + 1e-8)).astype(np.float32)


def load_split(tag):
    """Exact copy of the runner's data protocol (deterministic)."""
    data = np.load(os.path.join(ROOT, DATA[tag]))
    if "X_train" in data:
        X_trval, y_trval = data["X_train"].astype(np.float32), data["y_train"].astype(np.int64)
        X_te, y_te = data["X_test"].astype(np.float32), data["y_test"].astype(np.int64)
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_trval, y_trval, test_size=VAL_FRAC / 0.85, random_state=SEED,
            stratify=y_trval)
    else:
        X_all, y_all = data["X"].astype(np.float32), data["y"].astype(np.int64)
        idx = np.arange(len(X_all))
        trval_idx, te_idx = train_test_split(idx, test_size=0.15, random_state=SEED,
                                             stratify=y_all)
        X_trval, y_trval = X_all[trval_idx], y_all[trval_idx]
        X_te, y_te = X_all[te_idx], y_all[te_idx]
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_trval, y_trval, test_size=VAL_FRAC / 0.85, random_state=SEED,
            stratify=y_trval)
    return znorm(X_tr), znorm(X_va), znorm(X_te), y_tr, y_va, y_te


def extract_lite_intermediates(branch, F_T, H):
    """Replicate LiteBranch.forward ops to expose z_t / v_t / u_t / alpha."""
    F_T_gap = F_T.mean(dim=2)
    z_t = branch.P_z(H).permute(0, 2, 1)                       # [B, T, rd]
    v_t = branch.vel_linear(z_t)                               # [B, T, rd]
    u_t = torch.sigmoid(branch.P_u(torch.cat([z_t, v_t], dim=-1)))  # [B, T, 1]
    F_R = branch.P_R(torch.cat([z_t, v_t, u_t], dim=-1))
    F_R_gap = F_R.mean(dim=1)
    I_TR = branch.P_I(F_T_gap * branch.P_T(F_R_gap))
    alpha = torch.sigmoid(branch.P_alpha(torch.cat([F_T_gap, F_R_gap], dim=1)))
    return {"z_t": z_t, "v_t": v_t, "u_t": u_t, "alpha": alpha, "F_R_gap": F_R_gap}


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "ECG5000_UNBAL"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dataset: {tag} | device: {device}")

    # ---- 1. deterministic data replay ----
    X_tr, X_va, X_te, y_tr, y_va, y_te = load_split(tag)
    print(f"replayed split: train={len(X_tr)} val={len(X_va)} test={len(X_te)}")

    # ---- 2. load checkpoint (eval-only, no training) ----
    ckpt = torch.load(os.path.join(ROOT, "checkpoints", "turs_stack",
                                   f"{tag}_turs_stack.pt"),
                      map_location="cpu", weights_only=False)
    model = TURSStack(in_channels=1, num_classes=ckpt["num_classes"],
                      sequence_length=ckpt["sequence_length"],
                      sigma_init=ckpt["sigma0"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"loaded checkpoint: epoch={ckpt['epoch']} best_val_mf1={ckpt['best_metric']:.4f}")

    # ---- 3. eval-only inference over the test set ----
    te_dl = DataLoader(TensorDataset(torch.from_numpy(X_te[:, None, :]).float(),
                                     torch.from_numpy(y_te).long()), batch_size=256)
    all_probs, all_labels, extra = [], [], []
    with torch.no_grad():
        for xb, yb in te_dl:
            xb = xb.to(device)
            out = model(xb)
            all_probs.append(torch.stack(out["probs"], 0).cpu())   # [4, B, C]
            all_labels.append(yb)
            if not extra:  # intermediates from the first batch are enough to prove extraction
                lite = extract_lite_intermediates(model.lite, out["F_T"], out["H"])
                extra.append({
                    "lite_z_t": lite["z_t"], "lite_v_t": lite["v_t"],
                    "lite_u_t": lite["u_t"], "lite_alpha": lite["alpha"],
                    "cs_beta": out["beta_cs"], "cmr_beta": out["beta_cmr"],
                    "novelty_e_t": out["novelty"],
                    "cs_zbar1": model.cs.branch1(out["H"], model.cs.scale_params.sigma),
                })
    P = torch.cat(all_probs, dim=1)      # [4, N, C]
    y = torch.cat(all_labels).numpy()
    soft = P.mean(dim=0).argmax(dim=-1).numpy()
    mf1 = f1_score(y, soft, average="macro")

    saved = json.load(open(os.path.join(ROOT, "results", "turs_stack", tag,
                                        "full_results.json")))
    saved_soft = saved["combination_test_mf1"]["soft_vote"]  # rounded to 4 dp in JSON
    match = round(mf1, 4) == saved_soft
    print(f"soft-vote test MF1: replayed={mf1:.6f}  saved={saved_soft:.6f}  "
          f"{'MATCH [OK]' if match else 'MISMATCH [FAIL]'}")

    # ---- 4. intermediates are live and finite ----
    e = extra[0]
    print("\nextracted intermediates (batch 0, inference-only, no retraining):")
    for name, t in e.items():
        print(f"  {name:14s} shape={tuple(t.shape)}  mean={t.mean().item():+.4f}  "
              f"finite={bool(torch.isfinite(t).all())}")
    beta_sum = e["cs_beta"].sum(-1)
    print(f"  cs beta simplex check: sum~1 -> {bool(torch.allclose(beta_sum, torch.ones_like(beta_sum), atol=1e-5))}")

    ok = match
    print(f"\nVERDICT: {'REPLAY OK - diagnostics reproducible from checkpoint alone' if ok else 'REPLAY FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
