"""Forensic check: why are R1 and R3 predictions identical?

R1 = [global MR | official-DRTN het | Hydra], R3 = [global MR | SSL-context het | Hydra].
They share the global and Hydra blocks and differ only in the heterogeneity
block (different learned regime sources), yet R1 == R3 exactly (0/308
prediction differences, identical alpha).

This script verifies at the array level:
  1. The two het blocks genuinely differ (no wiring alias).
  2. RidgeClassifierCV selects a heavily-penalized alpha (155 train+val
     samples, 12044 features).
  3. A control model on [global | Hydra] ONLY (het block dropped)
     reproduces R1/R3 predictions exactly -> the het block is
     decision-irrelevant at the CV-selected alpha.
  4. Decision-value margins vs the between-model differences.

No test labels are used for any decision.
"""
import json
import os

import numpy as np
import torch
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import f1_score

from experiments.rcmkn_haptics_seed42.config import SEED, HYDRA, N_GLOBAL
from experiments.rcmkn_haptics_seed42 import kernel_features as kf
from experiments.rcmkn_haptics_seed42.runner import (
    load_dataset, load_official_drtn, extract_drtn_regimes,
    extract_context_regimes, set_seed, OUT_DIR)
from experiments.rcmkn_haptics_seed42.model import RCMKNContextModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_seed(SEED)
data = load_dataset("Haptics")
Xtr, ytr = data["Xtr"], data["ytr"]
Xva, yva = data["Xva"], data["yva"]
Xte, yte = data["Xte"], data["yte"]
ytrva = np.concatenate([ytr, yva])


def znorm(X):
    return ((X - X.mean(-1, keepdims=True)) /
            (X.std(-1, keepdims=True) + 1e-8)).astype(np.float32)


Xtr_z, Xva_z, Xte_z = znorm(Xtr), znorm(Xva), znorm(Xte)
Xtrva_z = np.vstack([Xtr_z, Xva_z])

# ---------------- regimes: official DRTN (R1) vs SSL context (R3) ------------
drtn, _ = load_official_drtn(device)
reg1_trva = extract_drtn_regimes(drtn, Xtrva_z, device=device)
reg1_te = extract_drtn_regimes(drtn, Xte_z, device=device)

ck = torch.load(os.path.join(OUT_DIR, "context_model_seed42.pt"),
                map_location=device, weights_only=False)
model = RCMKNContextModel(n_classes=data["n_classes"]).to(device)
model.load_state_dict(ck["model_state"])
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
reg3_trva = extract_context_regimes(model, Xtrva_z, device)
reg3_te = extract_context_regimes(model, Xte_z, device)

n_diff = int((reg1_trva != reg3_trva).sum())
print(f"[1] regime arrays differ: {n_diff}/{reg1_trva.size} trainva positions "
      f"({n_diff / reg1_trva.size:.2%}); test "
      f"{int((reg1_te != reg3_te).sum())}/{reg1_te.size}; "
      f"shares_memory={np.shares_memory(reg1_trva, reg3_trva)}")

# determinism cross-check: R2 test regimes saved by the official run
saved = np.load(os.path.join(OUT_DIR, "_regimes_R2_te.npy"))
print(f"    re-extracted R3 test regimes match saved _regimes_R2_te.npy "
      f"(first 20 rows): {np.array_equal(reg3_te[:20], saved)}")

# ---------------- shared blocks: global MR + Hydra ---------------------------
from aeon.transformations.collection.convolution_based import MiniRocket

extractor = MiniRocket(random_state=SEED, n_jobs=-1)
extractor.fit(Xtr_z[:, None, :].astype(np.float32))
F_trva = extractor.transform(Xtrva_z[:, None, :].astype(np.float32))
F_te = extractor.transform(Xte_z[:, None, :].astype(np.float32))
G_tr, G_va = F_trva[:len(Xtr), :N_GLOBAL], F_trva[len(Xtr):, :N_GLOBAL]
G_trva, G_te = F_trva[:, :N_GLOBAL], F_te[:, :N_GLOBAL]

Hd_trva, _ = kf.compute_hydra_features(Xtrva_z, data["L"], k=HYDRA["k"], g=HYDRA["g"])
Hd_te, _ = kf.compute_hydra_features(Xte_z, data["L"], k=HYDRA["k"], g=HYDRA["g"])
Hd_trva_s, Hd_te_s = kf.scale_hydra_features(Hd_trva, Hd_te)

# ---------------- het blocks -------------------------------------------------
_, valid = kf.compute_raw_activations(extractor, Xtrva_z[:4])
valid_het = valid[N_GLOBAL:]
H = {}
for name, r_trva, r_te in [("R1", reg1_trva, reg1_te), ("R3", reg3_trva, reg3_te)]:
    H_tr = kf.compute_heterogeneity_chunked(extractor, Xtrva_z, r_trva,
                                            valid_het)
    act_va, _ = kf.compute_raw_activations(extractor, Xva_z)
    H_va = kf.compute_regime_heterogeneity(act_va[:, N_GLOBAL:], valid_het,
                                           r_trva[len(Xtr):])
    act_te, _ = kf.compute_raw_activations(extractor, Xte_z)
    H_te = kf.compute_regime_heterogeneity(act_te[:, N_GLOBAL:], valid_het, r_te)
    H[name] = (H_tr, H_va, H_te)
    del act_va, act_te

d_het = max(float(np.max(np.abs(H["R1"][i] - H["R3"][i]))) for i in range(3))
print(f"[2] het blocks: max|R1het - R3het| over train/val/test = {d_het:.4e} "
      f"(differ: {d_het > 0})")

# ---------------- Ridge models ----------------------------------------------
alphas = np.logspace(-4, 4, 20)
X1 = np.hstack([G_trva, H["R1"][0], Hd_trva_s])
X3 = np.hstack([G_trva, H["R3"][0], Hd_trva_s])
XGH = np.hstack([G_trva, Hd_trva_s])
T1 = np.hstack([G_te, H["R1"][2], Hd_te_s])
T3 = np.hstack([G_te, H["R3"][2], Hd_te_s])
TGH = np.hstack([G_te, Hd_te_s])

m1 = RidgeClassifierCV(alphas=alphas).fit(X1, ytrva)
m3 = RidgeClassifierCV(alphas=alphas).fit(X3, ytrva)
mgh = RidgeClassifierCV(alphas=alphas).fit(XGH, ytrva)
p1, p3, pgh = m1.predict(T1), m3.predict(T3), mgh.predict(TGH)

print(f"[3] alphas: R1={m1.alpha_:.4f} R3={m3.alpha_:.4f} [G|Hydra]={mgh.alpha_:.4f}")
print(f"    test MF1: R1={f1_score(yte, p1, average='macro'):.4f} "
      f"R3={f1_score(yte, p3, average='macro'):.4f} "
      f"[G|Hydra]={f1_score(yte, pgh, average='macro'):.4f}")
print(f"    predictions identical: R1==R3 {np.array_equal(p1, p3)}; "
      f"R1==[G|Hydra] {np.array_equal(p1, pgh)}; "
      f"R3==[G|Hydra] {np.array_equal(p3, pgh)}")

D1, D3, DGH = m1.decision_function(T1), m3.decision_function(T3), \
    mgh.decision_function(TGH)
print(f"[4] decision values: max|D_R1-D_R3|={np.max(np.abs(D1 - D3)):.4e}, "
      f"max|D_R1-D_GH|={np.max(np.abs(D1 - DGH)):.4e}")
for nm, D in [("R1", D1), ("R3", D3), ("[G|Hydra]", DGH)]:
    s = np.sort(D, axis=1)
    margins = s[:, -1] - s[:, -2]
    print(f"    {nm}: top1-top2 margin min={margins.min():.4f} "
          f"median={np.median(margins):.4f}")

out = {
    "regime_positions_differ_trainva": n_diff,
    "het_block_max_abs_diff": d_het,
    "alphas": {"R1": float(m1.alpha_), "R3": float(m3.alpha_),
               "global_hydra_only": float(mgh.alpha_)},
    "r1_equals_r3_predictions": bool(np.array_equal(p1, p3)),
    "r1_equals_global_hydra_only": bool(np.array_equal(p1, pgh)),
    "r3_equals_global_hydra_only": bool(np.array_equal(p3, pgh)),
    "test_macro_f1": {"R1": float(f1_score(yte, p1, average="macro")),
                      "R3": float(f1_score(yte, p3, average="macro")),
                      "global_hydra_only": float(f1_score(yte, pgh,
                                                          average="macro"))},
}
with open(os.path.join(OUT_DIR, "forensic_r1_r3.json"), "w") as f:
    json.dump(out, f, indent=2)
print("saved forensic_r1_r3.json")
