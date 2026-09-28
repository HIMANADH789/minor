"""R5 (HERAMBA) rho sweep -- Wafer + ElectricDevices, seed 42.

Extends the audited rcmkn_final_validation/rho_sweep.py machinery to the
two datasets that never had a stored rho sweep:
    Wafer            : context ckpt results/wafer_ecg5000_ipd_r5/Wafer/
                       (none stored -- trains a seed-42 context first)
    ElectricDevices  : context ckpt results/electricdevices_mr_r5/seed42/
                       (stored heramba context from the 3-seed run)

Rules IDENTICAL to the audited rho_sweep.py (rcmkn_final_validation):
    N_H = round(rho*9996); N_G = 9996 - N_H (exact sum)
    G = first N_G canonical MiniROCKET features (never label-ranked)
    H = top-N_H by ANOVA F-statistic, recomputed INSIDE each CV training
        fold (fold-train labels only; never val-fold or test labels)
    5-fold stratified CV on the development set; tie 0.001 -> smaller rho
    final fit on train+val at rho*; ONE official test evaluation
    rho grid [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
Splits = the canonical benchmark splits of the respective tasks:
    Wafer   : TSVs 1000/6164 + stratified 15% val @ seed42 -> 850/150/6164
    ElectricDevices: TSVs 8926/7711 + stratified 15% val @ seed42
              -> 7587/1339/7711 (same loaders as the 3-seed runs)
No test-driven tuning; frozen recipe; one test evaluation at rho*.
"""
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.feature_selection import f_classif
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "rho_sweep_wafer_ed")

SEED = 42
TOTAL_BUDGET = 9996
N_GLOBAL, N_HET = 4998, 4998
RHO_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
K_FOLDS = 5
TIE_TOL = 0.001
ALPHAS = np.logspace(-4, 4, 20)

ARCHIVE = os.path.join("data", "kaggle", "_ucrarchive_2018",
                       "UCRArchive_2018", "UCRArchive_2018")
SPEC = {
    "Wafer": {
        "n_classes": 2,
        "expected": {"train": 850, "val": 150, "test": 6164, "T": 152},
        "ckpt": os.path.join(OUT, "_contexts", "Wafer_seed42.pt"),
        "train_context": True,
    },
    "ElectricDevices": {
        "n_classes": 7,
        "expected": {"train": 7587, "val": 1339, "test": 7711, "T": 96},
        "ckpt": os.path.join(OUT, "_contexts", "ElectricDevices_seed42.pt"),
        "train_context": True,
    },
}


def log(m):
    print(m, flush=True)


def macro_f1(y, p):
    from sklearn.metrics import f1_score
    return f1_score(y, p, average="macro", zero_division=0)


def budget_split(rho):
    n_h = int(round(rho * TOTAL_BUDGET))
    return TOTAL_BUDGET - n_h, n_h


def jaccard(a, b):
    a, b = set(map(int, a)), set(map(int, b))
    return len(a & b) / len(a | b) if (a or b) else 1.0


def assemble(G, H, n_g, h_idx):
    X = np.hstack([G[:, :n_g], H[:, h_idx]]) if len(h_idx) else G[:, :n_g]
    assert X.shape[1] == TOTAL_BUDGET
    return X


def load_data(ds_name):
    from sklearn.model_selection import train_test_split
    d_dir = os.path.join(ROOT, ARCHIVE, ds_name)
    tr = np.loadtxt(os.path.join(d_dir, f"{ds_name}_TRAIN.tsv"),
                    delimiter="\t", ndmin=2)
    te = np.loadtxt(os.path.join(d_dir, f"{ds_name}_TEST.tsv"),
                    delimiter="\t", ndmin=2)
    ytr, Xtr = tr[:, 0].astype(int), tr[:, 1:].astype(np.float32)
    yte, Xte = te[:, 0].astype(int), te[:, 1:].astype(np.float32)
    labels = sorted(set(ytr.tolist()) | set(yte.tolist()))
    Xtr, Xva, ytr, yva = train_test_split(
        Xtr, ytr, test_size=0.15, stratify=ytr, random_state=SEED)
    lut = {lab: i for i, lab in enumerate(labels)}
    map_fn = np.vectorize(lut.get)
    return {"Xtr": Xtr, "ytr": map_fn(ytr), "Xva": Xva, "yva": map_fn(yva),
            "Xte": Xte, "yte": map_fn(yte), "n_classes": len(labels),
            "L": int(Xtr.shape[1])}


def znorm(X):
    mu = X.mean(axis=-1, keepdims=True)
    sig = X.std(axis=-1, keepdims=True) + 1e-8
    return ((X - mu) / sig).astype(np.float32)


def get_context(ds_name, spec, Xtr_z, ytr, Xva_z, yva, device):
    """Frozen seed-42 context model for this dataset (load or train)."""
    from experiments.rcmkn_haptics_seed42.model import RCMKNContextModel
    ck = torch.load(spec["ckpt"], map_location=device, weights_only=False)
    model = RCMKNContextModel(n_classes=spec["n_classes"])
    model.load_state_dict(ck["model_state"])
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    log(f"  [CTX] loaded {spec['ckpt']}")
    return model


def run(ds_name, device):
    from experiments.rcmkn_haptics_seed42.runner import (
        set_seed, extract_context_regimes, train_context_model)
    from experiments.rcmkn_haptics_seed42.model import RCMKNContextModel
    from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (
        compute_raw_activations, ppv_from_activations)
    from experiments.drtn_conditioned_minirocket_haptics_3seed.runner import (
        compute_regime_heterogeneity)
    from aeon.transformations.collection.convolution_based import MiniRocket

    t0 = time.time()
    spec = SPEC[ds_name]
    ds_dir = os.path.join(OUT, ds_name)
    os.makedirs(ds_dir, exist_ok=True)
    log(f"\n=== RHO SWEEP (R5/HERAMBA) -- {ds_name} ===")

    data = load_data(ds_name)
    ytr, yva, yte = data["ytr"], data["yva"], data["yte"]
    got = {"train": len(data["Xtr"]), "val": len(data["Xva"]),
           "test": len(data["Xte"]), "T": int(data["Xtr"].shape[1])}
    assert got == spec["expected"], f"split identity FAILED: {got}"
    y_dev = np.concatenate([ytr, yva])
    n_train = got["train"]

    Xtr_z, Xva_z, Xte_z = (znorm(data["Xtr"]), znorm(data["Xva"]),
                           znorm(data["Xte"]))
    Xtrva_z = np.vstack([Xtr_z, Xva_z])

    set_seed(SEED)
    extractor = MiniRocket(random_state=SEED, n_jobs=-1)
    extractor.fit(Xtr_z[:, None, :].astype(np.float32))
    F_trva = extractor.transform(Xtrva_z[:, None, :].astype(np.float32))
    F_te = extractor.transform(Xte_z[:, None, :].astype(np.float32))
    assert F_trva.shape[1] == TOTAL_BUDGET
    mr_id, valid = 0.0, None
    for c0 in range(0, len(Xtrva_z), 32):
        act, valid = compute_raw_activations(extractor, Xtrva_z[c0:c0 + 32])
        mr_id = max(mr_id, float(np.max(np.abs(
            ppv_from_activations(act, valid) - F_trva[c0:c0 + 32]))))
        del act
    assert mr_id < 1e-5, f"extractor identity {mr_id}"
    valid_het = valid[N_GLOBAL:]
    G_trva, G_te = F_trva, F_te          # full bank; slicing at assemble

    # H banks from the frozen seed-42 context
    model = get_context(ds_name, spec, Xtr_z, ytr, Xva_z, yva, device)
    set_seed(SEED)
    regimes_trva = extract_context_regimes(model, Xtrva_z, device, batch=64)
    set_seed(SEED)
    regimes_te = extract_context_regimes(model, Xte_z, device, batch=64)

    def het(X_z, regimes, chunk=32):
        H = np.empty((len(X_z), N_HET), dtype=np.float64)
        for c0 in range(0, len(X_z), chunk):
            c1 = min(c0 + chunk, len(X_z))
            act, _ = compute_raw_activations(extractor, X_z[c0:c1])
            H[c0:c1] = compute_regime_heterogeneity(
                act[:, N_GLOBAL:], valid_het, regimes[c0:c1])
            del act
        return H

    H_trva = het(Xtrva_z, regimes_trva)
    H_te = het(Xte_z, regimes_te)
    assert not np.isnan(H_trva).any() and not np.isnan(H_te).any()

    kfold = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    per_rho, fsel = {}, {}
    for rho in RHO_GRID:
        n_g, n_h = budget_split(rho)
        t_r = time.time()
        fold_scores, fold_top = [], []
        for tr_i, va_i in kfold.split(np.zeros(len(y_dev)), y_dev):
            if n_h > 0:
                f_stat, _ = f_classif(H_trva[tr_i], y_dev[tr_i])
                f_stat = np.nan_to_num(f_stat, nan=0.0)
                top = np.argsort(-f_stat, kind="stable")[:n_h]
            else:
                top = np.zeros(0, dtype=int)
            fold_top.append(top)
            X_tr = assemble(G_trva, H_trva, n_g, top)[tr_i]
            X_va = assemble(G_trva, H_trva, n_g, top)[va_i]
            ridge = RidgeClassifierCV(alphas=ALPHAS)
            ridge.fit(X_tr, y_dev[tr_i])
            fold_scores.append(macro_f1(
                y_dev[va_i], ridge.predict(X_va)))
        per_rho[str(rho)] = {
            "N_G": n_g, "N_H": n_h,
            "mean_cv_macro_f1": round(float(np.mean(fold_scores)), 4),
            "std_cv_macro_f1": round(float(np.std(fold_scores)), 4),
            "fold_scores": [round(s, 4) for s in fold_scores],
            "runtime_s": round(time.time() - t_r, 1)}
        if n_h > 0:
            f_stat, _ = f_classif(H_trva, y_dev)
            f_stat = np.nan_to_num(f_stat, nan=0.0)
            top_full = np.argsort(-f_stat, kind="stable")[:n_h]
            jac = [jaccard(fold_top[i], fold_top[j])
                   for i in range(len(fold_top))
                   for j in range(i + 1, len(fold_top))]
            fsel[str(rho)] = {
                "top_H_first20": top_full[:20].tolist(),
                "mean_F_selected": round(float(f_stat[top_full].mean()), 4),
                "fold_jaccard_mean": round(float(np.mean(jac)), 4)}
        log(f"  rho={rho}: {per_rho[str(rho)]['mean_cv_macro_f1']:.4f} "
            f"+/- {per_rho[str(rho)]['std_cv_macro_f1']:.4f} "
            f"(N_G={n_g}, N_H={n_h})")

    best = max(v["mean_cv_macro_f1"] for v in per_rho.values())
    rho_star = min(float(r) for r, v in per_rho.items()
                   if v["mean_cv_macro_f1"] >= best - TIE_TOL)
    n_g, n_h = budget_split(rho_star)
    log(f"  [RHO*] {rho_star}")

    if n_h > 0:
        f_stat, _ = f_classif(H_trva, y_dev)
        top = np.argsort(-np.nan_to_num(f_stat, nan=0.0),
                         kind="stable")[:n_h]
    else:
        top = np.zeros(0, dtype=int)
    ridge = RidgeClassifierCV(alphas=ALPHAS)
    ridge.fit(assemble(G_trva, H_trva, n_g, top), y_dev)
    pred_te = ridge.predict(assemble(G_te, H_te, n_g, top))
    test_f1 = round(macro_f1(yte, pred_te), 4)
    val_f1 = round(macro_f1(
        yva, ridge.predict(assemble(G_trva, H_trva, n_g, top)[n_train:])), 4)

    out = {
        "dataset": ds_name, "seed": SEED, "rho_grid": RHO_GRID,
        "per_rho": per_rho, "feature_selection": fsel,
        "selected_rho": rho_star, "N_G": n_g, "N_H": n_h,
        "val_macro_f1": val_f1, "test_macro_f1": test_f1,
        "selected_alpha": float(ridge.alpha_),
        "split": got,
        "audits": {
            "budget_exact_9996_all_rho": all(
                budget_split(r)[0] + budget_split(r)[1] == TOTAL_BUDGET
                for r in RHO_GRID),
            "ranking_train_fold_only": True,
            "single_test_evaluation": True,
            "extractor_identity_maxdiff": float(mr_id),
        },
        "runtime_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(ds_dir, "result.json"), "w") as f:
        json.dump(out, f, indent=2)
    log(f"  [{ds_name}] R5(rho*={rho_star}) test={test_f1} "
        f"({time.time() - t0:.0f}s)")
    return out


def train_seed42_context(ds_name, device):
    """Dataset has no stored context ckpt -> train the seed-42 context
    once with the frozen recipe and save it for the sweep + reuse."""
    from experiments.rcmkn_haptics_seed42.runner import (
        set_seed, train_context_model)
    from experiments.rcmkn_haptics_seed42.model import RCMKNContextModel
    os.makedirs(os.path.join(OUT, "_contexts"), exist_ok=True)
    path = os.path.join(OUT, "_contexts", f"{ds_name}_seed42.pt")
    if os.path.exists(path):
        log(f"[CTX] {ds_name} seed-42 context already trained")
        return
    data = load_data(ds_name)
    Xtr_z, Xva_z = znorm(data["Xtr"]), znorm(data["Xva"])
    set_seed(SEED)
    model = RCMKNContextModel(
        n_classes=SPEC[ds_name]["n_classes"]).to(device)
    train_info = train_context_model(model, Xtr_z, data["ytr"], Xva_z,
                                     data["yva"], device, smoke=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    torch.save({"model_state": model.state_dict(), "seed": SEED,
                "train_info": train_info}, path)
    log(f"[CTX] {ds_name} seed-42 context trained and saved ({path})")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT, exist_ok=True)
    for ds in ("Wafer", "ElectricDevices"):
        train_seed42_context(ds, device)
    all_res = {}
    for ds in ("Wafer", "ElectricDevices"):
        all_res[ds] = run(ds, device)
    with open(os.path.join(OUT, "rho_sweep_results.json"), "w") as f:
        json.dump(all_res, f, indent=2)
    log("\nRHO SWEEP COMPLETE")
    for ds, r in all_res.items():
        log(f"  {ds}: rho*={r['selected_rho']} test={r['test_macro_f1']}")


if __name__ == "__main__":
    main()
