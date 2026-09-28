"""TURS-RRMT-V2 diagnostics (D1-D6) on frozen V1 features.

Whereas the V1 diagnostic suite probed the routed transport model end-to-end,
V2 changed only the readout; so the questions here are about READOUTS:

  D1  Prediction reconstruction: all variant predictions are recomputed from
      cached artifacts (results/turs_rrmt_v2/<DS>/<V>_results.json hold only
      metrics; per-sample predictions were never persisted). R0/R1 rerun
      cheaply; R2/R3 refit from the cached feature matrices.
  D2  Per-variant error analysis: per-class F1, error rate, and
      macro-F1-of-minority-classes (the failure mode that matters here).
  D3  Uncertainty: routing entropy + ridge-margin confidence as error
      detectors (AUROC vs errors) on the frozen V1 representation.
  D4  Flavor usage: routing weight mass per flavor and its correlation with
      class identity (is the router class-aware?).
  D5  Lambda sensitivity: val-MF1 curves across the ridge lambda grid
      (recomputed cheaply from cached features).
  D6  Soft vs hard routing agreement: how often R4's soft-routing features
      change the ridge decision vs R2 hard-routed features (paired).

All randomness flows through src.diagnostics.statistics seeds.
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from experiments.turs_rrmt.data import SEED, DATASETS, NUM_CLASSES
from src.diagnostics import statistics as S

S.BOOTSTRAP_SEED = 8200
S.PERMUTATION_SEED = 8300
S.RNG_POOL = {}

MAX_N = 2000  # cap for AUROC/permutation experiments


# ---------------------------------------------------------------- helpers
def _load_v1_features(ds_tag, results_root):
    """Load cached V1 A4 features + routing intermediates for one dataset."""
    rep = os.path.join(results_root, ds_tag, "representation")
    f = np.load(os.path.join(rep, "v1_features_M128.npz"))
    out = dict(Xtr=f["Xtr"], ytr=f["ytr"], Xva=f["Xva"], yva=f["yva"],
               Xte=f["Xte"], yte=f["yte"])
    rp = os.path.join(rep, "v1_routing_M128.npz")
    if os.path.exists(rp):
        rd = np.load(rp)
        for k in rd.files:
            out[f"r_{k}"] = rd[k]
    return out


def _grid_align(a, target=160):
    a = np.asarray(a)
    ax = 2 if a.ndim == 3 else 1
    T = a.shape[ax]
    if T <= target:
        return a
    step = int(np.ceil(T / target))
    sl = [slice(None)] * a.ndim
    sl[ax] = slice(None, None, step)
    return a[tuple(sl)]


def routing_entropy(w):
    """Entropy of routing weights [N, J, T] -> [N] (mean over time)."""
    w = np.asarray(w, float)
    w = w / np.clip(w.sum(axis=1, keepdims=True), 1e-8, None)
    H = -(w * np.log(np.clip(w, 1e-12, None))).sum(axis=1)  # [N, T]
    return H.mean(axis=1)


def _margin_confidence(probs):
    """Top1 - top2 probability margin (higher = more confident)."""
    s = np.sort(probs, axis=1)
    return s[:, -1] - s[:, -2]


def _auroc(y_true, scores):
    from sklearn.metrics import roc_auc_score
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, float)
    ok = np.isfinite(scores)
    y_true, scores = y_true[ok], scores[ok]
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def _ridge_refit_mf1(Xtr, ytr, Xva, yva, lam):
    """Cheap sklearn ridge refit -> (val predictions, fitted pipeline)."""
    from sklearn.linear_model import RidgeClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    clf = make_pipeline(StandardScaler().fit(Xtr), RidgeClassifier(alpha=lam))
    clf.fit(Xtr, ytr)
    return clf.predict(Xva), clf


def _build_routed_psi(Tv_arr, Psc_arr, w_arr):
    """Routing-weighted kernel features Psi(x) for R5 (mirrors the runner).

    Args:
        Tv_arr: [B, J, T] transport features
        Psc_arr: [B, 2] precomputed pattern scalars (mean-of-mean, mean-of-max)
        w_arr: [B, J, T] routing weights
    Returns:
        Psi: [B, 2J+2]
    """
    Tv_t = torch.from_numpy(np.asarray(Tv_arr)).float()
    Psc_t = torch.from_numpy(np.asarray(Psc_arr)).float()
    w_t = torch.from_numpy(np.asarray(w_arr)).float()
    B, J, T = Tv_t.shape
    w_sqrt = torch.sqrt(w_t.clamp(min=1e-8))
    parts = []
    for j in range(J):
        parts.append((w_sqrt[:, j, :] * Tv_t[:, j, :]).mean(dim=-1))
    for j in range(J):
        parts.append(Tv_t[:, j, :].mean(dim=-1))
    parts.append(Psc_t[:, 0])
    parts.append(Psc_t[:, 1])
    return torch.stack(parts, dim=1)


# ================================================================ D1
def d1_prediction_reconstruction(ds_tag, features, results_root, device,
                                 hyp, variants=("R0", "R1", "R2", "R3",
                                                "R4", "R5", "R6", "R7")):
    """Recompute per-sample test predictions for each variant and persist.

    R0/R2/R3 come from cached V1 features (exact replay).
    R1/R6/R7 redo the sklearn ridge fit (deterministic, exact).
    R4's soft-routing forward and R5's kernel ridge are replayed from the
    cached routing intermediates.
    """
    from experiments.turs_rrmt.train_eval import _mf1

    yte = features["yte"]
    n_cls = NUM_CLASSES[ds_tag]
    out = {}
    pred_dir = os.path.join(results_root, ds_tag, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    # --- R0: V1 A4 predictions straight from the V1 run artifacts
    v1_root = os.path.join(os.path.dirname(results_root.rstrip("/")), "turs_rrmt")
    v1_npz = os.path.join(v1_root, ds_tag, "routing_outputs_test.npz")
    if os.path.exists(v1_npz):
        z = np.load(v1_npz)
        out["R0"] = dict(probs=z["probs"], pred=z["pred"])
    else:
        log = f"[{ds_tag}] D1: V1 routing_outputs_test.npz missing, R0 skipped"
        print(log)
        return out

    # --- R1: frozen ridge (exact replay: same grid, same seed-free pipeline)
    rp = os.path.join(results_root, ds_tag, "R1_results.json")
    best = None
    if os.path.exists(rp):
        best = json.load(open(rp)).get("selected_lambda")
    if best is not None:
        _, clf = _ridge_refit_mf1(features["Xtr"], features["ytr"],
                                  features["Xva"], features["yva"], best)
        out["R1"] = dict(pred=clf.predict(features["Xte"]))

    # --- R2/R3: differentiable ridge from cached features (exact replay)
    for v, rt in (("R2", "ridge"), ("R3", "class_weighted_ridge")):
        rp = os.path.join(results_root, ds_tag, f"{v}_results.json")
        if not os.path.exists(rp):
            continue
        meta = json.load(open(rp))
        lam = meta.get("selected_lambda")
        if lam is None:
            continue
        Ztr = torch.from_numpy(features["Xtr"]).float()
        ytr_t = torch.from_numpy(features["ytr"]).long()
        Zte = torch.from_numpy(features["Xte"]).float()
        C = n_cls
        D = Ztr.shape[1]
        Zm, Zs = Ztr.mean(0), Ztr.std(0).clamp_min(1e-6)
        Ztr_n = (Ztr - Zm) / Zs
        Zte_n = (Zte - Zm) / Zs
        Yoh = F.one_hot(ytr_t, C).to(Ztr.dtype)
        if rt == "class_weighted_ridge":
            from models.turs_rrmt_v2.readout import ClassWeightedRidge
            cw = ClassWeightedRidge(C, lam=1.0)
            cw.set_weights_from_train(features["ytr"])
            w = cw.class_weights[ytr_t]
            Zw = Ztr_n * w.unsqueeze(1)
        else:
            Zw = Ztr_n
        A = Zw.T @ Ztr_n + (lam + 1e-6) * torch.eye(D)
        W = torch.linalg.solve(A, Zw.T @ Yoh)
        out[v] = dict(pred=(Zte_n @ W).argmax(-1).numpy())

    # --- R4: soft-routing features (tau=2.0, same recipe as the runner)
    rp = os.path.join(results_root, ds_tag, "R4_results.json")
    if os.path.exists(rp) and all(k in features for k in
                                  ("r_Tv_tr", "r_w_tr", "r_Psc_tr")):
        soft = _soft_routing_features(ds_tag, results_root, device)
        if soft is not None:
            meta = json.load(open(rp))
            lam = meta.get("selected_lambda")
            if lam is not None:
                # NOTE: the original R4 fit used the torch one-hot closed-form
                # ridge (fit_ridge_readout), NOT sklearn RidgeClassifier —
                # replay the exact same formulation for a faithful replay.
                Ztr = torch.from_numpy(soft["Xtr"]).float()
                ytr_t = torch.from_numpy(soft["ytr"]).long()
                Zte = torch.from_numpy(soft["Xte"]).float()
                Zm, Zs = Ztr.mean(0), Ztr.std(0).clamp_min(1e-6)
                Ztr_n, Zte_n = (Ztr - Zm) / Zs, (Zte - Zm) / Zs
                Yoh = F.one_hot(ytr_t, n_cls).to(Ztr.dtype)
                D4d = Ztr.shape[1]
                A = Ztr_n.T @ Ztr_n + (lam + 1e-6) * torch.eye(D4d)
                W = torch.linalg.solve(A, Ztr_n.T @ Yoh)
                out["R4"] = dict(pred=(Zte_n @ W).argmax(-1).numpy())

    # --- R5: kernel ridge replay from cached Psi builder
    rp = os.path.join(results_root, ds_tag, "R5_results.json")
    if os.path.exists(rp) and all(k in features for k in
                                  ("r_Tv_tr", "r_w_tr", "r_Psc_tr")):
        from models.turs_rrmt_v2.readout import KernelRidgeReadout
        meta = json.load(open(rp))
        lam = meta.get("selected_lambda")
        if lam is not None:
            Psi_tr = _build_routed_psi(features["r_Tv_tr"], features["r_Psc_tr"],
                                       features["r_w_tr"])
            Psi_te = _build_routed_psi(features["r_Tv_te"], features["r_Psc_te"],
                                       features["r_w_te"])
            mu, sd = Psi_tr.mean(0), Psi_tr.std(0).clamp_min(1e-6)
            Psi_tr = (Psi_tr - mu) / sd
            Psi_te = (Psi_te - mu) / sd
            kr = KernelRidgeReadout(n_cls, J=4, lam=lam, kernel_type="linear")
            probs = kr(Psi_tr, torch.from_numpy(features["ytr"]).long(), Psi_te)
            out["R5"] = dict(pred=probs.argmax(-1).numpy(), probs=probs.numpy())

    # --- R6/R7: bigger-bank ridge (features cached by the runner)
    for v, M in (("R6", 256), ("R7", 512)):
        fpath = os.path.join(results_root, ds_tag, "representation",
                             f"v1_features_M{M}.npz")
        if not os.path.exists(fpath):
            continue
        fb = np.load(fpath)
        rp = os.path.join(results_root, ds_tag, f"{v}_results.json")
        if not os.path.exists(rp):
            continue
        lam = json.load(open(rp)).get("selected_lambda")
        if lam is None:
            continue
        _, clf = _ridge_refit_mf1(fb["Xtr"], fb["ytr"], fb["Xva"], fb["yva"], lam)
        out[v] = dict(pred=clf.predict(fb["Xte"]))

    # persist + sanity-check the reconstruction against recorded metrics
    recon = {}
    for v, rec in out.items():
        pred = np.asarray(rec["pred"])
        np.savez_compressed(os.path.join(pred_dir, f"{v}_test_pred.npz"),
                            pred=pred,
                            probs=rec.get("probs", np.zeros(0)))
        correct = (pred == yte)
        recon[v] = dict(
            n=len(pred),
            test_mf1_recomputed=_mf1(yte, pred, n_cls),
            error_rate=float(1 - correct.mean()),
            pred_class_counts=np.bincount(pred, minlength=n_cls).tolist())
    d1 = dict(dataset=ds_tag, n_test=int(len(yte)), variants=recon)

    # compare against recorded metric (reconstruction fidelity)
    for v, rec in recon.items():
        rp = os.path.join(results_root, ds_tag, f"{v}_results.json")
        if os.path.exists(rp):
            rec["recorded_mf1"] = json.load(open(rp))["test"]["macro_f1"]
            rec["replay_delta"] = round(rec["test_mf1_recomputed"]
                                        - rec["recorded_mf1"], 6)

    # second return value: raw per-sample predictions (NOT json-serialized)
    return d1, out


def _soft_routing_features(ds_tag, results_root, device):
    """Recompute R4 soft-routing features (tau=2.0) from the V1 checkpoint.

    Runs on the RAW time series (canonical split), not the cached pooled
    features — the V1 backbone needs [N, 1, T] inputs.
    """
    import torch
    from models.turs_rrmt.model import TURSRRMT as V1Model
    from experiments.turs_rrmt.data import load_split
    V1_CKPT = os.path.join(os.path.dirname(results_root.rstrip("/")),
                           "turs_rrmt", "checkpoints")

    v1_path = os.path.join(V1_CKPT, f"{ds_tag}_A4.pt")
    if not os.path.exists(v1_path):
        return None
    ds = load_split(ds_tag)
    ckpt = torch.load(v1_path, map_location="cpu", weights_only=False)
    m = V1Model(num_classes=ckpt["num_classes"], seq_len=ckpt["seq_len"],
                M=ckpt["M"], J=ckpt["J"], topk=ckpt["topk"],
                ppv_window=ckpt["ppv_window"], variant=ckpt["variant"],
                seed=ckpt["seed"], head_width=ckpt.get("head_width", 192))
    m.load_state_dict(ckpt["model_state_dict"])
    m.flavors.ref_q.copy_(torch.tensor(ckpt["flavor_ref"]["ref_q"]))
    m.flavors.ref_q_fine.copy_(torch.tensor(ckpt["flavor_ref"]["ref_q_fine"]))
    m.flavors.ref_lag.copy_(torch.tensor(ckpt["flavor_ref"]["ref_lag"]))
    m.flavors.fitted = True
    m.eval().to(device)

    tau = 2.0
    out = {}
    for split, y, ykey in [("Xtr", ds["y_train"], "ytr"),
                           ("Xva", ds["y_val"], "yva"),
                           ("Xte", ds["y_test"], "yte")]:
        X = ds[split]
        hs = []
        with torch.no_grad():
            for i in range(0, len(X), 256):
                xb = torch.from_numpy(X[i:i + 256]).float().to(device)
                R = m.bank(xb)
                A, Sv = m.activity(R)
                P = torch.cat([A, torch.log1p(Sv)], 1)
                Tv = m.flavors(xb, window=m.ppv_window)
                Tp = min(P.shape[-1], Tv.shape[-1])
                P, Tv = P[..., :Tp], Tv[..., :Tp]
                w_logits = m.router(P.permute(0, 2, 1))
                w_soft = F.softmax(w_logits / tau, dim=-1).permute(0, 2, 1)
                # A4 transport_proj maps J+1=5 channels (kept-4 + rest); the
                # soft variant keeps all 4 flavors at full mass and routes the
                # deficit into the 5th channel so the layer shapes match.
                Tv_padded = torch.cat([Tv, torch.zeros_like(Tv[:, :1, :])], dim=1)
                rest = 1.0 - w_soft.sum(dim=1, keepdim=True)
                w_padded = torch.cat([w_soft, rest.clamp(min=0)], dim=1)
                F_T = m.transport_proj(w_padded * Tv_padded)
                F_P = m.pattern_path(P.permute(0, 2, 1)).permute(0, 2, 1)
                F_all = torch.cat([F_T, F_P], dim=1)
                mu = F_all.mean(-1)
                mx = F_all.max(-1).values
                sd = F_all.std(-1) if F_all.shape[-1] > 1 else torch.zeros_like(mu)
                hs.append(torch.cat([mu, mx, sd], dim=1).cpu().numpy())
        out[split] = np.concatenate(hs, 0)
        out[ykey] = y
    return out


# ================================================================ D2
def d2_error_analysis(ds_tag, features, preds, hyp):
    """Per-class F1 and minority-class MF1 for every variant."""
    from sklearn.metrics import f1_score
    yte = features["yte"]
    n_cls = NUM_CLASSES[ds_tag]
    counts = np.bincount(features["ytr"], minlength=n_cls)
    minority = [c for c in range(n_cls) if counts[c] < np.median(counts)]
    if not minority:
        # balanced data: fall back to the least frequent classes
        minority = [int(np.argmin(counts))]
    out = dict(minority_classes=minority, variants={})
    for v, rec in preds.items():
        pred = np.asarray(rec["pred"])
        if len(pred) != len(yte):
            continue
        per_class = f1_score(yte, pred, average=None, labels=list(range(n_cls)),
                             zero_division=0)
        out["variants"][v] = dict(
            per_class_f1=[round(float(x), 4) for x in per_class],
            minority_mf1=round(float(np.mean(per_class[minority])), 4),
            error_rate=round(float(1 - (pred == yte).mean()), 4))
        hyp.add("D2_error_analysis", "per-class F1 (descriptive)",
                f"{ds_tag} {v} minority-class MF1",
                out["variants"][v]["minority_mf1"], None)
    return out


# ================================================================ D3
def d3_uncertainty(ds_tag, features, preds, hyp):
    """Routing entropy + confidence margin as error detectors (AUROC + CI)."""
    yte = features["yte"]
    if "r_w_te" not in features:
        return {}
    H = routing_entropy(features["r_w_te"])
    probs_r0 = preds.get("R0", {}).get("probs")
    if probs_r0 is None or len(probs_r0) == 0:
        return {}
    err_r0 = (np.asarray(preds["R0"]["pred"]) != yte).astype(int)
    margin = _margin_confidence(probs_r0)
    out = {"n": int(len(yte)), "signals": {}}
    for name, scores in (("routing_entropy", H), ("confidence_neg", -margin)):
        au, lo, hi = S.bootstrap_auroc_ci(err_r0, scores, n_boot=500)
        out["signals"][name] = dict(auroc=round(au, 4),
                                    ci95=[round(lo, 4), round(hi, 4)])
        if np.isfinite(au):
            hyp.add("D3_uncertainty", "AUROC + bootstrap CI (error detection)",
                    f"{ds_tag} {name} vs R0 errors", au, None)
    finite = {k: v["auroc"] for k, v in out["signals"].items()
              if np.isfinite(v["auroc"])}
    if finite:
        best = max(finite, key=finite.get)
        out["best_signal"] = dict(name=best, auroc=finite[best])
    return out


# ================================================================ D4
def d4_flavor_usage(ds_tag, features, hyp, max_n=MAX_N):
    """Routing mass per flavor + point-biserial-style link to class identity."""
    if "r_w_te" not in features:
        return {}
    W = _grid_align(features["r_w_te"])          # [N, J, T]
    yte = features["yte"]
    n_cls = int(yte.max() + 1)
    mass = W.mean(axis=(0, 2))                   # [J]
    per_sample_mass = W.mean(axis=2)             # [N, J]
    sub = slice(0, min(len(yte), max_n))
    # correlation of per-sample flavor mass with one-hot class indicators
    rows = {}
    for j in range(W.shape[1]):
        best = None
        for c in range(n_cls):
            r = S.corr_with_inference(per_sample_mass[sub, j],
                                      (yte[sub] == c).astype(float),
                                      n_perm=200)
            if best is None or abs(r["rho"]) > abs(best["rho"]):
                best = r
        rows[f"flavor_{j}"] = dict(mass=round(float(mass[j]), 4),
                                   best_class_rho=round(best["rho"], 4),
                                   p_perm=best["p_perm"])
        hyp.add("D4_flavor_usage", "Spearman + permutation",
                f"{ds_tag} flavor {j} mass vs class", best["rho"], best["p_perm"])
    return dict(per_flavor=rows)


# ================================================================ D5
def d5_lambda_sensitivity(ds_tag, features, results_root, hyp):
    """Val-MF1 vs lambda for R1-style ridge (cheap, from cached features)."""
    from experiments.turs_rrmt.train_eval import _mf1
    lam_grid = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    n_cls = NUM_CLASSES[ds_tag]
    curve = {}
    for lam in lam_grid:
        pred_va, _ = _ridge_refit_mf1(features["Xtr"], features["ytr"],
                                      features["Xva"], features["yva"], lam)
        curve[str(lam)] = round(_mf1(features["yva"], pred_va, n_cls), 4)
    best = max(curve, key=curve.get)
    out = dict(curve=curve, best_lambda=float(best),
               best_val_mf1=curve[best],
               sensitivity=round(max(curve.values()) - min(curve.values()), 4))
    return out


# ================================================================ D6
def d6_soft_vs_hard_agreement(ds_tag, features, preds, hyp):
    """How often R4 (soft routing) changes the decision vs R2 (hard top-K)."""
    if "R2" not in preds or "R4" not in preds:
        return {}
    p2 = np.asarray(preds["R2"]["pred"])
    p4 = np.asarray(preds["R4"]["pred"])
    agree = float((p2 == p4).mean())
    yte = features["yte"]
    # McNemar between the two readouts over the same representation
    c2, c4 = p2 == yte, p4 == yte
    chi2, p = S.mcnemar(c4, c2)
    hyp.add("D6_soft_vs_hard", "McNemar", f"{ds_tag} R4 vs R2", agree, p)
    return dict(agreement=round(agree, 4),
                n_disagree=int((p2 != p4).sum()),
                mcnemar_p=round(p, 5))


# ================================================================ entry point
def run_all_diagnostics(results_root, device, hyp, datasets=None):
    """Run D1-D6 for each dataset; returns {tag: {...}}."""
    datasets = datasets or list(DATASETS)
    all_out = {}
    for tag in datasets:
        print(f"[{tag}] V2 diagnostics D1-D6", flush=True)
        features = _load_v1_features(tag, results_root)
        d1, preds = d1_prediction_reconstruction(tag, features, results_root,
                                                 device, hyp)
        all_out[tag] = dict(
            D1=d1,
            D2=d2_error_analysis(tag, features, preds, hyp),
            D3=d3_uncertainty(tag, features, preds, hyp),
            D4=d4_flavor_usage(tag, features, hyp),
            D5=d5_lambda_sensitivity(tag, features, results_root, hyp),
            D6=d6_soft_vs_hard_agreement(tag, features, preds, hyp))
    return all_out
