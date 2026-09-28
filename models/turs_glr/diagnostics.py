"""TURS-GLR diagnostics.

Reuses the RRMT diagnostic machinery (src/diagnostics/statistics.py,
calibration.py, perturb.py) and the RRMT local-descriptor/correlation
helpers, and adds GLR-specific first-class analyses:

    - block contribution decomposition (exact, linear): f(x) = f_g(x) + f_l(x)
    - global-vs-local contribution by class / correctness / per sample
    - lambda- adaptation record (dataset-level, validation-selected)
    - routing descriptor correlations (w_j(t) vs independent descriptors)
    - routing intervention (uniform / shuffled / single-flavor)
    - routing faithfulness (targeted vs random masking)
    - routing stability under benign perturbations
    - counterfactual routing (noise in high-routing region)
    - uncertainty / selective prediction (confidence, entropy, margin,
      routing entropy, block disagreement)
"""

import numpy as np
import torch
import torch.nn.functional as F

from src.diagnostics import statistics as S
from src.diagnostics import perturb as PB
from src.diagnostics.calibration import (ece, adaptive_ece, brier, nll as nll_fn,
                                         reliability_curve)
from experiments.turs_rrmt.train_eval import full_metrics
from experiments.turs_rrmt.diagnostics import _local_descriptors, _rho

MAX_DIAG_N = 400
DEG_SPECS = {
    "gaussian_noise":  {"levels": [0.05, 0.10, 0.20, 0.40, 0.80]},
    "amplitude_scale": {"levels": [0.6, 0.8, 1.2]},
    "baseline_shift":  {"levels": [0.2, 0.4, 0.8]},
    "temporal_jitter": {"levels": [2, 8, 16]},
    "localized_mask":  {"levels": [0.05, 0.15, 0.30]},
}
BENIGN_SPECS = {
    "amp_scale_0.95": {"kind": "amplitude_scale", "level": 0.95},
    "amp_scale_1.05": {"kind": "amplitude_scale", "level": 1.05},
    "baseline_0.1sd": {"kind": "baseline_shift", "level": 0.1},
    "noise_0.02":     {"kind": "gaussian_noise", "level": 0.02},
    "jitter_2":       {"kind": "temporal_jitter", "level": 2},
}


def _model_device(model):
    """Device of the module's parameters (CPU if parameterless)."""
    import torch
    return next(model.parameters()).device if any(True for _ in model.parameters()) \
        else torch.device("cpu")


# =================================================================
# Block contribution decomposition (exact for the linear readout)
# =================================================================
def block_contributions(ridge, Zg, Zl, y, probs):
    """Exact linear decomposition f(x) = f_g(x) + f_l(x).

    Returns per-sample contribution magnitudes and correctness conditioning.
    Magnitudes are L2 norms of the per-block logit vectors; they are NOT
    normalized into percentages (no mathematically justified partition).
    """
    fg, fl = ridge.block_logits(Zg, Zl)
    fg, fl = np.asarray(fg), np.asarray(fl)
    pred = probs.argmax(1)
    out = dict(
        global_logit_norm=np.linalg.norm(fg, axis=1),
        local_logit_norm=np.linalg.norm(fl, axis=1),
        total_logit_norm=np.linalg.norm(fg + fl, axis=1),
        correct=(pred == np.asarray(y)).astype(int),
        pred=pred, y=np.asarray(y),
        fg=fg, fl=fl,
        ratio=np.linalg.norm(fl, axis=1) /
              (np.linalg.norm(fg, axis=1) + 1e-12),
    )
    return out


def block_contribution_summary(bc):
    """Aggregate block contribution record."""
    gl, ll = bc["global_logit_norm"], bc["local_logit_norm"]
    corr = bc["correct"].astype(bool)
    out = dict(
        n=int(len(gl)),
        global_norm_mean=float(gl.mean()), global_norm_std=float(gl.std()),
        local_norm_mean=float(ll.mean()), local_norm_std=float(ll.std()),
        ratio_median=float(np.median(bc["ratio"])),
        local_share_median=float(np.median(ll / (ll + gl + 1e-12))),
        correct_global=float(gl[corr].mean()) if corr.any() else None,
        correct_local=float(ll[corr].mean()) if corr.any() else None,
        incorrect_global=float(gl[~corr].mean()) if (~corr).any() else None,
        incorrect_local=float(ll[~corr].mean()) if (~corr).any() else None,
    )
    return out


def global_local_disagreement(ridge, Zg, Zl):
    """Fraction of samples where global-only and local-only argmax differ."""
    fg, fl = ridge.block_logits(Zg, Zl)
    return float((np.asarray(fg).argmax(1) != np.asarray(fl).argmax(1)).mean())


# =================================================================
# Routing diagnostics (descriptor correlations)
# =================================================================
def routing_descriptor_correlations(model, X, w, hyp, family="D2_routing_validity",
                                    max_n=200):
    """w: [N, J, T] numpy routing weights vs independent local descriptors."""
    n = min(len(X), max_n)
    W = w[:n]
    desc = _local_descriptors(X[:n])
    names = model.FLAVOR_NAMES
    out = {"per_flavor": {}}
    for j in range(W.shape[1]):
        rec = {}
        for name, Dv in desc.items():
            r = _rho(Dv, W[:, j, :])
            rec[name] = r
            hyp.add(family, "Spearman + permutation",
                    f"w_{names[j]}(t) vs {name}", r["rho"], r["p_perm"],
                    {"ci": [r["ci_lo"], r["ci_hi"]]})
        out["per_flavor"][names[j]] = rec
    # aggregate activity A(t) vs descriptors
    return out


def routing_summary_stats(w):
    """w: [N, J, T] -> dataset-level routing summary record."""
    ent = -(w * np.log(w + 1e-12)).sum(1).mean(1)
    top = w.argmax(1)
    persist = (top[:, 1:] == top[:, :-1]).mean(1) if w.shape[2] > 1 else np.ones(len(w))
    return dict(
        n=int(w.shape[0]), J=int(w.shape[1]), T=int(w.shape[2]),
        mean_w=float(w.mean()), std_w=float(w.std()),
        routing_entropy_mean=float(ent.mean()), routing_entropy_std=float(ent.std()),
        route_switch_rate=float(1 - persist.mean()),
        flavor_freq=[float(f) for f in
                     np.bincount(top.ravel(), minlength=w.shape[1])
                     / top.size],
        weight_margin_mean=float((np.sort(w, axis=1)[:, -1]
                                  - np.sort(w, axis=1)[:, -2]).mean()),
    )


# =================================================================
# Routing interventions on the frozen model + ridge
# =================================================================
def predict_with_modified_routing(model, ridge, scaler, Zg_fn, X, y, mode,
                                  seed=42, batch=128):
    """Recompute Z_local under a modified router, predict with FROZEN ridge.

    mode: 'uniform' | 'shuffled' | 'fixed_j' (j = flavor index)
    Zg_fn: callable X -> Z_global (features unchanged; global block frozen)
    Returns probs [N, C].
    """
    g = torch.Generator().manual_seed(seed)
    n = len(X)
    dev = _model_device(model)
    probs = []
    for i in range(0, n, batch):
        xb = torch.from_numpy(X[i:i + batch]).float().to(dev)
        B = xb.shape[0]
        # whole recompute is an inference-time control: run under no_grad so
        # build_local_block's numpy conversion stays legal
        with torch.no_grad():
            ls = model.local_stream(xb)
            P = ls["P"]
            Tv = model.flavors(xb, window=model.ppv_window)
            Tp = min(P.shape[-1], Tv.shape[-1])
            P, Tv = P[..., :Tp], Tv[..., :Tp]
            J = model.J
            if mode == "uniform":
                w = torch.full((B, J, Tp), 1.0 / J, device=xb.device)
            elif mode == "shuffled":
                w_real = model.router(P.permute(0, 2, 1)).permute(0, 2, 1)
                flat = w_real.detach().permute(0, 2, 1).reshape(-1, J)
                idx = torch.randperm(flat.shape[0], generator=g).to(flat.device)
                w = flat[idx].reshape(B, Tp, J).permute(0, 2, 1)
            elif mode.startswith("fixed_"):
                j = int(mode.split("_")[1])
                w = torch.zeros(B, J, Tp, device=xb.device)
                w[:, j, :] = 1.0
            else:
                raise ValueError(mode)
            U = w * Tv
            A = ls["A"][..., :Tp]
            S_norm = ls["S_norm"][..., :Tp]
            from models.turs_glr.feature_blocks import build_local_block
            Zl = build_local_block(U, w, A, S_norm)
        Zg = Zg_fn(X[i:i + batch])
        Zgs, Zls = scaler.transform(Zg, Zl)
        probs.append(ridge.predict_proba(Zgs, Zls))
    return np.concatenate(probs, 0)


# =================================================================
# Faithfulness, stability, counterfactual
# =================================================================
def routing_faithfulness(model, ridge, scaler, Zg_fn, X, y, hyp=None,
                         rng_seed=7400, n_max=200, frac=0.10):
    """Perturb top-routing regions vs matched random regions (mask to mean)."""
    rng = np.random.default_rng(rng_seed)
    n = min(len(X), n_max)
    idx = rng.choice(len(X), n, replace=False)
    X, y = X[idx], np.asarray(y)[idx]
    T = X.shape[-1]
    width = max(3, int(frac * T))
    dev = _model_device(model)

    # routing importance map (router weight x flavor energy)
    imps = []
    base_probs = []
    for i in range(0, n, 128):
        xb = torch.from_numpy(X[i:i + 128]).float().to(dev)
        with torch.no_grad():
            d = model.forward_diag(xb)
            imp = (d["w"].max(1).values * d["Tv"].abs().mean(1)).cpu().numpy()
            ls = model.local_stream(xb)
            P = ls["P"][..., :d["w"].shape[-1]]
            A = ls["A"][..., :d["w"].shape[-1]]
            S_norm = ls["S_norm"][..., :d["w"].shape[-1]]
            U = d["w"] * d["Tv"]
            from models.turs_glr.feature_blocks import build_local_block
            Zl = build_local_block(U, d["w"], A, S_norm)
            Zg = Zg_fn(X[i:i + 128])
            Zgs, Zls = scaler.transform(Zg, Zl)
            base_probs.append(ridge.predict_proba(Zgs, Zls))
        imps.append(imp)
    imp = np.concatenate(imps, 0)
    base_probs = np.concatenate(base_probs, 0)
    base_pred, base_conf = base_probs.argmax(1), base_probs.max(1)

    target_drops, rand_drops, flips = [], [], 0
    for i in range(n):
        order = np.argsort(-imp[i])[:width]
        r_start = rng.integers(0, T - width)
        r_center = np.arange(r_start, r_start + width)
        for kind, center, store in [("target", order, target_drops),
                                    ("random", r_center, rand_drops)]:
            Xp = X[i:i + 1].copy()
            Xp[0, 0, center] = X[i, 0].mean()
            xb = torch.from_numpy(Xp).float().to(dev)
            with torch.no_grad():
                d = model.forward_diag(xb)
                ls = model.local_stream(xb)
                P = ls["P"][..., :d["w"].shape[-1]]
                A = ls["A"][..., :d["w"].shape[-1]]
                S_norm = ls["S_norm"][..., :d["w"].shape[-1]]
                U = d["w"] * d["Tv"]
                Zl = build_local_block(U, d["w"], A, S_norm)
                Zg = Zg_fn(Xp)
                Zgs, Zls = scaler.transform(Zg, Zl)
                p = ridge.predict_proba(Zgs, Zls)
            store.append(base_conf[i] - p.max(1)[0])
            if kind == "target" and p.argmax(1)[0] != base_pred[i]:
                flips += 1
    t, r = np.array(target_drops), np.array(rand_drops)
    diff = t - r
    _, lo, hi = S.bootstrap_ci(diff)
    _, p_perm = S.paired_permutation_test(t, r)
    out = dict(level="router", frac=frac, width=width, n=n,
               target_drop=float(t.mean()), random_drop=float(r.mean()),
               diff=float(diff.mean()), ci95=[float(lo), float(hi)],
               p_perm=float(p_perm), flip_rate=float(flips / n))
    if hyp is not None:
        hyp.add("D6_routing_faithfulness", "paired permutation",
                "targeted vs random region confidence drop", out["diff"],
                out["p_perm"], {"ci": out["ci95"], "target": out["target_drop"],
                                "random": out["random_drop"]})
    return out


def routing_stability(model, ridge, scaler, Zg_fn, X, y, hyp, rng_seed=7600,
                      n_max=200):
    """Benign perturbations: routing cosine + prob KL on frozen model."""
    rng = np.random.default_rng(rng_seed)
    X = X[:n_max]
    y = np.asarray(y)[:n_max]
    out = {"specs": {}}
    dev = _model_device(model)

    def full_predict(Xa):
        ps = []
        for i in range(0, len(Xa), 128):
            xb = torch.from_numpy(Xa[i:i + 128]).float().to(dev)
            with torch.no_grad():
                d = model.forward_diag(xb)
                ls = model.local_stream(xb)
                P = ls["P"][..., :d["w"].shape[-1]]
                A = ls["A"][..., :d["w"].shape[-1]]
                S_norm = ls["S_norm"][..., :d["w"].shape[-1]]
                U = d["w"] * d["Tv"]
                from models.turs_glr.feature_blocks import build_local_block
                Zl = build_local_block(U, d["w"], A, S_norm)
                Zg = Zg_fn(Xa[i:i + 128])
                Zgs, Zls = scaler.transform(Zg, Zl)
            ps.append(ridge.predict_proba(Zgs, Zls))
        return np.concatenate(ps, 0)

    p0 = full_predict(X)
    w0 = None
    with torch.no_grad():
        d0 = model.forward_diag(torch.from_numpy(X).float().to(dev))
        w0 = d0["w"].cpu().numpy()
    for name, spec in BENIGN_SPECS.items():
        Xp = np.stack([PB.apply_spec(x[0], spec["kind"], spec["level"], rng)[0]
                       for x in X])[:, None, :]
        with torch.no_grad():
            d1 = model.forward_diag(torch.from_numpy(Xp).float().to(dev))
        w1 = d1["w"].cpu().numpy()
        Tm = min(w0.shape[-1], w1.shape[-1])
        cos = float(np.mean(np.sum(w0[..., :Tm] * w1[..., :Tm], axis=1) /
                            (np.linalg.norm(w0[..., :Tm], axis=1) *
                             np.linalg.norm(w1[..., :Tm], axis=1) + 1e-12)))
        p1 = full_predict(Xp)
        kl = float((p0 * np.log((p0 + 1e-12) / (p1 + 1e-12))).sum(1).mean())
        top1_agree = float((w0.argmax(1)[:, :Tm] == w1.argmax(1)[:, :Tm]).mean())
        out["specs"][name] = dict(cosine_routing=cos, top1_route_agreement=top1_agree,
                                  prob_KL=kl)
        hyp.add("D7_routing_stability", "descriptive", f"{name}: routing cosine",
                cos, None, {})
    return out


def counterfactual_routing(model, ridge, scaler, Zg_fn, X, y, rng_seed=7700,
                           n_max=200):
    """Replace the top-routing region with noise; does routing + prediction move?"""
    rng = np.random.default_rng(rng_seed)
    X = X[:n_max]
    y = np.asarray(y)[:n_max]
    T = X.shape[-1]
    width = max(3, int(0.10 * T))
    dev = _model_device(model)

    def extract(Xa):
        ds, ls_out = [], []
        for i in range(0, len(Xa), 128):
            xb = torch.from_numpy(Xa[i:i + 128]).float().to(dev)
            with torch.no_grad():
                d = model.forward_diag(xb)
                ls = model.local_stream(xb)
            ds.append(d)
            ls_out.append(ls)
        return ds, ls_out

    def assemble(d_list, ls_list):
        ws = torch.cat([d["w"] for d in d_list], 0)
        Tvs = torch.cat([d["Tv"] for d in d_list], 0)
        As = torch.cat([l["A"] for l in ls_list], 0)
        Ss = torch.cat([l["S_norm"] for l in ls_list], 0)
        Ps = torch.cat([l["P"] for l in ls_list], 0)
        Tp = min(ws.shape[-1], As.shape[-1], Ps.shape[-1])
        from models.turs_glr.feature_blocks import build_local_block
        with torch.no_grad():
            U = ws[..., :Tp] * Tvs[..., :Tp]
            Zl = build_local_block(U, ws[..., :Tp], As[..., :Tp], Ss[..., :Tp])
        return ws, U, Zl

    d0, l0 = extract(X)
    w0, U0, Zl0 = assemble(d0, l0)
    with torch.no_grad():
        imp = (w0.max(1).values * U0.abs().mean(1)).cpu().numpy()
    Xp = X.copy()
    for i in range(len(X)):
        c = int(np.argmax(imp[i]))
        lo, hi = max(0, c - width // 2), min(T, c + width // 2)
        if hi > lo:
            Xp[i, 0, lo:hi] = rng.normal(0, 1, hi - lo)
    d1, l1 = extract(Xp)
    w1, U1, Zl1 = assemble(d1, l1)
    Tm = min(w0.shape[-1], w1.shape[-1])
    with torch.no_grad():
        shift = float((w1[..., :Tm] - w0[..., :Tm]).abs().mean().item())
        top1_change = float((w0.argmax(1)[:, :Tm] != w1.argmax(1)[:, :Tm])
                            .float().mean().item())
    # predictions
    Zg = Zg_fn(X)
    Zgp = Zg_fn(Xp)
    Zgs0, Zls0 = scaler.transform(Zg, Zl0)
    Zgs1, Zls1 = scaler.transform(Zgp, Zl1)
    p0 = ridge.predict_proba(Zgs0, Zls0)
    p1 = ridge.predict_proba(Zgs1, Zls1)
    out = dict(level="router", routing_weight_shift=shift,
               top1_route_change_rate=top1_change,
               pred_flip_rate=float((p1.argmax(1) != p0.argmax(1)).mean()),
               mean_conf_change=float((p1.max(1) - p0.max(1)).mean()),
               note="counterfactual: noise replaced in top-routing region")
    return out


# =================================================================
# Uncertainty / selective prediction
# =================================================================
def uncertainty_analysis(probs, y, w, bc):
    """Baseline uncertainty measures for a ridge model + routing signals.

    Terminology note: these are *scores correlated with error*, evaluated as
    error-detection AUROC; they are not calibrated posterior uncertainties.
    """
    y = np.asarray(y)
    yerr = (probs.argmax(1) != y).astype(int)
    conf = probs.max(1)
    ent = -(probs * np.log(probs + 1e-12)).sum(1)
    srt = np.sort(probs, 1)
    margin = srt[:, -1] - srt[:, -2]
    r_ent = -(w * np.log(w + 1e-12)).sum(1).mean(1)
    # block disagreement: global-only vs local-only argmax mismatch
    dis = (bc["fg"].argmax(1) != bc["fl"].argmax(1)).astype(float)
    scores = {
        "confidence_neg": -conf, "predictive_entropy": ent, "margin_neg": -margin,
        "routing_entropy": r_ent,
        "block_disagreement": dis,
        "local_share": bc["local_logit_norm"] /
                       (bc["local_logit_norm"] + bc["global_logit_norm"] + 1e-12),
    }
    out = {"signals": {}}
    for name, sc in scores.items():
        auc, lo, hi = S.bootstrap_auroc_ci(yerr, sc, n_boot=1000)
        from sklearn.metrics import average_precision_score
        out["signals"][name] = dict(auroc=auc, ci=[lo, hi],
                                    auprc=float(average_precision_score(yerr, sc)))
    # risk-coverage for confidence + entropy
    def rc(score):
        order = np.argsort(-score)
        correct = (probs.argmax(1) == y).astype(float)
        rows = []
        for cov in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
            k = max(1, int(round(cov * len(y))))
            sel = order[:k]
            rows.append(dict(coverage=cov, risk=float(1 - correct[sel].mean())))
        cum = np.cumsum(1 - correct[order]) / np.arange(1, len(y) + 1)
        aurc = float(np.trapezoid(cum, np.arange(1, len(y) + 1) / len(y)))
        return dict(rows=rows, aurc=aurc)
    out["risk_coverage"] = {"confidence": rc(-conf), "entropy": rc(ent),
                            "routing_entropy": rc(r_ent)}
    return out


# =================================================================
# Calibration
# =================================================================
def calibration_record(probs, y):
    y = np.asarray(y)
    return dict(
        ece=ece(probs, y), adaptive_ece=adaptive_ece(probs, y),
        brier=brier(probs, y), nll=nll_fn(probs, y),
        reliability_curve=reliability_curve(probs, y),
        mean_confidence=float(probs.max(1).mean()))


# =================================================================
# Robustness (degradations)
# =================================================================
def degradation_analysis(model, ridge, scaler, Zg_fn, X, y, hyp,
                         rng_seed=7500, n_max=MAX_DIAG_N):
    rng = np.random.default_rng(rng_seed)
    n = min(len(X), n_max)
    idx = rng.choice(len(X), n, replace=False)
    X, y = X[idx], np.asarray(y)[idx]
    out = {"kinds": {}}
    from models.turs_glr.feature_blocks import build_local_block
    dev = _model_device(model)

    def predict_and_route(Xa):
        ps, ws = [], []
        for i in range(0, len(Xa), 128):
            xb = torch.from_numpy(Xa[i:i + 128]).float().to(dev)
            with torch.no_grad():
                d = model.forward_diag(xb)
                ls = model.local_stream(xb)
                P = ls["P"][..., :d["w"].shape[-1]]
                A = ls["A"][..., :d["w"].shape[-1]]
                S_norm = ls["S_norm"][..., :d["w"].shape[-1]]
                U = d["w"] * d["Tv"]
                Zl = build_local_block(U, d["w"], A, S_norm)
                Zg = Zg_fn(Xa[i:i + 128])
                Zgs, Zls = scaler.transform(Zg, Zl)
                ps.append(ridge.predict_proba(Zgs, Zls))
                ws.append(d["w"].cpu().numpy())
        return np.concatenate(ps, 0), np.concatenate(ws, 0)

    for kind, spec in DEG_SPECS.items():
        recs = []
        for lev in spec["levels"]:
            Xd = np.stack([PB.apply_spec(x, kind, lev, rng)[0] for x in X])
            probs, w = predict_and_route(Xd)
            pred = probs.argmax(1)
            ent = -(w * np.log(w + 1e-12)).sum(1).mean(1)
            recs.append(dict(
                level=float(lev),
                error_rate=float((pred != y).mean()),
                macro_f1=float(__import__("sklearn").metrics.f1_score(
                    y, pred, average="macro", zero_division=0)),
                mean_conf=float(probs.max(1).mean()),
                routing_entropy=float(ent.mean()),
                route_switch=float((w.argmax(1)[:, 1:] != w.argmax(1)[:, :-1]).mean())))
        lev = np.array([r["level"] for r in recs])
        for key in ["error_rate", "mean_conf", "routing_entropy"]:
            vals = np.array([r[key] for r in recs])
            rho = float(np.corrcoef(np.argsort(np.argsort(lev)),
                                    np.argsort(np.argsort(vals)))[0, 1]) \
                if len(lev) > 2 and np.std(vals) > 0 else None
            recs[-1][f"severity_vs_{key}_rho"] = rho
            if rho is not None:
                hyp.add("D12_robustness", "Spearman over severity levels",
                        f"{kind}: severity vs {key}", rho, None,
                        {"n_levels": len(lev)})
        out["kinds"][kind] = recs
    return out
