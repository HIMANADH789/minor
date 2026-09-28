"""TURS-RRMT diagnostic validation (D1-D9) on the frozen A4 model."""
import os

import numpy as np
import torch
import torch.nn.functional as F

from experiments.turs_rrmt.data import SEED
from src.diagnostics import statistics as S
from src.diagnostics import perturb as PB

S.BOOTSTRAP_SEED = 7200
S.PERMUTATION_SEED = 7300
S.RNG_POOL = {}

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
MAX_DIAG_N = 400       # cap per-experiment sample count for runtime


def _local_descriptors(X):
    """Independent per-position input descriptors. X: [N, 1, T] -> dict of [N, T].
    Windows downsampled to ~160 positions per sample to bound correlation cost."""
    x2 = torch.from_numpy(X[:, 0, :]).float()          # [N, T]
    x = x2.unsqueeze(1)                                # [N, 1, T]
    N, T = x2.shape
    eps = 1e-8
    k = 9
    xp = F.pad(x, (k // 2, k // 2), mode="reflect")     # [N, 1, T+8]
    mean = F.avg_pool1d(xp, k, stride=1)               # [N, 1, T]
    var = F.avg_pool1d(xp ** 2, k, stride=1) - mean ** 2
    m2 = var + eps
    xc = (x - mean)                                    # centered [N, 1, T]
    m3 = F.avg_pool1d(F.pad(xc ** 3, (k // 2, k // 2), mode="reflect"),
                      k, stride=1)
    m4 = F.avg_pool1d(F.pad(xc ** 4, (k // 2, k // 2), mode="reflect"),
                      k, stride=1)
    lvar = var.squeeze(1)
    d1 = (x2[:, 1:] - x2[:, :-1]).abs()
    d1 = F.pad(d1.unsqueeze(1), (0, 1)).squeeze(1)
    skew = (m3 / m2 ** 1.5).squeeze(1)
    kurt = (m4 / m2 ** 2 - 3.0).squeeze(1)
    out = dict(local_variance=lvar, local_energy=lvar, first_diff=d1,
               skewness=skew, kurtosis=kurt)
    if T > 160:  # align temporal lengths across descriptors and routing maps
        step = int(np.ceil(T / 160))
        for kk in out:
            out[kk] = out[kk][:, ::step]
    return out


def _rho(x, y, max_pts=20000, rng_seed=7900):
    """Spearman with permutation p + bootstrap CI, subsampled for tractability.
    Arrays may be [N, T_a] and [N, T_b]: truncated to the common length along
    axis 1 before flattening (window centers approximately aligned)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.ndim == 2 and y.ndim == 2:
        tmin = min(x.shape[1], y.shape[1])
        x, y = x[:, :tmin], y[:, :tmin]
    x = x.ravel()
    y = y.ravel()
    if len(x) > max_pts:
        idx = np.random.default_rng(rng_seed).choice(len(x), max_pts, replace=False)
        x, y = x[idx], y[idx]
    return S.corr_with_inference(x, y, "spearman")


# ------------------------------------------------ D1 pattern-activity validity
def _grid_align(arr, target=160):
    """Temporal subsample to ~target positions.
    [N, T] -> subsample axis 1; [N, J, T] -> subsample axis 2."""
    arr = np.asarray(arr)
    if arr.ndim == 3:
        T = arr.shape[2]
        if T <= target:
            return arr
        step = int(np.ceil(T / target))
        return arr[:, :, ::step]
    T = arr.shape[1]
    if T <= target:
        return arr
    step = int(np.ceil(T / target))
    return arr[:, ::step]


def d1_pattern_activity(model, ds, device, hyp, ext):
    e = ext
    A = _grid_align(e["A_mean"])                       # [N, ~T] mean over M
    desc = _local_descriptors(e["X"])
    out = {"level": "router_input", "descriptors": {}}
    for name, Dv in desc.items():
        r = _rho(Dv, A)
        out["descriptors"][name] = r
        hyp.add("D1_pattern_activity", "Spearman + permutation",
                f"A(t) vs {name}", r["rho"], r["p_perm"],
                {"ci": [r["ci_lo"], r["ci_hi"]]})
    return out


# ------------------------------------------------ D2 routing validity
def d2_routing_validity(model, ds, device, hyp, ext):
    e = ext
    W = _grid_align(e["routing_weights"])              # [N, J, ~T]
    desc = _local_descriptors(e["X"])
    out = {"level": "router", "per_flavor": {}}
    J = W.shape[1]                                     # NB: layout is [N, J, T]
    names = model.FLAVORS
    for j in range(J):
        wj = W[..., j]
        rec = {}
        for name, Dv in desc.items():
            r = _rho(Dv, wj)
            rec[name] = r
            hyp.add("D2_routing_validity", "Spearman + permutation",
                    f"w_{names[j]}(t) vs {name}", r["rho"], r["p_perm"],
                    {"ci": [r["ci_lo"], r["ci_hi"]]})
        # lag-disagreement descriptor: local |first difference|
        ld = np.abs(np.diff(e["X"][:, 0, :], axis=1, prepend=e["X"][:, 0, :1]))
        r = _rho(ld, wj)
        rec["lag_disagreement"] = r
        hyp.add("D2_routing_validity", "Spearman + permutation",
                f"w_{names[j]}(t) vs lag_disagreement", r["rho"], r["p_perm"],
                {"ci": [r["ci_lo"], r["ci_hi"]]})
        out["per_flavor"][names[j]] = rec
    return out


# ------------------------------------------------ D3 routing intervention
def d3_routing_intervention(model, ds, device, hyp, A4_probs, uni, fixed0, shuf):
    y = ds["y_test"]
    from experiments.turs_rrmt.train_eval import full_metrics
    from src.diagnostics.statistics import mcnemar, cliffs_delta, paired_permutation_test
    res = {"level": "final", "variants": {}}
    from src.diagnostics.statistics import cliffs_delta as _cd
    for name, probs in [("actual", A4_probs), ("uniform", uni),
                        ("fixed_flavor0", fixed0), ("shuffled", shuf)]:
        pred = probs.argmax(1)
        conf = probs.max(1)
        nll = -np.log(probs[np.arange(len(y)), y] + 1e-12)
        res["variants"][name] = dict(
            **full_metrics(y, pred, ds["n_cls"]),
            mean_conf=float(conf.mean()), mean_nll=float(nll.mean()))
    a4c = (A4_probs.argmax(1) == y).astype(float)
    res["flip_vs_actual"] = {
        k: float((v.argmax(1) != A4_probs.argmax(1)).mean())
        for k, v in [("uniform", uni), ("fixed_flavor0", fixed0), ("shuffled", shuf)]}
    res["shuffled_vs_actual"] = dict(
        mcnemar_p=mcnemar((shuf.argmax(1) == y).astype(float), a4c)[1],
        cliffs_delta_acc=_cd((shuf.argmax(1) == y).astype(int), a4c.astype(int)))
    hyp.add("D3_routing_intervention", "McNemar paired",
            "shuffled vs actual routing accuracy",
            res["shuffled_vs_actual"]["cliffs_delta_acc"],
            res["shuffled_vs_actual"]["mcnemar_p"], {})
    return res


# ------------------------------------------------ D4 routing faithfulness
def d4_routing_faithfulness(model, ds, device, hyp, rng_seed=7400, n_max=200):
    """Perturb top-routing regions vs matched random regions."""
    rng = np.random.default_rng(rng_seed)
    X = ds["Xte"]
    n = min(len(X), n_max)
    idx = rng.choice(len(X), n, replace=False)
    X = X[idx]
    y = ds["y_test"][idx]
    T = X.shape[-1]
    frac = 0.10                                        # [PRE-REGISTERED]
    width = max(3, int(frac * T))
    model.eval()

    def routed_score(xb):
        with torch.no_grad():
            xt = torch.from_numpy(xb).float().to(device)
            R = model.bank(xt)
            A, Sv = model.activity(R)
            P = torch.cat([A, torch.log1p(Sv)], 1)
            Tv = model.flavors(xt, window=model.ppv_window)
            Tp = min(P.shape[-1], Tv.shape[-1])
            P, Tv = P[..., :Tp], Tv[..., :Tp]
            w = torch.softmax(model.router(P.permute(0, 2, 1)),
                              -1).permute(0, 2, 1)
            # routing importance: max weight * total flavor energy
            imp = w.max(1).values * Tv.abs().mean(1)   # [B, T]
        return imp.cpu().numpy()

    def perturb_eval(xb_t, regions):
        with torch.no_grad():
            xt = torch.from_numpy(xb_t).float().to(device)
            logits = model(xt)
            p = F.softmax(logits, 1).cpu().numpy()
        return p

    target_drops, rand_drops, flips = [], [], 0
    base_probs = perturb_eval(X, None)
    base_pred = base_probs.argmax(1)
    base_conf = base_probs.max(1)
    imp = routed_score(X)                              # [n, T]
    for i in range(n):
        order = np.argsort(-imp[i])
        t_center = order[:width]
        r_start = rng.integers(0, T - width)
        r_center = np.arange(r_start, r_start + width)
        for mask_kind, center, store in [("target", t_center, target_drops),
                                         ("random", r_center, rand_drops)]:
            Xp = X[i:i + 1].copy()
            Xp[0, 0, center] = 0.0                     # masking perturbation
            p = perturb_eval(Xp, None)
            store.append(base_conf[i] - p.max(1)[0])
            if mask_kind == "target" and p.argmax(1)[0] != base_pred[i]:
                flips += 1
    target_drops = np.array(target_drops)
    rand_drops = np.array(rand_drops)
    diff = target_drops - rand_drops
    ci = S.bootstrap_ci(diff)[1:]
    perm = S.paired_permutation_test(target_drops, rand_drops)
    out = dict(level="router", frac=frac, width=width, n=n,
               target_drop=float(target_drops.mean()),
               random_drop=float(rand_drops.mean()),
               diff=float(diff.mean()), ci95=[float(ci[0]), float(ci[1])],
               p_perm=float(perm[1]), flip_rate=float(flips / n))
    hyp.add("D4_routing_faithfulness", "paired permutation",
            "targeted vs random region conf-drop", out["diff"], out["p_perm"],
            {"ci": out["ci95"], "target": out["target_drop"],
             "random": out["random_drop"]})
    return out


# ------------------------------------------------ D5 controlled degradation
def d5_degradation(model, ds, device, hyp, rng_seed=7500, n_max=MAX_DIAG_N):
    rng = np.random.default_rng(rng_seed)
    X, y = ds["Xte"], ds["y_test"]
    n = min(len(X), n_max)
    idx = rng.choice(len(X), n, replace=False)
    X, y = X[idx], y[idx]
    out = {"level": "router+final", "kinds": {}}
    for kind, spec in DEG_SPECS.items():
        recs = []
        for lev in spec["levels"]:
            Xd = np.stack([PB.apply_spec(x, kind, lev, rng)[0] for x in X])
            with torch.no_grad():
                xt = torch.from_numpy(Xd).float().to(device)
                logits, o = model(xt, return_aux=True)
                probs = F.softmax(logits, 1).cpu().numpy()
                w = o["w"].cpu()                            # [B, J, T]
            pred = probs.argmax(1)
            r_ent = -(w.numpy() * np.log(w.numpy() + 1e-12)).sum(1).mean(1)
            recs.append(dict(
                level=float(lev),
                error_rate=float((pred != y).mean()),
                mean_conf=float(probs.max(1).mean()),
                final_entropy=float(-(probs * np.log(probs + 1e-12)).sum(1).mean()),
                routing_entropy=float(r_ent.mean()),
                top1_rate=float((w.argmax(1) == 0).float().mean()),
                route_switch=float((w.argmax(1)[:, 1:] !=
                                    w.argmax(1)[:, :-1]).float().mean())))
        # severity vs signal: Spearman over levels
        lev = np.array([r["level"] for r in recs])
        for key in ["error_rate", "mean_conf", "final_entropy", "routing_entropy"]:
            vals = np.array([r[key] for r in recs])
            rho = float(np.corrcoef(np.argsort(np.argsort(lev)),
                                    np.argsort(np.argsort(vals)))[0, 1]) \
                if len(lev) > 2 else None
            recs[-1][f"severity_vs_{key}_rho"] = rho
            if rho is not None:
                hyp.add("D5_degradation", "Spearman over severity levels",
                        f"{kind}: severity vs {key}", rho, None,
                        {"n_levels": len(lev)})
        out["kinds"][kind] = recs
    return out


# ------------------------------------------------ D6 uncertainty value
def d6_uncertainty(model, ds, device, hyp, A4_probs, ext):
    y = ds["y_test"]
    yerr = (A4_probs.argmax(1) != y).astype(int)
    W = _grid_align(ext["routing_weights"])            # [N, J, T]
    r_ent = -(W * np.log(W + 1e-12)).sum(1).mean(1)    # entropy over J, mean over T
    conf = A4_probs.max(1)
    ent = -(A4_probs * np.log(A4_probs + 1e-12)).sum(1)
    srt = np.sort(A4_probs, 1)
    margin = srt[:, -1] - srt[:, -2]
    scores = {"confidence_neg": -conf, "predictive_entropy": ent,
              "margin_neg": -margin, "routing_entropy": r_ent,
              "routing_concentration_neg": -W.max(-1).mean(1)}
    out = {"level": "final+router", "signals": {}}
    for name, sc in scores.items():
        auc, lo, hi = S.bootstrap_auroc_ci(yerr, sc)
        from sklearn.metrics import average_precision_score
        out["signals"][name] = dict(auroc=auc, ci=[lo, hi],
                                    auprc=float(average_precision_score(yerr, sc)))
    # routing entropy adds beyond confidence?
    hyp.add("D6_uncertainty", "bootstrap AUROC",
            "routing_entropy error detection", out["signals"]["routing_entropy"]["auroc"],
            None, {"ci": out["signals"]["routing_entropy"]["ci"]})
    hyp.add("D6_uncertainty", "bootstrap AUROC",
            "confidence error detection", out["signals"]["confidence_neg"]["auroc"],
            None, {"ci": out["signals"]["confidence_neg"]["ci"]})
    # risk-coverage with routing entropy vs confidence
    def rc(score, higher_uncertain=True):
        order = np.argsort(-score) if higher_uncertain else np.argsort(score)
        correct = (A4_probs.argmax(1) == y).astype(float)
        rows = []
        for cov in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
            k = max(1, int(round(cov * len(y))))
            sel = order[:k]
            rows.append(dict(coverage=cov,
                             risk=float(1 - correct[sel].mean())))
        cum = np.cumsum(1 - correct[order]) / np.arange(1, len(y) + 1)
        aurc = float(np.trapezoid(cum, np.arange(1, len(y) + 1) / len(y)))
        return dict(rows=rows, aurc=aurc)
    out["risk_coverage"] = {k: rc(v) for k, v in
                            [("confidence", conf), ("entropy", ent),
                             ("routing_entropy", r_ent)]}
    return out


# ------------------------------------------------ D7 routing stability
def d7_routing_stability(model, ds, device, hyp, rng_seed=7600, n_max=200):
    rng = np.random.default_rng(rng_seed)
    X = ds["Xte"][:n_max]
    out = {"level": "router", "specs": {}}
    for name, spec in BENIGN_SPECS.items():
        Xp = np.stack([PB.apply_spec(x, spec["kind"], spec["level"], rng)[0]
                       for x in X])
        with torch.no_grad():
            wa = model(torch.from_numpy(X).float().to(device), return_aux=True)[1]["w"]
            wb = model(torch.from_numpy(Xp).float().to(device), return_aux=True)[1]["w"]
            pa = F.softmax(model(torch.from_numpy(X).float().to(device)), 1).cpu().numpy()
            pb = F.softmax(model(torch.from_numpy(Xp).float().to(device)), 1).cpu().numpy()
        cos = F.cosine_similarity(wa.flatten(1), wb.flatten(1)).mean().item()
        top1_agree = float((wa.argmax(1) == wb.argmax(1)).float().mean())
        kl = float((pa * np.log((pa + 1e-12) / (pb + 1e-12))).sum(1).mean())
        out["specs"][name] = dict(cosine_routing=cos, top1_route_agreement=top1_agree,
                                  prob_KL=kl)
        hyp.add("D7_routing_stability", "descriptive (no test), CI via bootstrap",
                f"{name}: routing cosine", cos, None, {})
    return out


# ------------------------------------------------ D8 counterfactual routing
def d8_counterfactual(model, ds, device, hyp, rng_seed=7700, n_max=200):
    """Modify a high-routing region: does routing adapt?"""
    rng = np.random.default_rng(rng_seed)
    X = ds["Xte"][:n_max]
    T = X.shape[-1]
    width = max(3, int(0.10 * T))
    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(X).float().to(device)
        R = model.bank(xt)
        A, Sv = model.activity(R)
        P = torch.cat([A, torch.log1p(Sv)], 1)
        Tv = model.flavors(xt, window=model.ppv_window)
        Tp = min(P.shape[-1], Tv.shape[-1])
        P, Tv = P[..., :Tp], Tv[..., :Tp]
        w0 = torch.softmax(model.router(P.permute(0, 2, 1)),
                           -1).permute(0, 2, 1)
        imp = (w0.max(1).values * Tv.abs().mean(1)).cpu().numpy()
    Xp = X.copy()
    for i in range(len(X)):
        c = int(np.argmax(imp[i]))
        lo, hi = max(0, c - width // 2), min(T, c + width // 2)
        if hi > lo:
            Xp[i, 0, lo:hi] = rng.normal(0, 1, hi - lo)    # replace with noise
    with torch.no_grad():
        xp = torch.from_numpy(Xp).float().to(device)
        Rp = model.bank(xp)
        Ap, Sp = model.activity(Rp)
        Pp = torch.cat([Ap, torch.log1p(Sp)], 1)
        Tvp = model.flavors(xp, window=model.ppv_window)
        Tpp = min(Pp.shape[-1], Tvp.shape[-1])
        Pp, Tvp = Pp[..., :Tpp], Tvp[..., :Tpp]
        w1 = torch.softmax(model.router(Pp.permute(0, 2, 1)),
                           -1).permute(0, 2, 1)
        p0 = F.softmax(model(xt), 1).cpu().numpy()
        p1 = F.softmax(model(xp), 1).cpu().numpy()
    shift = (w1 - w0[..., :w1.shape[2]]).abs().mean().item()  # temporal-length guard
    top1_change = float((w0.argmax(1) != w1.argmax(1)).float().mean())
    flip = float((p1.argmax(1) != p0.argmax(1)).mean())
    dconf = float((p1.max(1) - p0.max(1)).mean())
    out = dict(level="router", routing_weight_shift=shift,
               top1_route_change_rate=top1_change,
               pred_flip_rate=flip, mean_conf_change=dconf,
               note="counterfactual: noise replaced in top-routing region")
    hyp.add("D8_counterfactual_routing", "descriptive effect",
            "routing shift under targeted pattern modification", shift, None, {})
    return out


# ------------------------------------------------ D9 simple baselines
def d9_baselines(model, ds, device, hyp, A4_probs, ext):
    y = ds["y_test"]
    yerr = (A4_probs.argmax(1) != y).astype(int)
    W = ext["routing_weights"]
    r_ent = -(W * np.log(W + 1e-12)).sum(-1).mean(1)
    X = ext["X"][:len(yerr)]
    d1 = np.abs(np.diff(X[:, 0, :], axis=1, prepend=X[:, 0, :1])).mean(1)
    le = PB.raw_local_energy(X[:, 0, :]).mean(1)
    conf = A4_probs.max(1)
    ent = -(A4_probs * np.log(A4_probs + 1e-12)).sum(1)
    cands = {"routing_entropy": r_ent, "raw_first_difference": d1,
             "raw_local_energy": le, "confidence_neg": -conf,
             "predictive_entropy": ent, "random": rng_random(len(yerr))}
    out = {"level": "router+final", "aurocs": {}}
    for name, sc in cands.items():
        auc, lo, hi = S.bootstrap_auroc_ci(yerr, sc)
        out["aurocs"][name] = dict(auroc=auc, ci=[lo, hi])
    best_routing = max(out["aurocs"]["routing_entropy"]["auroc"], 0.5)
    best_simple = max(out["aurocs"][k]["auroc"] for k in
                      ["raw_first_difference", "raw_local_energy", "random"])
    hyp.add("D9_baselines", "bootstrap AUROC",
            "routing_entropy vs best raw baseline", best_routing - best_simple,
            None, {"routing": best_routing, "simple": best_simple})
    return out


def rng_random(n):
    return np.random.default_rng(7800).random(n)


def run_all_diagnostics(model, ds, device, A4_probs, ext, log=print):
    hyp = S.HypothesisRegistry()
    log("  [diag] D1 pattern-activity validity")
    r1 = d1_pattern_activity(model, ds, device, hyp, ext)
    log("  [diag] D2 routing validity")
    r2 = d2_routing_validity(model, ds, device, hyp, ext)
    log("  [diag] D3 routing intervention")
    from experiments.turs_rrmt.train_eval import routing_intervention
    uni_p = _uniform_routing(model, ds, device)
    fixed0 = routing_intervention(model, ds, device, "fixed_0")
    shuf = routing_intervention(model, ds, device, "shuffled")
    r3 = d3_routing_intervention(model, ds, device, hyp, A4_probs, uni_p,
                                 fixed0, shuf)
    log("  [diag] D4 routing faithfulness")
    r4 = d4_routing_faithfulness(model, ds, device, hyp)
    log("  [diag] D5 degradation")
    r5 = d5_degradation(model, ds, device, hyp)
    log("  [diag] D6 uncertainty")
    r6 = d6_uncertainty(model, ds, device, hyp, A4_probs, ext)
    log("  [diag] D7 stability")
    r7 = d7_routing_stability(model, ds, device, hyp)
    log("  [diag] D8 counterfactual")
    r8 = d8_counterfactual(model, ds, device, hyp)
    log("  [diag] D9 baselines")
    r9 = d9_baselines(model, ds, device, hyp, A4_probs, ext)
    rows = hyp.finalize()
    return dict(D1=r1, D2=r2, D3=r3, D4=r4, D5=r5, D6=r6, D7=r7, D8=r8, D9=r9,
                _hyp_rows=rows)


def _uniform_routing(model, ds, device, batch=256):
    """Frozen A4 evaluated with uniform routing weights (D3 control)."""
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(ds["Xte"]), batch):
            xb = torch.from_numpy(ds["Xte"][i:i + batch]).float().to(device)
            B, _, T = xb.shape
            R = model.bank(xb)
            A, Sv = model.activity(R)
            P = torch.cat([A, torch.log1p(Sv)], 1)
            Tv = model.flavors(xb, window=model.ppv_window)
            Tp = min(P.shape[-1], Tv.shape[-1])
            P, Tv = P[..., :Tp], Tv[..., :Tp]
            w = torch.full((B, model.J, Tp), 1.0 / model.J, device=xb.device)
            top2 = w.topk(min(model.topk, model.J), dim=1).indices
            mask = torch.zeros_like(w).scatter_(1, top2, 1.0)
            kept = w * mask * Tv
            rest = w.sum(1, keepdim=True) - (w * mask).sum(1, keepdim=True)
            F_T = model.transport_proj(torch.cat([kept, rest], dim=1))
            F_P = model.pattern_path(P[..., :Tp].permute(0, 2, 1)).permute(0, 2, 1)
            F_all = torch.cat([F_T, F_P], dim=1)   # NB: not 'F' (shadows functional)
            mu, mx = F_all.mean(-1), F_all.max(-1).values
            sd = F_all.std(-1) if F_all.shape[-1] > 1 else torch.zeros_like(mu)
            logits = model.head(torch.cat([mu, mx, sd], dim=1))
            outs.append(F.softmax(logits, 1).cpu().numpy())
    return np.concatenate(outs)
