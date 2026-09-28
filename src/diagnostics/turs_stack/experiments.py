"""Stack diagnostic experiments (Phases 5-16, 23, 24).

Level discipline (Phase 4): every result dict is tagged with its level:
  level="branch" (intrinsic mechanism), "ensemble" (derived from branch
  outputs), or "final" (final predictor property).
"""
import json
import os

import numpy as np

from . import config as C
from src.diagnostics import perturb as P
from src.diagnostics import statistics as S
from src.diagnostics import calibration as CAL
from .statistics import HypothesisRegistry

BRANCHES = ["lite", "rv", "cs", "cmr"]


def _cls_feats(raw_key, ext_split):
    """Pooled representation per sample: native sample-level state + pooled
    temporal stats (mean/std/max/range) of z_t -> [N, 16*5] when available."""
    z = ext_split[raw_key]
    parts = [z.mean(1)]
    parts += [z.std(1), np.abs(z).max(1), z.max(1) - z.min(1)]
    return np.concatenate(parts, axis=1)


# ------------------------------------------------ Phase 5: latent validity
def exp_latent(ext, ds, hyp):
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                                 calinski_harabasz_score, adjusted_rand_score,
                                 normalized_mutual_info_score, f1_score,
                                 balanced_accuracy_score, confusion_matrix)
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    out = {"level": "branch"}
    n_cls = ds["n_cls"]
    zkeys = {"lite": "z_t_lite", "rv": "z_t_rv", "cs": "z_t_cs", "cmr": "z_t_cmr"}
    raw_feats = None
    for b, zk in zkeys.items():
        res = {}
        try:
            tr = _cls_feats(zk, ext["train"]); te = _cls_feats(zk, ext["test"])
        except KeyError:
            out[b] = {"error": "missing key"}
            continue
        ytr = ext["train"]["y"]; yte = ext["test"]["y"]
        res["silhouette"] = float(silhouette_score(te, yte))
        res["davies_bouldin"] = float(davies_bouldin_score(te, yte))
        res["calinski_harabasz"] = float(calinski_harabasz_score(te, yte))
        km = KMeans(n_clusters=n_cls, n_init=10, random_state=C.SEED).fit(te)
        res["kmeans_ari"] = float(adjusted_rand_score(yte, km.labels_))
        res["kmeans_nmi"] = float(normalized_mutual_info_score(yte, km.labels_))
        probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        probe.fit(tr, ytr)
        pred = probe.predict(te)
        res["probe_test_mf1"] = float(f1_score(yte, pred, average="macro",
                                               zero_division=0, labels=list(range(n_cls))))
        res["probe_test_balacc"] = float(balanced_accuracy_score(yte, pred))
        res["probe_confusion_matrix"] = confusion_matrix(
            yte, pred, labels=list(range(n_cls))).tolist()
        if raw_feats is None:
            def rawf(X):
                X = X[:, 0, :]
                return np.stack([X.mean(1), X.std(1),
                                 np.abs(np.diff(X, axis=1)).mean(1),
                                 np.quantile(X, 0.05, axis=1),
                                 np.quantile(X, 0.95, axis=1)], 1)
            rf_tr = rawf(ds["Xtr"]); rf_te = rawf(ds["Xte"])
            pr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            pr.fit(rf_tr, ytr)
            raw_feats = float(f1_score(yte, pr.predict(rf_te), average="macro",
                                       zero_division=0, labels=list(range(n_cls))))
        res["probe_test_mf1_raw_stats"] = raw_feats
        out[b] = res
    return out


# ------------------------------------------------ Phase 6: velocity
def exp_velocity(tag, model, ds, device, hyp, ext):
    """s_t = ||v_t|| localization on controlled synthetic changes (test)."""
    from .extraction import forward_split
    rng = np.random.default_rng(C.SYNTH_SEED)
    X = ds["Xte"][:, 0, :]
    y = ds["y_test"]
    L = X.shape[1]
    W = max(4, int(round(C.SYNTH_REGION_FRAC * L)))
    xs, centers = [], []
    for c in range(ds["n_cls"]):
        cidx = np.where(y == c)[0]
        take = min(C.SYNTH_N_PER_CLASS, len(cidx))
        for i in rng.choice(cidx, size=take, replace=False):
            x0 = X[i].copy()
            start = int(rng.integers(0, L - W + 1))
            x0[start:start + W] *= 2.0
            xs.append(x0); centers.append(start + W // 2)
    X_syn = np.stack(xs)[:, None, :]
    center = np.array(centers)

    raw = forward_split(model, X_syn, device, heavy_temporal=False)
    scal = {}
    for b, key in [("lite", "v_t_lite"), ("rv", "vtilde_rv"), ("cs", "v_t_cs"),
                   ("cmr", "v_t_cmr")]:
        if key in raw:
            scal[b] = np.linalg.norm(raw[key], axis=2)     # [N, T]
    T = scal["lite"].shape[1]
    ratio = L / T
    true_T = np.clip((center / ratio).astype(int), 0, T - 1)

    res = {"level": "branch", "n": int(len(center)), "region": int(W)}
    def hitk(score, k):
        return np.array([true_T[i] in np.argsort(-score[i])[:k]
                         for i in range(len(true_T))])
    turs_hit5 = {}
    for b, s in scal.items():
        pred1 = s.argmax(1)
        h5 = hitk(s, 5)
        turs_hit5[b] = h5
        res[b] = dict(
            top1_err_steps=float(np.abs(pred1 - true_T).mean()),
            hit_at_1=float((pred1 == true_T).mean()),
            hit_at_5=float(h5.mean()))
    # baselines
    fd = np.abs(np.diff(X_syn[:, 0, :], axis=1, prepend=0))
    en = P.raw_local_energy(X_syn[:, 0, :])
    rnd = rng.random((len(center), T))
    def pool(a):
        idx = (np.arange(T) * (a.shape[1] / T)).astype(int)
        return a[:, np.clip(idx, 0, a.shape[1] - 1)]
    baselines = {"raw_first_diff": pool(fd), "raw_local_energy": pool(en),
                 "random": rnd}
    res["baselines"] = {}
    for name, sc in baselines.items():
        res["baselines"][name] = dict(
            top1_err_steps=float(np.abs(sc.argmax(1) - true_T).mean()),
            hit_at_5=float(hitk(sc, 5).mean()))
        # paired test: best TURS branch vs baseline
        best_b = max(turs_hit5, key=lambda b: turs_hit5[b].mean())
        chi2, p = S.mcnemar(turs_hit5[best_b], hitk(sc, 5))
        d, lo, hi = S.bootstrap_diff_ci(turs_hit5[best_b].astype(float),
                                        hitk(sc, 5).astype(float))
        hyp.add("velocity_localization", "McNemar paired",
                f"{tag} hit@5 {best_b} vs {name}", d, p,
                {"turs": float(turs_hit5[best_b].mean()),
                 "baseline": float(hitk(sc, 5).mean()), "ci": [lo, hi]})
    return res


# ------------------------------------------------ Phase 7: intrinsic u
def exp_intrinsic_uncertainty(ext, hyp):
    out = {"level": "branch"}
    for split in ["val", "test"]:
        e = ext[split]
        res = {}
        for b in BRANCHES:
            uk = f"u_{b}_mean"
            if uk not in e:
                continue
            u = e[uk]
            corr, incorr = u[e["final_correct"] == 1], u[e["final_correct"] == 0]
            s = dict(n_correct=int(len(corr)), n_incorrect=int(len(incorr)))
            if len(incorr) >= 3 and len(corr) >= 3:
                m, lo, hi = S.bootstrap_diff_ci(incorr, corr)
                from scipy.stats import mannwhitneyu
                stat, p = mannwhitneyu(incorr, corr, alternative="greater")
                cd = S.cliffs_delta(incorr, corr)
                s.update(u_correct_mean=float(corr.mean()),
                         u_incorrect_mean=float(incorr.mean()),
                         mean_diff=m, mean_diff_ci=[lo, hi],
                         cliffs_delta=cd, cliffs_label=S.cliffs_label(cd),
                         mannwhitney_p=float(p))
                if split == "test":
                    hyp.add("intrinsic_uncertainty", "Mann-Whitney U (greater)",
                            f"{b}: u incorrect > correct", cd, p,
                            {"n_inc": len(incorr)})
            # error detection + baselines (branch probs)
            yerr = (e["final_correct"] == 0).astype(int)
            if 0 < yerr.sum() < len(yerr):
                cands = {f"u_{b}_mean": e[uk], f"u_{b}_max": e[f"u_{b}_max"],
                         "1_minus_maxsoftmax": 1 - e["final_conf"],
                         "final_entropy": e["final_entropy"],
                         "margin_neg": -e["final_margin"],
                         "js_disagreement": e["js_disagreement"]}
                rows = {}
                for name, score in cands.items():
                    auc, lo, hi = S.bootstrap_auroc_ci(yerr, score)
                    from sklearn.metrics import average_precision_score
                    rows[name] = dict(auroc=auc, ci=[lo, hi],
                                      auprc=float(average_precision_score(yerr, score)))
                s["error_detection"] = rows
            res[b] = s
        out[split] = res
    return out


# ------------------------------------------------ Phase 8: ensemble signals
ENSEMBLE_SIGNALS = ["branch_pred_disagreement", "vote_entropy", "prob_var",
                    "conf_dispersion", "js_disagreement",
                    "mean_branch_entropy", "n_agree_max_neg"]


def exp_ensemble(ext, hyp):
    out = {"level": "ensemble"}
    for split in ["val", "test"]:
        e = ext[split]
        yerr = (e["final_correct"] == 0).astype(int)
        res = {}
        cands = {name: (e["n_agree_max"] if name == "n_agree_max_neg" else e[name])
                 for name in ENSEMBLE_SIGNALS}
        cands["n_agree_max_neg"] = -e["n_agree_max"]
        if 0 < yerr.sum() < len(yerr):
            for name, score in cands.items():
                auc, lo, hi = S.bootstrap_auroc_ci(yerr, score)
                from sklearn.metrics import average_precision_score
                res[name] = dict(auroc=auc, ci=[lo, hi],
                                 auprc=float(average_precision_score(yerr, score)),
                                 n=int(len(yerr)), n_err=int(yerr.sum()))
                if split == "test":
                    # vs final confidence baselines
                    for bname, bscore in [("final_entropy", e["final_entropy"]),
                                          ("1_minus_conf", 1 - e["final_conf"])]:
                        aucB, _, _ = S.bootstrap_auroc_ci(yerr, bscore)
                        hyp.add("ensemble_vs_confidence", "bootstrap AUROC diff",
                                f"{name} vs {bname}", auc - aucB, None,
                                {"auroc_signal": auc, "auroc_baseline": aucB})
        # risk-coverage for primary signal
        out[f"risk_coverage_{split}"] = CAL.risk_coverage(
            e, score=e[C.PRIMARY["ensemble_signal"]])
        out[split] = res
    return out


# ------------------------------------------------ Phase 9: pairwise overlap
def exp_pairwise(ext, hyp):
    e = ext["test"]
    bp = e["branch_pred"]; bc = e["branch_correct"]
    K = len(BRANCHES)
    out = {"level": "ensemble", "pairs": {}}
    for a in range(K):
        for b in range(a + 1, K):
            na, nb = BRANCHES[a], BRANCHES[b]
            agree = float((bp[a] == bp[b]).mean())
            err_overlap = float(((bc[a] == 0) & (bc[b] == 0)).sum()
                                / max((bc[a] == 0).sum(), 1))
            helpful = float((((bp[a] != bp[b])) & (bc[a] != bc[b])).mean())
            out["pairs"][f"{na}-{nb}"] = dict(
                agreement=agree, error_overlap=err_overlap,
                disagreement_complement=helpful)
    # disagreement about final error: Spearman js vs final nll
    r = S.corr_with_inference(e["js_disagreement"], e["final_nll"], "spearman")
    out["js_vs_final_nll"] = r
    hyp.add("ensemble_disagreement_error", "Spearman + permutation",
            "js_disagreement vs final NLL", r["rho"], r["p_perm"],
            {"ci": [r["ci_lo"], r["ci_hi"]]})
    return out


# ------------------------------------------------ Phase 10: degradation
def exp_degradation(tag, model, ds, device, hyp, max_n=None):
    from .extraction import forward_split, compute_scalars
    rng = np.random.default_rng(C.SYNTH_SEED + 1)
    n = min(max_n or C.DEG_MAX_SAMPLES, len(ds["Xte"]))
    idx = rng.permutation(len(ds["Xte"]))[:n]
    X_clean = ds["Xte"][idx][:, 0, :]
    y = np.asarray(ds["y_test"])[idx]

    results = {"level": "all", "n": int(n), "kinds": {}}
    series = {}
    for kind, spec in C.DEGRADATION_SPECS.items():
        kind_res = {"levels": {}}
        for lv in spec["levels"]:
            Xd = np.stack([P.apply_spec(x, kind, lv, rng)[0] for x in X_clean])
            raw = forward_split(model, Xd[:, None, :], device, heavy_temporal=False)
            sc = compute_scalars(raw, y)
            rec = dict(
                macro_f1=_mf1(y, sc["final_pred"], ds["n_cls"]),
                accuracy=float(sc["final_correct"].mean()),
                final_confidence=float(sc["final_conf"].mean()),
                final_entropy=float(sc["final_entropy"].mean()),
                error_rate=float(1 - sc["final_correct"].mean()),
                branch_disagreement_js=float(sc["js_disagreement"].mean()),
                vote_entropy=float(sc["vote_entropy"].mean()),
                prob_var=float(sc["prob_var"].mean()),
                u_lite=float(sc.get("u_lite_mean", np.array([np.nan])).mean()),
                u_cs=float(sc.get("u_cs_mean", np.array([np.nan])).mean()),
                u_cmr=float(sc.get("u_cmr_mean", np.array([np.nan])).mean()),
            )
            kind_res["levels"][str(lv)] = rec
            for sig in ["u_lite", "u_cs", "u_cmr", "branch_disagreement_js",
                        "final_entropy", "final_confidence"]:
                series.setdefault(sig, ([], []))
                sev = abs(1 - lv) if kind == "amplitude_scale" else lv
                series[sig][0].append(sev)
                series[sig][1].append(rec.get(sig if sig != "final_confidence"
                                              else "final_confidence"))
        results["kinds"][kind] = kind_res

    # hypotheses per signal
    for sig, (sev, vals) in series.items():
        rho = S.corr_with_inference(sev, vals, "spearman")
        results[f"severity_vs_{sig}_spearman"] = rho
        hyp.add("degradation_response", "Spearman + permutation",
                f"{tag} severity -> {sig}", rho["rho"], rho["p_perm"],
                {"ci": [rho["ci_lo"], rho["ci_hi"]]})
    return results


def _mf1(y, pred, n_cls):
    from sklearn.metrics import f1_score
    return float(f1_score(y, pred, average="macro", zero_division=0,
                          labels=list(range(n_cls))))


# ------------------------------------------------ Phase 11: alpha
def exp_alpha(ext, ds, hyp):
    out = {"level": "branch"}
    for split in ["test"]:
        e = ext[split]
        X = ds["X" + {"train": "tr", "val": "va", "test": "te"}[split]][:, 0, :]
        desc = {
            "amplitude_std": X.std(1),
            "local_variance_mean": np.diff(X, axis=1).var(1),
            "ks_stat_uniform": _ks_uniform(X),
            "hist_entropy": _hist_entropy(X),
            "lag_disagreement": np.abs(X[:, :-8] - X[:, 8:]).mean(1),
        }
        res = {}
        for b in BRANCHES:
            ak = f"alpha_{b}"
            if ak not in e:
                continue
            alpha = e[ak]
            rr = {}
            for name, v in desc.items():
                r = S.corr_with_inference(alpha, v, "spearman")
                rr[name] = r
                hyp.add("alpha_validity", "Spearman + permutation",
                        f"{split}: alpha_{b} vs {name}", r["rho"], r["p_perm"],
                        {"ci": [r["ci_lo"], r["ci_hi"]]})
            rr["alpha_mean"] = float(alpha.mean())
            res[b] = rr
        out[split] = res
    return out


def _ks_uniform(X):
    from scipy.stats import kstest, norm
    z = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-8)
    cdf = norm.cdf(z)
    return np.array([kstest(row, "uniform")[0] for row in cdf])


def _hist_entropy(X, bins=32):
    ent = np.empty(len(X))
    for i, x in enumerate(X):
        h, _ = np.histogram(x, bins=bins)
        p = h / max(h.sum(), 1); p = p[p > 0]
        ent[i] = -(p * np.log(p)).sum()
    return ent


# ------------------------------------------------ Phase 12: novelty + beta
def exp_novelty_beta(ext, hyp):
    out = {"level": "branch"}
    e = ext["test"]
    # novelty vs confidence/error/entropy/disagreement
    nov = e["novelty"]
    yerr = (e["final_correct"] == 0).astype(int)
    auc, lo, hi = S.bootstrap_auroc_ci(yerr, nov)
    from sklearn.metrics import average_precision_score
    out["novelty"] = dict(auroc_vs_error=auc, ci=[lo, hi],
                          auprc=float(average_precision_score(yerr, nov)),
                          mean=float(nov.mean()), std=float(nov.std()))
    r = S.corr_with_inference(nov, e["final_entropy"], "spearman")
    out["novelty_vs_entropy"] = r
    hyp.add("novelty_validity", "Spearman + permutation",
            "novelty vs final entropy", r["rho"], r["p_perm"])
    hyp.add("novelty_validity", "AUROC bootstrap",
            "novelty detects final errors", auc, None, {"ci": [lo, hi]})
    # beta stats (cs / cmr): temporal mean/entropy + relation to input scale
    for b in ["cs", "cmr"]:
        beta = e[f"beta_{b}"]                  # [N, T, 3]
        ent = -(beta * np.log(beta + 1e-12)).sum(-1).mean(1)
        dom = beta.argmax(-1)
        dom_frac = np.stack([(dom == k).mean(1) for k in range(3)], 1)
        out[f"beta_{b}"] = dict(
            mean=beta.mean((0, 1)).tolist(), entropy_mean=float(ent.mean()),
            dominance_fraction_mean=dom_frac.mean(0).tolist(),
            entropy_std=float(ent.std()))
        hyp.add("beta_stats", "descriptive", f"beta_{b} entropy", float(ent.mean()),
                None, {"std": float(ent.std())})
    return out


# ------------------------------------------------ Phase 13/14: faithfulness
def _predict_stack(model, X_B1L, device):
    import torch
    with torch.no_grad():
        out = model(torch.from_numpy(X_B1L).float().to(device))
    return torch.stack(out["probs"], 0).mean(0).float().cpu().numpy()


def exp_faithfulness(tag, model, ds, device, ext_val, hyp):
    """Targeted (u_t / s_t peaks) vs matched-random regions. Uses cs branch
    u_t (primary pre-registered) + s_t_cs. Region frac from VAL if desired;
    primary fixed at config. Level: branch-intrinsic + final response."""
    from .extraction import forward_split
    rng = np.random.default_rng(C.SYNTH_SEED + 2)
    n = min(C.FAITH_MAX_SAMPLES, len(ds["Xte"]))
    idx = np.random.default_rng(C.SEED).permutation(len(ds["Xte"]))[:n]
    X = ds["Xte"][idx][:, 0, :]
    y = np.asarray(ds["y_test"])[idx]
    raw = forward_split(model, X[:, None, :], device, heavy_temporal=False)
    u_map = raw["u_t_cs"][:, :, 0]             # [N, T]
    s_map = np.linalg.norm(raw["v_t_cs"], axis=2)
    ratio = X.shape[1] / u_map.shape[1]
    frac = C.FAITH_PRIMARY_FRAC

    results = {"level": "branch", "n": int(n), "frac": frac, "by_source": {}}
    for source, score_map in [("u_t_cs", u_map), ("s_t_cs", s_map)]:
        diffs = {m: [] for m in ["pdrop", "confdrop", "entdrop", "flip"]}
        rnd = {m: [] for m in diffs}
        for i in range(n):
            x0 = X[i]
            sm = np.repeat(score_map[i], int(np.ceil(ratio)))[:len(x0)]
            w = max(1, int(round(frac * len(x0))))
            tstart = int(np.argmax(np.convolve(sm, np.ones(w), mode="valid")))
            base = _predict_stack(model, x0[None, None, :], device)[0]
            from src.diagnostics.experiments_b import _perturb_region, _logits
            xp = _perturb_region(x0, C.FAITH_PERTURB, tstart, tstart + w, rng)
            pp = _predict_stack(model, xp[None, None, :], device)[0]
            diffs["pdrop"].append(base[y[i]] - pp[y[i]])
            diffs["confdrop"].append(base.max() - pp.max())
            diffs["entdrop"].append(-(pp * np.log(pp + 1e-12)).sum()
                                    - (-(base * np.log(base + 1e-12)).sum()))
            diffs["flip"].append(float(pp.argmax() != y[i]))
            for _ in range(C.FAITH_N_RANDOM):
                rs = int(rng.integers(0, len(x0) - w))
                xr = _perturb_region(x0, C.FAITH_PERTURB, rs, rs + w, rng)
                pr = _predict_stack(model, xr[None, None, :], device)[0]
                rnd["pdrop"].append(base[y[i]] - pr[y[i]])
                rnd["confdrop"].append(base.max() - pr.max())
                rnd["entdrop"].append(-(pr * np.log(pr + 1e-12)).sum()
                                      - (-(base * np.log(base + 1e-12)).sum()))
                rnd["flip"].append(float(pr.argmax() != y[i]))
        rec = {}
        for m in diffs:
            t = np.array(diffs[m]); r_all = np.array(rnd[m])
            ravg = r_all.reshape(len(t), -1).mean(1)
            d, lo, hi = S.bootstrap_diff_ci(t, ravg, paired=True)
            stat, p = S.paired_permutation_test(t, ravg)
            g = S.cohens_d_paired(t, ravg)
            rec[m] = dict(targeted_mean=float(t.mean()), random_mean=float(ravg.mean()),
                          diff=d, ci=[lo, hi], p_perm=p, cohens_d=g)
            if m == "pdrop":
                hyp.add("faithfulness", "paired permutation",
                        f"{tag} {source}: targeted>random p-drop", d, p,
                        {"ci": [lo, hi], "cohens_d": g})
        results["by_source"][source] = rec
    return results


def exp_counterfactual(tag, model, ds, device, hyp):
    from .extraction import forward_split
    from src.diagnostics.experiments_b import _perturb_region
    rng = np.random.default_rng(C.SYNTH_SEED + 4)
    n = min(C.FAITH_MAX_SAMPLES, len(ds["Xte"]))
    idx = np.random.default_rng(C.SEED).permutation(len(ds["Xte"]))[:n]
    X = ds["Xte"][idx][:, 0, :]
    y = np.asarray(ds["y_test"])[idx]
    raw = forward_split(model, X[:, None, :], device, heavy_temporal=False)
    s_map = np.linalg.norm(raw["v_t_cs"], axis=2)
    ratio = X.shape[1] / s_map.shape[1]
    frac = C.FAITH_PRIMARY_FRAC
    tp, rp, tf, rf = [], [], [], []
    for i in range(n):
        x0 = X[i]
        sm = np.repeat(s_map[i], int(np.ceil(ratio)))[:len(x0)]
        w = max(1, int(round(frac * len(x0))))
        tstart = int(np.argmax(np.convolve(sm, np.ones(w), mode="valid")))
        base = _predict_stack(model, x0[None, None, :], device)[0]
        xt = x0.copy(); xt[tstart:tstart + w] *= 2.5
        rs = int(rng.integers(0, len(x0) - w))
        xr = x0.copy(); xr[rs:rs + w] *= 2.5
        pt = _predict_stack(model, xt[None, None, :], device)[0]
        pr = _predict_stack(model, xr[None, None, :], device)[0]
        tp.append(base[y[i]] - pt[y[i]]); rp.append(base[y[i]] - pr[y[i]])
        tf.append(float(pt.argmax() != y[i])); rf.append(float(pr.argmax() != y[i]))
    tp, rp, tf, rf = map(np.array, (tp, rp, tf, rf))
    d, lo, hi = S.bootstrap_diff_ci(tp, rp, paired=True)
    stat, p = S.paired_permutation_test(tp, rp)
    g = S.cohens_d_paired(tp, rp)
    chi2, pm = S.mcnemar(tf, rf)
    hyp.add("counterfactual", "paired permutation",
            f"{tag}: high-s_t edit > random edit", d, p, {"ci": [lo, hi], "d": g})
    return {"level": "branch", "n": int(n), "frac": frac,
            "probdrop_target": float(tp.mean()), "probdrop_random": float(rp.mean()),
            "diff": d, "ci": [lo, hi], "p_perm": p, "cohens_d": g,
            "flip_target": float(tf.mean()), "flip_random": float(rf.mean()),
            "mcnemar_p": pm}


# ------------------------------------------------ Phase 15: benign
def exp_benign(tag, model, ds, device, hyp):
    from .extraction import forward_split, compute_scalars
    rng = np.random.default_rng(C.SYNTH_SEED + 3)
    n = min(C.FAITH_MAX_SAMPLES, len(ds["Xte"]))
    idx = np.random.default_rng(C.SEED).permutation(len(ds["Xte"]))[:n]
    X = ds["Xte"][idx][:, 0, :]
    y = np.asarray(ds["y_test"])[idx]
    raw0 = forward_split(model, X[:, None, :], device, heavy_temporal=False)
    sc0 = compute_scalars(raw0, y)
    results = {"level": "all"}
    for name, spec in C.BENIGN_SPECS.items():
        Xt = np.stack([P.apply_spec(x, spec["kind"], spec["level"], rng)[0] for x in X])
        raw1 = forward_split(model, Xt[:, None, :], device, heavy_temporal=False)
        sc1 = compute_scalars(raw1, y)

        def cos(a, b):
            num = (a * b).sum(1)
            den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
            return float(np.nanmean(num / den))
        results[name] = dict(
            final_agreement=float((sc0["final_pred"] == sc1["final_pred"]).mean()),
            final_kl=float((sc0["final_probs"] * (np.log(sc0["final_probs"] + 1e-12)
                                                  - np.log(sc1["final_probs"] + 1e-12))).sum(1).mean()),
            z_cs_cosine=cos(raw0["z_t_cs"].mean(1), raw1["z_t_cs"].mean(1)),
            u_cs_change=float(np.abs(sc0["u_cs_mean"] - sc1["u_cs_mean"]).mean()),
            alpha_cs_change=float(np.abs(sc0["alpha_cs"] - sc1["alpha_cs"]).mean()),
            js_disagreement_change=float(np.abs(sc0["js_disagreement"]
                                                - sc1["js_disagreement"]).mean()),
            vote_consistency=float((sc0["branch_pred"] == sc1["branch_pred"]).mean()),
        )
        hyp.add("benign_stability", "descriptive + CI",
                f"{tag} {name}: final agreement", results[name]["final_agreement"],
                None, {})
    return results


# ------------------------------------------------ Phase 23: risk-coverage
def exp_risk_coverage(ext):
    e = ext["test"]
    out = {"level": "all"}
    for name, score in [("final_1_conf", 1 - e["final_conf"]),
                        ("final_entropy", e["final_entropy"]),
                        ("branch_disagreement_js", e["js_disagreement"]),
                        ("u_cs_mean", e.get("u_cs_mean", e["final_entropy"]))]:
        out[name] = CAL.risk_coverage(e, score=score)
    return out


# ------------------------------------------------ Phase 24: error typology
def exp_error_typology(ext):
    e = ext["test"]
    wrong = e["final_correct"] == 0
    if wrong.sum() == 0:
        return {"level": "final", "n_errors": 0}
    u_med = np.median(e["u_cs_mean"])
    dis_med = np.median(e["js_disagreement"])
    conf_med = np.median(e["final_conf"])
    nov_med = np.median(e["novelty"])
    cats = {
        "high_agreement_wrong": wrong & (e["n_agree_max"] >= 3),
        "high_disagreement_wrong": wrong & (e["js_disagreement"] > dis_med),
        "low_u_wrong": wrong & (e["u_cs_mean"] <= u_med),
        "high_u_wrong": wrong & (e["u_cs_mean"] > u_med),
        "high_conf_wrong": wrong & (e["final_conf"] > conf_med),
        "high_novelty_wrong": wrong & (e["novelty"] > nov_med),
    }
    out = {"level": "final", "n_errors": int(wrong.sum()),
           "n_test": int(len(wrong)), "categories": {}}
    for name, m in cats.items():
        out["categories"][name] = dict(count=int(m.sum()),
                                       frac_of_errors=float(m.sum() / max(wrong.sum(), 1)))
    return out


# ------------------------------------------------ Phase 16: baselines summary
def exp_baseline_summary(ext, hyp, tag):
    e = ext["test"]
    yerr = (e["final_correct"] == 0).astype(int)
    cands = {"branch_u_cs_mean": e.get("u_cs_mean", e["final_entropy"] * 0),
             "js_branch_disagreement": e["js_disagreement"],
             "vote_entropy": e["vote_entropy"],
             "1_minus_max_softmax": 1 - e["final_conf"],
             "final_entropy": e["final_entropy"],
             "top1_top2_margin_neg": -e["final_margin"]}
    rows = {}
    for name, score in cands.items():
        auc, lo, hi = S.bootstrap_auroc_ci(yerr, score)
        rows[name] = dict(auroc=auc, ci=[lo, hi], n=int(len(yerr)),
                          n_errors=int(yerr.sum()))
    names = [n for n in cands if n.startswith(("branch", "js", "vote"))]
    simple = [n for n in cands if n.startswith(("1_minus", "final", "top1"))]
    from sklearn.metrics import roc_auc_score
    r = np.random.default_rng(C.BOOTSTRAP_SEED)
    for a in names:
        for b in simple:
            diffs = []
            for _ in range(500):
                ii = r.integers(0, len(yerr), len(yerr))
                try:
                    diffs.append(roc_auc_score(yerr[ii], cands[a][ii]) -
                                 roc_auc_score(yerr[ii], cands[b][ii]))
                except ValueError:
                    continue
            d = float(np.mean(diffs)); lo, hi = np.percentile(diffs, [2.5, 97.5])
            rp = np.random.default_rng(C.PERMUTATION_SEED)
            cnt = 0
            for _ in range(1000):
                yy = rp.permutation(yerr)
                try:
                    dd = roc_auc_score(yy, cands[a]) - roc_auc_score(yy, cands[b])
                    if abs(dd) >= abs(d):
                        cnt += 1
                except ValueError:
                    continue
            hyp.add("baseline_uncertainty", "bootstrap+permutation AUROC diff",
                    f"{tag}: {a} vs {b}", d, (cnt + 1) / 1001, {"ci": [lo, hi]})
    return rows
