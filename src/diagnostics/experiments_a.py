"""Experiments 1-5 (Part A): latent validity, velocity localization,
uncertainty validity, controlled degradation, alpha validation."""
import os

import numpy as np

from . import config as C
from . import perturb as P
from . import statistics as S
from .statistics import HypothesisRegistry


def _pooled_z_features(ext_split, mode="mean"):
    """Justified pooled latent representation per sample (Phase 7):
    native z [N,16] concatenated with pooled temporal-trace statistics
    (mean / std / max-magnitude / range of z_t), each [N,16] -> [N,80].
    Falls back to native z alone when temporal traces are unavailable."""
    z = ext_split["z"]
    parts = [z]
    if "z_t" in ext_split:
        zt = ext_split["z_t"]                     # [N, T, 16]
        parts += [zt.mean(1), zt.std(1), np.abs(zt).max(1), zt.max(1) - zt.min(1)]
    return np.concatenate(parts, axis=1)


# --------------------------------------------------------- Experiment 1
def exp_latent_validity(ext, ds, device, hyp: HypothesisRegistry):
    """Does pooled z contain class/regime structure? (Phase 7)"""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                                 calinski_harabasz_score,
                                 adjusted_rand_score, normalized_mutual_info_score,
                                 f1_score, balanced_accuracy_score, accuracy_score,
                                 confusion_matrix)
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    out = {}
    for split in ["val", "test"]:
        y = ext[split]["y"].astype(int)
        feats = _pooled_z_features(ext[split])
        n_cls = ds["n_cls"]
        if len(np.unique(y)) < 2:
            continue
        s = dict(
            silhouette=float(silhouette_score(feats, y)),
            davies_bouldin=float(davies_bouldin_score(feats, y)),
            calinski_harabasz=float(calinski_harabasz_score(feats, y)),
        )
        km = KMeans(n_clusters=n_cls, n_init=10, random_state=C.SEED).fit(feats)
        s["kmeans_ari"] = float(adjusted_rand_score(y, km.labels_))
        s["kmeans_nmi"] = float(normalized_mutual_info_score(y, km.labels_))

        # within/between distances (euclidean, subsampled for speed)
        rng = np.random.default_rng(C.SEED)
        idx = rng.permutation(len(feats))[:2000]
        f, yy = feats[idx], y[idx]
        D = np.linalg.norm(f[:, None] - f[None], axis=2)
        same = yy[:, None] == yy[None]
        s["within_class_dist"] = float(D[same].mean())
        if (~same).any():
            s["between_class_dist"] = float(D[~same].mean())
            s["between_within_ratio"] = s["between_class_dist"] / max(s["within_class_dist"], 1e-9)

        if split == "test":
            # supervised linear probe trained on TRAIN only
            ztr = _pooled_z_features(ext["train"])
            ytr = ext["train"]["y"].astype(int)
            zv = _pooled_z_features(ext["val"])
            yv = ext["val"]["y"].astype(int)
            probe = make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=2000, C=1.0))
            probe.fit(ztr, ytr)
            svm = make_pipeline(StandardScaler(), LinearSVC(C=1.0, max_iter=5000))
            svm.fit(ztr, ytr)
            pred = svm.predict(feats)
            s["probe_test_mf1_lr"] = float(f1_score(y, probe.predict(feats),
                                                    average="macro", zero_division=0))
            s["probe_test_mf1_svm"] = float(f1_score(y, pred, average="macro",
                                                     zero_division=0))
            s["probe_test_balacc_svm"] = float(balanced_accuracy_score(y, pred))
            s["probe_test_acc_svm"] = float(accuracy_score(y, pred))
            s["probe_confusion_matrix"] = confusion_matrix(
                y, pred, labels=list(range(n_cls))).tolist()

            # raw-input baseline features: simple statistical descriptors of x
            def raw_feats(Xn):
                X = Xn[:, 0, :]
                return np.stack([X.mean(1), X.std(1),
                                 np.abs(np.diff(X, axis=1)).mean(1),
                                 np.quantile(X, 0.05, axis=1),
                                 np.quantile(X, 0.95, axis=1),
                                 np.percentile(X, 75, axis=1) - np.percentile(X, 25, axis=1)],
                                axis=1)
            rf_tr = raw_feats(ds["Xtr"]); rf_te = raw_feats(ds["Xte"])
            probe_raw = make_pipeline(StandardScaler(),
                                      LogisticRegression(max_iter=2000, C=1.0))
            probe_raw.fit(rf_tr, ytr)
            s["probe_test_mf1_raw_stats"] = float(f1_score(
                y, probe_raw.predict(rf_te), average="macro", zero_division=0))

            # also mean-pooled raw signal as trivial pooled representation
            rf2_tr = ds["Xtr"][:, 0, :]; rf2_te = ds["Xte"][:, 0, :]
            svc_raw = make_pipeline(StandardScaler(), LinearSVC(C=0.01, max_iter=5000))
            svc_raw.fit(rf2_tr, ytr)
            s["probe_test_mf1_raw_linear"] = float(f1_score(
                y, svc_raw.predict(rf2_te), average="macro", zero_division=0))
        out[split] = s
    return out


# --------------------------------------------------------- Experiment 2
def exp_velocity_localization(tag, model, ds, device, hyp: HypothesisRegistry,
                              max_samples=120):
    """Controlled synthetic temporal changes: does s_t=||v_t|| localize them?
    (Phase 8) Uses the frozen temporal trace. Compares against raw
    first-difference, raw local energy, and random."""
    from .extraction import extract_split
    rng = np.random.default_rng(C.SYNTH_SEED)
    X = ds["Xte"][:, 0, :]
    y = ds["y_test"]
    L = X.shape[1]
    W = max(4, int(round(C.SYNTH_REGION_FRAC * L)))
    raw = ds["Xte"][:, 0, :].copy()

    # build synthetic set: per class, take N clean samples, inject a localized
    # amplitude bump at a random location
    xs, locs, kinds = [], [], []
    for c in range(ds["n_cls"]):
        cidx = np.where(y == c)[0]
        take = min(C.SYNTH_N_PER_CLASS, len(cidx))
        for i in rng.choice(cidx, size=take, replace=False):
            x0 = raw[i].copy()
            start = int(rng.integers(0, L - W + 1))
            x0[start:start + W] *= 2.0
            xs.append(x0); locs.append(start + W // 2); kinds.append("amp_bump")
    X_syn = np.stack(xs)[:, None, :]
    center = np.array(locs)

    # TURS temporal trace on synthetic inputs
    ext_syn = extract_split(model, X_syn, np.zeros(len(X_syn), int), device,
                            temporal=True)
    s_t = ext_syn["s_t"]                       # [N, T]
    T = s_t.shape[1]
    ratio = L / T

    def center_to_T(c):
        return int(np.clip(round(c / ratio), 0, T - 1))

    true_T = np.array([center_to_T(c) for c in center])
    pred_T = s_t.argmax(1)

    def loc_metrics(pred, truth):
        err1 = np.abs(pred - truth)
        d5 = np.sort(np.abs(pred[:, None] - np.arange(T)[None]), axis=1)  # not used
        hit5 = np.array([truth[i] in np.argsort(-s_t[i])[:5] for i in range(len(truth))])
        hit1 = err1 == 0
        return dict(
            top1_err=int(err1.mean() * ratio),
            top5_err=float(np.mean([np.abs((np.argsort(-s_t[i])[:5] - truth[i])).min()
                                     for i in range(len(truth))]) * ratio),
            hit_at_1=float(hit1.mean()),
            hit_at_5=float(hit5.mean()),
            precision_5=float(hit5.mean()),   # precision@5 == hit@5 for 1 truth
            recall_5=float(hit5.mean()),
        )

    res = {"n_synthetic": int(len(center)), "region_width": int(W),
           "turs": loc_metrics(pred_T, true_T)}

    # trivial baselines on raw synthetic signals
    fd = np.abs(np.diff(X_syn[:, 0, :], axis=1, prepend=0)) / C.VEL_TO_RAW_RATIO
    en = P.raw_local_energy(X_syn[:, 0, :]) / C.VEL_TO_RAW_RATIO
    rand = rng.random((len(center), T))
    # align raw to T by pooling (truncate or pad to a multiple of the ratio)
    def pool_t(a):
        ratio = a.shape[1] / T
        idx = (np.arange(T) * ratio).astype(int)
        idx = np.clip(idx, 0, a.shape[1] - 1)
        return a[:, idx]
    for name, score in [("raw_first_diff", pool_t(fd)),
                        ("raw_local_energy", pool_t(en)),
                        ("random", rand)]:
        res[name] = dict(
            top1_err=int(np.abs(score.argmax(1) - true_T).mean() * ratio),
            hit_at_5=float(np.mean([true_T[i] in np.argsort(-score[i])[:5]
                                    for i in range(len(true_T))])),
        )

    # statistical comparison: TURS vs each baseline on hit@5 (paired McNemar)
    from .statistics import mcnemar, bootstrap_diff_ci
    turs_hit = np.array([true_T[i] in np.argsort(-s_t[i])[:5] for i in range(len(true_T))])
    for name, score in [("raw_first_diff", pool_t(fd)),
                        ("raw_local_energy", pool_t(en)),
                        ("random", rand)]:
        b_hit = np.array([true_T[i] in np.argsort(-score[i])[:5]
                          for i in range(len(true_T))])
        chi2, p = mcnemar(turs_hit, b_hit)
        d, lo, hi = bootstrap_diff_ci(turs_hit.astype(float), b_hit.astype(float))
        hyp.add("velocity_localization", "McNemar paired",
                f"hit@5 TURS s_t vs {name}", d, p,
                {"turs_hit": float(turs_hit.mean()), "baseline_hit": float(b_hit.mean()),
                 "diff_ci": [lo, hi]})
    res["true_vs_pred_centers"] = dict(turs_center_err_mean=float(
        np.abs(pred_T - true_T).mean()))
    return res


# --------------------------------------------------------- Experiment 3
def exp_uncertainty_validity(ext, ds, hyp: HypothesisRegistry):
    """Intrinsic u vs correctness, error detection, baselines. (Phase 9)"""
    out = {}
    primary = C.PRIMARY["uncertainty_agg"]
    for split in ["val", "test"]:
        e = ext[split]
        u = e[f"u_{primary}"]
        corr, incorr = u[e["correct"] == 1], u[e["correct"] == 0]
        s = {"n_correct": int(len(corr)), "n_incorrect": int(len(incorr))}
        if len(incorr) >= 3 and len(corr) >= 3:
            m, lo, hi = S.bootstrap_diff_ci(incorr, corr)
            s["u_correct_mean"] = float(corr.mean())
            s["u_incorrect_mean"] = float(incorr.mean())
            s["u_correct_median"] = float(np.median(corr))
            s["u_incorrect_median"] = float(np.median(incorr))
            s["mean_diff"] = m; s["mean_diff_ci"] = [lo, hi]
            s["cliffs_delta"] = S.cliffs_delta(incorr, corr)
            s["cliffs_label"] = S.cliffs_label(s["cliffs_delta"])
            # independent samples -> Mann-Whitney U (one-sided: incorrect > correct)
            from scipy.stats import mannwhitneyu
            stat, p = mannwhitneyu(incorr, corr, alternative="greater")
            s["mannwhitney_p"] = float(p)
            hyp.add("uncertainty_validity", "Mann-Whitney U (greater)",
                    f"{split}: u for incorrect > correct", s["cliffs_delta"], p,
                    {"cliffs": s["cliffs_delta"], "n_inc": len(incorr)})

        # error detection: score = higher means "predict error"
        yerr = (e["correct"] == 0).astype(int)
        if 0 < yerr.sum() < len(yerr):
            cands = {
                "turs_u_mean": e["u_mean"],
                "turs_u_max": e["u_max"],
                "1_minus_maxsoftmax": 1 - e["confidence"],
                "predictive_entropy": e["entropy"],
                "margin_neg": -e["margin"],
            }
            best = {"turs_u_mean": e["u_mean"], "turs_u_max": e["u_max"],
                    "1_minus_maxsoftmax": 1 - e["confidence"],
                    "predictive_entropy": e["entropy"],
                    "top1_top2_margin": -e["margin"]}
            rows = {}
            for name, score in best.items():
                auc, lo, hi = S.bootstrap_auroc_ci(yerr, score)
                from sklearn.metrics import average_precision_score
                ap = float(average_precision_score(yerr, score))
                rows[name] = dict(auroc=auc, auroc_ci=[lo, hi], auprc=ap)
            s["error_detection"] = rows
            if split == "test":
                for name in best:
                    if name.startswith("turs"):
                        continue
                    aucT = rows["turs_u_mean"]["auroc"]
                    aucB = rows[name]["auroc"]
                    hyp.add("uncertainty_vs_baselines", "bootstrap diff AUROC",
                            f"test AUROC turs_u_mean vs {name}", aucT - aucB, None,
                            {"auroc_turs": aucT, "auroc_baseline": aucB})
        out[split] = s
    return out


# --------------------------------------------------------- Experiment 4
def exp_controlled_degradation(tag, model, ds, device, hyp: HypothesisRegistry,
                               max_samples=400):
    """Clean vs degraded pairs; u responds to corruption. (Phase 10)"""
    from .extraction import extract_split
    rng = np.random.default_rng(C.SYNTH_SEED + 1)
    n = min(max_samples, len(ds["Xte"]))
    idx = rng.permutation(len(ds["Xte"]))[:n]
    X_clean = ds["Xte"][idx][:, 0, :]
    y = np.asarray(ds["y_test"])[idx]

    results = {"n": int(n), "kinds": {}}
    for kind, spec in C.DEGRADATION_SPECS.items():
        kind_res = {"levels": {}}
        sev_scores, err_scores, u_scores = [], [], []
        for lv in spec["levels"]:
            Xd = np.stack([P.apply_spec(x, kind, lv, rng)[0] for x in X_clean])
            ext = extract_split(model, Xd[:, None, :], y, device, temporal=False)
            flip = float((ext["pred"] != _clean_pred(tag, model, ds, device, idx)).mean()) \
                if lv == spec["levels"][0] else None
            rec = dict(
                macro_f1=float(mf1(y, ext["pred"], ds["n_cls"])),
                accuracy=float((ext["pred"] == y).mean()),
                mean_confidence=float(ext["confidence"].mean()),
                mean_entropy=float(ext["entropy"].mean()),
                mean_u=float(ext["u_mean"].mean()),
                error_rate=float((ext["correct"] == 0).mean()),
                pred_flip_rate=None,
            )
            kind_res["levels"][str(lv)] = rec
            sev_scores.append(lv if kind != "amplitude_scale" else abs(1 - lv))
            err_scores.append(rec["error_rate"])
            u_scores.append(rec["mean_u"])
        # severity vs u and vs error (per-sample paired, over levels)
        rho_u = S.corr_with_inference(sev_scores, u_scores, "spearman")
        rho_e = S.corr_with_inference(sev_scores, err_scores, "spearman")
        kind_res["severity_vs_u_spearman"] = rho_u
        kind_res["severity_vs_error_spearman"] = rho_e
        hyp.add("degradation_response", "Spearman + permutation",
                f"{tag} {kind}: severity -> mean u", rho_u["rho"], rho_u["p_perm"],
                {"ci": [rho_u["ci_lo"], rho_u["ci_hi"]]})
        hyp.add("degradation_response", "Spearman + permutation",
                f"{tag} {kind}: severity -> error rate", rho_e["rho"],
                rho_e["p_perm"], {"ci": [rho_e["ci_lo"], rho_e["ci_hi"]]})
        results["kinds"][kind] = kind_res

    # H3: uncertainty correlates with actual unreliability (per-sample,
    # pooled across degradations): per-sample error indicator vs per-sample u
    per_sample = []
    for kind, spec in C.DEGRADATION_SPECS.items():
        for lv in spec["levels"]:
            Xd = np.stack([P.apply_spec(x, kind, lv, rng)[0] for x in X_clean])
            ext = extract_split(model, Xd[:, None, :], y, device, temporal=False)
            per_sample.append((ext["u_mean"], (ext["correct"] == 0).astype(float)))
    uu = np.concatenate([p[0] for p in per_sample])
    ee = np.concatenate([p[1] for p in per_sample])
    h3 = S.corr_with_inference(uu, ee, "spearman")
    auc, lo, hi = S.bootstrap_auroc_ci(ee.astype(int), uu)
    results["H3_u_vs_error"] = dict(**h3, auroc=auc, auroc_ci=[lo, hi])
    hyp.add("degradation_response", "Spearman + permutation",
            f"{tag} pooled degradations: u -> error", h3["rho"], h3["p_perm"],
            {"auroc": auc, "ci": [lo, hi]})
    return results


_CLEAN_PRED_CACHE = {}


def _clean_pred(tag, model, ds, device, idx):
    key = tag
    if key not in _CLEAN_PRED_CACHE:
        from .extraction import extract_split
        ext = extract_split(model, ds["Xte"], ds["y_test"], device, temporal=False)
        _CLEAN_PRED_CACHE[key] = ext["pred"]
    return _CLEAN_PRED_CACHE[key][idx]


def mf1(y, pred, n_cls):
    from sklearn.metrics import f1_score
    return f1_score(y, pred, average="macro", zero_division=0,
                    labels=list(range(n_cls)))


# --------------------------------------------------------- Experiment 5
def exp_alpha_validation(ext, ds, hyp: HypothesisRegistry):
    """alpha (transport reliance) vs independent input descriptors. (Phase 11)"""
    out = {}
    for split in ["val", "test"]:
        e = ext[split]
        X = ds["X" + {"train": "tr", "val": "va", "test": "te"}[split]][:, 0, :]
        alpha = e["alpha"][:, 0]
        # independent input descriptors (NOT derived from the model)
        desc = {
            "amplitude_std": X.std(1),
            "local_variance_mean": np.diff(X, axis=1).var(1),
            "ks_stat_uniform": _ks_uniform(X),
            "hist_entropy": _hist_entropy(X),
            "lag_disagreement": _lag_disagreement(X),
        }
        res = {}
        for name, v in desc.items():
            r = S.corr_with_inference(alpha, v, "spearman")
            res[name] = r
            if split == "test":
                hyp.add("alpha_validity", "Spearman + permutation",
                        f"{split}: alpha vs {name}", r["rho"], r["p_perm"],
                        {"ci": [r["ci_lo"], r["ci_hi"]]})
        res["alpha_mean"] = float(alpha.mean())
        res["alpha_std"] = float(alpha.std())
        res["alpha_deciles"] = np.quantile(alpha, np.linspace(0.1, 0.9, 9)).tolist()
        out[split] = res
    return out


def _ks_uniform(X):
    from scipy.stats import kstest
    z = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-8)
    from scipy.stats import norm
    cdf = norm.cdf(z)
    return np.array([kstest(row, "uniform")[0] for row in cdf])


def _hist_entropy(X, bins=32):
    ent = np.empty(len(X))
    for i, x in enumerate(X):
        h, _ = np.histogram(x, bins=bins)
        p = h / max(h.sum(), 1)
        p = p[p > 0]
        ent[i] = -(p * np.log(p)).sum()
    return ent


def _lag_disagreement(X, lag=8):
    a = X[:, :-lag]; b = X[:, lag:]
    return np.abs(a - b).mean(1)
