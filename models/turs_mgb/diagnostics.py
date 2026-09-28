"""TURS-MGB diagnostics (Phases 13-26).

Information-addition, complementarity, redundancy, class-conditional value,
group contributions, zeroing ablation, perturbation response, faithfulness,
stability, calibration, selective prediction, significance, FDR.

All statistics flow through src.diagnostics.statistics (deterministic seeds)
and src.diagnostics.calibration; perturbations through
src.diagnostics.perturb.apply_spec with the shared spec table.
"""

import numpy as np
import torch

from src.diagnostics import statistics as S
from src.diagnostics import perturb as PB
from src.diagnostics.calibration import ece, adaptive_ece, brier, nll as nll_fn
from experiments.turs_rrmt.train_eval import _mf1

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
MAX_N = 600


def _mcnemar_perm(y, pred_a, pred_b, hyp, family, label):
    cA, cB = pred_a == y, pred_b == y
    chi2, p = S.mcnemar(cB, cA)
    mf1_a, mf1_b = _mf1(y, pred_a, len(np.unique(y))), _mf1(y, pred_b, len(np.unique(y)))
    hyp.add(family, "McNemar paired", label, float(mf1_b - mf1_a), p)
    return dict(label=label, mf1_a=round(mf1_a, 4), mf1_b=round(mf1_b, 4),
                mf1_delta=round(mf1_b - mf1_a, 4), mcnemar_p=round(p, 6))


# ===================================================== Phase 13/25: addition
def info_addition(y_te, probs_by_model, hyp, n_boot=2000):
    """A0 vs A5/A6/A7 on identical test samples: MF1/NLL/Brier deltas with
    bootstrap CI, paired permutation, McNemar, effect size."""
    out = {}
    base = "A0"
    y = np.asarray(y_te)
    for name, probs in probs_by_model.items():
        if name == base:
            continue
        p0, p1 = probs_by_model[base], probs
        d0, d1 = p0.argmax(1), p1.argmax(1)
        rec = dict(
            mf1_delta=round(_mf1(y, d1, p1.shape[1]) - _mf1(y, d0, p0.shape[1]), 4),
            nll_delta=round(float(nll_fn(p1, y) - nll_fn(p0, y)), 4),
            brier_delta=round(float(brier(p1, y) - brier(p0, y)), 4))
        # McNemar
        rec["mcnemar"] = _mcnemar_perm(y, d0, d1, hyp,
                                       "model_comparisons", f"{name} vs {base}")
        # paired permutation on per-sample correctness
        c0, c1 = (d0 == y).astype(float), (d1 == y).astype(float)
        _, p_perm = S.paired_permutation_test(c1, c0)
        hyp.add("model_comparisons", "paired permutation",
                f"{name} vs {base} accuracy", float(c1.mean() - c0.mean()),
                float(p_perm))
        rec["perm_p"] = round(float(p_perm), 5)
        # bootstrap CI on the accuracy diff
        acc_d, lo, hi = S.bootstrap_diff_ci(c1, c0, paired=True, n_boot=n_boot)
        rec["acc_diff_ci95"] = [round(float(lo), 4), round(float(hi), 4)]
        rec["cohens_d"] = round(float(S.cohens_d_paired(c1, c0)), 3)
        out[name] = rec
    return out


# ============================================ Phases 14-15: complementarity
def geometry_similarity(Z_blocks, keys, max_n=MAX_N):
    """Pairwise correlation / cosine / CKA between geometry feature blocks."""
    n = min(len(Z_blocks[0]), max_n)
    out = {"keys": keys, "pairs": {}}
    reps = []
    for Z in Z_blocks:
        X = Z[:n].astype(np.float64)
        X = X - X.mean(0, keepdims=True)
        nrm = np.linalg.norm(X)
        reps.append(X / (nrm if nrm > 0 else 1.0))
    for i in range(len(reps)):
        for j in range(i + 1, len(reps)):
            a, b = reps[i], reps[j]
            # feature-space correlation (align columns by pairing top-k?) —
            # use sample-space cosine + linear CKA (both well-defined for
            # differing column counts)
            cos = float((a * b).sum())
            # linear CKA
            ka, kb = a @ a.T, b @ b.T
            ka_c = ka - ka.mean(0, keepdims=True) - ka.mean(1, keepdims=True) + ka.mean()
            kb_c = kb - kb.mean(0, keepdims=True) - kb.mean(1, keepdims=True) + kb.mean()
            cka = float((ka_c * kb_c).sum() /
                        (np.linalg.norm(ka_c) * np.linalg.norm(kb_c) + 1e-12))
            # effective rank of each block's Gram matrix
            def eff_rank(K):
                ev = np.linalg.eigvalsh(K)
                ev = np.clip(ev, 0, None)
                p = ev / (ev.sum() + 1e-12)
                p = p[p > 1e-12]
                return float(np.exp(-(p * np.log(p)).sum()))
            out["pairs"][f"{keys[i]}|{keys[j]}"] = dict(
                sample_cosine=round(cos, 4), linear_cka=round(cka, 4),
                eff_rank_a=round(eff_rank(ka), 1), eff_rank_b=round(eff_rank(kb), 1))
    return out


def incremental_gain(y_te, y_va, probs_by_keys, hyp):
    """leave-one-geometry-out: delta_j = perf(all) - perf(all minus j)."""
    out = {}
    full = probs_by_keys["ALL"]
    mf1_full = _mf1(y_te, full.argmax(1), full.shape[1])
    for k, probs in probs_by_keys.items():
        if k == "ALL":
            continue
        mf1 = _mf1(y_te, probs.argmax(1), probs.shape[1])
        d = mf1_full - mf1
        out[k] = dict(mf1_without=round(mf1, 4), delta_j=round(d, 4))
        # McNemar full vs without-k
        _mcnemar_perm(y_te, full.argmax(1), probs.argmax(1), hyp,
                      "geometry_complementarity", f"ALL vs ALL-minus-{k}")
    out["full_mf1"] = round(mf1_full, 4)
    return out


# =============================================== Phase 16: class-conditional
def class_conditional(y_te, preds_by_model, n_cls):
    out = {}
    from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
    for name, pred in preds_by_model.items():
        per_f1 = f1_score(y_te, pred, average=None, labels=list(range(n_cls)),
                          zero_division=0)
        out[name] = dict(per_class_f1=[round(float(x), 4) for x in per_f1])
    return out


# ================================================== Phase 17: contributions
def group_contributions(model, feats_test, y_te, hyp):
    """Exact per-geometry logit contributions from the linear readout."""
    gl = model.group_logits(feats_test)          # list of [N, C]
    keys = model.keys_
    probs = model.predict_split(feats_test)
    pred = probs.argmax(1)
    conf = probs.max(1)
    out = {"keys": keys, "per_group": {}}
    for j, k in enumerate(keys):
        g = gl[j]
        out["per_group"][k] = dict(
            logit_norm_mean=round(float(np.linalg.norm(g, axis=1).mean()), 4),
            logit_norm_median=round(float(np.median(np.linalg.norm(g, axis=1))), 4))
    # contribution by correctness (top half vs bottom half by |logit|)
    for j, k in enumerate(keys):
        gn = np.linalg.norm(gl[j], axis=1)
        correct = pred == y_te
        out["per_group"][k]["logit_norm_correct"] = round(float(gn[correct].mean()), 4)
        out["per_group"][k]["logit_norm_wrong"] = round(float(gn[~correct].mean()), 4)
    return out


# ============================================== Phase 18: zeroing ablation
def zeroing_ablation(model, feats_test, y_te, hyp):
    """Zero each standardized geometry block (no retraining) and re-evaluate."""
    out = {}
    keys = model.keys_
    probs0 = model.predict_split(feats_test)
    pred0 = probs0.argmax(1)
    for j, k in enumerate(keys):
        # build a zeroed copy of the model matrix
        import copy
        all_keys = ["raw"] + list(model.ex.geometry_names)
        blocks = [model.scaler_all.transform_block(m, feats_test[kk])
                  for m, kk in enumerate(all_keys)]
        by_key = dict(zip(all_keys, blocks))
        by_key[k] = np.zeros_like(by_key[k])
        Z = np.concatenate([by_key[kk] for kk in keys], axis=1)
        probs = model._rd.predict_proba(Z)
        d = probs0.max(1) - probs.max(1)
        out[k] = dict(
            mf1_zeroed=round(_mf1(y_te, probs.argmax(1), model.n_classes), 4),
            mean_prob_change=round(float(np.abs(d).mean()), 4),
            flip_rate=round(float((probs.argmax(1) != pred0).mean()), 4),
            nll_change=round(float(nll_fn(probs, y_te) - nll_fn(probs0, y_te)), 4))
    return out


# ================================================ Phase 20: perturbations
def perturbation_response(model, X_te, feats_test_fn, y_te, max_n=MAX_N):
    """Apply controlled degradations; measure per-geometry feature change,
    prediction change, confidence, entropy, error rate."""
    rng = np.random.default_rng(8800)
    n = min(len(X_te), max_n)
    X = X_te[:n]
    y = np.asarray(y_te)[:n]
    out = {"kinds": {}}
    base_probs = model.predict_split(feats_test_fn(X[:n])) if feats_test_fn else None
    for kind, spec in DEG_SPECS.items():
        rec = {"levels": {}}
        for lv in spec["levels"]:
            Xp = np.stack([PB.apply_spec(x[0], kind, lv, rng)[0] for x in X])[:, None, :]
            feats = model.ex.extract_features(Xp) if feats_test_fn is None else feats_test_fn(Xp)
            probs = model.predict_split(feats)
            pred = probs.argmax(1)
            ent = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(1)
            rec["levels"][str(lv)] = dict(
                error_rate=round(float((pred != y).mean()), 4),
                mean_conf=round(float(probs.max(1).mean()), 4),
                mean_entropy=round(float(ent.mean()), 4),
                mf1=round(_mf1(y, pred, model.n_classes), 4))
        out["kinds"][kind] = rec
    return out


# ================================================== Phase 21: faithfulness
def geometry_faithfulness(model, X_te, y_te, hyp, max_n=MAX_N, frac=0.10):
    """Phase 21: for samples where a geometry contributes strongly (top-30%
    by mean |group logit|), mask the highest-magnitude window of its
    transport FIELD (transport-space intervention; raw view untouched) vs a
    matched random window, and compare confidence drops. Batched; registers
    targeted-vs-random paired permutation tests in the registry."""
    rng = np.random.default_rng(7700)
    n = min(len(X_te), max_n)
    X = X_te[:n]
    f0 = model.ex.extract_features(X)
    p0 = model.predict_split(f0)
    c0, pred0 = p0.max(1), p0.argmax(1)
    gl = model.group_logits(f0)
    views = model.ex.extract_views(X)             # dict k -> [n, T]
    arr_keys = [k for k, v in f0.items() if isinstance(v, np.ndarray)]
    out = {}
    for j, k in enumerate(model.keys_):
        V = views[k]
        n_sel_len = V.shape[-1]
        width = max(3, int(frac * n_sel_len))
        imp = np.abs(gl[j]).mean(1)
        idx = np.argsort(-imp)[: max(8, int(0.3 * n))]
        idx = np.sort(idx)
        # smooth magnitude -> targeted window center per selected sample
        kernel = np.ones(width) / width
        mag = np.abs(V)
        smooth = np.apply_along_axis(
            lambda r: np.convolve(r, kernel, mode="valid"), 1, mag[idx])
        t_center = smooth.argmax(1)               # start index of best window
        r_center = rng.integers(0, n_sel_len - width + 1, size=len(idx))
        drops, flips = {}, {}
        fill = np.median(V[idx], axis=1, keepdims=True)   # neutral field value
        for kind, centers in (("target", t_center), ("random", r_center)):
            Vp = V[idx].copy()
            for row, c in enumerate(centers):
                Vp[row, c:c + width] = fill[row]
            feats = {kk: f0[kk][idx] for kk in arr_keys}
            feats[k] = model.ex.bank.features(Vp[:, None, :],
                                              stats=model.ex.stats)
            p = model.predict_split(feats)
            drops[kind] = c0[idx] - p.max(1)
            flips[kind] = (p.argmax(1) != pred0[idx]).astype(float)
        t, r = drops["target"], drops["random"]
        diff = t - r
        _, lo, hi = S.bootstrap_ci(diff, n_boot=500)
        _, p_perm = S.paired_permutation_test(t, r, n_perm=500)
        rec = dict(n_selected=int(len(idx)), window=width,
                   target_drop=round(float(t.mean()), 4),
                   random_drop=round(float(r.mean()), 4),
                   diff=round(float(diff.mean()), 4),
                   ci95=[round(float(lo), 4), round(float(hi), 4)],
                   p_perm=round(float(p_perm), 5),
                   flip_rate_targeted=round(float(flips["target"].mean()), 4),
                   flip_rate_random=round(float(flips["random"].mean()), 4))
        out[k] = rec
        if hyp is not None:
            hyp.add("perturbation", "paired permutation",
                    f"{k} targeted vs random geometry-field masking",
                    rec["diff"], rec["p_perm"])
    return out


# ===================================================== Phase 22: stability
def stability(model, X_te, y_te, max_n=MAX_N):
    """Class-preserving benign transforms: geometry-feature cosine + pred agreement."""
    rng = np.random.default_rng(8600)
    n = min(len(X_te), max_n)
    X = X_te[:n]
    y = np.asarray(y_te)[:n]
    f0 = model.ex.extract_features(X)
    p0 = model.predict_split(f0)
    out = {"specs": {}}
    for name, spec in BENIGN_SPECS.items():
        Xp = np.stack([PB.apply_spec(x[0], spec["kind"], spec["level"], rng)[0]
                       for x in X])[:, None, :]
        f1 = model.ex.extract_features(Xp)
        p1 = model.predict_split(f1)
        rec = {"per_geometry": {}, "pred_agreement": None}
        for k in model.ex.geometry_names:
            a = f0[k][:n].astype(np.float64)
            b = f1[k][:n].astype(np.float64)
            num = (a * b).sum(1)
            den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
            rec["per_geometry"][k] = round(float((num / den).mean()), 4)
        rec["pred_agreement"] = round(float((p0.argmax(1) == p1.argmax(1)).mean()), 4)
        out["specs"][name] = rec
    return out


# ================================================= Phase 23: calibration
def calibration_table(probs_by_model, y_te):
    out = {}
    for name, probs in probs_by_model.items():
        out[name] = dict(
            ece=round(float(ece(probs, y_te)), 4),
            adaptive_ece=round(float(adaptive_ece(probs, y_te)), 4),
            brier=round(float(brier(probs, y_te)), 4),
            nll=round(float(nll_fn(probs, y_te)), 4))
    return out


# ============================================ Phase 24: selective prediction
def selective_prediction(probs_by_model, y_te):
    from sklearn.metrics import roc_auc_score, average_precision_score
    out = {}
    for name, probs in probs_by_model.items():
        pred = probs.argmax(1)
        conf = probs.max(1)
        ent = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(1)
        margin = np.sort(probs, 1)[:, -1] - np.sort(probs, 1)[:, -2]
        err = (pred != np.asarray(y_te)).astype(int)
        def _aurc(scores, err):
            order = np.argsort(-scores)
            e = err[order]
            risks = np.cumsum(e) / np.arange(1, len(e) + 1)
            return float(risks.mean())
        rec = {}
        for sig_name, s in (("confidence", conf), ("neg_entropy", -ent),
                            ("margin", margin)):
            try:
                au = roc_auc_score(err, -s) if sig_name != "confidence" \
                    else roc_auc_score(err, s)
            except ValueError:
                au = float("nan")
            rec[sig_name] = dict(auroc=round(float(au), 4),
                                 auprc=round(float(average_precision_score(err, s)), 4),
                                 aurc=round(_aurc(s, err), 4))
        out[name] = rec
    return out
