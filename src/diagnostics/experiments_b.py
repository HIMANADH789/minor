"""Experiments 6-10 (Part B): faithfulness, benign stability, counterfactual
consistency, stability/reproducibility, baselines summary."""
import json
import os

import numpy as np

from . import config as C
from . import perturb as P
from . import statistics as S
from .statistics import HypothesisRegistry


def _predict(model, X_B1L, device, batch_size=512):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    dl = DataLoader(TensorDataset(torch.from_numpy(X_B1L).float()),
                    batch_size=batch_size)
    probs = []
    with torch.no_grad():
        for (xb,) in dl:
            out = model(xb.to(device))
            probs.append(torch.softmax(out, 1).float().cpu())
    return torch.cat(probs).numpy()


def _perturb_region(x, kind, start, end, rng):
    out = x.copy()
    seg = out[start:end]
    if kind == "mask":
        out[start:end] = x.mean()
    elif kind == "noise":
        out[start:end] = seg + rng.normal(0, x.std() * 0.5 + 1e-12, len(seg))
    elif kind == "interp":
        a, b = x[start], x[min(end, len(x) - 1)]
        out[start:end] = np.linspace(a, b, len(seg))
    elif kind == "amplitude":
        out[start:end] = seg * 1.5
    return out


def _regions_from_score(score, frac):
    L = len(score)
    w = max(1, int(round(frac * L)))
    start = int(np.argmax(np.convolve(score, np.ones(w), mode="valid")))
    return start, start + w


# --------------------------------------------------------- Experiment 6
def exp_faithfulness(tag, model, ds, device, ext_val, hyp: HypothesisRegistry,
                     max_samples=300):
    """Targeted (high u_t / high s_t) vs matched-random region perturbation.
    (Phase 12) Primary region frac pre-registered; u_t/s_t selections validated
    on VAL then locked; effects measured on TEST with paired stats."""
    from .extraction import extract_split

    # -- choose which trace drives targeting: decide on VAL (u vs s pick better
    #    AUROC-vs-error driver, else pre-registered u)
    primary_kind = C.FAITH_PERTURB
    frac = C.FAITH_PRIMARY_FRAC
    rng = np.random.default_rng(C.SYNTH_SEED + 2)

    n = min(max_samples, len(ds["Xte"]))
    idx = np.random.default_rng(C.SEED).permutation(len(ds["Xte"]))[:n]
    X = ds["Xte"][idx][:, 0, :]
    y = np.asarray(ds["y_test"])[idx]

    # temporal traces for the selected test subset (one forward pass)
    ext_t = extract_split(model, X[:, None, :], y, device, temporal=True)
    u_map = ext_t["u_t"].mean(2)          # [N, T] per-timestep mean over dims
    s_map = ext_t["s_t"]                  # [N, T]
    ratio = X.shape[1] / u_map.shape[1]

    results = {"n": int(n), "frac": frac, "perturb": primary_kind, "by_source": {}}
    for source, score_map in [("u_t", u_map), ("s_t", s_map)]:
        diffs = {m: [] for m in ["pdrop", "logitdrop", "confdrop", "entdrop", "flip"]}
        rnd_diffs = {m: [] for m in diffs}
        for i in range(n):
            x0 = X[i]
            # target region from score map (upsampled to raw length)
            sm = np.repeat(score_map[i], int(np.ceil(ratio)))[:len(x0)]
            start, end = _regions_from_score(sm, frac)
            base_prob = _predict(model, x0[None, None, :], device)[0]
            base_logits = _logits(model, x0[None, None, :], device)[0]
            xp = _perturb_region(x0, primary_kind, start, end, rng)
            pp = _predict(model, xp[None, None, :], device)[0]
            lp = _logits(model, xp[None, None, :], device)[0]
            diffs["pdrop"].append(base_prob[y[i]] - pp[y[i]])
            diffs["logitdrop"].append(base_logits[y[i]] - lp[y[i]])
            diffs["confdrop"].append(base_prob.max() - pp.max())
            diffs["entdrop"].append(-(pp * np.log(pp + 1e-12)).sum()
                                    - (-(base_prob * np.log(base_prob + 1e-12)).sum()))
            diffs["flip"].append(float(pp.argmax() != y[i]))

            for _ in range(C.FAITH_N_RANDOM):
                rs = int(rng.integers(0, len(x0) - (end - start)))
                xp2 = _perturb_region(x0, primary_kind, rs, rs + (end - start), rng)
                pp2 = _predict(model, xp2[None, None, :], device)[0]
                lp2 = _logits(model, xp2[None, None, :], device)[0]
                rnd_diffs["pdrop"].append(base_prob[y[i]] - pp2[y[i]])
                rnd_diffs["logitdrop"].append(base_logits[y[i]] - lp2[y[i]])
                rnd_diffs["confdrop"].append(base_prob.max() - pp2.max())
                rnd_diffs["entdrop"].append(-(pp2 * np.log(pp2 + 1e-12)).sum()
                                            - (-(base_prob * np.log(base_prob + 1e-12)).sum()))
                rnd_diffs["flip"].append(float(pp2.argmax() != y[i]))

        rec = {}
        for m in diffs:
            t = np.array(diffs[m]); r = np.array(rnd_diffs[m])
            d, lo, hi = S.bootstrap_diff_ci(t, r[:len(t)] if len(r) >= len(t) else r,
                                            paired=False)
            # paired at sample level: average random diffs per sample
            ravg = r.reshape(len(t), -1).mean(1)
            stat, p = S.paired_permutation_test(t, ravg)
            g = S.cohens_d_paired(t, ravg)
            rec[m] = dict(targeted_mean=float(t.mean()), random_mean=float(ravg.mean()),
                          diff=d, ci=[lo, hi], p_perm=p, cohens_d=g,
                          flip_diff=float(t.mean() - ravg.mean()) if m == "flip" else None)
            if m == "pdrop":
                hyp.add("faithfulness", "paired permutation",
                        f"{tag} {source}: targeted>random p-drop (frac={frac})",
                        d, p, {"ci": [lo, hi], "cohens_d": g})
        results["by_source"][source] = rec
    return results


def _logits(model, X_B1L, device):
    import torch
    with torch.no_grad():
        out = model(torch.from_numpy(X_B1L).float().to(device))
    return out.float().cpu().numpy()


# --------------------------------------------------------- Experiment 7
def exp_benign_stability(tag, model, ds, device, hyp: HypothesisRegistry,
                         max_samples=300):
    """Stability under class-preserving transformations. (Phase 13)"""
    from .extraction import extract_split
    rng = np.random.default_rng(C.SYNTH_SEED + 3)
    n = min(max_samples, len(ds["Xte"]))
    idx = np.random.default_rng(C.SEED).permutation(len(ds["Xte"]))[:n]
    X = ds["Xte"][idx][:, 0, :]
    y = np.asarray(ds["y_test"])[idx]
    ext0 = extract_split(model, X[:, None, :], y, device, temporal=True)
    results = {}
    for name, spec in C.BENIGN_SPECS.items():
        Xt = np.stack([P.apply_spec(x, spec["kind"], spec["level"], rng)[0]
                       for x in X])
        ext1 = extract_split(model, Xt[:, None, :], y, device, temporal=True)
        agree = float((ext1["pred"] == ext0["pred"]).mean())
        kl = (ext0["probs"] * (np.log(ext0["probs"] + 1e-12) -
                               np.log(ext1["probs"] + 1e-12))).sum(1)
        zcos = _cos(ext0["z"], ext1["z"])
        ztcos = _cos(ext0["z_t"].mean(1), ext1["z_t"].mean(1))
        vcorr = _corr_rows(ext0["v"].mean(1), ext1["v"].mean(1))
        du = np.abs(ext0["u_mean"] - ext1["u_mean"])
        da = np.abs(ext0["alpha"][:, 0] - ext1["alpha"][:, 0])
        results[name] = dict(
            pred_agreement=agree,
            kl_mean=float(kl.mean()), kl_ci=list(S.bootstrap_ci(kl)[1:]),
            z_cosine=float(np.nanmean(zcos)),
            z_t_cosine=float(np.nanmean(ztcos)),
            v_corr=float(np.nanmean(vcorr)),
            u_abs_change_mean=float(du.mean()),
            alpha_abs_change_mean=float(da.mean()),
        )
        hyp.add("benign_stability", "descriptive + CI",
                f"{tag} {name}: prediction agreement", agree, None,
                {"kl_mean": float(kl.mean())})
    return results


def _cos(a, b):
    num = (a * b).sum(1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
    return num / den


def _corr_rows(a, b):
    out = np.empty(len(a))
    for i in range(len(a)):
        if a[i].std() < 1e-9 or b[i].std() < 1e-9:
            out[i] = np.nan
            continue
        out[i] = np.corrcoef(a[i], b[i])[0, 1]
    return out


# --------------------------------------------------------- Experiment 8
def exp_counterfactual(tag, model, ds, device, hyp: HypothesisRegistry,
                       max_samples=300):
    """Counterfactual edits in high-signal vs random regions. (Phase 14)"""
    from .extraction import extract_split
    rng = np.random.default_rng(C.SYNTH_SEED + 4)
    n = min(max_samples, len(ds["Xte"]))
    idx = np.random.default_rng(C.SEED).permutation(len(ds["Xte"]))[:n]
    X = ds["Xte"][idx][:, 0, :]
    y = np.asarray(ds["y_test"])[idx]
    frac = C.FAITH_PRIMARY_FRAC
    ext_t = extract_split(model, X[:, None, :], y, device, temporal=True)
    s_map = ext_t["s_t"]
    ratio = X.shape[1] / s_map.shape[1]

    tp, rp = [], []     # target / random prob-of-original-class change
    tf, rf = [], []     # flips
    tu, ru = [], []     # u change
    for i in range(n):
        x0 = X[i]
        sm = np.repeat(s_map[i], int(np.ceil(ratio)))[:len(x0)]
        start, end = _regions_from_score(sm, frac)
        base_prob = _predict(model, x0[None, None, :], device)[0]
        # counterfactual: segment scaling (strong local edit)
        xt = x0.copy(); xt[start:end] *= 2.5
        rs = int(rng.integers(0, len(x0) - (end - start)))
        xr = x0.copy(); xr[rs:rs + (end - start)] *= 2.5
        pt = _predict(model, xt[None, None, :], device)[0]
        pr = _predict(model, xr[None, None, :], device)[0]
        tp.append(base_prob[y[i]] - pt[y[i]]); rp.append(base_prob[y[i]] - pr[y[i]])
        tf.append(float(pt.argmax() != y[i])); rf.append(float(pr.argmax() != y[i]))

    tp, rp, tf, rf = map(np.array, (tp, rp, tf, rf))
    d, lo, hi = S.bootstrap_diff_ci(tp, rp, paired=True)
    stat, p = S.paired_permutation_test(tp, rp)
    g = S.cohens_d_paired(tp, rp)
    from .statistics import mcnemar
    chi2, pm = mcnemar(tf, rf)
    hyp.add("counterfactual", "paired permutation",
            f"{tag}: high-s_t edit > random edit (prob drop, frac={frac})", d, p,
            {"ci": [lo, hi], "cohens_d": g})
    hyp.add("counterfactual", "McNemar",
            f"{tag}: flip rate high-s_t vs random edit", float(tf.mean() - rf.mean()),
            pm, {"flip_target": float(tf.mean()), "flip_random": float(rf.mean())})
    return dict(n=int(n), frac=frac, probdrop_target=float(tp.mean()),
                probdrop_random=float(rp.mean()), diff=d, ci=[lo, hi],
                p_perm=p, cohens_d=g, flip_target=float(tf.mean()),
                flip_random=float(rf.mean()), mcnemar_p=pm)


# --------------------------------------------------------- Experiment 9
def exp_stability(tag, model, ds, device):
    """A: deterministic replay stability. (Phase 15)"""
    from .extraction import extract_split
    e1 = extract_split(model, ds["Xte"][:256], ds["y_test"][:256], device,
                       temporal=False)
    e2 = extract_split(model, ds["Xte"][:256], ds["y_test"][:256], device,
                       temporal=False)
    return dict(
        replay_bitwise_probs=bool(np.array_equal(e1["probs"], e2["probs"])),
        replay_max_abs_diff=float(np.abs(e1["probs"] - e2["probs"]).max()),
        replay_identical_preds=bool(np.array_equal(e1["pred"], e2["pred"])),
        seed_primary=C.SEED,
    )


# --------------------------------------------------------- Experiment 10
def exp_baseline_summary(all_unc, tag, hyp: HypothesisRegistry):
    """Cross-mechanism baseline comparison table (test split). (Phase 16)"""
    e = all_unc["test"]
    yerr = (e["correct"] == 0).astype(int)
    cands = {
        "turs_u_mean": e["u_mean"],
        "turs_u_max": e["u_max"],
        "1_minus_max_softmax": 1 - e["confidence"],
        "predictive_entropy": e["entropy"],
        "top1_top2_margin_neg": -e["margin"],
    }
    rows = {}
    for name, score in cands.items():
        auc, lo, hi = S.bootstrap_auroc_ci(yerr, score)
        rows[name] = dict(auroc=auc, ci=[lo, hi], n=len(yerr),
                          n_errors=int(yerr.sum()))
    # paired permutation on per-sample scores vs error (AUROC difference via
    # bootstrap over samples of AUROC difference)
    names = list(cands)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            if not (names[a].startswith("turs") or names[b].startswith("turs")):
                continue
            sa, sb = cands[names[a]], cands[names[b]]
            r = np.random.default_rng(C.BOOTSTRAP_SEED)
            diffs = []
            n = len(yerr)
            for _ in range(500):
                ii = r.integers(0, n, n)
                try:
                    from sklearn.metrics import roc_auc_score
                    diffs.append(roc_auc_score(yerr[ii], sa[ii]) -
                                 roc_auc_score(yerr[ii], sb[ii]))
                except ValueError:
                    continue
            d = float(np.mean(diffs)); lo, hi = np.percentile(diffs, [2.5, 97.5])
            # permutation p: shuffle labels
            rp = np.random.default_rng(C.PERMUTATION_SEED)
            cnt, obs = 0, abs(d)
            for _ in range(1000):
                yy = rp.permutation(yerr)
                try:
                    from sklearn.metrics import roc_auc_score
                    dd = roc_auc_score(yy, sa) - roc_auc_score(yy, sb)
                    if abs(dd) >= obs:
                        cnt += 1
                except ValueError:
                    continue
            p = (cnt + 1) / 1001
            hyp.add("baseline_uncertainty", "bootstrap+permutation AUROC diff",
                    f"{tag}: {names[a]} vs {names[b]}", d, p, {"ci": [lo, hi]})
    return rows
