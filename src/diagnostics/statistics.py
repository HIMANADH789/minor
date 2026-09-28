"""Shared statistical utilities: bootstrap CIs, permutation tests, Wilcoxon,
paired effect sizes, FDR correction, serialization helpers."""
import json
import os

import numpy as np
from scipy import stats

BOOTSTRAP_SEED = 4200
PERMUTATION_SEED = 4300
N_BOOTSTRAP = 2000
N_PERMUTATION = 2000
RNG_POOL = {}


def rng(name):
    """Deterministic per-purpose RNG."""
    if name not in RNG_POOL:
        seed = {"boot": BOOTSTRAP_SEED, "perm": PERMUTATION_SEED}.get(name, 0)
        RNG_POOL[name] = np.random.default_rng(seed)
    return RNG_POOL[name]


# ---------------------------------------------------------------- bootstrap
def bootstrap_ci(x, stat=np.mean, n_boot=N_BOOTSTRAP, ci=0.95, seed_name="boot"):
    """CI for stat(x). x: 1-D array. Returns (point, lo, hi)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(stat(x))
    if len(x) == 1:
        return point, point, point
    r = rng(seed_name)
    idx = r.integers(0, len(x), size=(n_boot, len(x)))
    boots = stat(x[idx], axis=1)
    lo, hi = np.percentile(boots, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return point, float(lo), float(hi)


def bootstrap_diff_ci(a, b, paired=True, n_boot=N_BOOTSTRAP, ci=0.95):
    """CI for mean(a) - mean(b); paired resampling only when lengths match."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if paired and len(a) == len(b):
        r = rng("boot")
        n = len(a)
        idx = r.integers(0, n, size=(n_boot, n))
        diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    else:
        r = rng("boot")
        diffs = np.empty(n_boot)
        for i in range(n_boot):
            diffs[i] = a[r.integers(0, len(a), len(a))].mean() - \
                       b[r.integers(0, len(b), len(b))].mean()
    if len(diffs) == 0 or not np.isfinite(diffs).all():
        d = float(np.mean(a) - np.mean(b)) if len(a) and len(b) else float("nan")
        return d, float("nan"), float("nan")
    d = float(np.mean(a) - np.mean(b))
    lo, hi = np.percentile(diffs, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return d, float(lo), float(hi)


def bootstrap_auroc_ci(y_true, scores, n_boot=N_BOOTSTRAP, ci=0.95):
    """AUROC point estimate and percentile bootstrap CI (positive-class AUC)."""
    from sklearn.metrics import roc_auc_score
    y_true = np.asarray(y_true); scores = np.asarray(scores, float)
    try:
        point = float(roc_auc_score(y_true, scores))
    except ValueError:
        return float("nan"), float("nan"), float("nan")
    pos = np.where(y_true == 1)[0]; neg = np.where(y_true == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return point, float("nan"), float("nan")
    r = rng("boot")
    vals = np.empty(n_boot)
    for i in range(n_boot):
        sp = r.choice(pos, len(pos), replace=True)
        sn = r.choice(neg, len(neg), replace=True)
        yy = np.concatenate([np.ones(len(sp)), np.zeros(len(sn))])
        ss = np.concatenate([scores[sp], scores[sn]])
        try:
            vals[i] = roc_auc_score(yy, ss)
        except ValueError:
            vals[i] = np.nan
    lo, hi = np.nanpercentile(vals, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return point, float(lo), float(hi)


# ------------------------------------------------------------- correlations
def corr_with_inference(x, y, method="spearman", n_perm=N_PERMUTATION):
    """Correlation with permutation p-value and bootstrap CI."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 10:
        return dict(rho=float("nan"), p_perm=float("nan"), ci_lo=float("nan"),
                    ci_hi=float("nan"), n=int(len(x)))
    fn = stats.spearmanr if method == "spearman" else stats.pearsonr
    rho = float(fn(x, y)[0])
    r = rng("perm")
    count = 0
    for _ in range(n_perm):
        ry = r.permutation(y)
        if abs(fn(x, ry)[0]) >= abs(rho):
            count += 1
    p = (count + 1) / (n_perm + 1)
    # bootstrap CI on the statistic
    r = rng("boot")
    vals = []
    for _ in range(N_BOOTSTRAP):
        idx = r.integers(0, len(x), len(x))
        try:
            vals.append(fn(x[idx], y[idx])[0])
        except Exception:
            pass
    lo, hi = np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)
    return dict(rho=rho, p_perm=float(p), ci_lo=float(lo), ci_hi=float(hi),
                n=int(len(x)))


# ------------------------------------------------------------ paired tests
def paired_permutation_test(a, b, n_perm=N_PERMUTATION):
    """Two-sided paired permutation test on mean(a-b)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = a - b
    d = d[np.isfinite(d)]
    obs = float(np.mean(d)) if len(d) else np.nan
    r = rng("perm")
    count = 0
    for _ in range(n_perm):
        signs = r.choice([-1.0, 1.0], size=len(d))
        if abs(np.mean(d * signs)) >= abs(obs):
            count += 1
    p = (count + 1) / (n_perm + 1)
    return obs, float(p)


def wilcoxon_signed(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = a - b
    d = d[np.isfinite(d)]
    if len(d) < 10 or np.allclose(d, 0):
        return float("nan"), float("nan")
    try:
        w = stats.wilcoxon(a, b, zero_method="wilcox")
        return float(w.statistic), float(w.pvalue)
    except Exception:
        return float("nan"), float("nan")


def mcnemar(pred_a_correct, pred_b_correct):
    """Paired binary correctness comparison. Returns (chi2, p)."""
    a = np.asarray(pred_a_correct, bool); b = np.asarray(pred_b_correct, bool)
    n01 = int(np.sum(~a & b)); n10 = int(np.sum(a & ~b))
    if n01 + n10 == 0:
        return 0.0, 1.0
    chi2 = (abs(n10 - n01) - 1) ** 2 / (n10 + n01)
    p = float(stats.chi2.sf(chi2, 1))
    return float(chi2), p


# ------------------------------------------------------------- effect sizes
def cohens_d_paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    if len(d) < 2 or np.std(d, ddof=1) < 1e-12:
        return float("nan")
    return float(np.mean(d) / np.std(d, ddof=1))


def cliffs_delta(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return float((gt - lt) / (len(a) * len(b)))


def cliffs_label(delta):
    d = abs(delta) if np.isfinite(delta) else 0
    if d < 0.147: return "negligible"
    if d < 0.33:  return "small"
    if d < 0.474: return "medium"
    return "large"


def rank_biserial_from_diffs(d):
    """Rank-biserial correlation for paired differences."""
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    d = d[d != 0]
    if len(d) == 0:
        return float("nan")
    ranks = stats.rankdata(np.abs(d))
    return float(np.sum(ranks * np.sign(d)) / ranks.sum())


# --------------------------------------------------------------------- FDR
def benjamini_hochberg(pvals):
    """BH-FDR. Returns adjusted q-values (input order preserved)."""
    p = np.asarray(pvals, float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(n)
    out[order] = q
    return out


class HypothesisRegistry:
    """Collects hypothesis tests across experiments; applies FDR per family."""
    def __init__(self):
        self.rows = []

    def add(self, family, test, comparison, estimate, p, extra=None):
        self.rows.append(dict(family=family, test=test, comparison=comparison,
                              estimate=None if estimate is None else float(estimate),
                              p_value=None if p is None else float(p),
                              extra=extra or {}))

    def finalize(self):
        fams = sorted(set(r["family"] for r in self.rows))
        for fam in fams:
            idx = [i for i, r in enumerate(self.rows) if r["family"] == fam]
            ps = [self.rows[i]["p_value"] if self.rows[i]["p_value"] is not None
                  else 1.0 for i in idx]
            qs = benjamini_hochberg(ps)
            for i, q in zip(idx, qs):
                self.rows[i]["q_value"] = float(q)
                self.rows[i]["alpha"] = 0.05
                self.rows[i]["significant"] = bool(q < 0.05 and
                    self.rows[i]["p_value"] is not None)
        return self.rows


# ------------------------------------------------------------ serialization
def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and (np.isnan(obj) if isinstance(obj, float) else False):
        return None
    return obj


def dump_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(jsonable(obj), f, indent=2)


def dump_csv(rows, path, columns=None):
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = jsonable(rows)
    if not rows:
        with open(path, "w") as f:
            csv.writer(f).writerow(columns or [])
        return
    if columns is None:
        columns = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})
