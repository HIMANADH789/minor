"""Calibration + selective prediction (Phase 9C/9D)."""
import numpy as np

from . import config as C
from . import statistics as S


def ece(probs, y, n_bins=15):
    conf = probs.max(1)
    pred = probs.argmax(1)
    y = np.asarray(y)
    correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() == 0:
            continue
        e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def adaptive_ece(probs, y, n_bins=15):
    """Equal-mass binning ECE."""
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == np.asarray(y)).astype(float)
    order = np.argsort(conf)
    e = 0.0
    n = len(conf)
    for i in range(n_bins):
        idx = order[i * n // n_bins:(i + 1) * n // n_bins]
        if len(idx) == 0:
            continue
        e += len(idx) / n * abs(correct[idx].mean() - conf[idx].mean())
    return float(e)


def brier(probs, y):
    y = np.asarray(y)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def nll(probs, y):
    y = np.asarray(y)
    return float(-np.log(probs[np.arange(len(y)), y] + 1e-12).mean())


def reliability_curve(probs, y, n_bins=10):
    conf = probs.max(1)
    correct = (probs.argmax(1) == np.asarray(y)).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    out = []
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() == 0:
            out.append(dict(bin_center=float((bins[i] + bins[i + 1]) / 2),
                            acc=None, conf=None, n=0))
            continue
        out.append(dict(bin_center=float((bins[i] + bins[i + 1]) / 2),
                        acc=float(correct[m].mean()), conf=float(conf[m].mean()),
                        n=int(m.sum())))
    return out


def calibration_summary(ext, split="test"):
    e = ext[split]
    probs = e["probs"]; y = e["y"]
    res = dict(
        ece=ece(probs, y), adaptive_ece=adaptive_ece(probs, y),
        brier=brier(probs, y), nll=nll(probs, y),
        mean_confidence=float(e["confidence"].mean()),
        reliability_curve=reliability_curve(probs, y),
        nll_ci=list(S.bootstrap_ci(-np.log(probs[np.arange(len(y)), y] + 1e-12))[1:]),
    )
    return res


def risk_coverage(e, score=None, ascending=False):
    """Selective prediction: sort by uncertainty score; risk at coverage levels.
    e: split dict; score: higher = more uncertain. Returns rows + AURC.
    Accepts Lite-style keys (correct/pred) or Stack-style (final_correct/final_pred).
    """
    if score is None:
        score = e["u_mean"]
    y = np.asarray(e["y"])
    correct = np.asarray(e["correct"] if "correct" in e else e["final_correct"])
    pred = e["pred"] if "pred" in e else e["final_pred"]
    correct = correct.astype(bool)
    order = np.argsort(-score) if not ascending else np.argsort(score)
    risks = 1 - correct[order].astype(float)
    n = len(risks)
    cum_risk = np.cumsum(risks) / np.arange(1, n + 1)
    aurc = float(np.trapezoid(cum_risk, np.arange(1, n + 1) / n) / 1.0)
    rows = []
    for cov in C.RISK_COVERAGE_LEVELS:
        k = max(1, int(round(cov * n)))
        sel = order[:k]
        rows.append(dict(coverage=cov, risk=float(1 - correct[sel].mean()),
                         aurc_prefix=float(cum_risk[k - 1])))
    # macro-F1 on the covered subset
    from sklearn.metrics import f1_score
    for cov in C.RISK_COVERAGE_LEVELS:
        k = max(1, int(round(cov * n)))
        sel = order[:k]
        rows_f1 = f1_score(y[sel], np.asarray(pred)[sel], average="macro", zero_division=0,
                           labels=sorted(set(y.tolist())))
        for r in rows:
            if r["coverage"] == cov:
                r["macro_f1"] = float(rows_f1)
    return dict(rows=rows, aurc=aurc, selective_risk_at_80=float(
        [r["risk"] for r in rows if r["coverage"] == 0.8][0]))


def risk_coverage_comparison(ext, split="test"):
    """TURS u vs max-softmax vs entropy selective prediction."""
    e = ext[split]
    scores = {
        "turs_u_mean": e["u_mean"],
        "max_softmax_confidence_neg": -e["confidence"],
        "predictive_entropy": e["entropy"],
    }
    out = {}
    for name, s in scores.items():
        out[name] = risk_coverage(e, score=s)
    return out
