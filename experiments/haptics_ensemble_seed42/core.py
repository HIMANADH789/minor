"""
Haptics Fixed-Feature + Learned-Regime Ensemble — Seed 42
=========================================================
Leakage-safe systems experiment: does the best existing DRTN learned-regime
model make errors complementary enough to canonical MiniROCKET that
validation-selected fusion/stacking beats MiniROCKET alone?

Leakage protocol (spec sec. 3/8/9):
  Stage A (selection): MiniROCKET extractor+Ridge fit on TRAIN only;
    DRTN official checkpoint (frozen long before this experiment) produces
    train/val logits. Fusion alpha and stacker C are selected on VAL only.
  Stage B (final): MiniROCKET Ridge refit on TRAIN+VAL (canonical protocol);
    DRTN checkpoint is unchanged (it never saw val gradient updates for
    selection beyond its own frozen early stopping). TEST evaluated exactly
    once per frozen system.

Hydra / MultiRocketHydra use aeon 1.5.0's canonical implementations with
random_state=42, following the same two-stage protocol.
"""
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

SEED = 42
ALPHAS = [round(a, 2) for a in np.arange(0.0, 1.0001, 0.05)]
C_GRID = [0.01, 0.1, 1.0, 10.0]


# ---------------------------------------------------------------------------
# Canonical MiniROCKET, two-stage leakage-safe protocol
# ---------------------------------------------------------------------------
class MiniRocketArm:
    """Canonical aeon MiniRocket (~10K kernels) + RidgeClassifierCV.

    Stage A: extractor fit on TRAIN only, Ridge fit on TRAIN only ->
             decision scores for train (OOF)/val.
    Stage B: extractor unchanged; Ridge refit on TRAIN+VAL for final TEST.
    """

    def __init__(self, n_jobs=-1):
        from aeon.transformations.collection.convolution_based import MiniRocket
        self.extractor = MiniRocket(random_state=SEED, n_jobs=n_jobs)
        self.alphas = np.logspace(-4, 4, 20)
        self.ridge = None
        self.classes = None

    @staticmethod
    def _3d(X):
        return X[:, None, :].astype(np.float32)

    def fit_extractor_train(self, Xtr):
        t0 = time.time()
        self.extractor.fit(self._3d(Xtr))
        return time.time() - t0

    def transform(self, X):
        return self.extractor.transform(self._3d(X))

    def fit_ridge(self, Ftr, ytr):
        self.classes = np.unique(ytr)
        self.ridge = RidgeClassifierCV(alphas=self.alphas)
        self.ridge.fit(Ftr, ytr)

    def decision_scores(self, F):
        """Canonical multiclass Ridge decision scores, aligned to self.classes."""
        d = self.ridge.decision_function(F)
        if d.ndim == 1:                       # binary edge case
            d = np.column_stack([-d, d])
        return d

    def predict(self, F):
        return self.ridge.predict(F)


# ---------------------------------------------------------------------------
# OOF decision scores for the stacker's TRAIN rows (leakage-safe)
# ---------------------------------------------------------------------------
def minirocket_oof_scores(Xtr, ytr, n_folds=5):
    """Per-fold: fit extractor+Ridge on K-1 folds, score the held-out fold.

    Every training sample receives an out-of-fold decision score produced by
    a model that never saw that sample -> no in-sample leakage into the
    stacker's training rows.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof = None
    order = []
    for tr_idx, te_idx in skf.split(Xtr, ytr):
        arm = MiniRocketArm(n_jobs=-1)
        arm.fit_extractor_train(Xtr[tr_idx])
        Ftr, Fte = arm.transform(Xtr[tr_idx]), arm.transform(Xtr[te_idx])
        arm.fit_ridge(Ftr, ytr[tr_idx])
        d = arm.decision_scores(Fte)
        if oof is None:
            oof = np.zeros((len(Xtr), d.shape[1]))
        oof[te_idx] = d
        order.append(te_idx)
    covered = np.sort(np.concatenate(order))
    assert np.array_equal(covered, np.arange(len(Xtr))), \
        "OOF folds do not cover every training sample exactly once"
    return oof


def drtn_oof_logits(model, Xtr, ytr, n_folds=5, device="cpu"):
    """OOF logits from the frozen DRTN checkpoint.

    The checkpoint weights are FROZEN (no fitting inside folds), so scoring
    any training sample is leakage-free by construction; this helper exists
    to keep the OOF machinery symmetric and auditable.
    """
    from torch.utils.data import DataLoader, TensorDataset
    zn = lambda X: ((X - X.mean(-1, keepdims=True)) /
                    (X.std(-1, keepdims=True) + 1e-8)).astype(np.float32)
    dl = DataLoader(TensorDataset(torch.from_numpy(zn(Xtr))[:, None, :],
                                  torch.from_numpy(ytr)), batch_size=64)
    logits = []
    model.eval()
    with torch.no_grad():
        for xb, _ in dl:
            lg, _ = model.forward_with_assign(xb.to(device))
            logits.append(lg.cpu().numpy())
    oof = np.concatenate(logits, 0)
    assert oof.shape[0] == len(Xtr)
    return oof


# ---------------------------------------------------------------------------
# DRTN checkpoint loading (frozen official artifact)
# ---------------------------------------------------------------------------
def load_official_drtn(device):
    """Load the official seed-42 R5 checkpoint (val-selected, frozen)."""
    from models.drtn.model import build_model
    rdir = os.path.join(ROOT, "results", "drtn_haptics_seed42", "R5")
    with open(os.path.join(rdir, "result.json")) as f:
        official = json.load(f)
    ck = torch.load(os.path.join(rdir, "checkpoint.pt"),
                    map_location=device, weights_only=False)
    cfg = ck["config"]
    model = build_model("R5", c_in=1, n_classes=5, d_model=cfg["d_model"],
                        n_codes=cfg["n_codes"], tau=cfg["tau"],
                        ema_decay=cfg["ema_decay"], beta=cfg["beta_commit"],
                        lam_div=cfg["lam_div"],
                        dead_threshold=cfg["dead_threshold"],
                        revival_patience=cfg["revival_patience"],
                        traj_layers=cfg["trajectory"]["layers"],
                        traj_heads=cfg["trajectory"]["heads"],
                        traj_ffn=cfg["trajectory"]["ffn"],
                        traj_dropout=cfg["trajectory"]["dropout"])
    model.load_state_dict(ck["model_state"])
    model.to(device).eval()
    return model, official


def drtn_logits(model, X, device):
    """Raw classifier logits for X, sample-index aligned."""
    from torch.utils.data import DataLoader, TensorDataset
    zn = lambda X: ((X - X.mean(-1, keepdims=True)) /
                    (X.std(-1, keepdims=True) + 1e-8)).astype(np.float32)
    dl = DataLoader(TensorDataset(torch.from_numpy(zn(X))[:, None, :]),
                    batch_size=64)
    out = []
    with torch.no_grad():
        for (xb,) in dl:
            lg, _ = model.forward_with_assign(xb.to(device))
            out.append(lg.cpu().numpy())
    return np.concatenate(out, 0)


def softmax_stable(S, temp=1.0):
    """Numerically stable softmax over class axis."""
    z = S / max(temp, 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def macro_f1(y, p):
    return float(f1_score(y, p, average="macro", zero_division=0))


# ---------------------------------------------------------------------------
# Hydra / MultiRocketHydra arms (aeon canonical)
# ---------------------------------------------------------------------------
class HydraArm:
    """aeon HydraClassifier, random_state=42, canonical defaults."""

    def __init__(self):
        from aeon.classification.convolution_based import HydraClassifier
        self.clf = HydraClassifier(random_state=SEED, n_jobs=-1)

    def fit(self, Xtr, ytr):
        self.clf.fit(Xtr[:, None, :].astype(np.float32), ytr)

    def predict(self, X):
        return self.clf.predict(X[:, None, :].astype(np.float32))


class MultiRocketHydraArm(HydraArm):
    def __init__(self):
        from aeon.classification.convolution_based import MultiRocketHydraClassifier
        self.clf = MultiRocketHydraClassifier(random_state=SEED, n_jobs=-1)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def correctness_table(y, mr_pred, dr_pred):
    A = int(np.sum((mr_pred == y) & (dr_pred == y)))
    B = int(np.sum((mr_pred == y) & (dr_pred != y)))
    C = int(np.sum((mr_pred != y) & (dr_pred == y)))
    D = int(np.sum((mr_pred != y) & (dr_pred != y)))
    n = len(y)
    return {"A_mr_and_drtn_correct": A, "B_mr_only_correct": B,
            "C_drtn_only_correct": C, "D_both_wrong": D, "n": n,
            "percentages": {"A": round(100 * A / n, 2),
                            "B": round(100 * B / n, 2),
                            "C": round(100 * C / n, 2),
                            "D": round(100 * D / n, 2)}}


def error_correlation(y, mr_pred, dr_pred):
    mr_e = (mr_pred != y).astype(int)
    dr_e = (dr_pred != y).astype(int)
    corr = float(np.corrcoef(mr_e, dr_e)[0, 1]) if mr_e.std() > 0 and dr_e.std() > 0 else 0.0
    # Cohen's kappa between correctness INDICATORS (1=correct)
    mr_c, dr_c = 1 - mr_e, 1 - dr_e
    po = float(np.mean(mr_c == dr_c))
    pe = float(np.mean(mr_c) * np.mean(dr_c) +
               (1 - np.mean(mr_c)) * (1 - np.mean(dr_c)))
    kappa = (po - pe) / (1 - pe) if pe < 1 else 0.0
    both_wrong = int(np.sum(mr_e & dr_e))
    union_wrong = int(np.sum(mr_e | dr_e))
    per_class = {}
    for c in np.unique(y):
        m = y == c
        per_class[int(c)] = {
            "n": int(m.sum()),
            "both_wrong": int(np.sum(mr_e[m] & dr_e[m])),
            "mr_only_wrong": int(np.sum(mr_e[m] & ~dr_e[m])),
            "drtn_only_wrong": int(np.sum(~mr_e[m] & dr_e[m])),
        }
    return {
        "binary_error_correlation": round(corr, 4),
        "cohens_kappa_correctness": round(float(kappa), 4),
        "prediction_agreement_rate": round(float(np.mean(mr_pred == dr_pred)), 4),
        "error_overlap": round(both_wrong / union_wrong, 4) if union_wrong else 0.0,
        "both_wrong": both_wrong,
        "union_wrong": union_wrong,
        "per_class": per_class,
    }
