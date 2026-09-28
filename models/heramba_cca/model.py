"""HERAMBA-CCA: frozen MiniROCKET G + CCA-residualized HERAMBA-unique H.

The model has NO mixing ratio:
    representation = [G || H_unique]
where H_unique = H - (component of H linearly explained by the shared
G-H canonical subspace), with every fit performed on TRAIN only and
frozen before val/test are transformed.

Classifier: the canonical project classifier
    RidgeClassifierCV(alphas=logspace(-4, 4, 20))
fitted on train+validation for the final single test evaluation
(exactly the R2/R5 final-fit convention).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import f1_score

from models.heramba_cca.cca import HerambaCCAProjection

ALPHAS = np.logspace(-4, 4, 20)
TAU_SHARED = 0.5          # predeclared shared-subspace threshold
PCA_ENERGY = 1 - 1e-10    # predeclared numerical-rank reduction energy


class HerambaCCAModel:
    """End-to-end HERAMBA-CCA (fit once on train, then frozen)."""

    def __init__(self, tau_shared: float = TAU_SHARED,
                 pca_energy: float = PCA_ENERGY):
        self.tau_shared = tau_shared
        self.pca_energy = pca_energy

    def fit_representation(self, G_train, H_train):
        self.proj = HerambaCCAProjection(
            tau_shared=self.tau_shared,
            pca_energy=self.pca_energy).fit(G_train, H_train)
        self.summary_ = self.proj.summary()
        return self

    def transform(self, G, H):
        """Frozen representation transform -> [G || H_unique]."""
        Hu = self.proj.transform_H_unique(H)
        X = np.hstack([np.asarray(G, dtype=np.float64), Hu])
        assert np.isfinite(X).all(), "non-finite values in representation"
        return X

    def fit_classifier(self, X_dev, y_dev):
        self.clf_ = RidgeClassifierCV(alphas=ALPHAS)
        self.clf_.fit(X_dev, y_dev)
        self.alpha_ = float(self.clf_.alpha_)
        return self

    def predict(self, X):
        return self.clf_.predict(X)

    @staticmethod
    def macro_f1(y_true, y_pred):
        return round(float(f1_score(y_true, y_pred, average="macro",
                                    zero_division=0)), 4)
