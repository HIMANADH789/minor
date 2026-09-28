"""TURS-MGB model API (Phase 33).

A thin orchestration layer over the extractor + scaler + dual ridge:
    fit()      : fit transport refs (TRAIN), extract features, fit scaler
                 (TRAIN), select lambda (VAL), fit dual ridge (TRAIN)
    predict()  : probabilities / predictions from raw X
    state()    : full serializable state

extraction_mode:
    "summary"  -> cached per-split feature matrices (default)
    "temporal" -> additionally returns per-geometry temporal transport
                  fields T_j(t) for diagnostics (Phase 19; memory-gated)
"""

import numpy as np
import torch

from models.turs_mgb.feature_extractor import MGBFeatureExtractor
from models.turs_mgb.grouped_scaler import GroupStandardizer
from models.turs_mgb.ridge_readout import DualRidge


class TURSMGB:
    def __init__(self, n_classes, M=4096, seed=42, stats=("ppv", "max"),
                 lag_set=(1, 2, 4, 8), device=None, feature_keys=None):
        self.n_classes = int(n_classes)
        self.feature_keys = feature_keys  # e.g. ("G1_standard", "G2_tail", ...)
        self.device = device or torch.device("cpu")
        self.ex = MGBFeatureExtractor(M=M, seed=seed, stats=stats,
                                      lag_set=lag_set, device=self.device)

    # ------------------------------------------------------------- fit
    def fit_features(self, X_train, X_val, X_test, window=None,
                     chunk=256, batch=128):
        """Extract features for all splits with TRAIN-fitted references.

        Returns dict of split -> (feats dict). The raw block is always
        extracted (A0 baseline uses the same code path).
        """
        self.ex.fit_transport_references(X_train)
        out = {}
        for split, X in (("train", X_train), ("val", X_val), ("test", X_test)):
            out[split] = self.ex.extract_features(X, window=window,
                                                  chunk=chunk, batch=batch)
        self.spec = self.ex.spec()
        return out

    # ------------------------------------------------------- readout
    def fit_readout(self, feats, y_train, y_val, lam_grid, scaler_blocks=None):
        """Group-standardize (TRAIN stats), select lambda on VAL, fit dual
        ridge on TRAIN with the frozen lambda. Returns selection record."""
        keys = self.feature_keys or [k for k in self.ex.geometry_names]
        self.keys_ = keys

        # group standardization per geometry block (Phase 6; raw block is
        # standardized as its own group when included)
        all_keys = ["raw"] + list(self.ex.geometry_names)
        blocks_tr = [feats["train"][k] for k in all_keys]
        self.scaler_all = GroupStandardizer([f.shape[1] for f in blocks_tr]).fit(blocks_tr)
        Ztr_blocks = self.scaler_all.transform(blocks_tr)

        # assemble the model matrix for the selected keys
        def assemble(blocks_by_key):
            return np.concatenate([blocks_by_key[k] for k in keys], axis=1)

        blocks_by_key_tr = dict(zip(all_keys, Ztr_blocks))
        Ztr = assemble(blocks_by_key_tr)
        self.block_dims_ = [feats["train"][k].shape[1] for k in keys]
        self.offsets_ = np.concatenate([[0], np.cumsum(self.block_dims_)])

        # standardize val/test with TRAIN stats
        Zva = self.transform_split(feats["val"])
        Zte = self.transform_split(feats["test"])

        # lambda selection on validation (Phase 8)
        from experiments.turs_rrmt.train_eval import _mf1
        best = dict(lam=None, val_mf1=-1, val_nll=float("inf"))
        from src.diagnostics.calibration import nll as nll_fn
        for lam in lam_grid:
            rd = DualRidge(self.n_classes, lam=lam, device=self.device)
            rd.fit(Ztr, y_train)
            pv = rd.predict_proba(Zva)
            m = _mf1(y_val, pv.argmax(1), self.n_classes)
            nl = float(nll_fn(pv, y_val))
            # primary: val MF1; tie-break: val NLL
            if (m > best["val_mf1"]) or (m == best["val_mf1"] and nl < best["val_nll"]):
                best = dict(lam=lam, val_mf1=m, val_nll=nl)
        self.lam_ = best["lam"]

        # final fit with frozen lambda
        self.readout = DualRidge(self.n_classes, lam=self.lam_,
                                 device=self.device)
        self.readout.fit(Ztr, y_train)
        self.selection_ = best
        return best

    def transform_split(self, feats_split):
        """Standardize a split's blocks (TRAIN-fitted scaler) + assemble."""
        all_keys = ["raw"] + list(self.ex.geometry_names)
        blocks = [self.scaler_all.transform_block(j, feats_split[k])
                  for j, k in enumerate(all_keys)]
        by_key = dict(zip(all_keys, blocks))
        return np.concatenate([by_key[k] for k in self.keys_], axis=1)

    # ------------------------------------------------------- inference
    def predict_split(self, feats_split):
        Z = self.transform_split(feats_split)
        return self.readout.predict_proba(Z)

    def group_logits(self, feats_split):
        Z = self.transform_split(feats_split)
        keys = self.keys_
        dims = [feats_split[k].shape[1] for k in keys]
        offs = np.concatenate([[0], np.cumsum(dims)])
        return self.readout.group_logits(Z, offs)

    # ----------------------------------------------------------- spec
    def spec(self):
        st = dict(self.ex.spec())
        st.update(selected_keys=list(self.keys_), lam=self.lam_,
                  block_dims=[int(d) for d in self.block_dims_],
                  feature_dim_model=int(self.offsets_[-1]),
                  selection={k: (float(v) if isinstance(v, (int, float))
                                 else v) for k, v in self.selection_.items()})
        return st
