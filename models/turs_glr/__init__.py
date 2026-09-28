"""TURS-GLR — Gated Global-Local Regime-Routed Ridge.

One architecture for all datasets:
  - global stream: large fixed kernel bank, MiniRocket-style PPV features
  - local stream: small fixed bank -> local PPV/strength -> soft router
    -> 4 transport flavors -> rich routed statistics
  - single closed-form BLOCK-REGULARIZED ridge readout (lambda_global,
    lambda_local selected on validation) is the ONLY global/local gate.
"""

from models.turs_glr.model import TURSGLR, count_params
from models.turs_glr.block_ridge import BlockRidge, fit_block_ridge_cv

__all__ = ["TURSGLR", "count_params", "BlockRidge", "fit_block_ridge_cv"]
