"""TURS-GLR local PPV + strength features.

The canonical local activity module (sliding-window local PPV and response
strength) is reused from models/turs_rrmt/model.py::LocalActivity. This file
documents the contract and re-exports it so the GLR package is
self-describing:

    a_m(t) = mean_{tau in W_t} I[r_m(tau) > 0]      (local PPV, window W)
    s_m(t) = mean_{tau in W_t} |r_m(tau)|           (local strength)

Both are [B, M, T'] full temporal resolution. No learned compression occurs
here; P(t) = [A(t) || log1p(s_m(t)/med_train)] feeds the router.
"""

from models.turs_rrmt.model import LocalActivity

__all__ = ["LocalActivity"]
