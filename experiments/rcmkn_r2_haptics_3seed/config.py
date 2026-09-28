"""Config for the R2 Haptics 3-seed robustness experiment.

All values predeclared from the validated R2 (rcmkn_haptics_seed42) config.
Nothing here is tuned; the ONLY per-seed variation is the outer random seed
applied to the learned context pipeline (SSL encoder + VQ).
"""
import os

import numpy as np

# config.py -> rcmkn_r2_haptics_3seed -> experiments -> ECG_Benchmark (repo root)
ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
RESULTS_ROOT = os.path.join(ROOT, "results")
OUT_DIR = os.path.join(RESULTS_ROOT, "rcmkn_r2_haptics_3seed")

DATASET = "Haptics"
SEEDS = [42, 43, 44]

# ---- Fixed MiniROCKET (identical for all seeds) ---------------------------
MINIROCKET_SEED = 42          # AUDIT 5: never varied by the outer seed
N_FEATURES = 9996
N_GLOBAL = 4998
N_HET = 4998

# ---- Canonical references (NOT to be silently replaced) --------------------
R2_REF_SEED42 = 0.5500        # validated R2 Haptics seed-42 test Macro-F1
R2_REF_TOL = 0.011            # repo-established seed tolerance (R0 gate scale)
M0_REF = 0.4974               # canonical Haptics MiniROCKET

# ---- R2 context config (must equal rcmkn_haptics_seed42.config) -----------
ENCODER = {
    "channels": (32, 32, 64, 64),
    "kernel_sizes": (3, 5, 7, 9),
    "dilations": (1, 2, 4, 8),
    "d_model": 32,
    "mask_ratio": 0.10,
    "span_len": 16,
    "batch_size": 8,
    "ssl_epochs": 120,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "patience": 20,
}
VQ = {
    "K": 8,
    "beta_vq": 0.25,
    "lam_div": 0.01,
    "ema_decay": 0.99,
    "dead_threshold": 1e-3,
    "revival_patience": 100,
}
JOINT = {
    "epochs": 60,
    "lr": 5e-4,
    "lambda_cls": 0.10,
    "patience": 10,
    "batch_size": 8,
}
MIN_OCCUPANCY = 0.01

# Ridge protocol (identical to validated R2)
ALPHAS = np.logspace(-4, 4, 20)
