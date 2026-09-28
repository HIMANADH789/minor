"""RCMKN configuration and version registry (all predeclared, none tuned)."""
import platform
import sys

SEED = 42

# ---- Stage 1: fixed MiniROCKET bank (canonical, unchanged) ----------------
N_FEATURES = 9996
N_GLOBAL = 4998
N_HET = 4998
K_CODES = 8

# ---- Stage 2: small causal SSL encoder ------------------------------------
ENCODER = {
    "channels": (32, 32, 64, 64),
    "kernel_sizes": (3, 5, 7, 9),
    "dilations": (1, 2, 4, 8),
    "d_model": 32,
    "mask_ratio": 0.10,
    "span_len": 16,           # contiguous span length (8-32 recommended range)
    "batch_size": 8,
    "ssl_epochs": 120,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "patience": 20,           # val masked-MSE early stopping
}

# ---- Stage 3: VQ (validated HardVQ machinery, reused) ---------------------
VQ = {
    "K": K_CODES,
    "beta_vq": 0.25,          # commitment coefficient (DRTN_CFG value)
    "lam_div": 0.01,          # population code-usage diversity
    "ema_decay": 0.99,
    "dead_threshold": 1e-3,
    "revival_patience": 100,
}

# ---- Stage 3B/5: joint context+classification fine-tuning -----------------
JOINT = {
    "epochs": 60,
    "lr": 5e-4,
    "lambda_cls": 0.10,       # predeclared aux classification pressure
    "patience": 10,
    "batch_size": 8,
}

# ---- Stage 4: heterogeneity (audited formula, unchanged) ------------------
MIN_OCCUPANCY = 0.01

# ---- Stage 5: Hydra competitive pooling (aeon _HydraInternal) -------------
HYDRA = {
    "k": 8,                   # kernels per group
    "g": 16,                  # groups
    "note": "uses aeon 1.5.0 _HydraInternal directly (the HydraTransformer "
            "core); count_max + count_min per (dilation, diff, group, kernel)",
}

# ---- Stage 6: Ridge -------------------------------------------------------
ALPHAS = None               # set in runner: np.logspace(-4, 4, 20)

# ---- R0 gate --------------------------------------------------------------
R0_REFERENCE = 0.5366       # audited Haptics M1, seed 42
R0_TOLERANCE = 0.011        # repo-established tolerance (e.g. M0 gate 0.0011
                            # scaled by observed seed spread ~0.01)


def versions():
    import aeon
    import numpy
    import sklearn
    import torch
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "aeon": aeon.__version__,
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "sklearn": sklearn.__version__,
    }
