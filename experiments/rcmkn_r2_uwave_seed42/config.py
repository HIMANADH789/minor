"""R2 + MiniROCKET on the local UWave gesture datasets (seed 42).

Single-seed Motion-domain generalization experiment. Reuses, unchanged, the
audited/validated project components:

    * R2 context model / SSL schedule / regime extraction:
      experiments/rcmkn_haptics_seed42 (config, model, vq, runner)
    * raw activations / canonical PPV / independent H recompute:
      experiments/drtn_conditioned_minirocket_transfer_seed42.core
    * audited heterogeneity H + M2/M3-style controls:
      experiments/drtn_conditioned_minirocket_haptics_3seed.runner
    * canonical per-sample z-normalization + stratified 15%-of-train val
      split policy (seed 42): the repository's established protocol

Datasets: UWaveGestureLibrary{All,X,Y,Z}, local .ts copies (official UCR
TRAIN/TEST split; official TEST is never touched until the single official
evaluation per model).
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
RESULTS_ROOT = os.path.join(ROOT, "results")
OUT_DIR = os.path.join(RESULTS_ROOT, "r2_uwave_seed42")

DATASETS = ["UWaveGestureLibraryAll", "UWaveGestureLibraryX",
            "UWaveGestureLibraryY", "UWaveGestureLibraryZ"]

SEED = 42
VAL_FRAC = 0.15

# Canonical UCR specifications (verified against the actual files at load time)
CANONICAL = {
    "UWaveGestureLibraryAll": {"train": 896, "test": 3582, "T": 945,
                               "n_classes": 8},
    "UWaveGestureLibraryX": {"train": 896, "test": 3582, "T": 315,
                             "n_classes": 8},
    "UWaveGestureLibraryY": {"train": 896, "test": 3582, "T": 315,
                             "n_classes": 8},
    "UWaveGestureLibraryZ": {"train": 896, "test": 3582, "T": 315,
                             "n_classes": 8},
}

# R2 config: identical to the validated Haptics R2 (rcmkn_haptics_seed42)
N_FEATURES = 9996
MINIROCKET_SEED = 42
M0_TOL = 0.0011                 # canonical M0 reproduction guard
R2_HAPTICS_REF = 0.5500         # context for the report

# ~10K-kernel MiniRocket is used for T=945/315 per the project's canonical
# configuration; aeon yields 9996 PPV features for univariate input.
