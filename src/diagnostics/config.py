"""Central configuration for the TURS-Lite diagnostic validation study.

Every tunable analysis constant lives here so the pipeline is auditable and
no threshold is tuned after looking at test results. The PRIMARY choices
marked [PRE-REGISTERED] are fixed before any test-set diagnostic evaluation.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------- output layout ----------------
DIAG_ROOT = os.path.join(ROOT, "results", "diagnostics", "turs_lite")
AUDIT_DIR = os.path.join(DIAG_ROOT, "audit")
EXTRACT_DIR = os.path.join(DIAG_ROOT, "extracted")
TABLE_DIR = os.path.join(DIAG_ROOT, "tables")
FIG_DIR = os.path.join(DIAG_ROOT, "figures")
CASE_DIR = os.path.join(DIAG_ROOT, "case_studies")
REPORT_PATH = os.path.join(DIAG_ROOT, "TURS_LITE_DIAGNOSTIC_VALIDATION_REPORT.md")
RUBRIC_PATH = os.path.join(DIAG_ROOT, "evidence_rubric.json")
RUN_META_PATH = os.path.join(DIAG_ROOT, "run_metadata.json")
MASTER_CSV = os.path.join(TABLE_DIR, "diagnostic_master_results.csv")
PIPELINE_LOG = os.path.join(ROOT, "logs", "turs_lite_diag_full.log")
PIPELINE_PID = os.path.join(ROOT, "turs_lite_diag.pid")

CKPT_DIR = os.path.join(ROOT, "checkpoints", "turs_lite")
HIST_RESULT = os.path.join(ROOT, "results", "turs_benchmark")

# ---------------- canonical benchmark protocol (fair_turs.py) ----------------
SEED = 42
VAL_FRAC = 0.15
MAX_EPOCHS = 30
PATIENCE = 8
LR = 3e-4
WD = 1e-2
BATCH_SIZE = 64
TURS_LOSS_LAMBDAS = dict(lambda_smooth=0.01, lambda_vel=0.005,
                         lambda_unc=0.01, lambda_inter=0.005)

DATASETS = [
    ("ECG5000_UNBAL", "data/ecg5000_resplit.npz", 5),
    ("ECG5000_BAL",   "data/ecg5000_fair_balanced.npz", 5),
    ("CWRU_UNBAL",    "data/cwru_unbalanced.npz", 4),
    ("CWRU_BAL",      "data/cwru_balanced.npz", 4),
]
NUM_CLASSES = {t: n for t, _, n in DATASETS}
DATA_FILE = {t: f for t, f, _ in DATASETS}

# ---------------- analysis seeds [FIXED] ----------------
BOOTSTRAP_SEED = 4200
PERMUTATION_SEED = 4300
N_BOOTSTRAP = 2000          # Phase 17 minimum
N_PERMUTATION = 2000

# ---------------- statistical / evaluation settings ----------------
RISK_COVERAGE_LEVELS = [1.00, 0.90, 0.80, 0.70, 0.60, 0.50]
FDR_ALPHA = 0.05

# ---------------- Experiment 2: velocity localization ----------------
SYNTH_N_PER_CLASS = 10      # synthetic perturbed samples per class (test)
SYNTH_REGION_FRAC = 0.15    # injected region width as fraction of T
SYNTH_SEED = 4400
LOC_BASELINE_STD = 25.0     # trivial-baseline noise std (signal units)
VEL_TO_RAW_RATIO = 4        # raw->H4 temporal downsampling factor (stride 2 x4)

# ---------------- Experiment 4: controlled degradation ----------------
DEGRADATION_SPECS = {
    "gaussian_noise":  {"levels": [0.05, 0.10, 0.20, 0.40, 0.80]},
    "amplitude_scale": {"levels": [0.6, 0.7, 0.8, 0.9, 1.1]},   # away from 1.0
    "baseline_shift":  {"levels": [0.2, 0.4, 0.6, 0.8, 1.2]},   # x signal std
    "temporal_jitter": {"levels": [2, 4, 8, 16, 32]},           # sample shift
    "localized_mask":  {"levels": [0.05, 0.10, 0.20, 0.30, 0.40]},  # width frac
}

# ---------------- Experiment 7: benign transformations ----------------
BENIGN_SPECS = {
    "amp_scale_0.95":  {"kind": "amplitude_scale", "level": 0.95},
    "amp_scale_1.05":  {"kind": "amplitude_scale", "level": 1.05},
    "baseline_0.1sd":  {"kind": "baseline_shift",  "level": 0.1},
    "noise_0.02":      {"kind": "gaussian_noise",  "level": 0.02},
    "jitter_2":        {"kind": "temporal_jitter", "level": 2},
}

# ---------------- Experiment 6: faithfulness ----------------
FAITH_REGION_FRACS = [0.05, 0.10, 0.20]
FAITH_PRIMARY_FRAC = 0.10   # [PRE-REGISTERED] primary region size
FAITH_PERTURB = "noise"     # noise replacement (local stats preserved)
FAITH_N_RANDOM = 3          # random matched regions per sample

# ---------------- primary aggregations [PRE-REGISTERED] ----------------
PRIMARY = {
    "uncertainty_agg": "mean",       # mean(u) is the primary scalar
    "velocity_agg": "max",           # max_t ||v_t|| for event detection
    "z_pooled": "mean",
}

# ---------------- case studies ----------------
CASE_STUDY_N = 2            # cases per category
CASE_SEED = 4500
