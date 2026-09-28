"""Central configuration for the TURS-Stack diagnostic validation study.

Mirrors src/diagnostics/config.py conventions; every analysis constant lives
here. PRIMARY [PRE-REGISTERED] choices are fixed before test-set evaluation.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

# ---------------- output layout ----------------
DIAG_ROOT = os.path.join(ROOT, "results", "diagnostics", "turs_stack")
AUDIT_DIR = os.path.join(DIAG_ROOT, "audit")
EXTRACT_DIR = os.path.join(DIAG_ROOT, "extracted")
REPLAY_DIR = os.path.join(DIAG_ROOT, "replay")
TABLE_DIR = os.path.join(DIAG_ROOT, "tables")
FIG_DIR = os.path.join(DIAG_ROOT, "figures")
CASE_DIR = os.path.join(DIAG_ROOT, "case_studies")
FAIL_DIR = os.path.join(DIAG_ROOT, "failure_analysis")
REPORT_PATH = os.path.join(DIAG_ROOT, "TURS_STACK_DIAGNOSTIC_VALIDATION_REPORT.md")
RUBRIC_PATH = os.path.join(DIAG_ROOT, "evidence_rubric.json")
RUN_META_PATH = os.path.join(DIAG_ROOT, "run_metadata.json")
MASTER_CSV = os.path.join(TABLE_DIR, "diagnostic_master_results.csv")
PIPELINE_LOG = os.path.join(ROOT, "logs", "turs_stack_diag_full.log")
PIPELINE_PID = os.path.join(ROOT, "turs_stack_diag.pid")

CKPT_DIR = os.path.join(ROOT, "checkpoints", "turs_stack")
STACK_RESULTS = os.path.join(ROOT, "results", "turs_stack")
LITE_DIAG_ROOT = os.path.join(ROOT, "results", "diagnostics", "turs_lite")

# ---------------- canonical benchmark protocol (run_turs_stack_benchmark.py) ----------------
SEED = 42
VAL_FRAC = 0.15
NUM_CLASSES = {"ECG5000_UNBAL": 5, "ECG5000_BAL": 5, "CWRU_UNBAL": 4, "CWRU_BAL": 4}
DATA_FILE = {
    "ECG5000_UNBAL": "data/ecg5000_resplit.npz",
    "ECG5000_BAL": "data/ecg5000_fair_balanced.npz",
    "CWRU_UNBAL": "data/cwru_unbalanced.npz",
    "CWRU_BAL": "data/cwru_balanced.npz",
}
DATASETS = list(NUM_CLASSES)
BRANCH_NAMES = ["lite", "rv", "cs", "cmr"]

# ---------------- analysis seeds [FIXED] ----------------
BOOTSTRAP_SEED = 5200
PERMUTATION_SEED = 5300
N_BOOTSTRAP = 2000
N_PERMUTATION = 2000
FDR_ALPHA = 0.05

# ---------------- Experiment: velocity localization ----------------
SYNTH_N_PER_CLASS = 10
SYNTH_REGION_FRAC = 0.15
SYNTH_SEED = 5400

# ---------------- degradation ----------------
DEGRADATION_SPECS = {
    "gaussian_noise":  {"levels": [0.05, 0.10, 0.20, 0.40, 0.80]},
    "amplitude_scale": {"levels": [0.6, 0.7, 0.8, 0.9, 1.1]},
    "baseline_shift":  {"levels": [0.2, 0.4, 0.6, 0.8, 1.2]},
    "temporal_jitter": {"levels": [2, 4, 8, 16, 32]},
    "localized_mask":  {"levels": [0.05, 0.10, 0.20, 0.30, 0.40]},
}
DEG_MAX_SAMPLES = 400

# ---------------- faithfulness ----------------
FAITH_REGION_FRACS = [0.05, 0.10, 0.20]
FAITH_PRIMARY_FRAC = 0.10     # [PRE-REGISTERED]
FAITH_PERTURB = "noise"
FAITH_N_RANDOM = 3
FAITH_MAX_SAMPLES = 300

# ---------------- primary aggregations [PRE-REGISTERED] ----------------
PRIMARY = {
    "uncertainty_agg": "mean",      # mean_t u_t per branch
    "velocity_agg": "max",          # max_t ||v_t|| for event detection
    "z_pooled": "mean",
    "ensemble_signal": "js_disagreement",  # primary ensemble disagreement
}

# ---------------- benign transformations ----------------
BENIGN_SPECS = {
    "amp_scale_0.95": {"kind": "amplitude_scale", "level": 0.95},
    "amp_scale_1.05": {"kind": "amplitude_scale", "level": 1.05},
    "baseline_0.1sd": {"kind": "baseline_shift", "level": 0.1},
    "noise_0.02":     {"kind": "gaussian_noise", "level": 0.02},
    "jitter_2":       {"kind": "temporal_jitter", "level": 2},
}

RISK_COVERAGE_LEVELS = [1.00, 0.90, 0.80, 0.70, 0.60, 0.50]
CASE_STUDY_N = 2
CASE_SEED = 5500
