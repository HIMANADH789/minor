"""TURS-Stack FULL STUDY (fresh canonical benchmark + frozen diagnostics).

Layout:
  results/turs_stack_final/
    audit/                     implementation/protocol/dataset/branch/artifact audits
    benchmark/                 fresh training: checkpoints, history, metrics
    replay/                    replay verification of the fresh run
    extracted/                 per-sample + internal-variable caches
    diagnostics/               diagnostic experiment JSONs, tables, figures
    tables/                    machine-readable CSVs
    TURS_STACK_FULL_BENCHMARK_AND_DIAGNOSTIC_REPORT.md

Reuses the verified Stack machinery:
  src/diagnostics/turs_stack/*        (extraction, experiments, stats, plotting)
  src/diagnostics/*                   (perturb, calibration, statistics)
The fresh benchmark lives in checkpoints/turs_stack_final/ so the OLD
unsatisfactory artifacts are never silently reused.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

# ---------------- layout ----------------
STUDY_ROOT = os.path.join(ROOT, "results", "turs_stack_final")
AUDIT_DIR = os.path.join(STUDY_ROOT, "audit")
BENCH_DIR = os.path.join(STUDY_ROOT, "benchmark")
REPLAY_DIR = os.path.join(STUDY_ROOT, "replay")
EXTRACT_DIR = os.path.join(STUDY_ROOT, "extracted")
DIAG_DIR = os.path.join(STUDY_ROOT, "diagnostics")
TABLE_DIR = os.path.join(STUDY_ROOT, "tables")
FIG_DIR = os.path.join(STUDY_ROOT, "figures")
CASE_DIR = os.path.join(STUDY_ROOT, "case_studies")
REPORT_PATH = os.path.join(STUDY_ROOT, "TURS_STACK_FULL_BENCHMARK_AND_DIAGNOSTIC_REPORT.md")
RUBRIC_PATH = os.path.join(DIAG_DIR, "evidence_rubric.json")
RUN_META_PATH = os.path.join(STUDY_ROOT, "run_metadata.json")
MASTER_CSV = os.path.join(TABLE_DIR, "diagnostic_master_results.csv")
PIPELINE_LOG = os.path.join(ROOT, "logs", "turs_stack_full_study.log")
PIPELINE_PID = os.path.join(ROOT, "turs_stack_full_study.pid")

CKPT_DIR = os.path.join(ROOT, "checkpoints", "turs_stack_final")
LITE_CKPT_DIR = os.path.join(ROOT, "checkpoints", "turs_lite")
LITE_DIAG = os.path.join(ROOT, "results", "diagnostics", "turs_lite")
OLD_STACK_RESULTS = os.path.join(ROOT, "results", "turs_stack")

# ---------------- canonical benchmark protocol ----------------
# (run_turs_stack_benchmark.py: seed 42, AdamW 3e-4/1e-2, OneCycleLR,
#  30 ep max, patience 8, batch 64, CE mean over branches, sigma0 warm start,
#  Phase-0 calibration epochs, combiners fit on VAL only)
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

# ---------------- analysis seeds ----------------
BOOTSTRAP_SEED = 6200
PERMUTATION_SEED = 6300
SYNTH_SEED = 6400
CASE_SEED = 6500

# ---------------- diagnostics settings (same as verified prior study) ----------------
SYNTH_N_PER_CLASS = 10
SYNTH_REGION_FRAC = 0.15
DEG_MAX_SAMPLES = 400
FAITH_MAX_SAMPLES = 300
FAITH_PRIMARY_FRAC = 0.10     # [PRE-REGISTERED] (validation-selected in Lite study)
FAITH_PERTURB = "noise"
FAITH_N_RANDOM = 3
RISK_COVERAGE_LEVELS = [1.00, 0.90, 0.80, 0.70, 0.60, 0.50]
PRIMARY = {
    "uncertainty_agg": "mean",
    "velocity_agg": "max",
    "ensemble_signal": "js_disagreement",
}
BENIGN_SPECS = {
    "amp_scale_0.95": {"kind": "amplitude_scale", "level": 0.95},
    "amp_scale_1.05": {"kind": "amplitude_scale", "level": 1.05},
    "baseline_0.1sd": {"kind": "baseline_shift", "level": 0.1},
    "noise_0.02":     {"kind": "gaussian_noise", "level": 0.02},
    "jitter_2":       {"kind": "temporal_jitter", "level": 2},
}
CASE_STUDY_N = 2
