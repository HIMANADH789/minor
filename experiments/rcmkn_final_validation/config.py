"""Final-validation phase configuration.

Paths to every canonical artifact used for the paper tables. Nothing is
retrained: frozen seed-42 checkpoints + stored results are the sources of
truth; the only new runs are explicitly-listed gaps.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
REPO_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))     # .../ECG_Benchmark
RESULTS_ROOT = os.path.join(ROOT, "results")
OUT_DIR = os.path.join(RESULTS_ROOT, "final_validation")
SEED = 42
SPLIT_DIR = os.path.join(RESULTS_ROOT, "r2_uwave_seed42")

# dataset -> directory holding context_model_seed42.pt (or .../checkpoints/)
CKPT_PATHS = {
    "Haptics": os.path.join(REPO_ROOT, "results", "rcmkn_haptics_seed42"),
    "Phoneme": os.path.join(REPO_ROOT, "results",
                            "rcmkn_ssl_context_transfer_seed42", "Phoneme"),
    "ECG5000_UNBAL": os.path.join(REPO_ROOT, "results",
                                  "rcmkn_ssl_context_transfer_seed42",
                                  "ECG5000_UNBAL"),
    "CWRU_UNBAL": os.path.join(REPO_ROOT, "results",
                               "rcmkn_ssl_context_transfer_seed42",
                               "CWRU_UNBAL"),
    "ECG5000_BAL": os.path.join(REPO_ROOT, "results",
                                "rcmkn_ssl_context_important2_seed42",
                                "ECG5000_BAL"),
    "CWRU_BAL": os.path.join(REPO_ROOT, "results",
                             "rcmkn_ssl_context_important2_seed42",
                             "CWRU_BAL"),
    "EpilepticSeizures": os.path.join(REPO_ROOT, "results",
                                      "rcmkn_ssl_context_transfer_seed42",
                                      "EpilepticSeizures"),
    "GunPoint": os.path.join(REPO_ROOT, "results",
                             "rcmkn_r2_gunpoint_seed42"),
    "ItalyPowerDemand": os.path.join(REPO_ROOT, "results",
                                     "rcmkn_r2_kaggle_context2_seed42",
                                     "ItalyPowerDemand"),
    "FordA": os.path.join(REPO_ROOT, "results",
                          "rcmkn_r2_kaggle_context2_seed42", "FordA"),
    "UWaveGestureLibraryAll": os.path.join(RESULTS_ROOT, "r2_uwave_seed42",
                                           "UWaveGestureLibraryAll"),
    "UWaveGestureLibraryX": os.path.join(RESULTS_ROOT, "r2_uwave_seed42",
                                         "UWaveGestureLibraryX"),
    "UWaveGestureLibraryY": os.path.join(RESULTS_ROOT, "r2_uwave_seed42",
                                         "UWaveGestureLibraryY"),
    "UWaveGestureLibraryZ": os.path.join(RESULTS_ROOT, "r2_uwave_seed42",
                                         "UWaveGestureLibraryZ"),
}

# canonical R2 results (per dataset) -> (path, results-dict-key for R2)
R2_RESULT_FILES = {
    "Haptics": (os.path.join(REPO_ROOT, "results", "rcmkn_haptics_seed42",
                             "report.json"), "R2"),
    "Phoneme": (os.path.join(REPO_ROOT, "results",
                             "rcmkn_ssl_context_transfer_seed42",
                             "per_dataset_results.json"), "Phoneme"),
    "ECG5000_UNBAL": (os.path.join(REPO_ROOT, "results",
                                   "rcmkn_ssl_context_transfer_seed42",
                                   "per_dataset_results.json"),
                      "ECG5000_UNBAL"),
    "CWRU_UNBAL": (os.path.join(REPO_ROOT, "results",
                                "rcmkn_ssl_context_transfer_seed42",
                                "per_dataset_results.json"), "CWRU_UNBAL"),
    "ECG5000_BAL": (os.path.join(REPO_ROOT, "results",
                                 "rcmkn_ssl_context_important2_seed42",
                                 "ECG5000_BAL", "result.json"), None),
    "CWRU_BAL": (os.path.join(REPO_ROOT, "results",
                              "rcmkn_ssl_context_important2_seed42",
                              "CWRU_BAL", "result.json"), None),
    "EpilepticSeizures": (os.path.join(REPO_ROOT, "results",
                                       "rcmkn_ssl_context_transfer_seed42",
                                       "per_dataset_results.json"),
                          "EpilepticSeizures"),
    "GunPoint": (os.path.join(REPO_ROOT, "results",
                              "rcmkn_r2_gunpoint_seed42", "result.json"),
                 None),
    "ItalyPowerDemand": (os.path.join(REPO_ROOT, "results",
                                      "rcmkn_r2_kaggle_context2_seed42",
                                      "per_dataset_results.json"),
                         "ItalyPowerDemand"),
    "FordA": (os.path.join(REPO_ROOT, "results",
                           "rcmkn_r2_kaggle_context2_seed42",
                           "per_dataset_results.json"), "FordA"),
    "UWaveGestureLibraryAll": (os.path.join(RESULTS_ROOT, "r2_uwave_seed42",
                                            "UWaveGestureLibraryAll",
                                            "result.json"), None),
    "UWaveGestureLibraryX": (os.path.join(RESULTS_ROOT, "r2_uwave_seed42",
                                          "UWaveGestureLibraryX",
                                          "result.json"), None),
    "UWaveGestureLibraryY": (os.path.join(RESULTS_ROOT, "r2_uwave_seed42",
                                          "UWaveGestureLibraryY",
                                          "result.json"), None),
    "UWaveGestureLibraryZ": (os.path.join(RESULTS_ROOT, "r2_uwave_seed42",
                                          "UWaveGestureLibraryZ",
                                          "result.json"), None),
}

# canonical M0 sources (verified MiniROCKET baseline results)
M0_SOURCES = {
    "Haptics": ("external_stack", "MiniROCKET"),
    "Phoneme": ("external_stack", "MiniROCKET"),
    "EpilepticSeizures": ("external_stack", "MiniROCKET"),
    "ECG5000_UNBAL": ("retest", None),
    "CWRU_UNBAL": ("retest", None),
    "ECG5000_BAL": ("ecg5000_bal_3seed", None),
    "CWRU_BAL": None,                       # -> needs new M0 (canonical path)
    "GunPoint": ("context3", None),
    "ItalyPowerDemand": ("context3", None),
    "FordA": ("context3", None),
    "UWaveGestureLibraryAll": ("r2_uwave", "M0"),
    "UWaveGestureLibraryX": ("r2_uwave", "M0"),
    "UWaveGestureLibraryY": ("r2_uwave", "M0"),
    "UWaveGestureLibraryZ": ("r2_uwave", "M0"),
}

# canonical R5 sources (existing runs)
R5_SOURCES = {
    "UWaveGestureLibraryAll": os.path.join(RESULTS_ROOT,
                                           "r5_uwave_hbudget_seed42",
                                           "UWaveGestureLibraryAll"),
    "UWaveGestureLibraryX": os.path.join(RESULTS_ROOT,
                                         "r5_uwave_hbudget_seed42",
                                         "UWaveGestureLibraryX"),
    "UWaveGestureLibraryY": os.path.join(RESULTS_ROOT,
                                         "r5_uwave_hbudget_seed42",
                                         "UWaveGestureLibraryY"),
    "UWaveGestureLibraryZ": os.path.join(RESULTS_ROOT,
                                         "r5_uwave_hbudget_seed42",
                                         "UWaveGestureLibraryZ"),
}
RHO_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
TOTAL_BUDGET = 9996
K_FOLDS = 5
TIE_TOL = 0.001
REPRESENTATIVE_RHO_DATASETS = ["Haptics", "Phoneme", "ECG5000_UNBAL",
                               "CWRU_UNBAL"]

# R3/R4 gating ablation locations (Haptics only; R4 exists only there)
R3R4_SOURCES = {
    "Haptics": os.path.join(REPO_ROOT, "results", "rcmkn_haptics_seed42",
                            "report.json"),
}

DOMAINS = {
    "ECG5000_UNBAL": "Biomedical", "ECG5000_BAL": "Biomedical",
    "CWRU_UNBAL": "Biomedical", "CWRU_BAL": "Biomedical",
    "EpilepticSeizures": "Biomedical", "Haptics": "Motion",
    "Phoneme": "Audio", "GunPoint": "Motion",
    "ItalyPowerDemand": "Energy", "FordA": "Sensor",
    "UWaveGestureLibraryAll": "Motion", "UWaveGestureLibraryX": "Motion",
    "UWaveGestureLibraryY": "Motion", "UWaveGestureLibraryZ": "Motion",
}
