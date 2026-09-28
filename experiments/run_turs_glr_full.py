"""TURS-GLR master pipeline.

Usage:
  python experiments/run_turs_glr_full.py --all            # full study (default resume)
  python experiments/run_turs_glr_full.py --dataset CWRU_UNBAL
  python experiments/run_turs_glr_full.py --variant A3
  python experiments/run_turs_glr_full.py --train-only | --features-only |
      --ridge-only | --diagnostics-only | --report-only | --plots-only
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from experiments.turs_glr_pipeline.core import (
    phase0_audit, run_dataset_ablations, paired_significance,
    RESULTS, TABLE_DIR, FIG_DIR, VARIANTS, log)
from experiments.turs_glr_pipeline.diag_runner import run_diagnostics
from experiments.turs_glr_pipeline.reporting import (
    write_tables, evidence_rubric, write_report, write_figures)
from experiments.turs_rrmt.data import DATASETS

PIPELINE_LOG = os.path.join(ROOT, "logs", "turs_glr_full.log")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", default=True)
    ap.add_argument("--dataset", type=str, default=None)
    ap.add_argument("--variant", type=str, default=None)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--features-only", action="store_true")
    ap.add_argument("--ridge-only", action="store_true")
    ap.add_argument("--diagnostics-only", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--plots-only", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"TURS-GLR master pipeline. Device: {device}")

    tags = [args.dataset] if args.dataset else list(DATASETS)

    all_ab, all_diag, sig_rows = {}, {}, []
    if not args.report_only and not args.plots_only:
        phase0_audit()

    for ds in tags:
        log(f"=== DATASET {ds} ===")
        ab_path = os.path.join(RESULTS, ds, "ablation_results.json")
        if args.report_only or args.plots_only:
            all_ab[ds] = json.load(open(ab_path))
            all_diag[ds] = json.load(open(os.path.join(RESULTS, ds, "diagnostics.json")))
            continue
        if args.features_only:
            from experiments.turs_glr_pipeline.core import extract_features
            from experiments.turs_rrmt.data import load_split
            dsd = load_split(ds)
            extract_features(ds, dsd, device, "A3", force=args.force, log=log)
            continue
        if args.ridge_only:
            # fit/eval A3 only
            ab = run_dataset_ablations(ds, device, force=args.force,
                                       variants=["A0", "A1", "A2", "A3"], log=log)
        else:
            variants = [args.variant] if args.variant else None
            ab = run_dataset_ablations(ds, device, force=args.force,
                                       variants=variants, log=log)
        all_ab[ds] = ab
        if args.train_only:
            continue
        diag = run_diagnostics(ds, device, force=args.force, log=log)
        all_diag[ds] = diag
        sig_rows.extend(paired_significance(ds, dsd_ncls(ds), log=log))

    if args.train_only or args.features_only or args.ridge_only:
        log(f"SELECTED STAGES COMPLETE in {(time.time()-t0)/60:.1f} min")
        return

    log("PHASE: tables / rubric / report / figures")
    write_tables(all_ab, all_diag, sig_rows, tags)
    rubric = evidence_rubric(all_ab, all_diag, sig_rows, tags)
    from src.diagnostics.statistics import dump_json
    dump_json(rubric, os.path.join(RESULTS, "evidence_rubric.json"))
    if not args.plots_only:
        report = write_report(all_ab, all_diag, sig_rows, tags, rubric,
                              elapsed_s=time.time() - t0)
        log(f"report -> {report}")
    if not args.report_only:
        write_figures(all_ab, all_diag, tags)
    log(f"COMPLETE in {(time.time()-t0)/60:.1f} min")


def dsd_ncls(ds):
    from experiments.turs_rrmt.data import NUM_CLASSES
    return NUM_CLASSES[ds]


if __name__ == "__main__":
    main()
