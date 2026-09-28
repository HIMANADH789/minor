# RUN_HANDOFF — External Stack Generalization Experiment (FINAL)

- **Status: COMPLETE** (2026-09-14 10:48 IST). Run PID 20508 finished all three
  datasets at 10:34; aggregates/figures/report rebuilt via `--report-only`
  (three small post-run bugs in the aggregation code were fixed; per-dataset
  results untouched and cache-resumable).

## Final results (TEST Macro-F1, seed 42)

| Dataset | MiniROCKET | InceptionTime | ResNet-1D | FCN | PatchTST | TURS-Stack | Δ(Stack−MR) | McNemar q |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EpilepticSeizures (PRIMARY) | **0.9194** | 0.9060 | 0.7935 | 0.3018 | 0.1331 | 0.8706 | −0.0488 | 0.0 (sig-neg) |
| Haptics | **0.4974** | 0.0636 | 0.1512 | 0.0634 | 0.1692 | 0.3553 | −0.1421 | 1.5e-05 |
| Phoneme | **0.0808** | 0.0445 | 0.0258 | 0.0015 | 0.0126 | 0.0785 | −0.0023 | 0.0 |

- **Outcome class C**: Stack wins on none of the three; the ECG5000 Stack
  advantage is dataset-specific. MiniROCKET ranks 1/6 on all three (STRONG).
- Biomedical generalization: NOT SUPPORTED. Cross-domain: NOT SUPPORTED.
- Stack val→test gaps (selected combiner): ES −0.129, Haptics −0.416,
  Phoneme −0.922 (combiner selection on tiny val sets overfits; soft-vote
  gaps much smaller).
- Complementarity: Stack errors only partially overlap MR's
  (0.19 / 0.49 / 0.70) and Stack fixes some MR mistakes (321 samples on ES),
  but MR fixes far more of Stack's (809) — net advantage MR everywhere.

## Deliverables in results/external_stack_generalization/
dataset_audit.md, RUN_HANDOFF.md, REPORT.md (reports/), master_comparison.csv,
baseline_results.csv, stack_results.csv, per_class_results.csv,
branch_results.csv, statistical_tests.csv, complementarity.csv,
validation_test_gap.csv, runtime.csv, parameter_counts.csv,
dataset_characteristics.csv, full_results.json, diagnostics.json,
leakage_audit.json (all PASS), figures/ (6), predictions/*.npy (all models +
branches), checkpoints/ (resume), configs/environment.json.

## Reproduce / extend
- Rebuild aggregates/report only:
  `python -m experiments.external_stack_generalization.runner --report-only`
- Rerun a dataset: `python -m experiments.external_stack_generalization.runner --dataset Phoneme`
- Robustness seeds 43-46 (PHASE 22; NOT yet run — results are single-seed):
  `python -m experiments.external_stack_generalization.runner --seeds 43 44 45 46`
