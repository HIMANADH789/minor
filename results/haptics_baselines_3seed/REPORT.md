# Haptics Baseline Study — 3 Seeds (42/43/44)

## 1. Dataset and protocol

Canonical Haptics (UCR, univariate, T=1092, 5 classes) with the frozen
project split: train 132 / validation 23 / test 308 (seed-42 stratified
15% of official train; identical indices to every R2/R5 Haptics run).
Per-sample z-normalization only. Data audit: dataset identity, 132/23/308,
T=1092, 5 classes, z-norm, frozen split — all PASS (`audit.json`).

## 2. Model definitions used

Canonical repository implementations, reused unchanged:

- InceptionTime — `models/inceptiontime.py` (404,197 trainable params)
- FCN — `models/external_baselines.py` (66,885)
- ResNet1D — `models/external_baselines.py` (168,037)
- PatchTST — `models/external_baselines.py` (105,605)

Training protocol: the canonical external-stack `run_neural_baseline`
(AdamW lr 3e-4, weight decay 1e-2, batch 64, max 15 epochs, OneCycleLR,
early-stopping patience 6, gradient clipping 1.0; `NEURAL_CONFIG`).
Checkpoint selection uses **validation Macro-F1 only**; the official test
is evaluated **exactly once** per run after the checkpoint is frozen.
No hyperparameter search was performed; the configuration is identical
across all seeds and models.

## 3. Seed protocol

Seeds 42, 43, 44; per-run determinism via the canonical `set_seed`
(torch + numpy + CUDA). The **seed-42 runs were reused** from the stored
canonical external-stack results (`results/external_stack_generalization/`,
metrics + per-sample predictions) rather than rerun; seeds 43/44 were run
fresh through the identical code path. 4 models × 3 seeds = 12 runs
(4 reused, 8 fresh).

## 4. Per-seed results (test Macro-F1)

| Model | Seed 42 | Seed 43 | Seed 44 | Mean ± SD |
|---|---|---|---|---|
| InceptionTime | 0.0636 | 0.2475 | 0.0945 | 0.1352 ± 0.0985 |
| FCN | 0.0634 | 0.1302 | 0.0637 | 0.0858 ± 0.0385 |
| ResNet1D | 0.1512 | 0.0715 | 0.0693 | 0.0973 ± 0.0467 |
| PatchTST | 0.1692 | 0.1339 | 0.0756 | 0.1262 ± 0.0473 |

Best validation Macro-F1 per run and best epochs are in
`per_run_results.csv`; full per-epoch histories for the fresh runs in each
run's `training_history.json`.

## 5. Mean ± SD

See table above. With n=3 seeds these are descriptive stability
summaries only; no significance claims are made. Seed-to-seed variation
is large for all four models (SD 0.04–0.10), consistent with training
deep networks on 132 samples.

## 6. Parameter counts

InceptionTime 404,197 · FCN 66,885 · ResNet1D 168,037 ·
PatchTST 105,605 — versus 0 trainable parameters for MiniROCKET+Ridge
(M0) and 61,414 for the R2/R5 context pipeline.

## 7. Training times

Mean per-run training time (s): InceptionTime 3.42 · FCN 0.31 ·
ResNet1D 0.88 · PatchTST 0.39 (CUDA). Fresh runs only; the reused
seed-42 rows retain their originally recorded times.

## 8. Comparison against existing M0/R5 references

| Model | Seed 42 | Seed 43 | Seed 44 | Mean ± SD |
|---|---|---|---|---|
| MiniROCKET (M0) | 0.4974 | — | — | deterministic, seed-independent |
| InceptionTime | 0.0636 | 0.2475 | 0.0945 | 0.1352 ± 0.0985 |
| FCN | 0.0634 | 0.1302 | 0.0637 | 0.0858 ± 0.0385 |
| ResNet1D | 0.1512 | 0.0715 | 0.0693 | 0.0973 ± 0.0467 |
| PatchTST | 0.1692 | 0.1339 | 0.0756 | 0.1262 ± 0.0473 |
| R5 (rho=0.5, R2 / 50–50) | 0.5500 | 0.5213 | 0.5387 | 0.5367 ± 0.0145 |
| R5 (rho=rho*) | 0.5429 | 0.5264 | 0.5342 | 0.5345 ± 0.0083 |

Notes on the R5 family rows (frozen references, not rerun here):

- R2 is the fixed rho=0.5 point inside the R5 family; seed 42 is the
  canonical stored result, seeds 43/44 come from
  `results/rcmkn_r2_haptics_3seed/`.
- R5 (rho=rho*) rows use the per-seed CV-selected rho
  (42: 0.4, 43: 0.4, 44: 0.3) from
  `C:/temp/results/final_validation/multiseed_haptics.json`.
- Every R2/R5 seed and every MiniROCKET run is above every deep-baseline
  seed on this dataset.

## 9. Limitations

- n=3 seeds; the SDs are descriptive only and no significance tests are
  appropriate at this sample size, in either direction.
- The deep baselines use the canonical small-epoch configuration tuned
  for the broader benchmark, not per-dataset tuning; their absolute
  levels on Haptics are low and highly seed-variable, so they should be
  read as reproducible reference points under the shared protocol rather
  than as the models' ceiling.
- Seed-42 baseline rows are reused stored runs (identical code path);
  their full training histories were not retained by the original run.
- Haptics is a single dataset; these numbers do not generalize claims
  about the architectures elsewhere.

Reproduce: `python -c "from
experiments.haptics_baselines_3seed.runner import main; main()"`
(from `ECG_Benchmark/`). Tests:
`python -m pytest tests/test_haptics_baselines_3seed.py` (4 passed).
