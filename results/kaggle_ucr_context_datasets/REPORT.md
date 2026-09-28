# KAGGLE UCR DATASET INSTALLATION — GunPoint, ItalyPowerDemand, FordA

Dataset installation + validation only. **No model experiment was run.**

## 1. Kaggle CLI

- `kaggle` CLI 2.1.0 installed and authenticated (API credentials present; not exposed).

## 2. Search summary

CLI searches run: `GunPoint`, `gun point`, `GunPoint time series`, `ItalyPowerDemand`,
`Italy power demand`, `FordA`, `forda`, `Ford time series classification`,
`UCR`, `UCR time series`, `UCR archive`, `UCRArchive`, `time series classification`.
Raw search outputs saved in the session transcript; key findings below.

## 3. Candidates considered

| Kaggle ref | Verdict | Reason |
|---|---|---|
| **qianhuan/ucrarchive-2018** | **SELECTED (all 3 datasets)** | Exact repack of the official **UCRArchive_2018** zip: per-dataset folders with `<NAME>_TRAIN.tsv` / `<NAME>_TEST.tsv` (column 0 = class label), per-dataset README, top-level LICENSE (MIT). Contains GunPoint, ItalyPowerDemand, FordA with byte sizes exactly matching canonical shapes. |
| hariwh0/ucr-archives | REJECTED | Custom `.npy` repack (`X_train/X_valid/y_valid`) with train+test **merged and re-split** — violates the canonical UCR train/test protocol. |
| khalidamiralam/italypowerdemand | REJECTED (superseded) | Unvetted single-dataset re-upload (arff/ts/txt variants, usability 0.25, 2 downloads); the canonical archive mirror is strictly preferable. |
| All other search hits | REJECTED | Unrelated datasets (videos, crime statistics, affordability tables, road anomaly images, etc.). |

## 4. Download commands and paths

```bash
# single command covering all three datasets (canonical archive mirror):
kaggle datasets download -d qianhuan/ucrarchive-2018 -p data/kaggle/_ucrarchive_2018 --unzip

# then per dataset (copied without modification):
data/kaggle/GunPoint/            GunPoint_TRAIN.tsv, GunPoint_TEST.tsv, README.md
data/kaggle/ItalyPowerDemand/    ItalyPowerDemand_TRAIN.tsv, ItalyPowerDemand_TEST.tsv, README.md
data/kaggle/FordA/               FordA_TRAIN.tsv, FordA_TEST.tsv, README.md
```

Full archive retained at `data/kaggle/_ucrarchive_2018/` for provenance.
No existing repository dataset or aeon source was modified.

## 5. File structure

Each folder: `<NAME>_TRAIN.tsv`, `<NAME>_TEST.tsv` (tab-separated, **no header**,
column 0 = class label, columns 1..T = raw values), `README.md` (official UCR
description: train/test sizes, length, classes), `desktop.ini` (zip artifact).

| File | Bytes |
|---|---|
| GunPoint/GunPoint_TRAIN.tsv | 83,332 |
| GunPoint/GunPoint_TEST.tsv | 251,451 |
| ItalyPowerDemand/ItalyPowerDemand_TRAIN.tsv | 17,832 |
| ItalyPowerDemand/ItalyPowerDemand_TEST.tsv | 273,602 |
| FordA/FordA_TRAIN.tsv | 20,094,049 |
| FordA/FordA_TEST.tsv | 7,364,408 |

## 6. Shapes (Kaggle vs canonical aeon `load_classification`)

| Dataset | Kaggle train | aeon train | Kaggle test | aeon test | T | classes | labels |
|---|---|---|---|---|---|---|---|
| GunPoint | 50 × 150 | 50 × 150 | 150 × 150 | 150 × 150 | 150 | 2 | {1, 2} |
| ItalyPowerDemand | 67 × 24 | 67 × 24 | 1029 × 24 | 1029 × 24 | 24 | 2 | {1, 2} |
| FordA | 3601 × 500 | 3601 × 500 | 1320 × 500 | 1320 × 500 | 500 | 2 | {−1, 1} |

All standard univariate UCR classification datasets; no forecasting,
multivariate, augmented, or re-split variants.

## 7. Value-level cross-check (no normalization applied)

Two independent deterministic checks, per split:

1. **Sorted per-sample SHA-256 hashes** of (label, raw values) — ordering-independent:
   - GunPoint: train 50/50, test 150/150 identical multisets
   - ItalyPowerDemand: train 67/67, test 1029/1029 identical multisets
   - FordA: train 3601/3601, test 1320/1320 identical multisets
2. **All-rows nearest-row numeric match** against the aeon split:
   every Kaggle row matches an aeon row with **max_abs_diff = 0.000e+00,
   mean_abs_diff = 0.000e+00** — and every aeon row is matched exactly once,
   i.e. **sample ordering is also identical** (row i ↔ row i in both splits).

The Kaggle copies are **bit-identical** to the canonical aeon/UCR data.

## 8. Label check

- Kaggle TSV labels parse as floats: GunPoint/ItalyPowerDemand {1.0, 2.0},
  FordA {−1.0, 1.0}; aeon returns the same labels as strings {'1','2'} /
  {'−1','1'}.
- Mapping (documented, applied nowhere silently): `str(int(label))`.
- Label sets are equal for every split of every dataset.

## 9. What was NOT done (per spec)

No z-normalization, resampling, interpolation, augmentation, balancing,
permanent shuffling, or validation-split creation. No benchmark loader was
rewritten. No MiniROCKET / DRTN / Ridge experiment was run.

## 10. Acceptance criteria

| Dataset | found | verified UCR | downloaded | train/test identified | shape verified | classes verified | aeon cross-check |
|---|---|---|---|---|---|---|---|
| GunPoint | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ PASS |
| ItalyPowerDemand | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ PASS |
| FordA | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ PASS |

Verification implementation: `experiments/kaggle_ucr_context_datasets/verify.py`
(details in `verification_details.json`; manifest in `dataset_manifest.json`).

## 11. Exact next command (for the future experiment — NOT run here)

```bash
cd ECG_Benchmark
python -m experiments.<context_dependence_namespace>.runner \
    --datasets GunPoint ItalyPowerDemand FordA --kaggle-data data/kaggle
```

(The context-dependence experiment namespace does not exist yet; when created,
its loader should read the TSVs via `np.loadtxt(path, delimiter='\t')` with
column 0 as labels, preserving canonical train/test splits. Per-sample
z-normalization then applies at experiment time, as in the established
benchmark protocol.)
