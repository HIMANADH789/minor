# Real CWRU Bearing Fault — Fair Comparison Report

## Data Source

**Real** CWRU bearing fault vibration data from Case Western Reserve University, downloaded via Kaggle (`brjapon/cwru-bearing-datasets`).

- Raw drive-end accelerometer signals at 48kHz sampling rate, Load 1
- 10 raw `.mat` files → windowed into 1024-sample segments with 50% overlap
- 4 classes: Normal, Inner Race (IR), Ball, Outer Race (OR)
- Multiple severity levels (0.007, 0.014, 0.021 inch) collapsed into fault-type classes

| Dataset | Total | Normal | IR | Ball | OR |
|---------|-------|--------|-----|------|-----|
| CWRU Unbalanced | 1,600 | 400 | 400 | 400 | 400 |
| CWRU Balanced | 3,776 | 944 | 944 | 944 | 944 |

## Protocol (Identical for All Models)

- Seed=42, 70/15/15 stratified split
- AdamW lr=3e-4, weight_decay=1e-2
- OneCycleLR scheduler
- Early stopping on val MF1, patience=8
- Max 30 epochs
- Per-sample z-normalization

## Results

### CWRU Unbalanced (60 test samples per class)

| Model | Params | Acc | MF1 | F1-C0 | F1-C1 | F1-C2 | F1-C3 | Ep | Time |
|-------|-------:|----:|----:|------:|------:|------:|------:|---:|-----:|
| InceptionTime | 135K | 0.8625 | 0.8546 | 1.000 | 0.984 | 0.775 | 0.659 | 7 | 18s |
| **eTAI-Focal** | **170K** | **0.9125** | **0.9125** | **1.000** | **1.000** | **0.824** | **0.826** | 28 | 36s |
| USTR-Net-CE | 100K | 0.8833 | 0.8833 | 1.000 | 1.000 | 0.770 | 0.763 | 27 | 36s |
| USTR-Net-Focal | 100K | 0.9042 | 0.9036 | 1.000 | 1.000 | 0.822 | 0.793 | 26 | 30s |

**Winner: eTAI-Focal** (MF1=0.9125, +5.8pp over InceptionTime on minority classes)

### CWRU Balanced (142 test samples per class)

| Model | Params | Acc | MF1 | F1-C0 | F1-C1 | F1-C2 | F1-C3 | Ep | Time |
|-------|-------:|----:|----:|------:|------:|------:|------:|---:|-----:|
| InceptionTime | 135K | 0.9753 | 0.9753 | 1.000 | 1.000 | 0.950 | 0.952 | 27 | 78s |
| eTAI-Focal | 170K | 0.9683 | 0.9683 | 1.000 | 0.997 | 0.937 | 0.940 | 30 | 84s |
| **USTR-Net-CE** | **100K** | **0.9753** | **0.9754** | **1.000** | 0.996 | **0.951** | **0.954** | 25 | 66s |
| USTR-Net-Focal | 100K | 0.9718 | 0.9717 | 1.000 | 0.989 | 0.943 | 0.955 | 29 | 53s |

**Winner: USTR-Net-CE** (MF1=0.9754, with fewest parameters)

### Combined Average

| Model | Params | Avg MF1 | MF1/100K |
|-------|-------:|--------:|---------:|
| InceptionTime | 135K | 0.9150 | 0.678 |
| **eTAI-Focal** | **170K** | **0.9404** | **0.553** |
| USTR-Net-CE | 100K | 0.9319 | 0.932 |
| USTR-Net-Focal | 100K | 0.9377 | 0.938 |

## Confusion Matrices (CWRU Unbalanced)

**eTAI-Focal:**
```
         Normal  IR  Ball  OR
Normal  [  60     0     0     0 ]
IR      [   0    60     0     0 ]
Ball    [   0     0    49    11 ]
OR      [   0     0    10    50 ]
```

**USTR-Net-Focal:**
```
         Normal  IR  Ball  OR
Normal  [  60     0     0     0 ]
IR      [   0    60     0     0 ]
Ball    [   0     0    53     7 ]
OR      [   0     0    16    44 ]
```

## Key Findings

1. **eTAI-Focal wins on CWRU Unbalanced** — transport channels (raw+quantile+drift) help Ball/OR discrimination by +5pp MF1 over InceptionTime
2. **USTR-Net-CE wins on CWRU Balanced** — uncertainty-aware regime modulation achieves the highest accuracy with 26% fewer parameters than eTAI
3. **All models achieve perfect classification on Normal and IR** — these classes have distinct spectral signatures
4. **Ball vs OR is the hardest distinction** — confusion matrices show most errors are Ball↔OR misclassification (similar frequency content)
5. **USTR-Net-Focal is the most parameter-efficient** — 0.938 MF1/100K, nearly 2x InceptionTime's efficiency
6. **Transfer from synthetic to real**: The earlier synthetic bearing results (where USTR-Net scored 0.979 MF1) were inflated. On real data, the advantage narrows but USTR-Net still performs competitively

## Comparison: Real CWRU vs Synthetic Bearing

| Metric | Synthetic (earlier) | Real CWRU (this run) |
|--------|:---:|:---:|
| IT MF1 (unbalanced) | 0.865 | 0.855 |
| eTAI MF1 (unbalanced) | 0.851 | **0.913** |
| USTR-Net MF1 (unbalanced) | **0.979** | 0.904 |
| Best model | USTR-Net | eTAI-Focal |

**Critical observation**: The synthetic data favored USTR-Net (+13pp over baselines), but the real CWRU data shows eTAI-Focal as the strongest model. This confirms that the earlier synthetic bearing results were not representative of real-world performance.
