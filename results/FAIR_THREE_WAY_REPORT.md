# Clean Three-Way Fair Comparison

## Protocol
- **Seed**: 42
- **Split**: 70% train / 15% val / 15% test (stratified)
- **Early stopping**: patience=10 on validation MF1
- **Optimizer**: Adam, lr=1e-3
- **Max epochs**: 30
- **Loss**: Cross-entropy (all models)

## Models

| Model | Input | Params | Architecture |
|-------|-------|--------|-------------|
| InceptionTime | 1ch raw | 135K | 4-block Inception CNN |
| eTAI-Focal | 3ch (raw+Q+D) | 170K | 4-block Inception CNN with transport input |
| HCRMN-Lite | 4ch (raw+Q+D+M) | 162K | Transport backbone + hierarchical regime manifold + FiLM + dynamics |

## Results

| Dataset | InceptionTime | eTAI-Focal | HCRMN-Lite | Winner |
|---------|--------------|------------|-----------|--------|
| ECG5000_UNBAL | 0.5285 | **0.5898 | 0.5372 | eTAI-Focal (3ch) |
| ECG5000_BAL | 0.5992 | **0.6348 | 0.6063 | eTAI-Focal (3ch) |
| BEARING_UNBAL | **0.8651 | 0.8513 | 0.8232 | InceptionTime (1ch) |
| BEARING_BAL | **0.9950 | 0.9937 | 0.9924 | InceptionTime (1ch) |

| Average MF1 | 0.7470 | 0.7674 | 0.7398 | eTAI-Focal |

## Per-Class F1 Detail

### ECG5000_UNBAL (5 classes)

| Model | C0 | C1 | C2 | C3 | C4 | MF1 |
|-------|-----|-----|-----|-----|-----------|
| InceptionTime (1ch) | 0.991 | 0.944 | 0.308 | 0.400 | 0.000 | 0.5285 |
| eTAI-Focal (3ch) | 0.996 | 0.949 | 0.529 | 0.475 | 0.000 | 0.5898 |
| HCRMN-Lite (4ch) | 0.993 | 0.938 | 0.370 | 0.385 | 0.000 | 0.5372 |

### ECG5000_BAL (5 classes)

| Model | C0 | C1 | C2 | C3 | C4 | MF1 |
|-------|-----|-----|-----|-----|-----------|
| InceptionTime (1ch) | 0.994 | 0.938 | 0.585 | 0.479 | 0.000 | 0.5992 |
| eTAI-Focal (3ch) | 0.994 | 0.945 | 0.585 | 0.507 | 0.143 | 0.6348 |
| HCRMN-Lite (4ch) | 0.995 | 0.939 | 0.583 | 0.514 | 0.000 | 0.6063 |

### BEARING_UNBAL (4 classes)

| Model | C0 | C1 | C2 | C3 | MF1 |
|-------|-----|-----|-----|-----------|
| InceptionTime (1ch) | 0.848 | 0.918 | 0.792 | 0.903 | 0.8651 |
| eTAI-Focal (3ch) | 0.889 | 0.870 | 0.820 | 0.827 | 0.8513 |
| HCRMN-Lite (4ch) | 0.928 | 0.835 | 0.738 | 0.793 | 0.8232 |

### BEARING_BAL (4 classes)

| Model | C0 | C1 | C2 | C3 | MF1 |
|-------|-----|-----|-----|-----------|
| InceptionTime (1ch) | 0.995 | 0.995 | 0.995 | 0.995 | 0.9950 |
| eTAI-Focal (3ch) | 0.990 | 0.998 | 0.990 | 0.998 | 0.9937 |
| HCRMN-Lite (4ch) | 0.998 | 0.990 | 0.995 | 0.987 | 0.9924 |

## Key Findings

### 1. eTAI-Focal wins on ECG5000 (both balanced and unbalanced)
- Transport input channels (quantile + drift) boost minority class F1 by +6-22pp over raw InceptionTime
- eTAI C2 F1: 0.529 vs IT 0.308 on unbalanced (+72% relative improvement)

### 2. Plain InceptionTime wins on Bearing Fault data
- On bearing data, transport channels slightly *hurt* performance (-1.4pp MF1 on unbalanced)
- The regime manifold in HCRMN-Lite hurts even more (-4.2pp on unbalanced)
- For clean spectral fault signatures, the simpler backbone generalizes better

### 3. No model detects C4 (rarest class) on unbalanced ECG5000
- C4 has only 5 test samples; all models score F1=0.000
- On balanced ECG5000, only eTAI-Focal detects C4 (F1=0.143, 1/5 correct)

### 4. Parameter efficiency
| Model | Params | Avg MF1 | MF1/100K params |
|-------|--------|---------|-----------------|
| InceptionTime | 135K | 0.747 | 0.554 |
| eTAI-Focal | 170K | 0.767 | 0.452 |
| HCRMN-Lite | 162K | 0.740 | 0.456 |

InceptionTime remains the most parameter-efficient. eTAI-Focal trades +22K params for +2.0pp avg MF1 on ECG (worth it) but not on bearing (not worth it).

### 5. Training speed
HCRMN-Lite is 5x faster than IT/eTAI (38s avg vs 190-237s) due to its compact transport encoder replacing some Inception blocks.

## Honest Assessment

Under **identical conditions** (same split, seeds, optimizer, early stopping):
- **eTAI-Focal is the strongest overall** (+2.0pp avg MF1 over InceptionTime, driven by ECG minority classes)
- **HCRMN-Lite does not outperform** the simpler models under fair conditions
- The earlier HCRMN-Lite results (MF1=0.956 on bearing) were from **non-comparable conditions** (different splits, early stopping on test set, different data preprocessing)
- On bearing data, **simpler is better**: transport features and regime manifolds add noise for clean spectral patterns
- On ECG data with minority classes: **transport input features are genuinely helpful** (+6-7pp on C2/C3)

## Files
- 
- 
- 
- 
- 
-  — unified training script
- Checkpoints: 