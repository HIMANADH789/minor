# Corrected Fair Benchmark Report

## Setup

All models run under **identical protocol**: same train/val/test split (seed=42), same optimizer (AdamW, lr=3e-4, wd=1e-2), same batch size (64), same early stopping (patience 6, max 15 epochs), same CPU device.

**Critical fix**: Previous results used a crippled InceptionTime backbone (77K params). This report uses the **correct backbone** (135K for 1ch, 170K for 3ch).

---

## Parameter Counts

| Model | Params | Backbone |
|-------|-------:|----------|
| InceptionTime (1ch) | 135,301 | Correct 128-ch Inception |
| eTAI-Focal (3ch) | 170,437 | Correct 128-ch Inception |
| USTR-Net-CE | 100,142 | Compact Inception |
| USTR-Net-Focal | 100,142 | Compact Inception |
| TURS-Lite | 137,547 | Compact Inception + transport + regime |
| TURS-Strong | 184,733 | Compact Inception + multi-scale fusion |

---

## Primary Results (Macro-F1)

| Model | Params | ECG-U | ECG-B | CWRU-U | CWRU-B | **Avg MF1** |
|-------|-------:|:-----:|:-----:|:------:|:------:|:-----------:|
| **InceptionTime** | 135K | 0.5868 | 0.6324 | 0.8662 | 0.9647 | 0.7625 |
| **eTAI-Focal** | 170K | 0.5956 | **0.6388** | 0.8791 | **0.9665** | 0.7700 |
| **USTR-Net-CE** | 100K | 0.5948 | 0.6198 | 0.8529 | 0.9312 | 0.7497 |
| **USTR-Net-Focal** | 100K | 0.5763 | 0.6301 | 0.8585 | 0.9489 | 0.7534 |
| **TURS-Lite** | 138K | **0.6046** | 0.6377 | **0.8997** | 0.9417 | **0.7709** |
| **TURS-Strong** | 185K | 0.5955 | 0.5605 | 0.8920 | 0.9090 | 0.7392 |

---

## Accuracy

| Model | ECG-U | ECG-B | CWRU-U | CWRU-B |
|-------|:-----:|:-----:|:------:|:------:|
| InceptionTime | 0.9530 | 0.9490 | 0.8667 | 0.9647 |
| eTAI-Focal | 0.9510 | 0.9400 | 0.8792 | 0.9665 |
| USTR-Net-CE | 0.9530 | 0.9480 | 0.8542 | 0.9312 |
| USTR-Net-Focal | 0.9460 | 0.9400 | 0.8583 | 0.9489 |
| TURS-Lite | 0.9520 | 0.9420 | **0.9000** | 0.9418 |
| TURS-Strong | **0.9540** | 0.9130 | 0.8917 | 0.9101 |

---

## Ranking by Average MF1

1. **TURS-Lite: 0.7709** — best overall, strongest on unbalanced datasets
2. **eTAI-Focal: 0.7700** — best on balanced datasets
3. **InceptionTime: 0.7625** — solid baseline
4. **USTR-Net-Focal: 0.7534** — lightweight but competitive
5. **USTR-Net-CE: 0.7497** — slightly behind focal variant
6. **TURS-Strong: 0.7392** — overfits on small datasets

---

## Key Findings

### 1. TURS-Lite wins on unbalanced data

| Dataset | TURS-Lite MF1 | IT MF1 | Δ | eTAI MF1 | Δ |
|---------|:------------:|:------:|:-:|:--------:|:-:|
| ECG5000 Unbalanced | 0.6046 | 0.5868 | **+1.8pp** | 0.5956 | **+0.9pp** |
| CWRU Unbalanced | 0.8997 | 0.8662 | **+3.4pp** | 0.8791 | **+2.1pp** |

TURS-Lite's adaptive transport-regime fusion helps detect minority classes.

### 2. eTAI-Focal wins on balanced data

| Dataset | eTAI MF1 | TURS-Lite MF1 | Δ |
|---------|:--------:|:------------:|:-:|
| ECG5000 Balanced | 0.6388 | 0.6377 | **+0.1pp** |
| CWRU Balanced | 0.9665 | 0.9417 | **+2.5pp** |

On well-represented data, the simple transport channels of eTAI suffice and the extra regime machinery of TURS-Lite adds noise.

### 3. TURS-Strong overfits catastrophically

- CWRU Balanced: validation MF1 reached 0.8874 (ep5) then collapsed to 0.1003 (ep11)
- ECG5000 Balanced: MF1 = 0.5605 vs TURS-Lite 0.6377 — multi-scale fusion hurts on small data
- 185K params is too many for datasets with <3000 training samples

### 4. USTR-Net underperforms despite lightweight design

USTR-Net (100K params) consistently ranks below IT and eTAI, suggesting the uncertainty-aware regime mechanism alone doesn't compensate for the simpler backbone.

---

## Training Times (CPU)

| Model | ECG-U | ECG-B | CWRU-U | CWRU-B |
|-------|------:|------:|-------:|-------:|
| InceptionTime | 99s | 167s | 193s | 399s |
| eTAI-Focal | 222s | 171s | 197s | 469s |
| USTR-Net-CE | 316s | 143s | 147s | 340s |
| USTR-Net-Focal | 256s | 142s | 149s | 338s |
| TURS-Lite | 573s | 167s | 166s | ~400s |
| TURS-Strong | 188s | 229s | 278s | 470s |

TURS-Lite is ~5× slower than IT on ECG5000_UNBAL due to transport computation + regime losses.

---

## Dataset Details

| Dataset | Train | Val | Test | Classes |
|---------|------:|----:|-----:|--------:|
| ECG5000 Unbalanced | 3,400 | 600 | 1,000 | 5 (imbalanced) |
| ECG5000 Balanced | 5,226 | 923 | 1,000 | 5 (balanced) |
| CWRU Unbalanced | 1,156 | 204 | 240 | 4 (imbalanced) |
| CWRU Balanced | 2,727 | 482 | 567 | 4 (balanced) |

---

## Scientific Conclusion

> **TURS-Lite's adaptive uncertainty-controlled fusion between transport and regime representations achieves the highest average Macro-F1 (0.7709) across all datasets, while maintaining comparable parameter count to InceptionTime (138K vs 135K).**

The key mechanism works as expected:
- On **unbalanced** data: TURS-Lite leverages transport features for minority class detection (+1.8pp on ECG, +3.4pp on CWRU)
- On **balanced** data: eTAI's simpler approach is sufficient
- TURS-Strong's multi-scale fusion **overfits** on small datasets, confirming that compact models are preferable

**The strongest practical recommendation**: Use **TURS-Lite** for imbalanced time-series classification, and **eTAI-Focal** for balanced data.
