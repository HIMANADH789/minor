# TURS-RRMT: Regime-Routed Multi-Transport TURS — Full Experiment Report

*Runtime 3.8 min · seed 42 · fixed pattern bank M=128 (lengths 7/11/15/23/31, 0 trainable params) · J=4 transport flavors · top-2 preservation · canonical protocol (AdamW 3e-4/1e-2, OneCycleLR, 30 ep max, patience 8, batch 64, val macro-F1 monitor)*

## 1. Motivation

TURS-Lite/Stack compress local temporal structure into a small regime
latent z_t BEFORE transport participates in prediction. TURS-RRMT
inverts this: expose many local patterns first (fixed MiniROCKET-style
bank, full temporal resolution), use their local activation to ROUTE
among 4 quantile-geometry transport flavors, preserve the top-2 routed
responses, and compress only late (mean/max/std pooling).

Central falsifiable hypothesis: **pattern-routed multi-transport (A4) >
uniform multi-transport (A3) > single transport (A2) > pattern-only (A1)**.
A3 is the critical control: identical to A4 except weights are uniform.

## 2. Predictive results (test Macro-F1)

| Dataset | A0 Ridge | A1 pattern | A2 1-flavor | A3 uniform | **A4 routed** | A5 no-topK | A6 shuffled | A7 fixed |
|---|---|---|---|---|---|---|---|---|
| CWRU_BAL | 0.8984 | 0.8278 | 0.7833 | 0.7795 | **0.8316** | 0.8237 | 0.4603 | 0.2642 |

## 3. The architectural test: A4 vs A3 (paired, same samples)

| Dataset | Δ Macro-F1 | Δ Acc [95% CI] | McNemar p | Δ NLL (perm p) | Cohen's d | Verdict |
|---|---|---|---|---|---|---|
| CWRU_BAL | +0.0521 | +0.0441 [+0.0159, +0.0741] | 0.0056 | -0.0804 (p=0.0005) | +0.122 | ROUTING IMPROVES (A4 > A3) |

Interpretation: A4> A3 consistently with CI excluding 0 => learned
routing justified. A4 ~ A3 => transport diversity helps but routing does
not; simplify. A4 < A3 => drop routing.

## 4. Routing behavior

| Dataset | routing entropy | top-1 flavor | route switch rate | A6 shuffle Δ |
|---|---|---|---|---|
| CWRU_BAL | (see routing_statistics.csv) | | | -0.3713 |

## 5. Diagnostic validation (D1-D9, frozen A4)

### CWRU_BAL
- **D2 routing validity**: strongest |rho| per flavor vs independent descriptors:
  - standard: lag_disagreement rho=-0.500 (p=0.0004997501249375312)
  - tail: lag_disagreement rho=-0.483 (p=0.0004997501249375312)
  - fine: lag_disagreement rho=-0.462 (p=0.0004997501249375312)
  - multilag: lag_disagreement rho=-0.442 (p=0.0004997501249375312)
- **D3 intervention**: shuffle flips 35.8% of predictions; McNemar p=6.92836204376785e-39
- **D4 faithfulness**: targeted drop 0.1204 vs random 0.0507 (diff +0.0697, p=0.0005)
- **D5 degradation**: mean |severity-rho| across signals = 0.893
- **D6 uncertainty**: error-detection AUROC routing-entropy 0.516 vs confidence 0.784

## 6. Interpretation discipline

- A6/A7 are inference-time interventions on frozen A4 (no retraining).
- If A6 (shuffled) ≈ A4, the learned pattern->flavor mapping is NOT
  functionally important — the model effectively uses an average flavor.
- Diagnostic correlations are associational, not causal.
- This is machine-learning diagnostic evidence, NOT clinical validation.

## 7. Complexity

- CWRU_BAL A4: 83,048 trainable + 3,968 fixed kernel params, 28.9s train, best ep 21

## 8. Limitations

- Single seed (42); no multi-seed variance reported.
- Transport flavors use fixed quantile grids; reference templates are
  TRAIN-only global means (not per-class).
- A0 Ridge uses global pooling; no tuning of Ridge alpha on validation.
- Synthetic/localization diagnostics use controlled perturbations; no
  real temporal event annotations exist in these datasets.
