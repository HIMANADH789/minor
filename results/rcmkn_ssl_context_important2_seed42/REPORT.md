# SSL-Context + Hard-VQ Heterogeneity — Transfer Part 2 (ECG5000_BAL, CWRU_BAL; seed 42)

Fixed-method transfer of the validated Haptics R2 architecture to the two
remaining canonical NPZ datasets. No M0, no Hydra, no architecture changes;
all hyperparameters transferred unchanged from Haptics R2.

## 1. Objective

Test whether the SSL-learned temporal-context mechanism (validated on
Haptics and transferred to Phoneme / ECG5000_UNBAL / CWRU_UNBAL in part 1)
transfers to ECG5000_BAL and CWRU_BAL.

## 2. Hypothesis

SSL-learned temporal context + fixed MiniROCKET responses produces more
useful features than the DRTN temporal context (R0), occupancy-matched
random partitioning (C1), or shuffled learned-regime alignment (C2).

## 3. Dataset Configurations

| Dataset | Train | Val | Test | T | Classes | Labels | M0 ref (not run) |
|---|---|---|---|---|---|---|---|
| ECG5000_BAL | 5226 | 923 | 1000 | 140 | 5 | {0,1,2,3,4} | 0.6553 |
| CWRU_BAL | 2727 | 482 | 567 | 1024 | 4 | {0,1,2,3} | 0.9947 |

Both univariate; per-sample z-normalization before MiniROCKET/SSL/DRTN
(canonical convention, unaltered). Splits resolved from the canonical NPZ
loaders and asserted (AUDIT 1).

## 4. Architecture

Identical to Haptics R2: aeon MiniRocket (9996) → SSL causal encoder
(4 blocks, k=3/5/7/9, dil=1/2/4/8, ch=32/32/64/64, d=32, ChannelNorm) →
HardVQ (K=8, EMA 0.99, revival) → audited heterogeneity
`H_m = Σ_k q_k (PPV_{m,k} − PPV_m)²` with per-feature valid regions →
RidgeClassifierCV(logspace(−4,4,20)) on train+val. 61,414 trainable params
(encoder 59,072 + decoder 2,177 + aux head 165).

## 5. Training Protocol

SSL 120 ep (lr 1e-3, wd 1e-4, batch 8, mask 10%, span 16, patience 20);
joint 60 ep (lr 5e-4, λ_cls=0.10, patience 10). Checkpoints frozen and
saved. DRTN (R0): ECG5000_BAL uses the **audited 3-seed study checkpoint**
(seed42, val MF1 0.8638@ep21); CWRU_BAL uses the frozen transfer-screen
checkpoint (val MF1 0.8719@ep17) — no audited reference exists, so its
in-run R0 is the first audited value. Runtimes: ECG5000_BAL 731.7s,
CWRU_BAL 503.0s.

## 6. Audit Results (all passed before any test evaluation)

| Audit | ECG5000_BAL | CWRU_BAL |
|---|---|---|
| 1 config | PASS | PASS |
| 2 extractor identity | 0.00e+00 max, 0.00e+00 mean | 0.00e+00 / 0.00e+00 |
| 3 budget 9996 | PASS | PASS |
| 4 4998+4998 feature-axis | PASS | PASS |
| 5 raw-response provenance | PASS (audited core) | PASS |
| 6 independent H recompute | 3.9e-09 | 2.3e-09 |
| 7 per-feature valid region | 170,328 flips → H exact | 879,392 flips → H exact |
| 8 causality | 0.0e+00 | 0.0e+00 |
| 9/10 SSL mask / no labels | PASS (unit tests) | PASS |
| 11/12 VQ / EMA | PASS (unit tests) | PASS |
| 13/14 C1/C2 occupancy | 0 failures | 0 failures |
| 15/16 C1≠C2, no shared mem | 78.6% differ | 53.9% differ |
| 17 Ridge train+val | PASS | PASS |
| 18 no test leakage | PASS (structural) | PASS |
| **19 R0 gate** | **0.6748 == 0.6748 exact PASS** | first audited ref 0.9930 |
| 20 NaN/Inf | features finite | features finite |

Note: SSL train-loss NaNs appeared intermittently on ECG5000_BAL (6149
samples); validation loss stayed finite and checkpoint selection is
val-based, so the frozen model is well-defined. This is a training-stability
observation of the transferred recipe on the largest dataset, documented
under Limitations.

## 7. R0 / R2 / C1 / C2 Results (test Macro-F1, seed 42)

| Dataset | R0 | R2 | C1 | C2 | Verdict |
|---|---|---|---|---|---|
| ECG5000_BAL | **0.6748** | 0.6409 | 0.6168 | 0.6210 | NEGATIVE |
| CWRU_BAL | 0.9930 | **0.9982** | 0.9842 | 0.9859 | STRONG TRANSFER |

## 8. Per-Dataset Deltas

| Dataset | R2−R0 | R2−C1 | R2−C2 | C1−C2 |
|---|---|---|---|---|
| ECG5000_BAL | −0.0339 | +0.0241 | +0.0199 | −0.0042 |
| CWRU_BAL | +0.0052 | +0.0140 | +0.0123 | −0.0017 |

Validation MF1 / alphas: ECG5000_BAL R0 0.9908 / 1.62, R2 0.9935 / 1.62;
CWRU_BAL R0 0.9930α0.23, R2 0.9982α0.09 (val 0.9982/0.9982 — R2's val was
not higher; see report.json).

## 9. Regime and H Diagnostics

VQ codebook (R2, trainva): ECG5000_BAL 8/8 codes, entropy 0.995,
perplexity 7.91, dominant 0.161; CWRU_BAL 8/8, 0.977, 7.62, 0.184. No
collapse on either dataset.

| Dataset | mean H (R0) | mean H (R2) | mean H (C1) | mean H (C2) |
|---|---|---|---|---|
| ECG5000_BAL | 0.0323 | 0.0263 | 0.0098 | 0.0099 |
| CWRU_BAL | 0.0177 | 0.0116 | 0.0015 | 0.0015 |

Both learned sources create far stronger kernel–context dependence than the
controls (2.7–7.7× above the random/shuffled noise floor), but **H magnitude
does not rank the learned sources' accuracy**: on ECG5000_BAL R0's higher H
accompanies its win; on CWRU_BAL R2 wins with *lower* H than R0. High H is
not sufficient — the classifier comparison decides.

## 10. Dataset-Level Verdicts

- **ECG5000_BAL: NEGATIVE.** R2 = 0.6409 < R0 = 0.6748 (−3.39pp). The SSL
  recipe underperforms the DRTN reference on the largest, T=140 dataset
  (the shortest series in the sweep — the causal encoder's receptive field
  and 10%/16-span masking were designed for T≈1024). Controls rank well
  below both learned sources, so the conditioning mechanism itself remains
  sound; the SSL context source specifically loses to DRTN here.
- **CWRU_BAL: STRONG TRANSFER.** R2 = 0.9982 > R0 0.9930 (+0.52pp) and
  > C1/C2 (+1.40/+1.23pp), and exceeds the canonical M0 reference (0.9947).
  Under a near-ceiling baseline this is a clean sweep of all three
  comparisons — reported as strong transfer with the caveat that margins
  are small in absolute terms.

## 11. Comparison with Prior Evidence (descriptive only, no pooling)

| Dataset | M0 | R0 | R2 | R2−R0 |
|---|---|---|---|---|
| Haptics (seed 42) | 0.4974 | 0.5366 | 0.5500 | +0.0134 |
| Phoneme | 0.0808 | 0.1119 | 0.1182 | +0.0063 |
| ECG5000_UNBAL | 0.5938 | 0.5859 | 0.5894 | +0.0035 |
| CWRU_UNBAL | 0.9917 | 0.9792 | 0.9917 | +0.0125 |
| ECG5000_BAL (3-seed audited R0 ref) | 0.6553 | 0.6748 | 0.6409 | **−0.0339** |
| CWRU_BAL | 0.9947 | 0.9930 | 0.9982 | +0.0052 |

Qualitative pattern across the 6-dataset sweep: SSL context beats the DRTN
reference on 5/6 datasets, with the single loss on ECG5000_BAL — the only
T=140 dataset with a large training set. No formal statistical test is
applied.

## 12. Which Dataset Deserves 3-Seed Confirmation?

- **CWRU_BAL: yes, conditionally.** The R2 sweep of R0/C1/C2 under a
  near-ceiling M0 is the strongest single-seed signature in the entire
  transfer program; a 3-seed run would establish whether +0.52pp over R0
  and the M0-exceedance are stable. Cheap to run (503s/seed).
- **ECG5000_BAL: no for R2 as-is.** The −3.39pp deficit is the largest
  effect in the sweep; confirmation would only solidify a negative. If
  revisited, the T=140 masking/receptive-field mismatch is the declared
  hypothesis — but that is an architecture change, out of scope for this
  fixed-method screen.

## 13. Limitations

- Single seed; no significance claims.
- Intermittent SSL train-loss NaNs on ECG5000_BAL (largest dataset);
  val-based selection kept the frozen model well-defined, but training
  stability of the transferred recipe is unverified beyond val curves.
- CWRU_BAL R0 has no prior audited reference; its in-run R0 (0.9930) is the
  first — the gate protection available on ECG5000_BAL did not exist here.
- Near-ceiling CWRU_BAL: small margins are less informative; the task's
  CWRU-specific interpretation rule was applied.
- ECG5000_BAL's T=140 mismatches the Haptics-tuned mask/span/receptive-field
  design; the negative is specific to this transferred configuration.

## 14. Reproduction Commands

```bash
cd ECG_Benchmark
python -m experiments.rcmkn_ssl_context_important2_seed42.runner            # full run
python -m experiments.rcmkn_ssl_context_important2_seed42.runner --smoke --datasets CWRU_BAL
python -m experiments.rcmkn_ssl_context_important2_seed42.figures           # 3 figures
python -m pytest tests/test_rcmkn_ssl_context_important2_seed42.py -q       # 18 tests
```

Full conditioned-MiniROCKET/RCMKN suite: **127 passed** (incl. these 18).

## Artifacts

`results/rcmkn_ssl_context_important2_seed42/`: `REPORT.md`, `report.json`,
`per_dataset_results.json`, `config.json`, `figures/` (3 PNGs),
`<DS>/result.json`, `<DS>/context_model_seed42.pt`, `<DS>/predictions.csv`,
`<DS>/diagnostics/audits.json`.

Per spec: **STOP** — 8 official test evaluations completed; no M0, no other
datasets, no seeds 43/44.
