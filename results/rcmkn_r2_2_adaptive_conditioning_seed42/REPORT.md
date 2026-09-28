# R2.2 — Adaptive Discrete Context-to-Kernel Conditioning (seed 42)

## 1. Objective

Screen whether directly conditioning fixed MiniROCKET temporal responses on the
learned discrete VQ context (`R2.2 = R2 + adaptive context-to-kernel
conditioning`) improves classification beyond the unconditioned R2
representation, using a single seed (42) on the two pre-declared datasets:

- **Haptics** — the strongest existing R2 demonstration (R2 = 0.5500 vs M0 = 0.4974)
- **CWRU_BAL** — near-ceiling baseline (R2 = 0.9982 vs M0 = 0.9947), different signal domain

Per the scope amendment: only these two datasets, seeds 43/44 excluded, M0 not
rerun, 2 × 5 = **10 official test evaluations** (not the original 35).

## 2. Hypothesis

A learned discrete temporal context (HardVQ code k_t) can adapt the
*interpretation* of fixed MiniROCKET kernel responses — the same kernel may be
amplified or attenuated depending on the learned temporal state — improving on
R2's indirect use of the regime (heterogeneity of unconditioned activations
within regimes).

## 3. R2.2 mathematical definition

For feature f with canonical bias b_f and raw response C_f(t):

```
s_{k,f}   = tanh(a_{k,f})                    ∈ [-1, 1]   (K × M table, init 0)
beta      = beta_max * sigmoid(b)            ∈ (0, 0.5), beta_max = 0.5
delta     = beta * s                          ∈ (-0.5, 0.5)
r̃_f(t)   = C_f(t) * (1 + delta_{k_t, f})     ∈ scale [0.5, 1.5], positive
act̃_f(t) = 1[ r̃_f(t) > b_f ]
```

Evaluated **exactly** in scale-invariant margin form without materializing
modulated responses:

```
u_f(t)      = (C_f(t) − b_f) / |b_f|         (±2 convention for b_f = 0)
τ_{k,f}     = sign(b_f) · ( −delta/(1+delta) )
act̃_f(t)    = 1[ u_f(t) > τ_{k_t, f} ]
```

The bias-sign fold (found by the direct-definition audit during development:
a naive unsigned threshold is wrong for the ~half of biases with b_f < 0) and
the |τ| < 1 clipping-losslessness property are unit-tested. Primary R2.2
feature vector: `[4998 global PPV | 4998 conditioned heterogeneity H2]` with
`H2_m = Σ_k q_k (PPV2_{m,k} − PPV2_m)²` from conditioned activations —
exactly 9996 features, thresholds untouched, per-feature valid regions
[padding_m, T−padding_m).

## 4. Architecture

Identical to validated R2 (frozen): causal multi-scale CNN encoder
(k=3/5/7/9, dil 1/2/4/8, ch 32/32/64/64, proj 32, ChannelNorm, no
downsampling) → HardVQ (K=8, EMA, straight-through, dead-code revival,
diversity) → hard regimes k_t. **Frozen R2 context checkpoints** are reused
(`rcmkn_haptics_seed42/context_model_seed42.pt`,
`rcmkn_ssl_context_important2_seed42/CWRU_BAL/context_model_seed42.pt`), so
B0-vs-B2 is perfectly controlled: identical encoder, codes, and global block.
Only the modulation table (a: 8×4998, b: scalar) and a surrogate aux head are
trained. MiniROCKET canonical aeon, fit on z-normed train only; the margin
kernel recomputes identical activations (bool-equality 0 diffs) plus float32
margins for the 4998 het-block features.

## 5. Ablation ladder

| Variant | Definition | Test purpose |
|---|---|---|
| B0 | unconditioned H from original activations | frozen R2 reference |
| B1 | full R2.2 path with delta ≡ 0 forced | capacity-only control; must equal B0 |
| B2 | full R2.2, learned (a, b), identity init | primary model |
| B3A | B2 table + occupancy-matched random codes (seed 900001-stream) | partition-only control |
| B3B | B2 table + per-sample shuffled codes (seed 900002-stream) | alignment control |

## 6. Dataset configurations

| Dataset | train | val | test | T | classes | frozen R2 ref | M0 ref |
|---|---|---|---|---|---|---|---|
| Haptics | 132 | 23 | 308 | 1092 | 5 | 0.5500 | 0.4974 |
| CWRU_BAL | 2727 | 482 | 567 | 1024 | 4 | 0.9982 | 0.9947 |

Both match the canonical project splits exactly (AUDIT 1). Preprocessing:
per-sample z-normalization feeding MiniROCKET, SSL encoder, and VQ alike.

## 7. Audit results

All audits passed before any test evaluation, on both datasets:

| Audit | Result |
|---|---|
| 1 dataset identity | exact match both datasets |
| 2 MiniROCKET identity (margin kernel vs audited extractor) | **0 differing positions**, both splits |
| 3/4 budget & axis | 9996 = 4998 + 4998 asserted per variant; sample/feature axes guarded |
| 5/6 H and H2 formulas | implementation vs independent float64 recompute: max 6.05e-09 (audited tolerance) |
| 7 per-feature valid region | corruption of invalid positions leaves H unchanged exactly |
| 8 VQ | hard assignment / EMA / revival / diversity from validated components; codebooks 8/8 active |
| 9 identity init | fresh module: delta max = 0.0 exactly |
| 10 modulation bounds | Haptics scale ∈ [0.9855, 1.0142]; CWRU_BAL scale ∈ [0.7049, 1.2827] — inside (0.5, 1.5) |
| 11 code-specific indexing | tau indexed by k_t only (hard codes from frozen VQ) |
| 12 **B1 == B0** | **max abs diff = 0.00e+00, both datasets, trainva and test** |
| 13/14 occupancy | 0 failures across all audited samples (B3A and B3B vs B2) |
| 15 B3A/B3B distinction | distinct hashes, >1% differing positions, no shared memory |
| 16 conditioning non-placeholder | H entries changed vs B0: Haptics 35.7%, CWRU_BAL 87.2% |
| 17–19 causality / SSL mask / no-leakage | encoder+VQ frozen from audited R2; modulation trained on train histograms only, val-selected; test touched only at final evaluation |
| 20 train+val Ridge | `RidgeClassifierCV(logspace(-4,4,20))` fit on train+val asserted |

NaN/Inf guards: no non-finite losses or features observed (the ECG5000_BAL
mask-starvation NaN bug previously fixed in `make_span_mask` is inherited by
all runs here; no starved masks occur at T≥1024 in any case).

## 8. Test results (test Macro-F1, seed 42, evaluated exactly once)

| Dataset | B0 | B1 | B2 | B3A | B3B |
|---|---|---|---|---|---|
| Haptics | 0.5500 | 0.5500 | **0.5500** | 0.5008 | 0.5011 |
| CWRU_BAL | **0.9982** | **0.9982** | 0.9930 | 0.9842 | 0.9859 |

Ridge details: Haptics alpha 4.281 (all variants), val MF1 0.9014 (B0/B1/B2).
CWRU_BAL alpha 0.0886 (B0/B1/B2; 0.2336 controls), val MF1 1.0000.

## 9. Per-dataset deltas

| Dataset | B2−B0 | B2−B1 | B2−B3A | B2−B3B | B3A−B3B | B2−M0 |
|---|---|---|---|---|---|---|
| Haptics | **0.0000** | 0.0000 | +0.0492 | +0.0489 | −0.0003 | +0.0526 |
| CWRU_BAL | **−0.0052** | −0.0052 | +0.0088 | +0.0071 | −0.0017 | −0.0017 |

## 10. Modulation diagnostics

| Statistic | Haptics | CWRU_BAL |
|---|---|---|
| beta | 0.2588 | 0.4060 |
| mean \|s\| | 0.0112 | 0.1611 |
| median \|s\| / max \|s\| | ~0 / 0.0562 | — / 0.7268 |
| pairwise cos mean | −0.028 | −0.081 |
| fraction near-zero code vectors | 0.0 | 0.0 |
| fraction near-identical code vectors | ~0 | ~0 |
| training | 47 epochs, best val 0.6688 | 58 epochs, best val 0.9876 |

Conditioning was **active and non-collapsed on both datasets** — the table is
code-specific (near-orthogonal code vectors), bounded, and identity-
initialized, and it changed 35.7% / 87.2% of heterogeneity entries. On
Haptics the learned modulation stayed small (mean|s| = 0.011); on CWRU_BAL it
trained aggressively (max|s| = 0.73, effective response scaling 0.70–1.28).

## 11. VQ diagnostics (frozen R2 context)

| Statistic | Haptics | CWRU_BAL |
|---|---|---|
| active codes | 8/8 | 8/8 |
| normalized entropy | 0.935 | 0.977 |
| perplexity | 6.99 | 7.62 |
| dominant fraction | 0.236 | 0.184 |

Codebooks are healthy and evenly used — the regime source is not degenerate,
so the null result is not attributable to context collapse.

## 12. Control analysis

- **B1 == B0 bit-exactly** on both datasets: added capacity alone contributes
  nothing; the implementation sanity check holds.
- B3A/B3B (occupancy-matched random / shuffled codes through the *learned*
  table) land 5.0–4.9 pp below B2 on Haptics and 0.7–0.9 pp below on CWRU_BAL
  — consistent with all prior rounds: real regime structure carries useful
  heterogeneity signal, but that signal is already fully exploited by B0.
- The decisive comparison is B2 vs B0: the *learned* modulation adds nothing
  on Haptics and actively hurts on CWRU_BAL.

## 13. Cross-dataset interpretation

**Haptics — NO MECHANISTIC BENEFIT (B2 == B0).** With 155 train+val rows
against 9996 features, Ridge shrinkage (alpha = 4.28) nullifies the small
learned modulation: B0, B1, B2 produce identical test predictions despite the
conditioning being active. This reproduces the known n≪p dilution pattern
from the Hydra forensics in the same namespace family.

**CWRU_BAL — NEGATIVE.** The context model has ample data (3209 rows) and the
modulation trained hard (β = 0.406, 87% of H entries changed), yet B2 loses
0.52 pp to B0 and drops below the canonical M0 reference (0.9947). Near
ceiling (val MF1 = 1.0), the extra adaptation only adds variance the Ridge
must shrink; per the CWRU-specific caveat this is a small negative under a
near-ceiling baseline, but the direction is consistent across B2−B0 and
B2−B1, and B0 = 0.9982 exactly reproduces the frozen R2 result.

## 14. Which datasets show evidence of useful adaptive conditioning

**None.** On neither dataset does the evidence support the intended claim.
The mechanism behaves exactly as designed (bounded, code-specific, identity-
initialized, active), yet:
- Haptics: improvement is structurally impossible to express through Ridge
  shrinkage at n ≪ p (identical predictions).
- CWRU_BAL: enough data to express the modulation, and it degrades accuracy.

The scientifically honest summary: **adaptive discrete context-to-kernel
modulation of MiniROCKET responses does not improve on the unconditioned R2
heterogeneity representation on either screened dataset.** The regime
*partition* remains informative (controls confirm), but re-weighting kernel
responses per code adds no information beyond what H_m already extracts — and
where it has room to act, it subtracts.

## 15. Limitations

- Single seed (42); no variance estimates. The CWRU_BAL gap (−0.52 pp,
  ~3 test errors) is within plausible seed noise.
- Haptics' n ≪ p regime (155 rows) means the modulation pathway cannot be
  expressed by the final Ridge; the screen cannot distinguish "no signal" from
  "shrinkage-nullified signal" there.
- The modulation trains through a histogram surrogate (soft threshold-
  crossing rates over 25 interior bins, width 0.03) on fixed precomputed
  margins; the exact path is used for all evaluation. The surrogate is exact
  in the delta→0 limit and audited for orientation, but end-to-end gradient
  quality was not independently validated.
- K = 8 codes bound the conditioning vocabulary; the frozen R2 checkpoints
  fix the regime source (by design, for perfect control), so no re-training
  of encoder/VQ interacted with the modulation.
- CWRU_BAL is near ceiling: small negatives are weak evidence.

## 16. Reproduction commands

```bash
cd ECG_Benchmark
# full suite (22 tests, R2.2) + all conditioned-MiniROCKET suites (229 total)
python -m pytest tests/test_rcmkn_r2_2_adaptive_conditioning_seed42.py -q
python -m pytest tests/ -q -k "rcmkn or minirocket or drtn"

# official run (10 test evaluations; ~30 min total on 1x RTX 4050)
python -m experiments.rcmkn_r2_2_adaptive_conditioning_seed42.runner \
    --datasets Haptics CWRU_BAL

# figures
python -m experiments.rcmkn_r2_2_adaptive_conditioning_seed42.figures
```

Artifacts: `results/rcmkn_r2_2_adaptive_conditioning_seed42/` — REPORT.md,
report.json, per_dataset_results.json, config.json, predictions/,
diagnostics/{Haptics,CWRU_BAL}_audits.json, figures/ (6), per-dataset
modulation checkpoints (modulation_seed42.pt), tau tables, bias signs.
