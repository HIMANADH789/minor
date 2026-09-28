# CORRECTED DRTN-CONDITIONED MINIROCKET — NEGATIVE-DATASET RE-TEST (seed 42)

Re-test of the two datasets classified **NEGATIVE** in the pre-audit 6-dataset
transfer screen (`drtn_conditioned_minirocket_transfer_seed42`), using the
**audited Stage-A implementation** (the corrected Haptics 3-seed runner semantics
that produced the corrected Haptics and ECG5000_BAL confirmations).

## 1. Why the old negatives are no longer sufficient

The old transfer screen violated two audited spec requirements:

1. **M2** drew regime IDs iid from the *global* occupancy histogram instead of
   preserving occupancy **per sample** (audit check failed on every seed).
2. Heterogeneity `H_m` averaged activation rates over **all T positions**,
   including aeon's zero-padded region, instead of the canonical valid region
   `[padding, T-padding)`.

Both affect the controls and the heterogeneity block directly, so the old
M1-vs-M2/M3 comparisons on these datasets were not scientifically interpretable.
This re-test therefore re-runs ECG5000_UNBAL and CWRU_UNBAL under the audited
implementation with the test-once policy (8 official test evaluations total).

## 2. Pre-test audit results (both datasets, all green)

| Audit | Evidence |
|---|---|
| 1. Canonical M0 identity | raw-extractor PPV vs aeon transform: **max\|diff\| = 0.00e+00** on both datasets |
| 2. Raw response path | inputs traced to per-timestep bool activations; independent float64 recompute of PPV_m, PPV_{m,k}, q_k, H_m: **max diff ≤ 4.3e-9** (wrong-formula scale ≥ 1e-6) |
| 3. Feature allocation | M0/M1/M2/M3 all **9996 = 4998 + 4998**; global blocks `array_equal` to M0's; ranges printed: global [0, 4998), het [4998, 9996) |
| 4. Occupancy preservation | per-sample histogram equality for M2 and M3 on every split: **0 failures** |
| 5. M2/M3 array difference | distinct SHA hashes (e.g. ECG5000_UNBAL test: `d58c72ec…` vs `d73d103a…`, 108,748/140,000 positions differ); `np.shares_memory = False` |
| 6. Valid-region invariance | flipping ALL out-of-valid activations (per-feature masks; 1.70e11 cells on ECG5000_UNBAL): **H unchanged exactly** |
| 7. Exact H formula | `H_m = Σ_k q_k (PPV_{m,k} − PPV_m)²` verified by independent recompute (audit 2) |
| 8. Train+validation fit | every Ridge fit on train+val (`ytrva`); validation used only for alpha selection; test touched exactly once per variant |

Canonical M0 gates: **PASS** on both datasets (0.5938 and 0.9917, exact).

## 3. Corrected seed-42 results (test Macro-F1)

### ECG5000_UNBAL (train 3400 / val 600 / test 1000, T=140, 5 classes)

| Variant | val MF1 | test MF1 | alpha |
|---|---|---|---|
| M0 | 0.8224 | **0.5938** | 11.2884 |
| M1 | 0.7153 | 0.5859 | 4.2813 |
| M2 | 0.6846 | 0.5547 | 4.2813 |
| M3 | 0.6846 | 0.5799 | 4.2813 |

Deltas: M1−M0 = **−0.0079**, M1−M2 = +0.0312, M1−M3 = +0.0060, M2−M3 = −0.0252.

Per-class F1 (M0): [0.9949, 0.9494, 0.5161, 0.5085, 0.0] — the last class has **zero test support** in Macro-F1 terms (F1 = 0 for every variant; the canonical convention is unchanged).

### CWRU_UNBAL (train 1156 / val 204 / test 240, T=1024, 4 classes)

| Variant | val MF1 | test MF1 | alpha |
|---|---|---|---|
| M0 | 1.0000 | **0.9917** | 0.2336 |
| M1 | 1.0000 | 0.9792 | 0.2336 |
| M2 | 1.0000 | 0.9624 | 0.2336 |
| M3 | 1.0000 | 0.9708 | 0.0886 |

Deltas: M1−M0 = **−0.0125**, M1−M2 = +0.0168, M1−M3 = +0.0084, M2−M3 = −0.0084.

## 4. OLD vs CORRECTED comparison (seed 42, test Macro-F1)

| Dataset | Variant | OLD (pre-audit) | CORRECTED | Δ |
|---|---|---|---|---|
| ECG5000_UNBAL | M0 | 0.5938 | 0.5938 | +0.0000 |
| ECG5000_UNBAL | M1 | 0.5859 | 0.5859 | +0.0000 |
| ECG5000_UNBAL | M2 | 0.5716 | 0.5547 | −0.0169 |
| ECG5000_UNBAL | M3 | 0.5565 | 0.5799 | +0.0234 |
| CWRU_UNBAL | M0 | 0.9917 | 0.9917 | +0.0000 |
| CWRU_UNBAL | M1 | 0.9792 | 0.9792 | +0.0000 |
| CWRU_UNBAL | M2 | 0.9625 | 0.9624 | −0.0001 |
| CWRU_UNBAL | M3 | 0.9583 | 0.9708 | +0.0125 |

M0 and M1 are unchanged (they never depended on the two fixed code paths).
The corrections moved only the controls (M2/M3), as expected: M3 now scores
*higher* than before on both datasets, narrowing M1's margin over M3.

## 5. Error complementarity M0 vs M1 (descriptive only)

- ECG5000_UNBAL: both correct 950, M0-only 4, M1-only 4, both wrong 42 (n=1000) — near-symmetric disagreement.
- CWRU_UNBAL: both correct 234, M0-only 4, M1-only 1, both wrong 1 (n=240) — M0's errors are a strict superset here.

## 6. Mechanistic note (diagnostic, not tuned on)

Regime diagnostics (train): 7/8 active codes, normalized entropy 0.884,
perplexity 6.29, dominant fraction 0.228, code 4 unused (ECG5000_UNBAL);
CWRU_UNBAL similar. Heterogeneity nonzero fractions are high for all variants
(0.87–0.98), so the features are not degenerate — but on these datasets the
alignment-specific signal in M1 does not convert into better test decisions.

## 7. Verdict

| Dataset | Classification | Reason |
|---|---|---|
| ECG5000_UNBAL | **CORRECTED NEGATIVE** (was NEGATIVE) | M1 ≤ M0 (−0.0079); M1 > M3 only marginally (+0.0060) with the corrected, *stronger* M3 |
| CWRU_UNBAL | **CORRECTED NEGATIVE** (was NEGATIVE) | M1 ≤ M0 (−0.0125) near ceiling (M0 = 0.9917); heterogeneity features inject noise |

**Neither dataset reverses under the audited implementation.** The corrected
controls are stronger (M3 improved on both datasets), so the old results were
if anything *optimistic* about M1's margin — the re-test confirms the negatives
rather than reversing them.

### Context (descriptive only, no pooling)

| Dataset | M0 | M1 | M2 | M3 |
|---|---|---|---|---|
| Haptics corrected 3-seed | 0.4948 ± 0.0022 | 0.5340 ± 0.0039 | 0.4807 ± 0.0107 | 0.4999 ± 0.0128 |
| ECG5000_BAL corrected 3-seed | 0.6498 ± 0.0044 | 0.6653 ± 0.0069 | 0.6171 ± 0.0101 | 0.6288 ± 0.0201 |
| ECG5000_UNBAL corrected seed-42 | 0.5938 | 0.5859 | 0.5547 | 0.5799 |
| CWRU_UNBAL corrected seed-42 | 0.9917 | 0.9792 | 0.9624 | 0.9708 |

## 8. Future multi-seed confirmation?

**No.** Per the follow-up seed policy, neutral/negative screening datasets do
not receive expensive multi-seed confirmation absent a specific scientific
reason. Neither dataset shows a positive M1 signal that a 3-seed run could
confirm.

## 9. Reproduction commands

```bash
cd ECG_Benchmark
# official re-test (8 test evaluations, seed 42)
python -m experiments.drtn_conditioned_minirocket_corrected_negative_retest.runner
# figures
python -m experiments.drtn_conditioned_minirocket_corrected_negative_retest.figures
# tests
python -m pytest tests/test_drtn_conditioned_minirocket_corrected_negative_retest.py -q
```

DRTN checkpoints: frozen, loaded from
`results/drtn_conditioned_minirocket_transfer_seed42/<dataset>/` (train-only
selection, val MF1 0.6595 @ epoch 18 for ECG5000_UNBAL; see per-dataset
`result.json`).

## Test-evaluation log

Total official test evaluations this experiment: **8** (2 datasets × 4 variants,
each exactly once). Two earlier launch attempts failed **before any test
evaluation** (a runner bug: train-only control arrays vs train+val activations;
then two bugs in the audit-6 harness itself: a boolean-index shape error and a
per-feature padding premise). No result was invalidated; the test-once policy
held for all 8 evaluations in the final clean run.
