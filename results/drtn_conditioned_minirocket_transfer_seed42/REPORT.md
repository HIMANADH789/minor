# DRTN-CONDITIONED MINIROCKET — SEED-42 CROSS-DATASET TRANSFER SCREEN

## Screening verdict per dataset (PROMISING / NEUTRAL / NEGATIVE / INVALID)

| Dataset | Verdict |
|---|---|
| ECG5000_BAL | **PROMISING** |
| EpilepticSeizures | NEUTRAL |
| Phoneme | NEUTRAL |
| CWRU_BAL | NEUTRAL |
| ECG5000_UNBAL | NEGATIVE |
| CWRU_UNBAL | NEGATIVE |

One dataset (ECG5000_BAL) shows the Haptics pattern (M1 > M0 AND M1 > M3 with meaningful margins). Two datasets show M1 < M0. This is a mixed single-seed screen, not a confirmation of generalization.

---

## 1. Objective

Determine whether the learned-regime conditioning effect confirmed on Haptics (3-seed: M1−M0 = +0.0319, M1−M3 = +0.0280, both 3/3 seeds positive) transfers beyond Haptics. Single seed (42), six benchmark datasets, four variants each, one test evaluation per variant. This is a screen, not a confirmation.

## 2. Exact protocol

- Seed 42 for every stochastic component (DRTN init/training, MiniROCKET kernel draw, control regimes).
- Canonical splits per dataset family:
  - external .ts (EpilepticSeizures, Phoneme): `experiments/external_stack_generalization/data.py` — provided canonical val for EpilepticSeizures; stratified 15% of train (seed 42) for Phoneme.
  - internal NPZ (ECG5000_*, CWRU_*): canonical benchmark split identical to `run_turs_skb.py`/`benchmark_baselines.py` — official X_train/X_test split (ECG5000) or stratified 85/15 (CWRU), val = stratified 15% of train (seed 42).
- Per-sample z-normalization (project-canonical formula) applied identically to all variants.
- Classifier: RidgeClassifierCV(alphas=np.logspace(-4, 4, 20)), fit on TRAIN+VAL features, test evaluated exactly once per variant.
- DRTN: existing R5 implementation (K=8, D=64, tau=0.5, EMA 0.99, beta 0.25, lam_div 0.01), trained on TRAIN only, checkpoint selected on VAL macro-F1, frozen before regime extraction. No architecture changes.

## 3. Datasets and splits (as loaded, canonical)

| Dataset | Train | Val | Test | T | Classes | DRTN val MF1 |
|---|---:|---:|---:|---:|---:|---:|
| EpilepticSeizures | 80 | 20 | 11,420 | 178 | 2 | 0.3333 |
| Phoneme | 185 | 29 | 1,896 | 1024 | 39 | 0.1408 |
| ECG5000_UNBAL | 3,400 | 600 | 1,000 | 140 | 5 | 0.6595 |
| ECG5000_BAL | 5,226 | 923 | 1,000 | 140 | 5 | 0.8736 |
| CWRU_UNBAL | 1,156 | 204 | 240 | 1024 | 4 | 0.8838 |
| CWRU_BAL | 2,727 | 482 | 567 | 1024 | 4 | 0.8719 |

## 4. Model definitions

- **M0**: canonical aeon MiniRocket(random_state=42), 9,996 PPV features.
- **M1**: 4,998 canonical global PPV + 4,998 DRTN-regime heterogeneity features.
- **M2**: identical to M1 but with random regimes (global occupancy-preserving draw, seed 42).
- **M3**: identical to M1 but with per-sample temporally shuffled DRTN regimes (exact per-sample histogram preserved).

Heterogeneity feature (unchanged validated formula, computed over each feature's canonical valid region):

    H_m = Σ_k q_k (PPV_{m,k} − PPV_m)²

with q_k = n_k/n_valid, PPV_{m,k} = activation rate of kernel m in regime k, PPV_m = global activation rate, and the 1%-of-T minimum-occupancy rule (weights renormalized over surviving regimes).

## 5. Feature-budget accounting

All four variants have exactly 9,996 features on every dataset (asserted in code). M1/M2/M3 global blocks are `array_equal` to the first 4,998 canonical features; the raw-activation PPV recomputation matched the aeon transform with max|diff| = 0.00e+00 on every dataset.

## 6. Master results table (test Macro-F1, seed 42)

| Dataset | M0 | M1 | M2 | M3 | M1−M0 | M1−M3 | M1−M2 | Category |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| EpilepticSeizures | 0.9194 | 0.9399 | 0.9397 | 0.9394 | +0.0205 | +0.0005 | +0.0002 | NEUTRAL |
| Phoneme | 0.0808 | 0.1119 | 0.1105 | 0.1088 | +0.0311 | +0.0031 | +0.0014 | NEUTRAL |
| ECG5000_UNBAL | 0.5938 | 0.5859 | 0.5716 | 0.5565 | −0.0079 | +0.0294 | +0.0143 | NEGATIVE |
| ECG5000_BAL | 0.6553 | **0.6764** | 0.6073 | 0.6159 | **+0.0211** | **+0.0605** | **+0.0691** | PROMISING |
| CWRU_UNBAL | 0.9917 | 0.9792 | 0.9625 | 0.9583 | −0.0125 | +0.0209 | +0.0167 | NEGATIVE |
| CWRU_BAL | 0.9947 | 0.9930 | 0.9824 | 0.9842 | −0.0017 | +0.0088 | +0.0106 | NEUTRAL |

## 7. Canonical M0 reproduction (integrity check)

All six datasets reproduce the repository's stored canonical MiniROCKET result **exactly** (delta +0.0000, tolerance 0.0011):

| Dataset | New M0 | Repository reference | Source artifact |
|---|---:|---:|---|
| EpilepticSeizures | 0.9194 | 0.9194 | results/external_stack_generalization |
| Phoneme | 0.0808 | 0.0808 | results/external_stack_generalization |
| ECG5000_UNBAL | 0.5938 | 0.5938 | results/baseline_bench |
| ECG5000_BAL | 0.6553 | 0.6553 | results/baseline_bench |
| CWRU_UNBAL | 0.9917 | 0.9917 | results/baseline_bench |
| CWRU_BAL | 0.9947 | 0.9947 | results/baseline_bench |

## 8. Selected Ridge alphas (train+val fit)

| Dataset | M0 | M1 | M2 | M3 |
|---|---:|---:|---:|---:|
| EpilepticSeizures | 0.1129 | 0.0168 | 0.0168 | 0.0168 |
| Phoneme | 206.91 | 29.76 | 29.76 | 29.76 |
| ECG5000_UNBAL | 11.29 | 4.28 | 4.28 | 4.28 |
| ECG5000_BAL | 1.62 | 1.62 | 1.62 | 1.62 |
| CWRU_UNBAL | 0.234 | 0.234 | 0.089 | 0.234 |
| CWRU_BAL | 0.616 | 0.234 | 0.234 | 0.234 |

## 9. Regime diagnostics (test split, frozen DRTN)

| Dataset | Active codes | Dominant fraction | Normalized entropy | Perplexity |
|---|---:|---:|---:|---:|
| EpilepticSeizures | 3 | 0.513 | 0.476 | 2.74 |
| Phoneme | 8 | 0.205 | 0.935 | 5.77 |
| ECG5000_UNBAL | 7 | 0.227 | 0.886 | 5.10 |
| ECG5000_BAL | 8 | 0.388 | 0.851 | 4.62 |
| CWRU_UNBAL | 8 | 0.205 | 0.969 | 7.19 |
| CWRU_BAL | 8 | 0.223 | 0.967 | 7.06 |

DRTN validation macro-F1 varies widely across datasets (0.14 on Phoneme to 0.88 on CWRU_UNBAL) — the DRTN is a regime provider, not a competitive classifier, consistent with its role.

## 10. Heterogeneity diagnostics (test split)

| Dataset | M1 mean | M1 nonzero frac | M1 max | M3 mean |
|---|---:|---:|---:|---:|
| EpilepticSeizures | 0.01421 | 0.954 | 0.164 | 0.01400 |
| Phoneme | 0.01923 | 0.963 | 0.130 | 0.01845 |
| ECG5000_UNBAL | 0.03550 | 0.870 | 0.231 | 0.03250 |
| ECG5000_BAL | 0.03152 | 0.859 | 0.229 | 0.02830 |
| CWRU_UNBAL | 0.01684 | 0.978 | 0.096 | 0.01500 |
| CWRU_BAL | 0.01762 | 0.976 | 0.104 | 0.01500 |

Features are not placeholders (85–98% nonzero), and M1 heterogeneity exceeds M3 slightly on every dataset — the learned regimes do modulate activations differently than shuffled ones, but the downstream benefit varies.

## 11. Error complementarity (M0 vs M1, test)

| Dataset | Both correct | M0-only | M1-only | Both wrong |
|---|---:|---:|---:|---:|
| EpilepticSeizures | 10,674 | 164 | 297 | 285 |
| Phoneme | 365 | 106 | 180 | 1,245 |
| ECG5000_UNBAL | 950 | 4 | 4 | 42 |
| ECG5000_BAL | 946 | 6 | 10 | 38 |
| CWRU_UNBAL | 234 | 4 | 1 | 1 |
| CWRU_BAL | 561 | 3 | 2 | 1 |

On the near-ceiling CWRU datasets M1 trades a few M0-only-correct samples for M1-only-correct ones (net negative). On ECG5000_BAL M1 nets +4 samples. On EpilepticSeizures/Phoneme M1 nets +133/+74 samples but the same gains appear in M2/M3.

## 12. Interpretation — the two-regime pattern

The screen reveals a sharp divide correlated with dataset structure:

**Sequence-structured datasets (transitions matter):**
- ECG5000_BAL: M1 (+0.0211 over M0) far above M2 (+0.0000 over M0) and M3 (−0.0394 below M1). Both controls sit clearly below M0 — temporal alignment of learned regimes is the active ingredient. This is the Haptics pattern.
- Haptics (previous experiment): M1 > M0 > M2 = M3, same signature.

**Near-ceiling fault datasets (CWRU_UNBAL/BAL):** M0 is already at 0.99+; the added heterogeneity features inject noise (M1 < M0) even though M1 still beats its own controls M2/M3. Conditioning cannot help when global PPV is already sufficient.

**EpilepticSeizures / Phoneme:** M1 > M0 clearly (+0.02/+0.03), but M2 ≈ M3 ≈ M1. Here any 8-way partition of the time axis adds useful variance features; the benefit is from partitioning itself, not learned alignment. (Note Phoneme has 39 classes and a near-chance M0 of 0.08 — the least reliable readout.)

M1 > M3 holds on all six datasets (mean +0.0203), but with very different margins: strong on ECG5000_BAL (+0.0605), negligible on EpilepticSeizures (+0.0005) and Phoneme (+0.0031).

## 13. Fairness / reproducibility audit (all enforced in code, per dataset)

- ✅ M0 = canonical aeon MiniRocket(random_state=42); raw-activation PPV matched the aeon transform with max|diff| = 0.00e+00 on all six datasets
- ✅ Exactly 9,996 features for all 24 evaluations (asserted)
- ✅ Global block of M1/M2/M3 `array_equal` to canonical features
- ✅ Heterogeneity independently recomputed for 12 random sample/kernel pairs per dataset: max abs error ≤ 6.7e-9 (float32 matmul vs float64 reference; a placeholder bug would give ~1e-2)
- ✅ Non-placeholder check: nonzero fraction 0.86–0.98
- ✅ M2 global occupancy preserved (<0.02 max abs diff); deterministic (exact rerun equality)
- ✅ M3 per-sample histogram exactly preserved (max abs diff 0.0); temporal alignment changed in >90% of samples
- ✅ DRTN frozen: state-dict equality verified across regime extraction
- ✅ Test labels touched only in the single final evaluation per variant
- ✅ Variant-local F_fit/F_va/F_te discipline (guards the historical Fte/F_te shadowing bug)
- ✅ Sample ordering preserved end-to-end (predictions.csv indexed by original order)

## 14. Unit-test results

`tests/test_drtn_conditioned_minirocket_transfer_seed42.py`: **17/17 passed**, covering heterogeneity zero-when-identical, positive-when-different, independent recomputation agreement, non-placeholder behavior, M2 determinism + occupancy preservation, M3 histogram preservation + alignment destruction, occupancy stats, canonical feature budget + raw-extractor/aeon identity on Haptics, all six dataset configurations with expected T, and no-label-path structural check.

## 15. Smoke-test status

`--smoke` run on ECG5000_UNBAL (reduced DRTN epochs, no checkpoint save): end-to-end pass, exact M0 reproduction (0.5938), all audits green, category computed. (Smoke results were not mixed with official results; official ECG5000_UNBAL was run fresh with full DRTN training.)

## 16. Bugs discovered

1. **Canonical valid-region convention (caught by test before any official run)**: the first raw-extractor draft averaged activations over all T timesteps, while aeon's canonical PPV for `_padding1==1` features averages over the interior `[padding : T-padding]` only. Detected by the unit test comparing against the aeon transform (max diff 0.99); fixed by tracking per-feature valid masks so Block A is bit-identical to canonical features. **No official test evaluation was performed before the fix** — the smoke test caught it during implementation.
2. **Train+val fit shape mismatch** (smoke run): the classifier initially fit on train features only while passing train+val labels. Fixed by assembling per-variant F_fit = vstack(F_tr, F_va). Again caught before any official evaluation.
3. Audit tolerance for independent recomputation set to 1e-7 to accommodate float32 vectorized matmul vs float64 reference (actual observed ≤ 6.7e-9).

No test results were invalidated: every official number in this report comes from a single clean run per dataset after the fixes above. The official run for each dataset was executed once (external pair, then the four NPZ datasets), each with test evaluated exactly once per variant.

## 17. Limitations

- Single seed, single run per dataset: no significance claims; category labels are screening heuristics.
- DRTN trained once per dataset with the Haptics-tuned configuration (no per-dataset tuning, per spec) — DRTN quality varies (val MF1 0.14–0.88), which confounds the M1 readout with DRTN quality.
- Heterogeneity is one of several possible regime-conditioned statistics.
- Phoneme's 39-class/185-train setting makes macro-F1 extremely noisy.

## 18. Follow-up seed policy (per spec, not executed)

Per the established precedent, only ECG5000_BAL shows the confirmable pattern (M1 > M0 AND M1 >> M2/M3). If multi-seed confirmation is pursued next, it should use the repository's established seed family for that dataset and pre-register the same M0–M3 protocol. EpilepticSeizures and Phoneme showed M1 > M0 gains that did not survive the M3 control and should not be prioritized for confirmation without a specific reason. CWRU datasets are near ceiling; the mechanism is not promising there.

---

## Artifacts

- `results/drtn_conditioned_minirocket_transfer_seed42/report.json` (consolidated, 6 datasets)
- `results/drtn_conditioned_minirocket_transfer_seed42/config.json`
- `results/drtn_conditioned_minirocket_transfer_seed42/REPORT.md` (this file)
- `results/drtn_conditioned_minirocket_transfer_seed42/<Dataset>/result.json`
- `results/drtn_conditioned_minirocket_transfer_seed42/<Dataset>/predictions.csv`
- `results/drtn_conditioned_minirocket_transfer_seed42/<Dataset>/diagnostics/{regime_statistics,feature_statistics,complementarity}.json`
- `results/drtn_conditioned_minirocket_transfer_seed42/<Dataset>/drtn_R5_seed42_checkpoint.pt` (frozen per-dataset DRTN)
- `results/drtn_conditioned_minirocket_transfer_seed42/figures/{performance_comparison,delta_M1_M0,delta_M1_M3,heterogeneity_diagnostic}.png`

**STOP**: per spec, no multi-seed confirmation was launched.
