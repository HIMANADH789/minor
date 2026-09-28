# STAGE A FORENSIC AUDIT — Haptics M2 == M3 Anomaly

**Status: RESOLVED. Two spec deviations found and fixed; full 3-seed rerun completed; previous Haptics M1/M2/M3 headline results invalidated and replaced.**

---

## 1. Forensic finding (A4 cases adjudicated)

The audit reconstructed, for every seed (42/43/44), the actual regime arrays at
each pipeline checkpoint (C0 DRTN labels → C1 M2 construction → C2 M3
construction → C3 pre-feature input → C4 heterogeneity features → C5
classifier input → C6 predictions).

**CASE 1 (identical arrays): RULED OUT.**
M2 and M3 regime arrays have different SHA-1 hashes for all seeds
(`regime_audit_summary.json`; e.g. seed 42 test: M2 `f2c338e2...`,
M3 `3906836d...`). No aliasing: `np.shares_memory(m2, m3) == False` for all
seeds (Cases 3/7 ruled out). M3's shuffle IS passed into the heterogeneity
builder (Case 4 ruled out — verified in `run_one_seed` wiring and re-derived
features). M2 is NOT derived from M3's output (Case 6 ruled out — different
constructions, different hashes). No axis errors (Case 8 — shapes (N, T)
verified). No restoration step (Case 9 — dataflow traced). No slicing bug
(Case 10 — feature-axis slicing verified).

**CASE 2 CONFIRMED + quantified — the actual root cause:**
M2/M3 arrays and features genuinely differ, but under alignment destruction
both controls collapse to a binomial-noise floor that Ridge cannot distinguish.

Evidence (feature-level audit, 5 test samples × 4998 kernels, seed 42):

| Variant | mean H | max H | zero fraction |
|---|---|---|---|
| M1 (DRTN regimes)  | 3.22e-3 | 1.57e-1 | 0.832 |
| M2 (random regimes)| 1.10e-4 | 4.7e-3  | 0.832 |
| M2spec (per-sample)| 0.96e-4 | 3.8e-3  | 0.832 |
| M3 (shuffled)      | 1.01e-4 | 4.2e-3  | 0.832 |

- The apparent "exact equality" was **entirely the shared zero block**:
  20,790 / 20,790 identical entries were both exactly 0 (kernels whose
  activation is regime-independent have H = 0 under ANY regime assignment).
  Where either feature is nonzero, corr(H2, H3) = 0.073 — genuinely different.
- Under a shuffled/random assignment, PPV_{m,k} deviates from PPV_m only by
  multinomial sampling noise ≈ p(1−p)/n_k, which is ~30× smaller than the
  aligned-regime heterogeneity. Ridge's decision margins exceed these
  ~1e-4-scale differences → 308/308 identical predictions and identical
  selected alphas (4.2813) per seed in the original run.

## 2. Spec deviations found and fixed (A11)

**DEV-1 (M2 definition, A6 / DOC-10 item 4).** The original M2 drew iid labels
from the GLOBAL occupancy histogram: per-sample occupancy was NOT preserved
(audit: `m2current_per_sample_occupancy_preserved_* == false` for all seeds).
**Fix:** `create_random_regime_control` now generates, per sample, a random
assignment with exactly the per-sample occupancy histogram (positions drawn
without replacement per regime), uniform over all count-preserving
arrangements, using an independent RNG stream.

**DEV-2 (valid-region convention, DOC-10 item 9 / BUG GUARD 5).** The original
heterogeneity computed activation rates over ALL T positions including aeon's
zero-padded region (responses tensor zero-filled outside the valid slice,
padded positions counted as non-activations).
**Fix:** the runner now uses the validated transfer-core implementation
(`heterogeneity_features`, `compute_raw_activations`, `ppv_from_activations`)
which computes all rates over each feature's valid region
`[padding, T − padding)` only. The legacy `compute_raw_minirocket_features`
was removed entirely.

**RNG hardening (A7).** M2 now uses `RandomState(seed + 900001)` and M3 uses
`RandomState(seed + 900002)` — documented, disjoint streams. In the original
code both were `RandomState(seed)` (same stream, though the constructions
still produced different arrays).

## 3. Corrected behavior (verified in the official rerun)

Per-seed in-run audit evidence (results/drtn_conditioned_minirocket_haptics_3seed/seed{42,43,44}/result.json):

- M2 vs M3 test arrays differ; no shared memory.
- M2 vs M3 heterogeneity features: max|d| = 0.556, only 7.4–7.6% of entries
  exactly equal (the shared-zero block) — previously 83.2% equal with
  max|d| ≈ 5e-3.
- M2 vs M3 test predictions now differ on **49/308, 60/308, 54/308** samples
  (seeds 42/43/44) — previously 0/308 on every seed.
- Per-sample occupancy preserved for both controls on every split (asserted).
- Independent H_m recomputation agrees to ≤ 1.3e-9 (float32 accumulation vs
  float64 reference; wrong-formula scale ≥ 1e-6).
- Raw-extractor PPV vs aeon transform: max|diff| = 0.00e+00 on every seed.

## 4. Corrected Haptics 3-seed results (A12)

| Seed | M0 | M1 | M2 | M3 | M1−M0 | M1−M3 | M1−M2 |
|---|---|---|---|---|---|---|---|
| 42 | 0.4974 | 0.5366 | 0.4830 | 0.5132 | +0.0392 | +0.0234 | +0.0536 |
| 43 | 0.4920 | 0.5285 | 0.4666 | 0.4827 | +0.0365 | +0.0458 | +0.0619 |
| 44 | 0.4950 | 0.5370 | 0.4925 | 0.5038 | +0.0420 | +0.0332 | +0.0445 |
| **mean** | **0.4948** | **0.5340** | **0.4807** | **0.4999** | **+0.0392** | **+0.0341** | **+0.0533** |
| std | 0.0022 | 0.0039 | 0.0107 | 0.0128 | 0.0022 | 0.0092 | 0.0071 |

- Δ(M1−M0): +0.0392 ± 0.0022 (se 0.0013), positive 3/3 seeds.
- Δ(M1−M3): +0.0341 ± 0.0092 (se 0.0053), positive 3/3 seeds.
- Δ(M1−M2): +0.0533 ± 0.0071 (se 0.0041), positive 3/3 seeds.
- M0 is UNCHANGED (0.4974 / 0.4920 / 0.4950) — the canonical aeon path was
  never touched.
- **The previous headline (M1 = 0.5267 mean, M2 = M3 = 0.4987) is INVALIDATED
  and replaced by the corrected numbers above.** The corrected result is
  STRONGER: both controls now fall clearly below M0, and the
  learned-alignment margin (M1−M3) increased from +0.0280 to +0.0341.
  M0 must still reproduce exactly: **PASS**.

## 5. DOC-10 checklist (on the fixed implementation)

| Item | Verdict | Evidence |
|---|---|---|
| 1 — Feature identity | **PASS** | Raw-extractor PPV vs aeon transform max\|diff\| = 0.00e+00 on every seed/split; global block of M1/M2/M3 asserted equal to M0's (`np.array_equal`); Haptics M0 reproduces 0.4974/0.4920/0.4950 exactly. |
| 2 — Heterogeneity from raw responses | **PASS** | Input trace: `compute_raw_activations` (bool per-timestep indicators) → `heterogeneity_features`; no PPV-pooled, DRTN-embedding, or logits inputs. Independent O(T) recompute at (sample 0, kernel 0) and (sample 5, kernel 1234) agrees ≤ 1.3e-9 per seed (values in each result.json `feature_check.independent_recompute`). |
| 3 — Matched kernel allocation | **PASS** | M0 = 9996; M1/M2/M3 = 4998 global + 4998 hetero (asserted per variant per split). Same `act[:, N_GLOBAL:, :]` FEATURE-axis slice for all variants (BUG GUARD 1 asserted by shape and by the equal global block check). |
| 4 — Regime occupancy preservation | **PASS** | Per-sample histogram equality asserted for M2 and M3 on trainva and test for all seeds (runner assertions); audit confirmed per-sample preservation for M3 and for the FIXED M2 (the original M2 failed this check — see DEV-1). |
| 9 — Exact formula + valid region | **PASS** | Worked numerical example (tests/test_drtn_conditioned_minirocket_haptics_audit.py::test_worked_numerical_example_independent_recompute): manual float64 recomputation of H_m from raw responses agrees ≤ 1e-8; padded-region labels/activations provably change nothing (test_heterogeneity_uses_valid_region_only, test_worked_example_padded_positions_excluded). Formula uses q_k-weighted squared deviations of per-regime PPV from global PPV with min-occupancy exclusion + renormalization — not the mean/unweighted/global-PPV variants. |

## 6. Test results

- `tests/test_drtn_conditioned_minirocket_haptics_audit.py`: 19 passed
- `tests/test_drtn_conditioned_minirocket_3seed.py`: 9 passed
- `tests/test_drtn_conditioned_minirocket_transfer_seed42.py`: 17 passed
- `tests/test_drtn_conditioned_minirocket.py`: 16 passed
- Total: **61 passed** across the conditioned-MiniROCKET suites.

## 7. Artifacts

- `results/drtn_conditioned_minirocket_haptics_audit/`
  - `regime_audit_summary.json`, `feature_audit_summary.json`
  - `seed{42,43,44}/audit_result.json`, `feature_audit.json`,
    `m2_regimes_test.npy`, `m2spec_regimes_test.npy`, `m3_regimes_test.npy`,
    `drtn_regimes_test.npy`, `feature_audit_slice.npz`
  - `INVALIDATION_NOTE.json`
- `results/drtn_conditioned_minirocket_haptics_3seed/seed{42,43,44}/`
  (corrected official rerun: `result.json`, `predictions.csv`)
- `results/drtn_conditioned_minirocket_haptics_3seed/report.json`

## 8. Reproducibility

```
cd ECG_Benchmark
python -m pytest tests/test_drtn_conditioned_minirocket_haptics_audit.py -q
python -m experiments.drtn_conditioned_minirocket_haptics_audit.audit_regimes
python -m experiments.drtn_conditioned_minirocket_haptics_audit.audit_features
python -m experiments.drtn_conditioned_minirocket_haptics_3seed.runner --seeds 42 43 44 --skip-drtn-train
```
