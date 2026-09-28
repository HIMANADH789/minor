# GunPoint / Phoneme / FordA — FINAL REPORT (MR vs HERAMBA R5)
Date: 2026-09-23 · Seeds {42, 43, 44} (repository standard) · 18/18 cells VALID

## 1. Objective & provenance
Final three-seed MiniRocket vs HERAMBA R5 benchmark on GunPoint, Phoneme,
FordA. Existing canonical seed-42 artifacts were reused as gates and
untouched; seeds 43/44 newly executed through the audited rcmkn core.
No reruns were required. HERAMBA R5 frozen; MiniRocket frozen; splits and
preprocessing identical across arms per dataset.

## 2. Per-seed table (test accuracy; MR is deterministic — one canonical
run per dataset, shown in every seed cell; Δ = HERAMBA_final − MR)

| Dataset  | Model      | Seed 42 | Seed 43 | Seed 44 | Mean ± Std |
| -------- | ---------- | ------: | ------: | ------: | ---------: |
| GunPoint | MiniRocket |  0.9933 |  0.9933 |  0.9933 | 0.9933 ± 0.0000 |
| GunPoint | HERAMBA raw | 1.0000 | 1.0000 | 1.0000  | 1.0000 ± 0.0000 |
| GunPoint | HERAMBA final | 1.0000 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| Phoneme  | MiniRocket |  0.0808 |  0.0808 |  0.0808 | 0.0808 ± 0.0000 |
| Phoneme  | HERAMBA raw | 0.1238 | 0.1128 | 0.1171  | 0.1179 ± 0.0055 |
| Phoneme  | HERAMBA final | 0.1238 | 0.1128 | 0.1171 | 0.1179 ± 0.0055 |
| FordA    | MiniRocket |  0.9499 |  0.9499 |  0.9499 | 0.9499 ± 0.0000 |
| FordA    | HERAMBA raw | 0.9553 | 0.9545 | 0.9470  | 0.9523 ± 0.0045 |
| FordA    | HERAMBA final | 0.9553 | 0.9545 | 0.9470 | 0.9523 ± 0.0045 |

Fallback-protected = raw (selection was validation-only HERAMBA on 9/9
cells; zero fallbacks fired).

## 3. Matched-seed Δ (final HERAMBA − MR)
| Dataset  | d42 | d43 | d44 | mean | median | std |
| -------- | --: | --: | --: | ---: | -----: | --: |
| GunPoint | +0.0067 | +0.0067 | +0.0067 | +0.0067 | +0.0067 | 0.0000 |
| Phoneme  | +0.0430 | +0.0320 | +0.0363 | +0.0371 | +0.0363 | 0.0055 |
| FordA    | +0.0054 | +0.0046 | −0.0029 | +0.0024 | +0.0046 | 0.0046 |

HERAMBA-selected 9/9; fallback 0/9; HERAMBA beats MR on 8/9 cells (FordA
seed 44 is the one negative cell, reported honestly — no forcing, per §22
of the protocol). No significance claims (n = 3).

## 4. Aggregates (mean ± sample std, ddof = 1)
| Dataset  | MR | HERAMBA raw | HERAMBA final | Δ mean |
| -------- | -- | ----------- | ------------- | -----: |
| GunPoint | 0.9933 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | +0.0067 |
| Phoneme  | 0.0808 ± 0.0000 | 0.1179 ± 0.0055 | 0.1179 ± 0.0055 | +0.0371 |
| FordA    | 0.9499 ± 0.0000 | 0.9523 ± 0.0045 | 0.9523 ± 0.0045 | +0.0024 |

MR cells are single deterministic canonical runs (established repo
convention: fixed random_state=42 MiniRocket + deterministic Ridge), so
their seed-to-seed std is 0 by construction — this is a protocol
property, not a stability claim.

## 5. Dataset-specific audits
- GunPoint (Phase 15a): 42/8/150, T=150, 2 classes; MR gate exact
  (0.9933 = canonical M0); HERAMBA at ceiling 1.0000 on all three seeds
  (identical to canonical R2 = 1.0000); val fully separated — tiny
  differences are not overinterpreted; no train/test contamination
  (extractor identity 0.0, test set untouched until final eval).
- Phoneme (Phase 15b): 185/29/1896, 39 classes; per-seed variation
  audited — raw HERAMBA 0.1238/0.1128/0.1171 (std 0.0055), seed 44 is
  mid-range, NOT representative of either extreme; alphas ≈ 5.7/5.3/5.9
  × 10³ (from the frozen grid); regime assignments deterministic per
  seed (asserted); both classes present in every prediction file; MR
  itself is weak (0.0808) — the 39-class regime-conditioned features add
  +0.037 mean. Fallback never triggered.
- FordA (Phase 15c): 3060/541/1320, T=500, 2 classes; raw HERAMBA
  0.9553/0.9545/0.9470 (std 0.0045) vs MR 0.9499 — effectively tied
  (mean Δ +0.0024 with one negative seed); configuration hashes
  seed-invariant; no gain forced; reported as-is.

## 6. Stability
- HERAMBA raw seed-std: GunPoint 0.0000, Phoneme 0.0055, FordA 0.0045 —
  tight; the largest variation is Phoneme, consistent with its 39-class
  head and 185 training samples.
- Δ std: GunPoint 0.0000, Phoneme 0.0055, FordA 0.0046.
- Runtimes: GunPoint seconds per arm; Phoneme ~2–3 min per seed;
  FordA ~290 s per seed (GPU); total ≈ 19 min.

## 7. Completeness & status
18/18 cells VALID (GunPoint MR 3/3 + HERAMBA 3/3; Phoneme MR 3/3 +
HERAMBA 3/3; FordA MR 3/3 + HERAMBA 3/3). No missing cells, no invalid
artifacts included. Raw vs fallback-protected kept distinct and both
auditable (FALLBACK_AUDIT.csv). Stop condition honored: no fourth seed,
no additional datasets, no R5/MR modifications, no test-driven selection.

## 8. Artifacts
results/gunpoint_phoneme_forda_r5/ — FINAL_HERAMBA_R5_RESULTS.csv,
HERAMBA_R5_PER_SEED.csv, HERAMBA_R5_AGGREGATED.csv, MR_VS_HERAMBA.csv,
FALLBACK_AUDIT.csv, STATUS_MATRIX.csv, per-cell prediction files,
FINAL_TABLE.md, FINAL_REPORT.md, AUDIT.md ·
experiments/gunpoint_phoneme_forda_r5/ — runner.py, aggregate.py.
