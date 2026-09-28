# GUNPOINT / PHONEME / FORDA — FINAL 3-SEED VALIDATION

**Final status: VALID — 18/18 cells audited.**

Canonical seed set: **{42, 43, 44}** (repository standard)

## Reused existing seeds
- MR: none were stored per-seed — MR is deterministic in this repo
  (MiniRocket random_state=42 + deterministic Ridge). The canonical
  stored M0 values are reproduced exactly as gates and shown in every
  seed cell: GunPoint 0.9933, Phoneme 0.0808, FordA 0.9499.
- HERAMBA R5: seed 42 canonical artifacts (GunPoint R2 = 1.0000,
  Phoneme R2 = 0.1182, FordA R2 = 0.9560) kept untouched as gate
  references. Fresh per-seed-context runs reproduce them within
  tolerance (GunPoint exact; Phoneme/FordA seed-level context variation
  within the repo's 0.011 tolerance).

## Newly executed seeds
- MiniRocket: none required (deterministic convention, gates verified).
- HERAMBA R5: 43 and 44 on all three datasets (6 new arms) plus seed-42
  gated runs per the repo's 3-seed convention.

## Reruns
None — no existing artifact was invalid.

## Final per-seed table (test accuracy)

| Dataset  | Model         | 42      | 43      | 44      | Mean ± Std      |
| -------- | ------------- | ------: | ------: | ------: | --------------- |
| GunPoint | MR            | 0.9933  | 0.9933  | 0.9933  | 0.9933 ± 0.0000 |
| GunPoint | HERAMBA raw   | 1.0000  | 1.0000  | 1.0000  | 1.0000 ± 0.0000 |
| GunPoint | HERAMBA final | 1.0000  | 1.0000  | 1.0000  | 1.0000 ± 0.0000 |
| Phoneme  | MR            | 0.0808  | 0.0808  | 0.0808  | 0.0808 ± 0.0000 |
| Phoneme  | HERAMBA raw   | 0.1238  | 0.1128  | 0.1171  | 0.1179 ± 0.0055 |
| Phoneme  | HERAMBA final | 0.1238  | 0.1128  | 0.1171  | 0.1179 ± 0.0055 |
| FordA    | MR            | 0.9499  | 0.9499  | 0.9499  | 0.9499 ± 0.0000 |
| FordA    | HERAMBA raw   | 0.9553  | 0.9545  | 0.9470  | 0.9523 ± 0.0045 |
| FordA    | HERAMBA final | 0.9553  | 0.9545  | 0.9470  | 0.9523 ± 0.0045 |

Fallback-protected = raw on all cells: validation-stage selection chose
HERAMBA 9/9; **fallback count 0/9** (and never test-based).

## Per-seed Δ (HERAMBA final − MR)
| Dataset  | d42 | d43 | d44 | mean | median | std |
| -------- | --: | --: | --: | ---: | -----: | --: |
| GunPoint | +0.0067 | +0.0067 | +0.0067 | +0.0067 | +0.0067 | 0.0000 |
| Phoneme  | +0.0430 | +0.0320 | +0.0363 | +0.0371 | +0.0363 | 0.0055 |
| FordA    | +0.0054 | +0.0046 | −0.0029 | +0.0024 | +0.0046 | 0.0046 |

Per-dataset Δ: GunPoint **+0.0067**, Phoneme **+0.0371**, FordA
**+0.0024**. HERAMBA-selected **9/9**; HERAMBA > MR on 8/9 cells
(FordA seed 44 negative, reported honestly). No significance claims
from n = 3.

## Dataset audits
- **GunPoint**: near-ceiling (42/8/150); MR gate exact; HERAMBA 1.0000
  on all seeds (= canonical); no contamination signals; differences not
  overinterpreted.
- **Phoneme** (seed-variance audit): raw spread 0.1128–0.1238
  (std 0.0055); seed 44 is mid-range, not an outlier representative;
  alphas ≈ 5.3–5.9 × 10³ from the frozen grid; regimes deterministic
  per seed; 39-class head benefits most (+0.037).
- **FordA**: tied within seed noise (Δ mean +0.0024, one negative
  seed); config hashes seed-invariant; stability good (std 0.0045);
  no gain forced.

## Artifacts
```text
results/gunpoint_phoneme_forda_r5/
├── FINAL_HERAMBA_R5_RESULTS.csv
├── HERAMBA_R5_PER_SEED.csv
├── HERAMBA_R5_AGGREGATED.csv
├── MR_VS_HERAMBA.csv
├── FALLBACK_AUDIT.csv
├── STATUS_MATRIX.csv
├── FINAL_TABLE.md
├── FINAL_REPORT.md
├── AUDIT.md
└── predictions/gunpoint|phoneme|forda_{mr,heramba}_seed{42,43,44}.npy
experiments/gunpoint_phoneme_forda_r5/{runner.py, aggregate.py}
```

## Hard-stop checks
No wrong dataset/split/seed detected; no configuration drift (single
frozen R2 recipe, seed-dependent fields only); no test leakage (fallback
selection on validation only, raw-HERAMBA test scores recorded but never
used for selection, no max() over test); HERAMBA R5 and MiniRocket
frozen; no artifacts missing after execution.
