# GunPoint / Phoneme / FordA — AUDIT
Date: 2026-09-23. Final verdict: 18/18 cells VALID.

## Initial status matrix (Phase 3, before runs)

| Dataset  | Model      | Seed 42 | Seed 43 | Seed 44 |
| -------- | ---------- | ------- | ------- | ------- |
| GunPoint | MiniRocket | VALID (canonical M0 0.9933) | MISSING | MISSING |
| GunPoint | HERAMBA R5 | VALID (canonical R2 1.0000) | MISSING | MISSING |
| Phoneme  | MiniRocket | VALID (ref 0.0808) | MISSING | MISSING |
| Phoneme  | HERAMBA R5 | VALID (canonical R2 0.1182) | MISSING | MISSING |
| FordA    | MiniRocket | VALID (canonical M0 0.9499) | MISSING | MISSING |
| FordA    | HERAMBA R5 | VALID (canonical R2 0.9560) | MISSING | MISSING |

Seed-42 canonical artifacts were REUSED as gates (untouched); all 43/44
cells were missing and were newly executed. No reruns were required.

## Gates (all PASS)
| Gate | Observed | Reference | Tol | Verdict |
|---|---|---|---|---|
| GunPoint MR seed42 | 0.9933 | 0.9933 | 0.0011 | PASS (exact) |
| GunPoint HERAMBA seed42 | 1.0000 | 1.0000 | 0.011 | PASS (exact) |
| Phoneme MR seed42 | 0.0808 | 0.0808 | 0.011 | PASS (exact) |
| Phoneme raw-HERAMBA seed42 | 0.1238 | 0.1182 | 0.011 | PASS |
| FordA MR seed42 | 0.9499 | 0.9499 | 0.011 | PASS (exact) |
| FordA raw-HERAMBA seed42 | 0.9553 | 0.9560 | 0.011 | PASS |

Note: Phoneme/FordA raw-HERAMBA seed-42 gates compare a freshly retrained
per-seed context against the stored canonical context (expected seed-level
variation; repo tolerance 0.011). GunPoint matched exactly.

## Freezes honored
- HERAMBA R5: unchanged (frozen R2 recipe; per-seed context is the only
  seed-varying component; no feature-budget/loss/selection changes).
- MiniRocket: unchanged (canonical protocol, no retuning).
- Split: GunPoint/FordA canonical UCR train/test TSVs (sha256-verified by
  the loader) + stratified 15%-of-train val (random_state=42); Phoneme
  provided canonical val; identical across experiment seeds; split seed
  never conflated with outer seed.

## Leakage / hard-stop checklist
[x] No test labels in any selection stage        [x] fallback uses
validation only (HERAMBA 9/9; 0 fallbacks)       [x] raw-HERAMBA tests
recorded for audit, never used for selection     [x] no max() over test
[x] extractor identity 0.0 on all datasets       [x] no NaN/Inf (asserted)
[x] regime determinism asserted per seed         [x] predictions correct
length (150/1896/1320), finite                   [x] configuration drift:
none (single frozen recipe; only seed-dependent fields vary)
[x] CWRU/ES/Haptics artifacts untouched

## Execution
START 12:58:20 (2026-09-23), COMPLETE ~13:17 (about 19 min; FordA arms
~290 s each on GPU, GunPoint seconds). Mechanical smoke passed before
launch.
