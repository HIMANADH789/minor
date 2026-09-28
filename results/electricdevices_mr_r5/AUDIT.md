# ElectricDevices — HERAMBA R5 vs MiniRocket (3 seeds) — AUDIT

Task (2026-09-23): install ElectricDevices, run the 3-seed MR vs R5
(HERAMBA) experiment. **NO FALLBACK** — both arms are reported directly;
raw deltas are final (no validation-stage model switching, no clamping).

## Dataset installation (verified)
- Source: repo's verified UCRArchive_2018 copy
  `data/kaggle/_ucrarchive_2018/UCRArchive_2018/UCRArchive_2018/ElectricDevices/`
- TRAIN 8926 x 97 (T=96), TEST 7711 x 97 — canonical UCR shapes.
- Classes: 7 ({1..7}); TRAIN counts 727/2231/851/1474/2406/509/728;
  TEST counts 667/1956/755/1165/1869/743/556 (imbalanced, handled by
  Macro-F1 with zero_division=0).
- sha256 (16-hex): TRAIN bf9b3e4ebb4bcecc, TEST 64ea1c73040b8c60.
- Split: canonical train/test TSVs (never re-split) + stratified 15% of
  train as val (random_state=42) — the repository's established split
  rule -> 7587/1339/7711, T=96, 7 classes (asserted in the runner).
- No prior ElectricDevices artifacts exist anywhere in the R2/rcmkn
  line (repo-wide grep: zero hits) -> per repo convention (cf. Phoneme,
  Wafer), the seed-42 values become the FIRST audited references
  (no gate possible; recorded in gates.json as first_reference).

## Protocol (identical frozen recipe; NO-FALLBACK variant)
- MR arm: aeon MiniRocket(random_state=42, fit TRAIN only) -> 9996
  features; RidgeClassifierCV(logspace(-4,4,20)) train-only fit -> val
  diagnostic; train+val refit -> ONE test evaluation. Deterministic
  across seeds (canonical convention, shown in every seed cell).
- HERAMBA R5 arm: per-seed RCMKNContextModel (SSL+VQ K=8+joint, frozen
  config byte-identical to the stored canonical configs), H =
  occupancy-weighted regime heterogeneity (min_occ 0.01), X_R5 =
  [G 4998 || H 4998], same Ridge/alpha grid, train-only fit -> val
  diagnostic; train+val refit -> ONE test evaluation.
- Seeds {42, 43, 44}; seed propagates through Python/NumPy/PyTorch/CUDA
  via the audited core set_seed; MiniRocket carriers fixed at 42.
- NO FALLBACK: no validation-stage model switching; both arms' test
  metrics are final; deltas = HERAMBA_test - MR_test as measured.

## Status matrix (initial)

| Dataset         | Model      | Seed 42 | Seed 43 | Seed 44 |
|-----------------|------------|---------|---------|---------|
| ElectricDevices | MiniRocket | MISSING | MISSING | MISSING |
| ElectricDevices | HERAMBA R5 | MISSING | MISSING | MISSING |

All 6 cells newly executed; nothing to reuse; nothing overwritten.
