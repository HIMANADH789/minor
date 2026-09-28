# CCWRU FINAL 3-SEED — AUDIT

Date: 2026-09-23. Verdict: MiniRocket 3/3 VALID, HERAMBA R5 3/3 VALID on
both CWRU_UNBAL and CWRU_BAL. All four seed-42 gates PASS.

## Scope decisions (user-confirmed)

1. Datasets: BOTH CWRU_UNBAL and CWRU_BAL.
2. HERAMBA R5 = the rcmkn R2/R5 line (user: "R5 is also called HERAMBA;
   R2 is a subset of R5 so they count as R5 / HERAMBA"). The
   drtn-transfer M1 line (which has its own literal `"rung": "R5"`
   checkpoint) is a DIFFERENT implementation and was excluded from the
   HERAMBA 3-seed object; its seed-42 M0 values were, however, consistent
   with the canonical MR references used here.

## Canonical seed set

{42, 43, 44} — the repository's established three-seed protocol
(drtn_conditioned_minirocket_ecg5000_bal_3seed, rcmkn_r2_haptics_3seed).
No other CCWRU seed protocol exists in the repo.

## Initial status matrix (Phase 1, before runs)

| Model | Seed | Artifact | Valid? | Action |
|---|---:|---|---|---|
| MR | 42 | drtn_transfer M0 (UNBAL 0.9917 / BAL 0.9947) | yes (canonical) | REUSE (gate) |
| MR | 43 | — | — | RUN (deterministic pipeline) |
| MR | 44 | — | — | RUN (deterministic pipeline) |
| HERAMBA R5 | 42 | rcmkn transfer R2 UNBAL 0.9917 / important2 R2 BAL 0.9982 | yes (canonical) | REUSE (as gate reference) |
| HERAMBA R5 | 43 | — | — | RUN |
| HERAMBA R5 | 44 | — | — | RUN |

Rationale for running fresh seed-42 R5 arms: no per-seed path existed for
CWRU; the fresh seed-42 run (new per-seed context) is gated against the
canonical stored R2 within the repo tolerance 0.011 and its artifacts are
kept separately (results/ccwru_final_3seed/); the original seed-42
artifacts are untouched and remain the references.

## Split identity (spec section 10)

- CWRU_UNBAL: 1156/204/240, T=1024, 4 classes (manifest:
  results/diagnostics/turs_lite/audit/dataset_manifest_CWRU_UNBAL.json;
  split seed 42, stratified 85/15 then 15% of trainval).
- CWRU_BAL: 2727/482/567 (same protocol).
- Split arrays are loaded once per dataset and shared by every seed; the
  split seed (42) is never conflated with the outer experiment seed
  (42/43/44). Asserted in the runner (EXPECTED) before any training.

## Protocol freezes honored

- MR: aeon MiniRocket rs=42 fit train-only; RidgeClassifierCV
  logspace(-4,4,20); train-only -> val diagnostic; train+val refit -> one
  test evaluation. Untouched.
- HERAMBA R5: R2 composition [G 4998 || H 4998]; H = occupancy-weighted
  regime heterogeneity (min_occupancy 0.01, valid-region masks); context =
  RCMKNContextModel SSL+VQ K=8+joint, config byte-identical across the
  CWRU seed-42 producers and the audited 3-seed core; per-seed context
  training is the ONLY seed-varying component; same Ridge protocol.
  Untouched.
- No test-driven tuning anywhere; test touched once per (model, seed).

## Determinism / seed propagation (spec section 12)

- MR: extractor random_state fixed at 42 (canonical convention — the
  MiniRocket bank is part of the frozen protocol, not the outer seed);
  Ridge deterministic. Verified: MR values identical across seeds.
- HERAMBA R5: outer seed propagated to Python/NumPy/torch (+CUDA) via the
  core set_seed before model init, SSL/joint training (dataloader
  generator seeded), and regime extraction (asserted deterministic by
  re-extraction under the same seed). Regime hash per seed recorded.
- VQ code identities are not compared across seeds (repo convention);
  only regime hashes and metrics are.

## Gates (spec sections 5/6/28)

| Gate | Observed | Reference | Tol | Verdict |
|---|---|---|---|---|
| MR CWRU_UNBAL seed42 | 0.9917 | 0.9917 | 0.0011 | PASS |
| MR CWRU_BAL seed42 | 0.9947 | 0.9947 | 0.0011 | PASS |
| R5 CWRU_UNBAL seed42 | 0.9833 | 0.9917 | 0.011 | PASS |
| R5 CWRU_BAL seed42 | 0.9965 | 0.9982 | 0.011 | PASS |

Note: the R5 seed-42 gates compare a freshly retrained per-seed context
against the stored canonical context model; within-tolerance agreement
(0.0084 / 0.0017) is the expected seed-level variation for this component
(repo tolerance 0.011 was set exactly for this purpose).

## Per-run validation checks (spec section 16)

All 12 arms: predictions present with correct length (240 UNBAL / 567 BAL),
finite integer labels, no NaN/Inf in feature matrices (asserted), regime
determinism asserted, alpha values finite and from the frozen grid,
val/test metrics recorded per seed. See PREDICTION_CHECKS.csv and
per_run_results.json.

## Artifact preservation

Nothing under results/rcmkn_ssl_context_transfer_seed42/,
results/rcmkn_ssl_context_important2_seed42/,
results/drtn_conditioned_minirocket_transfer_seed42/, or any other prior
directory was modified. All new artifacts live under
results/ccwru_final_3seed/ and experiments/ccwru_final_3seed/.

## Execution log

Production run started 11:37:32 (2026-09-23), survived a Freebuff restart
mid-run (CWRU_BAL seed-42 SSL training), completed ~12:20 with no errors.
Mechanical smoke (24-sample subset through MiniRocket -> context training
-> regimes -> H -> shapes) passed before launch.
