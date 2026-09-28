# HIER-HIGH-SUP — one-change ablation (UWaveGestureLibraryY, seed 42)

Only change vs HIER-HIGH: the hierarchical candidate pool (449,820) is
ranked by **supervised ANOVA-F, top 1,999** (the exact R5-HIGH selector,
final stage on train+val per the canonical protocol) instead of the
label-free energy allocator (level_budgets/select_carriers — quotas
K2=4,437/K4=3,767/K8=3,128/K16=3,662 NOT reused). Everything else locked
and asserted identical: expanded MiniROCKET (aeon n_kernels=19,992,
random_state=42, train-only; MR-HIGH reproduction gate diff 0.0000),
frozen K=1→2→4→8→16 recursive-2-means hierarchy (train latents only,
regression-equal to the stored hierarchy_assignments.npy), occupancy-
weighted parent/child delta definition (30 edges × 14,994 kernels),
G_high = first 17,993 (R5-HIGH convention, rho=0.1), canonical
RidgeClassifierCV (alphas=logspace(-4,4,20)) with ONE test evaluation.

## Comparison (saved references; only HIER-HIGH-SUP was run)

| method | final feats | G | H | H pool | Macro-F1 | acc |
|---|---|---|---|---|---|---|
| canonical_mr | 9,996 | 9,996 | 0 | 9,996 | 0.7543 | 0.7577 |
| mr_high | 19,992 | 19,992 | 0 | 19,992 | 0.7477 | 0.7521 |
| r5_high | 19,992 | 17,993 | 1,999 | 14,994 | 0.7772 | 0.7797 |
| hier_high | 19,992 | 4,998 | 14,994 | 449,820 | 0.7230 | 0.7289 |
| **hier_high_sup** | 19,992 | 17,993 | 1,999 | 449,820 | **0.7823** | 0.7848 |

Deltas of HIER-HIGH-SUP: canonical_mr +0.0280, mr_high +0.0346,
r5_high +0.0051, hier_high +0.0593.
alpha = 11.2884; CV (fold-internal selection, diagnostic):
0.7867 ± 0.0267.

## Diagnostic: where the supervised selector spent the 1,999 H slots

| level | pool columns | selected | share of H |
|---|---|---|---|
| K=2 | 29,988 | 1,810 | 90.5% |
| K=4 | 59,976 | 181 | 9.1% |
| K=8 | 119,952 | 7 | 0.35% |
| K=16 | 239,904 | 1 | 0.05% |

The labels concentrate the budget on the COARSE level — the opposite of
the energy allocator, which spread the budget across levels by intrinsic
variance. (K=2's two child columns are exact anti-correlated rescalings
under the locked pool definition, so both can enter the top-1,999; the
ablated allocator deduped one of them.)

## Status

- Focused tests: 19 passed (tests/test_uwavey_hier_high_sup.py).
- Full repository suite: 642 passed, 1 skipped.
- Determinism: two independent full runs — byte-identical predictions,
  comparison.csv, selection metadata; only wall-clock runtime differs.
- Leakage: selection consumed train+val labels exactly as the canonical
  R5-HIGH protocol does; test labels only inside ridge_eval for scoring;
  hierarchy/pool label-free (see diagnostics/leakage_report.json and
  diagnostics/selection_protocol.md).
- Runtime ≈ 330–360 s per full run; peak working set observed ≈ 2.6 GB
  (the in-process psapi probe returns 0 on this machine — known issue).
