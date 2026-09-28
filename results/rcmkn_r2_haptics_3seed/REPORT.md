# R2 Haptics -- 3-Seed Robustness Experiment

Replication of the finalized R2 architecture across learned-model seeds 42/43/44 under a fixed MiniROCKET bank, canonical split, per-sample z-normalization, and the canonical Ridge protocol.

## 1. Motivation

The R2 result on Haptics (test Macro-F1 0.5500 vs canonical MiniROCKET M0 0.4974, seed 42) is a single-run number. This experiment asks whether the improvement survives re-training the learned context pipeline under different random seeds.

## 2. R2 architecture (unchanged)

```
X_R2 = [G || H]           (9,996 features)
G_m  = MiniROCKET PPV_m   (4,998 global, random_state=42)
H_m  = sum_k q_k (PPV_{m,k} - PPV_m)^2   (4,998 het)
      PPV_{m,k} over hard-VQ regimes k_t in {1..8}
      -> RidgeClassifierCV(alphas=logspace(-4,4,20))
```

## 3. Why Haptics

Haptics is R2's strongest and most-cited win (R2 0.55 vs M0 0.4974); it is the claim most in need of a seed-robustness check.

## 4. Exact 3-seed protocol

- Seeds: 42, 43, 44 (outer seed of the learned context only).
- Dataset: Haptics, canonical split train=132 / val=23 / test=308, T=1092, 5 classes; never reshuffled.
- Preprocessing: per-sample z-normalization, identical in all runs.
- Ridge: fit on train+val, alpha via internal LOO-CV on the predeclared grid; exactly ONE official test evaluation per seed (3 total).
- No per-seed tuning of any kind (architecture, K, masking, feature budget, Ridge policy all fixed).

## 5. Fixed MiniROCKET protocol

- `aeon MiniRocket(random_state=42, n_jobs=-1)`, fit on the z-normed TRAIN rows only.
- Identical kernels, thresholds, valid regions, feature order, and selected 4,998 global features for all three seeds.
- Raw-activation extractor verified against the canonical aeon transform (max |diff| < 1e-5).

## 6. SSL / VQ seed variation

- SSL encoder (4 causal dilated Conv1d blocks, d=32 projection) and HardVQ (K=8, EMA, commitment 0.25, diversity 0.01) are initialized and trained independently per seed with the exact validated R2 schedule: SSL epochs<=120 (patience 20), joint fine-tune epochs<=60 (patience 10, lambda_cls=0.10).
- Regime assignments are deterministic given a seed's trained model (verified by re-extraction).

## 7. Ridge protocol

`RidgeClassifierCV(alphas=np.logspace(-4,4,20))` on [G || H]; identical to the validated R2 run. No SGD, no neural classifier, no new alpha policy.

## 8. Audit results

| Audit | Result |
|---|---|
| audit1_dataset_identity | pass=True |
| audit2_split_identical_across_seeds | pass=True |
| audit3_znorm | pass=True |
| audit4_extractor_identity | pass=True |
| audit5_minirocket_seed_fixed_42 | pass=True |
| audit6_G_budget_4998 | pass=True |
| audit7_ssl_architecture | pass=True |
| seed42_reproduction_gate | pass=True |
| seed42 audits (18 checks incl. 8/9/10/11/12/13/14/15/16/17/18) | all PASS |
| seed43 audits (18 checks incl. 8/9/10/11/12/13/14/15/16/17/18) | all PASS |
| seed44 audits (18 checks incl. 8/9/10/11/12/13/14/15/16/17/18) | all PASS |

## 9. Per-seed results

| Seed | Val Macro-F1 | Test Macro-F1 | Delta vs M0 | Alpha | Runtime (s) |
|---|---|---|---|---|---|
| 42 | 0.9014 | 0.5470 | +0.0496 | 4.2813 | 22.7 |
| 43 | 0.9014 | 0.5213 | +0.0239 | 4.2813 | 21.3 |
| 44 | 0.8618 | 0.5387 | +0.0413 | 4.2813 | 14.4 |

## 10. Mean +/- std

- Test Macro-F1: **0.5357 +/- 0.0107** (min 0.5213, max 0.5470)
- Delta vs M0: +0.0383 +/- 0.0107
- Seeds improving over M0: 3/3

## 11. Comparison with M0

- Canonical Haptics MiniROCKET M0 = 0.4974 (seed 42).
- Per-seed deltas: seed 42: +0.0496, seed 43: +0.0239, seed 44: +0.0413.
- Note: M0 is a single fixed-MiniROCKET baseline; the R2 MiniROCKET bank is identical across seeds, so deltas reflect only the learned-context variation.

## 12. VQ diagnostics

| Seed | Active codes | Perplexity | Norm. entropy | Dominant frac. |
|---|---|---|---|---|
| 42 | 8/8 | 7.004 | 0.936 | 0.233 |
| 43 | 8/8 | 5.819 | 0.847 | 0.290 |
| 44 | 6/8 | 5.594 | 0.828 | 0.245 |

Code identities are not semantically comparable across seeds; only distributional statistics are compared.

## 13. Runtime

seed 42: 23s, seed 43: 21s, seed 44: 14s

## 14. Limitations

- n=3 seeds, single dataset, single split: descriptive statistics only; no significance testing is meaningful here.
- M0 is evaluated once (its MiniROCKET bank is deterministic and shared), so paired differences are per-seed scalars.
- The seed-42 replication gate uses the repo-established tolerance of 0.011 Macro-F1.

## 15. Final robustness conclusion

- Verdict: **ROBUST ACROSS TESTED SEEDS**
- All three seeds improve over M0: the Haptics improvement is consistent across the tested seeds.
- Mean R2 = 0.5357 vs M0 = 0.4974 (mean delta +0.0383); std across seeds 0.0107.

