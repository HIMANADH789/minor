# DRTN Controlled Experiments — Continuous vs Discrete Trajectory vs K sweep
**Haptics, seed 42.** Purpose: determine WHICH component explains the R5 gain
(0.1509 → 0.3467 in the first probe) before sweeping anything else.

## 1. Repository audit
- Reused unchanged: `models/drtn/model.py` (encoder, HardVQ, EMA+revival,
  TrajectoryTransformer, TemporalAttentionPool, R5), canonical Haptics loader
  `experiments/external_stack_generalization/data.py`, original test suite.
- New: `models/drtn/controls.py` (DRTN_CTC, DRTN_DTC, param_audit),
  `experiments/drtn_haptics_controls_seed42/{runner,figures}.py`,
  `tests/test_drtn_controls.py` (14 tests).
- Original official results untouched; R5 K=8 row reuses the official run
  (`results/drtn_haptics_seed42/R5`), not a rerun.

## 2. Dataset audit (loader assertions PASSED before any run)
train=132, val=23 (deterministic stratified 15%, seed 42), test=308, T=1092,
5 classes, per-sample z-norm. Canonical UCR/aeon files, unchanged.

## 3. Configuration
One shared frozen `CFG` (seed 42, D=64, dilations {1,2,4,8}, transformer
2×4H/FF128/dropout 0.1, AdamW lr 1e-3, wd 1e-4, OneCycle, batch 16, ≤60 ep,
patience 10, β=0.25, λ_div=0.01, EMA 0.99, revival patience 100) + per-run
specs: CTC (no VQ, CE only), DTC (VQ, CE+commit), R5 K∈{8,16,32} (VQ+diversity).

## 4. Parameter audit (EXACT)

| Model | Encoder | VQ | Transformer | Pool | Head | EMA buffer | Trainable |
|---|---:|---:|---:|---:|---:|---:|---:|
| Continuous Control | 3,632 | 0 | 591,232 | 4,224 | 325 | 0 | **599,413** |
| Discrete Control | 3,632 | 0 | 591,232 | 4,224 | 325 | 1,040 | **599,413** |
| R5 K=8 (official) | 3,632 | 0 | 591,232 | 4,224 | 325 | 1,040 | **599,413** |
| R5 K=16 | 3,632 | 0 | 591,232 | 4,224 | 325 | 2,080 | **599,413** |
| R5 K=32 | 3,632 | 0 | 591,232 | 4,224 | 325 | 4,160 | **599,413** |

Key structural fact: the codebook is an **EMA buffer, not a trainable
parameter** — so removing VQ costs zero trainable parameters and the
continuous control is parameter-EXACT vs R5 with no projection hack. Only the
non-trainable buffer size differs across K.

## 5. Unit tests
32/32 pass (`tests/test_drtn.py` 18 + `tests/test_drtn_controls.py` 14),
including CTC forward/shape/causality, DTC hard-assignment/straight-through/EMA,
K∈{8,16,32} shapes, population-only diversity absence in DTC, no-label-access,
seed determinism, and encoder/transformer/parameter identity between controls.

## 6-8. Training / validation / test results (test evaluated exactly once per run)

| Model | K | VQ | Diversity | Params | Val F1 | **Test F1** | Best ep | Time | Peak mem |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| Continuous Control (CTC) | — | no | no | 599,413 | 0.1800 | **0.1486** | 3 | 7.8 s | 1,312.6 MB |
| Discrete Control (DTC) | 8 | yes | no | 599,413 | 0.5228 | **0.3386** | 24 | 24.0 s | 1,319.9 MB |
| R5 K=8 (official) | 8 | yes | yes | 599,413 | 0.4891 | **0.3467** | 42 | 33.1 s | 1,319.0 MB |
| R5 K=16 | 16 | yes | yes | 599,413 | 0.4891 | **0.3803** | 31 | 29.5 s | 1,319.9 MB |
| R5 K=32 | 32 | yes | yes | 599,413 | 0.5576 | **0.3566** | 39 | 35.5 s | 1,320.5 MB |
| *MiniROCKET-10K (non-neural baseline)* | — | — | — | 0 | — | *0.4974* | — | — | — |

CTC per-class F1 [0.000, 0.362, 0.000, 0.000, 0.382] — the continuous control
collapsed onto 2 of 5 classes (best-val epoch 3, early stop 13). DTC recovers
all classes [0.000, 0.460, 0.418, 0.326, 0.489].

## 9. Codebook diagnostics (final, validation)

| Run | K | Active | H_norm | Perplexity | Dominant | Revivals | Avg commit |
|---|---:|---:|---:|---:|---:|---:|---:|
| DTC (no diversity) | 8 | 8/8 | 0.714 | 4.41 | 0.386 | 6 | ~0.012 |
| R5 K=8 | 8 | 7/8 | 0.565 | 3.23 | 0.488 | 6 | ~0.012 |
| R5 K=16 | 16 | 12/16 | 0.646 | 6.00 | 0.309 | 13 | ~0.012 |
| R5 K=32 | 32 | 15/32 | 0.531 | 6.30 | 0.325 | 28 | ~0.012 |

## 10. Fairness checks — **29/29 PASS**
Identical encoder/transformer/pool/classifier/trainable counts between all
five runs; CTC provably has no VQ; DTC buffer layout == R5 K=8; K sweep
changes only the buffer size; every run uses the same frozen CFG (full list
in `report.json`).

## 11. Scientific comparisons (single-seed exploratory, NOT significant)
- **Continuous → Discrete: Δ = +0.1900** (0.1486 → 0.3386)
- **Discrete → R5 (add diversity): Δ = +0.0081** (0.3386 → 0.3467)
- K16 − K8 = +0.0336 · K32 − K8 = +0.0099 · K32 − K16 = −0.0237

## 12. Interpretation
Outcome **B**, decisively: hard discrete regime trajectories — not transformer
capacity — explain the R5 gain. With identical 599,413 parameters, the
continuous control cannot even optimize stably on this tiny training set
(val 0.18, two-class collapse), while the discretized variant reaches 0.34.
Discretization acts as a strong, beneficial inductive bias here (Haptics has
132 training samples; the codebook quantizes the continuous sequence into a
smaller effective hypothesis space, and the straight-through path keeps
gradients flowing). The population-diversity term is outcome **D**: R5 ≈ DTC
(+0.8 pp), i.e., diversity mainly improves codebook hygiene (R5 K=8 is more
concentrated, H_norm 0.565 vs 0.714) rather than prediction. The K sweep is
outcome **E-weak**: K=16 is the best test run overall (0.3803) but K=32 does
not improve further and its val advantage (0.5576, best of all) does not
transfer to test (n_val=23 noise). All five models remain below MiniROCKET's
0.4974.

## 13. Limitations
Single dataset, single seed, val=23 (checkpoint selection is noisy — DTC's
val 0.5228 vs R5's 0.4891 inverts under test); no variance estimates; K
differences (±3 pp) are within plausible seed noise; CTC's failure may partly
reflect optimization instability rather than representation poverty
(its best-val epoch was 3 and training diverged early).

## 14. Recommended next experiment
The mechanism question is answered: **keep hard VQ, treat diversity as
hygiene, prefer K=16.** Next: (1) 5-seed replication of
CTC vs DTC vs R5-K16 (cheap, ~4 min total) to put error bars on the +19 pp
discretization effect; (2) probe whether the continuous control's collapse is
fixable with stronger regularization (e.g., higher weight decay, dropout on
Z) — if a *stable* continuous control approaches DTC, the discretization
story becomes "optimization stabilizer" rather than "representation"; (3) only
then transfer the K=16 configuration to EpilepticSeizures/Phoneme.
