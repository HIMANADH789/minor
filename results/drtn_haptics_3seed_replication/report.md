# DRTN Strict 3-Seed Replication — Haptics (seeds 42 / 43 / 44)

**Purpose.** Determine whether the seed-42 discretization effect
(CTC 0.1486 → DTC 0.3386, Δ = +19.0 pp under the 60/10 budget) replicates
across independent seeds. Central hypothesis under test: *hard discrete
latent regimes provide a useful inductive bias for temporal trajectory
modeling under the same trainable parameter budget.* MiniROCKET (0.4974) is
an external reference only.

---

## 1. Pre-registered budget change (frozen before any new test evaluation)

The user pre-registered a training-budget change **before** this stage's test
results existed: `max_epochs 60→100`, `patience 10→15`, documented in the
immutable config with rationale (the 60/10 budget ended the seed-42 CTC run
at epoch 13 with best epoch 3, depriving the shared transformer of
optimization time). The 100/15 budget applies **identically** to all nine
runs; no other hyperparameter, architecture, or data change was made. Note
the consequence, stated plainly: **the seed-42 numbers below are not
comparable to the earlier controlled-experiment numbers** — both budgets were
run under frozen, single-evaluation discipline, but they are different
protocols. All 9 runs here used the 100/15 budget from the start.

## 2. Dataset / protocol audit
Canonical Haptics loader (unchanged): train 132 / val 23 / test 308, T=1092,
5 classes, per-sample z-norm — asserted before every run. AdamW lr 1e-3,
wd 1e-4, OneCycle, batch 16, best-val-Macro-F1 checkpoint selection, test
evaluated exactly once per run after freeze.

## 3. Model definitions (unchanged implementations)
- **CTC** `models/drtn/controls.py::DRTN_CTC`: encoder → continuous Z → same
  causal transformer → attention pool → classifier. L = CE.
- **DTC** `DRTN_DTC`: encoder → hard VQ K=8 (straight-through, EMA 0.99,
  revival patience 100) → same transformer. L = CE + 0.25·L_commit.
- **R5-K16** `models/drtn/model.py::DRTN_R5` (K=16): adds population
  diversity λ=0.01. L = CE + 0.25·L_commit + 0.01·L_div.

## 4. Parameter audit — automatic gate, 9/9 PASS
Every run audited before training: **599,413 trainable parameters for all
three models at all seeds** (codebook = EMA buffer, 0 trainable; K=16 only
enlarges the non-trainable buffer 1,040 → 2,080 elements). The runner raises
and stops on any mismatch; none occurred.

## 5. Seed configuration
Each (seed × model) combination independently reset python/numpy/torch-CPU/
torch-CUDA RNGs before construction and training — no shared advancing
stream. 9 independent runs.

## 6. Unit tests: 32/32 pass (18 DRTN + 14 controls), incl. causality,
straight-through, EMA, revival, population-only diversity, determinism.
Smoke run (seed 43, 2 epochs, sandboxed) verified forward/backward/checkpoint/
val/diagnostics before launch; smoke outputs deleted.

## 7-8. Per-seed results (test Macro-F1, ONE evaluation per run)

| Seed | Model | Params | Val F1 | Test F1 | Best ep | Time |
|---|---|---:|---:|---:|---:|---:|
| 42 | CTC | 599,413 | 0.6264 | 0.3821 | 54 | 41.3 s |
| 42 | DTC | 599,413 | 0.5212 | 0.3577 | 47 | 45.7 s |
| 42 | R5-K16 | 599,413 | 0.6345 | 0.3726 | 83 | 71.6 s |
| 43 | CTC | 599,413 | 0.4879 | 0.3563 | 70 | 49.8 s |
| 43 | DTC | 599,413 | 0.4832 | 0.3220 | 31 | 33.0 s |
| 43 | R5-K16 | 599,413 | 0.5908 | 0.3266 | 66 | 56.4 s |
| 44 | CTC | 599,413 | 0.4489 | 0.3213 | 45 | 35.3 s |
| 44 | DTC | 599,413 | 0.5267 | 0.3483 | 42 | 40.4 s |
| 44 | R5-K16 | 599,413 | 0.5297 | 0.3386 | 23 | 26.5 s |

## 9. Aggregate (sample SD, n=3)

| Model | Mean | SD | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
| CTC | **0.3532** | 0.0305 | 0.3563 | 0.3213 | 0.3821 |
| DTC | 0.3427 | 0.0185 | 0.3483 | 0.3220 | 0.3577 |
| R5-K16 | 0.3459 | 0.0239 | 0.3386 | 0.3266 | 0.3726 |

## 10. Paired seed-wise effects (primary: DTC − CTC)

| Seed | DTC − CTC | R5-K16 − DTC |
|---|---:|---:|
| 42 | **−0.0244** | +0.0149 |
| 43 | **−0.0343** | +0.0046 |
| 44 | **+0.0270** | −0.0097 |
| **mean** | **−0.0106** | **+0.0033** |
| SD | 0.0329 | 0.0124 |

(The paired test that would accompany these is not reported as inference:
n=3 is underpowered by construction. Replication consistency, below, is the
evidence.)

## 11. Codebook diagnostics (final, validation; mean ± SD over seeds)

| Model | Active | H_norm | Perplexity | Dominant | Revivals |
|---|---|---|---|---|---|
| DTC K=8 | 7.67 ± 0.58 (of 8) | 0.732 ± 0.074 | 4.62 ± 0.72 | 0.390 ± 0.054 | 4.3 ± 2.1 |
| R5 K=16 | 11.7 ± 1.2 (of 16) | 0.573 ± 0.035 | 4.91 ± 0.47 | 0.411 ± 0.056 | 11.7 ± 4.2 |

Entropy is reported as a diagnostic only. As before, the better-performing
run at seed 42 (R5-K16 val 0.6345) had *lower* entropy than DTC — consistent
with the earlier finding that informative concentration, not maximal
diversity, tracks performance.

## 12. Runtime
26.5–71.6 s training per run (~7 min total), inference ~1.3 s, peak GPU
1.31–1.32 GB.

## 13. Test-evaluation audit
9 runs → **exactly 9 final test evaluations**, one per completed run,
each after its checkpoint was frozen (`test_touch_audit.one_per_run = true`
in report.json). No test metric entered any selection decision.

## 14. Replication consistency
- **DTC > CTC on 1/3 seeds** (seed 44 only).
- **R5-K16 > DTC on 2/3 seeds.**

## 15. Scientific interpretation — CASE 3 (+ CASE 5)

The seed-42 discretization effect **did not replicate**. This is **CASE 3**:
DTC ≈ CTC (wins only 1/3, mean Δ = −1.1 pp). Under the pre-registered 100/15
budget the continuous control trains stably — its per-class F1s show no
two-class collapse (e.g. seed 42: [0.21, 0.39, 0.35, 0.45, 0.51], best epoch
54 vs 3 under 60/10) — and it matches or beats the discrete variants
(CTC mean 0.3532 vs DTC 0.3427 vs R5-K16 0.3459). The earlier +19 pp gap was
therefore **not a property of discretization as a representation**: the most
consistent explanation is that it was an **optimization-budget artifact** —
the continuous model needed more than 13 epochs to escape its early
instability, while the VQ straight-through path happened to start faster at
seed 42 under the short budget.

For the diversity/full-model ladder this is **CASE 5**: R5-K16 ≈ DTC
(mean +0.3 pp, 2/3 seeds) — the full objective does not add reliable
predictive value beyond hard VQ at K=16 on this dataset.

Honest paper language: the three-seed, single-dataset evidence is
*consistent with* hard discretization acting as an optimization stabilizer
under short budgets, and does **not support** a representation-level
advantage for discrete regimes over a continuous trajectory of identical
capacity once the budget is adequate. Nothing here is statistically
significant; all effects (±1–3 pp) are within seed noise (SD ≈ 2–3 pp).

## 16. Limitations
Three seeds on one small dataset (132 train / 23 val); val-based checkpoint
selection is noisy and CTC's seed-42 val (0.6264) notably did not transfer
(test 0.3821); the 100/15 budget itself was chosen after observing the 60/10
CTC failure — although frozen before any new test evaluation, a fully
pre-registered study would have fixed it before seed 42 as well; MiniROCKET
(0.4974) remains above every neural variant under every budget tried.

## 17. Recommended next step
The discretization question on Haptics is now answered negatively at the
representation level: stop iterating architectures on this dataset. If DRTN
is pursued, the two live options are (1) test the *optimization-stabilizer*
framing directly — train CTC and DTC under a deliberately short budget
across seeds and show the gap reappears/disappears as a function of budget,
which would turn the artifact into a citable finding about VQ
straight-through training dynamics; or (2) accept that on tiny-data Haptics
none of the 599K-parameter trajectory models closes the gap to MiniROCKET
and evaluate the architecture on a dataset where neural models are
competitive (EpilepticSeizures train=80 is not it; a larger dataset would be
required before any DRTN claim is worth testing).
