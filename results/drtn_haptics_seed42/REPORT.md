# DRTN — Discrete Regime Trajectory Network: First Probe (Haptics, seed 42)

**Status:** first implementation probe, single dataset (Haptics), single seed (42).
The test set was untouched until each rung's best-validation checkpoint was
frozen; one final test evaluation per rung. Selection metric: validation
Macro-F1. No architecture or hyperparameter was changed after any test contact.

---

## 1. Executive summary

The DRTN hypothesis — end-to-end learned **discrete** latent regimes whose
**trajectory** carries class information — is **PARTIALLY SUPPORTED** on
Haptics. The full model (R5) is the clear best neural model at test
(Macro-F1 **0.3467** vs 0.1903 for the R0 neural floor, +15.6 pp), and the
gains arrive exactly where the hypothesis predicts: the trajectory component
(R5 vs R4, +19.6 pp) and healthy code usage. The two hypothesized collapse
modes were both *observed and characterized*: R2's continuous soft assignment
collapsed monotonically (normalized usage entropy 0.403 → 0.015 in 11 epochs),
and R3's hard VQ without diversity regularization started with 2 of 8 codes
alive (population-level codebook collapse). Revival + (in R4) the diversity
regularizer fixed codebook health (7→6-7 active codes, H_norm 0.72→0.82), but
codebook health alone did **not** translate into better prediction (R4 test
0.1509 < R3 0.1604) — a genuine negative result for the diversity mechanism's
direct predictive value. What the probe does *not* yet show: R5 remains far
below MiniROCKET (0.4974 test MF1 in the frozen external baseline), so the
regime trajectory helps the neural family but not the benchmark.

## 2. Scientific hypothesis

Canonical MiniROCKET shows strong local pattern detection, but PPV pooling
compresses temporal activation evidence into a static vector. DRTN replaces
that with a learned pipeline — temporally resolved encoder → discrete VQ
regimes → regime trajectory → attention pooling → linear classifier — with no
hand-engineered velocity/uncertainty/quantile/phase/drift features. The
ablation ladder isolates each mechanism:

| Rung | Architecture | Purpose |
|---|---|---|
| R0 | encoder → GAP → linear | neural floor |
| R1 | encoder → attention pool → linear | temporal preservation |
| R2 | + **soft** codebook attention (τ=0.5) | continuous regime control (collapse watch) |
| R3 | + **hard** VQ (commitment only, EMA, revival) | discreteness alone |
| R4 | + population-diversity regularizer (λ=0.01) | the proposed mechanism |
| R5 | + causal trajectory transformer (2L, 4H) | full DRTN |

## 3. Data audit (verbatim from `experiment_audit.json`)

- Source: canonical UCR/aeon `.ts` from
  `data/external/extracted/Haptics/` (timeseriesclassification.com), the exact
  frozen loader of `experiments/external_stack_generalization/data.py`.
- Split: train 132 (15/29/29/31/28), val 23 (3/5/5/5/5, deterministic
  per-class stratified 15% of train, seed 42), test 308 (60/58/59/64/67).
- Sequence length **T=1092**, univariate (1 channel), 5 classes,
  per-sample z-normalization (project canonical). 0 NaN/Inf/duplicates.

## 4. Main results (TEST, seed 42, best-val checkpoint)

| Model | Macro-F1 | Δ vs prev | Accuracy | W-F1 | Params | Δ params vs R0 | Best ep | Train s | Peak GPU MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 (GAP) | 0.1903 | — | 0.2565 | 0.1937 | 3,957 | 0 | 33 | 13.1 | 59.3 |
| R1 (attention) | 0.1034 | −8.7 pp | 0.2078 | 0.0980 | 8,181 | +4,224 | 10 | 6.1 | 70.2 |
| R2 (soft codebook) | 0.0643 | −3.9 pp | 0.1916 | 0.0616 | 8,693 | +4,736 | 1 | 4.1 | 81.0 |
| R3 (hard VQ) | 0.1604 | +9.6 pp | 0.2500 | 0.1633 | 8,181 | +4,224 | 31 | 9.9 | 66.0 |
| R4 (R3+diversity) | 0.1509 | −0.9 pp | 0.2662 | 0.1509 | 8,181 | +4,224 | 21 | 7.6 | 66.0 |
| R5 (full DRTN) | **0.3467** | **+19.6 pp** | **0.3831** | 0.3491 | 599,413 | +595,456 | 42 | 33.1 | 1,319.0 |

Reference anchors on the identical split: MiniROCKET-10K **0.4974**,
TURS-Stack 0.3553, best other neural baseline (ResNet-1D) 0.1512.

Per-class F1:
- R0 [0.000, 0.087, 0.271, 0.382, 0.211] — class 0 never predicted.
- R5 [0.063, 0.413, 0.404, 0.411, 0.443] — every class except 0 substantially
  recovered; class 0 (the 60-sample minority) remains hard.

Validation→test gaps are large for all rungs (val is n=23): e.g. R5
val 0.4891 → test 0.3467. Single-seed results; no variance estimate.

## 5. Regime diagnostics (the core of the probe)

### R2 — continuous soft assignment: concentration confirmed
Soft assignments **do** collapse, and measurably: mean max assignment
probability rose 0.774 → 0.997 and normalized batch-usage entropy fell
0.403 → **0.015** within 11 epochs (effectively one dominant code at 99.7%
mass). This is exactly the continuous analogue of the project's alpha/beta/KTA
gate-collapse history. R2's best-val checkpoint landed at epoch 1 (before
collapse deepened), so its test MF1 (0.0643) reflects a *degenerate,
uninformative* regime representation: every test sample predicted class 2.

### R3 — discreteness alone does not prevent codebook collapse
With commitment loss + EMA + revival but **no** diversity regularizer, R3's
usage histogram pinned to **2 active codes for the first 11 epochs**
(H_norm = 0.333, i.e. two equally-used codes, six dead). The 5 revivals all
fired at step 99 (patience=100), confirming the dead-code detector works —
but revivals alone did not restore diversity early. After the revival wave
(codebook re-seeded from real latents) usage spread to 6-7 codes with
H_norm ≈ 0.70. Descriptive status: **concentration early, moderate diversity
late**; not full collapse, but far from healthy without help.

### R4 — population diversity regularizer: mechanism works, prediction doesn't move
The label-independent batch-histogram entropy regularizer did exactly what it
claims: R4 reached H_norm **0.823** (perplexity 5.54/8, dominant code 0.273,
min-nonzero usage 0.080 vs R3's 0.003) — the healthiest codebook of all
non-trajectory rungs, with 6-7 codes in genuine use and **zero**
per-timestep softening (assignments stayed one-hot; instance sharpness is
preserved by construction). The weighted diversity term was tiny
(L_div·λ = −0.015 vs CE 1.56), so prediction remained the dominant objective.
**But** test Macro-F1 did not improve over R3 (0.1509 vs 0.1604, −0.9 pp).
Healthier population structure ≠ better classification at this scale: with
K=8 and ~2,000 timesteps per batch, CE alone evidently saturates the useful
regime granularity, and forcing extra codes into use mainly redistributes
mass without adding discriminative symbols. Honest verdict: the diversity
mechanism is *mechanistically validated* but *predictively inert* here.

### R5 — trajectory modeling is where the gain lives
Adding the causal trajectory transformer over the code-embedding sequence
produced the single largest jump in the whole ladder: **+19.6 pp test MF1**
over R4 (0.1509 → 0.3467) and +15.6 pp over R0. Learning curve evidence: R5's
validation MF1 climbed 0.16 → 0.49 across ~40 epochs while its codebook
entropy *fell* from 0.80 to ~0.56 — i.e., as the trajectory model became
useful, the codebook **concentrated** into fewer, more specialized symbols
(dominant code 0.49). Classification benefits from *informative*, not
maximally diverse, regime vocabularies. R5 also revived one code mid-training
(step 398) and kept 7/8 codes active.

Figure: `figures/fig_usage.png` (usage histograms + entropy trajectories);
`figures/fig_trajectories.png` (input waveform with discrete k(t) steps and
segment boundaries for 3 val examples — the qualitative trajectory picture).

## 6. Answers to the scientific questions (sec. 30)

- **Q1 — Does preserving temporal info help? R1 vs R0:** **No** here
  (0.1034 vs 0.1903). Scalar attention pooling collapsed R1 onto class 1
  (96% of test mass). Temporal preservation alone, without an explicit
  structure to exploit it, hurt under this seed/budget.
- **Q2 — Does continuous regime assignment help? R2 vs R1:** No
  (0.0643 vs 0.1034).
- **Q3 — Does continuous assignment collapse?** **Yes — demonstrated.**
  H_norm 0.403→0.015, max assignment prob →0.997 in 11 epochs.
- **Q4 — Does hard VQ alone solve it? R3:** Partially — no continuous
  hedging (assignments are exactly one code), but codebook population
  collapsed to 2/8 codes early; revivals restored 7/8 only after ~100 steps.
- **Q5 — Does population-level diversity change code usage?** **Yes —
  demonstrated.** H_norm 0.72→0.82, perplexity 4.47→5.54, dominant fraction
  0.41→0.27, min nonzero usage ×27.
- **Q6 — Does healthier usage translate into better prediction?** **No**
  (R4 0.1509 vs R3 0.1604, −0.9 pp).
- **Q7 — Does trajectory modeling add predictive information?** **Yes**
  (R5 vs R4: +19.6 pp; R5 vs R0: +15.6 pp). The trajectory over discrete
  regimes is the most informative component in the ladder.
- **Q8 — Is the R5 gain worth the cost?** Mixed: +595K params, ~4× R3
  runtime, 1.3 GB peak GPU. The gain is real but R5 (0.3467) still trails
  MiniROCKET (0.4974) on this dataset.
- **Q9 — Central hypothesis?** **PARTIALLY SUPPORTED.** Both mechanistic
  predictions (soft collapse; diversity fixes population health) were
  confirmed; the discreteness+diversity stack alone does not improve
  prediction, but trajectory dynamics over discrete regimes do — the largest
  within-ladder effect. The gap to MiniROCKET remains open.

## 7. Parameter counts (exact)

| Model | Total | Trainable | Non-trainable | Δ vs R0 |
|---|---:|---:|---:|---:|
| R0 | 3,957 | 3,957 | 0 | 0 |
| R1 | 8,181 | 8,181 | 0 | +4,224 |
| R2 | 8,693 | 8,693 | 0 | +4,736 |
| R3 | 8,181 | 8,181 | 0 | +4,224 |
| R4 | 8,181 | 8,181 | 0 | +4,224 |
| R5 | 599,413 | 599,413 | 0 | +595,456 |

(R5's transformer = 2 layers × [4 heads, d=64, ffn=128, dropout 0.1] ≈ 591K
params; the codebook itself is an EMA buffer, not a gradient parameter.)

## 8. Integrity notes

- Test evaluated exactly once per rung, after checkpoint freeze.
- Diversity regularizer is label-independent by construction (unit-tested);
  no class-conditioned code semantics; codes are latent symbols.
- Causality: encoder (causal convs, per-position LayerNorm) and trajectory
  transformer (causal mask) verified by future-leakage unit tests.
- 18/18 unit tests pass (incl. straight-through gradients, EMA math, revival,
  population-vs-instance entropy distinction, determinism at seed 42).
- Known limitation: val set is 23 samples — checkpoint selection is noisy;
  all results are single-seed and single-dataset.

## 9. Next-experiment decision (sec. 33K)

Proceed to **EpilepticSeizures and Phoneme** with the architecture unchanged
in structure but with the trajectory component as the default (it carries the
entire gain), keeping the R3→R4→R5 ladder for diagnosis. Do **not** scale the
diversity coefficient hoping for accuracy; consider reporting R3 vs R4 as a
mechanism study. Priority next probes: (1) K ∈ {16, 32} on the trajectory
rung (R5's entropy *decline* during successful training suggests the useful
vocabulary is larger and more concentrated than K=8 forces), (2) a
parameter-matched R5 (e.g. trajectory width 32-40) to separate "regime
trajectory" from "+595K params", (3) multiscale codebooks only after (1)-(2).
