# UWaveY — CAPACITY-CONTROLLED NESTED/HIERARCHICAL REGIME-CONDITIONED H

**Experiment:** `experiments/uwavey_nested_hierarchical/` (seed 42)
**Modules:** `models/nested_regimes/model.py` (reused, untouched math core), `models/hierarchical_budget/model.py` (new: budget/allocation/selection)
**Results:** `results/uwavey_nested_hierarchical/seed42/`
**Companion to:** Haptics nested-regimes experiment (`results/haptics_nested_regimes/seed42/evidence_report.md`)

---

## 1. Research question

Does a nested regime hierarchy provide a useful, multi-resolution H
representation when the total H feature capacity is fixed to the same
**4,998-feature budget** as the existing flat HERAMBA representation?

This separates two quantities the Haptics experiment conflated:
(A) intrinsic hierarchical depth (label-free) and (B) classifier
feature budget (fixed by design).

## 2. Why the Haptics full-depth result required capacity control

The Haptics hierarchical experiment retained L\*=16 (all levels passed the
label-free null) but evaluated the **full detail bank (~150k features) on
155 train+val rows** and scored below both references. That result showed
only that *uncontrolled feature count* destroys Ridge generalization —
structural depth ≠ allowable classifier capacity. The present experiment
repeats the intrinsic pipeline on a larger dataset and constrains the H
side to exactly 4,998 features.

## 3. UWaveY dataset and canonical protocol

- Dataset `UWaveGestureLibraryY` via the repository's canonical loader
  (`experiments/rcmkn_r2_uwave_seed42/data.load_and_split`): official split,
  **train 761 / validation 135 / test 3582**, T = 315, 8 classes.
- Per-sample z-normalization (project convention); MiniRocket fit on
  train-z-normed signals only, `random_state=42`.
- Canonical Ridge protocol: `RidgeClassifierCV(alphas=logspace(-4,4,20))`
  fit on train+validation after all representation decisions were frozen;
  one official test evaluation.
- Canonical feature split: MiniRocket yields 9,996 features; **G bank =
  first 4,998**, **H-bank kernels = features 4,998:9,996** (4,998 kernel
  units). Full 9,996 is the M0 (MiniROCKET-only) representation.

## 4. Existing G and flat H representations

No stored UWaveY frozen bank files exist in the repository; both gates were
recomputed **in-run** from the frozen R2 machinery (unchanged code paths):

- G: fresh canonical MiniRocket transform (verified: aeon direct transform
  vs per-timestep activation PPVs agree to ≤ 1e-12).
- Flat H: `compute_regime_heterogeneity(act[:, 4998:], valid[4998:], regimes)`
  with the frozen flat K=8 VQ regimes from the stored seed-42 UWaveY
  context checkpoint (`results/r2_uwave_seed42/context_model.pt`,
  `SSLTemporalEncoder` d=32) — the exact R2/RCMKN code, unmodified.

## 5. Nested hierarchy construction

**What is partitioned (precise):** the FROZEN `SSLTemporalEncoder` temporal
latents z ∈ R³² of the seed-42 RCMKN UWaveY checkpoint, pooled over TRAIN
samples only: 761 × 315 = 239,715 latent vectors.

Construction: recursive seeded 2-means. Root = all latents; every current
node split with `KMeans(n_clusters=2, n_init=10, random_state=42)` fit on
its own member latents; children get binary path ids (2p, 2p+1; bit-0 =
lower first-centroid coordinate). 0 singleton fallbacks. The tree is then
FROZEN and applied transform-only to validation and test.

## 6. Parent-child detail definition

For kernel m, parent p, child c with S16 count n_c:

- PPV_{m,p} = Σ_c n_c·PPV_{m,c} / Σ_c n_c (occupancy-weighted, over
  occupied children)
- Δ_{m,c} = PPV_{m,c} − PPV_{m,p}  → Σ_c n_c·Δ_{m,c} = 0 (exact)

Zero-occupancy regimes are excluded (no invented PPVs); min-occupancy
safeguard: regimes with count < ⌈0.01·T⌉ at level 16 are excluded, with
argmax-count fallback so the support is never empty.

## 7. Exact ANOVA/orthogonality construction

All levels are restricted to the occupied-finest-regime support S16
(occupancy is nested, so coarse exclusions are vacuous); per-kernel
quantities are computed per padding group with the group's own valid slice
and summed. Weighted population variance at K=16 telescopes exactly:

    Var16 = E2 + E4 + E8 + E16

with E_l(m) = Σ_edges (n_c/|S16|)·Δ². Cross-level inner products
⟨f_l, f_l'⟩ (6 pairs × 4,998 kernels × 761 samples, joint-count weighted)
vanish by construction — no Gram-Schmidt, no CCA anywhere (AST-enforced in
tests).

**Measured:** identity relative error **5.8e-16**, zero-sum max **2.3e-16**,
max |cross-level inner product| **1.9e-13** (machine precision).

## 8. Intrinsic level energies (train only)

| K | Real E | EnergyFrac |
|---|---|---|
| 2 | 75,148.85 | 0.308 |
| 4 | 62,702.49 | 0.257 |
| 8 | 50,224.80 | 0.206 |
| 16 | 56,167.28 | 0.230 |

HI_hier (median over train samples of total hierarchical energy) = 322.0.

## 9. Shuffled-regime null

S = 500 permutations, seed 52042: per-sample permutation of the level-16
regime sequence (preserves each sample's leaf-occupancy multiset exactly;
coarser levels derived by tree ancestry), then the identical estimator.
No hierarchy or classifier change.

## 10. Multiple-testing correction

Empirical p with the plus-one convention p = (1 + #{null ≥ real})/(S+1),
Benjamini–Hochberg FDR across the 4 tested levels.

## 11. Automatically selected L\*

| K | Real E | Null mean | Null p95 | p | q | keep |
|---|---|---|---|---|---|---|
| 2 | 75,148.85 | 2,479.47 | 2,545.69 | 0.00200 | 0.00200 | ✓ |
| 4 | 62,702.49 | 4,362.71 | 4,453.53 | 0.00200 | 0.00200 | ✓ |
| 8 | 50,224.80 | 5,442.40 | 5,541.92 | 0.00200 | 0.00200 | ✓ |
| 16 | 56,167.28 | 7,191.98 | 7,292.66 | 0.00200 | 0.00200 | ✓ |

Every level exceeds the null 95th percentile by 7.7–29.5×; **L\* = 16**,
frozen before any classifier was evaluated. Real-vs-null surplus is
level-flat (≈9–14× on the finer levels), i.e. the structure is not
concentrated at one resolution.

## 12. Fixed H capacity rationale

B_H = 4,998 is PREDECLARED: equal to flat H, making [G‖H_hier] (9,996
features) capacity-identical to [G‖H_flat]. The budget is not claimed
optimal; it exists for the fair comparison.

## 13. Energy-based budget allocation

w_l = E_l/ΣE over retained levels; largest-remainder integer rounding
(floor + largest fractional remainder, tie-break lower K first); sums to
exactly 4,998:

    K=2: 1,538   K=4: 1,283   K=8: 1,028   K=16: 1,149   (Σ = 4,998)

## 14. Label-free within-level selection

Per candidate (kernel m, child c, level l): e_{m,c,l} = mean over TRAIN
samples of π_c·Δ², computed EXACTLY inside the chain per padding group.
Top-e_l carriers per level; deterministic ties by (kernel, child id).
Selected counts: {2: 1538, 4: 1283, 8: 1028, 16: 1149} — exactly the
budget. No labels, no validation, no test, no classifier coefficients.

## 15. Leakage audit

- No label-like argument exists in either intrinsic module (AST test).
- Poisoning validation signals, test signals, or labels leaves the tree,
  energies, null, L\*, budgets, and selected feature ids unchanged (tests).
- Val/test get the frozen tree transform-only; L\* and the feature set were
  frozen before any classifier ran.
- No existing experiment artifacts were modified (test-enforced).

## 16. Baseline results

| Model | Stored | Reproduced | Diff |
|---|---|---|---|
| M0: MiniROCKET (full 9,996) | 0.7539 | **0.7543** | 0.0004 ✓ |
| R2: flat [G‖H] | 0.7551 | **0.7568** | 0.0017 ✓ |

Both gates pass within the canonical 0.002 reproduction tolerance.

## 17. Hierarchical capacity-controlled result

**PRIMARY [G ‖ H_hier(B=4,998)] = 0.7299** test Macro-F1.

Optional full-detail diagnostic [G ‖ H_hier(full 149,940)] = 0.7201 —
consistent with the Haptics finding that the unconstrained bank does not
generalize even at 896 train+val rows.

## 18. Comparison with flat H

Δ(hierarchical_budget − flat) = 0.7299 − 0.7568 = **−0.0269**.
Also below MiniROCKET-only (−0.0244). This is **CASE C** of the
predeclared success criteria.

## 19. Limitations

1. One dataset, one seed; no variance estimate over seeds.
2. The energy-share allocation is one predeclared rule; the surplus is
   level-flat, so the allocation is close to uniform by construction — the
   experiment does not test whether a different (still label-free)
   allocation could close the gap.
3. Top-e per-edge selection picks strong *marginal* structural carriers;
   the flat H features are aggregation-based (per-kernel heterogeneity
   scores), which may suit Ridge's linear structure better than raw detail
   coefficients.
4. UWaveY H-bank kernels are the second half of the canonical MiniRocket
   draw — kernel identity is arbitrary in [0, 9996); results are not
   kernel-set-specific in construction but were not re-run on a permuted
   split.
5. The regime tree partitions frozen SSL latents, not raw signal values;
   the hierarchy inherits the checkpoint's inductive bias.

## 20. What the experiment establishes

- UWaveY contains statistically distinguishable regime-conditioned
  structure at **all four nested resolutions** (q = 0.002 at every level,
  7.7–29.5× above the null 95th percentile), measurable without labels,
  with an exact nested ANOVA decomposition (identity 5.8e-16, cross-level
  inner products ≤ 1.9e-13).
- The fixed-capacity pipeline works end-to-end: label-free budget
  allocation and carrier selection produce exactly 4,998 H features, and
  the complete decision chain is label-free, deterministic (two full
  independent runs produced byte-identical intrinsic outputs and
  predictions), and validation/test-free.

## 21. What it does not establish

- That the hierarchical detail representation converts its statistically
  real structure into predictive value under the current fixed-budget
  rule: at equal H capacity it scores **below** flat H (0.7299 vs 0.7568).
  Per §39, this means the current capacity-allocation mechanism did not
  exploit the detected structure sufficiently for prediction — NOT that
  the hierarchical regime structure does not exist (§8–§11 establish that
  it does).
- That L\*=16 is the optimal depth, that 4,998 is the optimal budget, or
  that structural energy predicts label relevance. Energy measures
  regime-conditioned structural variation, not class relevance.
