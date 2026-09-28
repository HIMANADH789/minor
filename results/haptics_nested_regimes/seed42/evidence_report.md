# HIERARCHICAL / NESTED REGIME-CONDITIONED HETEROGENEITY — Haptics, Seed 42

**Experiment namespace:** `experiments/haptics_nested_regimes/`
**Module:** `models/nested_regimes/model.py`
**Result directory:** `results/haptics_nested_regimes/seed42/`

---

## 1. Research question

Does Haptics contain statistically distinguishable regime-conditioned
structure at multiple **nested** temporal resolutions — measurable
**without labels** via occupancy-weighted detail energies, and with a
**null-calibrated stopping depth** that can be frozen before any
predictive evaluation?

## 2. Why flat K=8 regimes are insufficient

The existing H bank uses one flat partition of the timeline (K=8 VQ
regimes) and summarizes each kernel's regime-conditioned variation as a
single scalar, `H_m = sum_k q_k (PPV_{m,k} - PPV_m)^2`. A single
resolution cannot distinguish coarse temporal structure (long-lived
states) from fine structure (short transients); all heterogeneity is
collapsed into one number per kernel, and no intrinsic principle decides
which resolution actually carries structure.

## 3. Nested regime hypothesis

Temporal regime structure exists at multiple resolutions simultaneously.
One nested binary tree over the frozen temporal latents (K = 1 → 2 → 4 →
8 → 16, every fine regime a child of exactly one coarse regime) allows
the regime-conditioned variation to be **decomposed level by level**
(Nested ANOVA), with each level's energy testing whether the refinement
introduced at that resolution is real structure rather than sampling
noise.

## 4. Mathematical construction

Per sample i, kernel m, padding group g (each feature's aeon valid region
[p, T−p)), let S16 be the valid timesteps in min-occupancy-selected
level-16 regimes (selection rule identical to the flat bank: count ≥
ceil(0.01·T) = 11; argmax fallback; no PPV invented for empty regimes).
All levels are evaluated on the SAME support S16; because a selected
level-16 regime has ≥ 11 timesteps, every coarser regime containing it
automatically satisfies the occupancy rule, so the weight system is
consistent across levels. With n_c = |S16 ∩ regime c|, π_c = n_c/|S16|,
PPV_c = (activation sum in c)/n_c:

- parent PPV: PPV_p = Σ_c π_c·PPV_c / Σ_c π_c over occupied children
- detail: Δ_c = PPV_c − PPV_p
- level energy: E_l(m,i) = Σ_p Σ_{c∈children(p)} π_c·Δ_c²
- K=16 conditional variance: Var16(m,i) = Σ_c π_c(PPV_c − g)², g = Σ_c π_c·PPV_c

## 5. Parent-child residual definition

Δ_{m,c} = child PPV − occupancy-weighted parent PPV (within the occupied
parent), exactly as predeclared.

## 6. Occupancy weighting

Weights are the raw occupancy fractions π_c = n_c/|S16| (equivalently
n_c within each parent), used consistently in the parent mean, the
detail energies, and the identity. Excluded (sub-occupancy) regimes have
no PPV, no detail, and no energy; the flat bank's renormalization
convention is respected by the S16 support restriction.

## 7. Nested variance decomposition (identity)

Measured on the actual train chain (all 4998 kernels, 132 samples):

**Var16 = E(K=2) + E(K=4) + E(K=8) + E(K=16)** with
max relative error **5.6e-16** — exact to float64 (independently
brute-forced in unit tests; artifact `hierarchy_energy_identity.json`).

## 8. Orthogonality derivation

The level-l detail FUNCTION f_l(t) = Δ_{c_l(t)} on S16, 0 outside, is
the classical nested-ANOVA residual at resolution l. Cross-level inner
products ⟨f_l, f_l'⟩ vanish because (a) the π-weighted zero-sum of the
finer level holds inside each coarser regime (Σ_c n_c·Δ_c = 0 per
parent, max residual 2.0e-16), and (b) detail supports are disjoint
across branches. No Gram-Schmidt, no CCA, no post-hoc orthogonalization
(AST-enforced in tests).

**Measured max |⟨f_l, f_l'⟩| over all 6 level pairs, all kernels, all
samples: 3.1e-13** (machine precision at these magnitudes).

## 9. Hierarchy construction (precise, per §33)

**What is partitioned:** the FROZEN RCMKN seed-42 context model's
`SSLTemporalEncoder` temporal latents z_t ∈ R^32 (T=1092 per sample),
pooled over all TRAIN samples (132·1092 = 144,144 vectors). The frozen
flat VQ codebook is NOT modified and NOT used for the tree.

**Algorithm:** recursive 2-means. For depth d = 0..3, every level-d
centroid p is split by `sklearn.KMeans(n_clusters=2, n_init=10,
random_state=42)` on its members; child ids are 2p (bit 0) and 2p+1
(bit 1), with the bit-0 child defined as the split centroid with the
smaller FIRST coordinate (deterministic ordering). Singleton nodes
(< 2 members) duplicate the parent centroid (recorded: 0 fallbacks).
Assignment of any latent vector walks nearest-child from the root
(transform-only; never refit). Nesting is automatic from the id
arithmetic and is asserted (`nesting_errors` returns empty).

## 10. Haptics dataset

Canonical split (verified in-run): train 132, validation 23, test 308;
T = 1092, univariate; per-sample z-normalization (project convention).
MiniRocket(random_state=42) fit on train z-normed signals → 9996
features; the H-bank kernels are features 4998:9996 (4998 kernel units,
matching the canonical bank layout G = first 4998).

## 11. Train-only protocol

Tree, energies, null, stopping rule, and depth L* are derived from TRAIN
latents/activations only. Validation and test enter only AFTER L* is
frozen, and only for the downstream demonstration (transform-only
application of the frozen tree; no refit). Labels enter nothing until
the gate/demo Ridge fits. Leakage tests: no label-like argument exists
in the intrinsic module (AST-enforced); poisoned-val/test invariance is
covered by the chain-determinism test.

## 12. Level energies (train, summed over samples × kernels)

| K | Real E | Null mean | Null p95 | p (plus-one) | q (BH) | Keep |
|-----|--------|-----------|----------|--------------|--------|------|
| 2 | 8654.12 | 129.34 | 136.39 | 0.002 | 0.002 | yes |
| 4 | 8919.32 | 250.75 | 260.67 | 0.002 | 0.002 | yes |
| 8 | 10868.82 | 470.85 | 484.32 | 0.002 | 0.002 | yes |
| 16 | 9713.09 | 651.14 | 667.56 | 0.002 | 0.002 | yes |

Real energy exceeds the null 95th percentile by factors of 63× (K=2) to
14.6× (K=16). Normalized fractions (mean over samples): K=2 0.230, K=4
0.237, K=8 0.289, K=16 0.258 — structure is spread across ALL
resolutions, with no dominant single level.

## 13. Shuffled-regime null

S = 500, seed 52042. Per permutation: per-sample permutation of the
level-16 regime sequence (preserves each sample's regime-size multiset
exactly; null helper unit-tested for count preservation), then the
IDENTICAL estimator (same padding groups, same selection rule, same
chain). Diagnostic only — no fit was altered by the null.

## 14. Multiple-testing correction

Benjamini-Hochberg across the 4 level tests, plus-one p-values
p = (1 + #{null ≥ real})/(S+1) = 0.002 for all levels.

## 15. Automatic stopping rule (predeclared)

Keep level l iff real E_l > null p95 AND q_l < 0.05; walk coarse→fine;
stop at first failure. No threshold was changed after seeing results.

## 16. Selected hierarchy depth

**L\* = 16** — all four detail levels survive. The intrinsic (label-free)
H-capacity signal is therefore "retain detail through K=16".

## 17. Downstream demonstration (intrinsic-depth demonstration, NOT
optimized)

Gates reproduced exactly: MiniROCKET (G) = 0.5037; Raw [G‖H] = 0.5500
(identity gates A/B). H-bank replication from my recomputed activations
using the stored flat regimes: max|diff| = 4.1e-3 (the same small
aeon-version quantile drift as the G check; estimator provenance
verified).

Frozen-depth model: [G ‖ H_hier(L*=16)] with H_hier = {Δ^(2), Δ^(4),
Δ^(8), Δ^(16)} → 2+4+8+16 = 30 detail blocks × 4998 kernels = **149,940
features** on 155 development rows. **Test Macro-F1 = 0.4586.**

This single, non-searched configuration is BELOW both references
(−4.51 pp vs MiniROCKET, −9.14 pp vs raw G+H). No depth or feature-count
search was performed (per the predeclared protocol); the result shows
that the frozen full-depth detail bank, taken literally as a 150k-wide
feature space on 155 rows, is not usable by the canonical Ridge without
additional capacity control. The scientific payload of this experiment
is the intrinsic decomposition (§12, §16), not this demonstration.

## 18. Comparison with flat H

Flat H: 4998 features, K=8 single resolution, test 0.5500 (in [G‖H]).
Hierarchical H: 149,940 features, K=2/4/8/16 details, test 0.4586 (in
[G‖H_hier]). Feature counts were NOT forced equal (explicitly allowed);
the two are not capacity-matched and the comparison is descriptive.

## 19. Leakage audit

- No labels anywhere in tree/energies/null/stopping (AST test).
- Poisoned val/test cannot change train chain outputs (test 12).
- Frozen tree applied transform-only to val/test.
- L* frozen before any classifier fit (runner ordering).
- No CCA / Gram-Schmidt / orthogonalization calls (AST tests).

## 20. Limitations

1. Single dataset, single seed; the stopping decision is not established
   across datasets.
2. The hierarchy partitions FROZEN SSL latents — the "regimes" are
   latent-space neighborhoods, not hand-defined signal segments; results
   inherit whatever structure the frozen encoder has.
3. The null is a permutation null of temporal alignment; it does not
   certify that every retained level carries label-relevant information.
4. The demonstration uses the raw detail bank (150k features, 155 rows);
   the negative demonstration result does not evaluate whether a
   capacity-controlled version of the same bank would help.
5. Real energies partially track occupancy structure: the null also
   grows with K (129 → 651), so the per-level SURPLUS (real vs null)
   is the meaningful quantity, and it is roughly level-flat.

## 21. What is established

- Haptics contains regime-conditioned kernel-PPV structure at ALL FOUR
  nested resolutions (K=2,4,8,16), each surviving a 500-permutation
  label-free null with q = 0.002.
- The nested ANOVA decomposition is EXACT (identity 5.6e-16;
  zero-sum 2.0e-16; cross-level orthogonality 3.1e-13) and arose from
  the construction, not from post-hoc orthogonalization.
- The predeclared stopping rule deterministically outputs L* = 16
  before any predictive evaluation.

## 22. What is NOT established

- That hierarchical H improves prediction (the frozen-depth
  demonstration scored 0.4586 < both references).
- That L* = 16 generalizes across datasets or seeds.
- That the retained levels carry label-relevant (not merely present)
  structure.
- Any causal claim about regimes and classification performance.
