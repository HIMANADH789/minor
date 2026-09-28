# FULL-DIMENSION CCA-ADAPTIVE GENERALIZED RIDGE ("Full Canonical Ridge") — Haptics, seed 42 — Evidence Report

**Experiment namespace**: `experiments/heramba_canonical_ridge_full_haptics_seed42/`
**Previous experiments preserved unchanged**: `results/heramba_canonical_ridge/haptics_seed42/` (117-dim Canonical Ridge, test 0.4825), `results/heramba_cca_ranked/haptics_seed42/` (hard-unique, 0.5138), `results/heramba_cca/haptics_seed42/` (full-rank CCA degeneracy).

---

## 1. Research question

When the complete original MiniROCKET (G, 4998) and HERAMBA (H, 4998) representations are preserved, does applying different ridge penalties to CCA-identified HERAMBA directions improve prediction over ordinary scalar-alpha Ridge?

## 2. Why the previous Canonical Ridge experiment was confounded

The previous experiment (test 0.4825) simultaneously tested (A) CCA-adaptive shrinkage and (B) compression of G from 4998 to 29 dimensions in the final predictive design. Any performance change could not be attributed to the shrinkage mechanism alone.

## 3. Experimental correction

The final predictive design now retains **all 9996 original features**: `Z = [G_full(4998) ‖ H_cca(29) ‖ H_perp(4969)]`, where H is re-expressed in a train-only orthonormal basis Q = [Q_cca | Q_perp] obtained by completing the 29 CCA-identified raw-H directions to a full basis of R^4998. This is a **change of basis, not compression**: reconstruction error 8.0e-16 (train), 7.6e-16 (val), 7.6e-16 (test), far below the 1e-10 requirement. PCA/CCA ran only on train rows; G keeps all 4998 original MiniROCKET features.

**PCA/CCA dimensionality reduction was used only to identify CCA directions. The final predictive model retained the complete 4998-dimensional MiniROCKET and 4998-dimensional HERAMBA representations.**

## 4. Data and split

Haptics, canonical split: train 132 / validation 23 / test 308; T=1092; 5 classes; sorted-label encoding; per-sample z-normalization (canonical pipeline, untouched). Labels via `experiments/external_stack_generalization/data.load_dataset("Haptics")`.

## 5. G/H feature sources

Frozen, identity-verified banks (loaded read-only):
- `C:/temp/results/haptics_inference_banks_G_trva.npy` (155×4998), `..._G_te.npy` (308×4998) — canonical MiniROCKET
- `C:/temp/results/haptics_inference_banks_H_trva.npy` (155×4998), `..._H_te.npy` (308×4998) — HERAMBA/R2 H bank

No feature was regenerated; canonical MiniROCKET and HERAMBA extraction were not modified.

## 6. Rank-controlled CCA discovery stage

Train-only RankPCA with the predeclared 95%-variance rank rule (identical to the ranked-CCA experiment): G rank 29, H rank 88. CCA on (G_low, H_low): K=29 canonical directions, rho in [0.674, 0.839], mean 0.780 — same spectrum as the ranked experiment (no degeneracy). No rho thresholding, no H_unique construction, every direction retained.

## 7. Full-dimensional basis construction

Raw-space directions `D = diag(ok) @ comp_H^T @ diag(1/hsd) @ B` (verified: raw scores correlate 1.0 with CCA-projected H_low scores). Orthonormalized via full QR with deterministic signs: `Q = [Q_cca(4998×29) | Q_perp(4998×4969)]`, `max|Q^T Q − I| < 1e-10` (asserted; measured ~1e-15). Reconstruction: `H_c QQ^T = H_c` to machine precision on all three splits.

## 8. Mathematical definition of the uniform model

`min_W ||Y − Z_c W||² + W^T (alpha_base · diag(lam0)) W` with lam0 = ones(9996), Y one-hot (classes fixed on train), Z_c train-centered. Solved by the dual (n-space) solver. Because Z is a rotation of [G‖H], this model is information-identical to ordinary full ridge at the same alpha.

## 9. Mathematical definition of the proposed adaptive model

Same objective and design; lam0 = [1_{G(4998)}, 1+delta_1..29, 1_{perp(4969)}] with delta_k = gamma·rho_k², **gamma = 1.0 fixed**. alpha_k = alpha_base·(1+rho_k²) along each canonical coordinate; alpha_base on G, on H_perp, and on every retained direction (nothing discarded, no threshold).

## 10. Exact alpha schedule

alpha_k/alpha_base = 1+rho_k² ∈ [1.4543, 1.7041] across the 29 CCA coordinates (rho 0.674–0.839); 1.0 for G (4998) and H_perp (4969). Full schedule in `alpha_schedule.csv` and `canonical_ridge_diagnostics.csv`; visual confirmation in `figures/rho_vs_alpha.pdf/png`. The formula is a **proposed** rule of this project, not a literature-established Canonical Ridge equation.

## 11. Train-only GCV

alpha_base ∈ logspace(−4, 4, 81) selected by GCV = n·RSS/(n−df)² with exact df = Σ theta/(theta+alpha) from one eigendecomposition of the n×n matrix S = Z_c diag(1/lam0) Z_c^T. Fit rows: train (132) only. No validation or test input anywhere in selection. Both uniform and adaptive selected the grid boundary alpha = 1e-4 (df 131.0 of 132 — the interpolating regime).

## 12. Basis-invariance validation

Ordinary RidgeClassifierCV on raw [G‖H] vs rotated [G‖H_cca‖H_perp] (same rows, same alphas grid): max per-sample decision difference **3.57e-14**, identical Macro-F1 0.5500. Dual-solver twin: **7.41e-14**. Tolerance 1e-8 — **passed**. This proves the transform preserves the full representation; the gate is enforced in the runner (`basis_invariance_report.json`, `figures/basis_invariance.*`).

## 13. Identity-gate results

| Gate | Model | Reproduced | Expected |
|---|---|---|---|
| A | MiniROCKET (G, RidgeClassifierCV train+val) | 0.5037 | ≈0.5037 |
| B | Raw [G‖H] | 0.5500 | ≈0.5500 |
| C | [G‖H_unique] (previous pipeline, imported unchanged) | 0.5138 | ≈0.5138 |

All exact. Gates ran before any new-model fitting.

## 14. Permutation-null CCA results

Diagnostic only (S=500, seed 52042, train rows; never touched any fit): real rho 0.674–0.839 vs null mean up to 0.808 — the null itself is inflated at n=132 with K=29, so raw rho values carry finite-sample inflation. 28/29 components have empirical p ≤ 0.05 (FDR-adjusted q in `canonical_correlations.csv`). G/H alignment is real but over-stated by raw rho; the null did not alter the model.

## 15. Final test results

| Method | Representation | Ridge type | alpha_base | Test Macro-F1 | Val Macro-F1 |
|---|---|---|---|---|---|
| MiniROCKET | G (4998) | scalar RidgeClassifierCV (train+val) | 4.281 | **0.5037** | — |
| Raw full G+H | [G‖H] (9996) | scalar RidgeClassifierCV (train+val) | 4.281 | **0.5500** | — |
| Hard CCA unique | [G‖H_unique] (5086) | scalar RidgeClassifierCV (train+val) | 1.624 | **0.5138** | — |
| Uniform rotated full Ridge | [G‖H_cca‖H_perp] (9996) | generalized lam0=1 (dual, train-only GCV) | 1e-4 | 0.5234 | 0.6277 |
| **Proposed full Canonical Ridge** | [G‖H_cca‖H_perp] (9996) | CCA-adaptive lam0 (dual, train-only GCV) | 1e-4 | **0.5319** | 0.6733 |
| Proposed full Canonical Ridge (final fit) | same | dual, alpha frozen at train-only GCV value | 1e-4 | 0.4780 | — |

Percentage-point deltas of the primary result (train-only GCV arm):
- vs MiniROCKET: **+2.82**
- vs Raw full G+H: **−1.81**
- vs Hard CCA unique: **+1.81**
- vs Uniform rotated Ridge: **+0.85**

Mechanism ablation: adaptive > uniform (+0.85 pp test; val 0.6733 vs 0.6277) with the predeclared schedule — the rho-informed penalties do change behavior in the intended direction. However this contrast is confounded in the strict sense that both dual arms share the GCV-selected alpha=1e-4 (interpolating regime, df 131/132), and the uniform arm's alpha is optimized for the adaptive objective, not independently.

## 16. Numerical stability

p=9996, n=132; all solves in dual n-space (no p×p system, no explicit inverse). At the selected alpha: rank(S)=131, eigenvalues of the regularized system within [alpha, 332.5+alpha]; condition number 3.33e6 (uniform) / 1.02e3 (adaptive); Cholesky succeeded for both (no fallback); coefficients finite; GCV RSS cross-checked against the fitted-residual identity. Dual solver verified against the primal brute-force reference to ≤1e-8 on synthetic problems (unit test 14).

## 17. Leakage audit

- PCA/CCA/basis: fitted on the first 132 rows only (unit tests 4–5: fits invariant to poisoned validation rows).
- Test matrices never enter any fit (unit test 15: poisoning test rows changes transforms only affinely, by a constant shift; fits unchanged).
- GCV: train rows only (asserted in test 15); validation used only for reporting (D/E arms) and as frozen-alpha rows in the clearly-labeled final-fit arm.
- Classes fixed on train; deterministic re-runs identical (test 16).
- No gamma/rho/tau/mixing search anywhere; gamma=1.0 predeclared; no thresholding of canonical correlations.

## 18. Interpretation

Measured outcomes only. (i) The full-dimensional correction moved the mechanism result from 0.4825 (confounded, 117-dim) to 0.5319 — recovering most of the gap to raw concatenation (0.5500) and beating MiniROCKET, so the earlier negative result was substantially attributable to G compression (spec decision-tree CASE 1 leaning CASE 3 vs uniform). (ii) Under identical 9996-dim designs and the same train-only GCV, the adaptive penalty beat its uniform twin by +0.85 pp — positive but small, single-seed, single-dataset, and inside the ±0.028 test-set bootstrap noise of M0 established earlier; it is not evidence of a reliable mechanism advantage. (iii) No configuration beat the canonical raw [G‖H] Ridge (0.5500) on this dataset/seed. All differences (0.85–2.82 pp) are well within the sampling uncertainty documented in the inferential study (e.g., paired bootstrap CI half-widths of ±0.04).

## 19. Limitations

Single dataset (Haptics), single seed (42); cross-seed uncertainty not established. GCV optimizes one-hot squared error, not Macro-F1, and selected a grid-boundary alpha (1e-4) in the interpolating regime — df 131/132 means both dual arms are near-interpolating, which limits the meaningfulness of shrinkage-structure comparisons at the selected alpha. The adaptive-vs-uniform contrast shares one alpha selected under the adaptive objective. The permutation-null CCA shows raw rho values are inflated at n=132, so the "shared/unique" language of earlier diagnostics should be read with that inflation in mind. The final-fit arm demonstrates sensitivity of the interpolating regime to added rows (0.4780), which cautions against over-reading any single arm. One Haptics seed-42 run establishes no general superiority for any method.

## 20. Next experiment

Natural follow-ups, in order of information value: (1) repeat the D-vs-E mechanism contrast across seeds 42–44 on Haptics to establish variability; (2) restrict the GCV alpha grid to the non-interpolating regime (df bounded well below n) so shrinkage structure is meaningfully engaged; (3) port the identical protocol to UWaveY (the other dataset with established R5 gains); (4) a supervised-discovery variant (PLS directions instead of PCA+CCA) as a separate experiment. None of these was run here to keep this experiment atomic.

---

**Reproducibility**: `python experiments/heramba_canonical_ridge_full_haptics_seed42/runner.py` (~5 s; frozen banks); tests `python -m pytest tests/test_canonical_ridge_full.py -q` (20 passed; full suite 534 passed). RNG: permutation seed 52042; bootstrap-independent; no other stochastic elements.

**This experiment does not establish general superiority from one Haptics seed-42 run.**
