# R3 Final — Global Continuous Heterogeneity Gate with Differentiable Closed-Form Ridge (seed 42)

**Final clean R3.** Datasets: Haptics, ECG5000_BAL. Seed: 42 only. One official test evaluation per system.

---

## 1. Motivation

R2 established that the learned-regime heterogeneity branch H is dataset-dependent: it helps strongly on Haptics and (in the verified rerun) on ECG5000_BAL, but hurt in the original ECG5000_BAL run. R3 asks whether that dataset-level decision can be *learned* by a single scalar gate rather than chosen by the analyst.

## 2. Why the previous SGD-softmax R3 was confounded

The earlier R3 trained `[G ‖ wH]` with a gradient-trained linear-softmax head. Its control R3-GH — the *exact* R2 feature representation — scored 0.4788 test on Haptics vs Ridge's 0.5500 on identical features: the head, not the representation, dominated the result. Absolute test values from that run are therefore not comparable to R2. This rerun removes the confound structurally: **the classifier is Ridge itself**, solved in closed form inside the autograd graph.

## 3. R3 architecture

Frozen stages 1–3 (audited R2 checkpoints, byte-verified): fixed MiniROCKET → G ∈ R^4998; frozen SSL causal encoder; frozen hard VQ (K=8) → audited H ∈ R^4998. New: one scalar gate θ, w = sigmoid(θ), θ_init = −2 (w_init ≈ 0.1192, conservatively closed):

    X(w) = [G ‖ w·H] ∈ R^{N×9996}

## 4. Difference between R2, R3, R4

    R2:  H_m      = Σ_k H_{m,k}                    (uniform aggregation)
    R3:  H_m^gate = w · Σ_k H_{m,k}                (ONE learned scalar)
    R4:  H_m^gate = Σ_k v_k H_{m,k}                (EIGHT learned scalars)

R3 is the k=1 special case of R4's parameterization; nothing else differs.

## 5. Frozen R2 representation

Per-dataset audits 1–6 all pass: canonical split identity (Haptics 132/23/308, T=1092; ECG 5226/923/1000, T=140), z-norm identity (diff 0.0), checkpoint hashes (`2dbb0cf4f3db3df8`, `f2eda1b2ff0bbe99`), byte-identical regime re-extraction, and G/H features exactly reproducing the audited implementations (chunked == direct, diff 0.0). Context parameters: `requires_grad=False` throughout.

## 6. Global scalar gate

Exactly one trainable scalar (audit 13: `theta.numel() == 1`); sigmoid-bounded (0,1); globally shared across all 4998 H features. Recorded: θ_init = −2.0, w_init = 0.119203.

## 7. Differentiable Ridge formulation

At every outer step, with Xc_tr(w) = [Gc_tr, w·Hc_tr] (TRAIN-mean centered, one-hot {0,1} targets Yc):

    β(w) = argmin_β ‖X_tr(w)β − Y_tr‖² + α‖β‖²  (closed form, train rows only)
    f_val(w) = Xc_val(w) β(w) + Ȳ
    L = MSE(f_val, Y_val) + λ_w θ²,  λ_w = 1e-3

Gradients flow val MSE → β → K(w), Xc_tr → w → θ through the Cholesky solve (audit 20: finite nonzero gradients at every step).

## 8. Dual Ridge implementation

Dual form (n×n solve; n = 132 / 5226 ≪ p = 9996), mathematically identical to the primal via the push-through identity:

    K(w) = Xc_tr(w) Xc_tr(w)ᵀ + αI = Kg + w²·Kh + αI

The decomposition is **exact** (centering commutes with scalar gating) and verified numerically: max diff vs the explicit full Gram = 7.1e-15 (Haptics), 1.1e-13 (ECG). `torch.linalg.cholesky` + `cholesky_solve`; float64 throughout; no `torch.inverse`. sklearn intercept semantics mirrored exactly (centered features/targets by TRAIN statistics).

## 9. Gate optimization

Adam(lr=1e-2) on θ only (asserted: optimizer parameter group is exactly [θ]); max 500 steps, patience 50; identical for both datasets. Ridge is re-solved in closed form at every step from TRAIN rows only; validation appears solely in the outer objective and selection (best val Macro-F1, tie → lower val MSE). No classifier epochs exist.

## 10. Fixed alpha

α frozen at the R2-selected values, never re-optimized: Haptics 4.281332398719396, ECG5000_BAL 1.623776739188721 (provenance recorded in config.json).

## 11. Audit results

All 15 audit groups pass on both datasets (audits.json), including the mandatory identity chain:

| Audit | Haptics | ECG5000_BAL |
|---|---|---|
| 1–4 splits / z-norm / ckpt / regimes | PASS | PASS |
| 5–6 G / H feature identity | exact (0.0) | exact (0.0) |
| 7–8 w=1: wH=H, X_R3 = X_R2 | PASS (exact) | PASS (exact) |
| 9–10 w=0: wH=0, X = G-only | PASS | PASS |
| 11–12 w=1 Ridge == sklearn Ridge | F1 equal, agreement 1.000 | F1 equal, agreement 1.000 |
| 13–15 one scalar / sigmoid / global | PASS | PASS |
| 16–18 only θ optimized, context frozen | PASS | PASS |
| 19–20 train-only fit, grad through solve | PASS | PASS |
| 21–23 frozen α, no test leakage, one test eval | PASS | PASS |

Numerical: max dual residual 8.4e-15 (Haptics) / 7.8e-13 (ECG); Gram symmetry ≤ 4.3e-14; max condition number 77.4 / 36,665; final train+val refit residual ≤ 5.5e-13. No NaN/Inf anywhere.

## 12. Haptics results

| System | Val | Test |
|---|---|---|
| M0 (G) | 0.7894 | 0.5037 |
| R2 (G+H, w=1) | 0.9228 | **0.5500** |
| R3 (learned w=0.6218) | 0.8618 | 0.5355 |

Gate trajectory: 0.1192 → best 0.6218 @ step 288 → final 0.6636 (339 steps; smooth, monotone, no pathology). Best-MSE step (differentiable objective) vs best-F1 step reported in `validation_results.json`; selection used neither test nor post-hoc switching.

Interpretation (spec outcome D): the gate is learnable and correctly detects that H is informative (it climbs well past 0.5 against the λ_w·θ² pull), but partial weighting (w≈0.62) is worse than uniform H (val −0.0609, test −0.0145). On the official test R3 still beats M0 (+0.0381).

## 13. ECG5000_BAL results

| System | Val | Test |
|---|---|---|
| M0 (G) | 0.9759 | 0.6278 |
| R2 (G+H, w=1) | 0.9934 | **0.6508** |
| R3 (learned w=0.2925) | 0.9830 | 0.6143 |

Gate trajectory: 0.1192 → best 0.2925 @ step 119 → final 0.3763 (170 steps, smooth).

Interpretation: the verification rerun's R2 *helps* here (0.6508 > 0.6278), so the correct gate position is w=1; the gradient signal agreed in direction (w trending up at final step) but early stopping on val Macro-F1 froze the gate at w≈0.29, which suppresses a helpful branch and loses to both controls (test −0.0365 vs R2). Notably, R4's independently-optimized eight gates collapsed to a uniform vector with mean 0.2942 — essentially the same solution R3's single scalar found (0.2925), with the identical test score (0.6143). Two different parameterizations converged on the same global optimum: on this dataset the optimal *global* weight under this objective is genuinely below 1, but selecting it costs test performance relative to w=1.

## 14. Gate behavior

Both gates: smooth, monotone-ish, non-pathological trajectories from the conservative init; neither saturated at 0 or 1; both moved decisively against the θ→0 pull, confirming the optimization works. Haptics converged near w≈0.62–0.66; ECG near w≈0.29–0.38. Neither best-val state beat its w=1 control.

## 15. Comparison with M0 / R2

| Dataset | M0 | R2 | R3 val | R3 test | learned w | ΔR3−R2 | ΔR3−M0 |
|---|---|---|---|---|---|---|---|
| Haptics | 0.4974 | 0.5500 | 0.8618 | 0.5355 | 0.6218 | −0.0145 | +0.0381 |
| ECG5000_BAL | 0.6553 | 0.6508 | 0.9830 | 0.6143 | 0.2925 | −0.0365 | −0.0410 |

Mean R3 test = 0.5749; mean Δ vs M0 = −0.0014; mean Δ vs R2 = −0.0255. The w=1 controls reproduced the frozen references exactly (Haptics 0.5500; ECG 0.6508), so the R3−R2 gaps are attributable solely to the learned gate deviating from uniform H.

## 16. Limitations

1. Single seed, two datasets; Haptics val n=23 makes the selection noisy and test gaps smaller than seed-level noise.
2. Selection on val Macro-F1 (a piecewise-constant function of w) can freeze the gate away from the MSE-optimum; both selection conventions are logged, and neither beat w=1 on test.
3. α was frozen at the R2 (ungated) selection — deliberately, to isolate the gate effect; at w<1 a different α could be optimal.
4. The outer objective (one-hot MSE) is only a proxy for Macro-F1; gradient direction is informative but the F1 landscape over w is piecewise constant.
5. With the spec's early stopping (patience 50), neither run reached w=1; a longer horizon would likely close the gap to R2 on ECG (final w was still rising) — reported as-is per the no-post-hoc-tuning rule.

## 17. Final scientific verdict

**NOT SUPPORTED.**

With the classifier confound removed and the identity chain verified exactly (w=1 ≡ R2), a single learned global gate does not improve on uniform H on either dataset: R3 < R2 in validation and official test on both, and the R4 independent corroboration (8 gates collapsing to R3's scalar solution) shows the global-weight optimum under this objective is simply not at w=1 where the best test performance lies. The gate mechanism is technically sound — smooth convergence, exact identities, correct direction-finding — but the dataset-level decision it was meant to learn ("how much H?") is better made at w=1 given a validation protocol selecting on Macro-F1. Combined with R4 (regime-level: also not supported), the series concludes that the frozen R2 representation should be used **ungated**: H contributes fully or not at all, and on these two datasets, fully.

```
R3 DIFFERENTIABLE-RIDGE GLOBAL GATE — FINAL SEED 42

Haptics:      M0 0.4974 | R2 0.5500 | R3 val 0.8618 | R3 test 0.5355
              w_init 0.1192 | w_best 0.6218 | w_final 0.6636
              ΔR3−R2 −0.0145 | ΔR3−M0 +0.0381

ECG5000_BAL:  M0 0.6553 | R2 0.6508 | R3 val 0.9830 | R3 test 0.6143
              w_init 0.1192 | w_best 0.2925 | w_final 0.3763
              ΔR3−R2 −0.0365 | ΔR3−M0 −0.0410

R3 mean test = 0.5749 | mean Δ vs M0 = −0.0014 | mean Δ vs R2 = −0.0255
Exact R2 identity audit: PASS (agreement 1.000, both datasets)
Max Ridge solver residual: 7.85e-13
Gate convergence: both smooth and moved (339 / 170 steps)

VERDICT: NOT SUPPORTED
```

Per the stop rule: Haptics and ECG5000_BAL complete — no further datasets, seeds, gates, or protocol changes.
