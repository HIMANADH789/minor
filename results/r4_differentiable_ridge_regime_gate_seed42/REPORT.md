# R4 — Differentiable-Ridge Regime-Level Heterogeneity Gate (seed 42)

**Final architecture experiment.** Datasets: Haptics, ECG5000_BAL. Seed: 42 only. One official test evaluation per system.

---

## 1. Motivation

R2 established that learned-regime PPV heterogeneity (H) helps on some datasets and hurts on others (Haptics +0.053 over M0; ECG5000_BAL negative in the first run). R3 tested whether a single learned scalar gate could make H optional, but its gradient-trained softmax classifier introduced a structural confound: at w=1 the softmax head did not reproduce the R2 Ridge result on identical features (R3-GH Haptics test 0.4788 vs R2 0.5500). R4 asks the finer, final question —

> Can the contribution of regime heterogeneity be learned **separately for each of the K=8 latent temporal regimes**, with the classifier confound eliminated by making the classifier Ridge itself?

## 2. R2 baseline architecture (frozen)

Exactly the audited seed-42 R2 pipeline, byte-verified per dataset:

- fixed MiniROCKET (aeon, `random_state=42`, train-only fit) → raw responses r_m(t), canonical thresholds b_m, per-feature valid regions;
- frozen SSL causal multi-scale encoder (Conv1D k=3/d=1/C32 → k=5/d=2/C32 → k=7/d=4/C64 → k=9/d=8/C64 → projection d=32; SSL: mask ratio 10%, span 16, masked MSE);
- frozen hard VQ, K=8, straight-through assignment: k_t = argmin_k ||z_t − c_k||².

Nothing in stages 1–3 was retrained, retuned, or reconfigured. Checkpoint hashes verified (`2dbb0cf4f3db3df8` Haptics, `f2eda1b2ff0bbe99` ECG5000_BAL); regime re-extraction is byte-identical (audit 4).

## 3. R4 architecture

The only new mechanisms:

1. **Eight regime-level gates** (stage 4): v_k = sigmoid(θ_k), θ_k init −2.0 (v ≈ 0.1192, conservatively closed), one scalar per VQ regime.
2. **A differentiable closed-form Ridge classifier** (stage 5), replacing the R3 softmax head entirely.

Everything else is the frozen representation.

## 4. Mathematical formulation

Per-regime contributions (frozen, exact audited semantics incl. padding groups, min-occupancy, renormalized q):

- H_{m,k} = q_k (PPV_{m,k} − PPV_m)², with PPV_{m,k} = Σ_{t∈V_m, k_t=k} 1[r_m(t)>b_m] / n_k, q_k = n_k/|V_m| (renormalized over occupancy-selected regimes).
- R2:  H_m = Σ_k H_{m,k}
- R3:  H_m^gate = w Σ_k H_{m,k}  (one scalar — **not** R4)
- **R4: H_m^gate(v) = Σ_{k=1}^{8} v_k H_{m,k}** (eight scalars)

X(v) = [G ‖ H^gate(v)] ∈ R^{N×9996}. The defining computational graph:

```
θ (8) → sigmoid → v → H_gated(v) → X_train(v)
      → closed-form dual Ridge (train rows only):
          K(v) = Xc_tr(v) Xc_tr(v)ᵀ + αI;   a(v) = K(v)⁻¹ Yc;
          β(v) = Xc_tr(v)ᵀ a(v)             [≡ (XᵀX+αI)⁻¹XᵀY]
      → f_val(v) = Xc_val(v) β(v) + Ȳ
      → L = MSE(f_val, Y_val one-hot) + λ_v Σθ_k²   → grad → θ
```

β depends on v through both K(v) and Xc_tr(v); predictions therefore depend on v; the validation MSE provides exact gradients to θ_k through the Cholesky solve. **Dual form** (n×n solve) is used because n ≪ p (Haptics 132, ECG 5226 vs p=9996); documented as mathematically identical to the primal via the push-through identity.

## 5. Regime decomposition

`compute_H_contributions` mirrors the audited `heterogeneity_features` (padding-group slicing, valid regions, min_count = ceil(0.01·T) selection, single-regime fallback, renormalized q) but returns the unaggregated (N, 4998, 8) tensor. Verified:

- independent per-position recompute: max diff 2.16e-9 (Haptics) / 1.70e-9 (ECG) — float32 storage rounding;
- Σ_k H_{m,k} vs audited H_m: max diff 2.98e-8 / 4.47e-8 = 1–1.5 ulp of max|H| (bound ≤ 8 ulps; identity exact in float64 — unit-tested).

## 6. Gate parameterization

Eight scalars, sigmoid-bounded (0,1), init θ=−2 ⇒ v≈0.1192 < 0.5 (conservative). No per-kernel / per-sample / per-timestep gates. Record: θ_init = −2.0, v_init = 0.1192 (both datasets).

## 7. Loss

During gate optimization (θ only):

L(θ) = MSE(f_val(v), Y_val) + λ_v Σ_k θ_k², λ_v = 1e-3.

The penalty is exactly λ_v·Σθ_k² (audit-checked against the closed-form value 1e-3·10.5625). No CE loss, no classifier weight decay, no classifier optimizer anywhere. Final model: closed-form Ridge refit on train+val at the frozen best-v (R2 convention), α frozen.

## 8. Training protocol

Identical for both datasets, no search: Adam(lr=1e-2) on θ only (asserted: the optimizer's parameter group is exactly [θ]); max 500 outer steps, patience 50; Ridge re-solved in closed form at every step from TRAIN rows only; validation used solely for the outer loss and best-step selection (best val Macro-F1, tie → lower val MSE); test untouched until all audits pass; one test evaluation per system. Gate trajectories: `training_logs/gate_trajectory.{json,csv}` per dataset.

## 9. Fairness / budget

Representation exactly 9996 dims in all systems (4998 G + 4998 H or gated H). Trainable parameters: **R4 = 8** (θ only) + no classifier parameters (Ridge is closed-form). Frozen parameters: 61,414 (context model) + the fixed MiniROCKET bank. Gates add negligible capacity; no budget, feature, or allocation change.

## 10. Audit results

All audits pass on both datasets (per-dataset `audits.json`; key values):

| Audit | Haptics | ECG5000_BAL |
|---|---|---|
| 1 dataset/split identity | PASS (132/23/308, T=1092, 5 classes) | PASS (5226/923/1000, T=140, 5 classes) |
| 2 z-norm identity | PASS (diff 0.0) | PASS (diff 0.0) |
| 3 checkpoint provenance | PASS | PASS |
| 4 regime re-extraction | PASS (byte-identical) | PASS (byte-identical) |
| 5 MiniRocket identity | PASS (diff 0.0) | PASS (diff 0.0) |
| 6/7 H_{m,k} independent recompute | 2.16e-9 | 1.70e-9 |
| 8 Σ_k H_{m,k} = H_m | 2.98e-8 | 4.47e-8 |
| v=1 → H / X identity | 1.77e-8 | 1.30e-8 |
| v=1 Ridge == sklearn Ridge (frozen α) | F1 equal, agreement 1.000 | F1 equal, agreement 1.000 |
| v=0 → H_gated = 0 | PASS (<1e-12) | PASS (<1e-12) |
| softmax classifier removed (tokenized scan) | PASS | PASS |
| only θ optimized; context frozen | PASS | PASS |
| α frozen at R2 selection (4.2813 / 1.6238) | PASS | PASS |
| train-only Ridge during gate opt | PASS | PASS |
| gradients finite/nonzero through solve | PASS (all steps) | PASS (all steps) |
| single test eval per system | PASS | PASS |

Solver diagnostics: max residual ‖Kα−Yc‖∞ = 1.0e-14 (Haptics) / 8.5e-13 (ECG); max condition number 77.2 / 36,674; final train+val refit residual 5.8e-15 / 5.9e-13; float64 throughout; no NaN/Inf.

## 11. Regime statistics (train+val scope)

**Haptics** (VQ: 8/8 codes active, entropy 0.935, perplexity 6.99):

| k | occupancy | frac | H-contribution frac | v_best |
|---|---|---|---|---|
| 0 | 39,972 | 0.236 | 0.121 | 0.4943 |
| 1 | 1,564 | 0.009 | 0.006 | 0.4978 |
| 2 | 24,206 | 0.143 | 0.100 | 0.5187 |
| 3 | 19,126 | 0.113 | 0.125 | 0.6349 |
| 4 | 23,225 | 0.137 | 0.135 | 0.5318 |
| 5 | 20,936 | 0.124 | 0.166 | 0.5794 |
| 6 | 17,685 | 0.104 | 0.227 | 0.5866 |
| 7 | 22,546 | 0.133 | 0.119 | 0.5019 |

**ECG5000_BAL** (VQ: 8/8 active, entropy 0.981, perplexity 7.69):

| k | occupancy | frac | H-contribution frac | v_best |
|---|---|---|---|---|
| 0 | 109,698 | 0.127 | 0.146 | 0.2950 |
| 1 | 131,530 | 0.153 | 0.160 | 0.2952 |
| 2 | 152,078 | 0.177 | 0.117 | 0.2967 |
| 3 | 48,946 | 0.057 | 0.084 | 0.2918 |
| 4 | 116,837 | 0.136 | 0.089 | 0.2935 |
| 5 | 81,749 | 0.095 | 0.115 | 0.2921 |
| 6 | 110,848 | 0.129 | 0.134 | 0.2946 |
| 7 | 109,174 | 0.127 | 0.154 | 0.2950 |

## 12. Gate trajectories

- **Haptics**: smooth monotone rise from 0.1192, no oscillation or saturation; mild differentiation emerges (k6/k3/k5 highest — the same regimes with the largest raw H contributions; k1, the 0.9%-occupancy regime, stays lowest among the mid group). v_best = v_final = the step-499 state (best val found at the step budget ⇒ selection did not truncate a better gate state).
- **ECG5000_BAL**: all eight gates move as one — 0.1192 → ~0.294 at best-val (step 129) → ~0.363 at the final step — with dispersion std 0.0015 / max−min 0.005. This is a single *global* H-weight learned by an 8-fold degenerate parameterization, not regime-level selection.

## 13. Validation results (trainva-fit Ridge Macro-F1)

| Dataset | G | G+H (v=1) | R4 (learned) |
|---|---|---|---|
| Haptics | 0.7894 | **0.9228** | 0.8618 |
| ECG5000_BAL | 0.9759 | **0.9934** | 0.9830 |

Deltas: R4 vs G+H = −0.0609 (Haptics), −0.0104 (ECG). R4 vs G = +0.0724 (Haptics), +0.0071 (ECG).

## 14. Official test results (one evaluation each)

| Dataset | M0 (G) | R2 (G+H, v=1) | R4 |
|---|---|---|---|
| Haptics | 0.5037 | **0.5500** | 0.5324 |
| ECG5000_BAL | 0.6278 | **0.6508** | 0.6143 |

Control-B identity: the v=1 control reproduces the frozen R2 references **exactly** — Haptics 0.5500 (spec value) and ECG5000_BAL 0.6508 (the repo-recorded R2 rerun; the user-frozen historical reference 0.6409 is retained in config). The comparison R4 vs controls is therefore structurally clean: same features, same split, same closed-form Ridge family, only the 8 gates differ.

## 15. Haptics analysis

Gates differentiated mildly (std 0.048, max−min 0.141) with the ordering correlated with raw H-contribution share (k6 0.227→v 0.587; k3 0.125→v 0.635; k1 0.006→v 0.498). But the learned gated representation is strictly worse than uniform H in validation (0.8618 < 0.9228) and test (0.5324 < 0.5500). Interpretation: the H block on Haptics works as a *joint* statistic; down-weighting individual regimes early in training reduces information faster than reallocation recovers it. The optimizer's best val state was simply "part-way to opening" (mean v 0.54 at best step 499 = budget end), i.e., it did not find a regime-selective configuration that beats uniform. Outcome C of the spec's taxonomy: differentiation learnable but not predictively useful.

## 16. ECG5000_BAL analysis

Zero regime differentiation (std 0.0015): the eight gates degenerate to one global weight ≈ 0.294 at best-val. R4 then behaves like an R3-style partial gate and inherits the R3 lesson quantitatively: partial suppression of H loses more than it saves (val 0.9830 < 0.9934; test 0.6143 < 0.6508 = R2; and < the G-only control's test 0.6278). Note this dataset's R2 rerun is *positive* vs M0 (0.6508 vs 0.6278 on identical protocol), so the optimal dataset-level decision here was v=1 — the gradient signal agreed (gates trending up at final step 0.363) but early stopping on val Macro-F1 selected the state at v≈0.294. Outcome B: uniform behavior, no regime-level benefit.

## 17. Cross-dataset comparison

| Dataset | M0 | R2 | R4-G val | R4-GH val | R4 val | v1…v8 (best) | R4-G test | R4-GH test | R4 test |
|---|---|---|---|---|---|---|---|---|---|
| Haptics | 0.4974 | 0.5500 | 0.7894 | 0.9228 | 0.8618 | 0.494 0.498 0.519 0.635 0.532 0.579 0.587 0.502 | 0.5037 | 0.5500 | 0.5324 |
| ECG5000_BAL | 0.6553 | 0.6409 | 0.9759 | 0.9934 | 0.9830 | 0.295 0.295 0.297 0.292 0.294 0.292 0.295 0.295 | 0.6278 | 0.6508 | 0.6143 |

Compact deltas:

| Dataset | R4−R2 (pp) | R4−M0 (pp) | gate mean | gate std | min | max |
|---|---|---|---|---|---|---|
| Haptics | −1.76 | +3.50 | 0.5432 | 0.0481 | 0.4943 | 0.6349 |
| ECG5000_BAL | −2.66 | −4.10 | 0.2942 | 0.0015 | 0.2918 | 0.2967 |

## 18. Scientific interpretation

With the classifier confound removed, the answer is clean: **regime-level gating does not add predictive value over uniform H on either dataset.**

- R4 < R2 in validation **and** test on both datasets (−1.8 and −2.7 pp test).
- Gate differentiation is weak (Haptics) or absent (ECG) — the useful variation of H is not aligned with the VQ-regime axis; regimes are not independently "helpful" or "harmful" components of H.
- On Haptics the mechanism R4 was designed for (selective per-regime weighting) had its best shot — gates differentiated in the direction of H-contribution share — and still lost to uniform H. On ECG the parameterization collapsed to a scalar, empirically reducing R4 to R3-style global gating.
- These results *strengthen* the R3 conclusion: the H branch is either useful as a whole (Haptics, ECG rerun) or harmful as a whole (ECG first run, ECG5000_BAL historical); the useful granularity for any H modulation is the dataset level, not the regime level. No outcome in the spec's taxonomy (A–D) occurred except B/C.

## 19. Limitations

1. Single seed, two datasets — Haptics n=132 makes all gaps smaller than typical seed noise (±0.03–0.05 observed across this project); ECG's val split (923) selects noisily.
2. Outer objective is validation one-hot MSE, while selection is val Macro-F1 — the two can disagree about the best step (ECG's v≈0.29 selection).
3. The gate path is smooth but the F1 landscape over v is piecewise constant (argmax of decisions); gradient information is only a proxy signal.
4. The frozen alphas were selected by R2 for the *ungated* representation; at other v the optimal α could differ. This was a deliberate freeze (isolate gate effect) per spec.
5. H_{m,k} decomposition semantics inherit R2's min-occupancy handling; regimes near the occupancy threshold (k1 Haptics) contribute almost nothing to H and therefore carry almost no gradient — their gates are structurally hard to learn.

## 20. Final stop-rule conclusion

This was the final planned architecture experiment; per the stop rule, no further datasets, seeds, gates, parameterizations, or protocol changes follow this report. Required console summary:

```
R4 DIFFERENTIABLE-RIDGE REGIME GATE — FINAL SEED 42 RESULTS

Haptics:      M0 0.4974 | R2 0.5500 | G val 0.7894 | G+H val 0.9228 | R4 val 0.8618 | R4 test 0.5324
              ΔR4−R2 −0.0176 | ΔR4−M0 +0.0350
              v = [0.4943, 0.4978, 0.5187, 0.6349, 0.5318, 0.5794, 0.5866, 0.5019]

ECG5000_BAL:  M0 0.6553 | R2 0.6409 | G val 0.9759 | G+H val 0.9934 | R4 val 0.9830 | R4 test 0.6143
              ΔR4−R2 −0.0266 | ΔR4−M0 −0.0410   (R2 rerun 0.6508; ΔR4−R2rerun −0.0365)
              v = [0.2950, 0.2952, 0.2967, 0.2918, 0.2935, 0.2921, 0.2946, 0.2950]

Identity audit: v=1 == R2 Ridge EXACT (agreement 1.000, both datasets)
Solver: max residual 1.0e-14 / 8.5e-13; cond ≤ 3.7e4; float64; deterministic
Gate convergence: Haptics mild differentiation (std 0.048, budget-limited);
                  ECG uniform collapse (std 0.0015)

VERDICT: NOT SUPPORTED
```

Regime-level continuous gating of the heterogeneity block is learnable but provides no predictive benefit over the uniform R2 aggregation under a classifier-confound-free Ridge protocol. R2's H_m = Σ_k H_{m,k} stands as the correct aggregation granularity; the dataset level (R3's question) remains the only axis where gating behavior was observed to matter.
