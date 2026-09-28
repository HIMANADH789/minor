# TURS-AMR Report — Adaptive Multiscale Response Field

## 1. Motivation

TURS-RV (0.8014 avg MF1) narrowed the gap to MiniROCKET (0.8089) to −0.8 pp by using
Rocket-style response *changes* to gate regime velocity. TURS-AMR asks whether a **richer
response representation** — response dynamics (ΔR, Δ²R), cross-scale response structure,
and regime-adaptive scale selection — can close the remaining gap while staying compact
(140–220K params).

## 2. Architecture (shared skeleton)

```
X → [Transport T=[X,Q,D]] + [Inception-lite backbone H]
H → multiscale response bank R (fixed random convs, K=64, dilations {1,2,4,8})
R → descriptors (μ, σ, PPV, max) → F_M (16–32 dims)
Response dynamics: ΔR = R_t − R_{t−1}, Δ²R = ΔR_t − ΔR_{t−1} → q = E_R(R, ΔR, Δ²R)
Regime: (μ_z, σ_z) = E_Z(H, F_M, q) → z = μ_z, u = σ_z
Velocity: v = z_t − z_{t−1};  ṽ = v + g_v ⊙ P_v(q)   [TURS-RV mechanism, inherited]
Fusion: F = αF_T + (1−α)F_R + λ_TR·I_TR + λ_RM·I_RM
Classifier: [GAP(H'), z, ṽ, a, u, F_T, q] → MLP → logits
```

Variant mechanisms:
- **D**  — richer response dynamics: q built from (R, ΔR, Δ²R) with a dedicated encoder
- **CS** — cross-scale response interaction: pairwise learned products C_ij = P_i(Rⁱ)⊙P_j(Rʲ) over 4 scale groups, aggregated into the regime encoder
- **AS** — adaptive scale selection: regime-conditioned softmax gates g_s over scale groups, R_adaptive = Σ g_s R^(s)
- **RK** — conditional kernel residual: low-rank regime-conditioned modulation ΔW of the response bank, W_eff = W⁰ + η·ΔW(z,u)
- **FULL** — all of the above

## 3. Seed-42 results (all 4 datasets, Macro-F1)

| Variant       | ECG-U  | ECG-B  | CWRU-U | CWRU-B | **Avg**  | Params |
|---------------|--------|--------|--------|--------|----------|--------|
| **TURS-AMR-RK**   | 0.5997 | **0.7327** | 0.9044 | **0.9613** | **0.7995** | 150,333 |
| TURS-AMR-CS   | **0.6747** | 0.6278 | 0.9121 | 0.9558 | 0.7926 | 153,677 |
| TURS-AMR-AS   | 0.5821 | 0.6573 | 0.9214 | 0.9539 | 0.7787 | 155,163 |
| TURS-AMR-RV   | 0.5834 | 0.6728 | 0.8923 | 0.9437 | 0.7731 | 149,653 |
| TURS-AMR-FULL | 0.5938 | 0.6159 | **0.9252** | 0.9560 | 0.7727 | 197,340 |
| TURS-AMR-D    | 0.5722 | 0.6435 | 0.9207 | 0.9489 | 0.7713 | 186,437 |
| TURS-Lite (rerun) | 0.5945 | 0.6567 | 0.8760 | 0.9452 | 0.7681 | 137,547 |
| LateFusion TURS+MR | 0.6026 | 0.6797 | 0.9917 | 0.9894 | 0.8158* | — |

\* Late fusion "wins" only because λ→0.10 delegates CWRU to MiniROCKET — selection, not synergy.

**References (unchanged):**

| Model | ECG-U | ECG-B | CWRU-U | CWRU-B | Avg |
|---|---|---|---|---|---|
| MiniROCKET | 0.5938 | 0.6553 | 0.9917 | 0.9947 | **0.8089** |
| TURS-RV (original) | 0.6778 | 0.6729 | 0.9078 | 0.9471 | **0.8014** |
| eTAI-Focal | 0.596 | 0.639 | 0.879 | 0.967 | 0.7700 |
| InceptionTime | 0.587 | 0.632 | 0.866 | 0.965 | 0.7625 |

## 4. Key findings

1. **TURS-AMR-RK is the best AMR variant (0.7995)** — statistically tied with TURS-RV
   (0.8014, −0.2 pp), with a different profile: much stronger on ECG-B (+6.0 pp vs RV)
   and CWRU-B (+1.4 pp), weaker on ECG-U (−7.8 pp).
2. **No AMR variant beats MiniROCKET on average** (best −0.9 pp). The CWRU gap
   (0.90–0.96 vs 0.99+) remains unclosed for any compact learned model in this study.
3. **The FULL model again underperforms its parts** (0.7727 < RK 0.7995). Mechanism
   stacking does not help; single well-chosen mechanisms do.
4. **Cross-scale interaction (CS) is the second useful mechanism** (0.7926) and is the
   only AMR variant that detects ECG C4 (recall 0.20, 1/5, F1 0.333).
5. **Richer dynamics alone (D) does not help** (0.7713) — the Δ²R encoder adds 36K params
   without gain. Adaptive scale selection (AS) is neutral (0.7787).
6. **CWRU per-class:** AMR variants improve the hard Ball/Outer-Race classes over
   TURS-Lite (CS: 0.844/0.821 vs Lite 0.773/0.757 on CWRU-U; RK: 0.922/0.922 on CWRU-B).

## 5. ECG-U C4 detail (support = 5)

| Model | C4 recall | C4 F1 | C4 confusion row |
|---|---|---|---|
| TURS-AMR-CS | 0.200 (1/5) | 0.333 | [1,1,2,0,1] |
| TURS-AMR-RK | 0.000 | 0.000 | [0,2,3,0,0] |
| TURS-Lite | 0.000 | 0.000 | [0,2,3,0,0] |

C4 detection is not consistent across variants — do not over-claim from 1/5 samples.

## 6. Ablation ladder verdict

```
TURS-RV (0.8014) ≥ AMR-RK (0.7995) > AMR-CS (0.7926) > AMR-AS (0.7787)
                      > AMR-RV* (0.7731) ≈ AMR-FULL (0.7727) > AMR-D (0.7713) > TURS-Lite (0.7681)
```

\* AMR-RV is a re-implementation inside the AMR framework (richer response bank); it does
not reproduce the original TURS-RV exactly (CPU run-to-run variance ±1–2 pp plus wiring
differences).

## 7. Decision

Per the pre-registered criteria (performance + mechanism + compactness + stability):

- **Keep TURS-RV as the final TURS-family architecture** (0.8014, 142K params, simplest
  mechanism: Rocket-gated regime velocity).
- **AMR-RK is the only AMR variant worth preserving** as a documented alternative
  (0.7995, 150K params) — it trades ECG-U strength for ECG-B/CWRU-B strength and its
  conditional kernel residual is a genuine novel mechanism, but it does not dominate.
- **The AMR hypothesis (richer response dynamics → better regime dynamics) is NOT
  confirmed.** Response *changes* (RV) capture most of the available signal; second-order
  dynamics, cross-scale products, and scale selection add complexity without average gain.

## 8. Runtime

| Item | Time |
|---|---|
| ECG datasets (per variant) | ~90–115 s |
| CWRU datasets (per variant) | ~330–440 s |
| MiniROCKET (per dataset) | ~15 s |
| Full 4-dataset × 8-model run | ~2.6 h (incl. interruption + resume) |

## 9. Limitations

- Single seed (42) per variant, as specified — differences within ±1 pp are not decisive.
- CPU-only training; run-to-run variance ±1–2 pp observed across identical configs.
- Per-sample fusion weights / response-field interpretability dumps were not persisted
  in this run; the interpretability analysis (§31 of the spec) remains open for RK/CS.

## 10. Recommendation

Do **not** replace TURS-RV with AMR. If a single paper model is needed, TURS-RV remains
the recommendation; AMR-RK can be reported as an ablation showing that the conditional
kernel residual helps the harder balanced datasets but does not generalize.
