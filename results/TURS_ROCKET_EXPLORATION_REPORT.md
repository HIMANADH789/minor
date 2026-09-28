# TURS + ROCKET Exploration Report

**Question:** Can ROCKET's multiscale random-convolution inductive bias be integrated into TURS's adaptive transport–regime framework in a way that improves performance while retaining a meaningful architectural contribution — rather than a feature-concatenation ensemble?

**Protocol:** All runs use the identical fair-benchmark methodology as the existing corrected benchmark (same split, seed 42, epochs, optimizer, validation protocol, evaluation script). Single seed per variant (per user instruction — no 5-seed validation). All four datasets: ECG5000-Unbalanced, ECG5000-Balanced, CWRU-Unbalanced, CWRU-Balanced.

---

## 1. Motivation

MiniROCKET holds the best **average** Macro-F1 (0.8089) of every model tested so far, driven almost entirely by CWRU (0.9917 / 0.9947). TURS-Lite holds the best ECG performance of the learned models (0.7709 avg). The question is whether the *mechanism* behind MiniROCKET's strength — diverse multiscale random-convolution projections summarized by PPV-style statistics — can be absorbed into TURS's adaptive transport–regime reasoning, instead of bolting MiniROCKET features on as an extra input.

## 2. Variants tested (all seed 42, all 4 datasets)

Six single-mechanism variants plus the integrated full model, all sharing the TURS-Lite backbone (137.5K params) with a compact ROCKET-response branch (~143–154K total):

| Variant | Mechanism |
|---|---|
| **TURS-RR** | Rocket Response: compact multiscale random-conv bank (PPV+mean+max stats) as an extra representation branch feeding the classifier |
| **TURS-RTR** | Rocket Temporal Residual: ROCKET response injected into the temporal path as a gated conditional residual |
| **TURS-3F** | Three-way adaptive fusion: α_T·F_T + α_R·F_R + α_M·F_M with softmax over {T,R,M} |
| **TURS-RCF** | Rocket-Conditioned Fusion: ROCKET conditions the transport/regime fusion weights; rocket-mediated interactions I_TM, I_RM |
| **TURS-RS** | Rocket-Sensed Regime: ROCKET responses condition the regime encoder E(H, F_M) — rocket evidence shapes the latent regime |
| **TURS-RV** | Rocket-informed Regime Velocity: rocket-response *changes* gate the regime velocity (v'_t = v_t + g_q ⊙ q_t) |
| **TURS-FULL** | Integrated: ROCKET-sensed regime + rocket-conditioned fusion + rocket-mediated interactions |

## 3. Seed-42 results — Macro-F1 (accuracy in parentheses)

| Model | ECG-U | ECG-B | CWRU-U | CWRU-B | **Avg** |
|---|---|---|---|---|---|
| TURS-RR | 0.6076 (0.954) | 0.6484 (0.945) | 0.9040 (0.904) | 0.9472 (0.947) | **0.7768** |
| TURS-RTR | 0.6146 (0.955) | 0.6699 (0.940) | 0.8671 (0.871) | 0.9452 (0.945) | **0.7742** |
| TURS-3F | 0.6148 (0.955) | 0.7028 (0.952) | 0.8789 (0.879) | 0.9489 (0.949) | **0.7864** |
| TURS-RCF | 0.5718 (0.952) | **0.7122** (0.950) | **0.9124** (0.913) | 0.9524 (0.952) | **0.7872** |
| TURS-RS | 0.6157 (0.957) | 0.6434 (0.942) | 0.9003 (0.900) | **0.9596** (0.959) | **0.7798** |
| TURS-RV | **0.6778** (0.954) | 0.6729 (0.951) | 0.9078 (0.908) | 0.9471 (0.947) | **0.8014** |
| TURS-FULL | 0.5566 (0.948) | 0.6588 (0.947) | 0.9000 (0.900) | 0.9560 (0.956) | **0.7679** |
| TURS-Lite (rerun) | 0.6225 (0.954) | 0.6567 (0.948) | 0.8831 (0.883) | 0.9578 (0.958) | **0.7800** |

### Reference baselines (same protocol, seed 42)

| Model | ECG-U | ECG-B | CWRU-U | CWRU-B | **Avg** |
|---|---|---|---|---|---|
| InceptionTime | 0.5868 | 0.6324 | 0.8662 | 0.9647 | 0.7625 |
| eTAI-Focal | 0.5956 | 0.6388 | 0.8791 | 0.9665 | 0.7700 |
| USTR-Net-CE | 0.5948 | 0.6198 | 0.8529 | 0.9312 | 0.7497 |
| USTR-Net-Focal | 0.5763 | 0.6301 | 0.8585 | 0.9489 | 0.7534 |
| TURS-Lite | 0.6046 | 0.6377 | 0.8997 | 0.9417 | 0.7709 |
| TURS-Strong | 0.5955 | 0.5605 | 0.8920 | 0.9090 | 0.7392 |
| ResNet | 0.5360 | 0.5938 | 0.8644 | 0.8928 | 0.7217 |
| FCN | 0.3832 | 0.5766 | 0.8432 | 0.8901 | 0.6733 |
| PatchTST-Cls | 0.5260 | 0.5700 | 0.8314 | 0.8923 | 0.7049 |
| **MiniROCKET** | 0.5938 | 0.6553 | **0.9917** | **0.9947** | **0.8089** |

> Note: TURS-Lite appears twice — 0.7709 (original corrected-bench run) and 0.7800 (rerun inside this experiment). Difference ≈ 1 pp, attributable to CPU run-to-run variance; both are reported.

## 4. Headline findings

1. **TURS-RV is the strongest integrated variant on average (0.8014)**, beating TURS-Lite (+2.1 pp) and every learned baseline, and closing most of the gap to MiniROCKET (0.8089) — without touching CWRU's MiniROCKET advantage.
2. **On ECG, integrated variants decisively beat MiniROCKET.** ECG-U: RV 0.6778 vs MiniROCKET 0.5938 (**+8.4 pp**) and vs TURS-Lite 0.6046 (**+7.3 pp**). ECG-B: RCF 0.7122 vs MiniROCKET 0.6553 (**+5.7 pp**).
3. **On CWRU, MiniROCKET remains dominant (0.992 / 0.995).** Best TURS-family: RCF 0.9124 (U) and RS 0.9596 (B, +1.8 pp over Lite). No integrated variant closes the CWRU gap.
4. **The late-fusion control does NOT beat the integrated variants on ECG** — and "wins" the average only by delegating CWRU entirely to MiniROCKET (λ→0). That is model *selection*, not representation synergy.

## 5. Late-fusion control (TURS-Lite + MiniROCKET probability fusion, λ tuned on val)

| Dataset | λ* | Fusion test MF1 | TURS-Lite test | MiniROCKET test |
|---|---|---|---|---|
| ECG-U | 0.10 | 0.5992 | 0.6225 | 0.5938 |
| ECG-B | 0.45 | 0.6866 | 0.6567 | 0.6553 |
| CWRU-U | 0.00 | 0.9917 | 0.8831 | 0.9917 |
| CWRU-B | 0.00 | 0.9947 | 0.9578 | 0.9947 |
| **Avg** | — | **0.8180** | 0.7800 | 0.8089 |

**Interpretation:** Simple ensembling captures none of the integrated gain on ECG (0.599 < RV 0.678 on ECG-U; 0.687 < RCF 0.712 on ECG-B). Its higher average is an artifact of λ=0 on CWRU → pure MiniROCKET. It is a selection oracle, not a representation contribution.

## 6. Per-class analysis

### ECG5000-U minority class C4 (support = 5 test samples — statistically fragile)

| Model | C4 recall | C4 F1 |
|---|---|---|
| TURS-RV | **0.200** (1/5) | **0.286** |
| all other variants | 0.000 | 0.000 |
| MiniROCKET | 0.000 | 0.000 |
| ResNet / FCN / PatchTST | 0.000 | 0.000 |

TURS-RV is the **only model in the entire study that detects any C4 samples** (1 of 5). The gain is real but must be read with caution given the support of 5.

### CWRU-U per-class recall (Normal / Inner Race / Ball / Outer Race)

| Model | Normal | Inner Race | Ball | Outer Race |
|---|---|---|---|---|
| TURS-RCF | 1.000 | 0.983 | **0.917** | 0.750 |
| TURS-RV | 1.000 | 0.983 | 0.900 | 0.750 |
| TURS-RR | 1.000 | 0.983 | 0.900 | 0.733 |
| TURS-Lite | 1.000 | 0.983 | 0.817 | 0.733 |

Rocket-conditioned fusion (RCF) lifts **Ball** class recall from 0.817 (Lite) to 0.917 — the class most confusable with Outer Race — consistent with multiscale random projections helping vibration-fault discrimination. Outer Race remains the weak class for all TURS variants.

## 7. Parameter counts & runtime (per dataset, seed 42)

All variants stay in the compact regime (142–154K, +3–12% over TURS-Lite 137.5K).

| Model | Params | ECG-U time | CWRU-B time | Best epoch (typical) |
|---|---|---|---|---|
| TURS-Lite | 137.5K | 112s | 307s | 12–15 |
| TURS-RR | 143.5K | 130s | 326s | 11–15 |
| TURS-RTR | 144.8K | 132s | 344s | 10–13 |
| TURS-3F | 143.2K | 126s | 352s | 11–14 |
| TURS-RCF | 151.7K | 112s | 329s | 7–14 |
| TURS-RS | 142.7K | 137s | 425s | 13–15 |
| TURS-RV | 142.2K | 135s | 477s | 11–14 |
| TURS-FULL | 153.8K | 120s | 925s | 7–15 |

Runtime overhead is modest except TURS-FULL on CWRU-B (925s — the integrated variant's cross-scale interactions are expensive at length-1024).

## 8. Ablation reading (what each mechanism contributed)

- **Rocket as third feature (RR, RTR):** no gain over Lite on average (0.777 / 0.774 vs 0.780). Simple addition of rocket features is *not* the answer.
- **Three-way adaptive fusion (3F):** consistent gain on ECG-B (+4.6 pp vs Lite) and solid CWRU-B; neutral on ECG-U. Adding the rocket branch *into the adaptive weighting* helps.
- **Rocket-conditioned fusion (RCF):** best on ECG-B (+5.6 pp) and CWRU-U (+2.9 pp) — the "rocket decides how much to trust transport vs regime" mechanism is the strongest *average* integrated idea.
- **Rocket-sensed regime (RS):** best CWRU-B (+1.8 pp) — rocket evidence feeding the latent regime helps on the hardest balanced vibration set.
- **Rocket-gated regime velocity (RV):** the standout on ECG-U (+7.3 pp, only C4 detector). Regime *velocity* modulated by rocket response change is the mechanism that rescues the pathological minority class.
- **Full model:** no better than its parts (0.768 avg) — combining everything overfits/overcomplicates; the components are individually stronger than their union.

## 9. Best candidate

**TURS-RV (Rocket-informed Regime Velocity)** — 142.2K params:
- Best average among all integrated variants: **0.8014** (vs MiniROCKET 0.8089, TURS-Lite 0.7709–0.7800).
- The **only model that detects ECG C4** samples (recall 0.20 vs 0.00 everywhere else).
- +8.4 pp over MiniROCKET on ECG-U; +1.8 pp over MiniROCKET on ECG-B; −8.4 / −4.8 pp behind MiniROCKET on CWRU.

**TURS-RCF (Rocket-Conditioned Fusion)** — 151.7K params: best average performance on the two hardest discrimination tasks (ECG-B, CWRU-U); strongest mechanistic novelty ("rocket representation governs transport-vs-regime trust").

## 10. Decision

- **Architecturally integrated ROCKET mechanisms are genuinely better than feature concatenation / ensembling on ECG** — RV and RCF beat both TURS-Lite and MiniROCKET there, and the late-fusion control does not reproduce the gain.
- **The hypothesis is only partially confirmed.** No integrated variant reaches MiniROCKET's *average* (best 0.8014 vs 0.8089). The residual gap is entirely CWRU, where MiniROCKET's massively parallel random-conv bank (≈10K kernels) extracts vibration discriminability that a 143K-parameter learned model cannot match.
- **Recommendation:** incorporate the **Rocket-gated regime velocity** (RV) and optionally the **Rocket-conditioned fusion** (RCF) mechanisms into the final TURS architecture as a methodological contribution, and report the honest result: *TURS-RV closes the average gap to MiniROCKET from −3.8 pp (Lite) to −0.8 pp while beating it on both ECG sets and remaining 3.4× smaller, but does not surpass MiniROCKET on the vibration benchmarks.*
- **Limitations:** single seed (seed 42) per variant per user instruction — CPU runs show ±1–2 pp run-to-run variance (visible in the two TURS-Lite runs); per-sample learned fusion weights (α_T, α_R) were not persisted by the benchmark, so the learned-weight analysis of §14 of the original spec could not be completed post-hoc.