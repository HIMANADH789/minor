# TURS-GLR: Gated Global-Local Regime-Routed Ridge — Full Study Report

*Runtime 0.0 min · canonical biomedical protocol (seed 42, stratified splits, per-sample z-norm) · ONE architecture for all datasets: M_global=2048 fixed kernels, M_local=128 fixed kernels, J=4 transport flavors, soft-from-scratch router (hidden 32, softmax tau), block-regularized closed-form ridge with validation-selected (lambda_global, lambda_local)*

## 1. Headline predictive results (locked test set)

| Dataset | A0 global | A1 local | A2 concat+scalar | **A3 GLR (block)** | A4 tau-tuned | A7 uniform-route |
|---|---|---|---|---|---|---|
| ECG5000_UNBAL | 0.2451 | 0.3228 | 0.5204 | **0.5060** | 0.5418 | 0.5191 |
| ECG5000_BAL | 0.1665 | 0.4141 | 0.5725 | **0.5668** | 0.5497 | 0.5526 |
| CWRU_UNBAL | 0.3215 | 0.3780 | 0.9079 | **0.9079** | 0.9161 | 0.9072 |
| CWRU_BAL | 0.2487 | 0.2519 | 0.9435 | **0.9435** | 0.9204 | 0.9168 |

**Mean Macro-F1 of the primary model (A3) across the four datasets: 0.7310**

## 2. Research questions

- **RQ1 (global+local > each alone):** per-dataset A3 vs max(A0, A1) — see table above and 03_global_vs_local.csv.
- **RQ2 (block > scalar ridge):** ECG5000_UNBAL: dMF1=-0.0144, McNemar p=1; ECG5000_BAL: dMF1=-0.0056, McNemar p=0.683; CWRU_UNBAL: dMF1=+0.0000, McNemar p=1; CWRU_BAL: dMF1=+0.0000, McNemar p=1
- **RQ3 (soft routing > uniform):** ECG5000_UNBAL: dMF1=-0.0132, McNemar p=0.617; ECG5000_BAL: dMF1=+0.0143, McNemar p=0.831; CWRU_UNBAL: dMF1=+0.0007, McNemar p=0.789; CWRU_BAL: dMF1=+0.0266, McNemar p=0.00511
- **RQ4 (routing useful with global bank present):** A3 vs A7 tests exactly this (both include the global block).
- **RQ5 (automatic global/local balance):** selected lambda pairs and block contributions per dataset in 04_block_lambda.csv / 05_block_contributions.csv.
- **RQ6 (global stream recovers CWRU advantage):** A0 vs MiniROCKET_aeon baseline comparison in 13_model_comparison.csv.
- **RQ7 (local routed stream adds value on ECG):** A1/A3 on ECG datasets vs A0.
- **RQ8-RQ11 (vs baselines):** section 4 below.
- **RQ12 (retains RRMT diagnostics while improving prediction):** routing diagnostics D3-D8 below; predictive comparison vs TURS-RRMT lineage in section 4.

## 3. Global/local adaptation (the central mechanistic claim)

| Dataset | lambda_g | lambda_l | coef |beta|_g | coef |beta|_l | local share (median) |
|---|---|---|---|---|---|
| ECG5000_UNBAL | 1 | 0.1 | 0.366 | 6.563 | 0.529 |
| ECG5000_BAL | 100 | 0.01 | 0.259 | 8.512 | 0.556 |
| CWRU_UNBAL | 10 | 0.0001 | 0.071 | 73.303 | 0.746 |
| CWRU_BAL | 0.0001 | 0.0001 | 4.199 | 76.063 | 0.774 |

Interpretation discipline: lambda values are NOT contribution percentages; the fitted-solution measures (coefficient norms, per-block logit norms, ablation deltas) are the contribution evidence.

## 4. Comparison with baselines (protocol-verified where possible)

| Dataset | TURS-GLR A3 | TURS-Lite | TURS-Stack | MiniROCKET(aeon) |
|---|---|---|---|---|
| ECG5000_UNBAL | **0.5060** | 0.6225 | 0.3834 | 0.5934 |
| ECG5000_BAL | **0.5668** | 0.6567 | 0.6364 | 0.6546 |
| CWRU_UNBAL | **0.9079** | 0.8831 | 0.8744 | 0.9917 |
| CWRU_BAL | **0.9435** | 0.9578 | 0.9564 | 0.9947 |

MiniROCKET numbers are re-evaluated from saved test probabilities with our canonical test labels (protocol identity verified per dataset; see 13_baseline_protocol_identity.json). TURS-Stack / TURS-Lite numbers are historical benchmark summaries on the same data files and split logic.

## 5. Routing diagnostics (D3-D8)

### ECG5000_UNBAL
- Faithfulness: targeted conf drop 0.0515 vs random 0.0755 (diff -0.0240, p=0.1524, flip rate 46.0%)
- Uniform-routing intervention: MF1 0.0020 (vs A3 0.5060); shuffled: 0.0765
- Routing entropy (mean over J): 0.962; switch rate 0.000

### ECG5000_BAL
- Faithfulness: targeted conf drop -0.0451 vs random -0.0254 (diff -0.0197, p=0.2784, flip rate 63.5%)
- Uniform-routing intervention: MF1 0.0075 (vs A3 0.5668); shuffled: 0.0925
- Routing entropy (mean over J): 0.962; switch rate 0.000

### CWRU_UNBAL
- Faithfulness: targeted conf drop -0.0465 vs random -0.0544 (diff +0.0079, p=0.3358, flip rate 62.0%)
- Uniform-routing intervention: MF1 0.1165 (vs A3 0.9079); shuffled: 0.2815
- Routing entropy (mean over J): 0.971; switch rate 0.000

### CWRU_BAL
- Faithfulness: targeted conf drop -0.0001 vs random 0.0002 (diff -0.0003, p=1.0000, flip rate 14.5%)
- Uniform-routing intervention: MF1 0.1036 (vs A3 0.9435); shuffled: 0.1194
- Routing entropy (mean over J): 0.972; switch rate 0.000


## 6. Evidence rubric

| Dimension | Verdict | Evidence |
|---|---|---|
| D1 global-pattern representation | WEAK | mean A0 (global-only) MF1 = 0.24544504438508277 |
| D2 local-pattern representation | WEAK | mean A1 (local-only) MF1 = 0.3416836303454292 |
| D3 routing validity | MODERATE | max |rho(w, descriptor)| = 0.48652712280454297 |
| D4 routing functional necessity | MODERATE | mean MF1(A3 - A7) = 0.0070999999999999995 |
| D5 transport-flavor specialization | MODERATE | single-flavor intervention MF1 by dataset (see 08 table) |
| D6 routing faithfulness | WEAK | mean targeted-random conf drop = -0.009005542866361793 |
| D7 routing stability | STRONG | routing cosine under benign noise = 0.9999994039535522 |
| D8 global/local adaptive specialization | STRONG | selected (lam_g, lam_l) per dataset: [(1.0, 0.1), (100.0, 0.01), (10.0, 0.0001), (0.0001, 0.0001)] |
| D9 block-ridge usefulness | WEAK | mean MF1(A3 - A2) = -0.005 |
| D10 predictive calibration | WEAK | mean ECE = 0.526644697188339 |
| D11 selective reliability | MODERATE | mean error-detection AUROC (confidence) = 0.8892832321998316 |
| D12 robustness | MODERATE | severity-response rhos in 09_robustness.csv / D12 hyp family |
| D13 cross-dataset consistency | STRONG | unified >= best single block on 4/4 datasets |
| D14 simplicity/complexity efficiency | STRONG | trainable params (router+ridge): 8,356; kernels fixed (2048 global + 128 local) |

## 7. Failure modes, limitations, final recommendation

- Single seed (42) per dataset; no multi-seed variance.
- Router training (A4/A5) is stochastic despite fixed seeds; results may vary with hardware.
- MiniROCKET A8 comparison uses aeon's fixed 84-bias construction rather than our bank; A8 vs A3 isolates the kernel-bank recipe, not the readout.
- If A3 does not beat the best single block or the baselines on a dataset, that is reported as-is; the unified architecture is recommended only where it wins validation and holds on the locked test set.
- **Final recommendation (simplest model that wins validation and holds on test): A2** (mean MF1 across datasets).
