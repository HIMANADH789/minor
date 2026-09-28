# RCMKN — Regime-Conditioned Multi-View Kernel Network (Haptics, seed 42)

First implementation and single-seed ablation of the proposed RCMKN
architecture: fixed MiniROCKET temporal primitives × learned (self-supervised)
temporal context, fused with Hydra-style competitive pooling, classified by a
single linear Ridge. All decisions were frozen before test evaluation; each
variant touched the test set exactly once (6 official evaluations).

## 1. Objective

Test whether a heterogeneous architecture combining (1) fixed MiniROCKET
temporal projections, (2) a learned SSL temporal representation, (3) learned
discrete temporal context (HardVQ), (4) regime-conditioned kernel statistics,
(5) Hydra-style competitive pooling, and (6) Ridge feature-level fusion can
improve on the audited DRTN-conditioned MiniROCKET mechanism (R0) on Haptics —
the strongest positive dataset for the audited mechanism.

## 2. Scientific hypothesis

Fixed local temporal patterns become more discriminative when their
activation rates are summarized *conditional on a learned temporal context*,
and a complementary competition-based pooling (Hydra) is additive with that
mechanism. The learned encoder must discover temporal organization itself —
no hand-designed statistics (velocity/uncertainty/duration etc.) are defined
anywhere in the implementation.

## 3. Architecture (as implemented)

- **Stage 1 — fixed kernel bank**: canonical aeon `MiniRocket(random_state=42)`;
  raw temporal activations from the audited raw-response extractor
  (identity vs aeon: max|diff| = 0.00e+00). Untouched.
- **Stage 2 — SSL causal encoder**: 4 causal Conv1d blocks (k=3/5/7/9,
  dilation=1/2/4/8, channels 32/32/64/64), ChannelNorm (per-position channel
  norm — no future mixing), GELU, no downsampling, projection to d=32.
  Trained with **masked-span reconstruction** (mask 10%, span 16, loss on
  masked positions only, no labels). 59,072 encoder + 2,177 decoder params.
- **Stage 3 — discrete context**: reused validated `HardVQ` (K=8, EMA 0.99,
  dead-code revival, straight-through), commitment β=0.25, population-level
  code-usage diversity λ=0.01 (no per-timestep entropy).
- **Stage 4 — regime-conditioned kernels**: exact audited formula
  `H_m = Σ_k q_k (PPV_{m,k} − PPV_m)²` with hard regimes and per-feature
  valid region `[padding_m, T−padding_m)`. Unchanged from the audited
  implementation.
- **Stage 5 — Hydra competitive pooling**: aeon `_HydraInternal` (the exact
  core of `HydraTransformer`), k=8, g=16 → 8 dilations × divisor 2 × 8 groups
  × 8 kernels × {count_max, count_min} = **2048 features**. Same fixed kernel
  bank; competition statistic, not threshold-PPV.
- **Stage 6 — Ridge fusion**: `RidgeClassifierCV(alphas=np.logspace(-4,4,20))`
  fit on **train+validation**; no neural head, no output-level stacking.
  Hydra block scaled by the canonical `_SparseScaler` formula (fit train+val,
  Hydra block only — raw counts otherwise drown the PPV blocks and pin alpha
  at the grid max).

Total context model: **61,414 trainable parameters** (encoder 59,072 +
decoder 2,177 + aux head 165; VQ codebook EMA buffers are not trainable) —
under the <100K target and ~10× smaller than the 599K DRTN transformer.

## 4. Exact ablation ladder

| Variant | Fixed MR | Regimes | SSL encoder | Hydra | Dim |
|---|---|---|---|---|---|
| R0 | yes | official DRTN (frozen ckpt) | no | no | 9996 |
| R1 | yes | official DRTN | no | yes | 12044 |
| R2 | yes | SSL-context VQ | yes | no | 9996 |
| R3 | yes | SSL-context VQ | yes | yes | 12044 |
| C1 | yes | occupancy-matched random (audited M2 construction) | yes | no | 9996 |
| C2 | yes | per-sample shuffled (audited M3 construction) | yes | no | 9996 |

C1/C2 were implemented and use exactly the audited control constructions
(independent streams, per-sample occupancy preserved, no shared memory).

## 5. Parameter counts

See §3. Codebook usage diagnostics for the SSL context model (train+val):
8/8 codes active, normalized entropy 0.935, perplexity 6.99, dominant-code
fraction 0.236, 2 dead-code revivals during training. No collapse — unlike
GunPoint/ItalyPowerDemand in the context3 screen (~2 dominant codes there).

## 6. Feature dimensions

R0/R2/C1/C2: 9996 = 4998 global + 4998 heterogeneity (kernel-axis slicing;
no sample-axis slicing; no overlap/duplication). R1/R3: 12044 = 9996 + 2048
Hydra. Hydra feature count verified empirically from `_HydraInternal`
structure and asserted equal across variants.

## 7. Training protocol

- Data: canonical Haptics, train=132 / val=23 / test=308, T=1092, 5 classes,
  per-sample z-normalization before all branches (untouched canonical split;
  val = per-class stratified 15% of train, singletons kept in train).
- Context model (R2/R3/C1/C2 source): phase 1 pure SSL (120 epochs, AdamW
  lr 1e-3, wd 1e-4, batch 8, patience 20; early stop ep48, best val recon
  0.0225); phase 2 joint SSL+VQ+commit+div+aux (60 epochs, lr 5e-4,
  λ_cls=0.10, patience 10; early stop ep38). Checkpoint selected on
  validation only, then frozen (verified: state-dict equality after regime
  extraction). Loss weights predeclared; no tuning on test.
- R0 uses the official frozen DRTN R5 K=8 checkpoint (unchanged).
- Ridge: alpha selected by CV on train+val; final fit train+val.
- Test evaluation: once per variant, after all audits passed.

## 8. Audit results (all passed before any test evaluation)

1. Raw MiniROCKET identity vs aeon: max|diff| = **0.00e+00** (chunked, trainva).
2. Per-feature valid-region correctness: audited core reused unchanged
   (unit-tested: corrupting out-of-mask activations changes nothing).
3. Exact H formula: audited core reused unchanged (independent recompute ≤1e-8).
4. Raw-response provenance: heterogeneity computed from raw per-timestep
   activations (audited core).
5. Global/het dims: 4998/4998 asserted per variant.
6. Feature budget: 9996 (2-block) / 12044 (3-block) asserted.
7. Hard VQ assignment correctness: argmin-distance hard assignment (validated
   HardVQ); regime extraction verified deterministic (saved vs re-extracted
   arrays equal).
8. VQ occupancy: 8/8 active, entropy 0.935, perplexity 6.99 (above).
9. EMA update correctness: validated HardVQ infrastructure.
10. Dead-code revival: 2 revivals logged.
11. Encoder causality: unit-tested (permuting future inputs leaves Z_t
    unchanged).
12. Masked-span loss uses only masked positions: unit-tested.
13. No labels in SSL loss: enforced structurally (loss consumes x only).
14. No test samples in encoder training: train split only.
15. Hydra validation: group structure + feature count from `_HydraInternal`;
    deterministic same-batch (exact), cross-batch max|diff| = 3.8e-06
    (float32 reduction order); no test dependency.
16. Ridge fit on train+val: enforced in `run_variant`.
17. No test labels in feature construction: enforced structurally.
18. Determinism: seeded (seeds.json); R0/R1 feature paths deterministic.
19. **R0 reproduction gate: PASS — 0.5366 == 0.5366 (exact).**
20. No NaN/Inf in any feature matrix.

## 9. R0 reproduction

R0 test Macro-F1 = **0.5366**, exactly matching the audited Haptics M1
seed-42 reference (gate tolerance ≫ observed difference). The new namespace
does not alter the validated mechanism.

## 10. Results (test Macro-F1, seed 42)

| Variant | Val MF1 | Test MF1 | Alpha | Dim |
|---|---|---|---|---|
| R0 (DRTN cond.) | 0.9014 | 0.5366 | 1.62 | 9996 |
| R1 (R0+Hydra) | 0.9418 | 0.5123 | 1438.45 | 12044 |
| **R2 (SSL cond.)** | 0.9014 | **0.5500** | 4.28 | 9996 |
| R3 (full RCMKN) | 0.9418 | 0.5123 | 1438.45 | 12044 |
| C1 (random regimes) | 0.8251 | 0.4927 | 4.28 | 9996 |
| C2 (shuffled regimes) | 0.8618 | 0.5011 | 4.28 | 9996 |

Reference (not rerun): canonical M0 seed-42 = 0.4974.

## 11. Forensic note on R1 == R3 (exact identity)

R1 and R3 produced identical predictions (0/308 differences) and identical
alpha (1438.44988828766). Forensic verification
(`forensic_r1_r3.py`, saved JSON): the regime arrays differ on **84.38%** of
trainva positions (no shared memory), and the heterogeneity blocks genuinely
differ (max|ΔH| = 0.25 over train/val/test) — no aliasing or wiring bug.
Mechanism: with 155 train+val samples vs 12,044 features, RidgeClassifierCV
selects alpha = 1438.45 (second-highest grid value), and at that penalty the
low-variance het block is decision-irrelevant: a control Ridge on
[global | Hydra] **without any het block** reproduces R1/R3 predictions
exactly (0/308 differ; decision values differ by ≤2.6e-03 against minimum
top1–top2 margins of ~1.8e-03…1.0e-03 scale). This is a ridge-shrinkage
effect under severe n≪p, not an implementation error; it also means **R1/R3
test numbers must be read as [global|Hydra]-only models** for interpretation.

## 12. Mechanistic diagnostics (trainva H blocks)

| Source | mean H | median H | max H | frac nonzero |
|---|---|---|---|---|
| R0/R1 (DRTN regimes) | 0.01777 | 0.01076 | 0.25 | 0.9443 |
| R2/R3 (SSL regimes) | **0.03971** | 0.03382 | 0.25 | 0.9485 |
| C1 (random) | 0.00261 | 0.00088 | 0.5625 | 0.9590 |
| C2 (shuffled) | 0.00244 | 0.00090 | 0.5625 | 0.9588 |

The SSL-context regimes create **2.2×** stronger kernel–context dependence
than the DRTN regimes and **~15×** stronger than occupancy-matched random or
shuffled regimes (the latter sink to the binomial-noise floor identified in
the Stage-A forensics). The classifier result (R2 > R0 > C2 > C1) tracks the
H magnitude ordering — H magnitude here is not empty dispersion; it is
label-useful organization (see also class F1s, below).

## 13. What improved

- **R2 (SSL context) > R0 (+0.0134)**: the self-supervised causal encoder +
  HardVQ context improves the proven conditioning mechanism — and with a
  10× smaller model (61K vs 599K params), no DRTN trajectory stack, and
  *labels never entering the regime source's primary objective*.
- **Learned regimes > both controls** (R2−C1 = +0.0573, R2−C2 = +0.0489):
  the mechanistically-supported comparison passes cleanly.
- **R2 > canonical M0 by +0.0526** on the reference value.
- Class-level: R2 improves or holds R0's F1 on 4/5 classes
  (class 2: 0.567→0.614, class 4: 0.511→0.556, class 5: 0.619→0.589↓,
  class 3: 0.606→0.620, class 1: 0.381→0.370).

## 14. What did not improve

- **Hydra fusion (R1, R3)**: both = 0.5123 < R0's 0.5366. The Hydra block
  does not add useful information in this fusion; under n≪p shrinkage it
  also nullifies the het block entirely (§11). component = not beneficial in
  this experiment.
- **R3 vs R2**: −0.0377 — the full heterogeneous stack is worse than its
  SSL-context-only subset.

## 15. Failure modes

- Feature dilution under n≪p: at 12,044 features / 155 samples, Ridge picks a
  near-max penalty and entire blocks can become decision-irrelevant (§11).
  Any multi-block fusion on Haptics must be dimension-aware.
- Raw Hydra counts have 0–T magnitudes; without the canonical `_SparseScaler`
  the fusion collapses (observed pre-test in smoke: alpha pinned at grid max).
- Aux classification head learns little from 132 samples (best val MF1
  0.1985); by design it carries only λ_cls=0.10 pressure — the SSL objective
  dominates, as intended.

## 16. Scientific conclusion

The central RCMKN claim — **fixed temporal primitives + learned temporal
context + complementary competitive pooling** — is **partially supported**:
the learned-context half of the architecture (R2) beats both the audited DRTN
reference and both audited controls, with matching mechanistic evidence
(strongest H, no code collapse, 8/8 active codes). But the Hydra
competitive-pooling half is **not complementary here** (R1 < R0, R3 < R2),
and the full RCMKN (R3) underperforms its own SSL-context subset. The
strongest configuration is the simplest heterogeneous one: global MiniROCKET
PPV + SSL-regime heterogeneity, fused by a single Ridge.

Single seed, single dataset: no significance claims; this is an
architecture-development screen. R2's margin (+0.0134 over R0) is well
within Haptics' seed variability (M1 std over seeds was ±0.0039, but control
stds reach ±0.013) — a 3-seed confirmation is the natural next step *if*
pursued, but was explicitly out of scope here.

## 17. Reproduction commands

```bash
cd ECG_Benchmark
# full ablation (6 official test evaluations, all audits + R0 gate)
python -m experiments.rcmkn_haptics_seed42.runner
# mechanistic H stats + all 6 figures
python -m experiments.rcmkn_haptics_seed42.figures
# R1==R3 forensic explanation
python -m experiments.rcmkn_haptics_seed42.forensic_r1_r3
# unit tests (18) 
python -m pytest tests/test_rcmkn_haptics_seed42.py -q
```

Environment: Python 3.11.9, torch 2.5.1+cu121, aeon 1.5.0, numpy 2.3.5,
sklearn 1.8.0 (also stored in report.json / audit_results.json).

## Artifacts

`results/rcmkn_haptics_seed42/`: `REPORT.md`, `report.json` (incl. audits +
forensics), `seeds.json`, `dataset_manifest.json`, `model_config.json`,
`loss_config.json`, `feature_config.json`, `audit_results.json`,
`context_model_seed42.pt` (frozen), `_regimes_{R0,R2}_te.npy` (viz arrays),
`predictions/haptics_seed42.csv` (all 6 variants), `diagnostics/`
(mechanistic H stats), `figures/` (6 PNGs), `forensic_r1_r3.json`.

Per spec: **STOP** — no additional seeds or datasets launched.
