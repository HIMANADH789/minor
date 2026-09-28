# CONTEXT-DEPENDENCE SCREEN — DRTN-CONDITIONED MINIROCKET (seed 42)

Datasets: GunPoint, ItalyPowerDemand, FordA — verified Kaggle UCRArchive_2018
copies (`qianhuan/ucrarchive-2018`), previously established bit-identical to
canonical aeon data. Implementation: the audited Stage-A core, reused unchanged
(per-sample-occupancy M2, valid-region H, train+val Ridge fit, test-once),
plus the mandated **A_SOFT** soft-assignment ablation.

## SECTION 1 — Dataset verification

| Dataset | source | train | val | test | T | classes | labels |
|---|---|---|---|---|---|---|---|
| GunPoint | data/kaggle/GunPoint | 42 | 8 | 150 | 150 | 2 | {1, 2} |
| ItalyPowerDemand | data/kaggle/ItalyPowerDemand | 56 | 11 | 1029 | 24 | 2 | {1, 2} |
| FordA | data/kaggle/FordA | 3060 | 541 | 1320 | 500 | 2 | {−1, 1} |

All structural gates asserted at load time against the canonical UCR shapes
recorded at install (TSV file SHA-256 hashes stored in per-dataset
`diagnostics/dataset_verification.json`). Val split = stratified 15% of TRAIN,
seed 42 (repository rule). No file was modified.

## SECTION 2 — Experimental configuration

- M0: `aeon MiniRocket(random_state=42)`, 9996 features (verified 9996 at T=24/150/500).
- M1: 4998 global PPV + 4998 DRTN hard-regime heterogeneity.
- M2: same, per-sample occupancy-matched random regimes (RNG seed+900001).
- M3: same, per-sample shuffled DRTN regimes (RNG seed+900002).
- A_SOFT: 4998 global + 4998 soft conditional-rate heterogeneity
  `PPV_{m,k} = Σ_t q_tk·a_mt / Σ_t q_tk`, `q_k = mean_t(q_tk)`, valid-mask
  restricted; soft assignments `softmax(−d²(z_t,c_k)/τ)` from the FROZEN R5
  codebook, τ=0.5 taken from the frozen DRTN config (SoftCodebook semantics;
  no architecture change, nothing tuned).
- Classifier: `RidgeClassifierCV(alphas=np.logspace(-4,4,20))`, fit on
  train+validation; test touched exactly once per variant.
- DRTN: R5 K=8, audited config unchanged; trained on TRAIN only, val-Macro-F1
  checkpoint selection, frozen before feature extraction (frozen-state verified
  in-run by state-dict comparison).

## SECTION 3 — Audit results (all green, every dataset)

| Audit | Evidence |
|---|---|
| M0 feature identity | chunked raw-extractor PPV vs aeon: **max\|diff\| = 0.00e+00** (train+val, all datasets); aeon transform verified deterministic |
| Feature budget | M0/M1/M2/M3/A_SOFT all **9996 = 4998 + 4998**; global blocks `array_equal` to M0's; ranges [0,4998) / [4998,9996) printed |
| M2/M3 arrays | distinct hashes every dataset (FordA test: `ad874896…` vs `2aa4c659…`, 556,296/660,000 positions differ); `shares_memory=False`; independent RNG offsets 900001/900002 |
| Occupancy | per-sample histogram equality M2 & M3, all splits: **0 failures** |
| H formula | independent float64 recompute: **max diff ≤ 1.2e-8** (wrong-formula scale ≥ 1e-3 here) |
| Valid region | per-feature-mask corruption: 3.5e10–3.99e9 out-of-valid activations flipped → **H unchanged exactly** |
| A_SOFT non-placeholder | recompute ≤ 1.7e-17; nonzero fractions 0.82–0.98; H_soft ≠ H_hard; sharp-assignment construction confirms non-collapsing |
| Fit protocol | every Ridge fit on train+val rows ( asserted by construction); no test labels in any fitting/selection path |

M0 reference: no stored canonical artifact exists for these three datasets in
the repository (first benchmark appearance). Per spec the canonical seed-42
reference was **computed** from the canonical aeon MiniRocket under the exact
benchmark protocol; integrity is established by the bit-exact extractor
identity check rather than a stored value.

## SECTION 4 — Test results (Macro-F1, seed 42)

| Dataset | M0 | M1 | M2 | M3 | A_SOFT |
|---|---|---|---|---|---|
| GunPoint | 0.9933 | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| ItalyPowerDemand | **0.9650** | 0.9572 | 0.9602 | 0.9592 | 0.9572 |
| FordA | 0.9499 | 0.9484 | 0.9431 | 0.9371 | **0.9522** |

Validation Macro-F1 and selected alphas are in per-dataset `result.json`
(GunPoint: all val 1.0; FordA: M0 val 0.9930, A_SOFT val 0.9852).

## SECTION 5 — Per-dataset deltas (test Macro-F1)

| Dataset | M1−M0 | M1−M2 | M1−M3 | M1−A_SOFT | A_SOFT−M0 |
|---|---|---|---|---|---|
| GunPoint | +0.0067 | 0.0000 | 0.0000 | 0.0000 | +0.0067 |
| ItalyPowerDemand | −0.0078 | −0.0030 | −0.0020 | 0.0000 | −0.0078 |
| FordA | −0.0015 | +0.0053 | +0.0113 | −0.0038 | **+0.0023** |

## SECTION 6 — Context-dependence diagnostics (test split)

Mean heterogeneity scale H(M1) vs destroyed-alignment controls:

| Dataset | mean H (M1) | mean H (M2) | mean H (M3) | mean H (A_SOFT) | H1/H2 ratio | regime norm-entropy | perplexity |
|---|---|---|---|---|---|---|---|
| GunPoint | 0.02052 | 0.00155 | 0.00167 | 0.02053 | **13.3×** | 0.33 | 1.99 |
| ItalyPowerDemand | 0.02591 | 0.01108 | 0.01100 | 0.02587 | **2.3×** | 0.31 | 1.92 |
| FordA | 0.02177 | 0.00377 | 0.00378 | 0.02215 | **5.8×** | 0.97 | 7.57 |

Learned temporal organization creates substantially stronger kernel–context
dependence than occupancy-matched arbitrary organization on all three datasets
(H1 ≫ H2, H3; A_SOFT's dispersion matches H1's scale). This mechanistic signal
did **not** convert into test Macro-F1 gains on ItalyPowerDemand/FordA — the
extra dispersion is present but not discriminatively useful under this
protocol. GunPoint/ItalyPowerDemand DRTN codebooks collapsed to ~2 dominant
codes (entropy ≈ 0.32); FordA used all 8 codes (entropy 0.97).

## SECTION 7 — Dataset-by-dataset verdict

| Dataset | Verdict | Reason |
|---|---|---|
| GunPoint | **NEUTRAL** (ceiling-saturated) | M1 = M2 = M3 = A_SOFT = 1.0000 ≥ M0: every variant solves the task perfectly; the Haptics signature (M1 above controls) is unassessable at ceiling. |
| ItalyPowerDemand | **NEGATIVE** | M1 ≤ M0 (−0.0078); controls equally below M0; DRTN regimes collapsed to ~2 codes. |
| FordA | **NEGATIVE** (M1), with a noteworthy A_SOFT observation | M1 ≤ M0 (−0.0015). A_SOFT is the only variant above M0 (+0.0023) and above M1/M2/M3 — consistent with the soft-context dispersion diagnostic, but single-seed and small-margin; not a confirmation. |

None of the three shows the Haptics/ECG5000_BAL signature (M1 > M0 **and**
M1 > M2/M3 with controls below M0).

## SECTION 8 — 3-seed confirmation?

**None recommended.** GunPoint is saturated (uninformative), ItalyPowerDemand
is negative, and FordA's A_SOFT margin (+0.0023) is too small to justify
confirmation on a single seed without a hypothesis change. The
context-dependence hypothesis is **not supported** on these three datasets.

## SECTION 9 — Reproducibility commands

```bash
cd ECG_Benchmark
# official screen (15 test evaluations, seed 42)
python -m experiments.drtn_conditioned_minirocket_context3_seed42.runner
# per-dataset:  --datasets GunPoint | ItalyPowerDemand | FordA
# smoke (6-epoch DRTN, not saved):  --smoke
python -m experiments.drtn_conditioned_minirocket_context3_seed42.figures
python -m pytest tests/test_drtn_conditioned_minirocket_context3_seed42.py -q
```

## SECTION 10 — Implementation issues (all before any affected test evaluation)

1. **Soft-heterogeneity broadcasting bug** `(K,Fp)/(1,K)` — caught by the
   mandated unit tests **before any run**; fixed (`[:, None]`).
2. **`HardVQ` has no `tau` attribute** — soft conversion now takes τ from the
   frozen DRTN config explicitly; no hack of the model.
3. **`log(flush=)` TypeError** — crashed before any audit/evaluation (smoke).
4. **`KeyError: 'M4'`** in deltas dict — smoke-only crash after classifier
   prints, before saving; fixed to `A_SOFT` keys.
5. **FordA OOM (6.14 GiB alloc) at AUDIT 6** — crashed **before any FordA
   test evaluation**, after GunPoint/ItalyPowerDemand had completed
   official runs. Fixed by chunking AUDIT 6 (per-128-sample blocks) and
   replacing the full train+val activation tensor (~21 GiB for FordA) with a
   bounded chunked computation (row-independent math, identity check still
   bit-exact). FordA was then rerun from its frozen DRTN checkpoint.

**Test-evaluation log:** 15 official evaluations total (3 datasets × 5
variants), each test set touched exactly once. GunPoint and ItalyPowerDemand
completed in the first official pass; FordA's two crashed attempts never
reached test evaluation. No result was invalidated; the test-once policy held.
