# LABEL-FREE INTRINSIC TEMPORAL HETEROGENEITY AUDIT — Haptics, seed 42

Experiment namespace: `experiments/haptics_intrinsic_heterogeneity/`
Module: `analysis/intrinsic_heterogeneity.py`
All prior experiment artifacts untouched (test-enforced).

## 1. Research question

Does Haptics exhibit a nontrivial, stable, **label-free** measure of temporal structural heterogeneity that can be converted into a deterministic H-capacity signal **before** any validation or classifier fitting?

## 2. Why validation-selected H budget is insufficient

Selecting the H budget by validation Macro-F1 is a search: it consumes validation data, risks selection overfitting with n=23, provides no explanation, and cannot transfer to a new dataset without re-running the whole sweep. A signal-intrinsic statistic would let the budget be fixed **before** validation, deterministically, and (in later multi-dataset work) be tested as an explanation of *when* H helps.

## 3. Definition of intrinsic temporal temporal heterogeneity

A raw-signal property: the degree to which the spectral content of X(t) changes across the temporal axis, measured without labels, without any learned representation, and without MiniROCKET/HERAMBA features. Operatively: divergence between the normalized power spectra of consecutive temporal windows, averaged over window pairs and channels.

## 4. Primary spectral index

**HI_spec(sample)** = mean over channels × mean over window pairs (i<j) of Jensen–Shannon divergence between the normalized power spectra of windows i and j. Predeclared preprocessing (frozen before any predictive result was consulted): W=8 equal contiguous windows; per-window linear (endpoint) detrending; hann window; rfft with n_fft = next power of two ≥ window length; one-sided power with epsilon 1e-12; per-window normalization to sum 1; natural-log JS convention (scipy `jensenshannon(base=e)`, squared); mean over the 28 window pairs. Haptics is univariate (C=1). No RNG anywhere; the index is exactly deterministic.

## 5. Secondary variance index

**HI_var(sample)** = variance of window means / (mean of window variances + 1e-12) — a scale-dependent but simple non-stationarity statistic. Diagnostic only; **not** used for the budget. Train values: mean 3.02, median 2.93, SD 1.13, range [1.04, 11.04].

## 6. Stationarity diagnostics

Supporting only, on the window-mean series of each train sample (documented aggregate; never used for the budget): ADF finds 0/132 stationary at α=0.05; KPSS flags 132/132 as non-stationary. Every Haptics training series is non-stationary by both tests — consistent with genuinely time-varying structure, though these tests are not the heterogeneity measure.

## 7. Raw Haptics data and split

Canonical loader `experiments/external_stack_generalization/data.load_dataset("Haptics")`: Xtr (132, 1092) float32, Xva (23, 1092), Xte (308, 1092); univariate; T=1092; 5 classes; raw amplitudes (unnormalized), finite. Class distribution (train): 15/29/29/31/28.

## 8. Train-only protocol

HI_spec and HI_var were computed on the 132 training raw series only. The dataset index, B_H, and the H group selection used train data exclusively. Validation/test indices were computed afterwards for reporting only. Labels were touched only for post-hoc figures after the index was frozen.

## 9. Dataset-level heterogeneity

- **HI_dataset = median(HI_spec over train) = 0.05770** (predeclared median rule)
- Train sample stats: mean 0.05786, median 0.05770, std 0.03605, IQR 0.05239, range [0.00361, 0.16180], 132 unique values (all distinct)
- val split index 0.02782, test split index 0.05284 (diagnostic only)

## 10. Sample-level heterogeneity distribution

Non-degenerate (variance 1.30e-3, all 132 sample values unique, CV 0.62): the index spans more than a 40-fold range across training samples, from nearly monochromatic series (0.0036) to strongly time-varying ones (0.162). Distribution and the post-hoc per-class overlay (index already frozen) are in `figures/heterogeneity_distribution.*`; per-window spectra for the lowest/median/highest samples in `figures/spectral_heterogeneity_examples.*`; per-sample values in `sample_heterogeneity.csv`.

## 11. Stability analysis

Dataset level: medians 0.05363 (W=7), **0.05770 (W=8, primary)**, 0.03953 (W=9) — same order of magnitude; converted B_H would be 0.077 / **0.083** / 0.057, i.e. the **"low" band is unchanged** across neighboring window counts. Sample level: rank correlation with the primary index is weak (Spearman 0.120 at W=7, 0.223 at W=9). **Honest assessment:** the dataset-level index and the resulting band are stable, but the *sample-level ranking* is sensitive to the window count — HI_spec at the sample level should be treated as windowing-dependent, and sample-level uses of the index would need a window-robust definition (e.g., multi-W averaging) before being trusted. W=7/9 were diagnostics only; W=8 remains primary.

## 12. Intrinsic H-budget rule

Predeclared, deterministic, label-free:
- **B_H = clip(HI_dataset / log(2), 0, 1)** with HI_reference = log 2, the theoretical maximum JS divergence under the natural-log convention (never fitted from performance)
- **B_H = 0.05770 / 0.69315 = 0.08324** → band **"low"** ([0.00, 0.20))
- Descriptive bands (predeclared): [0,0.20) low · [0.20,0.50) moderate · [0.50,0.75) high · [0.75,1] very high

B_H is an **intrinsic H-capacity signal**, not the optimal H fraction and not a validated budget.

## 13. Leakage audit

`leakage_audit.json`: the train index is bit-identical (0.05769635185101932) when recomputed with validation and test signals poisoned (+1e6 on every value) in memory — held-out signals cannot enter the budget by construction (the index pipeline only receives Xtr). No index function has a label argument (AST-verified in tests). Unit tests 11–12 pass; unit tests 1–10, 13–16 all pass (16/16).

## 14. Downstream demonstration (secondary; NOT an optimized model)

A single demonstration was run to place the index in context. Identity gates first: MiniROCKET (G, canonical Ridge, train+val) = **0.5037** ✓ exact; raw [G‖H] = **0.5500** ✓ exact. Then three deterministic label-free capacity mappings from B_H=0.0832 (train-only H-carrier ranking = train-median of the normalized heterogeneity statistic; deterministic index tie-break), each evaluated once on test:

| Configuration | H features kept | Test Macro-F1 |
|---|---|---|
| G only (gate) | 0 | 0.5037 |
| Intrinsic-budget demonstration (quantile: ceil(B_H·4998)) | 417 | 0.5155 |
| Intrinsic-budget demonstration (64-kernel blocks: ceil(B_H·78)) | 384 | 0.5149 |
| Intrinsic-budget demonstration (8-slice hierarchy) | 8 | 0.5037 |
| Full [G‖H] (gate) | 4998 | 0.5500 |

The intrinsic-budget models improve over G-only (+1.18 pp at 417 features) but do not reach full H (0.5500). This is a demonstration that a configuration **can be fixed before validation from raw signal structure** — it is not a tuned or optimized model, and no H-capacity value was searched.

## 15. Relationship to known Haptics G vs G+H result (post-hoc)

The intrinsic index was computed independently of labels and predictive performance; the known Haptics predictive effect (Δ_H = 0.5500 − 0.5037 = +4.63 pp) is shown only afterward as an external diagnostic. Read as such: Haptics receives a **"low"** intrinsic spectral-heterogeneity score (B_H = 0.083) while exhibiting a **substantial** H benefit. This is directionally *inconsistent* with the naive reading "more spectral heterogeneity → more H helps," on this single dataset. It does not refute the cross-dataset hypothesis — Haptics may carry non-spectral structure (e.g., regime non-stationarity of activation patterns rather than spectral diversity), and one dataset cannot test the mapping — but it removes any basis for claiming the proposed index already explains Haptics' large Δ_H.

## 16. Limitations

Single dataset, single split convention. W=8 is arbitrary-but-fixed; sample-level rank stability across W is weak (Spearman 0.12–0.22), so the sample-level index is windowing-dependent even though the dataset-level band is not. JS divergence on 137-sample hann-windowed spectra has modest frequency resolution; detrending/hann choices are predeclared, not justified as optimal. B_H's denominator (log 2) is a theoretical ceiling rarely approached by real window spectra, which compresses B_H toward the low band — the band edges are conventions, not calibrated operating points. The carrier-strength group ranking (train-median heterogeneity statistic) is one of several defensible label-free rules; the three capacity mappings (417/384/8 features) straddle B_H·capacity by construction, so the demonstration's insensitivity to the exact mapping is untested. No significance or causality claims are made anywhere.

## 17. What this experiment establishes

On Haptics seed 42: (1) HI_spec is well-defined, exactly deterministic, and **non-degenerate** (all 132 train values unique, 40-fold dynamic range); (2) it is dataset-level **stable in band** across neighboring W (low at W=7/8/9); (3) the predeclared rule produces a clear deterministic B_H = 0.083 ("low") computable **before** any classifier is fitted; (4) the entire budget path uses train raw signals only — leakage-audited and unit-tested; (5) a label-free capacity mapping can be executed deterministically from B_H (demonstrated at 417/384/8 features).

## 18. What it does NOT establish

That B_H predicts when H helps (cross-dataset question, untested). That B_H = 0.083 is the optimal H fraction for Haptics (it is not; full H still wins on test). That the index is robust at the sample level (it is not, across W). That spectral heterogeneity is the right intrinsic quantity — Haptics' large Δ_H despite a low spectral score suggests the relevant structure may be non-spectral. That the downstream demonstration's +1.18 pp over G is meaningful (single test evaluation, within documented test-set noise).

## 19. Required multi-dataset follow-up

The mapping heterogeneity → H benefit can only be tested across datasets with varying Δ_H: compute HI_dataset and B_H with the identical frozen rule on the remaining canonical datasets (ECG5000_UNBAL/BAL, CWRU_UNBAL/BAL, Phoneme, EpilepticSeizures, UWave×4, GunPoint, IPD, FordA — all with existing verified M0 and R5 numbers), then assess rank association between B_H and Δ_H(ρ\*/50-50) post-hoc. A sample-level version additionally requires a window-robust index (e.g., fixed multi-W ensemble) decided before that study. This is a separate experiment; nothing here was tuned to anticipate its outcome.
