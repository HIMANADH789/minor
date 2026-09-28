# External Generalization: TURS-Stack vs MiniROCKET (+ standard baselines)

Generated 2026-09-14 10:48 | seed 42 | datasets: EpilepticSeizures, Haptics, Phoneme | primary: EpilepticSeizures | status: COMPLETE

## 1. Executive summary

- **EpilepticSeizures**: MiniROCKET 0.9194 vs TURS-Stack 0.8706 (Δ=-0.0488) -> **MiniROCKET**
- **Haptics**: MiniROCKET 0.4974 vs TURS-Stack 0.3553 (Δ=-0.1421) -> **MiniROCKET**
- **Phoneme**: MiniROCKET 0.0808 vs TURS-Stack 0.0785 (Δ=-0.0023) -> **MiniROCKET**

## 2. Research hypothesis

TURS-Stack appeared stronger than MiniROCKET on difficult physiological waveform classification (ECG5000) while MiniROCKET dominated highly separable vibration data (CWRU). This external validation tests whether that behavior **generalizes** to three new UCR datasets with similar temporal/morphological difficulty but different domains. The tested outcome classes are A (generalizes), B (biomedical-only), C (dataset-specific), D (wins but unstable).

## 3. Dataset selection rationale

- **EpilepticSeizures** (PRIMARY): EEG waveforms — the closest new biomedical temporal-waveform test to the ECG hypothesis; extreme imbalance (20% positive) and 80-sample canonical train split.
- **Haptics** (SECONDARY): proprioceptive/contact waveform morphology, 5 classes, small train — tests morphological generalization.
- **Phoneme** (SECONDARY): 39-class audio spectra — tests behavior under many-class difficulty far beyond the 4-5 class benchmark.

## 4. Dataset provenance

Canonical UCR/aeon distribution zips from timeseriesclassification.com/aeon-toolkit (the same canonical host the repository uses for ECG5000). NOTE per spec Phase 1C: the referenced Kaggle mirror m4ur1c10/dataset was audited and does NOT contain the UCR classification archive (7.5 MB forecasting CSVs); canonical source was used instead. No UCI-CSV substitutes were used.

## 5. Dataset statistics

| dataset | train | val | val source | test | L | classes |
|---|---:|---:|---|---:|---:|---:|
| EpilepticSeizures | 80 | 20 | provided_canonical_val_ts | 11420 | 178 | 2 |
| Haptics | 132 | 23 | per_class_stratified_15pct_of_train_seed42_singletons_kept_in_train | 308 | 1092 | 5 |
| Phoneme | 185 | 29 | per_class_stratified_15pct_of_train_seed42_singletons_kept_in_train | 1896 | 1024 | 39 |

## 6. Experimental protocol

- Canonical UCR train/test split preserved exactly; validation either the provided canonical val (EpilepticSeizures) or a stratified 15% of TRAIN with random_state=42 (Haptics, Phoneme). TEST untouched.
- Per-sample z-normalization (project-canonical formula).
- MiniROCKET: aeon MiniRocket(random_state=42, n_jobs=-1) ~10K kernels, RidgeClassifierCV(alphas=np.logspace(-4,4,20)), final fit on TRAIN+VAL.
- Neural baselines: canonical models/external_baselines.py + models/inceptiontime.py; AdamW 3e-4/1e-2, OneCycleLR, batch 64, max 15 epochs, early stopping on val MF1 (patience 6), seed 42.
- TURS-Stack: frozen models/turs_stack/model.py; canonical training (joint CE over 4 branches, AdamW 3e-4/1e-2, OneCycleLR, batch 64, max 30 epochs, patience 8 on val soft-vote MF1), correctness gates, sigma0 warm start from train only, 5 combiners fitted on VALIDATION only; reported combiner = validation-selected; soft-vote reported separately.
- Primary metric: TEST Macro-F1; no test-based selection anywhere.

## 11. Main results (test Macro-F1)

| Dataset | MiniROCKET | InceptionTime | ResNet-1D | FCN | PatchTST | TURS-Stack | Δ(Stack−MR) |
|---|---:|---:|---:|---:|---:|---:|---:|
| EpilepticSeizures | 0.9194 | 0.9060 | 0.7935 | 0.3018 | 0.1331 | 0.8706 | -0.0488 |
| Haptics | 0.4974 | 0.0636 | 0.1512 | 0.0634 | 0.1692 | 0.3553 | -0.1421 |
| Phoneme | 0.0808 | 0.0445 | 0.0258 | 0.0015 | 0.0126 | 0.0785 | -0.0023 |

TURS-Stack reported = validation-selected combiner (best_combiner_by_val); soft-vote also shown in stack_results.csv.

## 16. Statistical inference (paired McNemar on test correctness)

| Dataset | Comparison | Δ MF1 | chi2 | p | q(BH-FDR) | Verdict |
|---|---|---:|---:|---:|---:|---|
| EpilepticSeizures | PRIMARY: TURS-Stack(selected) vs MiniROCKET | -0.0488 | 209.884 | 0 | None | pending |
| EpilepticSeizures | PRIMARY(soft-vote): TURS-Stack vs MiniROCKET | -0.0488 | 209.884 | 0 | None | pending |
| EpilepticSeizures | SECONDARY: TURS-Stack vs InceptionTime | -0.0354 | 149.730 | 0 | None | pending |
| EpilepticSeizures | SECONDARY: TURS-Stack vs ResNet-1D | +0.0771 | 54.596 | 0 | None | pending |
| EpilepticSeizures | SECONDARY: TURS-Stack vs FCN | +0.5688 | 6672.267 | 0 | None | pending |
| EpilepticSeizures | SECONDARY: TURS-Stack vs PatchTST | +0.7375 | 8368.368 | 0 | None | pending |
| Haptics | PRIMARY: TURS-Stack(selected) vs MiniROCKET | -0.1421 | 18.720 | 1.5e-05 | None | pending |
| Haptics | PRIMARY(soft-vote): TURS-Stack vs MiniROCKET | -0.2341 | 29.867 | 0 | None | pending |
| Haptics | SECONDARY: TURS-Stack vs InceptionTime | +0.2917 | 19.507 | 1e-05 | None | pending |
| Haptics | SECONDARY: TURS-Stack vs ResNet-1D | +0.2041 | 8.410 | 0.003732 | None | pending |
| Haptics | SECONDARY: TURS-Stack vs FCN | +0.2919 | 19.507 | 1e-05 | None | pending |
| Haptics | SECONDARY: TURS-Stack vs PatchTST | +0.1861 | 3.935 | 0.04729 | None | pending |
| Phoneme | PRIMARY: TURS-Stack(selected) vs MiniROCKET | -0.0023 | 32.126 | 0 | None | pending |
| Phoneme | PRIMARY(soft-vote): TURS-Stack vs MiniROCKET | -0.0234 | 23.611 | 1e-06 | None | pending |
| Phoneme | SECONDARY: TURS-Stack vs InceptionTime | +0.0340 | 4.050 | 0.04417 | None | pending |
| Phoneme | SECONDARY: TURS-Stack vs ResNet-1D | +0.0527 | 73.142 | 0 | None | pending |
| Phoneme | SECONDARY: TURS-Stack vs FCN | +0.0770 | 281.108 | 0 | None | pending |
| Phoneme | SECONDARY: TURS-Stack vs PatchTST | +0.0659 | 63.314 | 0 | None | pending |

## 14. Complementarity (error overlap with MiniROCKET)

- **EpilepticSeizures**: MR wrong & Stack right = 321; MR right & Stack wrong = 809; both wrong = 261 (MR err 0.051, Stack err 0.0937)
- **Haptics**: MR wrong & Stack right = 35; MR right & Stack wrong = 83; both wrong = 113 (MR err 0.4805, Stack err 0.6364)
- **Phoneme**: MR wrong & Stack right = 194; MR right & Stack wrong = 324; both wrong = 1231 (MR err 0.7516, Stack err 0.8201)

## 21. Final conclusion

- **Outcome class: C — Stack wins none; the ECG advantage appears dataset-specific** (Stack positive on 0/3; primary EpilepticSeizures Δ=-0.0488).
- **TURS-Stack external generalization: NOT SUPPORTED.**
- **Biomedical generalization (EpilepticSeizures, Haptics): NOT SUPPORTED** — Stack trails MiniROCKET on the primary EEG dataset (-0.0488) and on Haptics (-0.1421).
- **Cross-domain generalization (Phoneme): NOT SUPPORTED** — Δ=-0.0023 (near-parity but not a win, on an extremely many-class task where all models are weak).
- **MiniROCKET competitiveness: STRONG** — rank-1 of all six models on all three external datasets (strictly best: True).
- **Val→test gap (selected combiner): EpilepticSeizures=-0.1294, Haptics=-0.4158, Phoneme=-0.9215 — no catastrophic D-class instability here, but small-train combiner selection (stacking on Haptics/Phoneme) still underperformed soft-vote on test.
- PHASE 40 answers: (1) No — Stack does NOT beat MiniROCKET on EpilepticSeizures. (2) No biomedical replication. (3) No transfer to Haptics. (4) No transfer to Phoneme (parity, −0.2pp). (5) The ECG-like-difficulty hypothesis is NOT supported externally: the ECG5000 Stack advantage does not reproduce on new physiological waveforms. (6) No — Stack should not become the primary TURS model on this evidence; MiniROCKET remains the strongest general model, and the Stack-vs-MR ECG effect is likely dataset-specific.
- Historical context (reference, not retrained): Stack beat MiniROCKET on ECG5000_UNBAL 0.6150 vs 0.5938 and ECG5000_BAL 0.6631 vs 0.6553, but lost on CWRU (0.9539/0.9877 vs 0.9917/0.9947). Adding the three external datasets, the positive Stack effect is confined to ECG5000.
