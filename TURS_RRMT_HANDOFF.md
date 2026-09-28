# TURS-RRMT — Final Handoff (2026-09-12, ~00:30 IST)

> **V2 UPDATE (2026-09-12, ~16:25 IST):** the TURS-RRMT-V2 readout-surgery
> pipeline (phases 8-10 were missing; now complete) finished with diagnostics,
> McNemar significance + BH-FDR, figures, and report. See
> `results/turs_rrmt_v2/TURS_RRMT_V2_REPORT.md` and the V2 section at the
> bottom of this file. Headline: every ridge readout significantly beats the
> V1 MLP head (R1: 4/4 datasets significant after FDR), best readout per
> dataset is R4 (soft-routing ridge) on ECG5000_UNBAL (0.647) and
> M=256/512-bank ridge on CWRU (0.925/0.950).

## Status
FULL PIPELINE COMPLETE with head_width=192. **3/4 datasets show statistically significant routing benefit.**

## Final Results (Test Macro-F1, head_width=192)

| Dataset | A0 Ridge | A3 Uniform | **A4 Routed** | A4-A3 Delta | McNemar p | Verdict |
|---|---|---|---|---|---|---|
| ECG5000_UNBAL | 0.376 | 0.407 | **0.417** | +0.011 | 0.118 | n.s. |
| ECG5000_BAL | 0.496 | 0.538 | **0.550** | +0.012 | **0.021** | SIGNIFICANT |
| CWRU_UNBAL | 0.886 | 0.597 | **0.695** | +0.097 | **0.016** | SIGNIFICANT |
| CWRU_BAL | 0.898 | 0.780 | **0.832** | +0.052 | **0.006** | HIGHLY SIG |

## Key Finding: The Classifier Head Was the Bottleneck

The original 64-C head wasted the representation. With head_width=192 (192->96->C):
- ECG5000_UNBAL: A4 test 0.381 -> 0.417 (+3.6pp)
- ECG5000_BAL: A4 test 0.544 -> 0.550, McNemar now significant (p=0.021)
- CWRU_UNBAL: A4 test 0.692 -> 0.695, McNemar now significant (p=0.016)
- CWRU_BAL: A4 test 0.820 -> 0.832, McNemar highly significant (p=0.006)

## Cross-Model Comparison

| Dataset | RRMT A4 | TURS-Stack softvote | Winner |
|---|---|---|---|
| ECG5000_UNBAL | 0.417 | **0.615** | TURS-Stack |
| ECG5000_BAL | 0.550 | **0.663** | TURS-Stack |
| CWRU_UNBAL | 0.695 | **0.954** | TURS-Stack |
| CWRU_BAL | 0.832 | **0.988** | TURS-Stack |

TURS-Stack still wins on all 4. The RRMT representation is strong (Ridge on A4 features gets 0.526/0.555/0.900/0.929) but the neural pipeline doesn't fully exploit it.

## Ablation Ladder (Test Macro-F1)

| Dataset | A0 Ridge | A1 Pattern | A2 1-flavor | A3 Uniform | A4 Routed | A5 NoTopK |
|---|---|---|---|---|---|---|
| ECG5000_UNBAL | 0.376 | 0.387 | 0.379 | 0.407 | 0.417 | 0.499 |
| ECG5000_BAL | 0.496 | 0.462 | 0.536 | 0.538 | 0.550 | 0.538 |
| CWRU_UNBAL | 0.886 | 0.703 | 0.629 | 0.597 | 0.695 | 0.676 |
| CWRU_BAL | 0.898 | 0.795 | 0.784 | 0.780 | 0.832 | 0.803 |

## Architecture
- Fixed pattern bank: M=128, lengths {7,11,15,23,31}, seed 42, 0 trainable params
- Transport flavors: J=4 (standard W1, tail-weighted, fine-grid W1, multi-lag drift)
- Router: 2M->32->4 MLP
- Top-K preservation: K=2
- Pattern residual: 2M->64->32 MLP
- Pooling: mean/max/std concatenation
- Classifier: Linear->ReLU->Dropout->Linear(192)->ReLU->Dropout->Linear(96)->C **[FIXED]**
- Total trainable: ~83K + 3,968 fixed kernel params

## Bugs Fixed (7 total)
1. `_uniform_routing()` P/Tv truncation mismatch
2. `d4_routing_faithfulness()` P/Tv truncation
3. `d8_counterfactual()` P/Tv truncation
4. `d4` argmax(0) -> argmax(1)[0]
5. `d5_degradation()` CUDA->numpy
6. `d7_routing_stability()` missing [0] index
7. `d7_routing_stability()` softmax outside no_grad

## Files
- Model: `models/turs_rrmt/model.py`
- Pipeline: `experiments/turs_rrmt/` (train_eval, data, diagnostics, reporting)
- Full runner: `experiments/run_turs_rrmt_full.py`
- Results: `results/turs_rrmt/<DATASET>/full_results.json`
- Comparison: `results/combined_comparison.md`
- Report: `results/turs_rrmt/TURS_RRMT_REPORT.md`

## Next Steps
1. **Multi-seed evaluation**: Run seeds 42, 123, 456, 789, 2024 to get mean+/-std and proper CI
2. **Enrich transport flavors**: T4 (multi-lag) has near-zero weight; consider more discriminating flavors
3. **Run on FI-2010**: Test with C=40 input channels
4. **Compare to MiniROCKET**: The A0 Ridge is essentially MiniROCKET-like
5. **Try larger head (256->128->C)**: Even more capacity may help further

---

# TURS-RRMT-V2 — Readout Surgery Results (2026-09-12, ~16:25 IST)

## Status
V2 pipeline COMPLETE (phases 0-10, incl. previously missing diagnostics D1-D6,
significance, figures, report). All variants replayed exactly from cached
artifacts (D1 replay delta = 0 on all but one BLAS-level 0.0018 drift).

## V2 Results (Test Macro-F1)

| Dataset | R0 MLP | R1 frz-Ridge | R2 diff-Ridge | R3 cw-Ridge | R4 soft-τ | R5 kernel | R6 M=256 | R7 M=512 |
|---|---|---|---|---|---|---|---|---|
| ECG5000_UNBAL | 0.417 | 0.487 | 0.547 | 0.436 | **0.647** | 0.475 | 0.546 | 0.479 |
| ECG5000_BAL | 0.550 | **0.586** | 0.543 | 0.516 | 0.578 | 0.492 | 0.554 | 0.547 |
| CWRU_UNBAL | 0.695 | 0.912 | 0.920 | 0.920 | 0.925 | 0.754 | 0.925 | **0.933** |
| CWRU_BAL | 0.832 | 0.934 | 0.934 | 0.934 | 0.927 | 0.781 | **0.951** | 0.940 |

## V2 Key Findings
1. **Frozen ridge beats the MLP head everywhere (R1: 4/4 significant after
   BH-FDR)** — confirms V1's suspicion that the head, not the representation,
   was the bottleneck.
2. **Best readout per dataset**: R4 soft-routing ridge on ECG5000_UNBAL
   (0.647, +23.0pp over R0, q=0.0009); R1 on ECG5000_BAL; R6/R7 larger-bank
   ridge on CWRU (0.925/0.951). Note R6/R7 use an UNTRAINED router at larger
   M — they probe bank capacity, not full-model capacity.
3. **Soft vs hard routing barely matters downstream** (D6 agreement 94-98%),
   yet soft-routing features + ridge (R4) is the single biggest ECG win.
4. **Uncertainty (D3)**: confidence margin is a consistently good error
   detector (AUROC 0.80-0.89); routing entropy is weak/inconsistent
   (0.39-0.66).
5. **R5 kernel readout (10-dim Psi) underperforms badly** — the severe
   per-flavor time-mean compression throws away the signal.

## V2 Files
- Runner: `experiments/run_turs_rrmt_v2_full.py` (phases 8-10 now implemented)
- Diagnostics: `experiments/turs_rrmt_v2/diagnostics.py`, results in
  `results/turs_rrmt_v2/diagnostics/v2_diagnostics.json`
- Significance: `results/turs_rrmt_v2/tables/hypothesis_tests.csv` (+
  `hypothesis_rows.json` for --report-only)
- Report: `results/turs_rrmt_v2/TURS_RRMT_V2_REPORT.md`
- Figures: `results/turs_rrmt_v2/figures/{variant_comparison,ridge_vs_mlp_delta}.{png,pdf}`
- Per-sample replays: `results/turs_rrmt_v2/<DS>/predictions/<V>_test_pred.npz`
- Rerun: `python experiments/run_turs_rrmt_v2_full.py` (variants cached;
  ~1 min; use `--force` to retrain, `--report-only` to rebuild the report)

---

# TURS-GLR — Gated Global-Local Regime-Routed Ridge (2026-09-12, ~20:00 IST)

## Status
FULL PIPELINE COMPLETE (all 4 datasets, A0-A9 ablations, D3-D8 diagnostics,
significance, evidence rubric, report + figures). The 10:33 run had died on a
CPU/CUDA device mismatch; fixed device placement in `models/turs_glr/model.py`,
`experiments/turs_glr_pipeline/{core,diag_runner}.py`,
`models/turs_glr/diagnostics.py`; made `build_local_block` grad-safe
(`keep_torch=True` twin for router training); fixed cache meta pickling and
the router-training OOM (static temporals now cached on CPU, moved per-chunk).

## GLR Results (Test Macro-F1, primary = A3 block ridge)

| Variant | ECG-U | ECG-B | CWRU-U | CWRU-B | Avg |
|---|---|---|---|---|---|
| A0 global-only | 0.245 | 0.167 | 0.322 | 0.249 | 0.246 |
| A1 local-only | 0.323 | 0.414 | 0.378 | 0.252 | 0.342 |
| A2 concat + scalar ridge | 0.520 | 0.573 | 0.908 | 0.944 | 0.736 |
| **A3 GLR block ridge (PRIMARY)** | 0.506 | 0.567 | 0.908 | 0.944 | **0.731** |
| A4 tau-tuned router | 0.542 | 0.550 | 0.916 | 0.920 | 0.732 |
| A7 uniform routing | 0.519 | 0.553 | 0.907 | 0.917 | 0.724 |
| **A8 MiniRocket-global + local** | **0.656** | **0.658** | **0.967** | **0.984** | **0.816** |
| A9 M_local=256 | 0.527 | 0.566 | 0.890 | 0.936 | 0.730 |

## GLR Key Findings
1. **A3 mean 0.731** — between TURS-Lite (0.771) and the V1 neural head.
2. **Block ridge ≈ scalar ridge** (A3 vs A2: n.s. on all four; RQ2 negative).
3. **Learned routing helps only CWRU_BAL** (A3 vs A7 +2.7pp, McNemar p=0.005);
   n.s. elsewhere (RQ3 mostly negative).
4. **Both single streams are WEAK** (A0 0.246 / A1 0.342 avg) — the power is
   in the concatenation, not the gating.
5. **A8 is the headline**: MiniRocket global bank + routed local stream =
   **0.816 avg, the best learned-model number in the benchmark** — beats
   MiniROCKET alone on both ECG datasets (0.656/0.658 vs 0.594/0.655), within
   ~2.5pp on CWRU. Global-local fusion with a strong global stream works.
6. Routing diagnostics: entropy ~0.96 (near-uniform soft router), switch rate
   0 (argmax route stable), faithfulness WEAK (targeted vs random drop n.s.).

## GLR Files
- Report: `results/turs_glr/TURS_GLR_REPORT.md` (+ tables/, figures/)
- Ablations: `results/turs_glr/<DS>/ablation_results.json`
- Diagnostics: `results/turs_glr/<DS>/diagnostics.json`
- Evidence rubric: `results/turs_glr/evidence_rubric.json`
- Runner: `experiments/run_turs_glr_full.py` (resumable; ~10 min total warm)
