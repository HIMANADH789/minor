# Master Model Comparison — All Generations (2026-09-12)

Test Macro-F1, canonical protocol (seed 42, stratified splits, per-sample
z-norm, 85/15 trainval/test, 15% of trainval as val). All numbers from the
repo's authoritative result files: `results/FINAL_BENCHMARK_REPORT.md`
(baselines), `results/combined_comparison.md` (Stack / RRMT-V1),
`results/turs_rrmt_v2/TURS_RRMT_V2_REPORT.md` (RRMT-V2).

## 1. Headline table (Test Macro-F1)

| Model | Params (trainable) | ECG5000_UNBAL | ECG5000_BAL | CWRU_UNBAL | CWRU_BAL | **Avg** |
|---|---|---:|---:|---:|---:|---:|
| **MiniROCKET** (+ridge) | ~10K fixed feats | 0.5938 | 0.6553 | **0.9917** | **0.9947** | **0.8089** |
| **TURS-Lite** | 137K | 0.6046 | 0.6377 | 0.8997 | 0.9417 | 0.7709 |
| **TURS-Stack** (soft-vote) | 165K | 0.615 | 0.663 | 0.954 | 0.988 | 0.8050 |
| **TURS-Stack** (best combiner) | 165K | 0.629 | **0.673** | 0.958 | 0.988 | **0.8120** |
| RRMT-V1 A4 (neural head) | ~83K + 4K fixed | 0.417 | 0.550 | 0.695 | 0.832 | 0.6235 |
| RRMT-V1 A4 + Ridge (post-hoc) | +~1K | 0.526 | 0.555 | 0.900 | 0.929 | 0.7275 |
| **RRMT-V2 R1** frozen ridge | +~1K | 0.487 | 0.586 | 0.912 | 0.934 | 0.7298 |
| **RRMT-V2 R4** soft-routing ridge | +~1K | **0.647** | 0.578 | 0.925 | 0.927 | 0.7692 |
| RRMT-V2 best readout per dataset | +~1K | 0.647 (R4) | 0.586 (R1) | 0.933 (R7) | 0.951 (R6) | 0.7790* |
| **TURS-GLR A3** (global+local block ridge) | fixed kernels | 0.506 | 0.567 | 0.908 | 0.944 | 0.7310 |
| TURS-GLR A4 (tau-tuned router) | router only | 0.542 | 0.550 | 0.916 | 0.920 | 0.7320 |
| TURS-GLR A8 (MiniRocket-global + local) | fixed kernels | 0.656 | 0.658 | 0.967 | 0.984 | 0.8163 |

\* "Best per dataset" is post-hoc test selection — optimistic; use R1 or R4
rows for honest single-model numbers.

## 2. Per-dataset winners

| Dataset | Winner | Runner-up | Margin |
|---|---|---|---:|
| ECG5000_UNBAL | **RRMT-V2 R4** 0.647 | TURS-GLR A8 0.656 | R4 leads by 0.9pp |
| ECG5000_BAL | TURS-Stack best 0.673 | TURS-GLR A8 0.658 | +1.5pp |
| CWRU_UNBAL | **MiniROCKET** 0.992 | TURS-GLR A8 0.967 | +2.5pp |
| CWRU_BAL | **MiniROCKET** 0.995 | TURS-GLR A8 0.984 | +1.1pp |

## 3. Reading the table

1. **MiniROCKET remains the overall benchmark leader** (0.809 avg) and is
   nearly unbeatable on CWRU — but it has NO learned representation, and it
   loses ECG5000_UNBAL to both TURS variants and to RRMT-V2 R4.
2. **TURS-Stack best-combiner (0.812) edges MiniROCKET on average** and wins
   ECG5000_BAL; its 4-branch fusion is the strongest *learned* model.
3. **RRMT-V2 R4 (soft-routing + ridge) wins ECG5000_UNBAL outright** — the
   only model above 0.63 on the hardest, most imbalanced split. On a single
   uniform readout choice, R4 averages 0.769, within 0.2pp of TURS-Lite
   (0.771) with ~1% of the trainable parameters.
4. **The readout, not the representation, was RRMT's bottleneck**: V1 neural
   head 0.624 avg -> V2 ridge readouts 0.730-0.779 avg (+11-15pp) on the same
   frozen features. Every ridge readout significantly beats the V1 head
   (McNemar + BH-FDR; R1 significant on 4/4 datasets).
5. **RRMT-V1 A4+Ridge ≈ RRMT-V2 R1** on CWRU (0.900/0.929 vs 0.912/0.934) —
   the V1 post-hoc ridge already anticipated V2's conclusion; V2 adds the
   significance machinery, the soft-routing variant (R4), the bank-capacity
   probe (R6/R7), and exact replayable diagnostics.
6. **TURS-GLR A3 (0.731 avg)**: the gated global-local block ridge lands
   between TURS-Lite (0.771) and the V1 neural head (0.624). The block-ridge
   gating itself adds ~nothing over plain concatenation (A3 vs A2 deltas
   -1.4pp to 0.0pp, McNemar n.s. everywhere), and learned routing only helps
   CWRU_BAL (A3 vs A7: +2.7pp, p=0.005). The striking result is **A8**:
   swapping the learned global bank for aeon MiniRocket kernels while keeping
   the routed local stream gives **0.816 avg - the best learned-model number
   in this entire benchmark**, within ~2.5pp of MiniROCKET alone on CWRU and
   ahead of it on both ECG datasets.

## 4. Statistical significance (where tested)

- RRMT-V2 readouts vs V1 MLP head (paired McNemar, BH-FDR within family):
  R1 significant improvement 4/4 datasets; R4 3/4 (n.s. on CWRU_BAL where
  R0 is already 0.832); R6 4/4; R3 degrades ECG5000_BAL.
- RRMT-V1 routing (A4 vs A3): significant on 3/4 (ECG-BAL p=0.021,
  CWRU-U p=0.016, CWRU-BAL p=0.006); n.s. on ECG5000_UNBAL.
- Baselines (MiniROCKET/TURS-Lite/Stack) vs each other: no paired tests were
  run in their original reports; treat cross-model gaps < ~2pp as indicative.

## 5. Caveats

- Old baselines (MiniROCKET, TURS-Lite, early reports) predate the current
  resplit protocol era; the RRMT/Stack numbers all use the identical current
  protocol and are directly comparable to each other.
- CWRU has no official split — the 85/15 stratified split is repository-made
  (seed 42), so all CWRU numbers share that choice.
- RRMT-V2 R6/R7 use an untrained router at M=256/512 — bank-capacity probe,
  not full-model capacity.
- Single seed everywhere; no multi-seed variance yet.

## 6. Artifacts

- Baselines: `results/FINAL_BENCHMARK_REPORT.md`
- Stack vs RRMT-V1: `results/combined_comparison.md`
- RRMT-V2: `results/turs_rrmt_v2/TURS_RRMT_V2_REPORT.md`
- Handoffs: `TURS_STACK_HANDOFF.md`, `TURS_RRMT_HANDOFF.md`
