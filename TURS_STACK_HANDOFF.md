# TURS-Stack — Handoff Note (2026-09-11, ~16:31 IST)

## Status
FULL SEQUENTIAL TRAINING RUNNING (PID 23112, CUDA RTX 4050 Laptop).
Log: `ECG_Benchmark/logs/turs_stack_full.log`
Progress: ECG5000_UNBAL ep 18/30, soft-vote val MF1 0.6608 (best @ ep17),
branch val MF1 (lite/rv/cs/cmr) ~0.60–0.67. ~9.5 s/epoch on ECG.

## What was done this session
1. GPU/process audit: no running project processes; stale `turs_cra_bench.pid`
   (PID 1115) pointed at a deleted launcher — no auto-restart risk. GPU was
   free before launch; nothing was terminated.
2. Repository audit: benchmark protocol source = `run_turs_cs_benchmark.py` /
   `fair_turs.py` (seed 42, stratified val 15%, z-norm, AdamW 3e-4/1e-2,
   OneCycleLR, CE, batch 64, max 30 ep, patience 8, grad clip 1.0).
   Branch mechanisms ported from `models/turs_cs/model.py` (soft dilation,
   ScaleParams, within-window EMA) and `models/tursnet.py` (Lite fusion).
   Historical TURS-Stack scaffold was incomplete → fully rewritten.
3. `models/turs_stack/model.py` — canonical implementation:
   shared TransportBuilder+TransportEncoder+InceptionBackbone (called ONCE,
   counter-verified), 4 branches (Lite / RV / CS / CMR) with independent
   projections, per-branch P_T/P_I/P_alpha fusion, compact classifiers,
   5 combiners (soft, hard w/ confidence tie-break, static θ(4),
   linear stacking 4C→C, diagnostic stacking 4C+1→C), novelty e_t = 
   tanh(1 − cos(z_t, zbar₁)) from Branch 3 (label-free).
4. `experiments/run_turs_stack_benchmark.py` — full-spec runner:
   correctness gates, train-only sigma0 warm start, joint training (mean CE
   over branches), freeze → val predictions → combiner fit on val ONLY →
   test eval of all five, per-branch/confusion/pred-dist diagnostics,
   RV correction/velocity ratio, beta stats, complementarity analysis,
   params incl. 4×-independent estimate, resumable checkpoints.
5. Gates passed on all 4 datasets (smoke): trunk calls=1/forward, branch
   independence, beta simplex, RV response active, no cross-window state,
   gradient flow, serialization, prob simplex, combiner fit/reload.
6. Profile: 171 ms/step, 375 samples/s, peak VRAM 298 MB (ECG, b=64).
   Params: trunk 80,784 | lite 11,511 | rv 25,332 | cs 15,438 | cmr 31,746
   | total 164,811 | 4×-independent est 659,244 → 75.0% saving.

## Remaining
- Let run finish (order: ECG5000_UNBAL → ECG5000_BAL → CWRU_UNBAL → CWRU_BAL).
  Expected: ~5–15 min/dataset + CWRU epochs slower (T=1024).
- Outputs: `results/turs_stack/<DS>/full_results.json`, `profile.json`,
  combiners in `checkpoints/turs_stack/`, per-epoch history in JSON + log.
- Then: assemble Part 76/77 tables, compare to historical baselines
  (TURS-Lite 0.6046/0.6377/0.8997/0.9417; MiniROCKET 0.5938/0.6553/0.9917/0.9947),
  complementarity analysis, scientific verdict (Parts 81–85).
- If interrupted: rerun `python experiments/run_turs_stack_benchmark.py`
  (resumes from checkpoints; completed datasets are skipped-fast via saved
  model state only if epoch>=MAX_EPOCHS, else retrain from scratch).

## Notes
- One accidental duplicate training process was killed (PID 5444); the
  surviving run (PID 23112) owns the checkpoints.
- CWRU npz has no official split → runner creates stratified 85/15 trainval/test
  (seed 42) then 15% val, consistent with repository handling.
