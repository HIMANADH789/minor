# TURS-MSW-10K — RUN HANDOFF

**Status: RUNNING (healthy)**

- **Start time:** 2026-09-14 15:41:39
- **PID:** 12720 (see `run.pid`)
- **Command:** `python experiments/turs_msw_10k/runner.py` (from `C:/temp/ECG_Benchmark`)
- **Frozen final five datasets:** ECG5000_UNBAL, ECG5000_BAL, CWRU_UNBAL, CWRU_BAL, EpilepticSeizures
  (documented derivation: `results/baseline_bench` = canonical core four; `experiments/external_stack_generalization/data.py` = ES primary, Haptics/Phoneme secondary diagnostics — excluded)
- **Current phase:** per-dataset experiment 2/5 (ECG5000_BAL, final test evaluation)
- **Datasets completed:** ECG5000_UNBAL — M0=0.5938 (exact canonical reproduction), selected=M0
- **Memory:** ~4.8 GB working set (within limits; chunked extraction active)
- **Expected remaining runtime:** ~10–15 min for the 3 remaining primaries
  (CWRU_BAL is the largest: ~3.2K train x L=1024), then locality/shift
  diagnostics, then 4x4-seed multiseed robustness (~10 min), then outputs.
- **Log file:** `results/turs_msw_10k/logs/runner.log` (also stdout.log; stderr.log currently 0 bytes = clean)
- **Results path:** `results/turs_msw_10k/`

**Inspect status:**
```bash
tail -30 ECG_Benchmark/results/turs_msw_10k/logs/runner.log
tasklist //FI "PID eq $(cat ECG_Benchmark/results/turs_msw_10k/run.pid)"
```

**Resume / re-check (safe, resumable):**
```bash
cd ECG_Benchmark && python experiments/turs_msw_10k/runner.py            # continues; cached datasets skipped
cd ECG_Benchmark && python experiments/turs_msw_10k/runner.py --report-only   # rebuild outputs from caches
```

**Gates passed so far:** equivalence (train/val/test) exactly 0.00e+00 on
ECG5000_UNBAL; all five variants at exactly 9996 features; M0 test MF1
reproduces the canonical 0.5938 reference bit-exactly.

**Warnings:** none. Unit tests 14/14 passed before launch.
