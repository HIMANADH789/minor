# HIER-CONTINUOUS-SUP — unified-pool controlled ablation (UWaveY, seed 42)

**One change vs HIER-HIGH-SUP**: the protected G/H boundary is removed.

| | HIER-HIGH-SUP (current) | HIER-CONTINUOUS-SUP (new) |
|---|---|---|
| Candidate pool | H only: 449,820 delta features | **C = [G_full ‖ Delta_full]: 19,992 + 449,820 = 469,812** |
| G | 17,993 protected (rho=0.1 slice of the 19,992 root matrix) | **not protected — roots are level-0 candidates** |
| Selection | `top_f_select(H, y_dev, 1,999)` | **`top_f_select(C, y_dev, 19,992)`** |
| Final dim | 17,993 + 1,999 = 19,992 | **19,992 (single ranking)** |
| rho / N_G / N_H / quota | fixed | **none — composition emerges from the ranking** |

Everything else is locked and identical: canonical split (761/135/3,582),
per-sample z-norm, expanded MiniROCKET (`MiniRocket(random_state=42,
n_kernels=19,992)` → 84 kernels × 238 quantiles), frozen seed-42 SSL
encoder + recursive 2-means hierarchy (K=1→2→4→8→16, train-only fit),
449,820-candidate telescoping delta pool (30 edges × 14,994 kernels,
occupancy-weighted parent/child PPVs), canonical Ridge protocol, seed 42.

The root PPV matrix is level-0 of the same hierarchy: PPV_m^(1) = G_m, and
hierarchical features are the telescoping residuals
Δ_{m,c}^{(ℓ)} = PPV_{m,c} − PPV_{m,parent(c)} for ℓ ∈ {2,4,8,16}. The
unified pool is therefore the complete multiresolution candidate space of
one representation.

## Interpretation map (§16)

- **Improves over HIER-HIGH-SUP** → the fixed G/H boundary was limiting.
- **Approximately equal** → the boundary was not materially limiting.
- **Decreases** → unrestricted competition adds selection noise/redundancy;
  the protected composition is more stable at n=896.

No moderated-F, no variance shrinkage, no occupancy penalties, no level
corrections, no multiple-testing corrections — those would add a second
experimental variable. `top_f_select` is reused as the same object.

## Run

```bash
python experiments/uwavey_hier_continuous_sup/runner.py            # production
python experiments/uwavey_hier_continuous_sup/runner.py --smoke    # subset e2e
pytest tests/test_uwavey_hier_continuous_sup.py -q                 # focused tests
```

Prior experiments (Canonical MR / MR-HIGH / R5-HIGH / HIER-HIGH /
HIER-HIGH-SUP) are never rerun or modified; their saved results are loaded
read-only for the comparison table. All outputs land under
`results/uwavey_hier_continuous_sup/seed42/`.
