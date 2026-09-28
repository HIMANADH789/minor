# HIER-HIGH-SUP — one-change ablation (UWaveY, seed 42)

Only change vs HIER-HIGH: hierarchical candidate pool (449,820) is
ranked by **supervised ANOVA-F** (top 1,999) instead of the
label-free energy allocator. Hierarchy, expanded MiniROCKET,
G_high (17,993, R5-HIGH convention), candidate definition and the
canonical Ridge protocol are locked and asserted identical.

| method | final feats | Macro-F1 |
|---|---|---|
| canonical_mr | 9,996 | 0.7543 |
| mr_high | 19,992 | 0.7477 |
| r5_high | 19,992 | 0.7772 |
| hier_high | 19,992 | 0.7230 |
| hier_high_sup | 19,992 | 0.7255 |

Deltas of HIER-HIGH-SUP: canonical_mr -0.0288, mr_high -0.0222, r5_high -0.0517, hier_high +0.0025

CV (fold-internal selection, diagnostic): 0.6971 +/- 0.0634
Selected per level (diagnostic only): {2: 1689, 4: 276, 8: 33, 16: 1}
MR-HIGH reproduction gate diff: 0.0382 (tol 0.002)
Runtime: 34s
