# UWaveY capacity-scaling experiment - summary

All high-capacity methods: final budget = 19992 = 2 x 9996.
Runtime: 16279s (original run) / 402s (deterministic rerun with cached null).

| method | final | Macro-F1 | alpha |
|---|---|---|---|
| MiniROCKET (canonical 9,996) | 9996 | 0.7543 | 29.764 |
| R2 flat [G||H] (canonical) | 9996 | 0.7568 | 4.281 |
| R5 rho=0.1 (canonical) | 9996 | 0.7712 | 11.288 |
| MR-HIGH | 19992 | 0.7477 | 78.476 |
| R5-HIGH | 19992 | 0.7772 | 11.288 |
| HIER-HIGH | 19992 | 0.723 | 206.914 |
