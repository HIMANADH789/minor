"""Paper-ready REPORT.md writer for the R2 Haptics 3-seed experiment."""
import os

import numpy as np

from experiments.rcmkn_r2_haptics_3seed.config import M0_REF, R2_REF_SEED42


def write_report(out_dir, seed_results, summary, all_audits, cfg):
    seeds = sorted(seed_results.keys())
    a = []
    a.append("# R2 Haptics -- 3-Seed Robustness Experiment\n")
    a.append("Replication of the finalized R2 architecture across learned-model "
             "seeds 42/43/44 under a fixed MiniROCKET bank, canonical split, "
             "per-sample z-normalization, and the canonical Ridge protocol.\n")

    a.append("## 1. Motivation\n")
    a.append("The R2 result on Haptics (test Macro-F1 0.5500 vs canonical "
             "MiniROCKET M0 0.4974, seed 42) is a single-run number. This "
             "experiment asks whether the improvement survives re-training "
             "the learned context pipeline under different random seeds.\n")

    a.append("## 2. R2 architecture (unchanged)\n")
    a.append("```")
    a.append("X_R2 = [G || H]           (9,996 features)")
    a.append("G_m  = MiniROCKET PPV_m   (4,998 global, random_state=42)")
    a.append("H_m  = sum_k q_k (PPV_{m,k} - PPV_m)^2   (4,998 het)")
    a.append("      PPV_{m,k} over hard-VQ regimes k_t in {1..8}")
    a.append("      -> RidgeClassifierCV(alphas=logspace(-4,4,20))")
    a.append("```\n")

    a.append("## 3. Why Haptics\n")
    a.append("Haptics is R2's strongest and most-cited win "
             f"(R2 {R2_REF_SEED42} vs M0 {M0_REF}); it is the claim most in "
             "need of a seed-robustness check.\n")

    a.append("## 4. Exact 3-seed protocol\n")
    a.append("- Seeds: 42, 43, 44 (outer seed of the learned context only).")
    a.append("- Dataset: Haptics, canonical split train=132 / val=23 / "
             "test=308, T=1092, 5 classes; never reshuffled.")
    a.append("- Preprocessing: per-sample z-normalization, identical in all runs.")
    a.append("- Ridge: fit on train+val, alpha via internal LOO-CV on the "
             "predeclared grid; exactly ONE official test evaluation per seed "
             "(3 total).")
    a.append("- No per-seed tuning of any kind (architecture, K, masking, "
             "feature budget, Ridge policy all fixed).\n")

    a.append("## 5. Fixed MiniROCKET protocol\n")
    a.append("- `aeon MiniRocket(random_state=42, n_jobs=-1)`, fit on the "
             "z-normed TRAIN rows only.")
    a.append("- Identical kernels, thresholds, valid regions, feature order, "
             "and selected 4,998 global features for all three seeds.")
    a.append("- Raw-activation extractor verified against the canonical aeon "
             "transform (max |diff| < 1e-5).\n")

    a.append("## 6. SSL / VQ seed variation\n")
    a.append("- SSL encoder (4 causal dilated Conv1d blocks, d=32 projection) "
             "and HardVQ (K=8, EMA, commitment 0.25, diversity 0.01) are "
             "initialized and trained independently per seed with the exact "
             "validated R2 schedule: SSL epochs<=120 (patience 20), joint "
             "fine-tune epochs<=60 (patience 10, lambda_cls=0.10).")
    a.append("- Regime assignments are deterministic given a seed's trained "
             "model (verified by re-extraction).\n")

    a.append("## 7. Ridge protocol\n")
    a.append("`RidgeClassifierCV(alphas=np.logspace(-4,4,20))` on "
             "[G || H]; identical to the validated R2 run. No SGD, no neural "
             "classifier, no new alpha policy.\n")

    a.append("## 8. Audit results\n")
    ga = all_audits["global"]
    a.append("| Audit | Result |")
    a.append("|---|---|")
    for k, v in ga.items():
        a.append(f"| {k} | pass={v.get('pass')} |")
    for s in seeds:
        fails = [k for k, v in seed_results[s]["audits"].items()
                 if isinstance(v, dict) and v.get("pass") is False]
        a.append(f"| seed{s} audits (18 checks incl. 8/9/10/11/12/13/14/15/"
                 "16/17/18) | "
                 + ("all PASS" if not fails else f"FAILED: {fails}") + " |")
    a.append("")

    a.append("## 9. Per-seed results\n")
    a.append("| Seed | Val Macro-F1 | Test Macro-F1 | Delta vs M0 | Alpha | "
             "Runtime (s) |")
    a.append("|---|---|---|---|---|---|")
    for s in seeds:
        r = seed_results[s]["results"]
        a.append(f"| {s} | {r['val_macro_f1']:.4f} | {r['test_macro_f1']:.4f} "
                 f"| {seed_results[s]['delta_vs_M0']:+.4f} | "
                 f"{r['selected_alpha']:.4f} | {seed_results[s]['runtime_s']} |")
    a.append("")

    a.append("## 10. Mean +/- std\n")
    a.append(f"- Test Macro-F1: **{summary['test_mean']:.4f} +/- "
             f"{summary['test_std']:.4f}** "
             f"(min {summary['test_min']:.4f}, max {summary['test_max']:.4f})")
    a.append(f"- Delta vs M0: {summary['delta_mean']:+.4f} +/- "
             f"{summary['delta_std']:.4f}")
    a.append(f"- Seeds improving over M0: {summary['n_improve_over_M0']}/3\n")

    a.append("## 11. Comparison with M0\n")
    a.append(f"- Canonical Haptics MiniROCKET M0 = {M0_REF} (seed 42).")
    a.append("- Per-seed deltas: "
             + ", ".join(f"seed {s}: {d:+.4f}"
                         for s, d in zip(seeds, summary['delta_vs_M0'])) + ".")
    a.append("- Note: M0 is a single fixed-MiniROCKET baseline; the R2 "
             "MiniROCKET bank is identical across seeds, so deltas reflect "
             "only the learned-context variation.\n")

    a.append("## 12. VQ diagnostics\n")
    a.append("| Seed | Active codes | Perplexity | Norm. entropy | Dominant frac. |")
    a.append("|---|---|---|---|---|")
    for s in seeds:
        v = seed_results[s]["vq_diagnostics"]
        a.append(f"| {s} | {v['active_codes']}/8 | {v['perplexity']:.3f} | "
                 f"{v['normalized_entropy']:.3f} | "
                 f"{v['dominant_fraction']:.3f} |")
    a.append("")
    a.append("Code identities are not semantically comparable across seeds; "
             "only distributional statistics are compared.\n")

    a.append("## 13. Runtime\n")
    a.append(", ".join(f"seed {s}: {seed_results[s]['runtime_s']:.0f}s"
                       for s in seeds) + "\n")

    a.append("## 14. Limitations\n")
    a.append("- n=3 seeds, single dataset, single split: descriptive "
             "statistics only; no significance testing is meaningful here.")
    a.append("- M0 is evaluated once (its MiniROCKET bank is deterministic "
             "and shared), so paired differences are per-seed scalars.")
    a.append("- The seed-42 replication gate uses the repo-established "
             "tolerance of 0.011 Macro-F1.\n")

    a.append("## 15. Final robustness conclusion\n")
    n_pos = summary["n_improve_over_M0"]
    if n_pos == 3:
        concl = ("All three seeds improve over M0: the Haptics improvement is "
                 "consistent across the tested seeds.")
    elif n_pos == 2:
        concl = ("Two of three seeds improve over M0: the improvement is "
                 "seed-sensitive (mixed robustness).")
    else:
        concl = ("At most one seed improves over M0: the R2 improvement does "
                 "not replicate robustly across seeds.")
    a.append(f"- Verdict: **{summary['verdict']}**")
    a.append(f"- {concl}")
    a.append(f"- Mean R2 = {summary['test_mean']:.4f} vs M0 = {M0_REF} "
             f"(mean delta {summary['delta_mean']:+.4f}); std across seeds "
             f"{summary['test_std']:.4f}.\n")

    path = os.path.join(out_dir, "REPORT.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(a) + "\n")
    return path
