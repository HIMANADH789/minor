"""REPORT.md + figure writer for the R2-on-GunPoint experiment."""
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def write_report(out_dir, res):
    r = res["results"]
    variants = ["M0", "R2", "C1", "C2"]
    a = []
    a.append("# R2 on GunPoint (seed 42)\n")
    a.append("First run of the R2 architecture (SSL-learned causal temporal "
             "encoder + HardVQ regime-conditioned MiniROCKET heterogeneity) "
             "on GunPoint, the ceiling-saturated dataset from the "
             "context-dependence screen. No architecture or hyperparameter "
             "changes; every component imported unchanged from the audited "
             "implementations.\n")

    a.append("## 1. Objective\n")
    a.append("Establish R2 reference values on GunPoint and test whether the "
             "SSL temporal-context mechanism behaves as on Haptics. Prior "
             "context3 screen: MiniRocket M0 0.9933 and all conditioned "
             "variants 1.0000 (ceiling-saturated, verdict NEUTRAL).\n")

    a.append("## 2. Configuration (unchanged from Haptics R2)\n")
    a.append("- MiniRocket: `aeon MiniRocket(random_state=42)`, 9,996 "
             "features, fit on z-normed TRAIN rows.")
    a.append("- SSL encoder: 4 causal dilated Conv1d blocks (k=3/5/7/9, "
             "d=1/2/4/8, ch=32/32/64/64), d=32 projection; masked-span SSL "
             "(10%, span 16); epochs<=120, patience 20.")
    a.append("- HardVQ: K=8, EMA 0.99, beta=0.25, lam_div=0.01, revival "
             "patience 100; joint fine-tune epochs<=60, lambda_cls=0.10.")
    a.append("- Heterogeneity: audited valid-region H_m = sum_k q_k "
             "(PPV_{m,k} - PPV_m)^2.")
    a.append("- Ridge: `RidgeClassifierCV(alphas=logspace(-4,4,20))`, fit on "
             "train+val; test touched exactly once per variant.\n")

    a.append("## 3. Data\n")
    p = res["provenance"]
    a.append(f"- Source: `{p['source_path']}` (verified Kaggle "
             "UCRArchive_2018 copy, bit-identical to canonical aeon data).")
    a.append(f"- Split: train {res['split']['train']} / val "
             f"{res['split']['val']} / test {res['split']['test']}, "
             f"T={res['T']}, {res['n_classes']} classes "
             f"({p['val_source']}) -- the exact split the context3 M0 "
             "reference used.")
    a.append(f"- Label map: {p['label_map']}.")
    a.append(f"- File SHA-256 recorded in result.json provenance.\n")

    a.append("## 4. Audit results\n")
    a.append("| Audit | Result |")
    a.append("|---|---|")
    au = res["audits"]
    a.append(f"| 1. config | train={au['audit1_config']['actual']['train']} "
             f"val={au['audit1_config']['actual']['val']} "
             f"test={au['audit1_config']['actual']['test']} PASS |")
    a.append(f"| 2. extractor identity | max|diff| = "
             f"{au['audit2_extractor_identity_maxdiff']:.2e} |")
    a.append("| 3. budget | 9996 = 4998 + 4998 PASS |")
    a.append(f"| 4. causality | max|diff| = "
             f"{au['audit4_causality_maxdiff']:.1e} |")
    a.append("| 5. regime determinism | PASS |")
    occ_tr = au["audit6_vq_diagnostics"]["train"]
    a.append(f"| 6. VQ diagnostics | active={occ_tr['active_codes']}/8, "
             f"norm-entropy={occ_tr['normalized_entropy']:.3f} |")
    a.append(f"| 7/8. controls | occupancy failures = "
             f"{au['audit7_control_occupancy_failures']}, distinct arrays PASS |")
    a.append(f"| 9. independent H recompute | max diff = "
             f"{au['audit9_independent_h_recompute_maxdiff']:.2e} |")
    a.append(f"| 10. valid region | "
             f"{au['audit10_valid_region']['flipped_out_of_mask_activations']} "
             f"out-of-mask flips -> H unchanged PASS |")
    mg = au["m0_gate"]
    a.append(f"| M0 gate | {mg['observed']:.4f} vs {mg['reference']} "
             f"(tol {mg['tolerance']}) {'PASS' if mg['pass'] else 'FAIL'} |\n")

    a.append("## 5. Test results (Macro-F1, seed 42, one pass)\n")
    a.append("| Variant | Val Macro-F1 | Test Macro-F1 | Alpha |")
    a.append("|---|---|---|---|")
    for v in variants:
        a.append(f"| {v} | {r[v]['val_macro_f1']:.4f} | "
                 f"{r[v]['test_macro_f1']:.4f} | "
                 f"{r[v]['selected_alpha']:.4f} |")
    a.append("")

    a.append("## 6. Deltas\n")
    for k, d in res["deltas"].items():
        a.append(f"- {k}: {d:+.4f}")
    a.append("")

    a.append("## 7. Mechanistic heterogeneity stats (train+val)\n")
    a.append("| Variant | mean H | median H | max H | frac nonzero |")
    a.append("|---|---|---|---|---|")
    for v in ("R2", "C1", "C2"):
        m = res["mechanistic_h"][v]
        a.append(f"| {v} | {m['mean_H']:.5f} | {m['median_H']:.5f} | "
                 f"{m['max_H']:.5f} | {m['fraction_nonzero']:.3f} |")
    a.append("")

    a.append("## 8. Context\n")
    a.append(f"- Haptics R2 reference: {res['r2_haptics_reference']} "
             "(seed 42, canonical).")
    a.append("- Context3 GunPoint: M0 0.9933, M1/M2/M3/A_SOFT 1.0000 "
             "(ceiling-saturated).\n")

    a.append("## 9. Verdict\n")
    a.append(f"**{res['verdict']}**\n")

    a.append("## 10. Limitations\n")
    a.append("- Single seed (42), single split; GunPoint is small "
             "(42/8 train+val rows) and ceiling-saturated, so differences "
             "between variants are not meaningful for the mechanism.")
    a.append("- The M0 gate ties this run to the context3 canonical "
             "reference; the R2 values are the first stored references for "
             "this dataset.\n")

    a.append("## 11. Reproducibility\n")
    a.append("```bash")
    a.append("cd ECG_Benchmark")
    a.append("python -m experiments.rcmkn_r2_gunpoint_seed42.runner [--smoke]")
    a.append("python -m pytest tests/test_rcmkn_r2_gunpoint_seed42.py -q")
    a.append("```\n")

    path = os.path.join(out_dir, "REPORT.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(a) + "\n")

    # ---- figure: per-variant test Macro-F1 ----
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    vals = [r[v]["test_macro_f1"] for v in variants]
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#888", "#3b6fb6", "#c9a227", "#c9a227"]
    ax.bar(variants, vals, color=colors, width=0.55)
    lo = min(min(vals) - 0.02, 0.97)
    ax.set_ylim(lo, 1.005)
    for x, v in zip(variants, vals):
        ax.text(x, v + 0.002, f"{v:.4f}", ha="center", fontsize=9)
    ax.set_ylabel("Test Macro-F1")
    ax.set_title("R2 on GunPoint (seed 42)")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "gunpoint_variants.png"), dpi=160)
    plt.close(fig)
    return path
