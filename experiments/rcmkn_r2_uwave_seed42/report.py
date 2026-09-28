"""REPORT.md writer for the R2 UWave Motion generalization experiment."""
import json
import os

SHORT = {"UWaveGestureLibraryAll": "All", "UWaveGestureLibraryX": "X",
         "UWaveGestureLibraryY": "Y", "UWaveGestureLibraryZ": "Z"}


def write_report(out_dir, all_res, cross):
    a = []
    a.append("# R2 + MiniROCKET on the Local UWave Gesture Datasets (seed 42)\n")
    a.append("Single-seed Motion-domain generalization experiment: the "
             "finalized R2 architecture vs the canonical MiniROCKET baseline "
             "on UWaveGestureLibrary{All,X,Y,Z}, with identical splits and "
             "preprocessing.\n")

    a.append("## 1. Motivation\n")
    a.append("Haptics is R2's primary development dataset "
             "(R2 0.5500 vs MiniROCKET M0 0.4974, seed 42). The UWave "
             "gesture family belongs to the same broad Motion domain; the "
             "scientific question is whether the temporal-context-conditioned "
             "heterogeneity mechanism transfers to closely related "
             "gesture-recognition datasets. R2 is NOT modified based on the "
             "Haptics result.\n")

    a.append("## 2. Dataset provenance\n")
    a.append("Manually downloaded local UCR copies located next to the "
             "repository (main project directory): flat `.ts` file pairs "
             "`<NAME>_TRAIN.ts` / `<NAME>_TEST.ts` (timeseriesclassification"
             ".com / aeon .ts format, header-verified univariate, "
             "equal-length). Files verified against canonical UCR shapes at "
             "load time; originals never modified.\n")

    a.append("## 3. Dataset specifications (verified from files)\n")
    a.append("| Dataset | T | Train | Val | Test | Classes |")
    a.append("|---|---|---|---|---|---|")
    for ds, r in all_res.items():
        a.append(f"| {ds} | {r['T']} | {r['split']['train']} | "
                 f"{r['split']['val']} | {r['split']['test']} | "
                 f"{r['n_classes']} |")
    a.append("")

    a.append("## 4. Official train/test structure\n")
    a.append("The official UCR split is used as-is: 896 TRAIN / 3582 TEST. "
             "Official TEST samples never enter training, SSL, VQ, or any "
             "selection step.\n")

    a.append("## 5. Internal validation split\n")
    a.append("Stratified 15% of the official TRAIN (seed 42, sklearn "
             "`train_test_split`, indices sorted) — the established R2 "
             "transfer protocol. Saved per dataset as `train_indices.npy`, "
             "`val_indices.npy`, `test_indices.npy`; both M0 and R2 use the "
             "identical indices.\n")

    a.append("## 6. Preprocessing\n")
    a.append("Per-sample z-normalization (x - mean(x)) / std(x), applied "
             "independently per series; no dataset-level statistics, no "
             "smoothing/filtering/resampling/augmentation. Identical for M0 "
             "and R2. Native lengths preserved (All T=945; X/Y/Z T=315).\n")

    a.append("## 7-15. R2 architecture (unchanged from the validated Haptics R2)\n")
    a.append("```")
    a.append("X_R2 = [G || H]")
    a.append("G_m  = MiniROCKET PPV_m   (random_state=42, shared with M0)")
    a.append("H_m  = sum_k q_k (PPV_{m,k} - PPV_m)^2")
    a.append("      PPV_{m,k} over per-feature valid regions [p_m, T-p_m)")
    a.append("      hard-VQ regimes k_t from the SSL temporal context, K=8")
    a.append("```")
    a.append("- SSL encoder: 4 CAUSAL dilated Conv1d blocks "
             "(k=3/5/7/9, d=1/2/4/8, ch=32/32/64/64), per-position channel "
             "norm + GELU, pointwise projection 64->32; no downsampling.")
    a.append("- SSL objective: masked-span reconstruction, mask ratio 10%, "
             "span 16, masked MSE; no labels.")
    a.append("- Hard VQ: K=8, EMA codebook (non-trainable buffers), "
             "commitment 0.25, population diversity 0.01, dead-code revival; "
             "hard argmin assignment; no labels.")
    a.append("- Regimes are learned latent temporal states, NOT the 8 "
             "gesture classes; labels enter only at the Ridge classifier.")
    a.append("- H heterogeneity: audited `heterogeneity_features` "
             "(valid-region, min-count handling, renormalized q). No gating, "
             "no scaling, no raw-response modulation, no Hydra/RPMS, no "
             "kernel x regime expansion: strictly [G || H].\n")

    a.append("## 16. Ridge classifier\n")
    a.append("`RidgeClassifierCV(alphas=np.logspace(-4,4,20))` fit on "
             "train+val (the repository's canonical protocol; the "
             "eigen-solver dual formulation handles p >> n). Alpha via "
             "internal LOO-CV on train+val only.\n")

    a.append("## 17. MiniROCKET baseline (M0)\n")
    a.append("`MiniRocket(random_state=42, n_jobs=-1)`, fit on internal "
             "TRAIN only, 9,996 PPV features, same Ridge protocol, one "
             "official test evaluation. Where a stored M0 reference existed "
             "it was checked (tolerance 0.0011); otherwise the new value was "
             "saved as the canonical local reference. The same frozen "
             "extractor feeds both M0 and R2 (no refit).\n")

    a.append("## 18-19. Results\n")
    a.append("### Per-dataset (test Macro-F1, seed 42, single official pass)\n")
    a.append("| Dataset | M0 Val | R2 Val | M0 Test | R2 Test | C1 Test | "
             "C2 Test | Delta R2-M0 | Verdict |")
    a.append("|---|---|---|---|---|---|---|---|---|")
    for ds, r in all_res.items():
        rr = r["results"]
        a.append(f"| {ds} | {rr['M0']['val_macro_f1']:.4f} | "
                 f"{rr['R2']['val_macro_f1']:.4f} | "
                 f"{rr['M0']['test_macro_f1']:.4f} | "
                 f"{rr['R2']['test_macro_f1']:.4f} | "
                 f"{rr['C1']['test_macro_f1']:.4f} | "
                 f"{rr['C2']['test_macro_f1']:.4f} | "
                 f"{r['deltas']['R2-M0']:+.4f} | {r['verdict']} |")
    a.append("")
    a.append("### Cross-dataset summary\n")
    a.append("| Dataset | Length | Train | Val | Test | Classes | M0 Test | "
             "R2 Test | Delta (pp) |")
    a.append("|---|---|---|---|---|---|---|---|---|")
    for ds, c in cross["per_dataset"].items():
        a.append(f"| {ds} | {c['T']} | {c['train']} | {c['val']} | "
                 f"{c['test']} | {c['classes']} | {c['M0_test']:.4f} | "
                 f"{c['R2_test']:.4f} | {c['delta_R2_M0_pp']:+.2f} |")
    a.append("")
    for b, dss in cross["buckets"].items():
        a.append(f"- Datasets {b}: {', '.join(dss) if dss else '(none)'}")
    a.append("")

    a.append("## 20. Audit results\n")
    a.append("| Audit | " + " | ".join(SHORT[d] for d in all_res) + " |")
    a.append("|---|" + "---|" * len(all_res))
    per = {}
    for ds, r in all_res.items():
        with open(os.path.join(out_dir, ds, "audits.json")) as f:
            per[ds] = json.load(f)
    keys = sorted({k for p in per.values() for k in p})
    for k in keys:
        row = []
        for ds in all_res:
            v = per[ds].get(k)
            if isinstance(v, dict):
                row.append("PASS" if v.get("pass") else "FAIL")
            else:
                row.append(str(v) if v is not None else "-")
        a.append(f"| {k} | " + " | ".join(row) + " |")
    a.append("")

    a.append("## 21. Relation to Haptics\n")
    a.append(f"Haptics R2 seed-42 reference: {0.5500} (M0 0.4974). The "
             "UWave datasets test the same mechanism under identical "
             "protocol on related gesture data; no component was retuned "
             "for UWave.\n")

    a.append("## 22. Limitations\n")
    a.append("- Single seed (42); no variance estimate. Descriptive "
             "dataset-wise verdicts only; no significance claims.")
    a.append("- VQ code identities are not semantically comparable across "
             "datasets; only distributional statistics are compared.")
    a.append("- M0 references for UWave are first-time local canonical "
             "references (no prior stored artifact).\n")

    a.append("## 23. Final conclusion\n")
    n_imp = len(cross["buckets"]["improved"])
    n_deg = len(cross["buckets"]["degraded"])
    if n_deg == 0 and n_imp >= 2:
        concl = ("R2 improves on several UWave datasets without degrading "
                 "any, supporting transfer of the temporal-context "
                 "heterogeneity mechanism within the Motion/gesture family.")
    elif n_imp == 0 and n_deg == 0:
        concl = ("R2 is approximately unchanged relative to MiniROCKET on "
                 "all four UWave datasets: the mechanism neither helps nor "
                 "hurts here.")
    elif n_deg > 0 and n_imp == 0:
        concl = ("R2 degrades relative to MiniROCKET on the UWave family: "
                 "the mechanism does not transfer positively to these "
                 "datasets under this protocol.")
    else:
        concl = ("Results are MIXED across the UWave family: the mechanism's "
                 "benefit is dataset-dependent within the Motion domain.")
    a.append(f"- {concl}")
    a.append(f"- Datasets improved: {n_imp}/4; degraded: {n_deg}/4.")
    a.append("- Per the interpretation rule, no single-dataset positive "
             "result is claimed as generalization.\n")

    path = os.path.join(out_dir, "REPORT.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(a) + "\n")
    return path
