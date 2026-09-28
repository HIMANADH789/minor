"""Final TURS-RRMT-V2 report: skeptical, paper-defensible framing.

Reads the variant results + diagnostics + significance registry and writes
results/turs_rrmt_v2/TURS_RRMT_V2_REPORT.md plus the hypothesis table CSV.
"""
import json
import os

import numpy as np

from src.diagnostics import statistics as S


def _verdict(delta, q):
    if q is not None and np.isfinite(q) and q < 0.05 and delta > 0:
        return "SIGNIFICANT IMPROVEMENT"
    if q is not None and np.isfinite(q) and q < 0.05 and delta < 0:
        return "SIGNIFICANT DEGRADATION"
    if delta > 0.02:
        return "improvement (n.s.)"
    if delta < -0.02:
        return "degradation (n.s.)"
    return "no meaningful change"


def _fmt(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def write_report(results_root, all_results, v1_results, sig_rows,
                 diagnostics, elapsed_s, datasets):
    """Write TURS_RRMT_V2_REPORT.md + hypothesis_tests.csv. Returns report path."""
    rep_path = os.path.join(results_root, "TURS_RRMT_V2_REPORT.md")
    a = []
    a.append("# TURS-RRMT-V2: Readout Surgery on the Frozen V1 Representation")
    a.append("")
    a.append(f"*Runtime {elapsed_s/60:.1f} min · seed 42 · V1 A4 backbone frozen "
             f"(fixed bank M=128, J=4 flavors, top-2 routing) · canonical split/protocol*")
    a.append("")
    a.append("## 1. Question")
    a.append("")
    a.append("V1 concluded that the RRMT **representation** is strong (Ridge on A4 "
             "features beats the neural head) but the MLP head wastes it. V2 keeps "
             "the representation frozen and swaps ONLY the readout:")
    a.append("")
    a.append("- **R0** V1 MLP head (baseline) — **R1** frozen sklearn Ridge (post-hoc)")
    a.append("- **R2** differentiable ridge — **R3** class-weighted ridge")
    a.append("- **R4** soft routing (τ=2) + ridge — **R5** kernelized routed readout")
    a.append("- **R6** M=256 bank + ridge — **R7** M=512 bank + ridge")
    a.append("")
    a.append("## 2. Headline results (test Macro-F1)")
    a.append("")
    variants = ["R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7"]
    header = "| Dataset | V1 A4 (R0) | " + " | ".join(f"**{v}**" if v == "R2" else v
                                                   for v in variants[1:]) + " |"
    a.append(header)
    a.append("|---|" + "---|" * len(variants))
    best = {}
    for tag in datasets:
        vals = [all_results.get(tag, {}).get(v, {}).get("test", {}).get("macro_f1")
                for v in variants]
        finite = [(i, v) for i, v in enumerate(vals) if v is not None]
        bi = max(finite, key=lambda t: t[1])[0] if finite else None
        best[tag] = variants[bi] if bi is not None else "n/a"
        cells = []
        for i, v in enumerate(vals):
            s = _fmt(v)
            if i == bi:
                s = f"**{s}**"
            cells.append(s)
        a.append(f"| {tag} | " + " | ".join(cells) + " |")
    a.append("")
    a.append("## 3. V2 vs V1 neural head (R0) — paired McNemar with BH-FDR")
    a.append("")
    a.append("| Dataset | Comparison | Δ MF1 | McNemar p | q (FDR) | Verdict |")
    a.append("|---|---|---|---|---|---|")
    for row in sig_rows:
        if row.get("family") == "V2_vs_R0":
            a.append(f"| {row['extra'].get('dataset','')} | {row['comparison']} | "
                     f"{row['extra'].get('mf1_delta', 0):+.4f} | "
                     f"{_fmt(row['p_value'])} | {_fmt(row.get('q_value'))} | "
                     f"{row['extra'].get('verdict','')} |")
    a.append("")
    a.append("## 4. Key findings")
    a.append("")
    # Best readout per dataset + the R1 vs R0 story
    for tag in datasets:
        r0 = all_results.get(tag, {}).get("R0", {}).get("test", {}).get("macro_f1")
        r1 = all_results.get(tag, {}).get("R1", {}).get("test", {}).get("macro_f1")
        r2 = all_results.get(tag, {}).get("R2", {}).get("test", {}).get("macro_f1")
        r4 = all_results.get(tag, {}).get("R4", {}).get("test", {}).get("macro_f1")
        if None in (r0, r1, r2):
            continue
        a.append(f"- **{tag}**: best readout = **{best[tag]}**. Frozen ridge "
                 f"(R1) {r1:.3f} vs MLP head (R0) {r0:.3f}; differentiable ridge "
                 f"(R2) {r2:.3f}; soft routing (R4) {_fmt(r4)}.")
    a.append("")
    d34 = [t for t in datasets if diagnostics.get(t, {}).get("D3", {}).get("signals")]
    if d34:
        a.append("- **Uncertainty (D3)**: " + "; ".join(
            f"{t}: routing-entropy AUROC "
            f"{diagnostics[t]['D3']['signals'].get('routing_entropy', {}).get('auroc', 'n/a')} "
            f"vs confidence AUROC "
            f"{diagnostics[t]['D3']['signals'].get('confidence_neg', {}).get('auroc', 'n/a')}"
            for t in d34))
    d6 = [t for t in datasets if diagnostics.get(t, {}).get("D6")]
    if d6:
        a.append("- **Soft vs hard routing (D6)**: " + "; ".join(
            f"{t}: agreement {diagnostics[t]['D6'].get('agreement', 'n/a')}, "
            f"McNemar p={diagnostics[t]['D6'].get('mcnemar_p', 'n/a')}" for t in d6))
    a.append("")
    a.append("## 5. Diagnostic detail (D1-D6)")
    a.append("")
    for tag in datasets:
        d = diagnostics.get(tag, {})
        if not d:
            continue
        a.append(f"### {tag}")
        d1 = d.get("D1", {})
        if d1.get("variants"):
            a.append("- **D1 replay fidelity** (recomputed MF1 vs recorded): " +
                     ", ".join(f"{v}: {rec.get('test_mf1_recomputed', 0):.4f} "
                               f"(Δ {rec.get('replay_delta', 'n/a')})"
                               for v, rec in d1["variants"].items()))
        d2 = d.get("D2", {})
        if d2.get("variants"):
            worst = min(d2["variants"].items(),
                        key=lambda kv: kv[1].get("minority_mf1", 1))
            a.append(f"- **D2 error analysis**: minority classes "
                     f"{d2.get('minority_classes')}; worst minority-MF1 variant "
                     f"{worst[0]} = {worst[1].get('minority_mf1')}")
        d3 = d.get("D3", {})
        if d3.get("signals"):
            for name, rec in d3["signals"].items():
                a.append(f"- **D3 uncertainty** {name}: AUROC {rec.get('auroc')} "
                         f"CI95 {rec.get('ci95')}")
        d4 = d.get("D4", {}).get("per_flavor", {})
        if d4:
            tops = ", ".join(f"{k}: mass {v['mass']}" for k, v in d4.items())
            a.append(f"- **D4 flavor usage**: {tops}")
        d5 = d.get("D5", {})
        if d5:
            a.append(f"- **D5 lambda sensitivity**: best λ={d5.get('best_lambda')} "
                     f"val-MF1 {d5.get('best_val_mf1')}, spread "
                     f"{d5.get('sensitivity')}")
        d6 = d.get("D6", {})
        if d6:
            a.append(f"- **D6 soft-vs-hard**: agreement {d6.get('agreement')} "
                     f"over {d6.get('n_disagree')} disagreements, McNemar "
                     f"p={d6.get('mcnemar_p')}")
        a.append("")
    a.append("## 6. Interpretation discipline")
    a.append("")
    a.append("- All readouts share the SAME frozen V1 representation and the SAME "
             "splits; differences are attributable to the readout alone.")
    a.append("- McNemar tests are paired on identical test samples; BH-FDR is "
             "applied within the V2_vs_R0 family (and within each diagnostic "
             "family separately).")
    a.append("- R6/R7 reuse the seed-fixed pattern bank at larger M with an "
             "UNTRAINED router/projection — they probe bank capacity, not full "
             "model capacity at that M.")
    a.append("- This is machine-learning benchmarking, NOT clinical validation.")
    a.append("")
    a.append("## 7. Limitations")
    a.append("")
    a.append("- Single seed (42); no multi-seed variance.")
    a.append("- The differentiable-ridge path (R2/R3) fits W* in closed form on "
             "the frozen features; end-to-end joint training was NOT run in V2.")
    a.append("- R5's kernel features reduce each flavor to its time-mean — a "
             "deliberately severe compression (10 dims).")
    a.append("")
    a.append("## 8. Artifacts")
    a.append("")
    a.append("- Per-variant results: `results/turs_rrmt_v2/<DS>/<V>_results.json`")
    a.append("- Replay predictions: `results/turs_rrmt_v2/<DS>/predictions/`")
    a.append("- Comparison CSV: `results/turs_rrmt_v2/tables/model_comparison.csv`")
    a.append("- Hypothesis tests: `results/turs_rrmt_v2/tables/hypothesis_tests.csv`")
    a.append("- V1 handoff: `TURS_RRMT_HANDOFF.md`")

    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("\n".join(a) + "\n")

    # hypothesis table
    hyp_path = os.path.join(results_root, "tables", "hypothesis_tests.csv")
    S.dump_csv(sig_rows, hyp_path)
    return rep_path
