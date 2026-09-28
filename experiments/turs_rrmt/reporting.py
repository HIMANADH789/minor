"""Final TURS-RRMT report (paper-defensible, skeptical framing)."""
import json
import os

import numpy as np

from src.diagnostics import statistics as S


def _verdict(d):
    if d is None or (isinstance(d, float) and not np.isfinite(d)):
        return "NOT TESTABLE"
    if d > 0.02:
        return "ROUTING IMPROVES (A4 > A3)"
    if d < -0.02:
        return "ROUTING HURTS (A4 < A3)"
    return "NO ROUTING BENEFIT (A4 ~ A3)"


def write_report(all_results, diag_rows, report_path, elapsed_s):
    lines = []
    a = lines.append
    a("# TURS-RRMT: Regime-Routed Multi-Transport TURS — Full Experiment Report")
    a("")
    a(f"*Runtime {elapsed_s/60:.1f} min · seed 42 · fixed pattern bank M=128 "
      f"(lengths 7/11/15/23/31, 0 trainable params) · J=4 transport flavors · "
      f"top-2 preservation · canonical protocol (AdamW 3e-4/1e-2, OneCycleLR, "
      f"30 ep max, patience 8, batch 64, val macro-F1 monitor)*")
    a("")
    a("## 1. Motivation")
    a("")
    a("TURS-Lite/Stack compress local temporal structure into a small regime")
    a("latent z_t BEFORE transport participates in prediction. TURS-RRMT")
    a("inverts this: expose many local patterns first (fixed MiniROCKET-style")
    a("bank, full temporal resolution), use their local activation to ROUTE")
    a("among 4 quantile-geometry transport flavors, preserve the top-2 routed")
    a("responses, and compress only late (mean/max/std pooling).")
    a("")
    a("Central falsifiable hypothesis: **pattern-routed multi-transport (A4) >")
    a("uniform multi-transport (A3) > single transport (A2) > pattern-only (A1)**.")
    a("A3 is the critical control: identical to A4 except weights are uniform.")
    a("")
    a("## 2. Predictive results (test Macro-F1)")
    a("")
    a("| Dataset | A0 Ridge | A1 pattern | A2 1-flavor | A3 uniform | **A4 routed** | A5 no-topK | A6 shuffled | A7 fixed |")
    a("|---|---|---|---|---|---|---|---|---|")
    for tag, R in all_results.items():
        v = R["variants"]
        def g(k):
            return round(v[k]["test"]["macro_f1"], 4) if k in v else "n/a"
        a(f"| {tag} | {g('A0')} | {g('A1')} | {g('A2')} | {g('A3')} | "
          f"**{g('A4')}** | {g('A5')} | {g('A6')} | {g('A7')} |")
    a("")
    a("## 3. The architectural test: A4 vs A3 (paired, same samples)")
    a("")
    a("| Dataset | Δ Macro-F1 | Δ Acc [95% CI] | McNemar p | Δ NLL (perm p) | Cohen's d | Verdict |")
    a("|---|---|---|---|---|---|---|")
    for tag, R in all_results.items():
        d = R.get("A4_vs_A3")
        if not d:
            continue
        ci = d["acc_diff_ci95"]
        a(f"| {tag} | {d['mf1_delta']:+.4f} | {d['acc_diff']:+.4f} "
          f"[{ci[0]:+.4f}, {ci[1]:+.4f}] | {d['mcnemar_p']:.4f} | "
          f"{d['nll_mean_delta']:+.4f} (p={d['nll_paired_perm_p']:.4f}) | "
          f"{d['cohens_d']:+.3f} | {_verdict(d['mf1_delta'])} |")
    a("")
    a("Interpretation: A4> A3 consistently with CI excluding 0 => learned")
    a("routing justified. A4 ~ A3 => transport diversity helps but routing does")
    a("not; simplify. A4 < A3 => drop routing.")
    a("")
    a("## 4. Routing behavior")
    a("")
    a("| Dataset | routing entropy | top-1 flavor | route switch rate | A6 shuffle Δ |")
    a("|---|---|---|---|---|")
    for tag, R in all_results.items():
        row = None
        try:
            row = [r for r in _routing_rows() if r["dataset"] == tag]
        except Exception:
            pass
        v = R["variants"]
        a6 = v.get("A6", {}).get("test", {}).get("macro_f1")
        a4 = v.get("A4", {}).get("test", {}).get("macro_f1")
        a(f"| {tag} | (see routing_statistics.csv) | | | "
          f"{'' if a6 is None else round(a6 - a4, 4)} |")
    a("")
    a("## 5. Diagnostic validation (D1-D9, frozen A4)")
    a("")
    for tag, R in all_results.items():
        d = R.get("diagnostics", {})
        if not d:
            continue
        a(f"### {tag}")
        d2 = d.get("D2", {}).get("per_flavor", {})
        if d2:
            a("- **D2 routing validity**: strongest |rho| per flavor vs independent "
              "descriptors:")
            for fname, rec in d2.items():
                best = max(rec.items(), key=lambda kv: abs(kv[1].get("rho", 0) or 0)
                           if isinstance(kv[1], dict) else (None, 0))
                if isinstance(best[1], dict):
                    a(f"  - {fname}: {best[0]} rho={best[1].get('rho', 0):.3f} "
                      f"(p={best[1].get('p_perm')})")
        d3 = d.get("D3", {})
        if d3:
            a(f"- **D3 intervention**: shuffle flips {d3.get('flip_vs_actual', {}).get('shuffled', 0):.1%} "
              f"of predictions; McNemar p={d3.get('shuffled_vs_actual', {}).get('mcnemar_p')}")
        d4 = d.get("D4", {})
        if d4:
            a(f"- **D4 faithfulness**: targeted drop {d4['target_drop']:.4f} vs random "
              f"{d4['random_drop']:.4f} (diff {d4['diff']:+.4f}, p={d4['p_perm']:.4f})")
        d5 = d.get("D5", {})
        if d5:
            rhos = [rec[-1].get(f"severity_vs_{k}_rho") for rec in d5.get("kinds", {}).values()
                    for k in ["error_rate", "routing_entropy", "mean_conf"]
                    if rec and rec[-1].get(f"severity_vs_{k}_rho") is not None]
            if rhos:
                a(f"- **D5 degradation**: mean |severity-rho| across signals = "
                  f"{np.nanmean([abs(r) for r in rhos]):.3f}")
        d6 = d.get("D6", {})
        if d6 and d6.get("signals"):
            sig = d6["signals"]
            a("- **D6 uncertainty**: error-detection AUROC routing-entropy "
              f"{sig['routing_entropy']['auroc']:.3f} vs confidence "
              f"{sig['confidence_neg']['auroc']:.3f}")
        a("")
    a("## 6. Interpretation discipline")
    a("")
    a("- A6/A7 are inference-time interventions on frozen A4 (no retraining).")
    a("- If A6 (shuffled) ≈ A4, the learned pattern->flavor mapping is NOT")
    a("  functionally important — the model effectively uses an average flavor.")
    a("- Diagnostic correlations are associational, not causal.")
    a("- This is machine-learning diagnostic evidence, NOT clinical validation.")
    a("")
    a("## 7. Complexity")
    a("")
    for tag, R in all_results.items():
        m = R.get("meta", {}).get("A4", {})
        if m:
            a(f"- {tag} A4: {m.get('params_trainable', 0):,} trainable + "
              f"{m.get('params_fixed', 0):,} fixed kernel params, "
              f"{m.get('elapsed_s', 0)}s train, best ep {m.get('best_epoch')}")
    a("")
    a("## 8. Limitations")
    a("")
    a("- Single seed (42); no multi-seed variance reported.")
    a("- Transport flavors use fixed quantile grids; reference templates are")
    a("  TRAIN-only global means (not per-class).")
    a("- A0 Ridge uses global pooling; no tuning of Ridge alpha on validation.")
    a("- Synthetic/localization diagnostics use controlled perturbations; no")
    a("  real temporal event annotations exist in these datasets.")
    a("")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return report_path


def _routing_rows():
    # helper kept simple; the routing_statistics.csv is written by the runner
    return []
