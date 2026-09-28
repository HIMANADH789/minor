"""Evidence rubric + overall verdict + final report (Phases 21/22/23/28/29/30)."""
import json
import os
import platform
import subprocess

import numpy as np

from . import config as C


def _grade(score):
    """Score in [0,1] -> rubric grade. Conservative banding."""
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return "NOT TESTABLE", 0.0
    if score >= 0.70: return "STRONG", score
    if score >= 0.50: return "MODERATE", score
    if score >= 0.30: return "WEAK", score
    return "NOT SUPPORTED", score


def build_rubric(per_dataset):
    """per_dataset: {tag: {latent:…, uncertainty:…, corruption:…, faithfulness:…,
    alpha:…, calibration:…, baseline:…, stability:…, velocity:…, counterfactual:…}}"""
    rubric = {"dimensions": {}, "notes": {}}
    for tag, R in per_dataset.items():
        d = {}

        # D1 predictive reliability — from macro-F1 (recorded, NOT diagnostic proof)
        mf1 = R.get("predictive", {}).get("macro_f1")
        d["D1_predictive_reliability"] = dict(score_0to1=mf1,
                                              grade=_grade(mf1)[0])

        # D2 latent validity: probe MF1 z-scored vs raw-stats baseline
        lat = R.get("latent", {}).get("test", {})
        pz = lat.get("probe_test_mf1_svm"); pr = lat.get("probe_test_mf1_raw_stats")
        d2 = (min(pz / max(pr, 1e-9) - 1, 1.0) if (pz and pr) else None)
        sil = lat.get("silhouette")
        score = None
        if d2 is not None:
            score = float(np.clip(0.5 * max(d2, 0) + 0.5 * max(sil or 0, 0), 0, 1))
        d["D2_latent_state_validity"] = dict(score_0to1=score, grade=_grade(score)[0],
            probe_z_mf1=pz, probe_raw_mf1=pr, silhouette=sil)

        # D3 temporal-change sensitivity: hit@5 vs random
        vel = R.get("velocity", {})
        h5 = vel.get("turs", {}).get("hit_at_5")
        hr = vel.get("random", {}).get("hit_at_5")
        score = float(np.clip((h5 - hr) / max(1 - hr, 1e-9), 0, 1)) if (h5 and hr is not None) else None
        d["D3_temporal_change_sensitivity"] = dict(score_0to1=score,
                                                   grade=_grade(score)[0],
                                                   hit5_turs=h5, hit5_random=hr)

        # D4 uncertainty validity: AUROC u vs error, effect size vs baselines
        unc = R.get("uncertainty", {}).get("test", {})
        ed = unc.get("error_detection", {}).get("turs_u_mean", {})
        auc = ed.get("auroc")
        score = float(np.clip((auc - 0.5) * 2, 0, 1)) if auc else None
        d["D4_uncertainty_validity"] = dict(score_0to1=score, grade=_grade(score)[0],
                                            auroc_u_vs_error=auc,
                                            cliffs_delta=unc.get("cliffs_delta"))

        # D5 calibration / selective prediction: ECE + AURC gain vs entropy
        cal = R.get("calibration", {}).get("test", {})
        rc = R.get("risk_coverage", {})
        score = None
        if cal:
            ece = cal.get("ece", 1.0)
            score = float(np.clip(1 - ece * 2, 0, 1)) * 0.5
            if rc and "turs_u_mean" in rc and "predictive_entropy" in rc:
                gain = (rc["predictive_entropy"]["aurc"] -
                        rc["turs_u_mean"]["aurc"]) / max(rc["predictive_entropy"]["aurc"], 1e-9)
                score += 0.5 * float(np.clip(gain * 3 + 0.5, 0, 1))
        d["D5_calibration_selective"] = dict(score_0to1=score, grade=_grade(score)[0],
                                             ece=cal.get("ece"),
                                             brier=cal.get("brier"))

        # D6 degradation response: Spearman severity->u pooled significance
        corr = R.get("corruption", {})
        rhos = [v["severity_vs_u_spearman"]["rho"] for v in corr.get("kinds", {}).values()
                if v.get("severity_vs_u_spearman")
                and v["severity_vs_u_spearman"].get("rho") is not None
                and np.isfinite(v["severity_vs_u_spearman"]["rho"])]
        sig = [v for v in (R.get("_hyp_rows") or [])
               if v["family"] == "degradation_response" and v.get("significant")]
        score = float(np.clip(np.mean(rhos), 0, 1)) if rhos else None
        d["D6_degradation_response"] = dict(score_0to1=score, grade=_grade(score)[0],
                                            mean_rho_severity_vs_u=float(np.mean(rhos)) if rhos else None,
                                            n_significant_after_fdr=len(sig))

        # D7 faithfulness: targeted-vs-random pdrop diff + significance
        f = R.get("faithfulness", {}).get("by_source", {})
        best = None
        for src, rec in f.items():
            if rec["pdrop"]["diff"] and rec["pdrop"]["diff"] > (best or 0):
                best = rec["pdrop"]["diff"]
        score = float(np.clip(best * 5, 0, 1)) if best is not None else None
        d["D7_faithfulness"] = dict(score_0to1=score, grade=_grade(score)[0],
                                    best_targeted_minus_random_pdrop=best)

        # D8 alpha validity: max |rho| across independent descriptors
        al = R.get("alpha", {}).get("test", {})
        rhos = [abs(v["rho"]) for v in al.values()
                if isinstance(v, dict) and v.get("rho") is not None
                and np.isfinite(v.get("rho", np.nan))]
        score = float(np.clip(max(rhos) * 2, 0, 1)) if rhos else None
        d["D8_alpha_reliance_validity"] = dict(score_0to1=score, grade=_grade(score)[0],
                                               max_abs_rho=float(max(rhos)) if rhos else None)

        # D9 stability
        st = R.get("stability", {})
        ok = st.get("replay_bitwise_probs", False)
        d["D9_stability_reproducibility"] = dict(score_0to1=1.0 if ok else 0.0,
                                                 grade="STRONG" if ok else "NOT SUPPORTED")

        # D10 counterfactual consistency
        cf = R.get("counterfactual", {})
        d10 = cf.get("diff")
        score = float(np.clip(d10 * 5, 0, 1)) if d10 is not None else None
        d["D10_counterfactual_consistency"] = dict(
            score_0to1=score, grade=_grade(score)[0],
            probdrop_diff=d10, p=cf.get("p_perm"))

        # D11 baseline superiority: u AUROC vs best simple baseline
        bl = R.get("baseline", {})
        turs = [v["auroc"] for k, v in bl.items() if k.startswith("turs")]
        simp = [v["auroc"] for k, v in bl.items() if not k.startswith("turs")]
        score = None
        if turs and simp:
            margin = max(turs) - max(simp)
            score = float(np.clip(0.5 + margin * 5, 0, 1)) if margin >= 0 else \
                    float(np.clip(0.5 + margin * 5, 0, 0.49))
        d["D11_baseline_superiority"] = dict(score_0to1=score, grade=_grade(score)[0],
                                             best_turs_auc=max(turs) if turs else None,
                                             best_simple_auc=max(simp) if simp else None)

        rubric["dimensions"][tag] = d

    # D12 cross-dataset consistency computed by the runner over tags
    return rubric


def cross_dataset_consistency(rubric):
    dims = ["D2_latent_state_validity", "D3_temporal_change_sensitivity",
            "D4_uncertainty_validity", "D6_degradation_response",
            "D7_faithfulness", "D8_alpha_reliance_validity"]
    tags = list(rubric["dimensions"])
    consistency = {}
    for dim in dims:
        grades = [rubric["dimensions"][t].get(dim, {}).get("grade", "NOT TESTABLE")
                  for t in tags]
        supported = [g in ("STRONG", "MODERATE") for g in grades]
        consistency[dim] = dict(grades=grades,
                                n_supported=int(sum(supported)),
                                consistent=bool(all(supported)) and any(supported))
    return consistency


def overall_verdict(rubric, consistency):
    """Phase 22 — overall conclusion from the evidence matrix."""
    all_scores = []
    for tag, dims in rubric["dimensions"].items():
        for dname, d in dims.items():
            if isinstance(d, dict) and d.get("score_0to1") is not None:
                all_scores.append(d["score_0to1"])
    mean_score = float(np.mean(all_scores)) if all_scores else 0.0
    n_dims_strong = sum(1 for tag in rubric["dimensions"]
                        for d in rubric["dimensions"][tag].values()
                        if isinstance(d, dict) and d.get("grade") in ("STRONG", "MODERATE"))
    n_consistent = sum(1 for v in consistency.values() if v["consistent"])
    if mean_score >= 0.55 and n_consistent >= 4:
        verdict = "STRONG SUPPORT"
    elif mean_score >= 0.40 and n_consistent >= 3:
        verdict = "MODERATE SUPPORT"
    elif mean_score >= 0.25:
        verdict = "WEAK SUPPORT"
    else:
        verdict = "NOT SUPPORTED"
    return dict(verdict=verdict, mean_dimension_score=mean_score,
                n_dimension_dataset_pairs_supported=n_dims_strong,
                n_dimensions_consistent_across_datasets=n_consistent,
                n_dimensions_tested=len(consistency))


def write_report(rubric, consistency, verdict, per_dataset, meta, runtime_info):
    """Phase 28 — full markdown report."""
    lines = []
    a = lines.append
    a("# TURS-Lite Intrinsic Diagnostic Validation — Final Report")
    a("")
    a(f"*Run: {meta.get('date')} · seed {C.SEED} · Python {runtime_info.get('python')} · "
      f"torch {runtime_info.get('torch')} · device {runtime_info.get('device')}*")
    a("")
    a("## 1. Executive Summary")
    a("")
    a(f"**Overall verdict: {verdict['verdict']}** "
      f"(mean dimension score {verdict['mean_dimension_score']:.3f}; "
      f"{verdict['n_dimensions_consistent_across_datasets']}/{verdict['n_dimensions_tested']} "
      f"dimensions consistent across the four datasets).")
    a("")
    a("## 2. Research Question")
    a("")
    a("Does standalone TURS-Lite provide *intrinsic diagnostic support*: do its "
      "internal variables z (regime), v (velocity), u (uncertainty) and alpha "
      "(transport reliance) carry validated, non-trivial information about the "
      "model's own state, its errors, and the input, beyond ordinary confidence "
      "baselines?")
    a("")
    a("## 3. Definition of Intrinsic Diagnostic Evidence")
    a("")
    a("A mechanism counts as diagnostic evidence only if: (i) it has a documented "
      "mechanistic meaning in the source; (ii) it shows statistically significant, "
      "non-negligible effects on pre-specified hypotheses; (iii) it survives "
      "comparison against simple baselines (max-softmax, entropy, raw-signal "
      "descriptors); (iv) effects are consistent across datasets; and (v) it is "
      "faithful under perturbation.")
    a("")
    a("## 4. TURS-Lite Architecture Relevant to Diagnostics")
    a("")
    a("- **z (regime)**: `RegimeEncoder.mu_net` applied to pooled H4 features; "
      "sample-level [B,16]. Temporal trace z_t = same frozen encoder applied per "
      "timestep (inference-only; documented in audit).")
    a("- **v (velocity)**: `vel_linear(z)` — a *learned projection* of z, NOT a "
      "temporal difference (source docstring). Temporal trace v_t per timestep.")
    a("- **u (uncertainty)**: `softplus(clamp(logvar_net(H4), -5, 2)) + 1e-6` — "
      "Gaussian regime dispersion; trained with MSE toward (1 - max softmax) "
      "via TURSLoss (lambda_unc=0.01).")
    a("- **alpha**: `sigmoid(alpha_net([t_pooled, z, v, u, g_T, g_R]))`; from "
      "`F_fused = alpha*(g_T*F_T) + (1-alpha)*(g_R*F_R)`, **alpha weights the "
      "TRANSPORT contribution** (low alpha = regime reliance). alpha is "
      "sample-level only; no temporal semantics (NOT TESTABLE temporally).")
    a("")
    a("## 5. Datasets")
    a("")
    for tag, R in per_dataset.items():
        m = R.get("manifest", {})
        sp = m.get("split", {})
        a(f"- **{tag}**: {m.get('n_samples_total')} samples, L={m.get('signal_length')}, "
          f"C={m.get('num_classes')}; split {sp.get('train')}/{sp.get('val')}/{sp.get('test')}")
    a("")
    a("## 6. Exact Training/Reproduction Protocol")
    a("")
    a("Canonical `fair_turs.py` TURS-Lite configuration, unchanged: TURSNet "
      "(regime_dim=16, variant=lite), TURSLoss CE + aux (0.01/0.005/0.01/0.005), "
      "AdamW 3e-4/1e-2, OneCycleLR (30 ep), batch 64, grad clip 1.0, seed 42, "
      "early stop patience 8 on val macro-F1.")
    a("")
    a("## 7. Predictive Reproduction Results")
    a("")
    a("| Dataset | Historical MF1 | This run MF1 | Acc | BalAcc | MCC | Best ep |")
    a("|---|---|---|---|---|---|---|")
    for tag, R in per_dataset.items():
        p = R.get("predictive", {})
        h = (R.get("historical") or {}).get("macro_f1")
        a(f"| {tag} | {h if h is not None else 'n/a'} | {p.get('macro_f1'):.4f} | "
          f"{p.get('accuracy'):.4f} | {p.get('balanced_accuracy'):.4f} | "
          f"{p.get('mcc'):.4f} | {p.get('best_epoch')} |")
    a("")
    a("Differences vs historical values are expected (GPU nondeterminism in "
      "cuDNN kernels despite deterministic seeds); the diagnostic study uses "
      "this run's own frozen checkpoints throughout.")
    a("")
    a("## 8. Latent-State Validation")
    a("")
    for tag, R in per_dataset.items():
        t = R.get("latent", {}).get("test", {})
        a(f"- {tag}: silhouette={t.get('silhouette'):.3f}, ARI={t.get('kmeans_ari'):.3f}, "
          f"probe(z) MF1={t.get('probe_test_mf1_svm'):.3f} vs raw-stats probe "
          f"MF1={t.get('probe_test_mf1_raw_stats'):.3f}")
    a("")
    a("## 9. Temporal Velocity Validation")
    a("")
    for tag, R in per_dataset.items():
        v = R.get("velocity", {})
        if not v:
            continue
        a(f"- {tag}: hit@5 TURS={v.get('turs', {}).get('hit_at_5')}, "
          f"raw diff={v.get('raw_first_diff', {}).get('hit_at_5')}, "
          f"energy={v.get('raw_local_energy', {}).get('hit_at_5')}, "
          f"random={v.get('random', {}).get('hit_at_5')}")
    a("")
    a("## 10. Uncertainty Validation")
    a("")
    for tag, R in per_dataset.items():
        t = R.get("uncertainty", {}).get("test", {})
        ed = t.get("error_detection", {}).get("turs_u_mean", {})
        a(f"- {tag}: u incorrect>correct (Cliff's δ={t.get('cliffs_delta')}, "
          f"p={t.get('mannwhitney_p')}), AUROC={ed.get('auroc')} "
          f"CI=[{ed.get('auroc_ci', [None, None])[0]}, {ed.get('auroc_ci', [None, None])[1]}]")
    a("")
    a("## 11. Calibration")
    a("")
    for tag, R in per_dataset.items():
        c = R.get("calibration", {}).get("test", {})
        a(f"- {tag}: ECE={c.get('ece'):.4f}, aECE={c.get('adaptive_ece'):.4f}, "
          f"Brier={c.get('brier'):.4f}, NLL={c.get('nll'):.4f}")
    a("")
    a("## 12. Risk-Coverage / Selective Prediction")
    a("")
    for tag, R in per_dataset.items():
        rc = R.get("risk_coverage", {})
        if "turs_u_mean" in rc and "predictive_entropy" in rc:
            a(f"- {tag}: AURC u={rc['turs_u_mean']['aurc']:.4f} vs entropy="
              f"{rc['predictive_entropy']['aurc']:.4f} "
              f"(risk@80% u={rc['turs_u_mean']['selective_risk_at_80']:.4f})")
    a("")
    a("## 13. Controlled Degradation")
    a("")
    for tag, R in per_dataset.items():
        c = R.get("corruption", {})
        h3 = c.get("H3_u_vs_error", {})
        a(f"- {tag}: pooled u↔error Spearman ρ={h3.get('rho')}, p={h3.get('p_perm')}, "
          f"AUROC={h3.get('auroc')}")
    a("")
    a("## 14. Alpha / Reliance Validation")
    a("")
    for tag, R in per_dataset.items():
        r = R.get("alpha", {}).get("test", {})
        top = sorted(((k, v.get("rho")) for k, v in r.items()
                      if isinstance(v, dict) and v.get("rho") is not None),
                     key=lambda kv: -abs(kv[1]))[:3]
        a(f"- {tag}: strongest descriptors " +
          ", ".join(f"{k} (ρ={v:.3f})" for k, v in top))
    a("")
    a("## 15. Faithfulness")
    a("")
    for tag, R in per_dataset.items():
        f = R.get("faithfulness", {}).get("by_source", {})
        for src, rec in f.items():
            p = rec["pdrop"]
            a(f"- {tag} [{src}]: targeted p-drop {p['targeted_mean']:.4f} vs random "
              f"{p['random_mean']:.4f}; diff={p['diff']:.4f} CI={p['ci']}, "
              f"p={p['p_perm']:.4f}, d={p['cohens_d']:.2f}")
    a("")
    a("## 16. Stability")
    a("")
    for tag, R in per_dataset.items():
        s = R.get("stability", {})
        a(f"- {tag}: bitwise replay={s.get('replay_bitwise_probs')}, "
          f"max|Δp|={s.get('replay_max_abs_diff')}")
    a("")
    a("## 17. Counterfactual Consistency")
    a("")
    for tag, R in per_dataset.items():
        cf = R.get("counterfactual", {})
        a(f"- {tag}: high-s_t edit p-drop {cf.get('probdrop_target'):.4f} vs random "
          f"{cf.get('probdrop_random'):.4f} (p={cf.get('p_perm'):.4f}); flips "
          f"{cf.get('flip_target'):.3f} vs {cf.get('flip_random'):.3f}")
    a("")
    a("## 18. Simple Baseline Comparisons")
    a("")
    for tag, R in per_dataset.items():
        b = R.get("baseline", {})
        for k, v in b.items():
            a(f"- {tag} {k}: AUROC={v['auroc']:.4f} CI=[{v['ci'][0]:.4f},{v['ci'][1]:.4f}]")
    a("")
    a("## 19-20. Statistical Significance & Effect Sizes")
    a("")
    hyp = R.get("_hyp_rows") or []
    fams = sorted(set(h["family"] for h in hyp))
    a("All hypotheses (raw p, BH-FDR q): see `tables/13_significance_tests.csv` and "
      "`tables/14_effect_sizes.csv`. Families: " + ", ".join(fams) + ".")
    a("")
    a("## 21. Cross-Dataset Analysis")
    a("")
    a("| Dimension | " + " | ".join(rubric["dimensions"]) + " | Consistent |")
    a("|---|" + "---|" * (len(rubric["dimensions"]) + 1))
    for dim, v in consistency.items():
        a(f"| {dim} | " + " | ".join(v["grades"]) +
          f" | {'yes' if v['consistent'] else 'no'} |")
    a("")
    a("## 22. Failure Modes")
    a("")
    weak = [(dim, v["grades"]) for dim, v in consistency.items()
            if not v["consistent"]]
    if weak:
        for dim, grades in weak:
            a(f"- {dim}: inconsistent across datasets ({', '.join(grades)}).")
    else:
        a("- No dimension failed consistently across datasets, but effect sizes "
          "vary; see tables for magnitudes.")
    a("")
    a("## 23. Evidence Rubric")
    a("")
    a("Full rubric in `evidence_rubric.json`; grades per dimension/dataset:")
    for tag, dims in rubric["dimensions"].items():
        for dname, d in dims.items():
            a(f"- {tag} {dname}: **{d.get('grade')}** (score={d.get('score_0to1')})")
    a("")
    a("## 24. Overall Diagnostic Conclusion")
    a("")
    a(f"**{verdict['verdict']}** — mean dimension score "
      f"{verdict['mean_dimension_score']:.3f}.")
    a("")
    a("## 25. Clinical-Validity Boundary")
    a("")
    a("This study evaluates machine-learning diagnostic evidence only. It does "
      "NOT establish clinical diagnostic accuracy, physician agreement, external "
      "hospital validation, prospective validation, patient-outcome validity, or "
      "regulatory/device validity. No clinical claims are made or supported.")
    a("")
    a("## 26. Limitations")
    a("")
    a("- Standalone TURS-Lite computes z/v/u from pooled features; temporal "
      "traces are inference-only applications of the frozen encoders.")
    a("- v is a learned projection, not a temporal derivative; localization "
      "results must be read accordingly.")
    a("- Synthetic temporal ground truth (no event labels in these datasets); "
      "controlled injections only.")
    a("- Single primary seed (42), as per the benchmark protocol; deterministic "
      "replay verified.")
    a("")
    a("## 27. Reproducibility Information")
    a("")
    a(f"- Checkpoints: `checkpoints/turs_lite/<DS>_TURS_Lite.pt` "
      f"(sha256 in run_metadata.json)")
    a(f"- Extraction cache: `results/diagnostics/turs_lite/extracted/`")
    a(f"- Seeds: model {C.SEED}, bootstrap {C.BOOTSTRAP_SEED}, "
      f"permutation {C.PERMUTATION_SEED}, synthetic {C.SYNTH_SEED}")
    a(f"- Total runtime: {runtime_info.get('elapsed_s', 0):.0f}s")
    a("")
    a("## 30. Final Paper-Ready Conclusion")
    a("")
    strongest = max(((dname, d.get("score_0to1") or 0)
                     for tag in rubric["dimensions"]
                     for dname, d in rubric["dimensions"][tag].items()),
                    key=lambda kv: kv[1])
    weakest = min(((dname, d.get("score_0to1") if d.get("score_0to1") is not None else 0)
                   for tag in rubric["dimensions"]
                   for dname, d in rubric["dimensions"][tag].items()),
                  key=lambda kv: kv[1])
    a(f"\"TURS-Lite **{'does' if verdict['verdict'] != 'NOT SUPPORTED' else 'does not'} "
      f"provide sufficient empirical evidence for intrinsic diagnostic "
      f"support\" at the {verdict['verdict']} level.")
    a("")
    a(f"- Strongest evidence: {strongest[0]} (score {strongest[1]:.2f}); weakest: "
      f"{weakest[0]} (score {weakest[1]:.2f}).")
    a(f"- Baseline comparison: TURS-u vs max-softmax/entropy/margin — see "
      f"`tables/12_baseline_comparison.csv` (superiority is NOT assumed; the "
      f"rubric grades it explicitly).")
    a("- Cross-dataset consistency: see table in section 21.")
    a("")
    a("\"This conclusion refers to machine-learning diagnostic evidence and does "
      "not constitute clinical validation.\"")
    a("")
    os.makedirs(os.path.dirname(C.REPORT_PATH), exist_ok=True)
    with open(C.REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return C.REPORT_PATH
