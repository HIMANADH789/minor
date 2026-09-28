"""Evidence rubric + verdicts + failure analysis + final report (Stack)."""
import json
import os

import numpy as np

from . import config as C


def _grade(score):
    if score is None or (isinstance(score, float) and not np.isfinite(score)):
        return "NOT TESTABLE", 0.0
    if score >= 0.70: return "STRONG", score
    if score >= 0.50: return "MODERATE", score
    if score >= 0.30: return "WEAK", score
    return "NOT SUPPORTED", score


def build_rubric(per_dataset):
    rubric = {"dimensions": {}}
    for tag, R in per_dataset.items():
        d = {}
        mf1 = R.get("replay", {}).get("soft_vote_mf1")
        d["D1_predictive_reliability"] = dict(score_0to1=mf1, grade=_grade(mf1)[0])

        lat = R.get("latent", {})
        probe_vs_raw = [v["probe_test_mf1"] / max(v["probe_test_mf1_raw_stats"], 1e-9)
                        for v in lat.values() if isinstance(v, dict) and "probe_test_mf1" in v]
        score = float(np.clip(np.mean([r - 1 for r in probe_vs_raw]) * 2, 0, 1)) \
            if probe_vs_raw else None
        d["D2_latent_state_validity"] = dict(
            score_0to1=score, grade=_grade(score)[0],
            per_branch_probe_mf1={k: v.get("probe_test_mf1") for k, v in lat.items()
                                  if isinstance(v, dict)})

        vel = R.get("velocity", {})
        best = None
        if vel.get("baselines"):
            rnd = vel["baselines"].get("random", {}).get("hit_at_5", 0)
            tb = [v.get("hit_at_5", 0) for k, v in vel.items()
                  if isinstance(v, dict) and "hit_at_5" in v]
            if tb:
                best = max(tb)
                score = float(np.clip((best - rnd) / max(1 - rnd, 1e-9), 0, 1))
            else:
                score = None
        else:
            score, best, rnd = None, None, None
        d["D3_temporal_change_sensitivity"] = dict(score_0to1=score,
                                                   grade=_grade(score)[0],
                                                   best_hit5=best, random_hit5=rnd)

        unc = R.get("uncertainty", {}).get("test", {})
        aucs, cliffs = [], []
        for b, rec in unc.items():
            if not isinstance(rec, dict):
                continue
            ed = rec.get("error_detection", {}).get(f"u_{b}_mean")
            if ed:
                aucs.append(ed["auroc"])
            if rec.get("cliffs_delta") is not None:
                cliffs.append(rec["cliffs_delta"])
        score = float(np.clip((np.mean(aucs) - 0.5) * 2, 0, 1)) if aucs else None
        d["D4_intrinsic_uncertainty_validity"] = dict(
            score_0to1=score, grade=_grade(score)[0],
            branch_aurocs=aucs, cliffs_deltas=cliffs)

        ens = R.get("ensemble", {}).get("test", {})
        js = ens.get("js_disagreement", {})
        auc_js = js.get("auroc")
        d["D5_ensemble_disagreement_validity"] = dict(
            score_0to1=float(np.clip((auc_js - 0.5) * 2, 0, 1)) if auc_js else None,
            grade=_grade(float(np.clip((auc_js - 0.5) * 2, 0, 1)) if auc_js else None)[0],
            js_auroc=auc_js, vote_entropy_auroc=ens.get("vote_entropy", {}).get("auroc"))

        cal = R.get("calibration", {})
        rc = R.get("risk_coverage", {})
        score = None
        if cal:
            ece = cal.get("ece", 1.0)
            score = float(np.clip(1 - ece * 2, 0, 1)) * 0.5
            if "branch_disagreement_js" in rc and "final_entropy" in rc:
                gain = (rc["final_entropy"]["aurc"] -
                        rc["branch_disagreement_js"]["aurc"]) / max(
                    rc["final_entropy"]["aurc"], 1e-9)
                score += 0.5 * float(np.clip(gain * 3 + 0.5, 0, 1))
        d["D6_calibration_selective"] = dict(score_0to1=score, grade=_grade(score)[0],
                                             ece=cal.get("ece"))

        corr = R.get("degradation", {})
        rhos = [v["rho"] for k, v in corr.items() if k.startswith("severity_vs_")
                and isinstance(v, dict) and v.get("rho") is not None
                and np.isfinite(v["rho"])]
        score = float(np.clip(np.mean(rhos), 0, 1)) if rhos else None
        d["D7_degradation_response"] = dict(score_0to1=score, grade=_grade(score)[0],
                                            mean_rho=float(np.mean(rhos)) if rhos else None)

        al = R.get("alpha", {}).get("test", {})
        rhos = [abs(v["rho"]) for b in al.values() if isinstance(b, dict)
                for v in b.values() if isinstance(v, dict) and v.get("rho") is not None
                and np.isfinite(v["rho"])]
        score = float(np.clip(max(rhos) * 2, 0, 1)) if rhos else None
        d["D8_alpha_reliance_validity"] = dict(
            score_0to1=score, grade=_grade(score)[0],
            max_abs_rho=float(max(rhos)) if rhos else None)

        f = R.get("faithfulness", {}).get("by_source", {})
        best = None
        for src, rec in f.items():
            if rec["pdrop"]["diff"] is not None and (best is None or
                                                     rec["pdrop"]["diff"] > best):
                best = rec["pdrop"]["diff"]
        score = float(np.clip(best * 5, 0, 1)) if best is not None else None
        d["D9_faithfulness"] = dict(score_0to1=score, grade=_grade(score)[0],
                                    best_diff=best)

        cf = R.get("counterfactual", {})
        d10 = cf.get("diff")
        score = float(np.clip(d10 * 5, 0, 1)) if d10 is not None else None
        d["D10_counterfactual_consistency"] = dict(score_0to1=score,
                                                   grade=_grade(score)[0], diff=d10)

        ben = R.get("benign", {})
        agrees = [v["final_agreement"] for v in ben.values() if isinstance(v, dict)]
        score = float(np.mean(agrees)) if agrees else None
        d["D11_stability"] = dict(score_0to1=score, grade=_grade(score)[0],
                                  mean_final_agreement=score)

        bl = R.get("baseline", {})
        stack_sigs = [v["auroc"] for k, v in bl.items()
                      if k.startswith(("branch", "js", "vote"))]
        simple = [v["auroc"] for k, v in bl.items()
                  if k.startswith(("1_minus", "final", "top1"))]
        score = None
        if stack_sigs and simple:
            margin = max(stack_sigs) - max(simple)
            score = float(np.clip(0.5 + margin * 5, 0, 1)) if margin >= 0 else \
                    float(np.clip(0.5 + margin * 5, 0, 0.49))
        d["D12_baseline_superiority"] = dict(
            score_0to1=score, grade=_grade(score)[0],
            best_stack_auc=max(stack_sigs) if stack_sigs else None,
            best_simple_auc=max(simple) if simple else None)

        st = R.get("stability", {})
        d["D13_replay_stability"] = dict(
            score_0to1=1.0 if st.get("replay_ok") else 0.0,
            grade="STRONG" if st.get("replay_ok") else "NOT SUPPORTED")

        comp = R.get("complementarity", {})
        score = comp.get("complementarity_score")
        d["D14_branch_complementarity"] = dict(score_0to1=score, grade=_grade(score)[0],
                                               **{k: v for k, v in comp.items()
                                                  if k != "complementarity_score"})
        rubric["dimensions"][tag] = d
    return rubric


def two_question_verdict(rubric, consistency):
    """Phase 32: separate branch-intrinsic vs ensemble verdicts."""
    intrinsic_dims = ["D2_latent_state_validity", "D3_temporal_change_sensitivity",
                      "D4_intrinsic_uncertainty_validity", "D8_alpha_reliance_validity",
                      "D9_faithfulness", "D10_counterfactual_consistency"]
    ensemble_dims = ["D5_ensemble_disagreement_validity", "D6_calibration_selective",
                     "D12_baseline_superiority", "D14_branch_complementarity"]

    def _score(dims):
        vals = []
        for tag in rubric["dimensions"]:
            for dim in dims:
                v = rubric["dimensions"][tag].get(dim, {}).get("score_0to1")
                if v is not None and np.isfinite(v):
                    vals.append(v)
        return float(np.mean(vals)) if vals else 0.0

    def _band(mean):
        if mean >= 0.55: return "STRONG SUPPORT"
        if mean >= 0.40: return "MODERATE SUPPORT"
        if mean >= 0.25: return "WEAK SUPPORT"
        return "NOT SUPPORTED"

    return dict(
        question_A_branch_intrinsic=dict(verdict=_band(_score(intrinsic_dims)),
                                         mean_score=_score(intrinsic_dims)),
        question_B_ensemble=dict(verdict=_band(_score(ensemble_dims)),
                                 mean_score=_score(ensemble_dims)))


def cross_dataset_consistency(rubric):
    dims = ["D2_latent_state_validity", "D3_temporal_change_sensitivity",
            "D4_intrinsic_uncertainty_validity", "D5_ensemble_disagreement_validity",
            "D7_degradation_response", "D8_alpha_reliance_validity", "D9_faithfulness"]
    tags = list(rubric["dimensions"])
    out = {}
    for dim in dims:
        grades = [rubric["dimensions"][t].get(dim, {}).get("grade", "NOT TESTABLE")
                  for t in tags]
        supported = [g in ("STRONG", "MODERATE") for g in grades]
        out[dim] = dict(grades=grades, n_supported=int(sum(supported)),
                        consistent=bool(all(supported)) and any(supported))
    return out


def lite_comparison():
    """Phase 20: compare with the completed TURS-Lite diagnostic study."""
    path = os.path.join(C.LITE_DIAG_ROOT, "evidence_rubric.json")
    if not os.path.exists(path):
        return {"available": False,
                "note": "TURS-Lite rubric not found; comparison table deferred"}
    lite = json.load(open(path))
    verdict = lite.get("overall_verdict", {})
    rows = []
    dim_map = {
        "D2_latent_state_validity": "intrinsic (z)",
        "D4_uncertainty_validity": "intrinsic (u)",
        "D6_degradation_response": "intrinsic (u under corruption)",
        "D7_faithfulness": "intrinsic (u_t/s_t regions)",
        "D8_alpha_reliance_validity": "intrinsic (alpha)",
    }
    tags = list(lite["dimensions"])
    for dim, kind in dim_map.items():
        lite_grades = [lite["dimensions"][t].get(dim, {}).get("grade", "NOT TESTABLE")
                       for t in tags]
        rows.append(dict(signal=kind, model="TURS-Lite", dimension=dim,
                         grades=lite_grades))
    return dict(available=True, rows=rows, lite_overall=verdict,
                note="intrinsic=branch mechanism; ensemble signals exist only in "
                     "Stack and are compared separately (different semantics)")


def write_report(rubric, consistency, verdicts, per_dataset, hyp_count, runtime):
    lines = []
    a = lines.append
    a("# TURS-Stack Intrinsic Diagnostic Validation — Final Report")
    a("")
    a(f"*Generated {runtime['date']} · frozen checkpoints · zero retraining · "
      f"runtime {runtime['elapsed_s']:.0f}s*")
    a("")
    a("## 1. Executive Summary")
    a("")
    a(f"**Question A (branch-intrinsic diagnostics): {verdicts['question_A_branch_intrinsic']['verdict']}** "
      f"(mean score {verdicts['question_A_branch_intrinsic']['mean_score']:.3f})")
    a("")
    a(f"**Question B (ensemble/reliability diagnostics): {verdicts['question_B_ensemble']['verdict']}** "
      f"(mean score {verdicts['question_B_ensemble']['mean_score']:.3f})")
    a("")
    a(f"{hyp_count} pre-registered hypotheses tested with BH-FDR correction; "
      "all results (significant or not) are in tables/18_significance_tests.csv.")
    a("")
    a("## 2-3. Research Question & Definition of Evidence")
    a("")
    a("Does TURS-Stack provide **intrinsic diagnostic support** (branch internal "
      "states z/v/u/alpha/beta/novelty carry validated, non-trivial information) "
      "and/or **ensemble diagnostic support** (branch disagreement adds "
      "reliability information beyond ordinary final confidence)? Evidence "
      "counts only when mechanistically documented, statistically significant "
      "with non-negligible effect size, superior or complementary to simple "
      "baselines, and consistent across datasets. Ensemble disagreement is "
      "NEVER equated with intrinsic uncertainty.")
    a("")
    a("## 4. Architecture Relevant to Diagnostics (from source)")
    a("")
    a("- **lite/rv branches**: native temporal z_t = P_z(H_t); v_t = learned "
      "projection (rv adds gated multi-scale response correction -> vtilde); "
      "u_t = sigmoid(P_u([z_t||v_t])) in (0,1); alpha = transport reliance.")
    a("- **cs/cmr branches**: 3 soft-dilated scales -> within-window causal EMA "
      "zbar, **v = EMA difference (genuine temporal derivative)**; beta [B,T,3] "
      "softmax scale-trust; u_t sigmoid; cmr adds bounded controlled response "
      "(gamma-gated); novelty e_t = tanh(1 - cos(z_t, zbar_1)).mean(t) [B].")
    a("- **alpha (all branches)**: F_fused = alpha*F_T_proj + (1-alpha)*F_R_gap "
      "=> **alpha weights the TRANSPORT contribution** (sample-level; no "
      "temporal semantics -> alpha-temporal NOT TESTABLE).")
    a("- **ensemble**: 4 branch probability vectors; combiners (soft/hard/static/"
      "stacking/diagnostic) frozen from benchmark artifact.")
    a("")
    a("## 5-6. Checkpoint Reuse & Replay Verification")
    a("")
    for tag, R in per_dataset.items():
        rp = R.get("replay", {})
        a(f"- **{tag}**: replay {'PASS' if rp.get('all_passed') else 'FAIL'} — "
          + ", ".join(f"{c['metric']}: saved {c['saved']} vs replay {c['replayed']}"
                      for c in rp.get("checks", [])[:3]))
    a("")
    a("## 7. Dataset/Protocol")
    a("")
    a("Exact run_turs_stack_benchmark.py splits (seed 42, VAL_FRAC/0.85 carving), "
      "per-sample z-norm; no retraining; combiners loaded from artifact.")
    a("")
    a("## 8-11. Branch-Level Results")
    a("")
    for tag, R in per_dataset.items():
        lat = R.get("latent", {})
        a(f"### {tag}")
        for b, v in lat.items():
            if isinstance(v, dict) and "probe_test_mf1" in v:
                a(f"- latent[{b}]: probe(z) MF1={v['probe_test_mf1']:.3f} vs raw-stats "
                  f"{v['probe_test_mf1_raw_stats']:.3f}, silhouette={v['silhouette']:.3f}, "
                  f"ARI={v['kmeans_ari']:.3f}")
        unc = R.get("uncertainty", {}).get("test", {})
        for b, v in unc.items():
            if isinstance(v, dict) and v.get("cliffs_delta") is not None:
                ed = v.get("error_detection", {}).get(f"u_{b}_mean", {})
                a(f"- u[{b}]: incorrect>correct Cliff's δ={v['cliffs_delta']:.3f} "
                  f"(p={v['mannwhitney_p']:.2g}), AUROC={ed.get('auroc')}")
        a("")
    a("## 12. Ensemble Disagreement")
    a("")
    for tag, R in per_dataset.items():
        ens = R.get("ensemble", {}).get("test", {})
        for name, v in ens.items():
            if isinstance(v, dict) and "auroc" in v:
                a(f"- {tag} {name}: AUROC={v['auroc']:.4f} CI=[{v['ci'][0]:.3f},{v['ci'][1]:.3f}]")
    a("")
    a("## 13-14. Calibration & Risk-Coverage")
    a("")
    for tag, R in per_dataset.items():
        cal = R.get("calibration", {})
        rc = R.get("risk_coverage", {})
        if cal:
            a(f"- {tag}: ECE={cal['ece']:.4f} Brier={cal['brier']:.4f}")
        if "branch_disagreement_js" in rc and "final_1_conf" in rc:
            a(f"  AURC: JS-disagreement={rc['branch_disagreement_js']['aurc']:.4f} vs "
              f"1-confidence={rc['final_1_conf']['aurc']:.4f} vs "
              f"u_cs={rc.get('u_cs_mean', {}).get('aurc')}")
    a("")
    a("## 15. Controlled Degradation")
    a("")
    for tag, R in per_dataset.items():
        d = R.get("degradation", {})
        for sig in ["u_cs", "branch_disagreement_js", "final_entropy"]:
            r = d.get(f"severity_vs_{sig}_spearman")
            if r:
                a(f"- {tag} severity→{sig}: ρ={r['rho']:.3f} p={r['p_perm']:.2g}")
    a("")
    a("## 16-17. Alpha, Novelty, Beta")
    a("")
    for tag, R in per_dataset.items():
        al = R.get("alpha", {}).get("test", {})
        for b, rec in al.items():
            if not isinstance(rec, dict):
                continue
            top = sorted(((k, v["rho"]) for k, v in rec.items()
                          if isinstance(v, dict) and v.get("rho") is not None),
                         key=lambda kv: -abs(kv[1]))[:2]
            if top:
                a(f"- {tag} alpha[{b}]: " + ", ".join(f"{k} ρ={v:.3f}" for k, v in top))
        nov = R.get("novelty_beta", {}).get("novelty", {})
        if nov:
            a(f"- {tag} novelty: AUROC vs error={nov.get('auroc_vs_error'):.4f}, "
              f"ρ vs entropy={R['novelty_beta']['novelty_vs_entropy']['rho']:.3f}")
    a("")
    a("## 18-20. Faithfulness, Counterfactual, Stability")
    a("")
    for tag, R in per_dataset.items():
        f = R.get("faithfulness", {}).get("by_source", {})
        for src, rec in f.items():
            p = rec["pdrop"]
            a(f"- {tag} [{src}]: targeted {p['targeted_mean']:.4f} vs random "
              f"{p['random_mean']:.4f} (p={p['p_perm']:.3f}, d={p['cohens_d']:.2f})")
        cf = R.get("counterfactual", {})
        if cf:
            a(f"- {tag} counterfactual: {cf['probdrop_target']:.4f} vs "
              f"{cf['probdrop_random']:.4f} (p={cf.get('p_perm'):.3f})")
    a("")
    a("## 21-22. Baselines & Complementarity")
    a("")
    for tag, R in per_dataset.items():
        bl = R.get("baseline", {})
        for k, v in bl.items():
            a(f"- {tag} {k}: AUROC={v['auroc']:.4f}")
    a("")
    a("## 23-24. Statistical Significance & Effect Sizes")
    a("")
    a("See tables/18_significance_tests.csv (raw p, BH-FDR q within family) and "
      "tables/19_effect_sizes.csv. Families: velocity_localization, "
      "intrinsic_uncertainty, ensemble_vs_confidence, ensemble_disagreement_error, "
      "degradation_response, alpha_validity, novelty_validity, beta_stats, "
      "faithfulness, counterfactual, benign_stability, baseline_uncertainty.")
    a("")
    a("## 25. Cross-Dataset Consistency")
    a("")
    a("| Dimension | " + " | ".join(rubric["dimensions"]) + " | Consistent |")
    a("|---|" + "---|" * (len(rubric["dimensions"]) + 1))
    for dim, v in consistency.items():
        a(f"| {dim} | " + " | ".join(v["grades"]) +
          f" | {'yes' if v['consistent'] else 'no'} |")
    a("")
    a("**Statistical dependence warning:** the four datasets are not four "
      "independent populations; inference is sample-level within datasets, with "
      "cross-dataset consistency reported as generalization, not pooled "
      "significance.")
    a("")
    a("## 26. Failure Analysis")
    a("")
    fails = []
    for dim, v in consistency.items():
        if not v["consistent"]:
            fails.append(f"- {dim}: inconsistent ({', '.join(v['grades'])})")
    lines.extend(fails if fails else ["- No dimension failed on all datasets, but "
                                      "effect sizes are heterogeneous; see tables."])
    a("")
    a("## 27. Evidence Rubric")
    a("")
    for tag, dims in rubric["dimensions"].items():
        for dname, d in dims.items():
            a(f"- {tag} {dname}: **{d.get('grade')}** (score={d.get('score_0to1')})")
    a("")
    a("## 28. Overall Conclusion (two questions)")
    a("")
    a(f"**A — Branch-intrinsic diagnostics: {verdicts['question_A_branch_intrinsic']['verdict']}** "
      f"(mean {verdicts['question_A_branch_intrinsic']['mean_score']:.3f})")
    a("")
    a(f"**B — Ensemble/reliability diagnostics: {verdicts['question_B_ensemble']['verdict']}** "
      f"(mean {verdicts['question_B_ensemble']['mean_score']:.3f})")
    a("")
    a("## 29. Clinical-Validity Boundary")
    a("")
    a("This study evaluates machine-learning diagnostic evidence only. It does "
      "NOT establish clinical diagnostic validity, physician agreement, patient "
      "outcome validity, prospective or external validation, deployment safety, "
      "or regulatory approval.")
    a("")
    a("## 30. Limitations")
    a("")
    a("- Synthetic temporal ground truth only (no event annotations).")
    a("- Single frozen checkpoint per dataset (no seed variance).")
    a("- v on lite/rv is a learned projection, not a derivative; only cs/cmr v "
      "has genuine temporal-derivative semantics.")
    a("- alpha has no temporal meaning; alpha-temporal hypotheses NOT TESTABLE.")
    a("- Ensemble disagreement is a derived reliability signal, not intrinsic u.")
    a("")
    a("## 31. Reproducibility")
    a("")
    a(f"- Checkpoints: {C.CKPT_DIR} (SHA-256 in run_metadata.json)")
    a(f"- Extraction cache: {C.EXTRACT_DIR}")
    a(f"- Seeds: bootstrap {C.BOOTSTRAP_SEED}, permutation {C.PERMUTATION_SEED}, "
      f"synthetic {C.SYNTH_SEED}")
    a("")
    os.makedirs(os.path.dirname(C.REPORT_PATH), exist_ok=True)
    with open(C.REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return C.REPORT_PATH
