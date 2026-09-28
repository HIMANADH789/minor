"""Evidence rubric (D1-D17), added-value analysis (Phase 34), and the final
full-study report (Phase 36)."""
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


def build_rubric(per_dataset, added_value):
    """D1-D17. added_value: per-dataset Lite-vs-Stack deltas."""
    rubric = {"dimensions": {}}
    for tag, R in per_dataset.items():
        d = {}
        d["D1_predictive_reliability"] = dict(
            score_0to1=R["benchmark"]["final"]["macro_f1"],
            grade=_grade(R["benchmark"]["final"]["macro_f1"])[0])
        lat = R["diag"].get("latent", {})
        ratios = [v["probe_test_mf1"] / max(v["probe_test_mf1_raw_stats"], 1e-9)
                  for v in lat.values() if isinstance(v, dict) and "probe_test_mf1" in v]
        score = float(np.clip(np.mean([r - 1 for r in ratios]) * 2, 0, 1)) if ratios else None
        d["D2_latent_state_validity"] = dict(score_0to1=score, grade=_grade(score)[0])

        vel = R["diag"].get("velocity", {})
        rnd = vel.get("baselines", {}).get("random", {}).get("hit_at_5", 0)
        tb = [v.get("hit_at_5", 0) for v in vel.values()
              if isinstance(v, dict) and "hit_at_5" in v]
        score = float(np.clip((max(tb) - rnd) / max(1 - rnd, 1e-9), 0, 1)) if tb else None
        d["D3_temporal_change_sensitivity"] = dict(score_0to1=score,
                                                   grade=_grade(score)[0])

        unc = R["diag"].get("uncertainty", {}).get("test", {})
        aucs, cliffs = [], []
        for b, rec in unc.items():
            if isinstance(rec, dict):
                ed = rec.get("error_detection", {}).get(f"u_{b}_mean")
                if ed: aucs.append(ed["auroc"])
                if rec.get("cliffs_delta") is not None: cliffs.append(rec["cliffs_delta"])
        score = float(np.clip((np.mean(aucs) - 0.5) * 2, 0, 1)) if aucs else None
        d["D4_intrinsic_uncertainty_validity"] = dict(
            score_0to1=score, grade=_grade(score)[0], branch_aurocs=aucs,
            cliffs=cliffs)

        ens = R["diag"].get("ensemble", {}).get("test", {})
        js = ens.get("js_disagreement", {}).get("auroc")
        d["D5_ensemble_disagreement_validity"] = dict(
            score_0to1=float(np.clip((js - 0.5) * 2, 0, 1)) if js else None,
            grade=_grade(float(np.clip((js - 0.5) * 2, 0, 1)) if js else None)[0],
            js_auroc=js)

        cal = R["diag"].get("calibration", {})
        rc = R["diag"].get("risk_coverage", {})
        score = float(np.clip(1 - cal.get("ece", 1) * 2, 0, 1)) * 0.5 if cal else None
        if score is not None and "branch_disagreement_js" in rc and "final_1_conf" in rc:
            gain = (rc["final_1_conf"]["aurc"] - rc["branch_disagreement_js"]["aurc"]) / \
                   max(rc["final_1_conf"]["aurc"], 1e-9)
            score += 0.5 * float(np.clip(gain * 3 + 0.5, 0, 1))
        d["D6_calibration_selective"] = dict(score_0to1=score, grade=_grade(score)[0],
                                             ece=cal.get("ece"))

        deg = R["diag"].get("degradation", {})
        rhos = [v["rho"] for k, v in deg.items() if k.startswith("severity_vs_")
                and isinstance(v, dict) and v.get("rho") is not None
                and np.isfinite(v["rho"])]
        score = float(np.clip(np.mean(rhos), 0, 1)) if rhos else None
        d["D7_degradation_behavior"] = dict(score_0to1=score, grade=_grade(score)[0],
                                            mean_rho=float(np.mean(rhos)) if rhos else None)

        al = R["diag"].get("alpha", {}).get("test", {})
        arho = [abs(v["rho"]) for b in al.values() if isinstance(b, dict)
                for v in b.values() if isinstance(v, dict) and v.get("rho") is not None
                and np.isfinite(v["rho"])]
        score = float(np.clip(max(arho) * 2, 0, 1)) if arho else None
        d["D8_alpha_reliance_validity"] = dict(score_0to1=score, grade=_grade(score)[0])

        nb = R["diag"].get("novelty_beta", {})
        nov_auc = nb.get("novelty", {}).get("auroc_vs_error")
        d["D9_beta_scale_validity"] = dict(
            score_0to1=None, grade="NOT TESTABLE",
            note="beta is an internal gate; no independent external reference "
                 "for scale selection exists in these datasets")
        score = float(np.clip((nov_auc - 0.5) * 2, 0, 1)) if nov_auc else None
        d["D10_novelty_validity"] = dict(score_0to1=score, grade=_grade(score)[0],
                                         novelty_auroc_vs_error=nov_auc)

        f = R["diag"].get("faithfulness", {}).get("by_source", {})
        best = None
        for src, rec in f.items():
            dv = rec["pdrop"]["diff"]
            if dv is not None and (best is None or dv > best):
                best = dv
        d["D11_faithfulness"] = dict(score_0to1=float(np.clip(best * 5, 0, 1))
                                     if best is not None else None,
                                     grade=_grade(float(np.clip(best * 5, 0, 1))
                                                  if best is not None else None)[0],
                                     best_diff=best)
        cf = R["diag"].get("counterfactual", {})
        d["D12_counterfactual_consistency"] = dict(
            score_0to1=float(np.clip(cf.get("diff", 0) * 5, 0, 1))
            if cf.get("diff") is not None else None,
            grade=_grade(float(np.clip(cf.get("diff", 0) * 5, 0, 1))
                         if cf.get("diff") is not None else None)[0],
            diff=cf.get("diff"), p=cf.get("p_perm"))
        ben = R["diag"].get("benign", {})
        agrees = [v["final_agreement"] for v in ben.values() if isinstance(v, dict)]
        score = float(np.mean(agrees)) if agrees else None
        d["D13_stability"] = dict(score_0to1=score, grade=_grade(score)[0])

        bl = R["diag"].get("baseline", {})
        stack_aucs = [v["auroc"] for k, v in bl.items()
                      if k.startswith(("branch", "js", "vote"))]
        simple = [v["auroc"] for k, v in bl.items()
                  if k.startswith(("1_minus", "final", "top1"))]
        score = None
        if stack_aucs and simple:
            margin = max(stack_aucs) - max(simple)
            score = float(np.clip(0.5 + margin * 5, 0, 1)) if margin >= 0 else \
                    float(np.clip(0.5 + margin * 5, 0, 0.49))
        d["D14_baseline_superiority"] = dict(
            score_0to1=score, grade=_grade(score)[0],
            best_stack=max(stack_aucs) if stack_aucs else None,
            best_simple=max(simple) if simple else None)

        comp = R["diag"].get("complementarity", {})
        score = comp.get("complementarity_score")
        d["D15_branch_complementarity"] = dict(
            score_0to1=score, grade=_grade(score)[0],
            oracle=comp.get("oracle_any_branch_correct"),
            prob_corr=comp.get("prob_corr_mean"))

        replay_ok = R.get("replay", {}).get("all_passed", False)
        d["D16_replay_integrity"] = dict(
            score_0to1=1.0 if replay_ok else 0.0,
            grade="STRONG" if replay_ok else "NOT SUPPORTED")

        av = added_value.get(tag, {})
        mf1_delta = av.get("mf1_delta")
        score = float(np.clip(0.5 + mf1_delta * 5, 0, 1)) if mf1_delta is not None else None
        d["D17_added_value_over_turs_lite"] = dict(
            score_0to1=score, grade=_grade(score)[0],
            mf1_delta=mf1_delta, mcnemar_p=av.get("mcnemar_p"),
            auroc_u_delta=av.get("auroc_u_delta"))
        rubric["dimensions"][tag] = d
    return rubric


def added_value_analysis(fresh_results, hyp_registry):
    """Phase 34: paired Lite-vs-Stack comparison from frozen Lite checkpoints."""
    from src.diagnostics import statistics as S
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {}
    from . import benchmark as BM
    from src.diagnostics.train_lite import load_frozen, evaluate_full
    from sklearn.metrics import f1_score

    for tag, R in fresh_results.items():
        entry = {}
        lite_ckpt = os.path.join(C.LITE_CKPT_DIR, f"{tag}_TURS_Lite.pt")
        if not os.path.exists(lite_ckpt):
            out[tag] = dict(available=False,
                            note="frozen TURS-Lite checkpoint missing")
            continue
        ds = BM.load_split(tag)
        lite_model, _ = load_frozen(tag, device)
        lite_eval = evaluate_full(lite_model, ds["Xte"], ds["y_test"],
                                  ds["n_cls"], device)
        stack_mf1 = R["benchmark"]["final"]["macro_f1"]
        lite_mf1 = lite_eval["macro_f1"]
        entry.update(available=True, lite_mf1=lite_mf1, stack_mf1=stack_mf1,
                     mf1_delta=stack_mf1 - lite_mf1,
                     rel_delta=(stack_mf1 - lite_mf1) / max(lite_mf1, 1e-9))
        # paired McNemar + bootstrap CI on accuracy difference
        stack_pred = R["extracted"]["test"]["final_pred"] \
            if "extracted" in R else None
        if stack_pred is not None:
            y = np.asarray(ds["y_test"])
            sc = (stack_pred == y).astype(float)
            lc = (lite_eval["preds"] == y).astype(float)
            chi2, p = S.mcnemar(sc, lc)
            d, lo, hi = S.bootstrap_diff_ci(sc, lc, paired=True)
            entry.update(mcnemar_p=p, acc_diff=d, acc_diff_ci=[lo, hi],
                         cohens_d=S.cohens_d_paired(sc, lc))
        # uncertainty AUROC delta (lite u vs stack u_cs + JS disagreement)
        try:
            yerr = (R["extracted"]["test"]["final_correct"] == 0).astype(int)
            auc_stack = R["diag"]["uncertainty"]["test"]["cs"]["error_detection"]["u_cs_mean"]["auroc"]
            lite_ext_test = None
            lite_diag = os.path.join(C.LITE_DIAG, "diagnostics_ECG5000_UNBAL.json")
            entry["auroc_u_delta"] = auc_stack - _lite_u_auroc(tag)
        except Exception:
            entry["auroc_u_delta"] = None
        out[tag] = entry
    return out


def _lite_u_auroc(tag):
    import json
    p = os.path.join(C.LITE_DIAG, f"diagnostics_{tag}.json")
    if not os.path.exists(p):
        return float("nan")
    d = json.load(open(p))
    ed = d.get("uncertainty", {}).get("test", {}).get("error_detection", {})
    return ed.get("turs_u_mean", {}).get("auroc", float("nan"))


def write_report(rubric, added_value, per_dataset, complexity, runtime, verdicts):
    lines = []
    a = lines.append
    a("# TURS-Stack Full Benchmark + Diagnostic Validation — Final Report")
    a("")
    a(f"*Fresh canonical run completed {runtime['date']} · seed 42 · "
      f"runtime {runtime['elapsed_s']:.0f}s · zero diagnostic retraining*")
    a("")
    a("## 1. Executive Summary")
    a("")
    for tag, av in added_value.items():
        if av.get("available"):
            a(f"- **{tag}**: Stack MF1 {av['stack_mf1']:.4f} vs TURS-Lite "
              f"{av['lite_mf1']:.4f} (Δ={av['mf1_delta']:+.4f}, "
              f"McNemar p={av.get('mcnemar_p', 'n/a')})")
        else:
            a(f"- **{tag}**: Lite comparison unavailable ({av.get('note')})")
    a("")
    a("## 32. Final Conclusions (separate questions)")
    a("")
    for q, v in verdicts.items():
        a(f"- **{q}: {v}**")
    a("")
    a("## 33. Clinical-Validity Boundary")
    a("")
    a("This work evaluates machine-learning diagnostic evidence and "
      "diagnostic-support behavior only. It does NOT establish clinical "
      "diagnostic validity, physician agreement, prospective validation, "
      "hospital/external validation, patient-outcome validity, or medical-device "
      "certification.")
    a("")
    a("## 34. Limitations")
    a("")
    a("- Single seed (42) for the fresh benchmark, per canonical protocol.")
    a("- Synthetic temporal ground truth; no real event annotations.")
    a("- Ensemble disagreement is a derived reliability signal, not intrinsic u.")
    a("- alpha is sample-level; temporal alpha NOT TESTABLE.")
    a("")
    a("## 28. Complexity / Cost")
    a("")
    for tag, cx in complexity.items():
        a(f"- {tag}: Stack {cx['stack_params']:,} params "
          f"({cx.get('params_ratio', 0) and round(cx['params_ratio'], 2)}× Lite), "
          f"{cx['stack_ms_per_sample']:.2f} ms/sample vs Lite "
          f"{cx.get('lite_ms_per_sample') and round(cx['lite_ms_per_sample'], 2)}, "
          f"peak mem {cx.get('peak_mem_mb') and round(cx['peak_mem_mb'], 1)} MB")
    a("")
    a("## 27. Evidence Rubric (full details in evidence_rubric.json)")
    a("")
    for tag, dims in rubric["dimensions"].items():
        for dname, d in dims.items():
            a(f"- {tag} {dname}: **{d.get('grade')}** (score={d.get('score_0to1')})")
    a("")
    os.makedirs(os.path.dirname(C.REPORT_PATH), exist_ok=True)
    with open(C.REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return C.REPORT_PATH
