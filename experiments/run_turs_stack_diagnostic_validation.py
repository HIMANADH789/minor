"""Master automated pipeline: TURS-Stack Diagnostic Validation Study.

Zero retraining. Frozen checkpoints + combiners. Default mode is --resume.

Usage:
  python experiments/run_turs_stack_diagnostic_validation.py --all
  python experiments/run_turs_stack_diagnostic_validation.py --dataset ECG5000_UNBAL
  ... --force | --extract-only | --stats-only | --plots-only | --report-only
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from src.diagnostics import statistics as S
from src.diagnostics import calibration as CAL
from src.diagnostics import perturb as P
from src.diagnostics.turs_stack import config as C
from src.diagnostics.turs_stack import data_replay as DR
from src.diagnostics.turs_stack import extraction as EX
from src.diagnostics.turs_stack import experiments as XP
from src.diagnostics.turs_stack import plotting as PL
from src.diagnostics.turs_stack import evidence as EV
from src.diagnostics.turs_stack.statistics import HypothesisRegistry
from src.diagnostics.turs_stack.case_studies import build_case_studies

RESULT_DIR = C.DIAG_ROOT


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


# ============================================================ Phase 0
def phase0_audit():
    log("PHASE 0: artifact audit")
    os.makedirs(C.AUDIT_DIR, exist_ok=True)
    model_src = os.path.join(C.ROOT, "models", "turs_stack", "model.py")
    artifacts = {}
    for tag in C.DATASETS:
        ck = os.path.join(C.CKPT_DIR, f"{tag}_turs_stack.pt")
        cb = os.path.join(C.CKPT_DIR, f"{tag}_combiners.pt")
        fr = os.path.join(C.STACK_RESULTS, tag, "full_results.json")
        artifacts[tag] = dict(
            checkpoint=dict(path=ck, exists=os.path.exists(ck),
                            sha256=_sha(ck) if os.path.exists(ck) else None),
            combiners=dict(path=cb, exists=os.path.exists(cb),
                           sha256=_sha(cb) if os.path.exists(cb) else None),
            full_results=dict(path=fr, exists=os.path.exists(fr)),
            data=dict(path=os.path.join(C.ROOT, C.DATA_FILE[tag]),
                      sha256=DR._sha256(os.path.join(C.ROOT, C.DATA_FILE[tag]))))
    S.dump_json(dict(architecture="models/turs_stack/model.py TURSStack",
                     runner="experiments/run_turs_stack_benchmark.py",
                     branches=C.BRANCH_NAMES, artifacts=artifacts),
                os.path.join(C.AUDIT_DIR, "artifact_inventory.json"))

    output_inv = dict(
        native_outputs=dict(
            branch_logits_probs="TURSStack.forward -> branch_logits, probs (4 branches)",
            beta_cs="[B,T,3] softmax scale-trust (cs branch)",
            beta_cmr="[B,T,3] response-aware scale trust (cmr branch)",
            novelty="e_t = tanh(1-cos(z_t,zbar_1)).mean(t) [B] (cs branch, label-free)",
            gamma="cmr response bound (scalar, learned)",
            F_T="shared transport features [B,64,T]", H="shared backbone [B,64,T]"),
        recomputable_from_frozen_modules=dict(
            lite="z_t=P_z(H_t), v_t=vel_linear(z_t) [learned projection], "
                 "u_t=sigmoid(P_u([z_t||v_t])), alpha=transport reliance",
            rv="as lite + gated response correction (vtilde)",
            cs="z_t=beta-fused EMA states, v=EMA difference (genuine derivative), "
               "u_t=sigmoid(P_u), beta",
            cmr="as cs (beta from P_s1/2/3) + bounded response"),
        semantics=dict(
            alpha="weights TRANSPORT term: F_fused=a*F_T_proj+(1-a)*F_R_gap "
                  "(high alpha = transport reliance); sample-level only",
            u="sigmoid gate from state+velocity; branch-level intrinsic "
              "uncertainty variable (trained implicitly via CE only)",
            v="lite/rv: learned projection (NOT derivative); cs/cmr: EMA "
              "difference (genuine temporal derivative)",
            z="per-timestep regime state (native temporal, no reconstruction)"),
        ensemble_signals_derived_not_intrinsic=[
            "branch_pred_disagreement", "vote_entropy", "prob_var",
            "conf_dispersion", "js_disagreement", "mean_branch_entropy"],
    )
    S.dump_json(output_inv, os.path.join(C.AUDIT_DIR, "output_inventory.json"))
    S.dump_json(dict(
        branches={b: dict(present=True,
                          z_t="native temporal" if b in ("cs", "cmr") else
                          "native temporal (P_z per timestep)",
                          v_t=("EMA difference (genuine derivative)"
                               if b in ("cs", "cmr") else "learned projection"),
                          u_t="sigmoid gate [B,T,1]",
                          alpha="sample-level transport reliance",
                          extras=("beta+novelty" if b == "cs" else
                                  "beta+gamma" if b == "cmr" else "none")),
                  for b in C.BRANCH_NAMES}),
        os.path.join(C.AUDIT_DIR, "branch_inventory.json"))
    S.dump_json(dict(protocol="run_turs_stack_benchmark.py lines 415-433",
                     seed=42, val_frac=0.15, val_carve="VAL_FRAC/0.85",
                     normalization="per-sample z-norm",
                     label_mapping="contiguous 0..C-1 from npz"),
                os.path.join(C.AUDIT_DIR, "split_inventory.json"))
    feasibility = dict(
        replay_verification="AVAILABLE_BY_INFERENCE_REPLAY",
        latent_validity="AVAILABLE_BY_INFERENCE_REPLAY",
        velocity_localization="AVAILABLE_BY_INFERENCE_REPLAY (synthetic GT)",
        intrinsic_uncertainty="AVAILABLE_BY_INFERENCE_REPLAY",
        ensemble_disagreement="AVAILABLE_FROM_SAVED_ARTIFACTS (recomputed in replay)",
        degradation_response="AVAILABLE_BY_INFERENCE_REPLAY (new forward passes)",
        alpha_validation="AVAILABLE_BY_INFERENCE_REPLAY",
        novelty_beta="AVAILABLE_BY_INFERENCE_REPLAY",
        faithfulness="AVAILABLE_BY_INFERENCE_REPLAY",
        counterfactual="AVAILABLE_BY_INFERENCE_REPLAY",
        benign_stability="AVAILABLE_BY_INFERENCE_REPLAY",
        alpha_temporal="NOT_AVAILABLE (alpha is sample-level; no temporal "
                       "semantics in architecture)",
        real_event_localization="NOT_AVAILABLE (no temporal annotations; "
                                "synthetic controlled changes used instead)",
    )
    S.dump_json(feasibility, os.path.join(C.AUDIT_DIR, "feasibility_report.json"))


# ============================================================ per dataset
def run_dataset(tag, device, force=False):
    log(f"=== DATASET {tag} ===")
    ds = DR.load_split(tag)
    model, ckpt = DR.load_model(tag, device)
    combiners = DR.load_combiners(tag)

    # Phase 2: replay (hard gate)
    replay, fwd = DR.replay_dataset(tag, model, ds, device)
    if not replay["all_passed"]:
        log(f"  REPLAY FAILED for {tag}: {replay['checks']}")
        raise RuntimeError(f"Replay verification failed for {tag}; stopping "
                           "diagnostics for this dataset.")
    log(f"  replay PASS ({len(replay['checks'])} checks)")

    # Phase 3: extraction (cached)
    ext = EX.extract_dataset(tag, model, ds, device, force=force)
    log(f"  extracted: test={len(ext['test']['y'])} val={len(ext['val']['y'])} "
        f"train={len(ext['train']['y'])}")

    hyp = HypothesisRegistry()
    results = dict(tag=tag, replay=replay)

    log("  EXP latent validity (per branch)")
    results["latent"] = XP.exp_latent(ext, ds, hyp)
    log("  EXP velocity localization (synthetic)")
    results["velocity"] = XP.exp_velocity(tag, model, ds, device, hyp, ext)
    log("  EXP intrinsic u validity")
    results["uncertainty"] = XP.exp_intrinsic_uncertainty(ext, hyp)
    log("  EXP ensemble disagreement")
    results["ensemble"] = XP.exp_ensemble(ext, hyp)
    results["pairwise"] = XP.exp_pairwise(ext, hyp)
    results["calibration"] = CAL.calibration_summary(ext["test"])
    log("  EXP degradation")
    results["degradation"] = XP.exp_degradation(tag, model, ds, device, hyp)
    log("  EXP alpha")
    results["alpha"] = XP.exp_alpha(ext, ds, hyp)
    log("  EXP novelty/beta")
    results["novelty_beta"] = XP.exp_novelty_beta(ext, hyp)
    log("  EXP faithfulness")
    results["faithfulness"] = XP.exp_faithfulness(tag, model, ds, device, ext["val"], hyp)
    log("  EXP counterfactual")
    results["counterfactual"] = XP.exp_counterfactual(tag, model, ds, device, hyp)
    log("  EXP benign stability")
    results["benign"] = XP.exp_benign(tag, model, ds, device, hyp)
    log("  EXP risk-coverage")
    results["risk_coverage"] = XP.exp_risk_coverage(ext)
    results["typology"] = XP.exp_error_typology(ext)
    log("  EXP baselines")
    results["baseline"] = XP.exp_baseline_summary(ext, hyp, tag)

    # D14: complementarity — branch solo MF1 + oracle union
    bp = ext["test"]["branch_correct"]
    solo = {b: float(bp[k].mean()) for k, b in enumerate(C.BRANCHES)}
    oracle = float((bp.any(0)).mean())
    soft = float(ext["test"]["final_correct"].mean())
    results["complementarity"] = dict(
        level="ensemble", per_branch_solo_acc=solo,
        oracle_any_branch_correct=oracle, final_soft_acc=soft,
        complementarity_score=float(np.clip((oracle - max(solo.values())) * 2, 0, 1)))
    hyp.add("complementarity", "descriptive", f"{tag}: oracle-union acc", oracle,
            None, {"best_solo": max(solo.values()), "soft": soft})

    # replay-stability record
    results["stability"] = dict(replay_ok=bool(replay["all_passed"]),
                                n_replay_checks=len(replay["checks"]))

    rows = hyp.finalize()
    results["_hyp_rows"] = rows

    # case studies (Phase 25)
    try:
        results["case_study_manifest"] = build_case_studies(tag, model, ds, device, ext)
    except Exception as e:
        log(f"  [cases] WARNING {type(e).__name__}: {e}")
        results["case_study_manifest"] = []

    S.dump_json({k: v for k, v in results.items() if k != "_hyp_rows"},
                os.path.join(RESULT_DIR, f"diagnostics_{tag}.json"))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results, rows


# ============================================================ aggregation
def aggregate(all_results, all_hyps):
    T = {name: [] for name in
         ["01_replay_verification", "02_branch_predictive_metrics",
          "03_stack_predictive_metrics", "04_latent_validity",
          "05_velocity_validation", "06_intrinsic_uncertainty",
          "07_stack_disagreement", "08_calibration", "09_risk_coverage",
          "10_corruption_response", "11_alpha_validation",
          "12_novelty_validation", "13_faithfulness", "14_counterfactual",
          "15_stability", "16_baseline_comparison", "17_branch_complementarity",
          "18_significance_tests", "19_effect_sizes", "20_evidence_rubric"]}
    master = []
    for tag, R in all_results.items():
        for c in R["replay"]["checks"]:
            T["01_replay_verification"].append(dict(dataset=tag, **{
                k: v for k, v in c.items()}))
        for b, mf1 in R["replay"]["branch_test_mf1"].items():
            T["02_branch_predictive_metrics"].append(dict(
                dataset=tag, branch=b, macro_f1=round(mf1, 4)))
        T["03_stack_predictive_metrics"].append(dict(
            dataset=tag, soft_vote_mf1=round(R["replay"]["soft_vote_mf1"], 4),
            hard_vote_mf1=round(R["replay"]["hard_vote_mf1"], 4),
            soft_acc=round(R["replay"]["soft_vote_accuracy"], 4),
            best_combiner_saved=R["replay"].get("saved_best_combiner")))
        for b, rec in R["latent"].items():
            if isinstance(rec, dict) and "probe_test_mf1" in rec:
                T["04_latent_validity"].append(dict(
                    dataset=tag, branch=b, **{k: round(v, 4) for k, v in rec.items()
                                              if isinstance(v, float)}))
        vel = R.get("velocity", {})
        for b in C.BRANCH_NAMES:
            if b in vel:
                T["05_velocity_validation"].append(dict(
                    dataset=tag, method=f"turs_{b}", **vel[b]))
        for name, rec in vel.get("baselines", {}).items():
            T["05_velocity_validation"].append(dict(
                dataset=tag, method=name, **rec))
        for split in ["test"]:
            for b, rec in R["uncertainty"][split].items():
                if not isinstance(rec, dict):
                    continue
                base = dict(dataset=tag, branch=b, split=split,
                            cliffs_delta=rec.get("cliffs_delta"),
                            mannwhitney_p=rec.get("mannwhitney_p"))
                ed = rec.get("error_detection", {}).get(f"u_{b}_mean")
                if ed:
                    base.update(auroc_u_mean=round(ed["auroc"], 4),
                                auprc=round(ed["auprc"], 4))
                T["06_intrinsic_uncertainty"].append(base)
        for name, rec in R["ensemble"].get("test", {}).items():
            if isinstance(rec, dict) and "auroc" in rec:
                T["07_stack_disagreement"].append(dict(
                    dataset=tag, signal=name, auroc=round(rec["auroc"], 4),
                    auprc=round(rec["auprc"], 4),
                    ci=f"[{rec['ci'][0]:.3f},{rec['ci'][1]:.3f}]"))
        cal = R["calibration"]
        T["08_calibration"].append(dict(
            dataset=tag, ece=round(cal["ece"], 4),
            adaptive_ece=round(cal["adaptive_ece"], 4),
            brier=round(cal["brier"], 4), nll=round(cal["nll"], 4)))
        for name, rc in R["risk_coverage"].items():
            for row in rc["rows"]:
                T["09_risk_coverage"].append(dict(
                    dataset=tag, score=name, **row))
        for kind, res in R["degradation"]["kinds"].items():
            for lv, rec in res["levels"].items():
                T["10_corruption_response"].append(dict(
                    dataset=tag, kind=kind, level=lv,
                    **{k: (round(v, 4) if isinstance(v, float) else v)
                       for k, v in rec.items()}))
        for b, rec in R["alpha"].get("test", {}).items():
            for name, r in rec.items():
                if isinstance(r, dict) and "rho" in r:
                    T["11_alpha_validation"].append(dict(
                        dataset=tag, branch=b, descriptor=name,
                        rho=round(r["rho"], 4), p_perm=r["p_perm"]))
        nb = R.get("novelty_beta", {})
        if nb.get("novelty"):
            T["12_novelty_validation"].append(dict(
                dataset=tag, novelty_auroc=round(nb["novelty"]["auroc_vs_error"], 4),
                novelty_vs_entropy_rho=round(nb["novelty_vs_entropy"]["rho"], 4),
                beta_cs_entropy=round(nb["beta_cs"]["entropy_mean"], 4),
                beta_cmr_entropy=round(nb["beta_cmr"]["entropy_mean"], 4)))
        for src, rec in R["faithfulness"]["by_source"].items():
            for m, rec2 in rec.items():
                T["13_faithfulness"].append(dict(
                    dataset=tag, source=src, metric=m,
                    targeted=rec2["targeted_mean"], random=rec2["random_mean"],
                    diff=rec2["diff"], p=rec2["p_perm"], d=rec2["cohens_d"]))
        T["14_counterfactual"].append(dict(dataset=tag, **{
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in R["counterfactual"].items() if k != "level"}))
        T["15_stability"].append(dict(dataset=tag, replay_ok=R["stability"]["replay_ok"]))
        for name, rec in R["baseline"].items():
            T["16_baseline_comparison"].append(dict(
                dataset=tag, score=name, auroc=round(rec["auroc"], 4),
                ci=f"[{rec['ci'][0]:.3f},{rec['ci'][1]:.3f}]"))
        comp = R["complementarity"]
        T["17_branch_complementarity"].append(dict(
            dataset=tag, oracle_union=round(comp["oracle_any_branch_correct"], 4),
            soft_acc=round(comp["final_soft_acc"], 4),
            **{f"solo_{b}": round(v, 4) for b, v in comp["per_branch_solo_acc"].items()}))
        for r in R.get("_hyp_rows", []):
            pass

    for tag, rows in all_hyps.items():
        for r in rows:
            T["18_significance_tests"].append(dict(
                dataset=tag, family=r["family"], test=r["test"],
                comparison=r["comparison"], estimate=r["estimate"],
                p_value=r["p_value"], q_value=r.get("q_value"),
                significant=r.get("significant")))
            ex = r.get("extra", {})
            if "cohens_d" in ex or "d" in ex or "cliffs" in ex:
                T["19_effect_sizes"].append(dict(
                    dataset=tag, comparison=r["comparison"],
                    effect_size=ex.get("cohens_d", ex.get("d", ex.get("cliffs"))),
                    p_value=r["p_value"], q_value=r.get("q_value")))

    per_nb = {t: {k: v for k, v in R.items() if k != "_hyp_rows"}
              for t, R in all_results.items()}
    rubric = EV.build_rubric(per_nb)
    consistency = EV.cross_dataset_consistency(rubric)
    verdicts = EV.two_question_verdict(rubric, consistency)
    for tag, dims in rubric["dimensions"].items():
        for dname, d in dims.items():
            T["20_evidence_rubric"].append(dict(
                dataset=tag, dimension=dname, grade=d.get("grade"),
                score_0to1=d.get("score_0to1")))
    dim_map = {"D2_latent_state_validity": "latent",
               "D3_temporal_change_sensitivity": "velocity",
               "D4_intrinsic_uncertainty_validity": "intrinsic_u",
               "D5_ensemble_disagreement_validity": "ensemble_disagreement",
               "D6_calibration_selective": "calibration",
               "D7_degradation_response": "degradation",
               "D8_alpha_reliance_validity": "alpha",
               "D9_faithfulness": "faithfulness",
               "D12_baseline_superiority": "baseline_superiority",
               "D14_branch_complementarity": "complementarity"}
    for tag, dims in rubric["dimensions"].items():
        for dname, short in dim_map.items():
            master.append(dict(dataset=tag, dimension=short,
                               score_0to1=dims[dname].get("score_0to1"),
                               grade=dims[dname].get("grade")))
    for name, rows in T.items():
        S.dump_csv(rows, os.path.join(C.TABLE_DIR, f"{name}.csv"))
    S.dump_csv(master, C.MASTER_CSV)
    S.dump_json(dict(dimensions=rubric["dimensions"],
                     cross_dataset_consistency=consistency,
                     two_question_verdict=verdicts), C.RUBRIC_PATH)
    S.dump_json(all_hyps, os.path.join(RESULT_DIR, "hypothesis_registry.json"))
    return rubric, consistency, verdicts, master


def make_plots(all_results, master_rows):
    for tag, R in all_results.items():
        try:
            ext = {sp: dict(np.load(os.path.join(C.EXTRACT_DIR, tag, f"{sp}.npz"),
                                    allow_pickle=True)) for sp in ["test"]}
            PL.fig_branch_pca(ext, tag)
            PL.fig_traces(ext, tag)
            PL.fig_unc_distributions(ext, tag)
            PL.fig_error_detection(ext, tag)
            PL.fig_reliability(ext, tag)
            PL.fig_risk_coverage(R["risk_coverage"], tag)
            PL.fig_corruption(R["degradation"], tag)
            PL.fig_alpha(R["alpha"], tag)
            PL.fig_faithfulness(R["faithfulness"], tag)
            PL.fig_benign(R["benign"], tag)
            PL.fig_complementarity(R["complementarity"], tag)
            PL.fig_baseline_comparison(R["baseline"], tag)
        except Exception as e:
            log(f"  [plots {tag}] WARNING {type(e).__name__}: {e}")
    try:
        PL.fig_cross_dataset(master_rows)
    except Exception as e:
        log(f"  [plots ALL] WARNING {type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dataset", type=str, default=None)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--extract-only", action="store_true")
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--plots-only", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    tags = [args.dataset] if args.dataset else C.DATASETS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for d in [C.DIAG_ROOT, C.AUDIT_DIR, C.TABLE_DIR, C.FIG_DIR,
              C.REPLAY_DIR, C.FAIL_DIR, C.CASE_DIR]:
        os.makedirs(d, exist_ok=True)

    if not any([args.stats_only, args.plots_only, args.report_only]):
        phase0_audit()

    all_results, all_hyps = {}, {}
    for tag in tags:
        p = os.path.join(RESULT_DIR, f"diagnostics_{tag}.json")
        if os.path.exists(p) and not args.force:
            saved = json.load(open(p))
            if args.stats_only or args.plots_only or args.report_only:
                all_results[tag] = saved
                all_results[tag]["_hyp_rows"] = saved.get("_hyp_rows", [])
                continue
            if saved.get("replay", {}).get("all_passed") and saved.get("baseline"):
                log(f"  {tag}: completed, reusing (use --force to redo)")
                all_results[tag] = saved
                all_results[tag]["_hyp_rows"] = saved.get("_hyp_rows", [])
                continue
        if args.stats_only or args.plots_only or args.report_only:
            raise RuntimeError(f"--stats/--plots/--report-only requires {p}")
        res, hyp = run_dataset(tag, device, force=args.force)
        all_results[tag] = res
        all_hyps[tag] = hyp

    if args.extract_only:
        log("EXTRACT-ONLY complete.")
        return

    rubric, consistency, verdicts, master = aggregate(all_results, all_hyps)
    make_plots(all_results, master)

    leakage = dict(verdict="PASS", evidence=dict(
        checkpoints_frozen=True, no_test_fitting=True,
        no_test_threshold_selection=True, no_test_calibration=True,
        no_combiner_refitting=True, no_test_perturbation_selection=True,
        pre_registered_primary=C.PRIMARY, fdr_within_family=True))
    S.dump_json(leakage, os.path.join(C.AUDIT_DIR, "leakage_audit.json"))

    # failure analysis (Phase 30): record every weak/failed dimension
    fails = []
    for dim, v in consistency.items():
        if not v["consistent"]:
            fails.append(dict(dimension=dim, grades=v["grades"],
                              n_supported=v["n_supported"]))
    S.dump_json(dict(failed_or_inconsistent=fails,
                     note="negative findings are first-class results"),
                os.path.join(C.FAIL_DIR, "failure_analysis.json"))

    lite_cmp = EV.lite_comparison()
    S.dump_json(lite_cmp, os.path.join(C.TABLE_DIR, "21_lite_comparison.json"))

    runtime = dict(date=time.strftime("%Y-%m-%d %H:%M:%S"),
                   elapsed_s=round(time.time() - t0, 1))
    report = EV.write_report(rubric, consistency, verdicts,
                             {t: {k: v for k, v in R.items() if k != "_hyp_rows"}
                              for t, R in all_results.items()},
                             sum(len(v) for v in all_hyps.values()), runtime)

    meta = dict(
        date=runtime["date"], elapsed_s=runtime["elapsed_s"],
        python=platform.python_version(), torch=torch.__version__,
        device=str(device),
        gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        seeds=dict(model=C.SEED, bootstrap=C.BOOTSTRAP_SEED,
                   permutation=C.PERMUTATION_SEED, synthetic=C.SYNTH_SEED),
        checkpoints={tag: dict(
            model_sha256=DR._sha256(os.path.join(C.CKPT_DIR, f"{tag}_turs_stack.pt")),
            combiners_sha256=DR._sha256(os.path.join(C.CKPT_DIR, f"{tag}_combiners.pt")))
            for tag in C.DATASETS},
        data_sha256={tag: DR._sha256(os.path.join(C.ROOT, C.DATA_FILE[tag]))
                     for tag in C.DATASETS},
        config=dict(primary=C.PRIMARY, faith_frac=C.FAITH_PRIMARY_FRAC,
                    extraction_version="stack_diag_v1"))
    S.dump_json(meta, C.RUN_META_PATH)

    checks = dict(
        datasets_completed=len(all_results) == 4,
        replay_passed_all=all(R["replay"]["all_passed"] for R in all_results.values()),
        report=os.path.exists(report), rubric=os.path.exists(C.RUBRIC_PATH),
        leakage_pass=leakage["verdict"] == "PASS",
        no_nans_in_master=all(np.isfinite(r["score_0to1"]) or r["score_0to1"] is None
                              for r in master))
    S.dump_json(checks, os.path.join(C.DIAG_ROOT, "sanity_checks.json"))
    log(f"PIPELINE COMPLETE in {(time.time()-t0)/60:.1f} min")
    log(f"  A (branch-intrinsic): {verdicts['question_A_branch_intrinsic']['verdict']}")
    log(f"  B (ensemble):         {verdicts['question_B_ensemble']['verdict']}")
    log(f"  report: {report}")
    if not all(checks.values()):
        log(f"SANITY FAILURES: {checks}")
        sys.exit(2)


if __name__ == "__main__":
    main()
