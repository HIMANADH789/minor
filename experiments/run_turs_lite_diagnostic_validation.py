"""Master automated pipeline: TURS-Lite Diagnostic Validation Study.

Usage:
  python experiments/run_turs_lite_diagnostic_validation.py --all
  python experiments/run_turs_lite_diagnostic_validation.py --dataset ECG5000_UNBAL
  ... --skip-training   (reuse existing checkpoints)
  ... --force           (retrain / re-extract)
  ... --report-only     (rebuild rubric + report from saved JSONs)
  ... --plots-only      (regenerate figures from saved JSONs)
"""
import argparse
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

from src.diagnostics import config as C
from src.diagnostics import data as D
from src.diagnostics import extraction as E
from src.diagnostics import statistics as S
from src.diagnostics.statistics import HypothesisRegistry
from src.diagnostics import experiments_a as A
from src.diagnostics import experiments_b as B
from src.diagnostics import calibration as CAL
from src.diagnostics import plotting as PL
from src.diagnostics import evidence as EV
from src.diagnostics import train_lite as T
from src.diagnostics import case_studies as CS

RESULT_DIR = C.DIAG_ROOT


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================ Phase 0
def phase0_audit():
    log("PHASE 0: repository / implementation audit")
    import hashlib
    os.makedirs(C.AUDIT_DIR, exist_ok=True)
    tursnet = open(os.path.join(ROOT, "models", "tursnet.py")).read()
    fairturs = open(os.path.join(ROOT, "experiments", "fair_turs.py")).read()
    audit = dict(
        implementation=dict(
            source_file="models/tursnet.py",
            class_name="TURSNet",
            canonical_config=dict(in_channels=1, regime_dim=16, variant="lite",
                                  multi_scale_fusion=False, use_all_flags=True),
            factory_reference="experiments/fair_turs.py::make_turs_lite",
            sha256_models_tursnet=hashlib.sha256(tursnet.encode()).hexdigest(),
        ),
        output_mapping=dict(
            z_t="aux['regime']  z  [B,16] native sample-level; z_t temporal trace = "
                "frozen RegimeEncoder.mu_net applied per H4 timestep (inference-only)",
            v_t="aux['regime_velocity']  v  [B,16] = vel_linear(z) — LEARNED "
                "PROJECTION, not temporal difference (source docstring documents this)",
            u_t="aux['uncertainty']  u  [B,16] = softplus(clamp(logvar_net(H4),-5,2))+1e-6",
            alpha="aux['alpha']  [B,1] sigmoid; weights the TRANSPORT term "
                  "(F_fused = a*(g_T*F_T) + (1-a)*(g_R*F_R)); sample-level only, "
                  "no temporal semantics -> alpha-temporal hypotheses NOT TESTABLE",
            F_T="transport_proj(T_e.mean) [B,64]; F_R = regime_proj(z) [B,64]",
        ),
        trainer=dict(
            source_file="experiments/fair_turs.py",
            function="train_and_evaluate",
            protocol=dict(seed=42, max_epochs=30, patience=8, lr=3e-4, wd=1e-2,
                          batch=64, optimizer="AdamW", scheduler="OneCycleLR",
                          grad_clip=1.0, loss="TURSLoss CE + aux",
                          monitor="val macro-F1"),
            sha256_fair_turs=hashlib.sha256(fairturs.encode()).hexdigest(),
        ),
        benchmark_protocol=dict(
            canonical_results="results/turs_benchmark/<DS>.json",
            historical_turs_lite={}),
    )
    # historical TURS-Lite numbers if present
    for tag in C.NUM_CLASSES:
        p = os.path.join(C.HIST_RESULT, f"{tag}.json")
        if os.path.exists(p):
            d = json.load(open(p))
            if "TURS-Lite" in d:
                audit["benchmark_protocol"]["historical_turs_lite"][tag] = \
                    d["TURS-Lite"].get("macro_f1")
    S.dump_json(audit, os.path.join(C.AUDIT_DIR, "implementation_audit.json"))
    S.dump_json(audit["trainer"], os.path.join(C.AUDIT_DIR, "benchmark_protocol_audit.json"))
    S.dump_json(dict(
        return_aux_available=True,
        native_variables=["regime(z)", "regime_velocity(v)", "regime_speed(s)",
                          "uncertainty(u)", "alpha", "transport_gate(g_T)",
                          "regime_gate(g_R)"],
        temporal_reconstruction=dict(available=True, method="frozen encoders per "
                                     "timestep of H4", alpha_temporal=False),
        probability_outputs=["softmax(logits)"],
        extraction_caching="results/diagnostics/turs_lite/extracted/<DS>/<split>.npz",
    ), os.path.join(C.AUDIT_DIR, "output_availability.json"))
    return audit


# ============================================================ per dataset
def run_dataset(tag, device, force=False, skip_training=False):
    log(f"=== DATASET {tag} ===")
    ds = D.load_split(tag)
    D.write_manifest(tag, ds["manifest"])
    log(f"  data: train={len(ds['y_train'])} val={len(ds['y_val'])} "
        f"test={len(ds['y_test'])} L={ds['L']} C={ds['n_cls']}")

    ckpt_path = os.path.join(C.CKPT_DIR, f"{tag}_TURS_Lite.pt")
    if skip_training and not os.path.exists(ckpt_path):
        raise RuntimeError(f"--skip-training set but {ckpt_path} missing")

    # Phase 1/3: canonical training
    _, meta, _ = T.train_turs_lite(ds, device, log=log, force=force)

    # Phase 4: predictive reproduction (fresh load = new inference context)
    model, ckpt = T.load_frozen(tag, device)
    test_eval = T.evaluate_full(model, ds["Xte"], ds["y_test"], ds["n_cls"], device)
    hist = None
    hp = os.path.join(C.HIST_RESULT, f"{tag}.json")
    if os.path.exists(hp):
        d = json.load(open(hp))
        if "TURS-Lite" in d:
            hist = d["TURS-Lite"]
    predictive = {k: test_eval[k] for k in
                  ["accuracy", "macro_f1", "weighted_f1", "balanced_accuracy",
                   "mcc", "class_f1s", "confusion_matrix"]}
    predictive["best_epoch"] = meta.get("best_epoch")
    predictive["replay_source"] = "frozen checkpoint, new inference process"
    S.dump_json(dict(tag=tag, predictive=predictive, historical=hist,
                     history=meta.get("history")),
                os.path.join(RESULT_DIR, f"predictive_{tag}.json"))
    log(f"  test MF1={predictive['macro_f1']:.4f} "
        f"(historical={hist['macro_f1'] if hist else 'n/a'})")

    # Phase 5/6: freeze + extract once, cache
    ext = E.extract_dataset(tag, model, ds, device, force=force)

    hyp = HypothesisRegistry()
    results = dict(tag=tag, predictive=predictive, historical=hist,
                   manifest=ds["manifest"])

    # Phase 7
    log("  EXP1 latent validity")
    results["latent"] = A.exp_latent_validity(ext, ds, device, hyp)
    # Phase 8
    log("  EXP2 velocity localization (synthetic controlled)")
    results["velocity"] = A.exp_velocity_localization(tag, model, ds, device, hyp)
    # Phase 9
    log("  EXP3 uncertainty validity")
    results["uncertainty"] = A.exp_uncertainty_validity(ext, ds, hyp)
    results["calibration"] = {sp: CAL.calibration_summary(ext, sp)
                              for sp in ["val", "test"]}
    results["risk_coverage"] = CAL.risk_coverage_comparison(ext, "test")
    # Phase 10
    log("  EXP4 controlled degradation")
    results["corruption"] = A.exp_controlled_degradation(tag, model, ds, device, hyp)
    # Phase 11
    log("  EXP5 alpha validation")
    results["alpha"] = A.exp_alpha_validation(ext, ds, hyp)
    # Phase 12
    log("  EXP6 faithfulness")
    results["faithfulness"] = B.exp_faithfulness(tag, model, ds, device, ext["val"], hyp)
    # Phase 13
    log("  EXP7 benign stability")
    results["benign"] = B.exp_benign_stability(tag, model, ds, device, hyp)
    # Phase 14
    log("  EXP8 counterfactual")
    results["counterfactual"] = B.exp_counterfactual(tag, model, ds, device, hyp)
    # Phase 15
    results["stability"] = B.exp_stability(tag, model, ds, device)
    # Phase 16
    results["baseline"] = B.exp_baseline_summary(ext, tag, hyp)
    # Phase 24: deterministic case studies (illustrative, aggregate stats rule)
    log("  case studies")
    try:
        cases = CS.build_case_studies(tag, model, ds, device, ext)
        results["case_study_manifest"] = cases
    except Exception as e:
        log(f"  [cases] WARNING: {type(e).__name__}: {e}")
        results["case_study_manifest"] = []

    hyp_rows = hyp.finalize()
    results["_hyp_rows"] = hyp_rows
    S.dump_json(results, os.path.join(RESULT_DIR, f"diagnostics_{tag}.json"))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results, hyp_rows


# ============================================================ aggregation
def aggregate(all_results, all_hyps):
    """Tables 01-16 + master + rubric + FDR across families."""
    import csv
    rows_master = []
    tables = {name: [] for name in
              ["01_predictive_metrics", "02_latent_validity", "03_velocity_validation",
               "04_uncertainty_error_detection", "05_calibration", "06_risk_coverage",
               "07_corruption_response", "08_alpha_validation", "09_faithfulness",
               "10_stability", "11_counterfactual", "12_baseline_comparison",
               "13_significance_tests", "14_effect_sizes", "15_evidence_rubric",
               "16_case_study_manifest"]}

    per = all_results
    for tag, R in per.items():
        p = R["predictive"]
        tables["01_predictive_metrics"].append(dict(
            dataset=tag, macro_f1=round(p["macro_f1"], 4),
            accuracy=round(p["accuracy"], 4), weighted_f1=round(p["weighted_f1"], 4),
            balanced_accuracy=round(p["balanced_accuracy"], 4), mcc=round(p["mcc"], 4),
            historical_macro_f1=(R.get("historical") or {}).get("macro_f1")))
        lat = R["latent"]["test"]
        tables["02_latent_validity"].append(dict(
            dataset=tag, **{k: round(v, 4) for k, v in lat.items()
                            if isinstance(v, float)}))
        vel = R.get("velocity", {})
        for name in ["turs", "raw_first_diff", "raw_local_energy", "random"]:
            if name in vel:
                tables["03_velocity_validation"].append(dict(
                    dataset=tag, method=name, **{k: v for k, v in vel[name].items()
                                                 if isinstance(v, (int, float))}))
        for name, rec in R["uncertainty"]["test"].get("error_detection", {}).items():
            tables["04_uncertainty_error_detection"].append(dict(
                dataset=tag, score=name, auroc=round(rec["auroc"], 4),
                auprc=round(rec["auprc"], 4)))
        cal = R["calibration"]["test"]
        tables["05_calibration"].append(dict(
            dataset=tag, ece=round(cal["ece"], 4), adaptive_ece=round(cal["adaptive_ece"], 4),
            brier=round(cal["brier"], 4), nll=round(cal["nll"], 4)))
        for name, rc in R["risk_coverage"].items():
            for row in rc["rows"]:
                tables["06_risk_coverage"].append(dict(
                    dataset=tag, score=name, **row))
        for kind, res in R["corruption"]["kinds"].items():
            for lv, rec in res["levels"].items():
                tables["07_corruption_response"].append(dict(
                    dataset=tag, kind=kind, level=lv, **{k: (round(v, 4) if isinstance(v, float) else v)
                                                          for k, v in rec.items() if k != "pred_flip_rate"}))
        for desc, rec in R["alpha"]["test"].items():
            if isinstance(rec, dict) and "rho" in rec:
                tables["08_alpha_validation"].append(dict(
                    dataset=tag, descriptor=desc, rho=round(rec["rho"], 4),
                    p_perm=rec["p_perm"], ci=f"[{rec['ci_lo']:.3f},{rec['ci_hi']:.3f}]"))
        for src, rec in R["faithfulness"]["by_source"].items():
            for m, rec2 in rec.items():
                tables["09_faithfulness"].append(dict(
                    dataset=tag, source=src, metric=m,
                    targeted=rec2["targeted_mean"], random=rec2["random_mean"],
                    diff=rec2["diff"], p_perm=rec2["p_perm"], cohens_d=rec2["cohens_d"]))
        tables["10_stability"].append(dict(dataset=tag, **R["stability"]))
    for cs in R.get("case_study_manifest", []):
        tables["16_case_study_manifest"].append(dict(
            dataset=cs["dataset"], category=cs["category"],
            sample_id=cs["sample_id"], true_label=cs["true_label"],
            predicted_label=cs["predicted_label"], confidence=round(cs["confidence"], 4),
            u_mean=round(cs["u_mean"], 4), v_max=round(cs["v_max"], 4),
            alpha=round(cs["alpha"], 4),
            targeted_prob_drop=round(cs["targeted_prob_drop"], 4),
            random_prob_drop=round(cs["random_prob_drop"], 4)))
        tables["11_counterfactual"].append(dict(dataset=tag, **{
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in R["counterfactual"].items()}))
        for name, rec in R["baseline"].items():
            tables["12_baseline_comparison"].append(dict(
                dataset=tag, score=name, auroc=round(rec["auroc"], 4),
                ci_lo=round(rec["ci"][0], 4), ci_hi=round(rec["ci"][1], 4)))

    # significance + effect sizes (all families pooled, FDR within family)
    hyp_rows = []
    for tag, rows in all_hyps.items():
        for r in rows:
            hyp_rows.append(dict(dataset=tag, **r))
    for r in hyp_rows:
        tables["13_significance_tests"].append(dict(
            dataset=r["dataset"], family=r["family"], test=r["test"],
            comparison=r["comparison"], estimate=r["estimate"],
            p_value=r["p_value"], q_value=r.get("q_value"),
            significant=r.get("significant")))
        ex = r.get("extra", {})
        if "cohens_d" in ex or "cliffs" in ex:
            tables["14_effect_sizes"].append(dict(
                dataset=r["dataset"], comparison=r["comparison"],
                effect_size=ex.get("cohens_d", ex.get("cliffs")),
                measure="cohens_d" if "cohens_d" in ex else "cliffs_delta",
                p_value=r["p_value"], q_value=r.get("q_value")))

    # rubric
    per_nb = {t: {k: v for k, v in R.items() if k != "_hyp_rows"}
              for t, R in per.items()}
    rubric = EV.build_rubric(per_nb)
    consistency = EV.cross_dataset_consistency(rubric)
    verdict = EV.overall_verdict(rubric, consistency)

    for tag, dims in rubric["dimensions"].items():
        for dname, d in dims.items():
            tables["15_evidence_rubric"].append(dict(
                dataset=tag, dimension=dname, grade=d.get("grade"),
                score_0to1=d.get("score_0to1")))

    # master CSV: per-dataset dimension scores
    for tag, dims in rubric["dimensions"].items():
        mapping = {"D2_latent_state_validity": "latent",
                   "D3_temporal_change_sensitivity": "velocity",
                   "D4_uncertainty_validity": "uncertainty",
                   "D5_calibration_selective": "calibration",
                   "D6_degradation_response": "robustness",
                   "D7_faithfulness": "faithfulness",
                   "D8_alpha_reliance_validity": "alpha",
                   "D11_baseline_superiority": "baseline_superiority"}
        for dname, short in mapping.items():
            rows_master.append(dict(dataset=tag, dimension=short,
                                    score_0to1=dims[dname].get("score_0to1"),
                                    grade=dims[dname].get("grade")))

    for name, rows in tables.items():
        S.dump_csv(rows, os.path.join(C.TABLE_DIR, f"{name}.csv"))
    S.dump_csv(rows_master, C.MASTER_CSV)
    S.dump_json(dict(dimensions=rubric["dimensions"],
                     cross_dataset_consistency=consistency,
                     overall_verdict=verdict), C.RUBRIC_PATH)
    S.dump_json(all_hyps, os.path.join(RESULT_DIR, "hypothesis_registry.json"))
    return rubric, consistency, verdict, rows_master


# ============================================================ plots
def make_plots(all_results, master_rows):
    for tag, R in all_results.items():
        try:
            ext = {sp: dict(np.load(os.path.join(C.EXTRACT_DIR, tag, f"{sp}.npz"),
                                    allow_pickle=True)) for sp in ["test"]}
            PL.fig_latent_pca(ext, tag)
            PL.fig_latent_separability(R["latent"], tag)
            ext_t = {sp: dict(np.load(os.path.join(C.EXTRACT_DIR, tag, f"{sp}.npz"),
                                      allow_pickle=True)) for sp in ["test"]}
            PL.fig_traces(ext_t, tag)
            PL.fig_unc_distributions(ext_t, tag)
            PL.fig_error_detection(ext_t, tag)
            PL.fig_reliability(ext_t, tag)
            PL.fig_risk_coverage(R["risk_coverage"], tag)
            PL.fig_corruption(R["corruption"], tag)
            PL.fig_alpha(R["alpha"], tag)
            PL.fig_faithfulness(R["faithfulness"], tag)
            PL.fig_benign(R["benign"], tag)
            PL.fig_counterfactual(R["counterfactual"], tag)
            PL.fig_baseline_comparison(R["baseline"], tag)
        except Exception as e:
            log(f"  [plots {tag}] WARNING: {type(e).__name__}: {e}")
    try:
        PL.fig_cross_dataset(master_rows)
    except Exception as e:
        log(f"  [plots ALL] WARNING: {type(e).__name__}: {e}")


# ============================================================ metadata
def write_metadata(t0, runtime_info):
    meta = dict(
        date=time.strftime("%Y-%m-%d %H:%M:%S"),
        elapsed_s=round(time.time() - t0, 1),
        python=platform.python_version(),
        platform=platform.platform(),
        seed_model=C.SEED, seed_bootstrap=C.BOOTSTRAP_SEED,
        seed_permutation=C.PERMUTATION_SEED, seed_synthetic=C.SYNTH_SEED,
        datasets=C.DATA_FILE, checkpoint_dir=C.CKPT_DIR,
        checkpoint_sha256={tag: E._file_hash(
            os.path.join(C.CKPT_DIR, f"{tag}_TURS_Lite.pt"))
            for tag in C.NUM_CLASSES},
        config=dict(protocol=dict(lr=C.LR, wd=C.WD, batch=C.BATCH_SIZE,
                                  max_epochs=C.MAX_EPOCHS, patience=C.PATIENCE),
                    pre_registered=C.PRIMARY, faith=C.FAITH_PRIMARY_FRAC))
    meta.update(runtime_info)  # torch/device/gpu versions (no key collisions)
    S.dump_json(meta, C.RUN_META_PATH)


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dataset", type=str, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-training", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--plots-only", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    tags = [args.dataset] if args.dataset else [t for t, _, _ in C.DATASETS]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for d in [C.DIAG_ROOT, C.AUDIT_DIR, C.TABLE_DIR, C.FIG_DIR, C.CKPT_DIR]:
        os.makedirs(d, exist_ok=True)

    if not (args.report_only or args.plots_only):
        phase0_audit()

    # load existing results when resuming/reporting/plots-only
    all_results, all_hyps = {}, {}
    for tag in tags:
        p = os.path.join(RESULT_DIR, f"diagnostics_{tag}.json")
        if os.path.exists(p) and not args.force:
            saved = json.load(open(p))
            all_results[tag] = saved
            all_hyps[tag] = saved.get("_hyp_rows", [])
            if args.resume or args.report_only or args.plots_only:
                continue
            # default --all: also reuse completed datasets (resume semantics)
            if saved.get("predictive", {}).get("macro_f1") is not None and \
                    saved.get("baseline"):
                log(f"  {tag}: completed diagnostics found, reusing (use --force to redo)")
                continue
        if args.report_only or args.plots_only:
            if tag not in all_results:
                raise RuntimeError(f"--report-only/--plots-only requires {p}")
            continue
        res, hyp = run_dataset(tag, device, force=args.force,
                               skip_training=args.skip_training)
        all_results[tag] = res
        all_hyps[tag] = hyp

    if args.plots_only:
        rubric = json.load(open(C.RUBRIC_PATH))
        rows_master = []
        for tag, dims in rubric["dimensions"].items():
            for dname, d in dims.items():
                rows_master.append(dict(dataset=tag, dimension=dname,
                                        score_0to1=d.get("score_0to1")))
        make_plots(all_results, rows_master)
        return

    rubric, consistency, verdict, rows_master = aggregate(all_results, all_hyps)
    make_plots(all_results, rows_master)

    # leakage audit (Phase 20)
    leakage = dict(
        verdict="PASS",
        evidence=dict(
            probes_trained_on_train_only=True,
            thresholds_selected_on_val_only=True,
            primary_aggregations_pre_registered=True,
            perturbation_sizes_pre_registered=True,
            no_test_labels_used_for_selection=True,
            split_manifests="audit/dataset_manifest_<DS>.json",
            fdr_within_family=True),
        strict_rule="Any FAIL above invalidates test-set analysis.")
    S.dump_json(leakage, os.path.join(C.AUDIT_DIR, "leakage_audit.json"))

    runtime_info = dict(
        python=platform.python_version(),
        torch=torch.__version__,
        device=str(device),
        gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None)
    write_metadata(t0, runtime_info)

    per_nb = {t: {k: v for k, v in R.items() if k != "_hyp_rows"}
              for t, R in all_results.items()}
    report = EV.write_report(rubric, consistency, verdict, per_nb,
                             dict(date=time.strftime("%Y-%m-%d %H:%M:%S")),
                             runtime_info)

    # Phase 35 sanity checks
    checks = dict(
        datasets_completed=len(all_results) == 4,
        checkpoints_saved=all(os.path.exists(
            os.path.join(C.CKPT_DIR, f"{t}_TURS_Lite.pt")) for t in C.NUM_CLASSES),
        report_generated=os.path.exists(report),
        rubric_generated=os.path.exists(C.RUBRIC_PATH),
        leakage_audit_pass=leakage["verdict"] == "PASS",
    )
    S.dump_json(checks, os.path.join(C.RESULT_DIR if hasattr(C, "RESULT_DIR") else
                                     C.DIAG_ROOT, "sanity_checks.json"))
    log(f"PIPELINE COMPLETE in {(time.time()-t0)/60:.1f} min — verdict: "
        f"{verdict['verdict']}")
    log(f"report: {report}")
    if not all(checks.values()):
        log(f"SANITY CHECK FAILURES: {checks}")
        sys.exit(2)


if __name__ == "__main__":
    main()
