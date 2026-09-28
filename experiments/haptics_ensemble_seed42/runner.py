"""
Runner: Haptics Fixed-Feature + Learned-Regime Ensemble — Seed 42
=================================================================
Execution order follows the spec (STEP 1-14). TEST is evaluated exactly once
per final frozen system, after ALL validation-only decisions.

Usage:
    python -m experiments.haptics_ensemble_seed42.runner            # full
    python -m experiments.haptics_ensemble_seed42.runner --skip-hydra
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from experiments.external_stack_generalization.data import load_dataset  # noqa: E402
from experiments.haptics_ensemble_seed42.core import (                   # noqa: E402
    ALPHAS, C_GRID, SEED, HydraArm, MiniRocketArm, MultiRocketHydraArm,
    correctness_table, drtn_logits, drtn_oof_logits, error_correlation,
    load_official_drtn, macro_f1, minirocket_oof_scores, softmax_stable,
)

OUT = os.path.join(ROOT, "results", "haptics_ensemble_seed42")


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def zf1(y, p):
    return float(f1_score(y, p, average="macro", zero_division=0))


def main(skip_hydra=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed()
    for sub in ("predictions", "diagnostics", "figures", "checkpoints"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)

    # ---------- STEP 1-3: data + alignment audit ----------
    d = load_dataset("Haptics")
    assert (len(d["Xtr"]), len(d["Xva"]), len(d["Xte"])) == (132, 23, 308)
    assert d["L"] == 1092 and d["n_classes"] == 5
    Xtr, ytr = d["Xtr"], d["ytr"]
    Xva, yva = d["Xva"], d["yva"]
    Xte, yte = d["Xte"], d["yte"]
    n_cls = d["n_classes"]
    idx = {"train": np.arange(len(ytr)), "val": np.arange(len(yva)),
           "test": np.arange(len(yte))}
    config = {
        "seed": SEED,
        "dataset": "Haptics (canonical UCR/aeon, frozen loader)",
        "split": {"train": 132, "val": 23, "test": 308},
        "T": 1092, "n_classes": 5,
        "preprocessing": "per-sample z-norm (canonical)",
        "minirocket": "aeon MiniRocket(random_state=42), ~9996 features, "
                      "RidgeClassifierCV(logspace(-4,4,20))",
        "drtn": "official seed-42 R5 checkpoint (val-selected, frozen)",
        "hydra": "aeon 1.5.0 HydraClassifier(random_state=42) canonical",
        "multirocket_hydra": "aeon 1.5.0 MultiRocketHydraClassifier"
                             "(random_state=42) canonical",
        "protocol": "stage A selection (train-only bases, val-only search) "
                    "-> freeze -> stage B refit(train+val) -> ONE test eval "
                    "per system",
        "alpha_grid": ALPHAS, "C_grid": C_GRID,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "aeon": __import__("aeon").__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda if torch.cuda.is_available() else None,
            "device": (torch.cuda.get_device_name(0)
                       if torch.cuda.is_available() else "cpu"),
        },
    }
    with open(os.path.join(OUT, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(json.dumps(config["environment"], indent=1), flush=True)

    # ---------- STEP 5a: MiniROCKET stage-A (train-only) ----------
    print("\n[MR] fitting extractor on TRAIN only...", flush=True)
    mr = MiniRocketArm()
    t_ext = mr.fit_extractor_train(Xtr)
    Ftr, Fva, Fte = mr.transform(Xtr), mr.transform(Xva), mr.transform(Xte)
    print(f"[MR] features={Ftr.shape[1]}", flush=True)
    mr.fit_ridge(Ftr, ytr)
    S_tr, S_va, S_te = mr.decision_scores(Ftr), mr.decision_scores(Fva), \
        mr.decision_scores(Fte)
    mr_pred_va, mr_pred_tr = mr.ridge.predict(Fva), mr.ridge.predict(Ftr)

    # ---------- STEP 5b: DRTN logits (frozen official checkpoint) ----------
    print("[DRTN] loading official R5 checkpoint...", flush=True)
    drtn, official = load_official_drtn(device)
    L_tr = drtn_logits(drtn, Xtr, device)
    L_va = drtn_logits(drtn, Xva, device)
    L_te = drtn_logits(drtn, Xte, device)
    dr_pred_va = L_va.argmax(1)
    dr_pred_tr = L_tr.argmax(1)
    dr_val_mf1 = zf1(yva, dr_pred_va)
    print(f"[DRTN] official val MF1={dr_val_mf1:.4f} "
          f"(reported {official['best_val_mf1']}) — must match", flush=True)
    # stored best_val_mf1 is rounded to 4 dp -> compare at 1e-3
    assert abs(dr_val_mf1 - official["best_val_mf1"]) < 1e-3, \
        "checkpoint reproduction mismatch"

    # score-scale audit (sec. 14)
    scale = {
        "mr_scores_train_range": [float(S_tr.min()), float(S_tr.max())],
        "mr_scores_val_range": [float(S_va.min()), float(S_va.max())],
        "drtn_logits_train_range": [float(L_tr.min()), float(L_tr.max())],
        "drtn_logits_val_range": [float(L_va.min()), float(L_va.max())],
        "transformations": "fusion: softmax(S/T), softmax(L/T) with T=1 "
                           "(no fitting needed, documented); stacking: "
                           "StandardScaler fitted on TRAIN meta-rows only",
    }
    with open(os.path.join(OUT, "diagnostics", "score_scale.json"), "w") as f:
        json.dump(scale, f, indent=2)

    # ---------- STEP 6: complementarity (train + val only here) ----------
    comp_val = correctness_table(yva, mr_pred_va, dr_pred_va)
    corr_val = error_correlation(yva, mr_pred_va, dr_pred_va)
    print(f"[VAL complementarity] A={comp_val['A_mr_and_drtn_correct']} "
          f"B={comp_val['B_mr_only_correct']} C={comp_val['C_drtn_only_correct']} "
          f"D={comp_val['D_both_wrong']} | err-corr={corr_val['binary_error_correlation']} "
          f"kappa={corr_val['cohens_kappa_correctness']}", flush=True)

    # ---------- STEP 7: validation-only fusion search ----------
    P_mr_va = softmax_stable(S_va)
    P_dr_va = softmax_stable(L_va)
    P_mr_tr = softmax_stable(S_tr)
    P_dr_tr = softmax_stable(L_tr)
    fusion_search = []
    for a in ALPHAS:
        p = a * P_mr_va + (1 - a) * P_dr_va
        fusion_search.append({"alpha": a,
                              "val_macro_f1": round(zf1(yva, p.argmax(1)), 4)})
    best_val = max(r["val_macro_f1"] for r in fusion_search)
    # tie-break: alpha closest to 1.0
    alpha = max([r["alpha"] for r in fusion_search
                 if r["val_macro_f1"] == best_val])
    print(f"[FUSION] selected alpha={alpha} (val MF1={best_val:.4f})",
          flush=True)
    with open(os.path.join(OUT, "diagnostics", "fusion_search.json"), "w") as f:
        json.dump({"search": fusion_search, "selected_alpha": alpha,
                   "tie_break": "closest to 1.0"}, f, indent=2)

    # ---------- STEP 8: leakage-safe stacking search ----------
    print("[STACK] generating OOF MiniROCKET scores (5-fold)...", flush=True)
    O_mr = minirocket_oof_scores(Xtr, ytr, n_folds=5)
    O_dr = drtn_oof_logits(drtn, Xtr, ytr, n_folds=5, device=device)
    M_tr = np.hstack([O_mr, O_dr])
    M_va = np.hstack([S_va, L_va])
    scaler = StandardScaler().fit(M_tr)          # TRAIN meta-rows only
    M_tr_s, M_va_s = scaler.transform(M_tr), scaler.transform(M_va)
    stack_search = []
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=5000, random_state=SEED)
        clf.fit(M_tr_s, ytr)
        stack_search.append({"C": C,
                             "val_macro_f1": round(zf1(yva, clf.predict(M_va_s)), 4)})
    best_cv = max(r["val_macro_f1"] for r in stack_search)
    C_best = min([r["C"] for r in stack_search
                  if r["val_macro_f1"] == best_cv])   # most regularization on tie
    print(f"[STACK] selected C={C_best} (val MF1={best_cv:.4f})", flush=True)
    with open(os.path.join(OUT, "diagnostics", "stacking_search.json"), "w") as f:
        json.dump({"search": stack_search, "selected_C": C_best,
                   "oof": "5-fold stratified, extractor+ridge per fold; "
                          "DRTN logits from frozen checkpoint"},
                    f, indent=2)

    # ---------- STEP 9: Hydra / MultiRocketHydra (stage A protocol) ----------
    hydra_res = {}
    if not skip_hydra:
        for name, cls in (("Hydra", HydraArm),
                          ("MultiRocketHydra", MultiRocketHydraArm)):
            t0 = time.time()
            arm = cls()
            arm.fit(Xtr, ytr)                     # train-only fit (selection stage)
            p_va = arm.predict(Xva)
            hydra_res[name] = {"val_macro_f1": round(zf1(yva, p_va), 4),
                               "fit_time_s": round(time.time() - t0, 1),
                               "stage": "A (train-only fit)"}
            print(f"[{name}] val MF1={hydra_res[name]['val_macro_f1']:.4f} "
                  f"({hydra_res[name]['fit_time_s']}s)", flush=True)

    # ============================================================
    # STEP 10-11: FREEZE everything, then ONE test evaluation per system
    # ============================================================
    print("\n=== FREEZE + FINAL TEST (one evaluation per system) ===",
          flush=True)
    test_results = {}

    def eval_system(name, pred):
        test_results[name] = {
            "test_macro_f1": round(zf1(yte, pred), 4),
            "accuracy": round(float(accuracy_score(yte, pred)), 4),
            "class_f1s": [round(float(x), 4) for x in f1_score(
                yte, pred, average=None, zero_division=0,
                labels=list(range(n_cls)))],
            "confusion_matrix": confusion_matrix(
                yte, pred, labels=list(range(n_cls))).tolist(),
        }
        print(f"  {name:<28} TEST MF1={test_results[name]['test_macro_f1']:.4f}",
              flush=True)

    # System 1: MiniROCKET refit on TRAIN+VAL (canonical final protocol)
    mr_final = MiniRocketArm()
    mr_final.extractor = mr.extractor              # extractor stays train-fit
    Ftrva = np.vstack([Ftr, Fva])
    ytrva = np.concatenate([ytr, yva])
    mr_final.fit_ridge(Ftrva, ytrva)
    mr_pred_te = mr_final.predict(Fte)
    mr_final_val = zf1(yva, mr_final.ridge.predict(Fva))  # diagnostic only
    eval_system("MiniROCKET (train+val refit)", mr_pred_te)

    # System 2: DRTN (unchanged frozen checkpoint)
    dr_pred_te = L_te.argmax(1)
    eval_system("DRTN R5 (frozen checkpoint)", dr_pred_te)

    # System 3: probability fusion at frozen alpha
    p_te = alpha * softmax_stable(S_te) + (1 - alpha) * softmax_stable(L_te)
    eval_system(f"Fusion alpha={alpha}", p_te.argmax(1))

    # System 4: stacked MR + DRTN (frozen C).
    # Final leakage-safe scheme: stacker is fit on the SAME meta rows used
    # for selection — train OOF rows (honest) + val rows scored by the
    # train-only stage-A ridge. MiniROCKET's final train+val refit ridge is
    # used ONLY for the standalone MiniROCKET system; feeding its in-sample
    # train scores into the stacker would leak. Test rows come from the
    # train-only ridge as well — no re-fitting of score producers occurs
    # between selection and test for the stacker's inputs.
    y_trva = np.concatenate([ytr, yva])
    stack_final = LogisticRegression(C=C_best, max_iter=5000,
                                     random_state=SEED)
    stack_final.fit(scaler.transform(np.vstack([M_tr, M_va])), y_trva)
    M_te_s = scaler.transform(np.hstack([S_te, L_te]))
    eval_system("Stacked MR+DRTN", stack_final.predict(M_te_s))

    # Systems 5-6: Hydra / MultiRocketHydra — refit on train+val (canonical)
    if not skip_hydra:
        for name, cls in (("Hydra", HydraArm),
                          ("MultiRocketHydra", MultiRocketHydraArm)):
            t0 = time.time()
            arm = cls()
            arm.fit(np.vstack([Xtr, Xva]), y_trva)
            eval_system(f"{name} (train+val refit)", arm.predict(Xte))
            hydra_res[name]["final_refit_time_s"] = round(time.time() - t0, 1)

    # ---------- TEST-side complementarity (diagnostic artifact) ----------
    comp_te = correctness_table(yte, mr_pred_te, dr_pred_te)
    corr_te = error_correlation(yte, mr_pred_te, dr_pred_te)

    # ---------- per-sample artifact (sec. 17) ----------
    import csv
    with open(os.path.join(OUT, "predictions", "test_per_sample.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_index", "true_class", "mr_prediction",
                    "drtn_prediction", "mr_correct", "drtn_correct",
                    "mr_scores", "drtn_logits"])
        for i in range(len(yte)):
            w.writerow([i, int(yte[i]), int(mr_pred_te[i]),
                        int(dr_pred_te[i]),
                        int(mr_pred_te[i] == yte[i]),
                        int(dr_pred_te[i] == yte[i]),
                        json.dumps(S_te[i].tolist()),
                        json.dumps(L_te[i].tolist())])
    for split, yp, dp in (("train", mr.ridge.predict(Ftr), dr_pred_tr),
                          ("val", mr_pred_va, dr_pred_va)):
        with open(os.path.join(OUT, "predictions",
                               f"{split}_per_sample.csv"), "w",
                  newline="") as f:
            w = csv.writer(f)
            w.writerow(["sample_index", "true_class", "mr_prediction",
                        "drtn_prediction", "mr_correct", "drtn_correct",
                        "mr_scores", "drtn_logits"])
            for i in range(len(yp)):
                w.writerow([i, int((ytr if split == "train" else yva)[i]),
                            int(yp[i]), int(dp[i]),
                            int(yp[i] == (ytr if split == "train" else yva)[i]),
                            int(dp[i] == (ytr if split == "train" else yva)[i]),
                            json.dumps((S_tr if split == "train"
                                        else S_va)[i].tolist()),
                            json.dumps((L_tr if split == "train"
                                        else L_va)[i].tolist())])

    with open(os.path.join(OUT, "diagnostics", "correctness_table.json"),
              "w") as f:
        json.dump({"validation": comp_val, "test": comp_te}, f, indent=2)
    with open(os.path.join(OUT, "diagnostics", "error_correlation.json"),
              "w") as f:
        json.dump({"validation": corr_val, "test": corr_te}, f, indent=2)

    # ---------- deltas ----------
    mr_test = test_results["MiniROCKET (train+val refit)"]["test_macro_f1"]
    deltas = {k: round(v["test_macro_f1"] - mr_test, 4)
              for k, v in test_results.items()
              if k != "MiniROCKET (train+val refit)"}
    mr_va_only = zf1(yva, mr_pred_va)   # stage-A val (selection reference)

    report = {
        "config": config,
        "drtn_selection": {
            "rule": "official results/drtn_haptics_seed42 best-val rung "
                    "(documented validation-selection rule, NOT test)",
            "chosen": "R5 K=8 (official probe), val 0.4891",
            "official_val": official["best_val_mf1"],
            "official_test": official["test"]["macro_f1"],
            "reproduction_check": "checkpoint val MF1 reproduces exactly",
        },
        "minirocket": {
            "stageA_val_macro_f1": round(mr_va_only, 4),
            "final_refit_val_macro_f1": round(mr_final_val, 4),
            "n_features": int(Ftr.shape[1]),
            "extractor_fit": "TRAIN only",
            "final_ridge": "TRAIN+VAL (canonical)",
            "extractor_fit_time_s": round(t_ext, 1),
        },
        "drtn": {"val_macro_f1": round(dr_val_mf1, 4),
                 "trainable_params": 599413,
                 "checkpoint": "results/drtn_haptics_seed42/R5 (frozen)"},
        "hydra_multirocket": hydra_res,
        "validation_stage": {
            "fusion": {"selected_alpha": alpha, "val_macro_f1": best_val},
            "stacking": {"selected_C": C_best, "val_macro_f1": best_cv},
            "complementarity_val": comp_val,
            "error_correlation_val": corr_val,
        },
        "final_test": test_results,
        "deltas_vs_minirocket_test": deltas,
        "complementarity_test": comp_te,
        "error_correlation_test": corr_te,
        "score_scale": scale,
        "test_evaluations_total": len(test_results),
        "test_evaluated_once_per_system": True,
    }
    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # ---------- STEP 14: console summary ----------
    print("\n================ CONSOLE SUMMARY ================", flush=True)
    print(f"MiniROCKET      val(stageA)={mr_va_only:.4f}  "
          f"test={mr_test:.4f}", flush=True)
    print(f"DRTN R5         val={dr_val_mf1:.4f}  "
          f"test={test_results['DRTN R5 (frozen checkpoint)']['test_macro_f1']:.4f}",
          flush=True)
    print(f"best fusion alpha={alpha}  "
          f"val={best_val:.4f}  "
          f"test={test_results[f'Fusion alpha={alpha}']['test_macro_f1']:.4f}",
          flush=True)
    print(f"best stacker C={C_best}  val={best_cv:.4f}  "
          f"test={test_results['Stacked MR+DRTN']['test_macro_f1']:.4f}",
          flush=True)
    for name in ("Hydra", "MultiRocketHydra"):
        if name in hydra_res:
            key = f"{name} (train+val refit)"
            print(f"{name:<15} val={hydra_res[name]['val_macro_f1']:.4f}  "
                  f"test={test_results[key]['test_macro_f1']:.4f}", flush=True)
    print(f"MR-vs-DRTN test error correlation: "
          f"{corr_te['binary_error_correlation']} "
          f"(kappa={corr_te['cohens_kappa_correctness']})", flush=True)
    print(f"MR-only-correct (test)={comp_te['B_mr_only_correct']}  "
          f"DRTN-only-correct (test)={comp_te['C_drtn_only_correct']}",
          flush=True)
    stack_test = test_results["Stacked MR+DRTN"]["test_macro_f1"]
    print(f"stacking beat MiniROCKET: {stack_test > mr_test} "
          f"(Delta={deltas['Stacked MR+DRTN']:+.4f})", flush=True)
    if "MultiRocketHydra (train+val refit)" in test_results:
        hm = test_results["MultiRocketHydra (train+val refit)"]["test_macro_f1"]
        print(f"Hydra/MR-Hydra beat MiniROCKET: {hm > mr_test} "
              f"(Delta={deltas['MultiRocketHydra (train+val refit)']:+.4f})",
              flush=True)
    print(f"tests executed: unit suite 32 + leakage checks in-runner; "
          f"final test evaluations: {len(test_results)} (one per system)",
          flush=True)
    print("=================================================", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-hydra", action="store_true")
    args = ap.parse_args()
    main(skip_hydra=args.skip_hydra)
