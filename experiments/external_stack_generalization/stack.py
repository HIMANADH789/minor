"""
Frozen TURS-Stack adapter for the external generalization experiment.

PHASE 8/11 ABSOLUTE RULE: the architecture is NOT modified. Everything here
imports the canonical, validated implementation read-only:

  - models/turs_stack/model.py      (TURSStack + 4 branches + 5 combiners)
  - experiments/run_turs_stack_benchmark.py (canonical protocol helpers:
    set_seed, compute_sigma0 calibration warm start, run_correctness_gates,
    train_model [joint CE over branches, AdamW 3e-4/1e-2, OneCycleLR max_lr
    3e-3, batch 64, max 30 ep, patience 8 on val soft-vote MF1, clip 1.0],
    predict_branch_probs, combiner fitting recipe [Adam lr 0.05, 300 steps,
    VALIDATION ONLY], val-selected combiner policy).

The ONLY adaptations (documented, non-architectural):
  - dataset loader / input dimensions (T, C) come from the new datasets
  - results are written to results/external_stack_generalization/
  - a resumable checkpoint is stored per dataset for --resume
"""
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.turs_stack.model import (  # noqa: E402
    TURSStack,
    StaticWeightCombiner,
    StackingCombiner,
    DiagnosticConditionedCombiner,
    soft_vote,
    hard_vote,
)
from experiments.run_turs_stack_benchmark import (  # noqa: E402  (read-only reuse)
    LAMBDA_I, MAX_EPOCHS, PATIENCE, LR, WD, BATCH_SIZE,
    set_seed, compute_sigma0, count_params, group_param_count,
    run_correctness_gates, predict_branch_probs, train_model, mf1,
)
from experiments.external_stack_generalization.baselines import (
    full_metrics,
)

BRANCH_NAMES = ["lite", "rv", "cs", "cmr"]
COMBINER_NAMES = ["soft_vote", "hard_vote", "static_weights", "stacking",
                  "diagnostic_stacking"]


def fit_combiners_canonical(P_va, y_va, e_va):
    """Canonical combiner fitting (validation ONLY), verbatim recipe."""
    y_va_t = torch.as_tensor(y_va)
    probs_va = [P_va[:, k] for k in range(4)]

    t0 = time.time()
    swc = StaticWeightCombiner()
    o = torch.optim.Adam(swc.parameters(), lr=0.05)
    for _ in range(300):
        o.zero_grad()
        pf, _ = swc(probs_va)
        loss = nn.functional.cross_entropy(pf, y_va_t)
        loss.backward()
        o.step()
    w_learned = torch.softmax(swc.theta.detach(), dim=0).cpu().numpy()

    sc = StackingCombiner(P_va.shape[-1])
    o = torch.optim.Adam(sc.parameters(), lr=0.05)
    for _ in range(300):
        o.zero_grad()
        loss = nn.functional.cross_entropy(sc(probs_va), y_va_t)
        loss.backward()
        o.step()

    dcc = DiagnosticConditionedCombiner(P_va.shape[-1])
    o = torch.optim.Adam(dcc.parameters(), lr=0.05)
    for _ in range(300):
        o.zero_grad()
        loss = nn.functional.cross_entropy(dcc(probs_va, e_va), y_va_t)
        loss.backward()
        o.step()
    return swc, sc, dcc, w_learned, time.time() - t0


@torch.no_grad()
def eval_all_combiners(swc, sc, dcc, P, y, e):
    probs = [P[:, k] for k in range(4)]
    preds = {
        "soft_vote": soft_vote(probs).argmax(-1).numpy(),
        "hard_vote": hard_vote(probs).numpy(),
        "static_weights": swc(probs)[0].argmax(-1).numpy(),
        "stacking": sc(probs).argmax(-1).numpy(),
        "diagnostic_stacking": dcc(probs, e).argmax(-1).numpy(),
    }
    return {k: v.astype(np.int64) for k, v in preds.items()}


def run_stack(d, device, ckpt_path, seed=42, log=print,
              max_epochs=None, patience=None):
    """Train/freeze/combine the canonical TURS-Stack on one external dataset.

    max_epochs/patience are ONLY overridden in smoke tests; the full run uses
    the canonical MAX_EPOCHS=30 / PATIENCE=8 unchanged.

    Returns (res, artifacts) where artifacts carries test predictions of the
    selected combiner, soft-vote, and each branch (for complementarity).
    """
    from torch.utils.data import DataLoader, TensorDataset

    set_seed(seed)
    n_cls = d["n_classes"]
    T = d["L"]
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    Xtr = d["Xtr"][:, None, :].astype(np.float32)
    Xva = d["Xva"][:, None, :].astype(np.float32)
    Xte = d["Xte"][:, None, :].astype(np.float32)
    ytr, yva, yte = d["ytr"], d["yva"], d["yte"]

    tr_dl = DataLoader(TensorDataset(torch.from_numpy(Xtr),
                                     torch.from_numpy(ytr)),
                       batch_size=BATCH_SIZE, shuffle=True,
                       num_workers=0, pin_memory=(device.type == "cuda"))
    va_dl = DataLoader(TensorDataset(torch.from_numpy(Xva),
                                     torch.from_numpy(yva)),
                       batch_size=256, shuffle=False)
    te_dl = DataLoader(TensorDataset(torch.from_numpy(Xte),
                                     torch.from_numpy(yte)),
                       batch_size=256, shuffle=False)

    model = TURSStack(in_channels=1, num_classes=n_cls, sequence_length=T,
                      lambda_I=LAMBDA_I).to(device)

    # ---- correctness gates (canonical, Part 53-61) ----
    log("  [stack] correctness gates...")
    gate_report = run_correctness_gates(model, device, n_cls)

    # ---- calibration warm start from TRAIN only (canonical) ----
    sigma0 = compute_sigma0(Xtr[:, 0, :])
    with torch.no_grad():
        model.cs.scale_params.raw_sigma.data.fill_(
            float(np.log(np.exp(sigma0 - 1.0) - 1)))
        model.cmr.scale_params.raw_sigma.data.fill_(
            float(np.log(np.exp(sigma0 - 1.0) - 1)))
    log(f"  [stack] sigma0 (train-only) = {sigma0:.3f}")

    trunk_p, branch_p = group_param_count(model)
    total_p = count_params(model)

    # ---- training with resume (canonical train_model, val-only stopping) ----
    start_ep, best_val_prev, history_prev = 0, -1.0, []
    if os.path.exists(ckpt_path):
        try:
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ck["model_state_dict"])
            start_ep = ck.get("epoch", 0)
            best_val_prev = ck.get("best_metric", -1.0)
            history_prev = ck.get("history", [])
            log(f"  [stack] resumed checkpoint at epoch {start_ep}")
        except Exception as e:  # noqa
            log(f"  [stack] resume failed ({e}); training fresh")
            start_ep = 0

    t_train0 = time.time()
    if start_ep >= MAX_EPOCHS:
        history = history_prev
        best_val = best_val_prev
        best_ep = ck.get("best_ep", 0)
        train_time = ck.get("time_s", 0.0)
    else:
        history, best_val, best_ep, train_time = train_model(
            model, tr_dl, va_dl, te_dl, yva, yte, n_cls,
            d["name"], device,
            epochs=max_epochs or MAX_EPOCHS,
            patience=patience or PATIENCE)
        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": MAX_EPOCHS,
            "best_metric": best_val,
            "best_ep": best_ep,
            "time_s": train_time,
            "history": history,
            "sigma0": sigma0,
            "num_classes": n_cls,
            "sequence_length": T,
        }, ckpt_path)

    # ---- freeze; held-out predictions (val + test) ----
    for p_ in model.parameters():
        p_.requires_grad_(False)
    model.eval()

    t_inf0 = time.time()
    P_va, y_va_t, e_va = predict_branch_probs(model, va_dl, device,
                                              return_novelty=True)
    P_te, y_te_t, e_te = predict_branch_probs(model, te_dl, device,
                                              return_novelty=True)
    infer_t = time.time() - t_inf0

    P_va, P_te = P_va.cpu(), P_te.cpu()
    e_va, e_te = e_va.cpu(), e_te.cpu()

    # ---- combiner fitting: VALIDATION ONLY (canonical recipe) ----
    swc, sc, dcc, w_learned, comb_t = fit_combiners_canonical(
        P_va, y_va_t.numpy(), e_va)

    preds_va = eval_all_combiners(swc, sc, dcc, P_va, y_va_t, e_va)
    preds_te = eval_all_combiners(swc, sc, dcc, P_te, y_te_t, e_te)

    combo_val = {k: round(mf1(y_va_t.numpy(), v), 4)
                 for k, v in preds_va.items()}
    combo_test = {k: round(mf1(y_te_t.numpy(), v), 4)
                  for k, v in preds_te.items()}
    best_combo_val = max(combo_val, key=combo_val.get)
    sel_te_preds = preds_te[best_combo_val]

    # ---- per-class metrics for soft-vote and selected combiner ----
    m_soft = full_metrics(y_te_t.numpy(), preds_te["soft_vote"], n_cls)
    m_sel = full_metrics(y_te_t.numpy(), sel_te_preds, n_cls)

    # ---- branch MF1 + confidence diagnostics (PHASE 17) ----
    branch_val = {n: round(mf1(y_va_t.numpy(),
                               P_va[:, k].argmax(-1).numpy()), 4)
                  for k, n in enumerate(BRANCH_NAMES)}
    branch_test = {n: round(mf1(y_te_t.numpy(),
                                P_te[:, k].argmax(-1).numpy()), 4)
                   for k, n in enumerate(BRANCH_NAMES)}

    def conf_stats(P, y_np):
        conf = P.max(-1).values.numpy()
        preds = P.argmax(-1).numpy()
        return {"mean_conf": round(float(conf.mean()), 4),
                "pred_dist": np.bincount(preds, minlength=P.shape[-1]).tolist(),
                "true_dist": np.bincount(y_np, minlength=P.shape[-1]).tolist()}

    diagnostics = {
        "val": {n: conf_stats(P_va[:, k], y_va_t.numpy())
                for k, n in enumerate(BRANCH_NAMES)},
        "test": {n: conf_stats(P_te[:, k], y_te_t.numpy())
                 for k, n in enumerate(BRANCH_NAMES)},
        "soft_vote_test": conf_stats(
            soft_vote([P_te[:, k] for k in range(4)]), y_te_t.numpy()),
    }

    res = {
        "model": "TURS-Stack",
        "n_classes": n_cls, "T": int(T),
        "n_train": len(ytr), "n_val": len(yva), "n_test": len(yte),
        "seed": seed,
        "params": {"shared_trunk": trunk_p,
                   "branches": branch_p,
                   "total": total_p},
        "gates": {k: v for k, v in gate_report.items()
                  if not isinstance(v, dict)},
        "calibration": {"sigma0_train_only": round(sigma0, 3)},
        "training": {"best_val_softvote_mf1": round(float(best_val), 4),
                     "best_epoch": best_ep,
                     "train_time_s": round(train_time, 1)},
        "branch_val_mf1": branch_val,
        "branch_test_mf1": branch_test,
        "combination_val_mf1": combo_val,
        "combination_test_mf1": combo_test,
        "best_combiner_by_val": best_combo_val,
        "preselected_test_mf1": combo_test[best_combo_val],
        "static_weights_learned": [round(float(w), 4) for w in w_learned],
        "val_softvote_mf1": combo_val["soft_vote"],
        "test_softvote_mf1": combo_test["soft_vote"],
        "gap_softvote": round(combo_test["soft_vote"]
                              - combo_val["soft_vote"], 4),
        "val_selected_mf1": combo_val[best_combo_val],
        "gap_selected": round(combo_test[best_combo_val]
                              - combo_val[best_combo_val], 4),
        "test_metrics_softvote": m_soft,
        "test_metrics_selected": m_sel,
        "time_train_s": round(train_time, 2),
        "time_combiner_fit_s": round(comb_t, 2),
        "time_inference_s": round(infer_t, 2),
        "device": str(device),
    }

    artifacts = {
        "test_preds_selected": sel_te_preds,
        "test_preds_softvote": preds_te["soft_vote"],
        "test_preds_branches": {n: P_te[:, k].argmax(-1).numpy()
                                for k, n in enumerate(BRANCH_NAMES)},
        "test_probs_branches": P_te.numpy(),
        "y_test": y_te_t.numpy(),
    }

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return res, artifacts
