"""
DRTN first probe: Haptics, seed 42, R0-R5 (spec sec. 27-29).

Usage:
    python -m experiments.drtn_haptics_seed42.runner            # full run
    python -m experiments.drtn_haptics_seed42.runner --rungs R4 # subset
    python -m experiments.drtn_haptics_seed42.runner --smoke    # tiny config

Output: results/drtn_haptics_seed42/{R0..R5}/ + audit + comparison tables.
Test set is touched exactly once per rung, after the best-val checkpoint is
frozen (sec. 3/19).
"""
import argparse
import copy
import json
import math
import os
import platform
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from experiments.external_stack_generalization.data import load_dataset  # noqa: E402
from models.drtn.model import build_model, count_params, usage_stats     # noqa: E402

SEED = 42
DATASET = "Haptics"
OUT_DIR = os.path.join(ROOT, "results", "drtn_haptics_seed42")

# ---- training configuration (fixed before any test contact, sec. 17) ----
D_MODEL = 64
K_CODES = 8
TAU = 0.5
EMA_DECAY = 0.99
BETA_COMMIT = 0.25
LAM_DIV = 0.01
DEAD_THRESHOLD = 1e-3
REVIVAL_PATIENCE = 100
BATCH_SIZE = 16
LR = 1e-3
WD = 1e-4
MAX_EPOCHS = 60
PATIENCE = 10
NUM_WORKERS = 0

RUNGS = ["R0", "R1", "R2", "R3", "R4", "R5"]
VQ_RUNGS = {"R2", "R3", "R4", "R5"}
DIVERSITY_RUNGS = {"R4", "R5"}


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loaders(batch_size):
    d = load_dataset(DATASET)
    zn = lambda X: ((X - X.mean(-1, keepdims=True)) /
                    (X.std(-1, keepdims=True) + 1e-8)).astype(np.float32)
    mk = lambda X, y, sh: DataLoader(
        TensorDataset(torch.from_numpy(zn(X))[:, None, :], torch.from_numpy(y)),
        batch_size=batch_size, shuffle=sh, num_workers=0,
        generator=torch.Generator().manual_seed(SEED) if sh else None)
    return d, mk(d["Xtr"], d["ytr"], True), mk(d["Xva"], d["yva"], False), \
        mk(d["Xte"], d["yte"], False)


def evaluate(model, loader, device):
    """Plain eval: logits + labels. Returns preds, targets, mean loss."""
    model.eval()
    preds, tgts, losses = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits, _ = model.forward_with_assign(xb)
            preds.append(logits.argmax(-1).cpu().numpy())
            tgts.append(yb.cpu().numpy())
            losses.append(float(F.cross_entropy(logits, yb)))
    return np.concatenate(preds), np.concatenate(tgts), float(np.mean(losses))


def mf1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def train_rung(rung, tr_dl, va_dl, te_dl, data, device, smoke=False):
    """Train one rung; returns full result dict. Test is evaluated ONCE at end."""
    max_ep = 8 if smoke else MAX_EPOCHS
    patience = 3 if smoke else PATIENCE
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = build_model(
        rung, c_in=1, n_classes=data["n_classes"], d_model=D_MODEL,
        n_codes=K_CODES, tau=TAU, ema_decay=EMA_DECAY, beta=BETA_COMMIT,
        lam_div=LAM_DIV if rung in DIVERSITY_RUNGS else 0.0,
        dead_threshold=DEAD_THRESHOLD, revival_patience=REVIVAL_PATIENCE,
    ).to(device)
    params = count_params(model)
    params["delta_vs_R0"] = params["total"] - R0_PARAMS["total"]

    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=LR, weight_decay=WD)
    steps_per_epoch = len(tr_dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, steps_per_epoch=steps_per_epoch, epochs=max_ep)

    history = []
    best_val, best_state, best_ep, no_imp = -1.0, None, 0, 0
    t0 = time.time()
    step = 0
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None

    for ep in range(max_ep):
        model.train()
        ep_loss = {}
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            z = model.encoder(xb)
            if rung in VQ_RUNGS and rung != "R2":
                q_st, assign, commit_raw = model.vq.quantize(z)
                h, _ = (model.pool(model.trajectory(q_st))
                        if rung == "R5" else model.pool(q_st))
                logits = model.classifier(h)
                ce = F.cross_entropy(logits, yb)
                commit = BETA_COMMIT * commit_raw
                terms = {"ce": float(ce), "commit": float(commit)}
                if rung in DIVERSITY_RUNGS:
                    div = model.diversity_loss_from_assign(assign)
                    loss = ce + commit + LAM_DIV * div
                    terms["div"] = float(div)
                    terms["div_w"] = float(LAM_DIV * div)
                    terms["total"] = float(loss)
                else:
                    loss = ce + commit
                    terms["total"] = float(loss)
                # EMA codebook update once per optimizer step (sec. 10)
                model.vq.ema_step(
                    z.detach().reshape(-1, z.shape[-1]),
                    assign.reshape(-1), step=step)
                step += 1
            else:
                logits, _ = model.forward_with_assign(xb)
                loss = F.cross_entropy(logits, yb)
                terms = {"ce": float(loss), "total": float(loss)}
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            for k, v in terms.items():
                ep_loss[k] = ep_loss.get(k, 0.0) + v
        nb = len(tr_dl)
        ep_loss = {k: v / nb for k, v in ep_loss.items()}

        # ---- validation + codebook diagnostics (sec. 19/21) ----
        vp, vt, vloss = evaluate(model, va_dl, device)
        val_mf1 = mf1(vt, vp)
        diag = {}
        if rung in VQ_RUNGS:
            diag = model.extract_diagnostics(va_dl, device)
            diag.pop("trajectories", None)
            diag.pop("attn", None)
            diag.pop("inputs", None)
            diag.pop("logits", None)
        history.append({
            "epoch": ep + 1, "val_mf1": round(val_mf1, 4),
            "val_loss": round(vloss, 4),
            **{k: round(v, 4) for k, v in ep_loss.items()},
            **{k: (round(v, 4) if isinstance(v, float) else v)
               for k, v in diag.items() if k != "usage"},
            "usage": diag.get("usage"),
            "revivals_so_far": (model.vq.total_revivals
                                if rung in {"R3", "R4", "R5"} else 0),
        })
        marker = ""
        if val_mf1 > best_val:
            best_val, best_ep, no_imp = val_mf1, ep + 1, 0
            best_state = copy.deepcopy(model.state_dict())
            marker = " *"
        else:
            no_imp += 1
        print(f"  [{rung}] ep{ep+1}/{max_ep} valMF1={val_mf1:.4f} "
              f"(best {best_val:.4f}@{best_ep}) "
              f"loss={ep_loss['total']:.4f} "
              f"{'enc=%0.3f ' % diag['normalized_entropy'] if 'normalized_entropy' in diag else ''}"
              f"[{time.time()-t0:.0f}s]{marker}", flush=True)
        if no_imp >= patience:
            print(f"  [{rung}] early stop at ep{ep+1}", flush=True)
            break

    train_time = time.time() - t0
    model.load_state_dict(best_state)

    # ---- FINAL test evaluation, exactly once (sec. 3) ----
    tp, tt, _ = evaluate(model, te_dl, device)
    test_mf1 = mf1(tt, tp)
    tr_p, tr_t, _ = evaluate(model, tr_dl, device)
    diag_test = model.extract_diagnostics(te_dl, device,
                                          save_trajectories=0)
    diag_val = model.extract_diagnostics(
        va_dl, device, save_trajectories=6, save_input=True)
    for k in ("logits",):
        diag_test.pop(k, None); diag_val.pop(k, None)

    result = {
        "rung": rung,
        "config": {
            "dataset": DATASET, "seed": SEED, "d_model": D_MODEL,
            "n_codes": K_CODES, "tau": TAU, "ema_decay": EMA_DECAY,
            "beta_commit": BETA_COMMIT, "lam_div": LAM_DIV,
            "dead_threshold": DEAD_THRESHOLD,
            "revival_patience": REVIVAL_PATIENCE,
            "batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WD,
            "max_epochs": max_ep, "patience": patience,
            "trajectory": ({"layers": 2, "heads": 4, "ffn": 128,
                            "dropout": 0.1} if rung == "R5" else None),
        },
        "params": params,
        "best_epoch": best_ep,
        "best_val_mf1": round(best_val, 4),
        "train_mf1_final": round(mf1(tr_t, tr_p), 4),
        "test": {
            "macro_f1": round(test_mf1, 4),
            "accuracy": round(float(accuracy_score(tt, tp)), 4),
            "weighted_f1": round(float(f1_score(tt, tp, average="weighted",
                                                zero_division=0)), 4),
            "class_f1s": [round(float(x), 4) for x in f1_score(
                tt, tp, average=None, zero_division=0,
                labels=list(range(data["n_classes"])))],
            "confusion_matrix": confusion_matrix(
                tt, tp, labels=list(range(data["n_classes"]))).tolist(),
        },
        "train_time_s": round(train_time, 1),
        "peak_mem_mb": (round(torch.cuda.max_memory_allocated(device) / 2**20, 1)
                        if device.type == "cuda" else None),
        "history": history,
        "final_diag_val": {k: v for k, v in diag_val.items()
                           if not isinstance(v, np.ndarray)},
        "final_diag_test": {k: v for k, v in diag_test.items()
                            if not isinstance(v, np.ndarray)},
        "codebook_final_usage_val": diag_val.get("usage"),
        "codebook_final_usage_test": diag_test.get("usage"),
        "revival_log": (model.vq.revival_log
                        if rung in {"R3", "R4", "R5"} else []),
        "total_revivals": (model.vq.total_revivals
                           if rung in {"R3", "R4", "R5"} else 0),
        "trajectories_val": diag_val.get("trajectories"),
        "inputs_val": diag_val.get("inputs"),
        "attn_val": diag_val.get("attn"),
    }

    # ---- checkpoint (sec. 24) ----
    ckpt = {
        "rung": rung, "config": result["config"], "epoch": best_ep,
        "best_val_mf1": best_val, "seed": SEED, "dataset": DATASET,
        "model_state": model.state_dict(),
        "optimizer_state": opt.state_dict(),
        "params": params,
    }
    return result, ckpt


def env_info():
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device": (torch.cuda.get_device_name(0)
                   if torch.cuda.is_available() else "cpu"),
        "numpy": np.__version__,
        "seed": SEED,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", nargs="*", default=RUNGS)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    set_seed(SEED)

    smoke_dir = ""
    if args.smoke:
        smoke_dir = os.path.join(ROOT, "results", "_drtn_smoke")

    data, tr_dl, va_dl, te_dl = make_loaders(
        8 if args.smoke else BATCH_SIZE)

    # ---- experiment audit (sec. 2) ----
    audit = {
        "dataset": DATASET,
        "source": data["paths"],
        "train": len(data["Xtr"]), "val": len(data["Xva"]),
        "test": len(data["Xte"]),
        "sequence_length": int(data["L"]),
        "channels": 1, "univariate": True,
        "n_classes": data["n_classes"],
        "class_names": data["class_names"],
        "train_class_distribution": {int(k): int(v) for k, v in
                                     zip(*np.unique(data["ytr"],
                                                    return_counts=True))},
        "val_class_distribution": {int(k): int(v) for k, v in
                                   zip(*np.unique(data["yva"],
                                                  return_counts=True))},
        "test_class_distribution": {int(k): int(v) for k, v in
                                    zip(*np.unique(data["yte"],
                                                   return_counts=True))},
        "normalization": "per-sample z-norm (project canonical)",
        "val_source": data["val_source"],
        "environment": env_info(),
        "config": {
            "d_model": D_MODEL, "n_codes": K_CODES, "tau": TAU,
            "ema_decay": EMA_DECAY, "beta_commit": BETA_COMMIT,
            "lam_div": LAM_DIV, "dead_threshold": DEAD_THRESHOLD,
            "revival_patience": REVIVAL_PATIENCE, "batch_size": BATCH_SIZE,
            "lr": LR, "weight_decay": WD, "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "selection": "best validation Macro-F1 checkpoint; "
                         "test evaluated once after freezing",
        },
    }
    base = smoke_dir or OUT_DIR
    os.makedirs(base, exist_ok=True)
    with open(os.path.join(base, "experiment_audit.json"), "w") as f:
        json.dump(audit, f, indent=2)
    print(json.dumps(audit, indent=2), flush=True)

    global R0_PARAMS
    R0_PARAMS = count_params(build_model("R0", n_classes=data["n_classes"]))

    summary = []
    for rung in args.rungs:
        print(f"\n===== {rung} =====", flush=True)
        result, ckpt = train_rung(rung, tr_dl, va_dl, te_dl, data, device,
                                  smoke=args.smoke)
        rdir = os.path.join(base, rung)
        os.makedirs(rdir, exist_ok=True)
        # strip ndarrays from the json-serializable result
        rjson = {k: v for k, v in result.items()
                 if not isinstance(v, np.ndarray)}
        with open(os.path.join(rdir, "result.json"), "w") as f:
            json.dump(rjson, f, indent=2)
        torch.save(ckpt, os.path.join(rdir, "checkpoint.pt"))
        if result.get("trajectories_val") is not None:
            np.save(os.path.join(rdir, "trajectories_val.npy"),
                    result["trajectories_val"])
            np.save(os.path.join(rdir, "inputs_val.npy"),
                    result["inputs_val"])
        if result.get("attn_val") is not None:
            np.save(os.path.join(rdir, "attn_val.npy"), result["attn_val"])
        summary.append({
            "rung": rung,
            "test_macro_f1": result["test"]["macro_f1"],
            "test_accuracy": result["test"]["accuracy"],
            "val_macro_f1": result["best_val_mf1"],
            "params": result["params"]["total"],
            "delta_vs_R0": result["params"]["delta_vs_R0"],
            "best_epoch": result["best_epoch"],
            "active_codes_val": result["final_diag_val"].get("active_codes"),
            "usage_entropy_val": result["final_diag_val"].get("entropy"),
            "normalized_entropy_val":
                result["final_diag_val"].get("normalized_entropy"),
            "perplexity_val": result["final_diag_val"].get("perplexity"),
            "dominant_fraction_val":
                result["final_diag_val"].get("dominant_fraction"),
            "total_revivals": result["total_revivals"],
            "train_time_s": result["train_time_s"],
        })
        print(f"  => {rung}: TEST MF1={result['test']['macro_f1']:.4f} "
              f"Acc={result['test']['accuracy']:.4f} "
              f"params={result['params']['total']:,} "
              f"({result['train_time_s']:.0f}s)", flush=True)

    with open(os.path.join(base, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSUMMARY:", flush=True)
    for s in summary:
        print(json.dumps(s), flush=True)


if __name__ == "__main__":
    main()
