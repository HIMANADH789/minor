"""TURS-RRMT training + evaluation (canonical protocol) and the A6/A7
routing interventions on the frozen A4 model."""
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (f1_score, accuracy_score, confusion_matrix,
                             balanced_accuracy_score, matthews_corrcoef)

from experiments.turs_rrmt.data import SEED, NUM_CLASSES

VARIANTS = ["A0", "A1", "A2", "A3", "A4", "A5"]


def _mf1(y, pred, n_cls):
    return float(f1_score(y, pred, average="macro", zero_division=0,
                          labels=list(range(n_cls))))


def full_metrics(y, pred, n_cls):
    y = np.asarray(y)
    pred = np.asarray(pred)
    from sklearn.metrics import cohen_kappa_score
    from src.diagnostics.calibration import ece, brier, nll as nll_fn
    return dict(
        macro_f1=_mf1(y, pred, n_cls),
        accuracy=float(accuracy_score(y, pred)),
        weighted_f1=float(f1_score(y, pred, average="weighted", zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        mcc=float(matthews_corrcoef(y, pred)),
        kappa=float(cohen_kappa_score(y, pred)),
        confusion_matrix=confusion_matrix(y, pred, labels=list(range(n_cls))).tolist(),
    )


def build_model(variant, ds, M=128, seed=SEED):
    from models.turs_rrmt.model import TURSRRMT
    return TURSRRMT(num_classes=ds["n_cls"], seq_len=ds["L"], M=M, J=4, topk=2,
                    variant=variant, seed=seed, head_width=192)


def fit_flavor_reference(model, Xtr):
    model.flavors.fit_reference(torch.from_numpy(Xtr).float())


def train_variant(variant, ds, device, log=print, force=False,
                  ckpt_dir=None, max_epochs=30, patience=8):
    """Train one variant with the canonical protocol. Returns (path, meta)."""
    from models.turs_rrmt.model import count_params
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"{ds['tag']}_{variant}.pt")
    meta_path = path.replace(".pt", "_meta.json")
    if os.path.exists(path) and os.path.exists(meta_path) and not force:
        return path, json.load(open(meta_path))

    if variant == "A0":
        return ridge_baseline(ds, ckpt_dir, force=force)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = build_model(variant, ds).to(device)
    fit_flavor_reference(model, ds["Xtr"])
    trainable, fixed = count_params(model)

    tr_dl = DataLoader(TensorDataset(torch.from_numpy(ds["Xtr"]).float(),
                                     torch.from_numpy(ds["y_train"]).long()),
                       batch_size=64, shuffle=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=3e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4,
                                                steps_per_epoch=len(tr_dl),
                                                epochs=max_epochs)
    history, best_val, best_state, best_ep, no_imp = [], -1, None, 0, 0
    t0 = time.time()

    def val_mf1():
        model.eval()
        with torch.no_grad():
            logits = []
            for i in range(0, len(ds["Xva"]), 256):
                xl = torch.from_numpy(ds["Xva"][i:i + 256]).float().to(device)
                logits.append(model(xl).cpu())
            pred = torch.cat(logits).argmax(1).numpy()
        return _mf1(ds["y_val"], pred, ds["n_cls"])

    for ep in range(max_epochs):
        model.train()
        ep_loss = 0.0
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters()
                                            if p.requires_grad], 1.0)
            opt.step(); sched.step()
            ep_loss += float(loss) * len(xb)
        vm = val_mf1()
        history.append(dict(epoch=ep + 1,
                            train_loss=round(ep_loss / len(ds["y_train"]), 6),
                            val_mf1=round(vm, 6)))
        if vm > best_val:
            best_val, best_ep, no_imp = vm, ep + 1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            no_imp += 1
        log(f"    [{variant} {ds['tag']}] ep {ep+1:2d}/{max_epochs} "
            f"loss={ep_loss/len(ds['y_train']):.4f} val_mf1={vm:.4f} best={best_val:.4f}")
        if no_imp >= patience:
            log(f"    [{variant} {ds['tag']}] early stop ep {ep+1}")
            break

    elapsed = time.time() - t0
    model.load_state_dict(best_state)
    torch.save(dict(model_state_dict=model.state_dict(), variant=variant,
                    M=128, J=4, topk=2, seq_len=ds["L"], num_classes=ds["n_cls"],
                    seed=SEED, ppv_window=model.ppv_window,
                    best_val_mf1=best_val, best_ep=best_ep,
                    head_width=192,
                    flavor_ref=dict(ref_q=model.flavors.ref_q.tolist(),
                                    ref_q_fine=model.flavors.ref_q_fine.tolist(),
                                    ref_lag=model.flavors.ref_lag.tolist()),
                    bank_spec_hash=model.bank.spec_hash), path)
    meta = dict(tag=ds["tag"], variant=variant, params_trainable=int(trainable),
                params_fixed=int(fixed), best_val_mf1=round(best_val, 6),
                best_epoch=best_ep, total_epochs=len(history),
                elapsed_s=round(elapsed, 1), seed=SEED,
                ppv_window=model.ppv_window, bank_spec_hash=model.bank.spec_hash,
                history=history)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log(f"    [{variant} {ds['tag']}] done val_mf1={best_val:.4f} "
        f"ep={best_ep} {elapsed:.0f}s params={trainable:,}+{fixed:,} fixed")
    return path, meta


def load_variant(variant, ds, device, ckpt_dir):
    from models.turs_rrmt.model import TURSRRMT
    path = os.path.join(ckpt_dir, f"{ds['tag']}_{variant}.pt")
    if variant == "A0":
        path = path.replace(".pt", ".npz")
        z = np.load(path, allow_pickle=True)
        return dict(coef=z["coef"], intercept=z["intercept"],
                    P=z["P_train"], feats=dict(z["feats"].item()))
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = TURSRRMT(num_classes=ckpt["num_classes"], seq_len=ckpt["seq_len"],
                     M=ckpt["M"], J=ckpt["J"], topk=ckpt["topk"],
                     ppv_window=ckpt["ppv_window"], variant=ckpt["variant"],
                     seed=ckpt["seed"],
                     head_width=ckpt.get("head_width", 192))
    model.load_state_dict(ckpt["model_state_dict"])
    model.flavors.ref_q.copy_(torch.tensor(ckpt["flavor_ref"]["ref_q"]))
    model.flavors.ref_q_fine.copy_(torch.tensor(ckpt["flavor_ref"]["ref_q_fine"]))
    model.flavors.ref_lag.copy_(torch.tensor(ckpt["flavor_ref"]["ref_lag"]))
    model.flavors.fitted = True
    model.eval().to(device)
    return model


@torch.no_grad()
def predict(model, X, device, batch=256, want_aux=False):
    """Probabilities + (optional) routing aux. X: [N,1,L] numpy."""
    model.eval()
    ps, auxs = [], []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).float().to(device)
        if want_aux:
            logits, o = model(xb, return_aux=True)
        else:
            logits = model(xb)
            o = None
        ps.append(F.softmax(logits, 1).cpu().numpy())
        if want_aux:
            auxs.append(o)
    probs = np.concatenate(ps)
    return (probs, auxs) if want_aux else probs


# ---------------------------------------------------------------- A0 Ridge
def ridge_baseline(ds, ckpt_dir, force=False):
    """A0: fixed bank -> local PPV+strength -> global summary -> Ridge."""
    import hashlib
    from models.turs_rrmt.model import FixedPatternBank, LocalActivity
    from sklearn.linear_model import RidgeClassifier
    from sklearn.metrics import f1_score as _f1

    path = os.path.join(ckpt_dir, f"{ds['tag']}_A0.npz")
    meta_path = os.path.join(ckpt_dir, f"{ds['tag']}_A0_meta.json")
    if os.path.exists(path) and os.path.exists(meta_path) and not force:
        return path, json.load(open(meta_path))
    t0 = time.time()
    bank = FixedPatternBank(M=128, seed=SEED)
    act = LocalActivity(max(3, round(0.05 * ds["L"])))

    def feats(X):
        with torch.no_grad():
            xt = torch.from_numpy(X).float()
            R = bank(xt)
            A, S = act(R)
            P = torch.cat([A, torch.log1p(S)], 1)
            f = torch.cat([P.mean(-1), P.max(-1).values, P.std(-1)], 1)
        return f.numpy()

    Ftr, Fva, Fte = feats(ds["Xtr"]), feats(ds["Xva"]), feats(ds["Xte"])
    clf = RidgeClassifier(alpha=1.0).fit(Ftr, ds["y_train"])
    pred_te = clf.predict(Fte)
    pred_va = clf.predict(Fva)
    mf1 = _f1(ds["y_test"], pred_te, average="macro", zero_division=0,
              labels=list(range(ds["n_cls"])))
    np.savez_compressed(path, coef=clf.coef_, intercept=clf.intercept_,
                        P=Ftr, feats=np.array(
                            dict(train=(Ftr, ds["y_train"]), val=(Fva, ds["y_val"]),
                                 test=(Fte, ds["y_test"])), dtype=object))
    meta = dict(tag=ds["tag"], variant="A0", best_val_mf1=round(float(
        _f1(ds["y_val"], pred_va, average="macro", zero_division=0,
            labels=list(range(ds["n_cls"])))), 6),
        test_macro_f1=round(float(mf1), 6), elapsed_s=round(time.time() - t0, 1),
        seed=SEED, params_trainable=int(clf.coef_.size + clf.intercept_.size),
        params_fixed=0, note="Ridge on fixed-bank global PPV/strength features")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return path, meta


# ---------------------------------------------- A6/A7 routing interventions
@torch.no_grad()
def routing_intervention(model, ds, device, kind, seed=SEED, batch=256):
    """Evaluate frozen A4 with modified routing. kind: shuffled | fixed_j.

    shuffled: routing weights permuted across temporal positions/samples
    (destroys pattern->flavor correspondence; keeps marginal weights).
    fixed_j: routing replaced by one-hot flavor j.
    """
    from models.turs_rrmt.model import TURSRRMT
    assert model.variant == "A4"
    model.eval()
    g = torch.Generator().manual_seed(seed)
    outs = []
    for i in range(0, len(ds["Xte"]), batch):
        xb = torch.from_numpy(ds["Xte"][i:i + batch]).float().to(device)
        B, _, T = xb.shape
        # recompute internals
        o = model._features(xb, diag=False)
        h_orig = o["h"]
        logits_orig = model.head(h_orig)

        R = model.bank(xb)
        A, S = model.activity(R)
        P = torch.cat([A, torch.log1p(S)], 1)
        Tv = model.flavors(xb, window=model.ppv_window)
        Tp = min(P.shape[-1], Tv.shape[-1])
        P, Tv = P[..., :Tp], Tv[..., :Tp]
        w = torch.softmax(model.router(P.permute(0, 2, 1)), -1).permute(0, 2, 1)

        if kind == "shuffled":
            # destroy pattern->flavor correspondence: shuffle the flattened
            # [N*T, J] routing rows across positions/samples (deterministic g)
            flat = w.permute(0, 2, 1).reshape(-1, model.J)
            idx = torch.randperm(flat.shape[0], generator=g).to(flat.device)
            w_mod = flat[idx].reshape(B, Tp, model.J).permute(0, 2, 1)
        elif kind.startswith("fixed_"):
            j = int(kind.split("_")[1])
            w_mod = torch.zeros_like(w)
            w_mod[:, j, :] = 1.0
        else:
            raise ValueError(kind)

        top2 = w_mod.topk(min(model.topk, model.J), dim=1).indices
        mask = torch.zeros_like(w_mod).scatter_(1, top2, 1.0)
        kept = w_mod * mask * Tv
        rest = w_mod.sum(1, keepdim=True) - (w_mod * mask).sum(1, keepdim=True)
        F_T = model.transport_proj(torch.cat([kept, rest], dim=1))
        F_P = model.pattern_path(P.permute(0, 2, 1)).permute(0, 2, 1)
        F_all = torch.cat([F_T, F_P], dim=1)   # NB: not 'F' (shadows functional)
        mu, mx = F_all.mean(-1), F_all.max(-1).values
        sd = F_all.std(-1) if F_all.shape[-1] > 1 else torch.zeros_like(mu)
        logits = model.head(torch.cat([mu, mx, sd], dim=1))
        outs.append(F.softmax(logits, 1).cpu().numpy())
    return np.concatenate(outs)
