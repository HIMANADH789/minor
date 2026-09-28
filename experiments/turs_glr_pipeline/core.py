"""TURS-GLR pipeline core: repository audit, feature extraction with caches,
the A0-A9 ablation suite, evaluation, and paired significance testing.

Protocol identity: experiments/turs_rrmt/data.py::load_split (seed 42,
stratified splits, per-sample z-norm, [N, 1, T] float32). No dataset-specific
architecture anywhere: M_global=2048, M_local=128, J=4, router hidden 32,
softmax tau, block ridge. Only lambdas / temperature / fitted references vary
per dataset, and only via TRAIN/VAL data.

Variant definitions (fixed bank counts; fixed kernels are NOT trainable):
  A0  global stream only, scalar ridge (lam_g == lam_l effectively; one lambda)
  A1  local/routed stream only, scalar ridge
  A2  global + local concatenation, scalar ridge
  A3  global + local, BLOCK ridge          <-- PRIMARY MODEL
  A4  A3 + soft routing temperature tuned on validation
  A5  A3 + differentiable block ridge (router trained through the solve)
  A6  A3 with SHUFFLED routing at extraction time (train-fit intact)
  A7  A3 with UNIFORM routing at extraction time (train-fit intact)
  A8  global bank replaced by aeon MiniRocket, 2016 kernels (scale check)
  A9  M_local = 256 (larger local bank)
"""

import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

from experiments.turs_rrmt.data import load_split, SEED, DATASETS, NUM_CLASSES
from models.turs_glr.model import TURSGLR, count_params_glr
from models.turs_glr.block_ridge import (BlockRidge, fit_block_ridge_cv,
                                         DEFAULT_LAM_GRID)
from models.turs_glr.feature_blocks import (BlockStandardizer,
                                            local_block_dims,
                                            build_local_block)

RESULTS = os.path.join(ROOT, "results", "turs_glr")
CACHE = os.path.join(RESULTS, "cache")
TABLE_DIR = os.path.join(RESULTS, "tables")
FIG_DIR = os.path.join(RESULTS, "figures")
AUDIT_DIR = os.path.join(RESULTS, "audit")

VARIANTS = ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9"]
LAM_GRID = DEFAULT_LAM_GRID
TAU_GRID = [0.5, 1.0, 2.0]

# Architecture constants (identical for ALL datasets - spec compliance)
ARCH = dict(M_global=2048, M_local=128, J=4, router_hidden=32,
            lengths=(7, 11, 15, 23, 31), seed=SEED, tau=1.0)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()[:16]


def _mf1(y, pred, n_cls):
    from sklearn.metrics import f1_score
    return float(f1_score(y, pred, average="macro", zero_division=0,
                          labels=list(range(n_cls))))


def full_metrics(y, pred, n_cls):
    from experiments.turs_rrmt.train_eval import full_metrics as _fm
    return _fm(np.asarray(y), np.asarray(pred), n_cls)


# ============================================================
# Phase 0: repository audit
# ============================================================
def phase0_audit():
    os.makedirs(AUDIT_DIR, exist_ok=True)
    S = dict(
        date=time.strftime("%Y-%m-%d %H:%M:%S"),
        python=sys.version.split()[0],
        torch=torch.__version__,
        cuda=torch.cuda.is_available(),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        protocol="experiments/turs_rrmt/data.py::load_split (seed 42, stratified, "
                 "per-sample z-norm)",
        datasets=DATASETS,
        num_classes=NUM_CLASSES,
        architecture=dict(arch=ARCH, J_flavors=["standard W1 coarse",
                                                "tail-weighted W1",
                                                "fine-grid W1",
                                                "multilag drift"],
                          router="Linear(2M_l->32)->ReLU->Linear(32->J)->softmax(tau), "
                                 "trained soft from scratch",
                          readout="block-regularized closed-form ridge, "
                                  "(lam_g, lam_l) validation-selected"),
        reuse=dict(
            data="experiments/turs_rrmt/data.py",
            banks="models/turs_rrmt/model.py::build_pattern_bank/"
                  "FixedPatternBank/LocalActivity",
            flavors="models/turs_rrmt/model.py::TransportFlavors",
            views="models/tursnet.py::TransportBuilder",
            stats="src/diagnostics/statistics.py",
            calibration="src/diagnostics/calibration.py",
            perturb="src/diagnostics/perturb.py",
            desc_corr="experiments/turs_rrmt/diagnostics.py::_local_descriptors/_rho",
            minirocket="aeon MiniRocket (A8 only)"),
        variant_definitions={v: (v == "A3" and "PRIMARY: global+local block ridge")
                             or v for v in VARIANTS},
    )
    import platform
    S["platform"] = platform.platform()
    try:
        import git
        S["git_commit"] = git.Repo(ROOT).head.commit.hexsha[:12]
    except Exception:
        S["git_commit"] = _sha(os.path.join(ROOT, "models", "turs_glr", "model.py")) \
            + " (model-file hash; git metadata unavailable)"
    from src.diagnostics.statistics import dump_json
    dump_json(S, os.path.join(AUDIT_DIR, "repository_audit.json"))
    log("  audit written")
    return S


# ============================================================
# Feature extraction (cached, resumable)
# ============================================================
def _cache_paths(ds_tag, variant):
    d = os.path.join(CACHE, ds_tag)
    os.makedirs(d, exist_ok=True)
    return (os.path.join(d, f"features_{variant}.npz"),
            os.path.join(d, f"diag_{variant}.npz"))


def build_model(ds, device, variant="A3", M_global=None, M_local=None, tau=None):
    """Build the TURSGLR feature model for one dataset (same arch everywhere)."""
    kw = dict(ARCH)
    if M_global is not None:
        kw["M_global"] = M_global
    if M_local is not None:
        kw["M_local"] = M_local
    if tau is not None:
        kw["tau"] = tau
    torch.manual_seed(kw["seed"])
    np.random.seed(kw["seed"])
    model = TURSGLR(seq_len=ds["L"], M_global=kw["M_global"],
                    M_local=kw["M_local"], J=kw["J"], tau=kw["tau"],
                    seed=kw["seed"], lengths=kw["lengths"])
    model.fit_train_only(ds["Xtr"])
    return model.to(device)


def extract_features(ds_tag, ds, device, variant="A3", force=False,
                     log=log, M_global=None, M_local=None, tau=None,
                     routing_override=None, router_state=None,
                     save_diag=False, use_minirocket=False):
    """Extract (Zg, Zl) for train/val/test with disk cache.

    variant-specific knobs:
      A8: use_minirocket=True (aeon MiniRocket global stream)
      A9: M_local=256
      A6/A7: routing_override in {'shuffled','uniform'} at extraction time
      A5/A4: router_state = trained router weights (else fresh soft router)
    Returns dict with Zg_*, Zl_*, y_*, and model/meta info.
    """
    cp, diag_cp = _cache_paths(ds_tag, variant)
    if os.path.exists(cp) and not force:
        z = np.load(cp, allow_pickle=True)
        out = {k: z[k] for k in z.files}
        if isinstance(out.get("meta"), np.ndarray) and out["meta"].shape == ():
            out["meta"] = out["meta"].item()  # un-box 0-d object array
        out["cached"] = True
        log(f"  [{ds_tag}/{variant}] features cached")
        return out

    t0 = time.time()
    if use_minirocket:
        from models.turs_glr.global_pattern_bank import MiniRocketGlobal
        model = build_model(ds, device, variant, M_global=84, M_local=M_local or 128,
                            tau=tau)  # bank unused; keeps local stream identical
        mr = MiniRocketGlobal(n_kernels=2016, seed=SEED).fit(ds["Xtr"])
        Zg = {k: mr.transform(X) for k, X in
              [("tr", ds["Xtr"]), ("va", ds["Xva"]), ("te", ds["Xte"])]}
    else:
        model = build_model(ds, device, variant, M_global=M_global,
                            M_local=M_local, tau=tau)
        if router_state is not None:
            model.router.load_state_dict(router_state)
            log(f"  [{ds_tag}/{variant}] loaded trained router state")
        Zg = {k: model.extract_global(X, batch=128 if (M_global or ARCH["M_global"]) > 1024 else 256)
              for k, X in [("tr", ds["Xtr"]), ("va", ds["Xva"]), ("te", ds["Xte"])]}

    Zl = {}
    diag_test = None
    for k, X in [("tr", ds["Xtr"]), ("va", ds["Xva"]), ("te", ds["Xte"])]:
        if routing_override is not None:
            Zl[k], diag_test = _extract_local_override(
                model, X, routing_override, device, save_diag=(k == "te"))
        else:
            Zl[k] = model.extract_local(X, batch=128 if (M_local or ARCH["M_local"]) > 128 else 256)

    out = dict(Zg_tr=Zg["tr"], Zg_va=Zg["va"], Zg_te=Zg["te"],
               Zl_tr=Zl["tr"], Zl_va=Zl["va"], Zl_te=Zl["te"],
               y_tr=ds["y_train"], y_va=ds["y_val"], y_te=ds["y_test"])
    out["meta"] = dict(
        elapsed_s=round(time.time() - t0, 1),
        d_global=int(out["Zg_tr"].shape[1]), d_local=int(out["Zl_tr"].shape[1]),
        model_spec_hash=model.spec_hash() if hasattr(model, "spec_hash") else None,
        routing_override=routing_override,
        tau=model.tau,
        ppv_window=model.ppv_window,
        n_train=int(len(ds["y_train"])), n_val=int(len(ds["y_val"])),
        n_test=int(len(ds["y_test"])))
    # meta is a dict -> save as a 0-d object array; loader un-boxes it.
    meta = out.pop("meta")
    np.savez_compressed(cp, **out, meta=np.array(meta, dtype=object))
    out["meta"] = meta
    if save_diag and diag_test is not None:
        np.savez_compressed(diag_cp, **{k: v.cpu().numpy() if torch.is_tensor(v)
                                        else v for k, v in diag_test.items()})
    log(f"  [{ds_tag}/{variant}] extracted Zg{out['Zg_tr'].shape} "
        f"Zl{out['Zl_tr'].shape} in {out['meta']['elapsed_s']}s")
    return out


def _extract_local_override(model, X, mode, device, save_diag=False):
    """Local block under uniform / shuffled routing (extraction-time control)."""
    import torch
    g = torch.Generator().manual_seed(SEED)
    Zs, diags = [], []
    for i in range(0, len(X), 128):
        xb = torch.from_numpy(X[i:i + 128]).float().to(device)
        B = xb.shape[0]
        ls = model.local_stream(xb)
        P = ls["P"]
        Tv = model.flavors(xb, window=model.ppv_window)
        Tp = min(P.shape[-1], Tv.shape[-1])
        P, Tv = P[..., :Tp], Tv[..., :Tp]
        J = model.J
        if mode == "uniform":
            w = torch.full((B, J, Tp), 1.0 / J, device=xb.device)
        elif mode == "shuffled":
            # this is an extraction-time (no_grad) control: detach the router
            # output so the numpy conversion in build_local_block stays legal
            with torch.no_grad():
                w_real = model.router(P.permute(0, 2, 1)).permute(0, 2, 1)
            flat = w_real.detach().permute(0, 2, 1).reshape(-1, J)
            idx = torch.randperm(flat.shape[0], generator=g).to(flat.device)
            w = flat[idx].reshape(B, Tp, J).permute(0, 2, 1)
        else:
            raise ValueError(mode)
        A = ls["A"][..., :Tp]
        S_norm = ls["S_norm"][..., :Tp]
        U = w * Tv
        with torch.no_grad():
            Zl_np = build_local_block(U, w, A, S_norm)
        Zs.append(torch.from_numpy(Zl_np))
        if save_diag:
            diags.append(dict(w=w, Tv=Tv, U=U, A=A, x=xb))
    Z = torch.cat(Zs, 0).numpy().astype(np.float32)
    diag = None
    if save_diag and diags:
        diag = {k: torch.cat([d[k] for d in diags], 0) for k in diags[0]}
    return Z, diag


# ============================================================
# Ridge fitting helpers (single scalar lambda = A0/A1/A2 path)
# ============================================================
def fit_scalar_ridge(Zg, Zl, y, Zg_va, Zl_va, y_va, n_cls, log=log, tag=""):
    """A2-style: concatenate blocks, ONE lambda selected on validation.

    Uses the same BlockRidge solver with lam_g == lam_l (exact scalar ridge;
    validated against sklearn.linear_model.Ridge to ~1e-10).
    """
    tmpl = BlockRidge(n_cls)
    tmpl.precompute_gram(Zg, Zl, y)
    best, best_rec = None, None
    for lam in LAM_GRID:
        m = BlockRidge(n_cls, lam_g=lam, lam_l=lam)
        m._cache = tmpl._cache
        m.fit(None, None, None, use_cache=True)
        probs_va = m.predict_proba(Zg_va, Zl_va)
        mf1 = _mf1(y_va, probs_va.argmax(1), n_cls)
        from src.diagnostics.calibration import nll as nll_fn
        nl = nll_fn(probs_va, y_va)
        if best is None or (mf1, -nl) > (best, -best_rec["val_nll"]):
            best, best_rec = mf1, dict(lam=lam, val_mf1=mf1, val_nll=nl)
    final = BlockRidge(n_cls, lam_g=best_rec["lam"], lam_l=best_rec["lam"])
    final._cache = tmpl._cache
    final.fit(None, None, None, use_cache=True)
    log(f"    [{tag}] scalar ridge selected lam={best_rec['lam']:g} "
        f"val_mf1={best_rec['val_mf1']:.4f}")
    return final, best_rec


def standardize_and_fit(feat, n_cls, mode="block", log=log, tag=""):
    """Fit scaler (train-only) + ridge on one feature set.

    mode: 'block' -> two-lambda CV (A3); 'scalar' -> single lambda (A0/A1/A2).
    Returns (ridge, scaler, selection_record).
    """
    scaler = BlockStandardizer(feat["Zg_tr"].shape[1], feat["Zl_tr"].shape[1])
    scaler.fit(feat["Zg_tr"], feat["Zl_tr"])
    Zg_tr, Zl_tr = scaler.transform(feat["Zg_tr"], feat["Zl_tr"])
    Zg_va, Zl_va = scaler.transform(feat["Zg_va"], feat["Zl_va"])
    if mode == "block":
        ridge, sel = fit_block_ridge_cv(Zg_tr, Zl_tr, feat["y_tr"],
                                        Zg_va, Zl_va, feat["y_va"], n_cls,
                                        log=log)
        sel = dict(kind="block", **sel["selected"],
                   grid=sel["grid"], d_global=feat["Zg_tr"].shape[1],
                   d_local=feat["Zl_tr"].shape[1])
    else:
        ridge, selrec = fit_scalar_ridge(Zg_tr, Zl_tr, feat["y_tr"],
                                         Zg_va, Zl_va, feat["y_va"], n_cls,
                                         log=log, tag=tag)
        sel = dict(kind="scalar", **selrec)
    return ridge, scaler, sel


def evaluate(ridge, scaler, feat, n_cls):
    """Full locked-test + val evaluation incl. probabilities."""
    out = {}
    for split in ["te", "va"]:
        Zg, Zl = scaler.transform(feat[f"Zg_{split}"], feat[f"Zl_{split}"])
        probs = ridge.predict_proba(Zg, Zl)
        pred = probs.argmax(1)
        y = feat[f"y_{split}"]
        rec = full_metrics(y, pred, n_cls)
        from src.diagnostics.calibration import ece, brier, nll as nll_fn
        rec["nll"] = nll_fn(probs, y)
        rec["brier"] = brier(probs, y)
        rec["ece"] = ece(probs, y)
        out[split] = rec
        out[f"probs_{split}"] = probs
    return out


# ============================================================
# Router training (A4/A5): soft-from-scratch through block ridge
# ============================================================
def train_router(ds_tag, ds, device, feat, tau=1.0, epochs=25, lr=1e-3,
                 log=log, seed=SEED):
    """Train the soft router end-to-end: features are recomputed per epoch on
    minibatches; loss = CE over logits from a block-ridge solve on the batch's
    accumulated features; gradients reach the router through
    torch.linalg.solve (implicit differentiation).

    Implementation: the ridge is re-solved every epoch on the FULL train set
    (features recomputed with the current router), then the epoch's batch CE
    gradients flow into the router. To keep this tractable, features are
    recomputed per epoch on the full train set in chunks with grad enabled on
    the router path only (global bank is fixed and cached).
    """
    torch.manual_seed(seed)
    model = build_model(ds, device, tau=tau)
    model.extract_global(ds["Xtr"])  # warm-up to set buffers
    # extract_global returns float32 numpy -> tensors on device for the solve
    Zg_tr = torch.from_numpy(model.extract_global(ds["Xtr"])).float().to(device)
    Zg_va = torch.from_numpy(model.extract_global(ds["Xva"])).float().to(device)

    # cached "static" local ingredients per sample (banks are fixed; the
    # router is the only trainable part, so P/Tv/A/S are fixed too)
    # NOTE: static ingredients live on CPU and are moved per-chunk. Holding
    # all four temporals for CWRU-sized train sets on the 6GB GPU OOMs
    # (P/Tv/A/S are [N, 128/4, T'] tensors; 2642 x ~2048 x 1010 x 4B x 4 ~ 11GB).
    log(f"  [{ds_tag}/router] caching static local ingredients (cpu)")
    cache_tr, cache_va = {}, {}
    for split, X in [("tr", ds["Xtr"]), ("va", ds["Xva"])]:
        Ps, Tvs, As, Ss = [], [], [], []
        with torch.no_grad():
            for i in range(0, len(X), 128):
                xb = torch.from_numpy(X[i:i + 128]).float().to(device)
                ls = model.local_stream(xb)
                P = ls["P"]
                Tv = model.flavors(xb, window=model.ppv_window)
                Tp = min(P.shape[-1], Tv.shape[-1])
                Ps.append(P[..., :Tp].cpu()); Tvs.append(Tv[..., :Tp].cpu())
                As.append(ls["A"][..., :Tp].cpu())
                Ss.append(ls["S_norm"][..., :Tp].cpu())
        cache = dict(P=torch.cat(Ps), Tv=torch.cat(Tvs),
                     A=torch.cat(As), S=torch.cat(Ss))
        (cache_tr if split == "tr" else cache_va).update(cache)

    y_tr = torch.from_numpy(ds["y_train"]).long()
    y_va = ds["y_val"]
    n_cls = ds["n_cls"]
    opt = torch.optim.AdamW(model.router.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=epochs,
                                                steps_per_epoch=1)
    best_val, best_state, hist = -1, None, []
    N = cache_tr["P"].shape[0]

    def local_block_from_w(w, Tv, A, S):
        """Grad-safe local block: keep_torch keeps the autograd graph so the
        closed-form ridge solve differentiates into the router."""
        U = w * Tv
        return build_local_block(U, w, A, S, keep_torch=True)

    for ep in range(epochs):
        # 1) recompute train features under the current router (chunked,
        #    with grad so the ridge solve differentiates through)
        Zls, ws_all = [], []
        for i in range(0, N, 256):
            P = cache_tr["P"][i:i + 256].to(device)
            Tv = cache_tr["Tv"][i:i + 256].to(device)
            A = cache_tr["A"][i:i + 256].to(device)
            S = cache_tr["S"][i:i + 256].to(device)
            w = model.router(P.permute(0, 2, 1)).permute(0, 2, 1)
            Zl = local_block_from_w(w, Tv, A, S)
            Zls.append(Zl)
            ws_all.append(w.detach())
        # chunks are grad-carrying torch tensors (possibly cuda) -> cat directly
        Zl_tr = torch.cat(Zls, 0).to(device)
        del Zls

        # 2) standardize (train stats) and solve the block ridge
        mu, sd = Zg_tr.mean(0), torch.clip(Zg_tr.std(0), 1e-6, None)
        ml, sl = Zl_tr.mean(0), torch.clip(Zl_tr.std(0), 1e-6, None)
        Zg_s = (Zg_tr - mu) / sd
        Zl_s = (Zl_tr - ml) / sl
        ridge = solve_block_ridge_tensors(Zg_s, Zl_s, y_tr.to(device), n_cls)

        # 3) CE over train logits; gradients flow into the router through
        #    the chunked feature recomputation (each chunk's graph retained)
        opt.zero_grad()
        losses = []
        for i in range(0, N, 256):
            P = cache_tr["P"][i:i + 256].to(device)
            Tv = cache_tr["Tv"][i:i + 256].to(device)
            A = cache_tr["A"][i:i + 256].to(device)
            S = cache_tr["S"][i:i + 256].to(device)
            w = model.router(P.permute(0, 2, 1)).permute(0, 2, 1)
            Zl = local_block_from_w(w, Tv, A, S)
            Zl_s2 = (Zl - ml.detach()) / sl.detach()
            logits = torch.cat([Zg_s[i:i + 256],
                                Zl_s2], dim=1) @ ridge.W_star
            losses.append(F.cross_entropy(logits.float(),
                                          y_tr[i:i + 256].to(device)))
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.router.parameters(), 1.0)
        opt.step(); sched.step()

        # 4) validation macro-F1 with the updated router
        model.eval()
        with torch.no_grad():
            Zls_va = []
            for i in range(0, cache_va["P"].shape[0], 256):
                P = cache_va["P"][i:i + 256].to(device)
                w = model.router(P.permute(0, 2, 1)).permute(0, 2, 1)
                Zls_va.append(local_block_from_w(
                    w, cache_va["Tv"][i:i + 256].to(device),
                    cache_va["A"][i:i + 256].to(device),
                    cache_va["S"][i:i + 256].to(device)))
            Zl_va_t = torch.cat(Zls_va, 0).to(device)
            Zg_va_s = (Zg_va - mu) / sd
            Zl_va_s = (Zl_va_t - ml) / sl
            probs_va = F.softmax((torch.cat([Zg_va_s, Zl_va_s], 1)
                                  @ ridge.W_star).float(), 1).cpu().numpy()
        vm = _mf1(y_va, probs_va.argmax(1), n_cls)
        hist.append(dict(epoch=ep + 1, loss=float(loss), val_mf1=vm))
        if vm > best_val:
            best_val, best_state = vm, {k: v.detach().cpu().clone()
                                        for k, v in model.router.state_dict().items()}
        log(f"  [{ds_tag}/router tau={tau:g}] ep {ep+1}/{epochs} "
            f"loss={float(loss):.4f} val_mf1={vm:.4f} best={best_val:.4f}")

    model.router.load_state_dict(best_state)
    return model.router.state_dict(), dict(best_val_mf1=best_val, history=hist,
                                           tau=tau)


def solve_block_ridge_tensors(Zg, Zl, y, n_cls, lam_g=1.0, lam_l=1.0, eps=1e-6):
    """Block ridge on torch tensors (differentiable path for router training).

    W* = (Z^T Z + Lam)^{-1} Z^T Y ; gradients flow via torch.linalg.solve.
    """
    Dg = Zg.shape[1]
    Z = torch.cat([Zg, Zl], dim=1)
    Y = F.one_hot(y.long(), n_cls).to(Z.dtype)
    A = Z.T @ Z
    reg = torch.cat([torch.full((Dg,), float(lam_g), device=Z.device),
                     torch.full((Z.shape[1] - Dg,), float(lam_l), device=Z.device)])
    A = A + torch.diag(reg + eps)
    W = torch.linalg.solve(A, Z.T @ Y)
    br = BlockRidge(n_cls, lam_g=lam_g, lam_l=lam_l)
    br.W_star = W
    return br


# ============================================================
# Phase: run all ablations for one dataset
# ============================================================
def run_dataset_ablations(ds_tag, device, force=False, variants=None,
                          log=log, train_only=False):
    """Run A0-A9 for one dataset; returns results dict (also written to disk)."""
    variants = variants or VARIANTS
    ds = load_split(ds_tag)
    n_cls = ds["n_cls"]
    os.makedirs(os.path.join(RESULTS, ds_tag), exist_ok=True)
    out_path = os.path.join(RESULTS, ds_tag, "ablation_results.json")
    if os.path.exists(out_path) and not force:
        log(f"  [{ds_tag}] ablation results cached")
        return json.load(open(out_path))

    results = dict(dataset=ds_tag, variants={}, timing={})

    # ---------- A0-A3 from ONE base extraction (same features) ----------
    base = extract_features(ds_tag, ds, device, "A3", force=force, log=log)
    t0 = time.time()
    ridge_A3, scaler_A3, sel_A3 = standardize_and_fit(base, n_cls, "block",
                                                      log=log, tag="A3")
    ev_A3 = evaluate(ridge_A3, scaler_A3, base, n_cls)
    results["variants"]["A3"] = dict(
        test={k: v for k, v in ev_A3["te"].items()},
        val={k: v for k, v in ev_A3["va"].items()},
        selection=sel_A3,
        coef_norms=ridge_A3.coef_norms,
        params=dict(trainable=sum(p.numel() for p in ridge_A3.parameters()
                                  if p.requires_grad) if False else
                    int(ridge_A3.W_star.numel()), fixed=0,
                    note="ridge coefficients only; all kernels fixed"))
    results["timing"]["A3"] = round(time.time() - t0, 1)
    log(f"  [{ds_tag}/A3] PRIMARY test MF1={ev_A3['te']['macro_f1']:.4f} "
        f"(lam_g={sel_A3['lam_g']:g}, lam_l={sel_A3['lam_l']:g})")

    # A0: global only (scalar ridge on Zg) - reuse base global features
    t0 = time.time()
    featA0 = dict(base)
    featA0["Zl_tr"] = np.zeros((len(base["y_tr"]), 0), np.float32)
    featA0["Zl_va"] = np.zeros((len(base["y_va"]), 0), np.float32)
    featA0["Zl_te"] = np.zeros((len(base["y_te"]), 0), np.float32)
    ridge_A0, scaler_A0, sel_A0 = _fit_single_block(featA0, n_cls, log=log)
    ev_A0 = evaluate(ridge_A0, scaler_A0, featA0, n_cls)
    results["variants"]["A0"] = dict(test=ev_A0["te"], val=ev_A0["va"],
                                     selection=sel_A0)
    results["timing"]["A0"] = round(time.time() - t0, 1)

    # A1: local only
    t0 = time.time()
    featA1 = dict(base)
    featA1["Zg_tr"] = np.zeros((len(base["y_tr"]), 0), np.float32)
    featA1["Zg_va"] = np.zeros((len(base["y_va"]), 0), np.float32)
    featA1["Zg_te"] = np.zeros((len(base["y_te"]), 0), np.float32)
    ridge_A1, scaler_A1, sel_A1 = _fit_single_block(featA1, n_cls, log=log)
    ev_A1 = evaluate(ridge_A1, scaler_A1, featA1, n_cls)
    results["variants"]["A1"] = dict(test=ev_A1["te"], val=ev_A1["va"],
                                     selection=sel_A1)
    results["timing"]["A1"] = round(time.time() - t0, 1)

    # A2: concat + scalar ridge (same features as A3)
    t0 = time.time()
    ridge_A2, scaler_A2, sel_A2 = standardize_and_fit(base, n_cls, "scalar",
                                                      log=log, tag="A2")
    ev_A2 = evaluate(ridge_A2, scaler_A2, base, n_cls)
    results["variants"]["A2"] = dict(test=ev_A2["te"], val=ev_A2["va"],
                                     selection=sel_A2)
    results["timing"]["A2"] = round(time.time() - t0, 1)
    log(f"  [{ds_tag}/A2] scalar-ridge test MF1={ev_A2['te']['macro_f1']:.4f}")

    # cache A2/A3 predictions for paired tests
    _save_predictions(ds_tag, "A2", ev_A2, base["y_te"], base["y_va"])
    _save_predictions(ds_tag, "A3", ev_A3, base["y_te"], base["y_va"])
    _save_predictions(ds_tag, "A0", ev_A0, base["y_te"], base["y_va"])
    _save_predictions(ds_tag, "A1", ev_A1, base["y_te"], base["y_va"])

    # ---------- A4: temperature-tuned soft router (validation-only tau) ----------
    if "A4" in variants:
        t0 = time.time()
        best_tau, best_router, best_rec = None, None, None
        for tau in TAU_GRID:
            router_state, rec = train_router(ds_tag, ds, device, base, tau=tau,
                                             epochs=15, log=log)
            if best_rec is None or rec["best_val_mf1"] > best_rec["best_val_mf1"]:
                best_tau, best_router, best_rec = tau, router_state, rec
        log(f"  [{ds_tag}/A4] selected tau={best_tau:g} "
            f"(val_mf1={best_rec['best_val_mf1']:.4f})")
        featA4 = extract_features(ds_tag, ds, device, "A4", force=True, log=log,
                                  tau=best_tau, router_state=best_router)
        ridge_A4, scaler_A4, sel_A4 = standardize_and_fit(featA4, n_cls, "block",
                                                          log=log, tag="A4")
        ev_A4 = evaluate(ridge_A4, scaler_A4, featA4, n_cls)
        results["variants"]["A4"] = dict(
            test=ev_A4["te"], val=ev_A4["va"], selection=sel_A4,
            tau=best_tau, router_training=best_rec)
        results["timing"]["A4"] = round(time.time() - t0, 1)
        _save_predictions(ds_tag, "A4", ev_A4, featA4["y_te"], featA4["y_va"])
        # persist router for A5/resume
        torch.save(best_router, os.path.join(CACHE, ds_tag, "router_A4.pt"))

    # ---------- A5: differentiable block ridge (end-to-end router training) ----
    if "A5" in variants:
        # A5 IS the end-to-end differentiable path; A4's router training is the
        # same machinery. For honesty: A5 = A4 router trained with the BEST tau,
        # evaluated WITHOUT re-tuning (documented in the report).
        t0 = time.time()
        featA5 = extract_features(ds_tag, ds, device, "A5", force=True, log=log,
                                  tau=best_tau if "A4" in results["variants"] else 1.0,
                                  router_state=best_router
                                  if "A4" in results["variants"] else None)
        ridge_A5, scaler_A5, sel_A5 = standardize_and_fit(featA5, n_cls, "block",
                                                          log=log, tag="A5")
        ev_A5 = evaluate(ridge_A5, scaler_A5, featA5, n_cls)
        results["variants"]["A5"] = dict(
            test=ev_A5["te"], val=ev_A5["va"], selection=sel_A5,
            note="same differentiable-router machinery as A4 without tau re-tuning")
        results["timing"]["A5"] = round(time.time() - t0, 1)
        _save_predictions(ds_tag, "A5", ev_A5, featA5["y_te"], featA5["y_va"])

    # ---------- A6/A7: routing interventions at extraction time ----------
    for vname, mode in [("A6", "shuffled"), ("A7", "uniform")]:
        if vname not in variants:
            continue
        t0 = time.time()
        featX = extract_features(ds_tag, ds, device, vname, force=True, log=log,
                                 routing_override=mode)
        ridge_X, scaler_X, sel_X = standardize_and_fit(featX, n_cls, "block",
                                                       log=log, tag=vname)
        ev_X = evaluate(ridge_X, scaler_X, featX, n_cls)
        results["variants"][vname] = dict(
            test=ev_X["te"], val=ev_X["va"], selection=sel_X,
            intervention=f"{mode} routing at feature construction "
                         "(router untrained; everything else identical)")
        results["timing"][vname] = round(time.time() - t0, 1)
        _save_predictions(ds_tag, vname, ev_X, featX["y_te"], featX["y_va"])

    # ---------- A8: MiniRocket global stream ----------
    if "A8" in variants:
        t0 = time.time()
        featA8 = extract_features(ds_tag, ds, device, "A8", force=True, log=log,
                                  use_minirocket=True)
        ridge_A8, scaler_A8, sel_A8 = standardize_and_fit(featA8, n_cls, "block",
                                                          log=log, tag="A8")
        ev_A8 = evaluate(ridge_A8, scaler_A8, featA8, n_cls)
        results["variants"]["A8"] = dict(
            test=ev_A8["te"], val=ev_A8["va"], selection=sel_A8,
            note="global stream = aeon MiniRocket (2016 kernels, PPV), local "
                 "stream unchanged, block ridge")
        results["timing"]["A8"] = round(time.time() - t0, 1)
        _save_predictions(ds_tag, "A8", ev_A8, featA8["y_te"], featA8["y_va"])

    # ---------- A9: larger local bank ----------
    if "A9" in variants:
        t0 = time.time()
        featA9 = extract_features(ds_tag, ds, device, "A9", force=True, log=log,
                                  M_local=256)
        ridge_A9, scaler_A9, sel_A9 = standardize_and_fit(featA9, n_cls, "block",
                                                          log=log, tag="A9")
        ev_A9 = evaluate(ridge_A9, scaler_A9, featA9, n_cls)
        results["variants"]["A9"] = dict(test=ev_A9["te"], val=ev_A9["va"],
                                         selection=sel_A9, M_local=256)
        results["timing"]["A9"] = round(time.time() - t0, 1)
        _save_predictions(ds_tag, "A9", ev_A9, featA9["y_te"], featA9["y_va"])

    # complexity record from the base model
    model = build_model(ds, device)
    tr_params, fx_params = count_params_glr(model)
    dims = local_block_dims(ARCH["M_local"], ARCH["J"])
    results["complexity"] = dict(
        trainable_params=int(tr_params), fixed_kernel_params=int(fx_params),
        n_global_kernels=ARCH["M_global"], n_local_kernels=ARCH["M_local"],
        d_global=int(base["Zg_tr"].shape[1]), d_local=int(base["Zl_tr"].shape[1]),
        local_dims=dims,
        ridge_matrix=(int(base["Zg_tr"].shape[1] + base["Zl_tr"].shape[1]),
                      int(base["Zg_tr"].shape[1] + base["Zl_tr"].shape[1])),
        n_train=int(len(ds["y_train"])))

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    return results


def _fit_single_block(feat, n_cls, log=log):
    """A0/A1 path: only one block is non-empty; scalar ridge on it."""
    has_global = feat["Zg_tr"].shape[1] > 0
    if has_global:
        Ztr, Zva = feat["Zg_tr"], feat["Zg_va"]
        d = Ztr.shape[1]
        scaler = BlockStandardizer(d, 0).fit(feat["Zg_tr"], feat["Zl_tr"])
    else:
        Ztr, Zva = feat["Zl_tr"], feat["Zl_va"]
        d = Ztr.shape[1]
        scaler = BlockStandardizer(0, d)
        scaler.g_mu = np.zeros(0); scaler.g_sd = np.zeros(0)
        scaler.l_mu = feat["Zl_tr"].mean(0)
        scaler.l_sd = np.clip(feat["Zl_tr"].std(0), 1e-6, None)
    best, best_rec = None, None
    tmpl = BlockRidge(n_cls)
    if has_global:
        tmpl.precompute_gram(feat["Zg_tr"], np.zeros((len(feat["y_tr"]), 0)), feat["y_tr"])
    else:
        tmpl.precompute_gram(np.zeros((len(feat["y_tr"]), 0)), feat["Zl_tr"], feat["y_tr"])
    from src.diagnostics.calibration import nll as nll_fn
    for lam in LAM_GRID:
        m = BlockRidge(n_cls, lam_g=lam if has_global else 1.0,
                       lam_l=1.0 if has_global else lam)
        m._cache = tmpl._cache
        m.fit(None, None, None, use_cache=True)
        probs_va = m.predict_proba(feat["Zg_va"] if has_global else np.zeros((len(feat["y_va"]), 0)),
                                   feat["Zl_va"] if not has_global else np.zeros((len(feat["y_va"]), 0)))
        mf1 = _mf1(feat["y_va"], probs_va.argmax(1), n_cls)
        nl = nll_fn(probs_va, feat["y_va"])
        if best is None or (mf1, -nl) > (best, -best_rec["val_nll"]):
            best, best_rec = mf1, dict(lam=lam, val_mf1=mf1, val_nll=nl)
    ridge = BlockRidge(n_cls, lam_g=best_rec["lam"] if has_global else 1.0,
                       lam_l=1.0 if has_global else best_rec["lam"])
    ridge._cache = tmpl._cache
    ridge.fit(None, None, None, use_cache=True)
    log(f"    [single-block] lam={best_rec['lam']:g} val_mf1={best_rec['val_mf1']:.4f}")
    return ridge, scaler, best_rec


def _save_predictions(ds_tag, variant, ev, y_te, y_va):
    d = os.path.join(RESULTS, ds_tag)
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, f"preds_{variant}.npz"),
                        probs_te=ev["probs_te"], probs_va=ev["probs_va"],
                        y_te=np.asarray(y_te), y_va=np.asarray(y_va))


# ============================================================
# Phase: paired significance for key comparisons
# ============================================================
def paired_significance(ds_tag, n_cls, log=log):
    """A3 vs A2/A0/A1/A7 paired tests on the same test samples."""
    from src.diagnostics.statistics import (bootstrap_diff_ci, mcnemar,
                                            paired_permutation_test,
                                            cohens_d_paired)
    rows = []
    preds = {}
    for v in ["A0", "A1", "A2", "A3", "A4", "A6", "A7", "A8", "A9"]:
        fp = os.path.join(RESULTS, ds_tag, f"preds_{v}.npz")
        if os.path.exists(fp):
            preds[v] = np.load(fp)
    if "A3" not in preds:
        return rows
    y = preds["A3"]["y_te"]
    pA3 = preds["A3"]["probs_te"]
    cA3 = (pA3.argmax(1) == y).astype(float)
    nA3 = -np.log(pA3[np.arange(len(y)), y] + 1e-12)
    for v in ["A0", "A1", "A2", "A4", "A6", "A7", "A8", "A9"]:
        if v not in preds:
            continue
        pv = preds[v]["probs_te"]
        yv = preds[v]["y_te"]
        assert np.array_equal(yv, y), f"split mismatch A3 vs {v}"
        cv = (pv.argmax(1) == y).astype(float)
        nv = -np.log(pv[np.arange(len(y)), y] + 1e-12)
        d, lo, hi = bootstrap_diff_ci(cA3, cv, paired=True)
        chi2, pm = mcnemar(cA3, cv)
        obs, pp = paired_permutation_test(nA3, nv)
        mf1_a3 = _mf1(y, pA3.argmax(1), n_cls)
        mf1_v = _mf1(y, pv.argmax(1), n_cls)
        rows.append(dict(
            dataset=ds_tag, comparison=f"A3_vs_{v}",
            mf1_a3=round(mf1_a3, 4), mf1_other=round(mf1_v, 4),
            mf1_delta=round(mf1_a3 - mf1_v, 4),
            acc_diff=round(float(d), 4), acc_ci_lo=round(float(lo), 4),
            acc_ci_hi=round(float(hi), 4),
            mcnemar_p=round(float(pm), 6),
            nll_perm_p=round(float(pp), 6),
            nll_delta=round(float(obs), 6),
            cohens_d=round(float(cohens_d_paired(cA3, cv)), 4),
            n_test=int(len(y))))
    return rows
