"""TURS-RRMT-V2 Full Automated Pipeline.

Stages:
  0. Repository audit
  1. V1 baseline verification
  2. V2 representation extraction (frozen V1)
  3. V2-A differentiable ridge (R2)
  4. V2-B class-weighted ridge (R3)
  5. V2-C soft routing ridge (R4)
  6. V2-D kernelized routed readout (R5)
  7. V2-E larger pattern bank (R6, R7)
  8. Diagnostics D1-D14
  9. Statistical significance + FDR
  10. Figures + tables + report

Usage:
  python experiments/run_turs_rrmt_v2_full.py --all
  python experiments/run_turs_rrmt_v2_full.py --dataset ECG5000_UNBAL
  python experiments/run_turs_rrmt_v2_full.py --variant R2
  ... --resume --force --train-only --diagnostics-only --report-only
"""

import argparse
import json
import os
import sys
import time
import hashlib
import copy

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from experiments.turs_rrmt.data import load_split, SEED, DATASETS, NUM_CLASSES
from experiments.turs_rrmt.train_eval import (
    full_metrics, load_variant, predict as v1_predict, _mf1
)
from models.turs_rrmt_v2.model import TURSRRMTV2, count_params
from models.turs_rrmt_v2.readout import KernelRidgeReadout
from src.diagnostics.statistics import HypothesisRegistry, dump_json

# === Configuration ===
RESULTS = os.path.join(ROOT, "results", "turs_rrmt_v2")
CKPT_DIR = os.path.join(ROOT, "checkpoints", "turs_rrmt_v2")
TABLE_DIR = os.path.join(RESULTS, "tables")
FIG_DIR = os.path.join(RESULTS, "figures")
AUDIT_DIR = os.path.join(RESULTS, "audit")
V1_CKPT = os.path.join(ROOT, "results", "turs_rrmt", "checkpoints")

VARIANTS = ["R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7"]
LAM_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
TAU_GRID = [0.5, 1.0, 2.0]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()[:16]


# ============================================================
# Phase 0: Repository audit
# ============================================================
def phase0_audit():
    log("PHASE 0: Repository audit")
    os.makedirs(AUDIT_DIR, exist_ok=True)
    
    # Check what exists
    audit = dict(
        date=time.strftime("%Y-%m-%d %H:%M:%S"),
        torch_version=torch.__version__,
        cuda=torch.cuda.is_available(),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        available_solvers=["torch.linalg.solve (differentiable)",
                          "torch.linalg.cholesky (differentiable)",
                          "sklearn.linear_model.RidgeClassifier",
                          "sklearn.kernel_ridge.KernelRidge"],
        available_kernels=["sklearn.metrics.pairwise.linear_kernel",
                          "sklearn.metrics.pairwise.rbf_kernel",
                          "src.kernels.product_kernel.ProductKernel"],
        project_modules=dict(
            statistics="src.diagnostics.statistics",
            perturb="src.diagnostics.perturb",
            safe_eigh="src.utils.safe_eigh",
            geodesic="src.classifiers.geodesic_prototype_classifier",
            swrst_kernel="swrst.kernel.builder",
        ),
        v1_source="models/turs_rrmt/model.py",
        v1_readout_source="experiments/turs_rrmt/train_eval.py",
    )
    
    with open(os.path.join(AUDIT_DIR, "repository_audit.json"), "w") as f:
        json.dump(audit, f, indent=2)
    log("  audit written")


# ============================================================
# Phase 1: V1 baseline verification
# ============================================================
def phase1_v1_baseline():
    log("PHASE 1: V1 baseline verification")
    os.makedirs(os.path.join(RESULTS, "baselines"), exist_ok=True)
    
    v1_results = {}
    for ds in DATASETS:
        fp = os.path.join(ROOT, "results", "turs_rrmt", ds, "full_results.json")
        if os.path.exists(fp):
            R = json.load(open(fp))
            v = R.get("variants", {})
            a4 = v.get("A4", {}).get("test", {})
            v1_results[ds] = dict(
                test_mf1=a4.get("macro_f1", 0),
                test_acc=a4.get("accuracy", 0),
                test_bal_acc=a4.get("balanced_accuracy", 0),
                params_trainable=R.get("meta", {}).get("A4", {}).get("params_trainable", 0),
                params_fixed=R.get("meta", {}).get("A4", {}).get("params_fixed", 0),
            )
            log(f"  {ds}: V1 A4 test MF1={a4.get('macro_f1', 0):.4f}")
        else:
            log(f"  {ds}: V1 results not found")
    
    with open(os.path.join(RESULTS, "baselines", "rrmt_v1_results.json"), "w") as f:
        json.dump(v1_results, f, indent=2)
    
    return v1_results


# ============================================================
# Phase 2: V2 representation extraction
# ============================================================
def extract_v2_representation(ds_tag, device, variant='R2', M=128):
    """Extract V1 representation frozen features for all splits.
    
    Uses the original TURSRRMTV2 model with M=128, but loads V1 A4 checkpoint.
    Since V1 A4 has transport_proj with 5 channels (J+1) and V2 uses J=4,
    we use a V1-extraction approach: build model with correct channel count.
    """
    cache_dir = os.path.join(RESULTS, ds_tag, "representation")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"v1_features_M{M}.npz")
    
    cache_path_routing = os.path.join(cache_dir, f"v1_routing_M{M}.npz")
    
    # FIX: require BOTH caches. The old check returned features-only when the
    # routing cache was missing, so R5 crashed on KeyError('r_Tv_tr').
    if os.path.exists(cache_path) and os.path.exists(cache_path_routing):
        log(f"  {ds_tag}: loading cached V1 features + routing")
        data = np.load(cache_path)
        result = dict(Xtr=data["Xtr"], ytr=data["ytr"],
                      Xva=data["Xva"], yva=data["yva"],
                      Xte=data["Xte"], yte=data["yte"])
        rd = np.load(cache_path_routing)
        for k in rd.files:
            result[f"r_{k}"] = rd[k]
        return result
    if os.path.exists(cache_path):
        log(f"  {ds_tag}: features cached but routing cache missing -> re-extracting")
    
    ds = load_split(ds_tag)
    
    # Use original V1 model (TURSRRMT) for feature extraction
    from models.turs_rrmt.model import TURSRRMT as V1Model
    
    v1_path = os.path.join(V1_CKPT, f"{ds_tag}_A4.pt")
    if os.path.exists(v1_path):
        ckpt = torch.load(v1_path, map_location="cpu", weights_only=False)
        if ckpt["M"] != M:
            # FIX: R6/R7 request larger banks -> checkpoint shapes don't match.
            # Build a fresh V1 model at the requested M: the pattern bank is
            # seed-fixed (meaningful), learned router/projections stay untrained.
            log(f"  {ds_tag}: ckpt M={ckpt['M']} != requested M={M} -> fresh model (untrained learned parts)")
            torch.manual_seed(SEED)
            model_v1 = V1Model(
                num_classes=ds["n_cls"], seq_len=ds["L"], M=M,
                J=4, topk=2, variant="A4", seed=SEED)
            # FIX: fit flavor references on TRAIN (old fallback left them at zero)
            torch.manual_seed(SEED)
            model_v1.flavors.fit_reference(torch.from_numpy(ds["Xtr"]).float())
            model_v1.flavors.fitted = True
        else:
            model_v1 = V1Model(
                num_classes=ckpt["num_classes"], seq_len=ckpt["seq_len"],
                M=ckpt["M"], J=ckpt["J"], topk=ckpt["topk"],
                ppv_window=ckpt["ppv_window"], variant=ckpt["variant"],
                seed=ckpt["seed"])
            model_v1.load_state_dict(ckpt["model_state_dict"])
            model_v1.flavors.ref_q.copy_(torch.tensor(ckpt["flavor_ref"]["ref_q"]))
            model_v1.flavors.ref_q_fine.copy_(torch.tensor(ckpt["flavor_ref"]["ref_q_fine"]))
            model_v1.flavors.ref_lag.copy_(torch.tensor(ckpt["flavor_ref"]["ref_lag"]))
            model_v1.flavors.fitted = True
            log(f"  {ds_tag}: loaded V1 A4 checkpoint")
    else:
        log(f"  {ds_tag}: no V1 checkpoint, using random init")
        torch.manual_seed(SEED)
        model_v1 = V1Model(
            num_classes=ds["n_cls"], seq_len=ds["L"], M=M,
            J=4, topk=2, variant="A4", seed=SEED)
        # FIX: fit flavor references on TRAIN (old fallback left them at zero)
        torch.manual_seed(SEED)
        model_v1.flavors.fit_reference(torch.from_numpy(ds["Xtr"]).float())
        model_v1.flavors.fitted = True
    
    model_v1.eval().to(device)
    
    # FIX: batch the forward pass. Full-split forwards allocate [B, M, T'] bank
    # activations; at M=512 on CWRU (T~1024) that OOMs a 6GB GPU.
    BS = 64 if M >= 512 else (128 if M >= 256 else 256)
    features = {}
    routing_feats = {"Tv": [], "w": [], "Psc": []}  # for R5 kernel
    for split, X, y, ykey in [("Xtr", ds["Xtr"], ds["y_train"], "ytr"),
                               ("Xva", ds["Xva"], ds["y_val"], "yva"),
                               ("Xte", ds["Xte"], ds["y_test"], "yte")]:
        hs, ws, Tvs, psc = [], [], [], []
        for i in range(0, len(X), BS):
            xb = torch.from_numpy(X[i:i + BS]).float().to(device)
            with torch.no_grad():
                o = model_v1._features(xb, diag=False)
                hs.append(o["h"].cpu().numpy())
                # Re-derive routing intermediates: V1 _features(diag=False)
                # returns neither Tv nor P (o["w"] IS returned for variant A4).
                R = model_v1.bank(xb)
                A_, S_ = model_v1.activity(R)
                P_ = torch.cat([A_, torch.log1p(S_)], dim=1)
                Tv_ = model_v1.flavors(xb, window=model_v1.ppv_window)
                Tp = o["F_P"].shape[-1]  # temporal dim aligned inside _features
                P_, Tv_ = P_[..., :Tp], Tv_[..., :Tp]
                if o.get("w") is not None:
                    ws.append(o["w"].cpu().numpy())
                    Tvs.append(Tv_.cpu().numpy())
                    # P is [B, 2M, T] -> too large to cache; keep only the two
                    # scalars the R5 kernel actually consumes.
                    psc.append(torch.stack([P_.mean(-1).mean(-1),
                                            P_.max(-1).values.mean(-1)], dim=1).cpu().numpy())
                del R, A_, S_, P_, Tv_
        features[split] = np.concatenate(hs, 0)
        features[ykey] = y
        if ws:
            routing_feats["w"].append(np.concatenate(ws, 0))
            routing_feats["Tv"].append(np.concatenate(Tvs, 0))
            routing_feats["Psc"].append(np.concatenate(psc, 0))
    
    np.savez_compressed(cache_path,
                        Xtr=features["Xtr"], ytr=features["ytr"],
                        Xva=features["Xva"], yva=features["yva"],
                        Xte=features["Xte"], yte=features["yte"])
    
    # Cache routing intermediates for R5
    if routing_feats["Tv"]:
        np.savez_compressed(cache_path_routing,
                            Tv_tr=routing_feats["Tv"][0],
                            w_tr=routing_feats["w"][0],
                            Psc_tr=routing_feats["Psc"][0],
                            Tv_va=routing_feats["Tv"][1],
                            w_va=routing_feats["w"][1],
                            Psc_va=routing_feats["Psc"][1],
                            Tv_te=routing_feats["Tv"][2],
                            w_te=routing_feats["w"][2],
                            Psc_te=routing_feats["Psc"][2])
        for k in ["Tv", "w", "Psc"]:
            features[f"r_{k}_tr"] = routing_feats[k][0]
            features[f"r_{k}_va"] = routing_feats[k][1]
            features[f"r_{k}_te"] = routing_feats[k][2]
    
    log(f"  {ds_tag}: extracted V1 features {features['Xtr'].shape} (+routing)")
    return features


# ============================================================
# Phase 3-7: Run V2 variants
# ============================================================
def run_v2_variant(ds_tag, variant, features, ds_data, device):
    """Run a single V2 variant on one dataset."""
    tag = f"{ds_tag}_{variant}"
    result_path = os.path.join(RESULTS, ds_tag, f"{variant}_results.json")
    
    os.makedirs(os.path.join(RESULTS, ds_tag), exist_ok=True)
    os.makedirs(os.path.join(CKPT_DIR, ds_tag), exist_ok=True)
    
    Xtr, ytr = features["Xtr"], features["ytr"]
    Xva, yva = features["Xva"], features["yva"]
    Xte, yte = features["Xte"], features["yte"]
    n_cls = ds_data["n_cls"]
    
    t0 = time.time()
    result = dict(dataset=ds_tag, variant=variant, n_cls=n_cls)
    
    if variant == "R0":
        # V1 MLP head - use existing V1 result directly
        log(f"  [{tag}] V1 MLP head (using existing V1 A4 result)")
        v1_fp = os.path.join(ROOT, "results", "turs_rrmt", ds_tag, "full_results.json")
        if os.path.exists(v1_fp):
            v1_R = json.load(open(v1_fp))
            a4_test = v1_R["variants"]["A4"]["test"]
            result["test"] = a4_test
            result["val_mf1"] = v1_R["variants"]["A4"]["val"]["macro_f1"]
            m4 = v1_R.get("meta", {}).get("A4", {})
            result["params"] = dict(
                trainable=m4.get("params_trainable", 0),
                fixed=m4.get("params_fixed", 0))
        else:
            result["test"] = dict(macro_f1=0, accuracy=0)
            result["val_mf1"] = 0
        
    elif variant == "R1":
        # Frozen Ridge (post-hoc baseline)
        log(f"  [{tag}] Frozen Ridge on V1 features")
        from sklearn.linear_model import RidgeClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        
        best_lam, best_mf1 = None, -1
        for lam in LAM_GRID:
            # FIX: standardize with TRAIN stats (raw pooled features are
            # badly scaled for closed-form ridge / RidgeClassifierCV-style solvers)
            clf = make_pipeline(StandardScaler().fit(Xtr),
                                RidgeClassifier(alpha=lam))
            clf.fit(Xtr, ytr)
            pred_va = clf.predict(Xva)
            vm = _mf1(yva, pred_va, n_cls)
            if vm > best_mf1:
                best_mf1, best_lam = vm, lam
        
        clf = make_pipeline(StandardScaler().fit(Xtr),
                            RidgeClassifier(alpha=best_lam))
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        result["test"] = full_metrics(yte, pred, n_cls)
        result["val_mf1"] = best_mf1
        result["selected_lambda"] = best_lam
        # clf is a Pipeline -> grab the RidgeClassifier step for param count
        _ridge = clf.steps[-1][1]
        result["params"] = dict(trainable=int(_ridge.coef_.size + _ridge.intercept_.size), fixed=0)
        
    elif variant in ("R2", "R3"):
        # Differentiable Ridge / Class-weighted Ridge
        readout_type = "ridge" if variant == "R2" else "class_weighted_ridge"
        log(f"  [{tag}] {readout_type} readout")
        
        model = TURSRRMTV2(
            num_classes=n_cls, seq_len=ds_data["L"],
            variant=variant, readout_type=readout_type, seed=SEED
        ).to(device)
        
        # Fit ridge on cached features
        Z_tr = torch.from_numpy(Xtr).float().to(device)
        Y_tr = torch.from_numpy(ytr).long().to(device)
        Z_va = torch.from_numpy(Xva).float().to(device)
        Y_va = torch.from_numpy(yva).long().to(device)
        Z_te = torch.from_numpy(Xte).float().to(device)
        
        best_lam, best_mf1 = model.fit_ridge_readout(
            Xtr, ytr, Xva, yva, lam_grid=LAM_GRID)
        
        # Predict
        probs = model.predict_with_ridge(Z_te)
        pred = probs.argmax(-1).cpu().numpy()
        
        result["test"] = full_metrics(yte, pred, n_cls)
        result["val_mf1"] = best_mf1
        result["selected_lambda"] = best_lam
        trainable = int(model.readout.W_star.numel()) if hasattr(model.readout, 'W_star') else 0
        result["params"] = dict(trainable=trainable, fixed=0)
        
    elif variant == "R4":
        # Soft routing + Ridge
        log(f"  [{tag}] Soft routing + Ridge")
        
        # Use original V1 model with soft routing simulation
        from models.turs_rrmt.model import TURSRRMT as V1Model
        
        v1_path = os.path.join(V1_CKPT, f"{ds_tag}_A4.pt")
        ckpt = torch.load(v1_path, map_location="cpu", weights_only=False)
        model_v1 = V1Model(
            num_classes=ckpt["num_classes"], seq_len=ckpt["seq_len"],
            M=ckpt["M"], J=ckpt["J"], topk=ckpt["topk"],
            ppv_window=ckpt["ppv_window"], variant=ckpt["variant"],
            seed=ckpt["seed"])
        model_v1.load_state_dict(ckpt["model_state_dict"])
        model_v1.flavors.ref_q.copy_(torch.tensor(ckpt["flavor_ref"]["ref_q"]))
        model_v1.flavors.ref_q_fine.copy_(torch.tensor(ckpt["flavor_ref"]["ref_q_fine"]))
        model_v1.flavors.ref_lag.copy_(torch.tensor(ckpt["flavor_ref"]["ref_lag"]))
        model_v1.flavors.fitted = True
        model_v1.eval().to(device)
        
        # Extract features with soft routing (replace hard top-K with uniform weights)
        features_soft = {}
        for split, X, y, ykey in [("Xtr", ds_data["Xtr"], ytr, "ytr"),
                                   ("Xva", ds_data["Xva"], yva, "yva"),
                                   ("Xte", ds_data["Xte"], yte, "yte")]:
            with torch.no_grad():
                xb = torch.from_numpy(X).float().to(device)
                # Get internals
                R = model_v1.bank(xb)
                A, Sv = model_v1.activity(R)
                P = torch.cat([A, torch.log1p(Sv)], 1)
                Tv = model_v1.flavors(xb, window=model_v1.ppv_window)
                Tp = min(P.shape[-1], Tv.shape[-1])
                P, Tv = P[..., :Tp], Tv[..., :Tp]
                
                # Soft routing: temperature-scaled softmax
                w_logits = model_v1.router(P.permute(0, 2, 1))
                tau = 2.0
                w_soft = F.softmax(w_logits / tau, dim=-1).permute(0, 2, 1)
                
                # Weighted transport - V1 A4 transport_proj expects [B, J+1=5, T]
                Tv_padded = torch.cat([Tv, torch.zeros_like(Tv[:, :1, :])], dim=1)
                rest = 1.0 - w_soft.sum(dim=1, keepdim=True)
                w_padded = torch.cat([w_soft, rest.clamp(min=0)], dim=1)
                F_T = model_v1.transport_proj(w_padded * Tv_padded)
                F_P = model_v1.pattern_path(P.permute(0, 2, 1)).permute(0, 2, 1)
                
                F_all = torch.cat([F_T, F_P], dim=1)
                mu = F_all.mean(-1)
                mx = F_all.max(-1).values
                sd = F_all.std(-1) if F_all.shape[-1] > 1 else torch.zeros_like(mu)
                h = torch.cat([mu, mx, sd], dim=1)
                
                features_soft[split] = h.cpu().numpy()
                features_soft[ykey] = y
        
        # Fit ridge on soft-routed features
        model_readout = TURSRRMTV2(
            num_classes=n_cls, seq_len=ds_data["L"],
            variant="R4", readout_type="ridge", seed=SEED)
        best_lam, best_mf1 = model_readout.fit_ridge_readout(
            features_soft["Xtr"], features_soft["ytr"],
            features_soft["Xva"], features_soft["yva"], lam_grid=LAM_GRID)
        
        probs = model_readout.predict_with_ridge(
            torch.from_numpy(features_soft["Xte"]).float().to(device))
        pred = probs.argmax(-1).cpu().numpy()
        
        result["test"] = full_metrics(yte, pred, n_cls)
        result["val_mf1"] = best_mf1
        result["selected_lambda"] = best_lam
        
    elif variant == "R5":
        # Kernelized routed readout
        log(f"  [{tag}] Kernel ridge readout")
        
        # Build routing-weighted features Psi(x) = concat_j aggregate_t sqrt(w_j(t)) Phi_j(t)
        def build_routed_psi(Tv_arr, Psc_arr, w_arr):
            """Build routing-weighted kernel features.
            
            Args:
                Tv_arr: [B, J, T] transport features
                Psc_arr: [B, 2] precomputed pattern scalars (mean-of-mean, mean-of-max)
                w_arr: [B, J, T] routing weights
            
            Returns:
                Psi: [B, D_routed] routing-weighted feature vector
            """
            Tv_t = torch.from_numpy(Tv_arr).float()
            Psc_t = torch.from_numpy(Psc_arr).float()
            w_t = torch.from_numpy(w_arr).float()
            B, J, T = Tv_t.shape
            
            # sqrt routing weights for PSD kernel: K = <sqrt(w)*Phi, sqrt(w)*Phi>
            w_sqrt = torch.sqrt(w_t.clamp(min=1e-8))
            
            # Per-flavor routing-weighted aggregation
            parts = []
            for j in range(J):
                # sqrt(w_j) * T_j aggregated over time
                psi_j = (w_sqrt[:, j, :] * Tv_t[:, j, :]).mean(dim=-1)  # [B]
                parts.append(psi_j)
            
            # Also include unweighted transport means (capturing raw flavor content)
            for j in range(J):
                parts.append(Tv_t[:, j, :].mean(dim=-1))  # [B]
            
            # Pattern features (precomputed in the extractor; P itself is not cached)
            parts.append(Psc_t[:, 0])  # mean of mean
            parts.append(Psc_t[:, 1])  # mean of max
            
            return torch.stack(parts, dim=1)  # [B, 2J+2]
        
        # Build features for each split
        Psi_tr = build_routed_psi(features["r_Tv_tr"], features["r_Psc_tr"], features["r_w_tr"])
        Psi_va = build_routed_psi(features["r_Tv_va"], features["r_Psc_va"], features["r_w_va"])
        Psi_te = build_routed_psi(features["r_Tv_te"], features["r_Psc_te"], features["r_w_te"])
        
        # FIX: standardize Psi with TRAIN stats. Raw transport means are ~1e-2
        # scale, so K = Psi Psi^T ~ 1e-4 and the lambda grid is meaningless.
        p_mu, p_sd = Psi_tr.mean(0), Psi_tr.std(0).clamp_min(1e-6)
        Psi_tr = (Psi_tr - p_mu) / p_sd
        Psi_va = (Psi_va - p_mu) / p_sd
        Psi_te = (Psi_te - p_mu) / p_sd
        
        ytr_t = torch.from_numpy(ytr).long()
        yva_t = torch.from_numpy(yva).long()
        yte_t = torch.from_numpy(yte).long()
        
        log(f"  [{tag}] Psi shapes: tr={Psi_tr.shape} va={Psi_va.shape} te={Psi_te.shape}")
        
        # Lambda selection on validation
        best_lam, best_mf1 = None, -1
        for lam in LAM_GRID:
            kr = KernelRidgeReadout(num_classes=n_cls, J=4, lam=lam, kernel_type='linear')
            probs = kr(Psi_tr, ytr_t, Psi_va)
            pred = probs.argmax(-1).numpy()
            vm = _mf1(yva, pred, n_cls)
            if vm > best_mf1:
                best_mf1, best_lam = vm, lam
        
        log(f"  [{tag}] Best lambda={best_lam} val_MF1={best_mf1:.4f}")
        
        # Final evaluation on test
        kr = KernelRidgeReadout(num_classes=n_cls, J=4, lam=best_lam, kernel_type='linear')
        probs = kr(Psi_tr, ytr_t, Psi_te)
        pred = probs.argmax(-1).numpy()
        
        result["test"] = full_metrics(yte, pred, n_cls)
        result["val_mf1"] = best_mf1
        result["selected_lambda"] = best_lam
        result["kernel_dim"] = Psi_tr.shape[1]
        
        # Kernel validation
        K_train = kr.kernel_matrix(Psi_tr)
        sym_err = (K_train - K_train.T).abs().max().item()
        eigvals = torch.linalg.eigvalsh(K_train)
        result["kernel_validation"] = dict(
            symmetry_error=round(sym_err, 8),
            min_eigenvalue=round(eigvals.min().item(), 6),
            n_negative_eigenvalues=int((eigvals < -1e-8).sum().item()),
            condition_number=round((eigvals.max() / (eigvals.min() + 1e-10)).item(), 2))
        
    elif variant in ("R6", "R7"):
        # Larger pattern bank
        M = 256 if variant == "R6" else 512
        log(f"  [{tag}] M={M} + Ridge readout")
        
        # Extract with larger bank
        features_big = extract_v2_representation(ds_tag, device, variant=variant, M=M)
        
        from sklearn.linear_model import RidgeClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        best_lam, best_mf1 = None, -1
        for lam in LAM_GRID:
            # FIX: standardize with TRAIN stats (see R1)
            clf = make_pipeline(StandardScaler().fit(features_big["Xtr"]),
                                RidgeClassifier(alpha=lam))
            clf.fit(features_big["Xtr"], features_big["ytr"])
            pred_va = clf.predict(features_big["Xva"])
            vm = _mf1(features_big["yva"], pred_va, n_cls)
            if vm > best_mf1:
                best_mf1, best_lam = vm, lam
        
        clf = make_pipeline(StandardScaler().fit(features_big["Xtr"]),
                            RidgeClassifier(alpha=best_lam))
        clf.fit(features_big["Xtr"], features_big["ytr"])
        pred = clf.predict(features_big["Xte"])
        result["test"] = full_metrics(features_big["yte"], pred, n_cls)
        result["val_mf1"] = best_mf1
        result["selected_lambda"] = best_lam
        result["M"] = M
        _ridge = clf.steps[-1][1]
        result["params"] = dict(trainable=int(_ridge.coef_.size + _ridge.intercept_.size), fixed=0)
    
    result["elapsed_s"] = round(time.time() - t0, 1)
    
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    
    log(f"  [{tag}] test MF1={result['test']['macro_f1']:.4f} ({result['elapsed_s']:.0f}s)")
    return result


# ============================================================
# Phase 8: Diagnostics (D1-D6) — see experiments/turs_rrmt_v2/diagnostics.py
# ============================================================
def phase8_diagnostics(tags, device, hyp):
    from experiments.turs_rrmt_v2 import diagnostics as V2D
    log("PHASE 8: V2 diagnostics (D1-D6)")
    diag = V2D.run_all_diagnostics(RESULTS, device, hyp, datasets=tags)
    out_path = os.path.join(RESULTS, "diagnostics", "v2_diagnostics.json")
    dump_json(diag, out_path)
    log(f"  diagnostics written -> {os.path.relpath(out_path, ROOT)}")
    return diag


# ============================================================
# Phase 9: Statistical significance (paired McNemar vs R0 + BH-FDR)
# ============================================================
def phase9_significance(tags, all_results, hyp):
    """Paired McNemar of each readout vs R0 (V1 MLP head) using the D1
    replayed per-sample predictions; BH-FDR within the V2_vs_R0 family."""
    from src.diagnostics.statistics import mcnemar, dump_csv
    log("PHASE 9: Statistical significance (McNemar vs R0 + BH-FDR)")

    def _pred_path(ds, v):
        return os.path.join(RESULTS, ds, "predictions", f"{v}_test_pred.npz")

    for ds in tags:
        y_path = os.path.join(RESULTS, ds, "representation", "v1_features_M128.npz")
        if not os.path.exists(y_path):
            log(f"  {ds}: no representation cache -> significance skipped")
            continue
        yte = np.load(y_path)["yte"]
        r0p = _pred_path(ds, "R0")
        v1_npz = os.path.join(ROOT, "results", "turs_rrmt", ds,
                              "routing_outputs_test.npz")
        if os.path.exists(r0p):
            pred0 = np.load(r0p)["pred"]
        elif os.path.exists(v1_npz):
            pred0 = np.load(v1_npz)["pred"]
        else:
            log(f"  {ds}: no R0 predictions -> significance skipped")
            continue
        correct0 = np.asarray(pred0) == yte

        for v in VARIANTS:
            if v == "R0":
                continue
            pp = _pred_path(ds, v)
            if not os.path.exists(pp):
                continue
            pred = np.load(pp)["pred"]
            if len(pred) != len(yte):
                log(f"  {ds}/{v}: length mismatch vs test split -> skipped")
                continue
            chi2, p = mcnemar(pred == yte, correct0)
            mf1_v = all_results.get(ds, {}).get(v, {}).get("test", {}) \
                .get("macro_f1")
            mf1_0 = all_results.get(ds, {}).get("R0", {}).get("test", {}) \
                .get("macro_f1")
            delta = (mf1_v - mf1_0) if (mf1_v is not None and mf1_0 is not None) \
                else None
            hyp.add("V2_vs_R0", "McNemar (paired, continuity-corrected)",
                    f"{ds}: {v} vs R0", delta, p,
                    extra=dict(dataset=ds, variant=v))
    rows = hyp.finalize()
    # attach verdicts
    for row in rows:
        if row.get("family") == "V2_vs_R0":
            d = row.get("estimate") or 0
            q = row.get("q_value")
            if row.get("p_value") is None:
                verdict = "n/a"
            elif q is not None and q < 0.05:
                verdict = "SIGNIFICANT " + ("IMPROVEMENT" if d > 0 else "DEGRADATION")
            else:
                verdict = "n.s." + (" (improves)" if d > 0 else
                                     " (degrades)" if d < 0 else "")
            row["extra"]["verdict"] = verdict
            row["extra"]["mf1_delta"] = row.get("estimate")
    out_path = os.path.join(TABLE_DIR, "hypothesis_tests.csv")
    os.makedirs(TABLE_DIR, exist_ok=True)
    dump_csv(rows, out_path)
    # persist raw rows so --report-only can rebuild the report later
    with open(os.path.join(TABLE_DIR, "hypothesis_rows.json"), "w") as f:
        json.dump(rows, f, indent=2)
    sig = [r for r in rows if r.get("significant")]
    log(f"  {len(rows)} hypothesis rows, {len(sig)} significant after FDR "
        f"-> {os.path.relpath(out_path, ROOT)}")
    return rows


# ============================================================
# Phase 10: Figures + tables + final report
# ============================================================
def phase10_plots(fig_dir, tags, all_results):
    """Comparison bar chart + best-readout summary figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(fig_dir, exist_ok=True)

    variants = ["R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7"]
    # 1. grouped bars: readout x dataset
    fig, ax = plt.subplots(figsize=(10, 4.5))
    width = 0.8 / len(variants)
    for i, v in enumerate(variants):
        vals = [all_results.get(t, {}).get(v, {}).get("test", {})
                .get("macro_f1", 0) for t in tags]
        pos = np.arange(len(tags)) + i * width - 0.4 + width / 2
        ax.bar(pos, vals, width, label=v)
    ax.set_xticks(np.arange(len(tags)))
    ax.set_xticklabels(tags, rotation=15)
    ax.set_ylabel("Test Macro-F1")
    ax.set_title("TURS-RRMT-V2: readout surgery on the frozen V1 representation")
    ax.legend(ncol=8, fontsize=8)
    ax.set_ylim(0, 1)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"variant_comparison.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. ridge-vs-MLP delta per dataset (the V2 headline)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    deltas, labels = [], []
    for t in tags:
        r0 = all_results.get(t, {}).get("R0", {}).get("test", {}).get("macro_f1")
        r1 = all_results.get(t, {}).get("R1", {}).get("test", {}).get("macro_f1")
        if r0 is not None and r1 is not None:
            deltas.append(r1 - r0)
            labels.append(t)
    ax.bar(np.arange(len(labels)), deltas, 0.6,
           color=["#2a9d8f" if d > 0 else "#e76f51" for d in deltas])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("MF1 delta (frozen ridge R1 - MLP head R0)")
    ax.set_title("How much the MLP head wastes the V1 representation")
    for i, d in enumerate(deltas):
        ax.text(i, d + (0.01 if d >= 0 else -0.03), f"{d:+.3f}",
                ha="center", fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"ridge_vs_mlp_delta.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  figures -> {os.path.relpath(fig_dir, ROOT)}")


def phase10_report(tags, all_results, v1_results, diag, sig_rows, elapsed_s):
    from experiments.turs_rrmt_v2.reporting import write_report
    rep = write_report(RESULTS, all_results, v1_results, sig_rows, diag,
                       elapsed_s, tags)
    log(f"  report -> {os.path.relpath(rep, ROOT)}")
    return rep


# ============================================================
# Main pipeline
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", default=True)
    ap.add_argument("--dataset", type=str, default=None)
    ap.add_argument("--variant", type=str, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--diagnostics-only", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--plots-only", action="store_true")
    args = ap.parse_args()
    
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    
    tags = [args.dataset] if args.dataset else list(DATASETS)
    variants = [args.variant] if args.variant else VARIANTS
    
    # Phase 0
    phase0_audit()
    
    # Phase 1
    v1_results = phase1_v1_baseline()
    
    # Phase 2: Extract V1 representations
    log("PHASE 2: V2 representation extraction")
    all_features = {}
    for ds in tags:
        all_features[ds] = extract_v2_representation(ds, device)
    
    if args.train_only:
        log("TRAIN-ONLY complete")
        return

    # Collect cached variant results (skip training when cached)
    all_results = {}
    for ds in tags:
        all_results[ds] = {}
        for variant in variants:
            fp = os.path.join(RESULTS, ds, f"{variant}_results.json")
            if os.path.exists(fp) and not args.force:
                all_results[ds][variant] = json.load(open(fp))
                log(f"  {ds}/{variant}: cached (use --force to redo)")

    if args.report_only:
        hyp = HypothesisRegistry()
        rows = json.load(open(os.path.join(TABLE_DIR, "hypothesis_rows.json"))) \
            if os.path.exists(os.path.join(TABLE_DIR, "hypothesis_rows.json")) \
            else []
        diag = json.load(open(os.path.join(RESULTS, "diagnostics",
                                           "v2_diagnostics.json"))) \
            if os.path.exists(os.path.join(RESULTS, "diagnostics",
                                           "v2_diagnostics.json")) else {}
        phase10_report(tags, all_results, v1_results, diag, rows,
                       time.time() - t0)
        return

    # Run any missing variants
    for ds in tags:
        for variant in variants:
            if variant in all_results[ds]:
                continue
            all_results[ds][variant] = run_v2_variant(ds, variant,
                                                      all_features[ds],
                                                      load_split(ds), device)

    hyp = HypothesisRegistry()
    diag = {}
    sig_rows = []
    if not args.plots_only:
        # Phase 8: diagnostics (D1-D6) incl. per-sample prediction replay
        diag = phase8_diagnostics(tags, device, hyp)
        # Phase 9: significance + FDR
        sig_rows = phase9_significance(tags, all_results, hyp)

    # Phase 10: Generate comparison table
    log("PHASE 10: Generating comparison table + figures + report")
    os.makedirs(TABLE_DIR, exist_ok=True)
    
    rows = []
    for ds in tags:
        row = dict(Dataset=ds)
        for v in variants:
            if v in all_results.get(ds, {}):
                r = all_results[ds][v]
                row[f"V2_{v}"] = round(r["test"]["macro_f1"], 4)
            else:
                row[f"V2_{v}"] = None
        # Add V1 baseline
        if ds in v1_results:
            row["V1_A4"] = round(v1_results[ds]["test_mf1"], 4)
        rows.append(row)
    
    # Write CSV
    import csv
    if rows:
        with open(os.path.join(TABLE_DIR, "model_comparison.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    
    # Print summary
    log("\n=== V2 MODEL COMPARISON (Test Macro-F1) ===")
    header = f"{'Dataset':14s} | " + " | ".join(f"{v:8s}" for v in ["V1_A4"] + [f"V2_{v}" for v in variants])
    log(header)
    log("-" * len(header))
    for ds in tags:
        vals = [f"{v1_results.get(ds, {}).get('test_mf1', 0):8.4f}"]
        for v in variants:
            r = all_results.get(ds, {}).get(v, {})
            mf1 = r.get("test", {}).get("macro_f1", 0)
            vals.append(f"{mf1:8.4f}")
        log(f"{ds:14s} | " + " | ".join(vals))
    
    # Figures + report (phase 10 cont.)
    phase10_plots(FIG_DIR, tags, all_results)
    phase10_report(tags, all_results, v1_results, diag, sig_rows,
                   time.time() - t0)

    # Save full results
    with open(os.path.join(RESULTS, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    elapsed = time.time() - t0
    log(f"\nCOMPLETE in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
