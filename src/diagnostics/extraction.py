"""Unified inference extraction layer (Phase 6).

Extracts, for every split and every sample:
  - logits, probabilities, confidence, entropy, margin, loss, correctness
  - canonical TURS-Lite internals (sample-level, native via return_aux=True):
      z  = aux["regime"]           [B, 16]
      v  = aux["regime_velocity"]  [B, 16]  (learned projection of z; the
           model's velocity is NOT a temporal difference - documented mapping)
      s  = aux["regime_speed"]     [B, 1]   = ||v||_2
      u  = aux["uncertainty"]      [B, 16]  (softplus of clamped logvar)
      alpha = aux["alpha"]         [B, 1]   (weights the TRANSPORT term in
           F_fused = alpha*(g_T*F_T) + (1-alpha)*(g_R*F_R) -> high alpha =
           transport reliance, low alpha = regime reliance)
      g_T / g_R gates            [B, 1]
      F_T, F_R fused components  [B, 64]
  - TEMPORAL traces (inference-only, canonical frozen weights): the regime
      encoder / velocity projection are per-vector MLPs, so applying them at
      every H4 timestep gives a justified per-timestep trace
      (z_t, v_t, u_t, s_t) of shape [B, T, d]. alpha has NO temporal meaning
      in this architecture and is not extracted temporally.

Everything is cached to disk (npz) so no experiment re-runs inference.
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from . import config as C


def _batch_tensors(model, X, device, batch_size=256):
    dl = DataLoader(TensorDataset(torch.from_numpy(X).float()), batch_size=batch_size)
    outs = []
    with torch.no_grad():
        for (xb,) in dl:
            xb = xb.to(device)
            logits, aux = model(xb, return_aux=True)
            B = xb.shape[0]
            rec = dict(
                logits=logits.float().cpu(),
                probs=F.softmax(logits, dim=1).float().cpu(),
                z=aux["regime"].float().cpu(),
                v=aux["regime_velocity"].float().cpu(),
                s=aux["regime_speed"].float().cpu(),
                u=aux["uncertainty"].float().cpu(),
                alpha=aux["alpha"].float().cpu().reshape(B, 1),
                g_T=aux["transport_gate"].float().cpu().reshape(B, 1),
                g_R=aux["regime_gate"].float().cpu().reshape(B, 1),
                F_T=(model.transport_proj(aux["_T_e_mean"] if "_T_e_mean" in aux
                     else torch.zeros(1)).reshape(B, -1)
                     if False else None),  # filled below
            )
            outs.append(rec)
    return outs


@torch.no_grad()
def extract_split(model, X, y, device, temporal=True, batch_size=256):
    """Run frozen inference and return dict of numpy arrays for one split."""
    from models.tursnet import TURSNet  # noqa: F401
    model.eval()
    X_in = torch.from_numpy(X).float()
    y = np.asarray(y)
    dl = DataLoader(TensorDataset(X_in), batch_size=batch_size)

    acc = {k: [] for k in ["logits", "probs", "z", "v", "s", "u", "alpha",
                           "g_T", "g_R", "F_T", "F_R", "F_fused", "z_t", "v_t",
                           "u_t", "s_t", "H4"]}
    for (xb,) in dl:
        xb = xb.to(device)
        B, _, L = xb.shape
        logits, aux = model(xb, return_aux=True)

        # --- recompute the Lite fusion internals from the SAME frozen modules
        # TransportEncoder.forward returns [B, C_T, L] (single tensor) —
        # identical to the canonical forward path.
        T_e = model.transport_builder(xb)
        t_pooled = model.transport_encoder(T_e).mean(dim=2)
        F_T = model.transport_proj(t_pooled)
        z, u = aux["regime"], aux["uncertainty"]
        v = aux["regime_velocity"]
        F_R = model.regime_proj(z)
        g_T = torch.sigmoid(model.gate_T(torch.cat([t_pooled, z, v, u], dim=1)))
        g_R = torch.sigmoid(model.gate_R(torch.cat([t_pooled, z, v, u], dim=1)))
        alpha = torch.sigmoid(model.alpha_net(torch.cat([t_pooled, z, v, u,
                                                         g_T, g_R], dim=1)))
        F_fused = alpha * (g_T * F_T) + (1 - alpha) * (g_R * F_R)

        rec = dict(
            logits=logits.float().cpu(),
            probs=F.softmax(logits, dim=1).float().cpu(),
            z=z.float().cpu(), v=v.float().cpu(),
            s=aux["regime_speed"].float().cpu(), u=u.float().cpu(),
            alpha=alpha.float().cpu().reshape(B, 1),
            g_T=g_T.float().cpu(), g_R=g_R.float().cpu(),
            F_T=F_T.float().cpu(), F_R=F_R.float().cpu(),
            F_fused=F_fused.float().cpu(),
        )

        if temporal:
            # Temporal traces with the canonical frozen encoders applied per
            # timestep of H4 ([B, 64, T] -> T per-timestep vectors [B, 64]).
            h = model.proj(xb)
            h = model.block1(h)
            h = model.block2(h)
            h = model.block3(h)
            H4 = model.block4(h)              # [B, 64, T]
            HT = H4.permute(0, 2, 1)          # [B, T, 64]
            # regime encoder applied per timestep (vector ops - broadcasting
            # through the Linear stacks is exact)
            m1 = torch.relu(model.regime_main.mu_net[0](HT))
            z_t = model.regime_main.mu_net[2](m1)                       # [B,T,16]
            l1 = torch.relu(model.regime_main.logvar_net[0](HT))
            logvar_t = model.regime_main.logvar_net[2](l1)
            logvar_t = torch.clamp(logvar_t, -5.0, 2.0)
            u_t = F.softplus(logvar_t) + 1e-6                           # [B,T,16]
            v_t = model.vel_linear(z_t)                                 # [B,T,16]
            s_t = v_t.norm(dim=2, keepdim=False)                        # [B,T]
            rec["H4"] = H4.mean(dim=2).float().cpu()   # pooled only (space)
            rec["z_t"], rec["v_t"], rec["u_t"], rec["s_t"] = (
                z_t.float().cpu(), v_t.float().cpu(),
                u_t.float().cpu(), s_t.float().cpu())

        for k, val in rec.items():
            acc[k].append(val)

    out = {k: torch.cat(v).numpy() for k, v in acc.items() if v and v[0] is not None}

    # ------- per-sample scalar diagnostics -------
    probs = out["probs"]
    eps = 1e-12
    out["confidence"] = probs.max(1)
    out["entropy"] = -(probs * np.log(probs + eps)).sum(1)
    srt = np.sort(probs, axis=1)
    out["margin"] = srt[:, -1] - srt[:, -2]
    out["pred"] = probs.argmax(1)
    out["y"] = y
    out["correct"] = (out["pred"] == y).astype(np.int8)
    out["nll_loss"] = -np.log(probs[np.arange(len(y)), y] + eps)
    out["u_mean"] = out["u"].mean(1)
    out["u_max"] = out["u"].max(1)
    out["u_q75"] = np.quantile(out["u"], 0.75, axis=1)
    out["u_var"] = out["u"].var(1)
    out["v_norm"] = np.linalg.norm(out["v"], axis=1)
    out["z_pooled"] = out["z"].mean(1)
    if "s_t" in out:
        out["s_t_max"] = out["s_t"].max(1)
        out["s_t_mean"] = out["s_t"].mean(1)
    return out


def extract_dataset(tag, model, ds, device, force=False):
    """Extract all three splits once; cache to EXTRACT_DIR/<tag>/{split}.npz."""
    out_dir = os.path.join(C.EXTRACT_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(C.CKPT_DIR, f"{tag}_TURS_Lite.pt")
    ckpt_hash = _file_hash(ckpt)
    results = {}
    for split, X, y in [("train", ds["Xtr"], ds["y_train"]),
                        ("val", ds["Xva"], ds["y_val"]),
                        ("test", ds["Xte"], ds["y_test"])]:
        path = os.path.join(out_dir, f"{split}.npz")
        if os.path.exists(path) and not force:
            results[split] = dict(np.load(path, allow_pickle=True))
            continue
        rec = extract_split(model, X, y, device)
        meta = dict(dataset=tag, split=split, checkpoint=os.path.basename(ckpt),
                    checkpoint_sha256=ckpt_hash, seed=C.SEED,
                    preprocessing="per-sample z-norm",
                    model="TURSNet variant=lite multi_scale=False regime_dim=16",
                    extraction="return_aux internals + frozen-encoder temporal traces")
        np.savez_compressed(path, **rec, _meta=np.array(json.dumps(meta)))
        results[split] = rec
    return results


def _file_hash(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
