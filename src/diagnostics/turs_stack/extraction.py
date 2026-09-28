"""Unified Stack diagnostic extraction (Phase 3). ONE forward pass per split.

Native internals per branch (from models/turs_stack/model.py source):
  lite/rv: z_t [B,T,16] (P_z per-timestep), v_t (learned projection; rv adds a
           gated response correction -> vtilde), u_t = sigmoid(P_u([z_t||v_t]))
           in (0,1) [B,T,1], alpha (transport reliance, sample-level [B,1]).
  cs:      beta [B,T,3] scale-trust (softmax), z_t/vtilde beta-fused, v = EMA
           difference (genuine temporal derivative), u_t = sigmoid(P_u) [B,T,1],
           e_t novelty [B] = tanh(1 - cos(z_t, zbar_1)).mean(t).
  cmr:     as cs + bounded controlled response, gamma; returns (logits, beta).
Alpha semantics: F_fused = alpha*F_T_proj + (1-alpha)*F_R_gap -> alpha weights
the TRANSPORT contribution (high alpha = transport reliance).
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from . import config as C


def _file_hash(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def forward_split(model, X, device, batch_size=128, heavy_temporal=True):
    """Run frozen inference; return dict of numpy arrays."""
    model.eval()
    X_t = torch.from_numpy(X).float()
    dl = DataLoader(TensorDataset(X_t), batch_size=batch_size)
    acc = {k: [] for k in [
        "branch_probs", "branch_logits", "z_t_lite", "v_t_lite", "u_t_lite",
        "alpha_lite", "z_t_rv", "vtilde_rv", "u_t_rv", "alpha_rv",
        "z_t_cs", "v_t_cs", "u_t_cs", "alpha_cs", "beta_cs",
        "z_t_cmr", "v_t_cmr", "u_t_cmr", "alpha_cmr", "beta_cmr",
        "novelty", "gamma", "F_T", "H"]}
    for (xb,) in dl:
        xb = xb.to(device)
        out = model(xb)

        # ---- recompute branch internals from the SAME frozen modules ----
        T_repr = model.transport_builder(xb)
        F_T = model.transport_encoder(T_repr)
        H = model.backbone(xb)

        # lite
        b = model.lite
        z1 = b.P_z(H).permute(0, 2, 1)
        v1 = b.vel_linear(z1)
        u1 = torch.sigmoid(b.P_u(torch.cat([z1, v1], -1)))
        a1 = torch.sigmoid(b.P_alpha(torch.cat([F_T.mean(2), b.P_R(
            torch.cat([z1, v1, u1], -1)).mean(1)], 1)))
        # rv
        b = model.rv
        z2 = b.P_z(H).permute(0, 2, 1)
        v2 = b.vel_linear(z2)
        R7 = b.dw7(H).permute(0, 2, 1); R15 = b.dw15(H).permute(0, 2, 1)
        R31 = b.dw31(H).permute(0, 2, 1)
        R_t = torch.cat([R7, R15, R31], -1)
        dR = torch.zeros_like(R_t); dR[:, 1:, :] = R_t[:, 1:, :] - R_t[:, :-1, :]
        g = torch.sigmoid(b.P_g(torch.cat([z2, v2, R_t], -1)))
        vtilde2 = v2 + g * b.P_Delta(dR)
        u2 = torch.sigmoid(b.P_u(torch.cat([z2, vtilde2], -1)))
        a2 = torch.sigmoid(b.P_alpha(torch.cat([F_T.mean(2), b.P_R(
            torch.cat([z2, vtilde2, u2], -1)).mean(1)], 1)))
        # cs
        b = model.cs
        sigma = b.scale_params.sigma; rho = b.scale_params.rho
        zA = b.branch1(H, sigma); zB = b.branch2(H, 3.0 * sigma); zC = b.branch3(H, 9.0 * sigma)
        zbar_g, v_g = __import__("models.turs_stack.model", fromlist=["ema_multi"]).ema_multi(
            torch.cat([zA, zB, zC], 1), rho)
        zbar1, zbar2, zbar3 = zbar_g.split(b.branch_dim, 1)
        vv1, vv2, vv3 = v_g.split(b.branch_dim, 1)

        def p(x):
            return x.permute(0, 2, 1)

        Q_t = torch.cat([p(zbar1), p(zbar2), p(zbar3), p(vv1), p(vv2), p(vv3)], -1)
        beta3 = F.softmax(b.P_beta(Q_t), -1)
        b1, b2, b3 = beta3[..., 0:1], beta3[..., 1:2], beta3[..., 2:3]
        z3 = b1 * p(zbar1) + b2 * p(zbar2) + b3 * p(zbar3) + p(zbar1)
        v3 = b1 * p(vv1) + b2 * p(vv2) + b3 * p(vv3)
        u3 = torch.sigmoid(b.P_u(torch.cat([z3, v3], -1)))
        a3 = torch.sigmoid(b.P_alpha(torch.cat([F_T.mean(2), b.P_R(
            torch.cat([z3, v3, u3], -1)).mean(1)], 1)))
        # cmr (includes response bank: R latents feed beta/u/F_R exactly as source)
        b = model.cmr
        zA = b.branch1(H, sigma); zB = b.branch2(H, 3.0 * sigma); zC = b.branch3(H, 9.0 * sigma)
        zbar_g, v_g = __import__("models.turs_stack.model", fromlist=["ema_multi"]).ema_multi(
            torch.cat([zA, zB, zC], 1), rho)
        zbar1, zbar2, zbar3 = zbar_g.split(b.branch_dim, 1)
        w1, w2, w3 = v_g.split(b.branch_dim, 1)
        R1 = b.resp1(H).permute(0, 2, 1)
        R2 = b.resp2(H).permute(0, 2, 1)
        R3 = b.resp3(H).permute(0, 2, 1)
        D1 = torch.zeros_like(R1); D1[:, 1:, :] = R1[:, 1:, :] - R1[:, :-1, :]
        D2 = torch.zeros_like(R2); D2[:, 1:, :] = R2[:, 1:, :] - R2[:, :-1, :]
        D3 = torch.zeros_like(R3); D3[:, 1:, :] = R3[:, 1:, :] - R3[:, :-1, :]
        desc = torch.cat([R1, R2, R3, R1.abs(), R2.abs(), R3.abs(),
                          D1, D2, D3, R1 * R1, R2 * R2, R3 * R3], -1)
        R_lat = b.P_Rdesc(desc)
        s1 = b.P_s1(torch.cat([p(zbar1), p(w1), R_lat], -1))
        s2 = b.P_s2(torch.cat([p(zbar2), p(w2), R_lat], -1))
        s3 = b.P_s3(torch.cat([p(zbar3), p(w3), R_lat], -1))
        beta4 = F.softmax(torch.cat([s1, s2, s3], -1), -1)
        c1, c2, c3 = beta4[..., 0:1], beta4[..., 1:2], beta4[..., 2:3]
        z4 = c1 * p(zbar1) + c2 * p(zbar2) + c3 * p(zbar3) + p(zbar1)
        v4 = c1 * p(w1) + c2 * p(w2) + c3 * p(w3)
        a_t4 = torch.sigmoid(b.P_a(torch.cat([z4, v4, b.gamma * torch.sigmoid(
            b.P_g(torch.cat([p(zbar1) + p(zbar2) + p(zbar3),
                             p(w1) + p(w2) + p(w3)], -1))) * R_lat], -1)))
        u4 = torch.sigmoid(b.P_u(torch.cat([z4, v4, R_lat, a_t4], -1)))
        a4 = torch.sigmoid(b.P_alpha(torch.cat([F_T.mean(2), b.P_R2(
            torch.cat([z4, v4, R_lat, a_t4, u4], -1)).mean(1)], 1)))

        rec = dict(
            # stack branch dim -> [4, B, C]; concat along BATCH dim later
            branch_probs=torch.stack(out["probs"], 0).cpu(),
            branch_logits=torch.stack([l.float() for l in out["branch_logits"]], 0).cpu(),
            z_t_lite=z1, v_t_lite=v1, u_t_lite=u1, alpha_lite=a1.reshape(-1, 1),
            z_t_rv=z2, vtilde_rv=vtilde2, u_t_rv=u2, alpha_rv=a2.reshape(-1, 1),
            z_t_cs=z3, v_t_cs=v3, u_t_cs=u3, alpha_cs=a3.reshape(-1, 1),
            beta_cs=beta3,
            z_t_cmr=z4, v_t_cmr=v4, u_t_cmr=u4, alpha_cmr=a4.reshape(-1, 1),
            beta_cmr=beta4,
            novelty=out["novelty"].cpu(), gamma=model.cmr.gamma.detach().cpu().reshape(1),
        )
        if heavy_temporal:
            rec["F_T"] = F_T.cpu(); rec["H"] = H.cpu()
        for k, val in rec.items():
            acc[k].append(val)

    out_np = {}
    for k, v in acc.items():
        if not v:
            continue
        if k in ("branch_probs", "branch_logits"):
            out_np[k] = torch.cat(v, dim=1).cpu().numpy()      # [4, N, C]
        else:
            out_np[k] = torch.cat(v).cpu().numpy()             # [N, ...]
    # gamma is a constant scalar per checkpoint; one value is enough
    if "gamma" in out_np:
        out_np["gamma"] = out_np["gamma"].reshape(-1)[:1]
    return out_np


def compute_scalars(raw, y):
    """Per-sample scalar diagnostics for every stored tensor family."""
    eps = 1e-12
    out = dict(y=np.asarray(y).astype(np.int64))
    B = raw["branch_probs"]                    # [4, N, C]
    K, N, Cl = B.shape
    out["branch_pred"] = B.argmax(-1)          # [4, N]
    out["branch_conf"] = B.max(-1)             # [4, N]
    out["branch_correct"] = (out["branch_pred"] == y[None, :]).astype(np.int8)
    out["branch_entropy"] = -(B * np.log(B + eps)).sum(-1)
    out["branch_nll"] = -np.log(B[np.arange(K)[:, None], np.arange(Cl)[None],
                                  np.broadcast_to(y, (K, N))] + eps).mean(0) \
        if False else np.stack([-np.log(B[k, np.arange(N), y] + eps) for k in range(K)])

    # final stack (soft vote)
    P = B.mean(0)
    out["final_probs"] = P
    out["final_pred"] = P.argmax(1)
    out["final_conf"] = P.max(1)
    out["final_entropy"] = -(P * np.log(P + eps)).sum(1)
    srt = np.sort(P, 1)
    out["final_margin"] = srt[:, -1] - srt[:, -2]
    out["final_correct"] = (out["final_pred"] == y).astype(np.int8)
    out["final_nll"] = -np.log(P[np.arange(N), y] + eps)

    # ensemble disagreement signals
    bp = out["branch_pred"]
    votes = np.eye(Cl)[bp]                     # [4, N, C]
    counts = votes.sum(0)
    out["n_agree_max"] = counts.max(1)
    out["branch_pred_disagreement"] = (K - counts.max(1)) / (K - 1)
    vote_probs = votes.mean(0)
    out["vote_entropy"] = -(vote_probs * np.log(vote_probs + eps)).sum(1)
    out["prob_var"] = B.var(0).mean(1)         # mean over classes of var across branches
    out["conf_dispersion"] = out["branch_conf"].std(0)
    # pairwise JS divergence (mean over pairs)
    js = np.zeros(N)
    pairs = [(a, b2) for a in range(K) for b2 in range(a + 1, K)]
    for a, b2 in pairs:
        m = 0.5 * (B[a] + B[b2])
        kl_am = (B[a] * np.log((B[a] + eps) / (m + eps))).sum(1)
        kl_bm = (B[b2] * np.log((B[b2] + eps) / (m + eps))).sum(1)
        js += 0.5 * (kl_am + kl_bm)
    out["js_disagreement"] = js / len(pairs)
    out["mean_branch_entropy"] = out["branch_entropy"].mean(0)

    # branch u / velocity scalars
    for name, key in [("lite", "u_t_lite"), ("rv", "u_t_rv"), ("cs", "u_t_cs"),
                      ("cmr", "u_t_cmr")]:
        if key in raw:
            u = raw[key][:, :, 0]              # [N, T]
            out[f"u_{name}_mean"] = u.mean(1)
            out[f"u_{name}_max"] = u.max(1)
            out[f"u_{name}_q75"] = np.quantile(u, 0.75, axis=1)
    for name, key in [("lite", "v_t_lite"), ("rv", "vtilde_rv"), ("cs", "v_t_cs"),
                      ("cmr", "v_t_cmr")]:
        if key in raw:
            s = np.linalg.norm(raw[key], axis=2)   # [N, T]
            out[f"s_{name}_max"] = s.max(1)
            out[f"s_{name}_mean"] = s.mean(1)
    for name in ["lite", "rv", "cs", "cmr"]:
        if f"alpha_{name}" in raw:
            out[f"alpha_{name}"] = raw[f"alpha_{name}"][:, 0]
    return out


def extract_dataset(tag, model, ds, device, force=False):
    """Extract train/val/test once; cache npz + scalars per split."""
    out_dir = os.path.join(C.EXTRACT_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_hash = _file_hash(os.path.join(C.CKPT_DIR, f"{tag}_turs_stack.pt"))
    results = {}
    for split, X, y in [("train", ds["Xtr"], ds["y_train"]),
                        ("val", ds["Xva"], ds["y_val"]),
                        ("test", ds["Xte"], ds["y_test"])]:
        path = os.path.join(out_dir, f"{split}.npz")
        if os.path.exists(path) and not force:
            results[split] = dict(np.load(path, allow_pickle=True))
            continue
        heavy = split == "test"      # keep big temporals for test only
        raw = forward_split(model, X, device, heavy_temporal=heavy)
        scal = compute_scalars(raw, y)
        sample_id = np.arange(len(y))
        meta = dict(dataset=tag, split=split, checkpoint_sha256=ckpt_hash,
                    seed=C.SEED, extraction_version="stack_diag_v1",
                    model="TURSStack frozen (4 branches)")
        payload = {**raw, **scal, "sample_id": sample_id}
        payload["_meta"] = np.array(json.dumps(meta))
        np.savez_compressed(path, **payload)
        results[split] = payload
    return results
