"""TURS-GLR routed transport statistics.

For each transport flavor j the routed field is U_j(t) = w_j(t) * T_j(t).
Instead of collapsing to a scalar (the R5 10-D mistake), this module derives
a rich, documented per-flavor summary. Feature definition (exactly, per
flavor j, computed on U_j and w_j):

    0  U_mean     temporal mean of U_j
    1  U_std      temporal std of U_j
    2  U_max      temporal max of U_j
    3  U_q90      90th percentile of U_j
    4  U_q10      10th percentile of U_j
    5  U_range    U_q90 - U_q10
    6  U_ppv      fraction of t with U_j(t) > 0 (activation rate)
    7  U_energy   mean of U_j(t)^2 (weighted temporal energy)
    8  w_mean     temporal mean of routing weight w_j
    9  w_std      temporal std of w_j
    10 w_max      temporal max of w_j
    11 w_min      temporal min of w_j
    12 w_ent      temporal mean of the routing entropy contribution -w_j log w_j
    13 persist    route persistence: fraction of t where argmax_j w stays the
                  same as the previous position (sample-level, flavor-agnostic)
    14 switch     route switch rate: 1 - persist (flavor-agnostic)

Z_routed = concat over j = 1..J  ->  J * 15 dims (60 for J=4).

persist/switch are duplicated across flavors by construction (they describe
the argmax route, not flavor j); they are kept in every flavor slot so the
layout stays regular. No raw temporal tensors reach the ridge matrix.
"""

import numpy as np
import torch


def routed_statistics(U, w):
    """Compute per-flavor routed summaries.

    Args:
        U: [B, J, T'] routed transport field  w_j(t) * T_j(t)
        w: [B, J, T'] routing weights (rows sum to 1 over j)
    Returns:
        Z: [B, J*15] float32 numpy
    """
    B, J, T = U.shape
    top_seq = w.argmax(1)                                    # [B, T]
    if T > 1:
        persist = (top_seq[:, 1:] == top_seq[:, :-1]).float().mean(1)   # [B]
    else:
        persist = torch.ones(B, device=U.device)
    switch = 1.0 - persist

    feats = []
    for j in range(J):
        u = U[:, j, :]                                       # [B, T]
        wj = w[:, j, :]
        mu = u.mean(1)
        sd = u.std(1) if T > 1 else torch.zeros_like(mu)
        mx = u.max(1).values
        q90 = torch.quantile(u, 0.9, dim=1)
        q10 = torch.quantile(u, 0.1, dim=1)
        rng = q90 - q10
        ppv = (u > 0).float().mean(1)
        energy = (u ** 2).mean(1)
        wm = wj.mean(1)
        ws = wj.std(1) if T > 1 else torch.zeros_like(wm)
        wmx = wj.max(1).values
        wmin = wj.min(1).values
        ent = -(wj * torch.log(wj + 1e-12)).sum(1) / float(J)  # contribution
        feats.append(torch.stack([mu, sd, mx, q90, q10, rng, ppv, energy,
                                  wm, ws, wmx, wmin, ent, persist, switch],
                                 dim=1))
    Z = torch.cat(feats, dim=1)                              # [B, J*15]
    return Z.cpu().numpy().astype(np.float32)
