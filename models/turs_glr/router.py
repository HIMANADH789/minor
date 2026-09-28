"""TURS-GLR soft regime/pattern router.

Trained SOFT FROM SCRATCH (spec block 3): the router consumes local pattern
evidence P(t) in [B, T', 2*M_local] and produces routing weights over J=4
transport flavors via a temperature-controlled softmax:

    w_j(t) = softmax(g(P(t)) / tau)_j

There is no hard/top-K routing anywhere in GLR (that was V1's design and the
source of R4's softened-after-the-fact artifacts). The router is trained
end-to-end with cross-entropy gradients that flow through the closed-form
block-ridge solution (implicit differentiation via torch.linalg.solve),
initializing from the frozen feature epoch when weights are provided.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftRouter(nn.Module):
    """Linear(2M->32) -> ReLU -> Linear(32->J) -> softmax(logits / tau)."""

    def __init__(self, M_local=128, J=4, hidden=32, tau=1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * M_local, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, J))
        self.tau = float(tau)

    def forward(self, P_btc):
        """P_btc: [B, T', 2M] -> w [B, T', J] (softmax rows sum to 1)."""
        return F.softmax(self.net(P_btc) / self.tau, dim=-1)

    def logits(self, P_btc):
        return self.net(P_btc)


def routing_entropy(w):
    """w: [..., J] -> entropy over last dim (nats)."""
    return -(w * torch.log(w + 1e-12)).sum(dim=-1)
