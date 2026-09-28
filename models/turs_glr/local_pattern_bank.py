"""TURS-GLR local stream: small fixed bank + local PPV + local strength.

Reuses the RRMT FixedPatternBank (full temporal resolution preserved) and
LocalActivity (sliding-window local PPV A(t) and |response| strength S(t)).
Strength normalization uses TRAIN-derived medians only (computed once and
passed in; never refit on val/test).
"""

import torch
import torch.nn as nn

from models.turs_rrmt.model import FixedPatternBank, LocalActivity


class LocalStream(nn.Module):
    """Small fixed bank -> R(t), local PPV A(t), normalized strength S~(t).

    forward returns dict(R, A, S, P) where P = [A || log1p(S~)] is the
    router-input evidence tensor [B, 2*M_local, T'].
    """

    def __init__(self, M_local=128, lengths=(7, 11, 15, 23, 31), seed=42,
                 ppv_window=None, seq_len=None):
        super().__init__()
        self.M = M_local
        self.bank = FixedPatternBank(M=M_local, lengths=lengths, seed=seed)
        if ppv_window is None:
            ppv_window = max(3, round(0.05 * (seq_len or 140)))
        self.ppv_window = int(ppv_window)
        self.activity = LocalActivity(self.ppv_window)
        # TRAIN-derived per-kernel strength medians (registered buffer; set
        # via set_train_medians, never refit on val/test)
        self.register_buffer("s_med", torch.ones(M_local))

    @torch.no_grad()
    def set_train_medians(self, X_train, batch=256):
        """Fit per-kernel strength medians on TRAIN only."""
        meds = []
        for i in range(0, len(X_train), batch):
            xb = torch.from_numpy(X_train[i:i + batch]).float()
            R = self.bank(xb)
            _, S = self.activity(R)
            meds.append(S.flatten(0, 2))
        all_s = torch.cat(meds)
        self.s_med.copy_(all_s.median())

    def forward(self, x):
        R = self.bank(x)                          # [B, M, T']
        A, S = self.activity(R)                   # [B, M, T'] each
        s_med = self.s_med.to(x.device).view(1, -1, 1)
        S_norm = S / (s_med + 1e-8)
        P = torch.cat([A, torch.log1p(S_norm)], dim=1)   # [B, 2M, T']
        return dict(R=R, A=A, S=S, S_norm=S_norm, P=P)
