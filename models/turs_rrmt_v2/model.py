"""TURS-RRMT-V2: Regime-Routed Multi-Transport TURS, Version 2.

V2 wraps the V1 RRMT representation (fixed pattern bank + transport flavors
+ router) and replaces only the downstream readout.

The representation contract:
    V1 representation produces: Z [B, D] (pre-readout features)
    V2 readout consumes: Z [B, D] and produces logits [B, C]

V2 variants:
    R0: V1 original MLP head (baseline)
    R1: V1 frozen features + sklearn Ridge (post-hoc baseline)
    R2: V2 differentiable ridge (V2-A)
    R3: V2 class-weighted ridge (V2-B)
    R4: V2 soft routing + differentiable ridge (V2-C)
    R5: V2 kernelized routed readout (V2-D)
    R6: V2 larger M=256 bank + best readout (V2-E)
    R7: V2 larger M=512 bank + best readout (V2-E)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import copy
import os
import json

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.turs_rrmt.model import (
    FixedPatternBank, LocalActivity, TransportFlavors,
    TURSRRMT, count_params
)
from models.turs_rrmt_v2.readout import (
    DifferentiableRidge, ClassWeightedRidge, KernelRidgeReadout
)


class TURSRRMTV2(nn.Module):
    """TURS-RRMT-V2: V1 representation + V2 readout.
    
    Uses the V1 pattern bank, transport flavors, and router to produce
    a rich representation, then applies a V2 readout.
    
    Args:
        num_classes: C
        seq_len: T
        M: pattern bank size
        J: number of transport flavors
        topk: top-K for hard routing
        variant: R0-R7 (ablation variant)
        readout_type: 'mlp' | 'ridge' | 'class_weighted_ridge' | 'soft_ridge' | 'kernel_ridge'
        lam: ridge regularization
        tau: soft routing temperature
        head_width: MLP head width (for R0 only)
        seed: deterministic seed
    """
    
    def __init__(self, num_classes, seq_len, M=128, J=4, topk=2,
                 variant='R2', readout_type='ridge', lam=1.0, tau=1.0,
                 head_width=192, seed=42, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.M = M
        self.J = J
        self.topk = topk
        self.variant = variant
        self.readout_type = readout_type
        self.lam = lam
        self.tau = tau
        self.seed = seed
        
        ppv_window = max(3, round(0.05 * seq_len))
        self.ppv_window = ppv_window
        
        # === V1 Representation (shared across all V2 variants) ===
        self.bank = FixedPatternBank(M=M, seed=seed)
        self.activity = LocalActivity(ppv_window)
        self.flavors = TransportFlavors(lag_set=(1, 2, 4, 8))
        
        P_dim = 2 * M
        self.router = nn.Sequential(
            nn.Linear(P_dim, 32), nn.ReLU(inplace=True),
            nn.Linear(32, J))
        
        # Transport projection
        self.transport_proj = nn.Conv1d(J, 32, 1)
        
        # Pattern projection
        self.pattern_proj = nn.Sequential(
            nn.Linear(P_dim, 64), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(64, 32))
        
        # Feature dimension after mean/max/std pooling: 32*3 + 32 = 128
        self.feature_dim = 32 * 3 + 32
        
        # === V2 Readout ===
        if readout_type == 'mlp':
            # R0: original V1 MLP head
            self.head = nn.Sequential(
                nn.Linear(self.feature_dim, head_width), nn.ReLU(inplace=True),
                nn.Dropout(dropout), nn.Linear(head_width, head_width // 2),
                nn.ReLU(inplace=True), nn.Dropout(dropout),
                nn.Linear(head_width // 2, num_classes))
            self.readout = None
        elif readout_type == 'ridge':
            # R2: differentiable ridge
            self.head = None
            self.readout = DifferentiableRidge(num_classes, lam=lam)
        elif readout_type == 'class_weighted_ridge':
            # R3: class-weighted ridge
            self.head = None
            self.readout = ClassWeightedRidge(num_classes, lam=lam)
        elif readout_type == 'kernel_ridge':
            # R5: kernelized routed readout
            self.head = None
            self.readout = KernelRidgeReadout(num_classes, J=J, lam=lam)
        else:
            raise ValueError(f"Unknown readout_type: {readout_type}")
        
        self.soft_routing = (variant == 'R4')
    
    def _compute_representation(self, x, diag=False):
        """Compute V1 representation (shared across all V2 variants).
        
        Returns dict with:
            h: [B, D] final pooled features (for MLP readout)
            Z: [B, D] same as h (alias for readout interface)
            w: [B, J, T] routing weights
            Tv: [B, J, T] transport features
            P: [B, 2M, T] pattern representation
            F_T: [B, 32, T] transport features (projected)
            F_P: [B, 32, T] pattern features (projected)
        """
        B, _, T = x.shape
        
        # Pattern bank
        R = self.bank(x)  # [B, M, T']
        A, S = self.activity(R)  # [B, M, T']
        P = torch.cat([A, torch.log1p(S)], dim=1)  # [B, 2M, T']
        
        # Transport flavors
        Tv = self.flavors(x, window=self.ppv_window)  # [B, J, T]
        
        # Align temporal dimensions
        Tp = min(P.shape[-1], Tv.shape[-1])
        P, Tv = P[..., :Tp], Tv[..., :Tp]
        
        # Router
        w = torch.softmax(self.router(P.permute(0, 2, 1)), dim=-1)
        w = w.permute(0, 2, 1)  # [B, J, T]
        
        # Soft routing for V2-C
        if self.soft_routing:
            w_used = F.softmax(self.router(P.permute(0, 2, 1)) / self.tau, dim=-1)
            w_used = w_used.permute(0, 2, 1)
        else:
            # Hard top-K routing
            top2 = w.topk(min(self.topk, self.J), dim=1).indices
            mask = torch.zeros_like(w).scatter_(1, top2, 1.0)
            kept = w * mask * Tv
            rest = w.sum(1, keepdim=True) - (w * mask).sum(1, keepdim=True)
            w_used = torch.cat([kept, rest], dim=1)  # [B, J+1, T]
        
        # Transport features
        if self.soft_routing:
            F_T = self.transport_proj((w_used * Tv).unsqueeze(1))  # [B, 32, T]
        else:
            F_T = self.transport_proj(w_used)  # [B, 32, T]
        
        # Pattern features
        F_P = self.pattern_proj(P.permute(0, 2, 1)).permute(0, 2, 1)  # [B, 32, T]
        
        # Pool
        mu = F_T.mean(-1)
        mx = F_T.max(-1).values
        sd = F_T.std(-1) if F_T.shape[-1] > 1 else torch.zeros_like(mu)
        h = torch.cat([mu, mx, sd, F_P.mean(-1)], dim=1)  # [B, D]
        
        out = dict(h=h, Z=h, w=w, Tv=Tv, P=P, F_T=F_T, F_P=F_P)
        if diag:
            out.update(A=A, S=S, R=R)
        return out
    
    def forward(self, x, Y_long=None, return_aux=False):
        """Forward pass.
        
        Args:
            x: [B, 1, T] input
            Y_long: [B] labels (only needed for ridge readout during training)
            return_aux: if True, return auxiliary info
            
        Returns:
            logits: [B, C]
        """
        rep = self._compute_representation(x)
        
        if self.readout is not None:
            # V2 readout (ridge or kernel ridge)
            if self.readout_type == 'kernel_ridge':
                # Kernel ridge needs special handling (train-time vs test-time)
                # For now, fall back to linear features
                logits = self.readout.ridge(rep['Z'], Y_long) if Y_long is not None \
                    else rep['Z'] @ self.readout.ridge.W_star if hasattr(self.readout.ridge, 'W_star') \
                    else rep['Z']
            else:
                logits = self.readout(rep['Z'], Y_long) if Y_long is not None \
                    else rep['Z']  # During eval, W_star should be cached
        else:
            # V1 MLP head
            logits = self.head(rep['Z'])
        
        if return_aux:
            return logits, rep
        return logits
    
    def extract_representation(self, x):
        """Extract V1 representation without readout (for caching)."""
        with torch.no_grad():
            rep = self._compute_representation(x, diag=True)
        return rep
    
    def fit_ridge_readout(self, Z_train, Y_train, Z_val=None, Y_val=None,
                          lam_grid=None):
        """Fit the ridge readout on training features.
        
        For V2-A: fits ridge with given lambda
        For V2-B: fits class-weighted ridge
        For lambda selection: uses validation set
        
        Args:
            Z_train: [N_train, D] training features
            Y_train: [N_train] training labels
            Z_val: [N_val, D] validation features (for lambda selection)
            Y_val: [N_val] validation labels
            lam_grid: list of lambda values to try
            
        Returns:
            best_lam: selected lambda
            val_mf1: validation MF1 at best lambda
        """
        if self.readout_type == 'kernel_ridge':
            return self._fit_kernel_ridge(Z_train, Y_train, Z_val, Y_val, lam_grid)
        
        if lam_grid is None:
            lam_grid = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
        
        best_lam, best_mf1 = None, -1
        
        # Standardize features with TRAIN-only stats (stored for inference)
        Z_t = torch.from_numpy(Z_train).float() if isinstance(Z_train, np.ndarray) else Z_train.float()
        Y_t = torch.from_numpy(Y_train).long() if isinstance(Y_train, np.ndarray) else Y_train.long()
        feat_mean = Z_t.mean(0)
        feat_std = Z_t.std(0).clamp_min(1e-6)
        Z_t = (Z_t - feat_mean) / feat_std
        
        # Compute W* in closed form
        C = self.num_classes
        D = Z_t.shape[1]
        Y_onehot = F.one_hot(Y_t, C).to(Z_t.dtype)
        
        if self.readout_type == 'class_weighted_ridge':
            _cw = ClassWeightedRidge(self.num_classes, lam=1.0)
            _cw.set_weights_from_train(Y_train)
            w = _cw.class_weights[Y_t]  # per-sample weights [N]
            Zw = Z_t * w.unsqueeze(1)
        else:
            Zw = Z_t
        
        for lam in lam_grid:
            if self.readout_type == 'class_weighted_ridge':
                # Weighted ridge normal equations: A = Z^T W Z, b = Z^T W y.
                # FIX: the old code used b = Z^T W (W y) -> double weighting -> R3 collapse.
                A = Zw.T @ Z_t + (lam + 1e-6) * torch.eye(D)
                ZtY = Zw.T @ Y_onehot
            else:
                A = Z_t.T @ Z_t + (lam + 1e-6) * torch.eye(D)
                ZtY = Z_t.T @ Y_onehot
            
            W_star = torch.linalg.solve(A, ZtY)
            
            # Evaluate on validation (standardized with TRAIN stats)
            if Z_val is not None and Y_val is not None:
                Z_v = torch.from_numpy(Z_val).float() if isinstance(Z_val, np.ndarray) else Z_val.float()
                Y_v = torch.from_numpy(Y_val).long() if isinstance(Y_val, np.ndarray) else Y_val.long()
                Z_v = (Z_v - feat_mean) / feat_std
                
                logits = Z_v @ W_star
                pred = logits.argmax(dim=-1).numpy()
                
                from sklearn.metrics import f1_score
                mf1 = f1_score(Y_v.numpy(), pred, average='macro', zero_division=0,
                              labels=list(range(C)))
                
                if mf1 > best_mf1:
                    best_mf1 = mf1
                    best_lam = lam
        
        # Refit with best lambda on full train
        if best_lam is not None:
            lam = best_lam
        else:
            lam = lam_grid[len(lam_grid) // 2]
        
        if self.readout_type == 'class_weighted_ridge':
            self.readout = ClassWeightedRidge(self.num_classes, lam=lam)
            self.readout.set_weights_from_train(Y_train)
        else:
            self.readout = DifferentiableRidge(self.num_classes, lam=lam)
        
        # Store W_star (+ train standardization stats) for inference
        if self.readout_type == 'class_weighted_ridge':
            A = Zw.T @ Z_t + (lam + 1e-6) * torch.eye(D)
            ZtY = Zw.T @ Y_onehot
        else:
            A = Z_t.T @ Z_t + (lam + 1e-6) * torch.eye(D)
            ZtY = Z_t.T @ Y_onehot
        
        self.readout.register_buffer('W_star', torch.linalg.solve(A, ZtY))
        self.readout.register_buffer('feat_mean', feat_mean)
        self.readout.register_buffer('feat_std', feat_std)
        
        return best_lam, best_mf1
    
    def predict_with_ridge(self, Z_test):
        """Predict using cached W_star (applies train-fit standardization)."""
        if hasattr(self.readout, 'W_star'):
            Z = Z_test
            if hasattr(self.readout, 'feat_mean'):
                mu = self.readout.feat_mean.to(Z_test.device)
                sd = self.readout.feat_std.to(Z_test.device)
                Z = (Z - mu) / sd
            W = self.readout.W_star.to(Z_test.device)
            logits = Z @ W
            return F.softmax(logits, dim=-1)
        return None
    
    def _fit_kernel_ridge(self, Z_train, Y_train, Z_val=None, Y_val=None, lam_grid=None):
        """Fit kernel ridge readout."""
        if lam_grid is None:
            lam_grid = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
        
        Z_t = torch.from_numpy(Z_train).float() if isinstance(Z_train, np.ndarray) else Z_train
        Y_t = torch.from_numpy(Y_train).long() if isinstance(Y_train, np.ndarray) else Y_train
        
        best_lam, best_mf1 = None, -1
        
        for lam in lam_grid:
            readout = KernelRidgeReadout(self.num_classes, J=self.J, lam=lam)
            
            if Z_val is not None and Y_val is not None:
                Z_v = torch.from_numpy(Z_val).float() if isinstance(Z_val, np.ndarray) else Z_val
                Y_v = torch.from_numpy(Y_val).long() if isinstance(Y_val, np.ndarray) else Y_val
                
                probs = readout(Z_t, Y_t, Z_v)
                pred = probs.argmax(dim=-1).numpy()
                
                from sklearn.metrics import f1_score
                mf1 = f1_score(Y_v.numpy(), pred, average='macro', zero_division=0,
                              labels=list(range(self.num_classes)))
                
                if mf1 > best_mf1:
                    best_mf1 = mf1
                    best_lam = lam
        
        lam = best_lam if best_lam is not None else 1.0
        self.readout = KernelRidgeReadout(self.num_classes, J=self.J, lam=lam)
        
        # Cache training data for kernel ridge
        self.readout.register_buffer('Z_train', Z_t)
        self.readout.register_buffer('Y_train', Y_t)
        
        # Compute and cache alpha
        K_train = self.readout.kernel_matrix(Z_t)
        reg = lam * torch.eye(Z_t.shape[0], device=Z_t.device)
        Y_onehot = F.one_hot(Y_t, self.num_classes).to(Z_t.dtype)
        alpha = torch.linalg.solve(K_train + reg, Y_onehot)
        self.readout.register_buffer('alpha', alpha)
        
        return best_lam, best_mf1
    
    def predict_kernel_ridge(self, Z_test):
        """Predict using cached kernel ridge."""
        if hasattr(self.readout, 'alpha') and hasattr(self.readout, 'Z_train'):
            K_test = self.readout.kernel_matrix(Z_test, self.readout.Z_train.to(Z_test.device))
            logits = K_test @ self.readout.alpha.to(Z_test.device)
            return F.softmax(logits, dim=-1)
        return None
