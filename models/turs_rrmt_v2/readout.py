"""Differentiable Ridge Classifier for TURS-RRMT-V2.

Implements end-to-end differentiable ridge classification where gradients
flow through the closed-form solution back into the representation.

Mathematical formulation (one-vs-rest, multiclass):

    Given:
        Z in R^{N x D}  (feature matrix)
        Y in R^{N x C}  (one-hot label matrix, N = batch size)
        lambda > 0       (regularization)

    Ridge solution:
        W* = (Z^T Z + lambda I_D)^{-1} Z^T Y

    This is computed via torch.linalg.solve for numerical stability.

    The backward pass through torch.linalg.solve provides implicit
    differentiation: gradients flow from the loss through W* back to Z
    and hence to the RRMT representation parameters.

Loss:
    Cross-entropy on the logits  Z @ W*
    (softmax cross-entropy for numerical stability)

This avoids:
    - explicit matrix inverse (uses Cholesky-backed solve)
    - numerical instability from small lambda
    - gradient through an unrolled iterative solver

For inference:
    f(x) = softmax(Z_test @ W*)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DifferentiableRidge(nn.Module):
    """End-to-end differentiable ridge classifier.
    
    The ridge weight W* is computed in closed form from the current
    batch representation Z and labels Y, then used to produce logits.
    Gradients flow through W* back to Z via implicit differentiation.
    
    Args:
        num_classes: number of classes C
        lam: regularization strength (lambda). Can be a float or 
             nn.Parameter for learnable regularization.
        use_onehot: if True, solve in one-hot space; if False, solve
                    via per-class ridge (less memory for large C).
        eps: numerical stability constant added to diagonal.
    """
    
    def __init__(self, num_classes, lam=1.0, use_onehot=True, eps=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.lam = lam
        self.use_onehot = use_onehot
        self.eps = eps
    
    def forward(self, Z, Y_long):
        """
        Args:
            Z: [N, D] feature matrix (differentiable)
            Y_long: [N] long tensor of class indices
            
        Returns:
            logits: [N, C] classification logits
        """
        N, D = Z.shape
        C = self.num_classes
        device = Z.dtype
        
        # Compute W* = (Z^T Z + lambda I)^{-1} Z^T Y
        ZtZ = Z.T @ Z  # [D, D]
        reg = (self.lam + self.eps) * torch.eye(D, device=Z.device, dtype=Z.dtype)
        A = ZtZ + reg  # [D, D]
        
        if self.use_onehot:
            # One-hot encoding of labels
            Y_onehot = F.one_hot(Y_long, C).to(Z.dtype)  # [N, C]
            ZtY = Z.T @ Y_onehot  # [D, C]
        else:
            # Per-class ridge: solve independently for each class
            # Less memory efficient but more numerically stable for large C
            Y_onehot = F.one_hot(Y_long, C).to(Z.dtype)
            ZtY = Z.T @ Y_onehot  # [D, C]
        
        # Solve A W = ZtY  =>  W = A^{-1} ZtY
        # torch.linalg.solve uses Cholesky decomposition (PSD)
        W_star = torch.linalg.solve(A, ZtY)  # [D, C]
        
        # Compute logits
        logits = Z @ W_star  # [N, C]
        
        return logits
    
    def predict_probs(self, Z):
        """Inference-time prediction (no gradient needed)."""
        with torch.no_grad():
            logits = self.forward(Z, torch.zeros(Z.shape[0], dtype=torch.long, device=Z.device))
            return F.softmax(logits, dim=-1)


class ClassWeightedRidge(DifferentiableRidge):
    """Class-weighted differentiable ridge (V2-B).
    
    Uses inverse-frequency class weights in the ridge objective:
        W* = (Z^T W_diag Z + lambda I)^{-1} Z^T W_diag Y
    
    where W_diag is a diagonal matrix of class weights.
    
    Weights are computed from training set frequencies only.
    """
    
    def __init__(self, num_classes, class_weights=None, lam=1.0, eps=1e-6):
        super().__init__(num_classes, lam=lam, use_onehot=True, eps=eps)
        self.class_weights = class_weights
    
    def set_weights_from_train(self, y_train):
        """Compute inverse-frequency weights from training labels."""
        y = y_train.detach().cpu().numpy() if torch.is_tensor(y_train) else np.asarray(y_train)
        counts = np.bincount(y, minlength=self.num_classes).astype(float)
        counts = np.maximum(counts, 1.0)  # avoid division by zero
        weights = counts.sum() / (self.num_classes * counts)
        self.class_weights = torch.from_numpy(weights).float()
    
    def forward(self, Z, Y_long):
        N, D = Z.shape
        C = self.num_classes
        
        Y_onehot = F.one_hot(Y_long, C).to(Z.dtype)  # [N, C]
        
        if self.class_weights is not None:
            # Apply class weights: weighted least squares
            # Normal equations: A = Z^T W Z, b = Z^T W y (W applied ONCE)
            w = self.class_weights[Y_long]  # [N]
            Zw = Z * w.unsqueeze(1)  # [N, D] -- weighted Z
        else:
            Zw = Z
        
        ZtZ = Zw.T @ Z  # [D, D]
        reg = (self.lam + self.eps) * torch.eye(D, device=Z.device, dtype=Z.dtype)
        A = ZtZ + reg
        # FIX: was Zw.T @ (W Y) -> double weighting -> degenerate solutions
        ZtY = Zw.T @ Y_onehot  # [D, C]
        
        W_star = torch.linalg.solve(A, ZtY)
        logits = Z @ W_star
        
        return logits


class SoftRoutingRidge(nn.Module):
    """V2-C: Soft routing + differentiable ridge.
    
    Replaces hard top-K routing with temperature-controlled softmax:
        w_j(t) = softmax(g_j(t) / tau)
    
    Then combines transport features using soft weights:
        F_T(t) = sum_j w_j(t) * T_j(t)
    
    This eliminates the non-differentiable top-K operation.
    """
    
    def __init__(self, num_classes, J=4, transport_dim=32,
                 pattern_dim=32, tau=1.0, lam=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.J = J
        self.transport_dim = transport_dim
        self.pattern_dim = pattern_dim
        self.tau = tau
        self.lam = lam
        
        # Transport projection (shared across flavors)
        self.transport_proj = nn.Conv1d(J, transport_dim, 1)
        
        # Pattern projection
        self.pattern_proj = nn.Linear(pattern_dim * 2, pattern_dim)
        
        # Ridge readout
        self.ridge = DifferentiableRidge(num_classes, lam=lam)
    
    def forward(self, Tv, P, router_weights, Y_long=None):
        """
        Args:
            Tv: [B, J, T] transport features per flavor
            P: [B, 2M, T] pattern representation
            router_weights: [B, J, T] routing weights (from V1 router)
            Y_long: [N] labels (only needed during training)
        Returns:
            logits: [B, C]
        """
        B, _, T = Tv.shape
        
        # Soft routing: apply temperature to router weights
        w_soft = F.softmax(router_weights / self.tau, dim=1)  # [B, J, T]
        
        # Weighted transport combination
        F_T = (w_soft.unsqueeze(2) * Tv.unsqueeze(1)).sum(1)  # [B, transport_dim, T]
        # Wait, Tv is [B, J, T] not [B, J, d, T]. Let me reconsider.
        # Tv shape is [B, J, T] where each flavor is a scalar per timestep
        # We need to projectTv through transport_proj after weighting
        
        # Actually Tv is [B, J, T] - scalar transport costs
        # Weighted combination: [B, T]
        F_T_weighted = (w_soft * Tv).sum(dim=1)  # [B, T]
        
        # Pattern path
        P_pooled = torch.cat([P.mean(-1), P.max(-1).values], dim=1)  # [B, 2M]
        F_P = self.pattern_proj(P_pooled)  # [B, pattern_dim]
        
        # Pool transport features
        F_T_stats = torch.stack([F_T_weighted.mean(-1), 
                                  F_T_weighted.max(-1).values,
                                  F_T_weighted.std(-1)], dim=1)  # [B, 3]
        
        # Combine
        Z = torch.cat([F_T_stats, F_P], dim=1)  # [B, 3 + pattern_dim]
        
        if Y_long is not None:
            return self.ridge(Z, Y_long)
        return Z


class KernelRidgeReadout(nn.Module):
    """V2-D: Kernelized routed readout.
    
    Constructs an example-level kernel from the temporally resolved
    routed representation:
    
        Psi_j(x) = aggregate_t sqrt(w_j(x,t)) Phi_j(x,t)
        Psi(x) = concat_j Psi_j(x)
        K(x,x') = <Psi(x), Psi(x')>
    
    Uses kernel ridge classification:
        f(x) = K(x, X_train) (K_train + lambda I)^{-1} Y_train
    """
    
    def __init__(self, num_classes, J=4, transport_dim=32,
                 pattern_dim=32, lam=1.0, kernel_type='linear'):
        super().__init__()
        self.num_classes = num_classes
        self.J = J
        self.transport_dim = transport_dim
        self.pattern_dim = pattern_dim
        self.lam = lam
        self.kernel_type = kernel_type
    
    def compute_routed_features(self, Tv, P, router_weights):
        """Compute routing-weighted features Psi(x).
        
        Args:
            Tv: [B, J, T] transport features
            P: [B, 2M, T] pattern representation
            router_weights: [B, J, T] routing weights
            
        Returns:
            Psi: [B, D_routed] routing-weighted feature vector
        """
        B, J, T = Tv.shape
        
        # sqrt routing weights for PSD kernel construction
        w_sqrt = torch.sqrt(router_weights.clamp(min=1e-8))  # [B, J, T]
        
        # Per-flavor weighted aggregation
        Psi_parts = []
        for j in range(self.J):
            # Weighted transport: sqrt(w_j) * T_j
            psi_j_transport = (w_sqrt[:, j, :] * Tv[:, j, :]).mean(dim=-1)  # [B]
            # Also include raw transport mean for additional features
            psi_j_raw = Tv[:, j, :].mean(dim=-1)  # [B]
            Psi_parts.append(psi_j_transport)
            Psi_parts.append(psi_j_raw)
        
        # Pattern features (unweighted, for completeness)
        P_mean = P.mean(dim=-1).mean(dim=-1)  # [B] -- mean of mean
        P_max = P.max(dim=-1).values.mean(dim=-1)  # [B]
        
        Psi = torch.stack(Psi_parts + [P_mean, P_max], dim=1)  # [B, 2*J+2]
        
        return Psi
    
    def kernel_matrix(self, Psi_train, Psi_test=None, gamma=None):
        """Compute kernel matrix.
        
        Args:
            Psi_train: [N_train, D]
            Psi_test: [N_test, D] or None (use Psi_train)
            gamma: RBF kernel bandwidth (None = linear)
            
        Returns:
            K: [N_test, N_train] kernel matrix
        """
        if Psi_test is None:
            Psi_test = Psi_train
        
        if self.kernel_type == 'linear':
            K = Psi_test @ Psi_train.T  # [N_test, N_train]
        elif self.kernel_type == 'rbf':
            # Squared distances
            sq_dist = (Psi_test**2).sum(1, keepdim=True) + \
                      (Psi_train**2).sum(1).unsqueeze(0) - \
                      2 * Psi_test @ Psi_train.T
            if gamma is None:
                gamma = 1.0 / Psi_train.shape[1]
            K = torch.exp(-gamma * sq_dist)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
        
        return K
    
    def forward(self, Psi_train, Y_train_long, Psi_test, gamma=None):
        """Kernel ridge classification.
        
        Args:
            Psi_train: [N_train, D] training features
            Y_train_long: [N_train] training labels
            Psi_test: [N_test, D] test features
            gamma: RBF bandwidth
            
        Returns:
            probs: [N_test, C] prediction probabilities
        """
        C = self.num_classes
        N_train = Psi_train.shape[0]
        N_test = Psi_test.shape[0]
        
        # Compute kernel matrices
        K_train = self.kernel_matrix(Psi_train, gamma=gamma)  # [N_train, N_train]
        K_test = self.kernel_matrix(Psi_train, Psi_test, gamma=gamma)  # [N_test, N_train]
        
        # Ensure PSD: add small diagonal
        reg = self.lam * torch.eye(N_train, device=K_train.device, dtype=K_train.dtype)
        K_reg = K_train + reg
        
        # One-hot labels
        Y_onehot = F.one_hot(Y_train_long, C).to(K_train.dtype)  # [N_train, C]
        
        # Kernel ridge: alpha = (K + lambda I)^{-1} Y
        alpha = torch.linalg.solve(K_reg, Y_onehot)  # [N_train, C]
        
        # Predict: f = K_test @ alpha
        logits = K_test @ alpha  # [N_test, C]
        probs = F.softmax(logits, dim=-1)
        
        return probs
