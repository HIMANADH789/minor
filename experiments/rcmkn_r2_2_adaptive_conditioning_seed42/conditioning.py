"""R2.2 core: discrete VQ-code-conditioned MiniROCKET kernels.

Mathematical core (all exact, no approximation at evaluation time)
------------------------------------------------------------------
For kernel/feature f with canonical MiniRocket bias b_f and raw response
C_f(t), the canonical activation is 1[C_f(t) > b_f].  R2.2 scales the
response by a bounded, code-specific factor

    s'_{k,f} = 1 + delta_{k,f},   delta_{k,f} = beta * tanh(a_{k,f})
    beta      = beta_max * sigmoid(b),  beta_max = 0.5  =>  beta in (0, 0.5)

so s' in (0.5, 1.5): positive, no sign inversion, bounded amplification.
The conditioned activation is

    1[C * s' > b]  <=>  u > tau_{k,f}

with the SCALE-INVARIANT margin

    u_f(t) = (C_f(t) - b_f) / |b_f|          (b_f != 0)
    u_f(t) = +2 / -2 for C > 0 / C <= 0      (b_f == 0; can never flip)

and the SIGN-FOLDED effective threshold (the inequality direction depends
on the sign of b_f, so the sign is folded into the threshold):

    tau_{k,f} = sign(b_f) * ( -delta_{k,f} / (1 + delta_{k,f}) )

Derivation: b_f > 0 => activation <=> u > -delta/(1+delta);  b_f < 0 =>
the division by b_f flips the comparison => u > +delta/(1+delta).
Zero biases give threshold 0 (sign 0), matching the +-2 convention.
Because delta in (-0.5, 0.5) => |tau_base| = |delta|/(1+delta) < 1, every
position with |u| >= 1 can NEVER flip under any admissible modulation, so
clipping u to [-1, 1] is exactly lossless for threshold decisions.

Identity initialization: a = 0 => tanh = 0 => delta = 0 => tau = 0 =>
u > 0 <=> C > b: the conditioned representation is EXACTLY the R2 one.
B1 (force_zero=True) holds delta identically zero => B1 == B0 bitwise.

The audited extractor thresholds responses inline and never stores C, so
this module recompiles the same numba kernel (identical C computation,
verified by exact bool equality of the emitted activations) with an
additional float32 margin output for the 4998 heterogeneity-block
features only.
"""

import numpy as np
import torch
from numba import prange
from numba import njit

N_GLOBAL = 4998
N_FEATURES = 9996
N_HET = N_FEATURES - N_GLOBAL          # 4998 conditioned kernels
BETA_MAX = 0.5

# Predeclared training-surrogate configuration (NOT tuned; see REPORT).
HIST_N_INTERIOR = 25                   # interior bins over [-1, 1]
HIST_WIDTH = 2.0 / HIST_N_INTERIOR
HIST_S = HIST_N_INTERIOR + 2           # + below / above catch-alls
SOFT_W = 0.03                          # sigmoid width for the surrogate

_KERNEL_CACHE = {}


# ====================================================================
# Margin kernel: identical C computation to the audited raw extractor,
# plus u = (C - b_f)/|b_f| for the heterogeneity-block features.
# ====================================================================
def compile_margin_kernel():
    """Compile the margin kernel; returns (kernel, indices).

    The C computation is byte-for-byte the audited one in
    experiments/drtn_conditioned_minirocket_transfer_seed42/core.py;
    correctness is enforced by exact bool-equality audits against it.
    """
    from experiments.drtn_conditioned_minirocket_transfer_seed42.core import \
        _compile_raw_response_kernel

    _, indices = _compile_raw_response_kernel()
    n_dilations = 6
    dilations = np.array([1, 2, 4, 8, 16, 32], dtype=np.int32)
    n_features_per_dilation = np.array([768, 1536, 3072, 3072, 768, 768],
                                       dtype=np.int32)

    @njit(fastmath=True, parallel=True, cache=True)
    def _margin_kernel(X, dilations, n_features_per_dilation, biases, indices,
                       n_global):
        n_cases, T = X.shape
        n_kernels = len(indices)
        n_dil = len(dilations)
        n_features = n_kernels * np.sum(n_features_per_dilation)
        act = np.zeros((n_cases, n_features, T), dtype=np.bool_)
        valid = np.zeros((n_features, T), dtype=np.bool_)
        u_het = np.zeros((n_cases, n_features - n_global, T),
                         dtype=np.float32)

        for i in prange(n_cases):
            _X = X[i]
            A = -_X
            G = 3.0 * _X
            f_start = 0
            for j in range(n_dil):
                _padding0 = j % 2
                dilation = dilations[j]
                padding = (8 * dilation) // 2
                n_feat_this_dil = n_features_per_dilation[j]

                C_alpha = np.zeros(T, dtype=np.float32)
                C_alpha[:] = A
                C_gamma = np.zeros((9, T), dtype=np.float32)
                C_gamma[4] = G

                start = dilation
                end = T - padding
                for gamma_index in range(4):
                    C_alpha[-end:] = C_alpha[-end:] + A[:end]
                    C_gamma[gamma_index, -end:] = G[:end]
                    end += dilation
                for gamma_index in range(5, 9):
                    C_alpha[:-start] = C_alpha[:-start] + A[start:]
                    C_gamma[gamma_index, :-start] = G[start:]
                    start += dilation

                for k in range(n_kernels):
                    f_end = f_start + n_feat_this_dil
                    _padding1 = (_padding0 + k) % 2
                    a, b, c = indices[k]
                    C = C_alpha + C_gamma[a] + C_gamma[b] + C_gamma[c]
                    if _padding1 == 0:
                        if i == 0:
                            for t in range(T):
                                valid[f_start:f_start + n_feat_this_dil, t] = \
                                    True
                        for f in range(n_feat_this_dil):
                            b_f = biases[f_start + f]
                            for t in range(T):
                                if C[t] > b_f:
                                    act[i, f_start + f, t] = True
                            if f_start + f >= n_global:
                                u_het[i, f_start + f - n_global, :] = \
                                    _margin_row(C, b_f)
                    else:
                        if i == 0:
                            for t in range(padding, T - padding):
                                valid[f_start:f_start + n_feat_this_dil, t] = \
                                    True
                        for f in range(n_feat_this_dil):
                            b_f = biases[f_start + f]
                            for t in range(padding, T - padding):
                                if C[t] > b_f:
                                    act[i, f_start + f, t] = True
                            if f_start + f >= n_global:
                                u_het[i, f_start + f - n_global,
                                      padding:T - padding] = \
                                    _margin_row(C[padding:T - padding], b_f)
                    f_start = f_end
        return act, u_het, valid

    return _margin_kernel, indices


@njit(fastmath=True, cache=True)
def _margin_row(C, b_f):
    """u = (C - b_f)/|b_f| with the +-2 zero-bias convention."""
    T = len(C)
    out = np.empty(T, dtype=np.float32)
    if b_f != 0.0:
        inv = 1.0 / abs(b_f)
        for t in range(T):
            out[t] = (C[t] - b_f) * inv
    else:
        for t in range(T):
            out[t] = 2.0 if C[t] > 0.0 else -2.0
    return out


def compute_activations_and_margins(extractor, X, chunk=16):
    """GENERATOR of per-chunk (act, u_het, valid) with the margin kernel.

    act   : (n_chunk, 9996, T) bool -- audited-identical activations
    u_het : (n_chunk, 4998, T) float32 -- scale-invariant margins
    valid : (9996, T) bool

    Memory safety: no full-split array is ever materialized (CWRU_BAL:
    3209 x 9996 x 1024 bools would be ~33 GB).  For audits/tests on tiny
    inputs use compute_activations_and_margins_full().
    """
    if "kernel" not in _KERNEL_CACHE:
        _KERNEL_CACHE["kernel"], _KERNEL_CACHE["indices"] = \
            compile_margin_kernel()
    kernel, indices = _KERNEL_CACHE["kernel"], _KERNEL_CACHE["indices"]

    X = np.ascontiguousarray(X, dtype=np.float32)
    _, _, dilations, n_features_per_dilation, biases = extractor.parameters
    dilations = np.ascontiguousarray(dilations, dtype=np.int32)
    n_features_per_dilation = np.ascontiguousarray(
        n_features_per_dilation, dtype=np.int32)
    biases = np.ascontiguousarray(biases, dtype=np.float32)

    for c0 in range(0, len(X), chunk):
        c1 = min(c0 + chunk, len(X))
        act, u_het, valid = kernel(X[c0:c1], dilations,
                                   n_features_per_dilation, biases, indices,
                                   N_GLOBAL)
        yield act, u_het, valid


def compute_activations_and_margins_full(extractor, X, chunk=16):
    """Stacked full-split (act, u_het, valid); tiny inputs ONLY."""
    acts, us, valid = [], [], None
    for act, u_het, v in compute_activations_and_margins(extractor, X,
                                                         chunk=chunk):
        acts.append(act)
        us.append(u_het)
        valid = v
    return np.concatenate(acts, axis=0), np.concatenate(us, axis=0), valid


# ====================================================================
# Exact conditioned activations from margins
# ====================================================================
def tau_from_delta(delta):
    """tau_base = -delta / (1 + delta);  delta (K, M) in (-0.5, 0.5).

    NOTE: |tau_base| < 1 for every admissible delta.  This is the UNSIGNED
    base threshold; fold the per-feature bias sign with fold_bias_sign()
    before comparing against u = (C - b)/|b|.
    """
    if isinstance(delta, torch.Tensor):
        return -delta / (1.0 + delta)
    d = np.asarray(delta, dtype=np.float64)
    return -d / (1.0 + d)


def fold_bias_sign(tau_base, bias_sign):
    """Fold per-feature sign(b_f) into the base thresholds.

    tau_base: (K, M); bias_sign: (M,) in {-1, 0, +1} from sign(b_f).
    Returns (K, M) signed thresholds; zero-bias features get threshold 0
    (their u is +-2, so they can never flip -- consistent).
    """
    bs = np.asarray(bias_sign, dtype=np.float64)
    if isinstance(tau_base, torch.Tensor):
        t = tau_base.double() * torch.as_tensor(
            bs, dtype=torch.float64, device=tau_base.device)[None, :]
        return t
    return np.asarray(tau_base, dtype=np.float64) * bs[None, :]


def signed_tau_from_delta(delta, bias_sign):
    """Effective per-(code, feature) thresholds including the bias sign:
    tau_{k,f} = sign(b_f) * (-delta/(1+delta)).  delta (K, M); returns
    (K, M) float64 (numpy) / float64 (torch).
    """
    return fold_bias_sign(tau_from_delta(delta), bias_sign)


def cond_act_from_u(u_het, tau_table, codes_chunk, valid_het):
    """Exact conditioned activations: act_cond[b, f, t] = u[b,f,t] > tau[codes[b,t], f].

    u_het      : (B, F, T) float32 margins
    tau_table  : (K, F) float32/float64 effective thresholds
    codes_chunk: (B, T) int hard VQ codes
    valid_het  : (F, T) bool (informational; invalid u stays outside H)
    Returns bool (B, F, T).
    """
    tau = np.asarray(tau_table, dtype=np.float32)
    tau_pos = tau[codes_chunk]                    # (B, T, F)
    return u_het > np.transpose(tau_pos, (0, 2, 1))


# ====================================================================
# Bounded code-modulation table
# ====================================================================
class CodeModulation(torch.nn.Module):
    """Per-(code, kernel) bounded modulation with identity initialization.

    delta_{k,m} = beta_max * sigmoid(b) * tanh(a_{k,m})  in (-0.5, 0.5)
    a init 0, b init 0  =>  delta = 0  =>  tau = 0  =>  identity (R2).

    force_zero=True implements the B1 control: delta is EXACTLY zero
    regardless of the learned parameters (no gradient can flow to a/b
    through the zero branch).
    """

    def __init__(self, K=8, M=N_HET, beta_max=BETA_MAX):
        super().__init__()
        self.K, self.M, self.beta_max = K, M, beta_max
        self.a = torch.nn.Parameter(torch.zeros(K, M))
        self.b = torch.nn.Parameter(torch.zeros(()))
        self.force_zero = False

    def delta(self):
        """(K, M) modulation deltas in (-0.5, 0.5); exactly 0 for B1."""
        if self.force_zero:
            return torch.zeros_like(self.a)
        beta = self.beta_max * torch.sigmoid(self.b)
        return beta * torch.tanh(self.a)

    def tau(self):
        """(K, M) effective thresholds tau = -delta/(1+delta)."""
        d = self.delta()
        return -d / (1.0 + d)

    def scale_bounds(self):
        """(min, max) of 1+delta over the table (must be within (0.5, 1.5))."""
        with torch.no_grad():
            d = self.delta()
            return float(1.0 + d.min()), float(1.0 + d.max())


# ====================================================================
# GPU histogram of margins x codes (training surrogate sufficient stat)
# ====================================================================
def bin_edges():
    """(HIST_S,) float32 evaluation points: below catch-all, interior bin
    centers, above catch-all."""
    e = np.empty(HIST_S, dtype=np.float32)
    e[0] = -1.0 - 0.5 * HIST_WIDTH
    for i in range(HIST_N_INTERIOR):
        e[1 + i] = -1.0 + (i + 0.5) * HIST_WIDTH
    e[-1] = 1.0 + 0.5 * HIST_WIDTH
    return e


def u_to_bin(u):
    """Vectorized bin index for margins: 0 below, 1..HIST_N_INTERIOR
    interior, HIST_S-1 above."""
    ui = np.clip(u, -1.0, 1.0)
    idx = 1 + np.floor((ui + 1.0) / HIST_WIDTH).astype(np.int64)
    idx = np.minimum(idx, 1 + HIST_N_INTERIOR - 1)
    idx = np.where(u < -1.0, 0, idx)
    idx = np.where(u > 1.0, HIST_S - 1, idx)
    return idx


def build_histograms(u_het, codes_chunk, valid_het, device):
    """Joint (feature, code, bin) counts over VALID positions for a chunk.

    u_het      : (B, F, T) float32 margins (heterogeneity block only)
    codes_chunk: (B, T) int
    valid_het  : (F, T) bool  OR  (9996, T) bool (het rows sliced off here)
    Returns int64 (B, F, K, HIST_S); invalid positions are discarded.
    """
    B, F, T = u_het.shape
    if valid_het.shape[0] != F:
        valid_het = valid_het[-F:]
    K = int(codes_chunk.max()) + 1 if codes_chunk.size else 1
    K = max(K, 8)
    dev = torch.device(device)
    u_t = torch.from_numpy(np.ascontiguousarray(u_het)).to(dev)
    bin_t = torch.from_numpy(u_to_bin(u_het)).to(dev)          # (B, F, T)
    valid_t = torch.from_numpy(np.ascontiguousarray(valid_het)).to(dev)
    codes_t = torch.from_numpy(np.ascontiguousarray(codes_chunk)).to(dev)

    # invalid positions -> dump bin HIST_S (dropped below)
    bin_t = torch.where(valid_t[None], bin_t,
                        torch.full_like(bin_t, HIST_S))
    flat = (codes_t[:, None, :] * (HIST_S + 1) + bin_t).long()  # (B, F, T)
    out = torch.zeros(B, F, K * (HIST_S + 1), dtype=torch.int32, device=dev)
    out.scatter_add_(2, flat, torch.ones_like(flat, dtype=torch.int32))
    # layout is [code0_bin0..code0_binS, code1_bin0..] -> split on the LAST
    # axis as (K, S+1) and drop the per-code dump column
    counts = out.reshape(B, F, K, HIST_S + 1)[..., :HIST_S]
    return counts.cpu().numpy().astype(np.int64)


# ====================================================================
# Differentiable surrogate features (training only)
# ====================================================================
def surrogate_H2(counts, delta, bias_sign, min_count, edges=None, w=SOFT_W):
    """Differentiable conditioned heterogeneity from histograms.

    counts : (B, F, K, S) float tensor (GPU) of signed-margin bin counts
    delta  : (K, F) tensor of modulation deltas (requires grad)
    bias_sign : (F,) sequence in {-1, 0, +1} (sign of the canonical bias)
    Returns (H2 (B, F) tensor, stats dict with detached tensors).
    """
    edges = (edges if edges is not None else torch.as_tensor(
        bin_edges(), device=counts.device, dtype=counts.dtype)).to(
        counts.device, counts.dtype)
    B, F, K, S = counts.shape
    # tau: (K, F) signed thresholds -> sigma over bins -> (K, F, S)
    tau = signed_tau_from_delta(delta, bias_sign).to(counts.device,
                                                     counts.dtype)
    sigma_kfs = torch.sigmoid((edges[None, None, :] - tau[:, :, None]) / w)
    sigma = sigma_kfs.permute(1, 0, 2)[None]         # (1, F, K, S)
    n_k = counts.sum(-1)                             # (B, F, K)
    assert sigma.shape[1:] == counts.shape[1:], "sigma/counts misalignment"
    n_valid = counts.sum((-1, -2)).clamp_min(1.0)    # (B, F)
    contrib = counts * sigma                         # (B, F, K, S), broadcast
    ppv2_k = contrib.sum(-1) / n_k.clamp_min(1.0)    # (B, F, K)
    ppv2_g = contrib.sum((-1, -2)) / n_valid         # (B, F)
    sel = (n_k >= min_count).float()                 # (B, F, K)
    any_sel = (sel.sum(-1, keepdim=True) == 0).float()
    sel = sel + any_sel * torch.nn.functional.one_hot(
        n_k.argmax(-1), K).float()                   # fallback: dominant code
    wq = torch.where(sel > 0, n_k / n_valid[..., None],
                     torch.zeros_like(n_k))
    wq = wq / wq.sum(-1, keepdim=True).clamp_min(1e-12)
    dev2 = ppv2_k - ppv2_g[..., None]                # (B, F, K)
    H2 = (wq * dev2 * dev2 * sel).sum(-1)            # (B, F)
    stats = {
        "n_k": n_k.detach(), "n_valid": n_valid.detach(),
        "ppv2_g": ppv2_g.detach(),
    }
    return H2, stats


# ====================================================================
# Diagnostics
# ====================================================================
def modulation_diagnostics(mod):
    """Predeclared modulation diagnostics from a CodeModulation."""
    with torch.no_grad():
        d = mod.delta().double().cpu().numpy()       # (K, M)
        s = np.tanh(mod.a.detach().double().cpu().numpy())
        beta = float(mod.beta_max * torch.sigmoid(mod.b).item())
        Sn = s / np.maximum(np.linalg.norm(s, axis=1, keepdims=True), 1e-12)
        cos = Sn @ Sn.T
        off = cos[~np.eye(len(cos), dtype=bool)]
        norm_s = np.linalg.norm(s, axis=1)
        return {
            "beta": round(beta, 6),
            "mean_abs_s": float(np.abs(s).mean()),
            "median_abs_s": float(np.median(np.abs(s))),
            "max_abs_s": float(np.abs(s).max()),
            "s_norms_L2": [round(float(x), 6) for x in norm_s],
            "mean_abs_delta": float(np.abs(d).mean()),
            "max_abs_delta": float(np.abs(d).max()),
            "pairwise_cos_mean": float(off.mean()),
            "pairwise_cos_max": float(off.max()),
            "fraction_near_zero_code_vec": float(
                (norm_s < 1e-3).mean()),
            "fraction_near_identical_code_vec": float(
                (off > 0.999).mean()),
            "scale_factor_min": float(1.0 + d.min()),
            "scale_factor_max": float(1.0 + d.max()),
        }


def conditioning_effect(act_before, act_after):
    """Fraction of activation positions changed by the conditioning."""
    n = act_before.size
    if n == 0:
        return {"flip_fraction": 0.0, "n_positions": 0}
    flips = int(np.count_nonzero(act_before != act_after))
    return {"flip_fraction": flips / n, "n_positions": int(n),
            "n_flips": flips}
