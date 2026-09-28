"""TURS-SKB kernel families: deterministic fixed temporal kernels.

Design contract (spec sections 5-11):
- 0 trainable parameters; every kernel parameter comes from a predefined,
  deterministic grid. No randomness, no test labels, no fitting.
- All base kernels are unit-L2-normalized discrete sequences.
- Families F1-F4 are signed linear kernels; F5 (Energy) is a NONLINEAR
  energy/envelope operator (weights q >= 0, applied to squared signal) and
  is documented as mathematically distinct from linear convolution.

Response convention (identical to MiniROCKET's model):
  r(t) = sum_j k_j x_{t + j d}   for a base kernel k and dilation d.
Banks enumerate (base kernel, dilation) pairs from fixed grids; the
extractor caps each pair's dilation at the largest power of two that fits
the series length (the same clip rule canonical MiniROCKET applies to its
dilation grid).

Support cap: base kernels have support <= 128 taps. Dataset lengths
T=140 (ECG5000) and T=1024 (CWRU) both admit the full dilation grid
(max base support 128 -> d=1 always fits; larger d supported at both).
"""
import numpy as np

# ============================================================================
# Deterministic parameter grids (predefined; NOT fitted, NOT random)
# ============================================================================
SIGMA_GRID = [1.0, 2.0, 4.0, 8.0, 16.0]
DILATION_GRID = [1, 2, 4, 8]
ALPHA_DO_G_GRID = [0.5, 1.0, 1.5]
SKEW_GRID = [-1.0, 0.0, +1.0]
DERIV_GRID = [1, 2]
WAVELET_TYPES = ["haar", "mexicanhat", "morlet"]
WAVELET_SCALES = [1, 2, 4, 8]
GABOR_F_GRID = [1.0 / 8, 1.0 / 16, 1.0 / 32]
GABOR_SIGMA_GRID = [2.0, 4.0, 8.0, 16.0]
GABOR_PHASE_GRID = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
ENERGY_W_GRID = [4, 8, 16, 32]
ENERGY_LAMBDA_GRID = [0.0, 0.5, 1.0]

MAX_SUPPORT = 128  # hard cap on base-kernel support (taps)

# Larger dilation grids for families whose base kernels are short
# (spec sec. 8: derivative family kept deliberately small; wavelet/gabor
# get the wider scale coverage).
LONG_DILATION_GRID = [1, 2, 4, 8, 16, 32, 64, 128]


def _norm(k):
    """Unit-L2 normalization; zero vector falls back to unit 1-norm shape."""
    k = np.asarray(k, dtype=np.float64)
    n = float(np.sqrt(np.sum(k ** 2)))
    if n < 1e-12:
        n = float(np.sum(np.abs(k))) or 1.0
        return k / n
    return k / n


def _support(sigma):
    """Truncate Gaussian-type kernels at ~+-4 sigma, capped at MAX_SUPPORT."""
    return int(min(2 * int(np.ceil(4 * sigma)) + 1, MAX_SUPPORT))


# ============================================================================
# F1 Morphological / shape kernels
# ============================================================================
def gaussian_kernel(sigma, skew=0.0):
    """(Possibly skewed) Gaussian bump; skew != 0 gives an asymmetric pulse.

    sigma: width. skew in [-1, 1]: piecewise sigma (left/right), a
    deterministic normalized asymmetric-bump basis (spec sec. 7).
    """
    sup = _support(max(sigma, 1.0 + abs(skew)))
    u = np.arange(sup, dtype=np.float64) - (sup - 1) / 2.0
    sig_l = max(sigma * (1.0 - 0.35 * skew), 0.5)   # left half-width
    sig_r = max(sigma * (1.0 + 0.35 * skew), 0.5)   # right half-width
    sig = np.where(u < 0, sig_l, sig_r)
    k = np.exp(-(u ** 2) / (2.0 * sig ** 2))
    return _norm(k)


def difference_of_gaussians_kernel(sigma, alpha):
    """DoG: exp(-u^2/(2 sigma^2)) - alpha * exp(-u^2/(2 (2 sigma)^2)).

    Detects peaks/valleys of a characteristic width (onset/offset contrast).
    """
    sup = _support(2 * sigma)
    u = np.arange(sup, dtype=np.float64) - (sup - 1) / 2.0
    g1 = np.exp(-(u ** 2) / (2.0 * sigma ** 2))
    g2 = np.exp(-(u ** 2) / (2.0 * (2.0 * sigma) ** 2))
    return _norm(g1 - alpha * g2)


def gauss_deriv_kernel(sigma, order):
    """Continuous derivative-of-Gaussian edge/transition operator (base).

    order 1: -d/du Gaussian (antisymmetric, rising-edge detector)
    order 2: second derivative (Laplacian-of-Gaussian, peakedness)
    Dilation is applied by the response convention, not here.
    """
    sup = _support(sigma)
    u = np.arange(sup, dtype=np.float64) - (sup - 1) / 2.0
    g = np.exp(-(u ** 2) / (2.0 * sigma ** 2))
    if order == 1:
        k = -(u / sigma ** 2) * g
    elif order == 2:
        k = ((u ** 2 / sigma ** 4) - (1.0 / sigma ** 2)) * g
    else:
        raise ValueError(f"unsupported derivative order: {order}")
    return _norm(k)


# ============================================================================
# F2 Derivative kernels
# ============================================================================
def derivative_kernel(order):
    """Centered finite-difference base kernel of order 1 or 2.

    order 1: [-1, 0, +1] / 2    (slope)
    order 2: [+1, -2, +1] / 2   (curvature)
    Dilation is applied by the response convention, not here.
    """
    if order == 1:
        base = np.array([-1.0, 0.0, 1.0]) / 2.0
    elif order == 2:
        base = np.array([1.0, -2.0, 1.0]) / 2.0
    else:
        raise ValueError(f"unsupported derivative order: {order}")
    return _norm(base)


# ============================================================================
# F3 Wavelet kernels
# ============================================================================
def haar_kernel(scale):
    """Discrete Haar psi at scale a: taps -0.5 at 0, +0.5 at a (support 2a+1)."""
    a = int(scale)
    k = np.zeros(2 * a + 1, dtype=np.float64)
    k[0] = -0.5
    k[a] = 0.5
    return _norm(k)


def mexican_hat_kernel(scale):
    """Mexican Hat (Ricker) psi(u) = (1-u^2) exp(-u^2/2), 1/sqrt(a) psi(u/a)."""
    a = float(scale)
    sup = _support(4 * a)
    u = np.arange(sup, dtype=np.float64) - (sup - 1) / 2.0
    psi = (1.0 - (u / a) ** 2) * np.exp(-((u / a) ** 2) / 2.0)
    return _norm(psi / np.sqrt(a))


def morlet_kernel(scale):
    """Morlet psi(u) = exp(-u^2/2) cos(5u); 1/sqrt(a) psi(u/a)."""
    a = float(scale)
    sup = _support(max(2 * a, 4))
    u = np.arange(sup, dtype=np.float64) - (sup - 1) / 2.0
    psi = np.exp(-((u / a) ** 2) / 2.0) * np.cos(5.0 * u / a)
    return _norm(psi / np.sqrt(a))


def wavelet_kernel(wtype, scale):
    if wtype == "haar":
        return haar_kernel(scale)
    if wtype == "mexicanhat":
        return mexican_hat_kernel(scale)
    if wtype == "morlet":
        return morlet_kernel(scale)
    raise ValueError(f"unsupported wavelet type: {wtype}")


# ============================================================================
# F4 Gabor / oscillatory kernels
# ============================================================================
def gabor_kernel(freq, sigma, phase):
    """k(u) = exp(-u^2/(2 sigma^2)) cos(2 pi f u + phase), truncated, unit-L2."""
    sup = _support(3 * sigma)
    u = np.arange(sup, dtype=np.float64) - (sup - 1) / 2.0
    k = np.exp(-(u ** 2) / (2.0 * sigma ** 2)) * \
        np.cos(2.0 * np.pi * freq * u + phase)
    return _norm(k)


# ============================================================================
# F5 Energy / burst kernels  (NONLINEAR operator - see module docstring)
# ============================================================================
def tent_window(n):
    """Deterministic nonnegative tent profile of length n, sum-normalized.

    q_j = clip(1 - |j - c| / (n/2 + 1), 0, inf), then sum-normalized to 1;
    peak at the window center. Used for both short and long energy windows.
    """
    n = int(n)
    c = (n - 1) / 2.0
    q = np.clip(1.0 - np.abs(np.arange(n) - c) / (n / 2.0 + 1.0), 0.0, None)
    return q / q.sum()


def energy_kernel(w, lam):
    """Energy-envelope weights q >= 0 for the burst operator (sec. 11).

    Returns (q_short, w_short, w_long, lam):
      E_short(t) = sum_j q_j x_{t+j}^2   (tent window w)
      E_long(t)  = sum_j q'_j x_{t+j}^2  (tent window w_long = 4w)
      burst B(t) = E_short(t) - lam * E_long(t)   (lambda >= 0)
    Both windows are sum-normalized (mean-energy operators), so lambda is a
    dimensionless relative weight; the long window is derived
    deterministically as 4w.
    """
    w = int(w)
    w_l = int(4 * w)
    return tent_window(w), w, w_l, float(lam)


# ============================================================================
# Full deterministic banks: (base_kernel, meta) with dilation in meta
# ============================================================================
def build_morphological_bank():
    """[(kernel, meta)] for F1.

    Grid: sigma {1,2,4,8,16} x dil {1,2,4,8} x 2 polarities = 40 Gaussian
    bumps; + sigma x dil x 3 alpha = 60 DoG; + 2 orders x sigma x dil = 40
    Gaussian-derivative ops. Total 140 kernels.
    """
    bank = []
    for s in SIGMA_GRID:
        for d in DILATION_GRID:
            for pol in (1.0, -1.0):
                bank.append((gaussian_kernel(s, 0.0),
                             {"kind": "gauss", "sigma": s, "dilation": d,
                              "polarity": pol}))
    for s in SIGMA_GRID:
        for d in DILATION_GRID:
            for al in ALPHA_DO_G_GRID:
                bank.append((difference_of_gaussians_kernel(s, al),
                             {"kind": "dog", "sigma": s, "dilation": d,
                              "alpha": al}))
    for order in (1, 2):
        for s in SIGMA_GRID:
            for d in DILATION_GRID:
                bank.append((gauss_deriv_kernel(s, order),
                             {"kind": "gaussderiv", "order": order,
                              "sigma": s, "dilation": d}))
    return bank


def build_derivative_bank():
    """[(kernel, meta)] for F2.

    Grid: 2 orders x 8 dilations {1..128 pow2} x 2 polarities = 32.
    Pure discrete derivative operators (spec sec. 8); kept deliberately
    small so the family cannot dominate by feature-count (sec. 8 note).
    """
    bank = []
    for order in DERIV_GRID:
        for d in LONG_DILATION_GRID:
            for pol in (1.0, -1.0):
                bank.append((derivative_kernel(order),
                             {"kind": "deriv", "order": order,
                              "dilation": d, "polarity": pol}))
    return bank


def build_wavelet_bank():
    """[(kernel, meta)] for F3.

    Grid: 3 types x 4 scales x 8 dilations {1..128 pow2} x 2 polarities
    = 192 kernels. All fixed analytic wavelets (spec sec. 9).
    """
    bank = []
    for wt in WAVELET_TYPES:
        for a in WAVELET_SCALES:
            for d in LONG_DILATION_GRID:
                for pol in (1.0, -1.0):
                    bank.append((wavelet_kernel(wt, a),
                                 {"kind": "wavelet", "type": wt,
                                  "scale": a, "dilation": d,
                                  "polarity": pol}))
    return bank


def build_gabor_bank():
    """[(kernel, meta)] for F4.

    Grid: 3 freq x 4 sigma {2,4,8,16} x 4 phase {0, pi/2, pi, 3pi/2}
    x 4 dilations = 192 kernels (cosine; sine quadrature captured by the
    pi/2 and 3pi/2 phase shifts).
    """
    bank = []
    for f in GABOR_F_GRID:
        for sg in GABOR_SIGMA_GRID:
            for ph in GABOR_PHASE_GRID:
                for d in DILATION_GRID:
                    bank.append((gabor_kernel(f, sg, ph),
                                 {"kind": "gabor", "freq": f, "sigma": sg,
                                  "phase": ph, "dilation": d}))
    return bank


def build_energy_bank():
    """[(q_short, meta)] for F5. Grid: 4 w x 3 lambda = 12 operators.

    Energy operators are expensive (window^2 work), so the family is kept
    small; w_long = 4w is derived deterministically. Documented as nonlinear
    (the signal is squared before the moving weighted average).
    """
    bank = []
    for w in ENERGY_W_GRID:
        for lam in ENERGY_LAMBDA_GRID:
            qs, ws, wl, lm = energy_kernel(w, lam)
            bank.append((qs, {"kind": "energy", "w_short": ws,
                              "w_long": wl, "lambda": lm}))
    return bank


BANK_BUILDERS = {
    "morphological": build_morphological_bank,
    "derivative": build_derivative_bank,
    "wavelet": build_wavelet_bank,
    "gabor": build_gabor_bank,
    "energy": build_energy_bank,
}
