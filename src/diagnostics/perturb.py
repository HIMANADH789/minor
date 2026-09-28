"""Perturbation / augmentation primitives shared by the degradation,
velocity-localization, faithfulness, benign-stability and counterfactual
experiments. All operate per-sample on [L] float signals or [B, L] batches."""
import numpy as np


def gauss_noise(x, level, rng):
    """Additive Gaussian noise, level = std in units of the signal's std."""
    return x + rng.normal(0.0, level * x.std() + 1e-12, size=x.shape)


def amplitude_scale(x, level):
    """Multiply amplitude by `level` (0.6..1.4)."""
    return x * level


def baseline_shift(x, level):
    """Additive constant baseline, level = fraction of signal std."""
    return x + level * x.std()


def temporal_jitter(x, level, rng):
    """Circular shift by `level` samples."""
    return np.roll(x, int(level))


def localized_mask(x, width_frac, rng, value=None):
    """Zero (or constant) out a random contiguous region. Returns (x', start, end)."""
    L = len(x)
    w = max(1, int(round(width_frac * L)))
    start = int(rng.integers(0, L - w + 1))
    fill = value if value is not None else float(x.mean())
    out = x.copy()
    out[start:start + w] = fill
    return out, start, start + w


def localized_noise(x, width_frac, rng, level=0.5):
    """Add localized noise in a random region. Returns (x', start, end)."""
    L = len(x)
    w = max(1, int(round(width_frac * L)))
    start = int(rng.integers(0, L - w + 1))
    out = x.copy()
    out[start:start + w] += rng.normal(0, level * x.std() + 1e-12, size=w)
    return out, start, start + w


def segment_scale(x, width_frac, rng, factor=1.8):
    """Scale amplitude inside a random region. Returns (x', start, end)."""
    L = len(x)
    w = max(1, int(round(width_frac * L)))
    start = int(rng.integers(0, L - w + 1))
    out = x.copy()
    out[start:start + w] *= factor
    return out, start, start + w


def apply_spec(x, kind, level, rng):
    """Apply a named (kind, level) transformation to one signal."""
    if kind == "gaussian_noise":
        return gauss_noise(x, level, rng), None
    if kind == "amplitude_scale":
        return amplitude_scale(x, level), None
    if kind == "baseline_shift":
        return baseline_shift(x, level), None
    if kind == "temporal_jitter":
        return temporal_jitter(x, level, rng), None
    if kind == "localized_mask":
        return localized_mask(x, level, rng)
    if kind == "localized_noise":
        return localized_noise(x, level, rng)
    if kind == "segment_scale":
        return segment_scale(x, level, rng)
    raise ValueError(f"unknown perturbation kind: {kind}")


# ---------------- trivial localization baselines ----------------
def raw_first_difference(x):
    return np.abs(np.diff(x, prepend=x[0]))


def raw_local_energy(x, win=9):
    """x: [N, L] or [L] -> local mean energy per sample."""
    k = np.ones(win) / win
    if x.ndim == 1:
        return np.convolve(x ** 2, k, mode="same")
    e = np.empty_like(x)
    for i in range(x.shape[0]):
        e[i] = np.convolve(x[i] ** 2, k, mode="same")
    return e
