"""Phase 24 — deterministic case studies from the frozen model (test split).

Categories (no cherry-picking; deterministic selection of the median-
representative sample per category):
  correct/low-u, correct/high-u, incorrect/low-u, incorrect/high-u,
  strong velocity, weak velocity, strong alpha, weak alpha.
For each: signal figure, traces, targeted vs random perturbation response.
Cases are illustrative only and never substitute aggregate statistics."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config as C
from . import perturb as P

plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


def _median_representative(values, mask):
    """Deterministic: sample whose value is closest to the category median."""
    v = values[mask]
    if len(v) == 0:
        return None
    med = np.median(v)
    candidates = np.where(mask)[0]
    return int(candidates[np.argmin(np.abs(v - med))])


def build_case_studies(tag, model, ds, device, ext, n_per_cat=C.CASE_STUDY_N):
    from .experiments_b import _predict, _perturb_region
    from .extraction import extract_split

    e = ext["test"]
    y = e["y"]; u = e["u_mean"]; v = e["s_t_max"]; al = e["alpha"][:, 0]
    correct = e["correct"] == 1
    X = ds["Xte"][:, 0, :]
    rng = np.random.default_rng(C.CASE_SEED)

    u_hi = u >= np.median(u)
    v_hi = v >= np.median(v)
    al_hi = al >= np.median(al)

    cats = {
        "correct_low_u": correct & ~u_hi,
        "correct_high_u": correct & u_hi,
        "incorrect_low_u": ~correct & ~u_hi,
        "incorrect_high_u": ~correct & u_hi,
        "strong_velocity": v_hi,
        "weak_velocity": ~v_hi,
        "strong_alpha_transport": al_hi,
        "weak_alpha_regime": ~al_hi,
    }
    out_dir = os.path.join(C.CASE_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)
    manifest = []

    ratio = X.shape[1] / e["s_t"].shape[1] if "s_t" in e else 1
    for cat, mask in cats.items():
        picks = []
        vv = u if "u" in cat else (v if "velocity" in cat else al)
        mm = mask.copy()
        for _ in range(n_per_cat):
            k = _median_representative(vv, mm)
            if k is None:
                break
            picks.append(k)
            mm[k] = False
        for j, k in enumerate(picks):
            x0 = X[k]
            base_p = _predict(model, x0[None, None, :], device)[0]

            # targeted (high s_t) vs random region perturbation
            sm = np.repeat(e["s_t"][k], int(np.ceil(ratio)))[:len(x0)]
            w = max(1, int(round(C.FAITH_PRIMARY_FRAC * len(x0))))
            tstart = int(np.argmax(np.convolve(sm, np.ones(w), mode="valid")))
            xt = _perturb_region(x0, "noise", tstart, tstart + w, rng)
            pt = _predict(model, xt[None, None, :], device)[0]
            rs = int(rng.integers(0, len(x0) - w))
            xr = _perturb_region(x0, "noise", rs, rs + w, rng)
            pr = _predict(model, xr[None, None, :], device)[0]

            case = dict(
                dataset=tag, category=cat, sample_id=int(k),
                true_label=int(y[k]), predicted_label=int(e["pred"][k]),
                confidence=float(e["confidence"][k]),
                u_mean=float(u[k]), v_max=float(v[k]), alpha=float(al[k]),
                class_probs=[float(p) for p in base_p],
                targeted_region=[int(tstart), int(tstart + w)],
                targeted_prob_drop=float(base_p[y[k]] - pt[y[k]]),
                random_region=[int(rs), int(rs + w)],
                random_prob_drop=float(base_p[y[k]] - pr[y[k]]),
            )
            # figure
            fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
            axes[0].plot(x0, lw=0.7)
            axes[0].axvspan(tstart, tstart + w, color="#D65F5F", alpha=0.3,
                            label="targeted (high s_t)")
            axes[0].axvspan(rs, rs + w, color="#888", alpha=0.3, label="random")
            axes[0].legend(fontsize=6)
            axes[0].set_title(
                f"{tag} · {cat} · sample {k} · y={y[k]} pred={e['pred'][k]} "
                f"conf={e['confidence'][k]:.2f} u={u[k]:.3f} α={al[k]:.3f}",
                fontsize=8)
            axes[1].plot(e["s_t"][k], color="#4878CF"); axes[1].set_ylabel("||v_t||")
            axes[2].plot(e["u_t"][k].mean(1), color="#D65F5F"); axes[2].set_ylabel("mean u_t")
            axes[2].set_xlabel("time (temporal-trace steps)")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"case_{cat}_{j}.png"),
                        bbox_inches="tight")
            plt.close(fig)
            manifest.append(case)

    with open(os.path.join(out_dir, "case_study_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest
