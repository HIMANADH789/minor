"""Phase 25 — deterministic Stack case studies (no cherry-picking; median-
representative sample per category). Illustrations, never statistical proof."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config as C
from src.diagnostics import perturb as P
from .experiments import _predict_stack
from src.diagnostics.experiments_b import _perturb_region

plt.rcParams.update({"figure.dpi": 130, "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False})


def build_case_studies(tag, model, ds, device, ext, n_per_cat=C.CASE_STUDY_N):
    e = ext["test"]
    y = e["y"]
    X = ds["Xte"][:, 0, :]
    rng = np.random.default_rng(C.CASE_SEED)
    correct = e["final_correct"] == 1
    u_med = np.median(e["u_cs_mean"])
    dis_med = np.median(e["js_disagreement"])
    nov_med = np.median(e["novelty"])
    v_med = np.median(e["s_cs_max"])
    al_med = np.median(e["alpha_cs"])

    cats = {
        "correct_high_agreement": correct & (e["n_agree_max"] >= 3),
        "correct_high_disagreement": correct & (e["js_disagreement"] > dis_med),
        "incorrect_high_agreement": (~correct) & (e["n_agree_max"] >= 3),
        "incorrect_high_disagreement": (~correct) & (e["js_disagreement"] > dis_med),
        "high_intrinsic_u": e["u_cs_mean"] > u_med,
        "low_intrinsic_u": e["u_cs_mean"] <= u_med,
        "strong_velocity": e["s_cs_max"] > v_med,
        "strong_alpha_transport": e["alpha_cs"] > al_med,
        "high_novelty": e["novelty"] > nov_med,
        "strong_branch_disagreement": e["js_disagreement"] > dis_med,
    }
    out_dir = os.path.join(C.CASE_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    bp = e["branch_probs"]                     # [4, N, C]
    ratio = X.shape[1] / e["u_t_cs"].shape[1]
    w = max(1, int(round(C.FAITH_PRIMARY_FRAC * X.shape[1])))

    for cat, mask in cats.items():
        values = {"high_intrinsic_u": e["u_cs_mean"],
                  "low_intrinsic_u": -e["u_cs_mean"],
                  "strong_velocity": e["s_cs_max"],
                  "strong_alpha_transport": e["alpha_cs"],
                  "high_novelty": e["novelty"]}.get(cat, e["js_disagreement"])
        mm = mask.copy()
        for j in range(n_per_cat):
            cand = np.where(mm)[0]
            if len(cand) == 0:
                break
            k = int(cand[np.argmin(np.abs(values[cand] - np.median(values[cand])))])
            mm[k] = False
            x0 = X[k]
            base = _predict_stack(model, x0[None, None, :], device)[0]
            sm = np.repeat(e["u_t_cs"][k][:, 0], int(np.ceil(ratio)))[:len(x0)]
            tstart = int(np.argmax(np.convolve(sm, np.ones(w), mode="valid")))
            xt = _perturb_region(x0, C.FAITH_PERTURB, tstart, tstart + w, rng)
            pt = _predict_stack(model, xt[None, None, :], device)[0]
            rs = int(rng.integers(0, len(x0) - w))
            xr = _perturb_region(x0, C.FAITH_PERTURB, rs, rs + w, rng)
            pr = _predict_stack(model, xr[None, None, :], device)[0]

            case = dict(dataset=tag, category=cat, sample_id=k,
                        true_label=int(y[k]),
                        final_prediction=int(e["final_pred"][k]),
                        branch_predictions={b: int(e["branch_pred"][bi, k])
                                            for bi, b in enumerate(C.BRANCH_NAMES)},
                        branch_probs={b: [round(float(v), 4) for v in
                                          bp[bi, k]] for bi, b in enumerate(C.BRANCH_NAMES)},
                        final_probs=[round(float(v), 4) for v in base],
                        confidence=float(e["final_conf"][k]),
                        entropy=float(e["final_entropy"][k]),
                        js_disagreement=float(e["js_disagreement"][k]),
                        u_cs_mean=float(e["u_cs_mean"][k]),
                        v_cs_max=float(e["s_cs_max"][k]),
                        alpha_cs=float(e["alpha_cs"][k]),
                        novelty=float(e["novelty"][k]),
                        targeted_region=[int(tstart), int(tstart + w)],
                        targeted_prob_drop=float(base[y[k]] - pt[y[k]]),
                        random_region=[int(rs), int(rs + w)],
                        random_prob_drop=float(base[y[k]] - pr[y[k]]))
            fig, axes = plt.subplots(4, 1, figsize=(8.6, 7.4), sharex=True)
            axes[0].plot(x0, lw=0.7)
            axes[0].axvspan(tstart, tstart + w, color="#D65F5F", alpha=0.3,
                            label="targeted")
            axes[0].axvspan(rs, rs + w, color="#888", alpha=0.3, label="random")
            axes[0].legend(fontsize=6)
            axes[0].set_title(
                f"{tag} · {cat} · s{k} · y={y[k]} pred={e['final_pred'][k]} "
                f"conf={e['final_conf'][k]:.2f} JS={e['js_disagreement'][k]:.3f} "
                f"u={e['u_cs_mean'][k]:.3f} α={e['alpha_cs'][k]:.3f} "
                f"nov={e['novelty'][k]:.3f}", fontsize=8)
            axes[1].plot(np.linalg.norm(e["v_t_cs"][k], axis=1), color="#6ACC64")
            axes[1].set_ylabel("||v_t|| cs")
            axes[2].plot(e["u_t_cs"][k][:, 0], color="#D65F5F")
            axes[2].set_ylabel("u_t cs")
            axes[3].plot(e["beta_cs"][k], lw=0.9)
            axes[3].set_ylabel("beta cs"); axes[3].set_xlabel("t")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"case_{cat}_{j}.png"),
                        bbox_inches="tight")
            plt.close(fig)
            manifest.append(case)
    with open(os.path.join(out_dir, "case_study_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest
