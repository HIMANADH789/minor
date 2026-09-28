"""
Generate DRTN figures (spec sec. 20) from saved run artifacts:
  fig_trajectories.png  : input waveform + discrete code k(t) for 3 val examples
  fig_usage.png         : code usage histograms R2/R3/R4/R5 + entropy evolution
Reads results/drtn_haptics_seed42/{R*}/; writes results/drtn_haptics_seed42/figures/.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

BASE = os.path.join(ROOT, "results", "drtn_haptics_seed42")
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

RUNGS = ["R0", "R1", "R2", "R3", "R4", "R5"]


def load(rung):
    p = os.path.join(BASE, rung, "result.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def usage_evolution(rung, field="usage"):
    """Extract the codebook usage histogram per epoch from history."""
    r = load(rung)
    if r is None:
        return None
    out = []
    for h in r.get("history", []):
        u = h.get("usage")
        out.append(u if u is not None else None)
    return out


def main():
    # ---------------- FIGURE 1: trajectories ----------------
    tpath = os.path.join(BASE, "R5", "trajectories_val.npy")
    ipath = os.path.join(BASE, "R5", "inputs_val.npy")
    if os.path.exists(tpath) and os.path.exists(ipath):
        trajs = np.load(tpath)          # (n_examples, T)
        inputs = np.load(ipath)         # (n_examples, 1, T)
        n_show = min(3, len(trajs))
        fig, axes = plt.subplots(2 * n_show, 1,
                                 figsize=(11, 2.1 * n_show), sharex=False)
        for i in range(n_show):
            ax_w = axes[2 * i]
            ax_c = axes[2 * i + 1]
            x = inputs[i, 0]
            k = trajs[i]
            T = len(x)
            ax_w.plot(np.arange(T), x, lw=0.6, color="#1f77b4")
            ax_w.set_ylabel("z-norm signal")
            ax_w.set_title(f"val example {i}", fontsize=9, loc="left")
            ax_c.step(np.arange(T), k, where="post", lw=0.9,
                      color="#d62728")
            ax_c.set_ylabel("code k(t)")
            ax_c.set_xlabel("time step")
            ax_c.set_yticks(range(0, 8, 1))
            # shade code segments for readability
            boundaries = np.nonzero(np.diff(k))[0]
            for b in boundaries:
                ax_w.axvline(b, color="gray", alpha=0.25, lw=0.5)
                ax_c.axvline(b, color="gray", alpha=0.25, lw=0.5)
        fig.suptitle("DRTN regime trajectories: waveform + discrete code k(t) "
                     "(R5, validation)", y=0.995)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "fig_trajectories.png"), dpi=160)
        plt.close(fig)
        print("wrote fig_trajectories.png")

    # ---------------- FIGURE 2: usage histograms + entropy evolution --------
    vq_rungs = ["R2", "R3", "R4", "R5"]
    fig, axes = plt.subplots(2, len(vq_rungs),
                             figsize=(3.2 * len(vq_rungs), 6))
    for j, rung in enumerate(vq_rungs):
        r = load(rung)
        # final usage histogram (val)
        u = (r or {}).get("codebook_final_usage_val")
        ax = axes[0, j]
        if u is not None:
            ax.bar(range(len(u)), u, color="#2ca02c")
            ax.set_title(f"{rung} final usage", fontsize=9)
            ax.set_ylim(0, 1)
            ne = (r or {}).get("final_diag_val", {}).get("normalized_entropy")
            ax.text(0.02, 0.95, f"H_norm={ne:.3f}" if ne is not None else "",
                    transform=ax.transAxes, va="top", fontsize=8)
        ax.set_xlabel("code")
        if j == 0:
            ax.set_ylabel("usage fraction")
        # entropy evolution across epochs
        ax = axes[1, j]
        ev = usage_evolution(rung)
        Hs = [None if u is None else float(-sum(p * np.log(p) for p in u if p > 0))
              for u in ev]
        xs = [i + 1 for i, h in enumerate(Hs) if h is not None]
        ys = [h for h in Hs if h is not None]
        if xs:
            ax.plot(xs, ys, marker="o", ms=3, color="#ff7f0e")
        ax.set_title(f"{rung} usage entropy", fontsize=9)
        ax.set_xlabel("epoch")
        if j == 0:
            ax.set_ylabel("H(q) nats")
    fig.suptitle("Codebook population diagnostics across the ablation ladder")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_usage.png"), dpi=160)
    plt.close(fig)
    print("wrote fig_usage.png")


if __name__ == "__main__":
    main()
