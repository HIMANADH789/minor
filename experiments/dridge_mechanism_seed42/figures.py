"""Figures for the differential-Ridge mechanism test."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _val_curve(ax, r):
    gammas = sorted(int(g) for g in r["cv_curve"])
    cv = [r["cv_curve"][str(g)]["mean_cv_macro_f1"] for g in gammas]
    sd = [r["cv_curve"][str(g)]["std_cv_macro_f1"] for g in gammas]
    ax.errorbar(gammas, cv, yerr=sd, fmt="o-", capsize=4,
                color="tab:blue", label="5-fold CV Macro-F1 (dev set)")
    ax.axhline(r["refs"]["R2_val"], ls="--", color="tab:red",
               label=f"canonical R2 val ({r['refs']['R2_val']})")
    ax.scatter([r["selected_gamma"]], [r["val_macro_f1"]], marker="*",
               s=200, color="tab:green", zorder=5,
               label=f"gamma*={r['selected_gamma']}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("gamma = alpha_H / alpha_G")
    ax.set_ylabel("Macro-F1")
    ax.legend(fontsize=8)


def make_figures(all_res, out_dir):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    h, u = all_res["Haptics"], all_res["UWaveGestureLibraryY"]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    _val_curve(ax, h)
    ax.set_title("Haptics: validation Macro-F1 vs gamma")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig1_haptics_val_vs_gamma.png"),
                dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    _val_curve(ax, u)
    ax.set_title("UWaveGestureLibraryY: validation Macro-F1 vs gamma")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig2_uwaveY_val_vs_gamma.png"),
                dpi=150)
    plt.close(fig)

    # FIG 3: Haptics M0 / R2 / DRidge
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    names = ["M0", "R2", f"DRidge(g*={h['selected_gamma']})"]
    vals = [h["refs"]["M0"], h["refs"]["R2"], h["test_macro_f1"]]
    bars = ax.bar(names, vals, color=["tab:gray", "tab:blue", "tab:green"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center",
                va="bottom", fontsize=9)
    ax.set_ylim(0.4, 0.65)
    ax.set_ylabel("official test Macro-F1")
    ax.set_title("Haptics: M0 vs R2 vs Differential Ridge")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig3_haptics_test.png"), dpi=150)
    plt.close(fig)

    # FIG 4: UWaveY M0 / R2 / R5 / DRidge (+ random-H band)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    names = ["M0", "R2", "R5 rho=0.1", f"DRidge(g*={u['selected_gamma']})"]
    vals = [u["refs"]["M0"], u["refs"]["R2"], u["refs"]["R5_test"],
            u["test_macro_f1"]]
    bars = ax.bar(names, vals,
                  color=["tab:gray", "tab:blue", "tab:orange", "tab:green"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center",
                va="bottom", fontsize=9)
    m, s = u["refs"]["random_H_mean"], u["refs"]["random_H_sd"]
    ax.axhspan(m - s, m + s, color="tab:purple", alpha=0.2,
               label=f"random-H mean ± SD ({m} ± {s})")
    ax.set_ylim(0.70, 0.82)
    ax.set_ylabel("official test Macro-F1")
    ax.set_title("UWaveGestureLibraryY: M0 vs R2 vs R5 vs Differential Ridge")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig4_uwaveY_test.png"), dpi=150)
    plt.close(fig)

    # FIG 5: ||beta_H|| vs gamma (both datasets)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for r, label, color in ((h, "Haptics", "tab:blue"),
                            (u, "UWaveY", "tab:orange")):
        gammas = sorted(int(g) for g in r["diagnostics"])
        bh = [r["diagnostics"][str(g)]["beta_H_l2"] for g in gammas]
        ax.plot(gammas, bh, "o-", label=label, color=color)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("gamma")
    ax.set_ylabel("||beta_H|| (train+val fit, original coords)")
    ax.set_title("H-block coefficient magnitude vs gamma")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig5_betaH_vs_gamma.png"), dpi=150)
    plt.close(fig)
