import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
from utils import norm_om, flatten_om
import os


# ── Palette ────────────────────────────────────────────────────────────────────
BG         = "#ffffff"
PANEL_BG   = "#ffffff"
GRID_COLOR = "#e0e0e0"
TEXT_COLOR = "#000000"#1a1a2e"
MUTED      = "#000000"#6b7280"

C_BLUE   = "#2563eb"   # KL
C_GREEN  = "#16a34a"   # variance
C_ORANGE = "#dc2626"   # best action count


plot_title = "Observation Model Learning — Soft Dirichlet Updates"
plot_name = 'Dir_acc'

top_low_y_lim = 1e-4
top_high_y_lim = 1e1
bottom_low_y_lim = -0.5
bottom_high_y_lim = 3.5
upper_plot_legend_loc = "center right"
lower_plot_legend_loc = "upper right"

# Relative panel heights: [KL/variance, best-action-count, heatmap row]
# Tweak these to resize panels, e.g. [1, 1, 2.5] makes the heatmap row bigger.
HEIGHT_RATIOS = [1.0, 1.0, 1.0]


def _apply_style(fig, axes):
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=MUTED, which="both", labelsize=9)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)
        ax.grid(True, axis="x", color=GRID_COLOR, linewidth=0.7, linestyle="-", alpha=1.0)
        ax.set_axisbelow(True)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which="minor", colors=GRID_COLOR, length=2)


def _styled_legend(ax, **kwargs):
    return ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR,
                     labelcolor=TEXT_COLOR, fontsize=8.5, framealpha=1.0, **kwargs)


def plotter(kls_eps, var_eps, bac_eps, true_A, learned_A,
                       height_ratios=None, figsize=(11, 13)):
    """
    Single figure, top to bottom:
      1. KL / covariance-norm panel (log scale)
      2. Best-action-count panel (linear)
      3. True vs. learned observation-model heatmaps (side by side)

    Resize panels with height_ratios=[h1, h2, h3], e.g. [1, 1, 2.5] for a bigger heatmap row.
    """
    height_ratios = height_ratios or HEIGHT_RATIOS

    x = np.arange(len(kls_eps[0]))
    kls_eps, var_eps, bac_eps = np.array(kls_eps), np.array(var_eps), np.array(bac_eps)
    kls_mean, kls_std = np.mean(kls_eps, axis=0), np.std(kls_eps, axis=0)
    var_mean, var_std = np.mean(var_eps, axis=0), np.std(var_eps, axis=0)
    bac_mean, bac_std = np.mean(bac_eps, axis=0), np.std(bac_eps, axis=0)

    A_norm = norm_om(flatten_om(learned_A))
    true_A_flat = flatten_om(true_A)

    fig = plt.figure(figsize=figsize, facecolor=BG)
    outer = gridspec.GridSpec(3, 1, figure=fig, height_ratios=height_ratios,
                              hspace=0.35, top=0.93, bottom=0.06, left=0.09, right=0.93)

    ax_top = fig.add_subplot(outer[0])
    ax_bot = fig.add_subplot(outer[1], sharex=ax_top)
    heat_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[2],
                                               width_ratios=[1, 1, 0.05], wspace=0.3)
    ax0 = fig.add_subplot(heat_gs[0])
    ax1 = fig.add_subplot(heat_gs[1])
    cax = fig.add_subplot(heat_gs[2])

    # ── Panel 1: KL and covariance norm (log scale) ─────────────────────────
    ax_top.plot(x, kls_mean, color=C_BLUE, lw=2.0, label=r'$\mathrm{KL}\left[ P_{true}(o|s)\| P_{avg}(o|s)\right]$')
    ax_top.fill_between(x, kls_mean+kls_std, kls_mean-kls_std, color=C_BLUE, alpha=0.12, label="KL ±1 std")
    ax_top.plot(x, var_mean, color=C_GREEN, lw=2.0, label=r'Norm Cov $P(o|s)$')
    ax_top.fill_between(x, var_mean+var_std, var_mean-var_std, color=C_GREEN, alpha=0.12, label="Cov ±1 std")
    ax_top.set_yscale("log")
    ax_top.set_ylim(top_low_y_lim, top_high_y_lim)
    ax_top.set_ylabel("Log scale", color=MUTED, fontsize=10)
    ax_top.yaxis.grid(True, color=GRID_COLOR, linewidth=0.7, linestyle="--", alpha=1.0)
    ax_top.spines["left"].set_edgecolor(GRID_COLOR)

    # ── Panel 2: best action count (linear) ──────────────────────────────────
    ax_bot.plot(x, bac_mean, color=C_ORANGE, lw=2.0, label="Best action count")
    ax_bot.fill_between(x, bac_mean-bac_std, bac_mean+bac_std, color=C_ORANGE, alpha=0.12, label="±1 std")
    ax_bot.set_ylabel("Best action count", color=C_ORANGE, fontsize=10)
    ax_bot.set_xlabel("Episode", fontsize=10, color=MUTED)
    ax_bot.tick_params(axis="y", colors=C_ORANGE)
    ax_bot.spines["left"].set_edgecolor(C_ORANGE)
    ax_bot.spines["left"].set_linewidth(1.5)
    ax_bot.set_ylim(bottom_low_y_lim, bottom_high_y_lim)
    ax_bot.set_yticks([0, 1, 2, 3])
    ax_bot.yaxis.grid(True, color=GRID_COLOR, linewidth=0.7, linestyle="--", alpha=1.0)

    n_eps = len(x)
    x_step = max(1, int(np.ceil(n_eps / 10)))
    ax_bot.set_xticks(np.arange(0, n_eps, x_step))
    plt.setp(ax_top.get_xticklabels(), visible=False)

    _styled_legend(ax_top, loc=upper_plot_legend_loc)
    _styled_legend(ax_bot, loc=lower_plot_legend_loc)

    # ── Panel 3: heatmaps ────────────────────────────────────────────────────
    im = None
    for ax, mat, title in zip([ax0, ax1], [true_A_flat, A_norm],
                              ["True observation model", "Learned observation model"]):
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, interpolation="nearest", aspect="auto")
        ax.set_title(title, color=TEXT_COLOR, fontsize=11, pad=10, fontweight="bold")
        ax.set_xlabel("State  $s$", color=MUTED, fontsize=9)
        ax.set_ylabel("Observation  $o$", color=MUTED, fontsize=9)
        ax.tick_params(colors=MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)
        ax.set_facecolor(PANEL_BG)

    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    cbar.outline.set_edgecolor(GRID_COLOR)
    cbar.set_label("Probability", color=MUTED, fontsize=9)

    _apply_style(fig, [ax_top, ax_bot, ax0, ax1])
    ax0.grid(False)
    ax1.grid(False)

    fig.suptitle(plot_title, color=TEXT_COLOR, fontsize=14, fontweight="bold", y=0.97)
    fig.text(0.5, 0.945, "Mean ± 1 std across runs · True vs. learned observation model",
             ha="center", fontsize=9, color=MUTED)

    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/{plot_name}.pdf", bbox_inches="tight", transparent=True)
    plt.show()
