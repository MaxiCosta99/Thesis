import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
from utils import norm_om, flatten_om

# ── Palette ────────────────────────────────────────────────────────────────────
BG         = "#fafafa"
PANEL_BG   = "#ffffff"
GRID_COLOR = "#e0e0e0"
TEXT_COLOR = "#1a1a2e"
MUTED      = "#6b7280"

C_BLUE   = "#2563eb"   # KL
C_GREEN  = "#16a34a"   # variance
C_ORANGE = "#dc2626"   # best action count



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
    return ax.legend(
        facecolor=PANEL_BG,
        edgecolor=GRID_COLOR,
        labelcolor=TEXT_COLOR,
        fontsize=8.5,
        framealpha=1.0,
        **kwargs
    )


def A_KL_plotter(kls_eps, var_eps, bac_eps):
    x = np.arange(len(kls_eps[0]))

    kls_eps = np.array(kls_eps)
    var_eps = np.array(var_eps)
    bac_eps = np.array(bac_eps)

    kls_mean, kls_std = np.mean(kls_eps, axis=0), np.std(kls_eps, axis=0)
    var_mean, var_std = np.mean(var_eps, axis=0), np.std(var_eps, axis=0)
    bac_mean, bac_std = np.mean(bac_eps, axis=0), np.std(bac_eps, axis=0)

    fig = plt.figure(figsize=(11, 8), facecolor=BG)
    gs  = gridspec.GridSpec(2, 1, hspace=0.08, figure=fig,
                            top=0.88, bottom=0.10, left=0.09, right=0.91)

    ax_top  = fig.add_subplot(gs[0])
    ax_bot  = fig.add_subplot(gs[1], sharex=ax_top)
    ax_twin = ax_top.twinx()

    N_TICKS = 5  # number of shared horizontal gridlines

    # ── Top: KL divergence ────────────────────────────────────────────────────
    ax_top.plot(x, kls_mean, color=C_BLUE, lw=2.0, label="KL divergence")
    ax_top.fill_between(x, kls_mean - kls_std, kls_mean + kls_std,
                        color=C_BLUE, alpha=0.12, label="±1 std")
    ax_top.set_ylabel("KL divergence", color=C_BLUE, fontsize=10)
    ax_top.tick_params(axis="y", colors=C_BLUE)
    ax_top.spines["left"].set_edgecolor(C_BLUE)
    ax_top.spines["left"].set_linewidth(1.5)

    # ── Top twin: covariance norm ─────────────────────────────────────────────
    ax_twin.plot(x, var_mean, color=C_GREEN, lw=2.0, linestyle="-",
                 label="Norm of OM covariances")
    ax_twin.fill_between(x, var_mean - var_std, var_mean + var_std,
                         color=C_GREEN, alpha=0.10, label="±1 std (var)")
    ax_twin.set_ylabel("Norm of covariances", color=C_GREEN, fontsize=10)
    ax_twin.tick_params(axis="y", colors=C_GREEN)
    ax_twin.spines["right"].set_edgecolor(C_GREEN)
    ax_twin.spines["right"].set_linewidth(1.5)

    # ── Helper: nice integer ticks and y-limits with padding below 0 ──────────
    def _nice_ticks(data_max, n_ticks=N_TICKS):
        """Return evenly spaced ticks 0, step, 2*step, ...
        where step is rounded to a clean number at the right magnitude."""
        if data_max <= 0:
            data_max = 1.0
        raw_step = data_max / (n_ticks - 1)
        magnitude = 10 ** np.floor(np.log10(raw_step))
        step = np.ceil(raw_step / magnitude) * magnitude
        ticks = np.arange(0, step * n_ticks, step)
        return ticks, step

    # ── Symlog scale for ax_top and ax_twin ───────────────────────────────────
    # linthresh = linear region around 0; tuned to the convergence scale of each quantity
    ax_top.set_yscale("symlog", linthresh=0.1)
    ax_twin.set_yscale("symlog", linthresh=0.0001)
    ax_top.yaxis.grid(True, color=GRID_COLOR, linewidth=0.7, linestyle="--", alpha=1.0)
    ax_twin.yaxis.grid(False)

    # ── Bottom: best action count ─────────────────────────────────────────────
    ax_bot.plot(x, bac_mean, color=C_ORANGE, lw=2.0, label="Best action count")
    ax_bot.fill_between(x, bac_mean - bac_std, bac_mean + bac_std,
                        color=C_ORANGE, alpha=0.12, label="±1 std")
    ax_bot.set_ylabel("Best action count", color=C_ORANGE, fontsize=10)
    ax_bot.set_xlabel("Episode", fontsize=10, color=MUTED)
    ax_bot.tick_params(axis="y", colors=C_ORANGE)
    ax_bot.spines["left"].set_edgecolor(C_ORANGE)
    ax_bot.spines["left"].set_linewidth(1.5)
    bac_ticks, bac_step = _nice_ticks((bac_mean + bac_std).max() * 1.05)
    ax_bot.set_ylim(-1.25 * bac_step, bac_ticks[-1])
    ax_bot.set_yticks(np.concatenate([[-bac_step], bac_ticks]))
    ax_bot.yaxis.grid(True, color=GRID_COLOR, linewidth=0.7, linestyle="--", alpha=1.0)

    # ── Integer x ticks ───────────────────────────────────────────────────────
    n_eps = len(x)
    x_step = max(1, int(np.ceil(n_eps / 10)))  # at most ~10 ticks
    ax_bot.set_xticks(np.arange(0, n_eps, x_step))

    plt.setp(ax_top.get_xticklabels(), visible=False)

    # ── Style ─────────────────────────────────────────────────────────────────
    _apply_style(fig, [ax_top, ax_bot, ax_twin])
    ax_twin.set_facecolor("none")
    ax_twin.spines["left"].set_visible(False)
    for spine in ["top", "bottom"]:
        ax_twin.spines[spine].set_visible(False)

    # ── Legends ───────────────────────────────────────────────────────────────
    lines1, labels1 = ax_top.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax_top.legend(lines1 + lines2, labels1 + labels2,
                  facecolor=PANEL_BG, edgecolor=GRID_COLOR,
                  labelcolor=TEXT_COLOR, fontsize=8.5, loc="upper right")
    _styled_legend(ax_bot, loc="lower right")

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.93, "Observation Model Learning — Training Curves",
             ha="center", fontsize=13, color=TEXT_COLOR, fontweight="bold")
    fig.text(0.5, 0.905, "Mean ± 1 std across runs",
             ha="center", fontsize=9, color=MUTED)

    plt.show()


def A_diff_plotter(true_A, learned_A):
    A      = norm_om(flatten_om(learned_A))
    true_A = flatten_om(true_A)

    # Reserve a narrow column on the right for the colorbar
    fig = plt.figure(figsize=(11, 5), facecolor=BG)
    gs  = gridspec.GridSpec(1, 3, figure=fig,
                            width_ratios=[1, 1, 0.05],
                            left=0.07, right=0.93,
                            top=0.82, bottom=0.12,
                            wspace=0.3)

    ax0  = fig.add_subplot(gs[0])
    ax1  = fig.add_subplot(gs[1])
    cax  = fig.add_subplot(gs[2])
    axes = [ax0, ax1]

    titles   = ["True observation model  $A$",
                "Learned observation model  $\\hat{A}$"]
    matrices = [true_A, A]

    for ax, mat, title in zip(axes, matrices, titles):
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1,
                       interpolation="nearest", aspect="auto")
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

    fig.suptitle("Observation Model Comparison",
                 color=TEXT_COLOR, fontsize=13, fontweight="bold", y=0.96)

    plt.show()
