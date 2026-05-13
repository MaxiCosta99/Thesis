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

low_y_lim = 1e-10 #1e-4
high_y_lim = 10**3 #1e+2
upper_plot_legend_loc = "upper right"
plot_title = "Observation Model Learning — Baseline EM"


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

    CLIP = 1e-10

    # Clip to avoid log(0) explosions
    kls_mean  = np.clip(kls_mean,  CLIP, None)
    kls_lo    = np.clip(kls_mean - kls_std,  CLIP, None)
    kls_hi    = np.clip(kls_mean + kls_std,  CLIP, None)
    var_mean  = np.clip(var_mean,  CLIP, None)
    var_lo    = np.clip(var_mean - var_std,  CLIP, None)
    var_hi    = np.clip(var_mean + var_std,  CLIP, None)

    fig = plt.figure(figsize=(11, 8), facecolor=BG)
    gs  = gridspec.GridSpec(2, 1, hspace=0.08, figure=fig,
                            top=0.88, bottom=0.10, left=0.09, right=0.93)

    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

    N_TICKS = 5

    # ── Top: KL and covariance norm on shared log axis ────────────────────────
    ax_top.plot(x, kls_mean, color=C_BLUE,  lw=2.0, label="KL(true A || learned A)")
    ax_top.fill_between(x, kls_lo, kls_hi,  color=C_BLUE,  alpha=0.12, label="KL ±1 std")
    ax_top.plot(x, var_mean, color=C_GREEN, lw=2.0, label="Norm of OM covariances")
    ax_top.fill_between(x, var_lo, var_hi,  color=C_GREEN, alpha=0.12, label="Cov ±1 std")

    ax_top.set_yscale("log")
    ax_top.set_ylim(low_y_lim, high_y_lim)
    ax_top.set_ylabel("Log scale", color=MUTED, fontsize=10)
    ax_top.yaxis.grid(True, color=GRID_COLOR, linewidth=0.7, linestyle="--", alpha=1.0)
    ax_top.spines["left"].set_edgecolor(GRID_COLOR)

    # ── Helper: nice ticks ────────────────────────────────────────────────────
    def _nice_ticks(data_max, n_ticks=N_TICKS):
        if data_max <= 0:
            data_max = 1.0
        raw_step = data_max / (n_ticks - 1)
        magnitude = 10 ** np.floor(np.log10(raw_step))
        step = np.ceil(raw_step / magnitude) * magnitude
        ticks = np.arange(0, step * n_ticks, step)
        return ticks, step

    # ── Bottom: best action count (linear) ────────────────────────────────────
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
    x_step = max(1, int(np.ceil(n_eps / 10)))
    ax_bot.set_xticks(np.arange(0, n_eps, x_step))

    plt.setp(ax_top.get_xticklabels(), visible=False)

    # ── Style ─────────────────────────────────────────────────────────────────
    _apply_style(fig, [ax_top, ax_bot])

    # ── Legends ───────────────────────────────────────────────────────────────
    _styled_legend(ax_top, loc= upper_plot_legend_loc)
    _styled_legend(ax_bot, loc="lower right")

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.93, plot_title,
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

    titles   = ["True observation model",
                "Learned observation model"]
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
