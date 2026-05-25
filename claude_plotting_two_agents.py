import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator

# ── Legend locations ───────────────────────────────────────────────────────────
# Options: "upper left", "upper right", "lower left", "lower right", "center left" etc.
TWO_AGENT_TOP_LEGEND_LOC = "upper left"
TWO_AGENT_MID_LEGEND_LOC = "upper right"
TWO_AGENT_BOT_LEGEND_LOC = "upper left"

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
    ax_top.plot(x, kls_mean, color=C_BLUE,  lw=2.0, label="KL divergence")
    ax_top.fill_between(x, kls_lo, kls_hi,  color=C_BLUE,  alpha=0.12, label="KL ±1 std")
    ax_top.plot(x, var_mean, color=C_GREEN, lw=2.0, label="Norm of OM covariances")
    ax_top.fill_between(x, var_lo, var_hi,  color=C_GREEN, alpha=0.12, label="Cov ±1 std")

    ax_top.set_yscale("log")
    ax_top.set_ylim(1e-10, 10**3)
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
    _styled_legend(ax_top, loc="upper right")
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


# ── Two-agent palettes: 4 shades per agent ─────────────────────────────────────
# Agent 1: blues, Agent 2: reds
AGENT1_COLORS = ["#1d4ed8", "#3b82f6", "#93c5fd", "#bfdbfe"]
AGENT2_COLORS = ["#b91c1c", "#ef4444", "#fca5a5", "#fee2e2"]


def two_agent_plotter(kls_eps_1, kls_eps_2, var_eps_1, var_eps_2,
                      policy_eps_1, policy_eps_2,
                      bac_eps_1, bac_eps_2):
    """
    Three-panel plot for a two-agent POMDP experiment.

    Parameters
    ----------
    kls_eps_1    : (n_runs, n_eps)    KL divergence for agent 1
    kls_eps_2    : (n_runs, n_eps)    KL divergence for agent 2
    var_eps_1    : (n_runs, n_eps)    Norm of OM covariances for agent 1
    var_eps_2    : (n_runs, n_eps)    Norm of OM covariances for agent 2
    policy_eps_1 : (n_runs, n_eps, 4) Action probabilities for agent 1
    policy_eps_2 : (n_runs, n_eps, 4) Action probabilities for agent 2
    bac_eps_1    : (n_runs, n_eps)    Best action count for agent 1
    bac_eps_2    : (n_runs, n_eps)    Best action count for agent 2
    """
    kls_eps_1    = np.array(kls_eps_1)
    kls_eps_2    = np.array(kls_eps_2)
    var_eps_1    = np.array(var_eps_1)
    var_eps_2    = np.array(var_eps_2)
    policy_eps_1 = np.array(policy_eps_1)
    policy_eps_2 = np.array(policy_eps_2)
    bac_eps_1    = np.array(bac_eps_1)
    bac_eps_2    = np.array(bac_eps_2)

    x    = np.arange(kls_eps_1.shape[1])
    CLIP = 1e-10

    # ── Means and stds ────────────────────────────────────────────────────────
    kls1_mean, kls1_std = np.mean(kls_eps_1, axis=0), np.std(kls_eps_1, axis=0)
    kls2_mean, kls2_std = np.mean(kls_eps_2, axis=0), np.std(kls_eps_2, axis=0)
    var1_mean, var1_std = np.mean(var_eps_1, axis=0), np.std(var_eps_1, axis=0)
    var2_mean, var2_std = np.mean(var_eps_2, axis=0), np.std(var_eps_2, axis=0)

    p1_mean = np.mean(policy_eps_1, axis=0)
    p1_std  = np.std(policy_eps_1,  axis=0)
    p2_mean = np.mean(policy_eps_2, axis=0)
    p2_std  = np.std(policy_eps_2,  axis=0)

    bac1_mean, bac1_std = np.mean(bac_eps_1, axis=0), np.std(bac_eps_1, axis=0)
    bac2_mean, bac2_std = np.mean(bac_eps_2, axis=0), np.std(bac_eps_2, axis=0)

    # ── Clip for log scale ────────────────────────────────────────────────────
    def _clip(mean, std):
        return (np.clip(mean, CLIP, None),
                np.clip(mean - std, CLIP, None),
                np.clip(mean + std, CLIP, None))

    kls1_mean, kls1_lo, kls1_hi = _clip(kls1_mean, kls1_std)
    kls2_mean, kls2_lo, kls2_hi = _clip(kls2_mean, kls2_std)
    var1_mean, var1_lo, var1_hi = _clip(var1_mean, var1_std)
    var2_mean, var2_lo, var2_hi = _clip(var2_mean, var2_std)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 12), facecolor=BG)
    gs  = gridspec.GridSpec(3, 1, hspace=0.10, figure=fig,
                            top=0.91, bottom=0.07, left=0.09, right=0.93)

    ax_top = fig.add_subplot(gs[0])
    ax_mid = fig.add_subplot(gs[1], sharex=ax_top)
    ax_bot = fig.add_subplot(gs[2], sharex=ax_top)

    N_TICKS = 5

    # ── Top: KL + covariance norm for both agents (log) ───────────────────────
    ax_top.plot(x, kls1_mean, color=AGENT1_COLORS[0], lw=2.0, label="KL — Agent 1")
    ax_top.fill_between(x, kls1_lo, kls1_hi, color=AGENT1_COLORS[0], alpha=0.12, label="±1 std")
    ax_top.plot(x, kls2_mean, color=AGENT2_COLORS[0], lw=2.0, label="KL — Agent 2")
    ax_top.fill_between(x, kls2_lo, kls2_hi, color=AGENT2_COLORS[0], alpha=0.12, label="±1 std")
    ax_top.plot(x, var1_mean, color=AGENT1_COLORS[1], lw=2.0, linestyle="--", label="Cov norm — Agent 1")
    ax_top.fill_between(x, var1_lo, var1_hi, color=AGENT1_COLORS[1], alpha=0.12, label="±1 std")
    ax_top.plot(x, var2_mean, color=AGENT2_COLORS[1], lw=2.0, linestyle="--", label="Cov norm — Agent 2")
    ax_top.fill_between(x, var2_lo, var2_hi, color=AGENT2_COLORS[1], alpha=0.12, label="±1 std")
    ax_top.set_yscale("log")
    ax_top.set_ylim(1e-10, 1e3)
    ax_top.set_ylabel("Log scale", color=MUTED, fontsize=10)
    ax_top.yaxis.grid(True, color=GRID_COLOR, linewidth=0.7, linestyle="--", alpha=1.0)
    ax_top.spines["left"].set_edgecolor(GRID_COLOR)
    _styled_legend(ax_top, loc=TWO_AGENT_TOP_LEGEND_LOC)

    ACTION_NAMES = ["Down", "Up", "Left", "Right"]

    # ── Middle: action probabilities ──────────────────────────────────────────
    for a in range(4):
        ax_mid.plot(x, p1_mean[:, a], color=AGENT1_COLORS[a], lw=1.8,
                    label=f"{ACTION_NAMES[a]}")
        ax_mid.fill_between(x, p1_mean[:, a] - p1_std[:, a],
                               p1_mean[:, a] + p1_std[:, a],
                            color=AGENT1_COLORS[a], alpha=0.12)
    for a in range(4):
        ax_mid.plot(x, p2_mean[:, a], color=AGENT2_COLORS[a], lw=1.8,
                    label=f"{ACTION_NAMES[a]}")
        ax_mid.fill_between(x, p2_mean[:, a] - p2_std[:, a],
                               p2_mean[:, a] + p2_std[:, a],
                            color=AGENT2_COLORS[a], alpha=0.12)
    ax_mid.set_ylim(-0.05, 1.05)
    ax_mid.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_mid.set_ylabel("Action probability", color=MUTED, fontsize=10)
    ax_mid.yaxis.grid(True, color=GRID_COLOR, linewidth=0.7, linestyle="--", alpha=1.0)
    _styled_legend(ax_mid, loc=TWO_AGENT_MID_LEGEND_LOC, ncol=2)

    # ── Bottom: best action counts ────────────────────────────────────────────
    def _nice_ticks(data_max, n_ticks=N_TICKS):
        if data_max <= 0:
            data_max = 1.0
        raw_step = data_max / (n_ticks - 1)
        magnitude = 10 ** np.floor(np.log10(raw_step))
        step = np.ceil(raw_step / magnitude) * magnitude
        ticks = np.arange(0, step * n_ticks, step)
        return ticks, step

    ax_bot.plot(x, bac1_mean, color=AGENT1_COLORS[0], lw=2.0, label="Agent 1 best action count")
    ax_bot.fill_between(x, bac1_mean - bac1_std, bac1_mean + bac1_std,
                        color=AGENT1_COLORS[0], alpha=0.12, label="Agent 1 ±1 std")
    ax_bot.plot(x, bac2_mean, color=AGENT2_COLORS[0], lw=2.0, label="Agent 2 best action count")
    ax_bot.fill_between(x, bac2_mean - bac2_std, bac2_mean + bac2_std,
                        color=AGENT2_COLORS[0], alpha=0.12, label="Agent 2 ±1 std")

    bac_max = max((bac1_mean + bac1_std).max(), (bac2_mean + bac2_std).max()) * 1.05
    bac_ticks, bac_step = _nice_ticks(bac_max)
    ax_bot.set_ylim(-1.25 * bac_step, bac_ticks[-1])
    ax_bot.set_yticks(np.concatenate([[-bac_step], bac_ticks]))
    ax_bot.set_ylabel("Best action count", color=MUTED, fontsize=10)
    ax_bot.set_xlabel("Episode", fontsize=10, color=MUTED)
    ax_bot.yaxis.grid(True, color=GRID_COLOR, linewidth=0.7, linestyle="--", alpha=1.0)
    _styled_legend(ax_bot, loc=TWO_AGENT_BOT_LEGEND_LOC)

    # ── Integer x ticks ───────────────────────────────────────────────────────
    n_eps  = len(x)
    x_step = max(1, int(np.ceil(n_eps / 10)))
    ax_bot.set_xticks(np.arange(0, n_eps, x_step))

    plt.setp(ax_top.get_xticklabels(), visible=False)
    plt.setp(ax_mid.get_xticklabels(), visible=False)

    # ── Style ─────────────────────────────────────────────────────────────────
    _apply_style(fig, [ax_top, ax_mid, ax_bot])

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.95, "Two-Agent POMDP — Training Curves",
             ha="center", fontsize=13, color=TEXT_COLOR, fontweight="bold")
    fig.text(0.5, 0.928, "Mean ± 1 std across runs",
             ha="center", fontsize=9, color=MUTED)

    plt.show()
