import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
import os


plot_title = "EM in the two maze case"
plot_name = "test"

top_ylim = [1e-1,1e4]
mid_ylim = [-0.25,1.1]
bot_ylim = [-0.5,4]


# ── Legend locations ───────────────────────────────────────────────────────────
# Options: "upper left", "upper right", "lower left", "lower right", "center left" etc.
TWO_AGENT_TOP_LEGEND_LOC = "lower left"
TWO_AGENT_MID_LEGEND_LOC = "upper left"
TWO_AGENT_BOT_LEGEND_LOC = "lower right"

# ── Palette ────────────────────────────────────────────────────────────────────
BG         = "#ffffff" #"fafafa"
PANEL_BG   = "#ffffff"
GRID_COLOR = "#e0e0e0"
TEXT_COLOR = "#000000" #1a1a2e"
MUTED      = "#000000"#6b7280"

C_BLUE   = "#2563eb"   # KL
C_GREEN  = "#16a34a"   # variance
C_ORANGE = "#dc2626"   # best action count


# ── Two-agent palettes: 4 shades per agent ─────────────────────────────────────
# Agent 1: blues, Agent 2: reds
AGENT1_COLORS = ["#1d4ed8", "#3b82f6", "#93c5fd", "#bfdbfe"]
AGENT2_COLORS = ["#b91c1c", "#ef4444", "#fca5a5", "#000000"]#fee2e2"]





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


    # ── Layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 12), facecolor=BG)
    gs  = gridspec.GridSpec(3, 1, hspace=0.10, figure=fig,
                            top=0.91, bottom=0.07, left=0.09, right=0.93,
                            height_ratios=[1, 2, 1])

    ax_top = fig.add_subplot(gs[0])
    ax_mid = fig.add_subplot(gs[1], sharex=ax_top)
    ax_bot = fig.add_subplot(gs[2], sharex=ax_top)


    # ── Top: KL + covariance norm for both agents (log) ───────────────────────
    #ax_top.plot(x, kls1_mean, color=AGENT1_COLORS[0], lw=2.0, label=r'$\mathrm{KL}\left[ P_{true}(o|s)\| P_{\theta}(o|s)\right]$')
    #ax_top.fill_between(x, kls1_mean - kls1_std, kls1_mean + kls1_std, color=AGENT1_COLORS[0], alpha=0.12)#, label="±1 std")
    ax_top.plot(x, kls2_mean, color=AGENT2_COLORS[0], lw=2.0, label=r"$\mathrm{KL}\left[ P_{true}(o|s)\| P_{\theta}(o|s)\right]$")
    ax_top.fill_between(x, kls2_mean - kls2_std, kls2_mean + kls2_std, color=AGENT2_COLORS[0], alpha=0.12)#, label="±1 std")
    #ax_top.plot(x, var1_mean, color=AGENT1_COLORS[1], lw=2.0, linestyle="--", label="Cov norm blue agent")
    #ax_top.fill_between(x, var1_mean - var1_std, var1_mean + var1_std, color=AGENT1_COLORS[1], alpha=0.12)#, label="±1 std")
    #ax_top.plot(x, var2_mean , color=AGENT2_COLORS[1], lw=2.0, linestyle="--", label=r"Cov of $P_{\Theta'}(\theta')$")
    #ax_top.fill_between(x, var2_mean - var2_std, var2_mean + var2_std, color=AGENT2_COLORS[1], alpha=0.12)#, label="±1 std")
    ax_top.set_yscale("log")
    ax_top.set_ylim(top_ylim[0], top_ylim[1])
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
    ax_mid.set_ylim(mid_ylim[0], mid_ylim[1])
    ax_mid.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_mid.set_ylabel("Action probability", color=MUTED, fontsize=10)
    ax_mid.yaxis.grid(True, color=GRID_COLOR, linewidth=0.7, linestyle="--", alpha=1.0)
    _styled_legend(ax_mid, loc=TWO_AGENT_MID_LEGEND_LOC, ncol=2)

    # ── Bottom: best action counts ────────────────────────────────────────────


    ax_bot.plot(x, bac1_mean, color=AGENT1_COLORS[0], lw=2.0, label="Best action count blue agent")
    ax_bot.fill_between(x, bac1_mean - bac1_std, bac1_mean + bac1_std,
                        color=AGENT1_COLORS[0], alpha=0.12)#, label="±1 std")
    ax_bot.plot(x, bac2_mean, color=AGENT2_COLORS[0], lw=2.0, label="Best action count red agent")
    ax_bot.fill_between(x, bac2_mean - bac2_std, bac2_mean + bac2_std,
                        color=AGENT2_COLORS[0], alpha=0.12)#, label="±1 std")

    ax_bot.set_ylim(bot_ylim[0], bot_ylim[1])
    ax_bot.set_ylabel("Best action count", color=MUTED, fontsize=10)
    ax_bot.set_yticks([0, 1, 2, 3])
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
    fig.text(0.5, 0.95, plot_title,
             ha="center", fontsize=13, color=TEXT_COLOR, fontweight="bold")
    fig.text(0.5, 0.928, "Mean ± 1 std across runs",
             ha="center", fontsize=9, color=MUTED)


    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/{plot_name}.pdf", bbox_inches="tight", transparent=True)
    plt.show()






def unstable_EM_two_maze(kls_eps_1, kls_eps_2, var_eps_1, var_eps_2,
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


    # ── Layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 12), facecolor=BG)
    gs  = gridspec.GridSpec(3, 1, hspace=0.10, figure=fig,
                            top=0.91, bottom=0.07, left=0.09, right=0.93,
                            height_ratios=[1, 2, 1])

    ax_top = fig.add_subplot(gs[0])
    ax_mid = fig.add_subplot(gs[1], sharex=ax_top)
    ax_bot = fig.add_subplot(gs[2], sharex=ax_top)


    # ── Top: KL + covariance norm for both agents (log) ───────────────────────
    #ax_top.plot(x, kls1_mean, color=AGENT1_COLORS[0], lw=2.0, label=r'$\mathrm{KL}\left[ P_{true}(o|s)\| P(o|s)\right]$')
    #ax_top.fill_between(x, kls1_mean - kls1_std, kls1_mean + kls1_std, color=AGENT1_COLORS[0], alpha=0.12)#, label="±1 std")
    ax_top.plot(x, kls2_mean, color=AGENT2_COLORS[0], lw=2.0, label=r'$\mathrm{KL}\left[ P_{true}(o|s)\| P_{\theta}(o|s)\right]$')
    ax_top.fill_between(x, kls2_mean - kls2_std, kls2_mean + kls2_std, color=AGENT2_COLORS[0], alpha=0.12)#, label="±1 std")
    #ax_top.plot(x, var1_mean, color=AGENT1_COLORS[1], lw=2.0, linestyle="--", label=r'Norm Cov P(o|s)')
    #ax_top.fill_between(x, var1_mean - var1_std, var1_mean + var1_std, color=AGENT1_COLORS[1], alpha=0.12)#, label="±1 std")
    #ax_top.plot(x, var2_mean , color=AGENT2_COLORS[1], lw=2.0, linestyle="--", label=r'Norm Cov $P(o|s)$')
    #ax_top.fill_between(x, var2_mean - var2_std, var2_mean + var2_std, color=AGENT2_COLORS[1], alpha=0.12)#, label="±1 std")
    ax_top.set_yscale("log")
    ax_top.set_ylim(top_ylim[0], top_ylim[1])
    ax_top.set_ylabel("Log scale", color=MUTED, fontsize=10)
    ax_top.yaxis.grid(True, color=GRID_COLOR, linewidth=0.7, linestyle="--", alpha=1.0)
    ax_top.spines["left"].set_edgecolor(GRID_COLOR)
    _styled_legend(ax_top, loc=TWO_AGENT_TOP_LEGEND_LOC)

    ACTION_NAMES = ["Down", "Up", "Left", "Right"]

    # ── Middle: action probabilities ──────────────────────────────────────────
    """for a in range(4):
        ax_mid.plot(x, p1_mean[:, a], color=AGENT1_COLORS[a], lw=1.8,
                    label=f"{ACTION_NAMES[a]}")
        ax_mid.fill_between(x, p1_mean[:, a] - p1_std[:, a],
                               p1_mean[:, a] + p1_std[:, a],
                            color=AGENT1_COLORS[a], alpha=0.12)"""
    for a in range(4):
        ax_mid.plot(x, p2_mean[:, a], color=AGENT2_COLORS[a], lw=1.8,
                    label=f"{ACTION_NAMES[a]}")
        ax_mid.fill_between(x, p2_mean[:, a] - p2_std[:, a],
                               p2_mean[:, a] + p2_std[:, a],
                            color=AGENT2_COLORS[a], alpha=0.12)
    ax_mid.set_ylim(mid_ylim[0], mid_ylim[1])
    ax_mid.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_mid.set_ylabel("Action probability", color=MUTED, fontsize=10)
    ax_mid.yaxis.grid(True, color=GRID_COLOR, linewidth=0.7, linestyle="--", alpha=1.0)
    _styled_legend(ax_mid, loc=TWO_AGENT_MID_LEGEND_LOC, ncol=2)

    # ── Bottom: best action counts ────────────────────────────────────────────


    #ax_bot.plot(x, bac1_mean, color=AGENT1_COLORS[0], lw=2.0, label="Best action count blue agent")
    #ax_bot.fill_between(x, bac1_mean - bac1_std, bac1_mean + bac1_std,
                        #color=AGENT1_COLORS[0], alpha=0.12)#, label="±1 std")
    ax_bot.plot(x, bac2_mean, color=AGENT2_COLORS[0], lw=2.0, label="Best action count red agent")
    ax_bot.fill_between(x, bac2_mean - bac2_std, bac2_mean + bac2_std,
                        color=AGENT2_COLORS[0], alpha=0.12)#, label="±1 std")

    ax_bot.set_ylim(bot_ylim[0], bot_ylim[1])
    ax_bot.set_ylabel("Best action count", color=MUTED, fontsize=10)
    ax_bot.set_yticks([0, 1, 2, 3])
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
    fig.text(0.5, 0.95, plot_title,
             ha="center", fontsize=13, color=TEXT_COLOR, fontweight="bold")
    """fig.text(0.5, 0.928, "Mean ± 1 std across runs",
             ha="center", fontsize=9, color=MUTED)"""


    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/{plot_name}.pdf", bbox_inches="tight", transparent=True)
    plt.show()

