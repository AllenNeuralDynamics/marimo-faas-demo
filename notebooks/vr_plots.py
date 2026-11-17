"""Figures and the PDF report for the VR-foraging daily summary.

Separate from :mod:`vr_data` so the notebook can show the stats first and pay the ~3 s
matplotlib import (in the browser) only afterwards.
"""

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from vr_data import OUTCOMES, align_to_events, leave_probability

# Colors: patch palette and event colors from the lab's VR-foraging dashboard/viewer, for parity.
PATCH_COLORMAP = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d"]
INK, MUTED, SHADE = "#222222", "#898781", "#ecebe6"
REWARD_COLOR, CHOICE_COLOR, LICK_COLOR = "#1f77b4", "#d62728", "#2ca02c"
# Outcome colors, validated as a set: adjacent CVD separation dE >= 15.
OUTCOME_COLORS = dict(zip(OUTCOMES, ["#2a78d6", "#e34948", "#eda100", "#e87ba4", "#d6d5cf"], strict=True))

plt.rcParams.update(
    {
        "figure.dpi": 100,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#c3c2b7",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "font.size": 9,
        "legend.frameon": False,
    }
)


def patch_colors(labels) -> dict[str, str]:
    """Stable color per patch label: sorted labels take palette slots in order. Never cycles."""
    unique = sorted(set(labels))
    if len(unique) > len(PATCH_COLORMAP):
        raise ValueError(f"{len(unique)} patch types > {len(PATCH_COLORMAP)} palette slots")
    return dict(zip(unique, PATCH_COLORMAP[: len(unique)], strict=True))


def plot_outcome_raster(reward_sites: pd.DataFrame):
    """Patch x site grid colored by outcome: whole session on top, one panel per patch type below."""
    labels = sorted(reward_sites["patch_label"].unique())
    widths = [reward_sites.loc[reward_sites["patch_label"] == lab, "patch_index"].nunique() for lab in labels]
    n_sites = int(reward_sites["site_index_in_patch_by_type"].max()) + 1
    cell = {"height": 1, "width": 1, "edgecolor": "white", "linewidth": 0.6}
    colors = reward_sites["outcome"].map(OUTCOME_COLORS).astype(str)

    fig = plt.figure(figsize=(11, 6), layout="constrained")
    grid = fig.add_gridspec(2, len(labels), width_ratios=widths)
    top = fig.add_subplot(grid[0, :])
    top.bar(reward_sites["patch_index"], bottom=reward_sites["site_index_in_patch_by_type"], color=colors, **cell)
    for boundary in reward_sites.loc[reward_sites["block_index"].diff() > 0, "patch_index"]:
        top.axvline(boundary - 0.5, color=INK, linestyle="--", linewidth=1)
    top.set(xlabel="Patch #", ylabel="Site in patch", ylim=(-0.2, n_sites))
    top.set_title("Choices, in session order (dashed = block change)")

    for col, label in enumerate(labels):
        ax = fig.add_subplot(grid[1, col])
        sub = reward_sites.loc[reward_sites["patch_label"] == label]
        ax.bar(sub["patch_index_by_type"], bottom=sub["site_index_in_patch_by_type"], color=colors[sub.index], **cell)
        ax.set(title=label, ylim=(-0.2, n_sites), xlabel="Patch # (this type)")
        ax.title.set_fontsize(8)
        if col:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("Site in patch")

    handles = [Patch(facecolor=c, label=k) for k, c in OUTCOME_COLORS.items()]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
    return fig


def plot_aligned_velocity(reward_sites, position, colors, threshold, window=(-1.0, 3.0)):
    """Velocity around odor onset at each patch's first site: one panel per patch type, median + IQR."""
    first = reward_sites.loc[
        reward_sites["site_index_in_patch_by_type"].eq(0) & reward_sites["odor_onset_time"].notna()
    ]
    groups = list(first.groupby("patch_label"))
    ts, vel = position["timestamp"].to_numpy(), position["velocity"].to_numpy()

    fig, axes = plt.subplots(
        1, len(groups), figsize=(max(2.3 * len(groups), 9), 3), sharey=True, layout="constrained", squeeze=False
    )
    for ax, (label, group) in zip(axes[0], groups, strict=True):
        t, traces = align_to_events(ts, vel, group["odor_onset_time"].to_numpy(), window)
        lo, med, hi = np.nanpercentile(traces, [25, 50, 75], axis=0)
        ax.axvspan(window[0], 0, color=SHADE, linewidth=0)
        ax.fill_between(t, lo, hi, color=colors[label], alpha=0.25, linewidth=0)
        ax.plot(t, med, color=colors[label], linewidth=2)
        if threshold is not None:
            ax.axhline(threshold, color=MUTED, linestyle="--", linewidth=1)
        ax.set(xlim=window, xlabel="Time from odor onset (s)", title=f"{label} (n={len(group)})")
        ax.title.set_fontsize(8)
    axes[0, 0].set_ylabel("Velocity (cm/s)")
    note = f"; dashed = stop threshold {threshold:g} cm/s" if threshold is not None else ""
    fig.suptitle(
        f"Velocity at odor onset, first site of each patch (median, IQR{note})",
        x=0.01,
        ha="left",
        fontsize=10,
        fontweight="bold",
    )
    return fig


def plot_stop_rate(reward_sites, colors, smoothing: int = 15):
    """Rolling P(stop) over session time, one line per patch type."""
    t0 = reward_sites["start_time"].min()
    fig, ax = plt.subplots(figsize=(11, 3), layout="constrained")
    for label, group in reward_sites.groupby("patch_label"):
        rolling = group["stopped"].astype(float).rolling(smoothing, min_periods=1, center=True).mean()
        ax.plot((group["start_time"] - t0) / 60, rolling, color=colors[label], linewidth=2, label=label)
    ax.set(ylim=(-0.03, 1.03), xlabel="Session time (min)", ylabel="P(stop)")
    ax.set_title(f"Stop probability over the session (rolling {smoothing} sites)")
    ax.grid(axis="y", color=SHADE)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    return fig


def plot_leave_heatmap(reward_sites, min_visits: int = 5):
    """P(leave) by rewards and consecutive failures before the site; sparse bins left blank."""
    p_leave = leave_probability(reward_sites, min_visits=min_visits)
    fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
    image = ax.imshow(p_leave.to_numpy(dtype=float), origin="lower", cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(p_leave.shape[1]), p_leave.columns)
    ax.set_yticks(range(p_leave.shape[0]), p_leave.index)
    ax.set(xlabel="Consecutive failures before site", ylabel="Rewards before site (in patch)")
    ax.set_title(f"P(leave) (bins with < {min_visits} visits blank)")
    fig.colorbar(image, ax=ax, label="P(leave)", shrink=0.8)
    return fig


def plot_timeline(sites, position, licks, colors, start: float, width: float = 60.0):
    """``width`` seconds of the session from ``start``: licks/cues/rewards above, velocity over site bands below."""
    t0 = sites["start_time"].min()
    lo, hi = t0 + start, t0 + start + width
    in_view = sites.loc[(sites["stop_time"] >= lo) & (sites["start_time"] <= hi)]

    fig, (events, vel) = plt.subplots(2, 1, figsize=(11, 3.6), sharex=True, height_ratios=[1, 3], layout="constrained")
    for row in in_view.itertuples():  # only the handful of sites in view
        band = colors.get(row.patch_label, SHADE) if row.site_label == "RewardSite" else SHADE
        vel.axvspan(row.start_time - t0, row.stop_time - t0, color=band, alpha=0.3, linewidth=0)
    window = position.loc[position["timestamp"].between(lo, hi)]
    vel.plot(window["timestamp"] - t0, window["velocity"], color=INK, linewidth=1)
    vel.set(ylabel="Velocity (cm/s)", xlabel="Session time (s)", xlim=(start, start + width))

    lick_times = licks.loc[licks["is_lick_onset"] & licks["timestamp"].between(lo, hi), "timestamp"] - t0
    events.eventplot(
        [lick_times, in_view["choice_cue_time"].dropna() - t0, in_view["reward_onset_time"].dropna() - t0],
        colors=[LICK_COLOR, CHOICE_COLOR, REWARD_COLOR],
        lineoffsets=[2, 1, 0],
        linelengths=0.7,
    )
    events.set_yticks([2, 1, 0], ["Lick", "Choice cue", "Reward"])
    events.spines["left"].set_visible(False)
    handles = [Patch(facecolor=c, alpha=0.4, label=k) for k, c in colors.items()]
    vel.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), title="Odor site")
    return fig


def pdf_report(session_id: str, stats: dict, table: pd.DataFrame, figures: list) -> bytes:
    """A multi-page PDF: a summary page, then one page per figure."""
    from matplotlib.backends.backend_pdf import PdfPages  # lazy: ~1 s to import in the browser

    cover = plt.figure(figsize=(11, 8.5))
    cover.text(0.05, 0.92, f"VR foraging daily summary: {session_id}", fontsize=16, fontweight="bold")
    lines = [
        f"Odor sites visited: {stats['odor_sites']}",
        f"Stopped: {stats['stop_rate']:.0%} ({stats['stops']} stops)",
        f"Stops rewarded: {stats['reward_rate']:.0%} ({stats['rewards']} rewards)",
        f"Water collected: {stats['water_ul']:.0f} µL",
        f"Distance run: {stats['distance_m']:.0f} m",
        f"Duration: {stats['duration_min']:.0f} min",
    ]
    cover.text(0.05, 0.86, "\n".join(lines), fontsize=11, va="top", linespacing=1.8)
    ax = cover.add_axes([0.05, 0.08, 0.9, 0.4])
    ax.axis("off")
    ax.table(cellText=table.astype(str).to_numpy(), colLabels=list(table.columns), loc="upper left").scale(1, 1.6)

    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        for fig in [cover, *figures]:
            pdf.savefig(fig)
    plt.close(cover)
    return buffer.getvalue()
