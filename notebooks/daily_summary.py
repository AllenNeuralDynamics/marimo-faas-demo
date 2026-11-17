# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.25.2",
#     "matplotlib>=3.10",
#     "numpy>=2",
#     "pandas>=2.2",
#     "pyarrow>=18",
# ]
# ///
"""Daily VR-foraging session summary.

Runs three ways from the same file:

* locally:      ``uv run marimo run notebooks/daily_summary.py``
* by URL:       ``.../?session=867425_2026-11-13_21-11-07``
* statically:   exported with ``marimo export html-wasm`` and served from GitHub Pages;
                Python runs in the viewer's browser (Pyodide), no server.

All data comes from the processed per-session assets in the public
``aind-open-data`` bucket, fetched over plain HTTPS (anonymous, CORS-enabled).
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="VR foraging - daily summary")

with app.setup:
    import asyncio
    import datetime as dt
    import io
    import re
    import sys
    import urllib.parse
    import urllib.request
    import xml.etree.ElementTree as ET

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    from matplotlib.patches import Patch

    OPEN_DATA = "https://aind-open-data.s3.us-west-2.amazonaws.com"
    DEFAULT_SESSION = "867425_2026-11-13_21-11-07"
    SESSION_RE = re.compile(r"^(?P<subject>\d+)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})$")

    # Reference categorical palette (light mode), fixed order, never cycled.
    CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    INK, INK_MUTED, GRID, NEUTRAL = "#0b0b0b", "#898781", "#e1e0d9", "#d6d5cf"

    # Site outcomes, in plotting order. Colors validated as a set (adjacent CVD dE >= 15).
    OUTCOME_COLORS = {
        "Stop, rewarded": "#2a78d6",
        "Stop, depleted": "#e34948",
        "Stop, unrewarded (probabilistic)": "#eda100",
        "Stop, aborted wait": "#e87ba4",
        "Leave": NEUTRAL,
    }

    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#c3c2b7",
            "axes.labelcolor": INK,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "font.size": 9,
            "legend.frameon": False,
        }
    )


# ---------------------------------------------------------------------------
# Data access: plain HTTPS so the same code runs under CPython and Pyodide.
# ---------------------------------------------------------------------------


@app.function
async def fetch_bytes(url: str) -> bytes:
    """GET ``url``. Uses the browser's fetch under Pyodide, urllib otherwise."""
    if sys.platform == "emscripten":
        from pyodide.http import pyfetch

        response = await pyfetch(url)
        if not response.ok:
            raise OSError(f"GET {url} -> HTTP {response.status}")
        return await response.bytes()

    def _get() -> bytes:
        with urllib.request.urlopen(url, timeout=60) as response:
            return response.read()

    return await asyncio.to_thread(_get)


@app.function
def parse_session_id(session_id: str) -> tuple[str, dt.date] | None:
    """Split ``<subject>_<YYYY-MM-DD>_<HH-MM-SS>`` into (subject, date); None if malformed."""
    match = SESSION_RE.match(session_id.strip())
    if match is None:
        return None
    return match["subject"], dt.date.fromisoformat(match["date"])


@app.function
def parse_listing(xml_bytes: bytes) -> dict[str, str]:
    """Map session_id -> URL of its newest processed asset, from an S3 ListObjectsV2 reply.

    Processed assets are named ``<session_id>_processed_<YYYY-MM-DD_HH-MM-SS>/``;
    the timestamp sorts lexically, so the max is the newest processing run.
    """
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    prefixes = [p.text.rstrip("/") for p in ET.fromstring(xml_bytes).iterfind("s3:CommonPrefixes/s3:Prefix", ns)]
    newest: dict[str, str] = {}
    for prefix in sorted(prefixes):
        session_id, sep, _ = prefix.partition("_processed_")
        if sep and SESSION_RE.match(session_id):
            newest[session_id] = f"{OPEN_DATA}/{prefix}"  # sorted -> last write wins = newest
    return newest


@app.function
async def list_processed_sessions(subject: str, day: dt.date) -> dict[str, str]:
    """All processed sessions of ``subject`` acquired on ``day`` (session_id -> asset URL)."""
    query = urllib.parse.urlencode(
        {"list-type": 2, "prefix": f"{subject.strip()}_{day.isoformat()}_", "delimiter": "/"}
    )
    return parse_listing(await fetch_bytes(f"{OPEN_DATA}/?{query}"))


@app.function
async def load_table(asset_url: str, table: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Read ``<asset>/<table>.parquet``; packaging provenance lands in ``df.attrs``."""
    arrow = pq.read_table(io.BytesIO(await fetch_bytes(f"{asset_url}/{table}.parquet")), columns=columns)
    df = arrow.to_pandas(ignore_metadata=True)  # keep e.g. `timestamp` a column, not the index
    meta = arrow.schema.metadata or {}
    df.attrs = {
        k: meta[k.encode()].decode() for k in ("packaging_version", "data_contract_version") if k.encode() in meta
    }
    return df


@app.function
def stop_velocity_threshold(software_events: pd.DataFrame) -> float | None:
    """The rig's stop threshold (cm/s) as logged by the task, or None if it was not logged."""
    logged = software_events.loc[software_events["event_name"] == "StopVelocityThreshold", "data"]
    return float(logged.iloc[0]) if len(logged) else None


# ---------------------------------------------------------------------------
# Analysis: pure functions of DataFrames, covered by tests/test_daily_summary.py
# ---------------------------------------------------------------------------


@app.function
def classify_outcomes(sites: pd.DataFrame) -> pd.Series:
    """One outcome label per site (see OUTCOME_COLORS); vectorised, no row loops."""
    stopped = sites["has_choice"].fillna(False).astype(bool)
    rewarded = sites["has_reward"].fillna(False).astype(bool)
    aborted = sites["has_waited_reward_delay"].eq(False)
    depleted = sites["reward_available"].le(0)
    # First matching rule wins; labels are spelled out so precedence can't drift from display order.
    rules = {
        "Stop, rewarded": stopped & rewarded,
        "Stop, aborted wait": stopped & aborted,
        "Stop, depleted": stopped & depleted,
        "Stop, unrewarded (probabilistic)": stopped,
    }
    labels = np.select(list(rules.values()), list(rules.keys()), default="Leave")
    return pd.Series(pd.Categorical(labels, categories=list(OUTCOME_COLORS)), index=sites.index, name="outcome")


@app.function
def prepare_reward_sites(sites: pd.DataFrame) -> pd.DataFrame:
    """Reward (odor) sites only, with outcome and within-patch history columns.

    ``rewards_before`` / ``failures_before`` count, within the current patch, the
    rewards collected and the *consecutive* unrewarded stops since the last reward,
    both strictly before the site. These are the axes of the leave-probability map.
    """
    rs = sites.loc[sites["site_label"] == "RewardSite"].sort_values("start_time").reset_index(drop=True)
    stopped = rs["has_choice"].fillna(False).astype(bool)
    rewarded = rs["has_reward"].fillna(False).astype(bool)
    failed = (stopped & ~rewarded).astype(int)

    rewards_before = rewarded.astype(int).groupby(rs["patch_index"]).cumsum() - rewarded.astype(int)
    failures_before = failed.groupby([rs["patch_index"], rewards_before]).cumsum() - failed
    return rs.assign(
        outcome=classify_outcomes(rs),
        stopped=stopped,
        rewarded=rewarded,
        rewards_before=rewards_before,
        failures_before=failures_before,
    )


@app.function
def session_summary(reward_sites: pd.DataFrame, position: pd.DataFrame) -> dict[str, float]:
    """Headline numbers for the session."""
    n_stops = int(reward_sites["stopped"].sum())
    n_rewarded = int(reward_sites["rewarded"].sum())
    return {
        "reward_sites": len(reward_sites),
        "stops": n_stops,
        "stop_rate": n_stops / max(len(reward_sites), 1),
        "rewarded_stops": n_rewarded,
        "reward_rate": n_rewarded / max(n_stops, 1),
        "water_ul": float(reward_sites.loc[reward_sites["rewarded"], "reward_amount"].sum()),
        "distance_m": float(position["position"].max() - position["position"].min()) / 100,
        "duration_min": float(position["timestamp"].iloc[-1] - position["timestamp"].iloc[0]) / 60,
    }


@app.function
def per_patch_table(reward_sites: pd.DataFrame) -> pd.DataFrame:
    """Per patch type: how many patches, sites, stops, rewards and water."""
    return (
        reward_sites.assign(water=reward_sites["reward_amount"].where(reward_sites["rewarded"], 0.0))
        .groupby("patch_label", observed=True)
        .agg(
            patches=("patch_index", "nunique"),
            sites=("site_index", "size"),
            stop_rate=("stopped", "mean"),
            rewards=("rewarded", "sum"),
            water_ul=("water", "sum"),
        )
        .round({"stop_rate": 2})
        .reset_index()
    )


@app.function
def patch_colors(labels) -> dict[str, str]:
    """Stable color per patch label: sorted labels take palette slots in order. Never cycles."""
    unique = sorted(set(labels))
    if len(unique) > len(CATEGORICAL):
        raise ValueError(f"{len(unique)} patch types > {len(CATEGORICAL)} palette slots; facet instead")
    return dict(zip(unique, CATEGORICAL[: len(unique)], strict=True))


@app.function
def align_to_events(
    timestamps: np.ndarray, values: np.ndarray, events: np.ndarray, window: tuple[float, float], step: float = 0.02
) -> tuple[np.ndarray, np.ndarray]:
    """Resample ``values`` on a common grid around each event: (grid, matrix[event, time]).

    One vectorised ``np.interp`` call; samples outside the recording are NaN.
    """
    grid = np.arange(window[0], window[1] + step / 2, step)
    query = np.asarray(events, dtype=float)[:, None] + grid[None, :]
    matrix = np.interp(query.ravel(), timestamps, values, left=np.nan, right=np.nan).reshape(query.shape)
    return grid, matrix


@app.function
def leave_probability(reward_sites: pd.DataFrame, max_count: int = 8, min_visits: int = 5) -> pd.DataFrame:
    """P(leave) by rewards-before x consecutive-failures-before; bins with < ``min_visits`` are NaN."""
    clipped = reward_sites.assign(
        rewards_before=reward_sites["rewards_before"].clip(upper=max_count),
        failures_before=reward_sites["failures_before"].clip(upper=max_count),
        left=~reward_sites["stopped"],
    )
    grouped = clipped.groupby(["rewards_before", "failures_before"])["left"]
    p_leave = grouped.mean().where(grouped.size() >= min_visits)
    return p_leave.unstack("failures_before")


# ---------------------------------------------------------------------------
# Figures: each returns a Figure; cells decide where to show it.
# ---------------------------------------------------------------------------


@app.function
def plot_outcome_raster(reward_sites: pd.DataFrame):
    """Patch x site grid, one cell per reward site, colored by outcome.

    Top: whole session in patch order. Bottom: one panel per patch type (named in
    its title; no patch hue here, so outcome is the only color encoding).
    """
    labels = sorted(reward_sites["patch_label"].unique())
    widths = [reward_sites.loc[reward_sites["patch_label"] == lab, "patch_index"].nunique() for lab in labels]
    n_sites = int(reward_sites["site_index_in_patch_by_type"].max()) + 1
    cells = {"height": 1, "width": 1, "edgecolor": "white", "linewidth": 0.6}
    cell_colors = reward_sites["outcome"].map(OUTCOME_COLORS).astype(str)

    fig = plt.figure(figsize=(12, 6.5), layout="constrained")
    grid = fig.add_gridspec(2, len(labels), width_ratios=widths)

    top = fig.add_subplot(grid[0, :])
    top.bar(reward_sites["patch_index"], bottom=reward_sites["site_index_in_patch_by_type"], color=cell_colors, **cells)
    for boundary in reward_sites.loc[reward_sites["block_index"].diff() > 0, "patch_index"]:
        top.axvline(boundary - 0.5, color=INK, linestyle="--", linewidth=1)
    top.set(
        xlabel="Patch #", ylabel="Site in patch", ylim=(-0.2, n_sites), xlim=(-1, reward_sites["patch_index"].max() + 1)
    )
    top.set_title("All patches, in session order (dashed = block change)", loc="left")

    for col, label in enumerate(labels):
        ax = fig.add_subplot(grid[1, col])
        sub = reward_sites.loc[reward_sites["patch_label"] == label]
        ax.bar(
            sub["patch_index_by_type"], bottom=sub["site_index_in_patch_by_type"], color=cell_colors[sub.index], **cells
        )
        ax.set_title(label, loc="left", fontsize=9)
        ax.set(ylim=(-0.2, n_sites), xlabel="Patch # (this type)")
        if col:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("Site in patch")

    handles = [Patch(facecolor=c, edgecolor="white", label=k) for k, c in OUTCOME_COLORS.items()]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
    return fig


@app.function
def plot_aligned_velocity(
    reward_sites: pd.DataFrame,
    position: pd.DataFrame,
    colors: dict[str, str],
    threshold: float | None,
    window: tuple[float, float],
    first_site_only: bool,
):
    """Velocity around odor onset: one panel per patch type, median line + IQR band."""
    sites = reward_sites.dropna(subset=["odor_onset_time"])
    if first_site_only:
        sites = sites.loc[sites["site_index_in_patch_by_type"] == 0]
    ts, vel = position["timestamp"].to_numpy(), position["velocity"].to_numpy()
    groups = list(sites.groupby("patch_label"))

    fig, axes = plt.subplots(
        1, len(groups), figsize=(2.6 * len(groups), 3.2), sharex=True, sharey=True, layout="constrained", squeeze=False
    )
    for ax, (label, group) in zip(axes[0], groups, strict=True):
        grid, traces = align_to_events(ts, vel, group["odor_onset_time"].to_numpy(), window)
        lo, med, hi = np.nanpercentile(traces, [25, 50, 75], axis=0)
        ax.axvspan(window[0], 0, color=GRID, alpha=0.5, linewidth=0)
        ax.fill_between(grid, lo, hi, color=colors[label], alpha=0.25, linewidth=0)
        ax.plot(grid, med, color=colors[label], linewidth=2)
        if threshold is not None:
            ax.axhline(threshold, color=INK_MUTED, linestyle="--", linewidth=1)
        ax.axvline(0, color=INK, linewidth=0.8)
        ax.set_title(f"{label}  (n={len(group)})", loc="left", fontsize=8)
        ax.set(xlim=window, xlabel="Time from odor onset (s)")
    axes[0, 0].set_ylabel("Velocity (cm/s)")
    note = f"; dashed = stop threshold {threshold:g} cm/s" if threshold is not None else ""
    fig.suptitle(
        f"Velocity around odor onset (median, IQR band{note})", x=0.01, ha="left", fontsize=11, fontweight="bold"
    )
    return fig


@app.function
def plot_stop_rate(reward_sites: pd.DataFrame, colors: dict[str, str], smoothing: int):
    """Rolling P(stop) across the session, per patch type, on session time."""
    t0 = reward_sites["start_time"].min()
    fig, ax = plt.subplots(figsize=(10, 3.2), layout="constrained")
    for label, group in reward_sites.groupby("patch_label"):
        rolling = group["stopped"].astype(float).rolling(smoothing, min_periods=1, center=True).mean()
        ax.plot((group["start_time"] - t0) / 60, rolling, color=colors[label], linewidth=2, label=label)
    ax.set(ylim=(-0.03, 1.03), xlabel="Session time (min)", ylabel="P(stop)")
    ax.set_title(f"Stop probability over the session (rolling {smoothing} sites)", loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    return fig


@app.function
def plot_leave_heatmap(p_leave: pd.DataFrame, min_visits: int):
    """Heatmap of P(leave); sequential single hue, empty bins left blank."""
    fig, ax = plt.subplots(figsize=(5, 4.2), layout="constrained")
    image = ax.imshow(p_leave.to_numpy(dtype=float), origin="lower", cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(p_leave.shape[1]), p_leave.columns)
    ax.set_yticks(range(p_leave.shape[0]), p_leave.index)
    ax.set(xlabel="Consecutive failures before site", ylabel="Rewards before site (in patch)")
    ax.set_title(f"P(leave) (bins with < {min_visits} visits blank)", loc="left")
    fig.colorbar(image, ax=ax, label="P(leave)", shrink=0.8)
    return fig


@app.function
def plot_timeline(
    sites: pd.DataFrame,
    position: pd.DataFrame,
    licks: pd.DataFrame,
    colors: dict[str, str],
    start: float,
    width: float,
):
    """A window of the session: events raster above, velocity with site shading below."""
    t0 = sites["start_time"].min()
    lo, hi = t0 + start, t0 + start + width

    fig, (ev, vel) = plt.subplots(2, 1, figsize=(12, 4), sharex=True, height_ratios=[1, 3], layout="constrained")
    in_view = sites.loc[(sites["stop_time"] >= lo) & (sites["start_time"] <= hi)]
    for row in in_view.itertuples():  # only the handful of sites in view
        color = colors.get(row.patch_label, NEUTRAL) if row.site_label == "RewardSite" else "#f0efec"
        vel.axvspan(row.start_time - t0, row.stop_time - t0, color=color, alpha=0.25, linewidth=0)

    window_pos = position.loc[position["timestamp"].between(lo, hi)]
    vel.plot(window_pos["timestamp"] - t0, window_pos["velocity"], color=INK, linewidth=1)
    vel.set(ylabel="Velocity (cm/s)", xlabel="Session time (s)", xlim=(start, start + width))

    lick_t = licks.loc[licks["is_lick_onset"] & licks["timestamp"].between(lo, hi), "timestamp"] - t0
    cue_t = in_view["choice_cue_time"].dropna() - t0
    reward_t = in_view["reward_onset_time"].dropna() - t0
    ev.eventplot(
        [lick_t, cue_t, reward_t], colors=[INK, CATEGORICAL[1], CATEGORICAL[0]], lineoffsets=[2, 1, 0], linelengths=0.7
    )
    ev.set_yticks([2, 1, 0], ["Lick", "Choice cue", "Reward"])
    ev.spines["left"].set_visible(False)

    handles = [Patch(facecolor=c, alpha=0.4, label=k) for k, c in colors.items()]
    vel.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), title="Reward site")
    return fig


# ---------------------------------------------------------------------------
# Notebook
# ---------------------------------------------------------------------------


@app.cell(hide_code=True)
def header():
    mo.md("""
    # VR foraging: daily session summary

    Pick a subject and a day, or open this page with `?session=<session_id>`.
    Everything is read from the session's processed asset in the public
    `aind-open-data` bucket, in your browser.
    """)
    return


@app.cell
def requested_session():
    # Source of truth for "which session": URL (?session=), then CLI (--session), then a default.
    query = mo.query_params()
    requested = query.get("session") or mo.cli_args().get("session") or DEFAULT_SESSION
    parsed = parse_session_id(requested)
    mo.stop(
        parsed is None,
        mo.callout(mo.md(f"`{requested}` is not a session id (`<subject>_<YYYY-MM-DD>_<HH-MM-SS>`)."), kind="danger"),
    )
    return parsed, query, requested


@app.cell
def pickers(parsed):
    subject_input = mo.ui.text(value=parsed[0], label="Subject")
    day_input = mo.ui.date(value=parsed[1], label="Day")
    mo.hstack([subject_input, day_input], justify="start")
    return day_input, subject_input


@app.cell
async def find_sessions(day_input, subject_input):
    with mo.status.spinner(title=f"Listing sessions of {subject_input.value} on {day_input.value}"):
        available = await list_processed_sessions(subject_input.value, day_input.value)
    mo.stop(
        not available,
        mo.callout(f"No processed sessions for subject {subject_input.value} on {day_input.value}.", kind="warn"),
    )
    return (available,)


@app.cell
def session_choice(available, query, requested):
    session_picker = mo.ui.dropdown(
        options=sorted(available),
        value=requested if requested in available else max(available),
        label="Session",
        on_change=lambda sid: query.set("session", sid),  # keeps the URL shareable
    )
    session_picker
    return (session_picker,)


@app.cell
async def load_session(available, session_picker):
    session_id = session_picker.value
    asset = available[session_id]
    with mo.status.spinner(title=f"Loading {session_id}"):
        sites, position, licks, software_events = await asyncio.gather(
            load_table(asset, "sites"),
            load_table(asset, "position_velocity"),
            load_table(asset, "licks"),
            load_table(asset, "software_events", columns=["event_name", "data"]),
        )
    reward_sites = prepare_reward_sites(sites)
    colors = patch_colors(reward_sites["patch_label"])
    threshold = stop_velocity_threshold(software_events)
    mo.md(
        f"Asset: [`{asset.rsplit('/', 1)[-1]}`]({asset}/sites.parquet) · "
        f"packaging `{sites.attrs.get('packaging_version', '?')}` · "
        f"data contract `{sites.attrs.get('data_contract_version', '?')}`"
    )
    return colors, licks, position, reward_sites, session_id, sites, threshold


@app.cell(hide_code=True)
def summary_stats(position, reward_sites, session_id):
    stats = session_summary(reward_sites, position)
    mo.vstack(
        [
            mo.md(f"## {session_id}"),
            mo.hstack(
                [
                    mo.stat(f"{stats['reward_sites']}", label="Odor sites visited"),
                    mo.stat(f"{stats['stop_rate']:.0%}", label="Stopped", caption=f"{stats['stops']} stops"),
                    mo.stat(
                        f"{stats['reward_rate']:.0%}",
                        label="Stops rewarded",
                        caption=f"{stats['rewarded_stops']} rewards",
                    ),
                    mo.stat(f"{stats['water_ul']:.0f} µL", label="Water collected"),
                    mo.stat(f"{stats['distance_m']:.0f} m", label="Distance run"),
                    mo.stat(f"{stats['duration_min']:.0f} min", label="Duration"),
                ],
                widths="equal",
            ),
            mo.ui.table(per_patch_table(reward_sites), selection=None, label="Per patch type"),
        ]
    )
    return


@app.cell(hide_code=True)
def raster(reward_sites):
    mo.vstack([mo.md("## Choices, patch by patch"), plot_outcome_raster(reward_sites)])
    return


@app.cell
def velocity_controls():
    first_site_only = mo.ui.switch(value=True, label="First site of each patch only")
    velocity_window = mo.ui.range_slider(-3, 6, step=0.5, value=[-1, 3], label="Window (s)", show_value=True)
    mo.hstack([first_site_only, velocity_window], justify="start")
    return first_site_only, velocity_window


@app.cell(hide_code=True)
def velocity(colors, first_site_only, position, reward_sites, threshold, velocity_window):
    plot_aligned_velocity(
        reward_sites, position, colors, threshold, tuple(velocity_window.value), first_site_only.value
    )
    return


@app.cell
def stop_rate_controls():
    smoothing = mo.ui.slider(1, 50, value=15, label="Smoothing (sites)", show_value=True)
    smoothing
    return (smoothing,)


@app.cell(hide_code=True)
def stop_rate(colors, reward_sites, smoothing):
    plot_stop_rate(reward_sites, colors, smoothing.value)
    return


@app.cell(hide_code=True)
def leave_map(reward_sites):
    min_visits = 5
    plot_leave_heatmap(leave_probability(reward_sites, min_visits=min_visits), min_visits)
    return


@app.cell
def timeline_controls(sites):
    session_seconds = float(sites["stop_time"].max() - sites["start_time"].min())
    timeline_width = mo.ui.slider(10, 300, step=10, value=60, label="Width (s)", show_value=True)
    timeline_start = mo.ui.slider(
        0, session_seconds, step=5, value=0, label="Start (s)", show_value=True, full_width=True
    )
    mo.vstack([mo.md("## Session timeline"), timeline_width, timeline_start])
    return timeline_start, timeline_width


@app.cell(hide_code=True)
def timeline(colors, licks, position, sites, timeline_start, timeline_width):
    plot_timeline(sites, position, licks, colors, timeline_start.value, timeline_width.value)
    return


@app.cell(hide_code=True)
def site_table(reward_sites):
    mo.accordion({"Reward-site table": mo.ui.table(reward_sites, page_size=15)})
    return


if __name__ == "__main__":
    app.run()
