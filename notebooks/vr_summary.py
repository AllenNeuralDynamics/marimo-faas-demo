"""Data access, analysis and figures for the VR-foraging daily summary notebook.

Plain module, imported by ``daily_summary.py``. It runs unchanged under CPython and
under Pyodide (the static GitHub Pages build), so all network I/O goes through
:func:`fetch_bytes`.

``session_id`` is the raw asset's DocDB ``name``: an opaque key. It is only ever
compared for equality, never parsed or used to build a path.
"""

import asyncio
import io
import json
import sys
import urllib.parse
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.patches import Patch

DOCDB = "https://api.allenneuraldynamics.org/v2/metadata_index/data_assets"
ACQUISITION_TYPE = "AindVrForaging"
S3_REGION = "us-west-2"
RECORD_FIELDS = {
    "name": 1,
    "location": 1,
    "subject.subject_id": 1,
    "acquisition.acquisition_start_time": 1,
    "data_description.data_level": 1,
    "data_description.creation_time": 1,
    "data_description.source_data": 1,
}

# Colors: patch palette and event colors from the lab's VR-foraging dashboard/viewer, for parity.
PATCH_COLORMAP = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d"]
INK, MUTED, SHADE = "#222222", "#898781", "#ecebe6"
OUTCOME_COLORS = {  # validated as a set: adjacent CVD separation dE >= 15
    "Stop, rewarded": "#2a78d6",
    "Stop, depleted": "#e34948",
    "Stop, unrewarded (probabilistic)": "#eda100",
    "Stop, aborted wait": "#e87ba4",
    "Leave": "#d6d5cf",
}

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


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------


async def fetch_bytes(url: str) -> bytes:
    """GET ``url`` with the browser's fetch under Pyodide, urllib otherwise."""
    if sys.platform == "emscripten":
        from pyodide.http import pyfetch

        response = await pyfetch(url)
        if not response.ok:
            raise OSError(f"GET {url} -> HTTP {response.status}")
        return await response.bytes()

    def get() -> bytes:
        with urllib.request.urlopen(url, timeout=60) as response:
            return response.read()

    return await asyncio.to_thread(get)


async def query_docdb(filter_query: dict, sort: dict | None = None, limit: int = 100) -> list[dict]:
    """Records from the public AIND metadata DB (the REST call behind aind-data-access-api)."""
    params = {"filter": json.dumps(filter_query), "projection": json.dumps(RECORD_FIELDS), "limit": limit, "skip": 0}
    if sort:
        params["sort"] = json.dumps(sort)
    return json.loads(await fetch_bytes(f"{DOCDB}?{urllib.parse.urlencode(params)}"))


def s3_to_https(uri: str) -> str:
    """``s3://bucket/key`` -> anonymous HTTPS URL of the same object."""
    bucket, _, key = uri.removeprefix("s3://").partition("/")
    return f"https://{bucket}.s3.{S3_REGION}.amazonaws.com/{key}"


def link_processed(raw_records: list[dict], derived_records: list[dict]) -> pd.DataFrame:
    """One row per raw session with its newest derived asset's location (NaN if not processed yet).

    Linked by exact key: a derived record lists the raw ``name`` in ``data_description.source_data``.
    """
    newest: dict[str, tuple[str, str]] = {}
    for record in derived_records:
        created = record["data_description"]["creation_time"]
        for source in record["data_description"].get("source_data") or []:
            if source not in newest or created > newest[source][0]:
                newest[source] = (created, record["location"])
    rows = [
        {
            "session_id": r["name"],
            "subject_id": r["subject"]["subject_id"],
            "start_time": pd.Timestamp(r["acquisition"]["acquisition_start_time"]),
            "processed_location": newest.get(r["name"], (None, None))[1],
        }
        for r in raw_records
    ]
    columns = ["session_id", "subject_id", "start_time", "processed_location"]
    return pd.DataFrame(rows, columns=columns).sort_values("start_time", ascending=False, ignore_index=True)


async def find_raw_session(name: str | None = None) -> dict | None:
    """The raw record called ``name``, or the newest VR-foraging session if no name is given."""
    query = {"data_description.data_level": "raw", "acquisition.acquisition_type": ACQUISITION_TYPE}
    if name:
        query["name"] = name
    records = await query_docdb(query, sort={"acquisition.acquisition_start_time": -1}, limit=1)
    return records[0] if records else None


async def sessions_on_day(subject_id: str, day) -> pd.DataFrame:
    """One subject's sessions starting on ``day`` (UTC), with their processed asset.

    One query returns raw and derived records together (derived records carry the raw
    acquisition time); filtering on ``source_data`` alone is ~50x slower, it is not indexed.
    """
    start = pd.Timestamp(day)
    records = await query_docdb(
        {
            "subject.subject_id": subject_id.strip(),
            "acquisition.acquisition_type": ACQUISITION_TYPE,
            "acquisition.acquisition_start_time": {
                "$gte": start.isoformat(),
                "$lt": (start + pd.Timedelta(days=1)).isoformat(),
            },
        }
    )
    by_level = {"raw": [], "derived": []}
    for record in records:
        by_level.setdefault(record["data_description"]["data_level"], []).append(record)
    return link_processed(by_level["raw"], by_level["derived"])


async def load_table(asset_location: str, table: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Read ``<asset>/<table>.parquet`` (all stored columns kept as columns)."""
    payload = await fetch_bytes(f"{s3_to_https(asset_location)}/{table}.parquet")
    return pq.read_table(io.BytesIO(payload), columns=columns).to_pandas(ignore_metadata=True)


async def load_session(asset_location: str) -> tuple[pd.DataFrame, pd.DataFrame, float | None]:
    """The three tables the summary needs, fetched concurrently: sites, velocity, stop threshold."""
    sites, position, events = await asyncio.gather(
        load_table(asset_location, "sites"),
        load_table(asset_location, "position_velocity"),
        load_table(asset_location, "software_events", columns=["event_name", "data"]),
    )
    threshold = events.loc[events["event_name"] == "StopVelocityThreshold", "data"]
    return sites, position, float(threshold.iloc[0]) if len(threshold) else None


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def classify_outcomes(sites: pd.DataFrame) -> pd.Series:
    """One outcome label per site (keys of OUTCOME_COLORS). First matching rule wins."""
    stopped = sites["has_choice"].fillna(False).astype(bool)
    rewarded = sites["has_reward"].fillna(False).astype(bool)
    rules = {
        "Stop, rewarded": stopped & rewarded,
        "Stop, aborted wait": stopped & sites["has_waited_reward_delay"].eq(False),
        "Stop, depleted": stopped & sites["reward_available"].le(0),
        "Stop, unrewarded (probabilistic)": stopped,
    }
    labels = np.select(list(rules.values()), list(rules.keys()), default="Leave")
    return pd.Series(pd.Categorical(labels, categories=list(OUTCOME_COLORS)), index=sites.index, name="outcome")


def prepare_reward_sites(sites: pd.DataFrame) -> pd.DataFrame:
    """Odor (reward) sites in time order, with outcome and boolean stopped/rewarded columns."""
    rs = sites.loc[sites["site_label"] == "RewardSite"].sort_values("start_time").reset_index(drop=True)
    return rs.assign(
        outcome=classify_outcomes(rs),
        stopped=rs["has_choice"].fillna(False).astype(bool),
        rewarded=rs["has_reward"].fillna(False).astype(bool),
    )


def session_summary(reward_sites: pd.DataFrame, position: pd.DataFrame) -> dict[str, float]:
    """Headline numbers for the session."""
    stops, rewards = int(reward_sites["stopped"].sum()), int(reward_sites["rewarded"].sum())
    return {
        "odor_sites": len(reward_sites),
        "stops": stops,
        "stop_rate": stops / max(len(reward_sites), 1),
        "rewards": rewards,
        "reward_rate": rewards / max(stops, 1),
        "water_ul": float(reward_sites.loc[reward_sites["rewarded"], "reward_amount"].sum()),
        "distance_m": float(position["position"].max() - position["position"].min()) / 100,
        "duration_min": float(position["timestamp"].iloc[-1] - position["timestamp"].iloc[0]) / 60,
    }


def per_patch_table(reward_sites: pd.DataFrame) -> pd.DataFrame:
    """Per patch type: patches, sites, stop rate, rewards and water (µL)."""
    return (
        reward_sites.assign(water_ul=reward_sites["reward_amount"].where(reward_sites["rewarded"], 0.0))
        .groupby("patch_label")
        .agg(
            patches=("patch_index", "nunique"),
            sites=("site_index", "size"),
            stop_rate=("stopped", "mean"),
            rewards=("rewarded", "sum"),
            water_ul=("water_ul", "sum"),
        )
        .round({"stop_rate": 2})
        .reset_index()
    )


def patch_colors(labels) -> dict[str, str]:
    """Stable color per patch label: sorted labels take palette slots in order. Never cycles."""
    unique = sorted(set(labels))
    if len(unique) > len(PATCH_COLORMAP):
        raise ValueError(f"{len(unique)} patch types > {len(PATCH_COLORMAP)} palette slots")
    return dict(zip(unique, PATCH_COLORMAP[: len(unique)], strict=True))


def align_to_events(timestamps, values, events, window: tuple[float, float], step: float = 0.02):
    """Resample ``values`` around each event on a common grid: (grid, matrix[event, time]); NaN off the record."""
    grid = np.arange(window[0], window[1] + step / 2, step)
    query = np.asarray(events, dtype=float)[:, None] + grid[None, :]
    matrix = np.interp(query.ravel(), timestamps, values, left=np.nan, right=np.nan).reshape(query.shape)
    return grid, matrix


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


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
