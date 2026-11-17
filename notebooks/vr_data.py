"""Session lookup, data loading and analysis for the VR-foraging daily summary (no plotting).

Runs unchanged under CPython and under Pyodide (the static GitHub Pages build), so all
network I/O goes through :func:`fetch_bytes`.

``session_id`` is the raw asset's DocDB ``name``: an opaque key. It is only ever
compared for equality, never parsed or used to build a path.
"""

import asyncio
import io
import json
import sys
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

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
OUTCOMES = ["Stop, rewarded", "Stop, depleted", "Stop, unrewarded (probabilistic)", "Stop, aborted wait", "Leave"]


async def fetch_bytes(url: str, cache: bool = False) -> bytes:
    """GET ``url`` with the browser's fetch under Pyodide, urllib otherwise.

    ``cache=True`` is for immutable files (a processed asset never changes in place): the
    browser then reuses its HTTP-cached copy, even across page reloads ("force-cache").
    The request is issued as soon as the coroutine starts, so it downloads in the
    background even while Python is busy.
    """
    if sys.platform == "emscripten":
        from pyodide.http import pyfetch

        response = await pyfetch(url, cache="force-cache" if cache else "default")
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
    payload = await fetch_bytes(f"{s3_to_https(asset_location)}/{table}.parquet", cache=True)
    return pq.read_table(io.BytesIO(payload), columns=columns).to_pandas(ignore_metadata=True)


_loaded_sessions: dict[str, tuple] = {}


async def load_session(asset_location: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float | None]:
    """Sites, velocity, licks and the stop threshold, fetched concurrently; kept in memory per asset."""
    if asset_location not in _loaded_sessions:
        sites, position, licks, events = await asyncio.gather(
            load_table(asset_location, "sites"),
            load_table(asset_location, "position_velocity"),
            load_table(asset_location, "licks"),
            load_table(asset_location, "software_events", columns=["event_name", "data"]),
        )
        threshold = events.loc[events["event_name"] == "StopVelocityThreshold", "data"]
        _loaded_sessions[asset_location] = (
            sites,
            position,
            licks,
            float(threshold.iloc[0]) if len(threshold) else None,
        )
    return _loaded_sessions[asset_location]


def classify_outcomes(sites: pd.DataFrame) -> pd.Series:
    """One outcome label per site (one of OUTCOMES). First matching rule wins."""
    stopped = sites["has_choice"].fillna(False).astype(bool)
    rewarded = sites["has_reward"].fillna(False).astype(bool)
    rules = {
        "Stop, rewarded": stopped & rewarded,
        "Stop, aborted wait": stopped & sites["has_waited_reward_delay"].eq(False),
        "Stop, depleted": stopped & sites["reward_available"].le(0),
        "Stop, unrewarded (probabilistic)": stopped,
    }
    labels = np.select(list(rules.values()), list(rules.keys()), default="Leave")
    return pd.Series(pd.Categorical(labels, categories=OUTCOMES), index=sites.index, name="outcome")


def prepare_reward_sites(sites: pd.DataFrame) -> pd.DataFrame:
    """Odor (reward) sites in time order, with outcome, stopped/rewarded and within-patch history.

    ``rewards_before`` / ``failures_before``: rewards collected, and *consecutive* unrewarded stops
    since the last reward, in the current patch and strictly before the site.
    """
    rs = sites.loc[sites["site_label"] == "RewardSite"].sort_values("start_time").reset_index(drop=True)
    stopped = rs["has_choice"].fillna(False).astype(bool)
    rewarded = rs["has_reward"].fillna(False).astype(bool)
    failed = (stopped & ~rewarded).astype(int)
    rewards_before = rewarded.astype(int).groupby(rs["patch_index"]).cumsum() - rewarded.astype(int)
    return rs.assign(
        outcome=classify_outcomes(rs),
        stopped=stopped,
        rewarded=rewarded,
        rewards_before=rewards_before,
        failures_before=failed.groupby([rs["patch_index"], rewards_before]).cumsum() - failed,
    )


def leave_probability(reward_sites: pd.DataFrame, max_count: int = 8, min_visits: int = 5) -> pd.DataFrame:
    """P(leave) by rewards-before x consecutive-failures-before; bins with < ``min_visits`` are NaN."""
    clipped = reward_sites.assign(
        rewards_before=reward_sites["rewards_before"].clip(upper=max_count),
        failures_before=reward_sites["failures_before"].clip(upper=max_count),
        left=~reward_sites["stopped"],
    )
    grouped = clipped.groupby(["rewards_before", "failures_before"])["left"]
    return grouped.mean().where(grouped.size() >= min_visits).unstack("failures_before")


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


def align_to_events(timestamps, values, events, window: tuple[float, float], step: float = 0.02):
    """Resample ``values`` around each event on a common grid: (grid, matrix[event, time]); NaN off the record."""
    grid = np.arange(window[0], window[1] + step / 2, step)
    query = np.asarray(events, dtype=float)[:, None] + grid[None, :]
    matrix = np.interp(query.ravel(), timestamps, values, left=np.nan, right=np.nan).reshape(query.shape)
    return grid, matrix
