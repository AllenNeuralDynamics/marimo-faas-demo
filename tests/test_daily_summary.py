"""Unit tests for the pure functions of notebooks/daily_summary.py (importable because they are @app.function)."""

import numpy as np
import pandas as pd
import pytest
from daily_summary import (
    align_to_events,
    classify_outcomes,
    leave_probability,
    link_processed,
    patch_colors,
    per_patch_table,
    prepare_reward_sites,
    s3_to_https,
)


def make_sites(rows: list[tuple]) -> pd.DataFrame:
    """rows: (patch_index, label, has_choice, has_reward, waited, reward_available)."""
    df = pd.DataFrame(
        rows,
        columns=[
            "patch_index",
            "patch_label",
            "has_choice",
            "has_reward",
            "has_waited_reward_delay",
            "reward_available",
        ],
    )
    n = len(df)
    return df.assign(
        site_label="RewardSite",
        start_time=np.arange(n, dtype=float),
        site_index=np.arange(n),
        reward_amount=np.where(df["has_reward"], 5.0, np.nan),
    )


def raw(name: str, start: str, subject: str = "s1") -> dict:
    return {
        "name": name,
        "location": f"s3://aind-open-data/{name}",
        "subject": {"subject_id": subject},
        "acquisition": {"acquisition_start_time": start},
    }


def derived(location: str, created: str, sources: list[str] | None) -> dict:
    return {"location": location, "data_description": {"creation_time": created, "source_data": sources}}


def test_link_processed_matches_by_source_data_and_keeps_newest():
    # Ids are deliberately not human readable: linking must never depend on their content.
    raws = [raw("opaque-A", "2026-11-13T09:00:00Z"), raw("opaque-B", "2026-11-13T21:00:00Z")]
    deriveds = [
        derived("s3://bucket/old", "2026-11-14T07:00:00Z", ["opaque-A"]),
        derived("s3://bucket/new", "2026-11-20T01:00:00Z", ["opaque-A"]),
        derived("s3://bucket/unrelated", "2026-11-21T00:00:00Z", None),
    ]
    table = link_processed(raws, deriveds).set_index("session_id")
    assert table.index.tolist() == ["opaque-B", "opaque-A"]  # newest acquisition first
    assert table.loc["opaque-A", "processed_location"] == "s3://bucket/new"
    assert pd.isna(table.loc["opaque-B", "processed_location"])


def test_link_processed_empty():
    assert link_processed([], []).empty


def test_s3_to_https():
    assert s3_to_https("s3://aind-open-data/some asset/x") == (
        "https://aind-open-data.s3.us-west-2.amazonaws.com/some asset/x"
    )


def test_classify_outcomes_covers_every_branch():
    sites = make_sites(
        [
            (0, "a", True, True, True, 10.0),  # rewarded
            (0, "a", True, False, False, 10.0),  # left during the reward delay
            (0, "a", True, False, True, 0.0),  # waited, nothing left
            (0, "a", True, False, True, 10.0),  # waited, unlucky draw
            (0, "a", False, None, None, None),  # skipped
        ]
    )
    assert classify_outcomes(sites).tolist() == [
        "Stop, rewarded",
        "Stop, aborted wait",
        "Stop, depleted",
        "Stop, unrewarded (probabilistic)",
        "Leave",
    ]


def test_history_counts_reset_per_patch_and_after_reward():
    # patch 0: R F F R F L | patch 1: F L
    sites = make_sites(
        [
            (0, "a", True, True, True, 5.0),
            (0, "a", True, False, True, 0.0),
            (0, "a", True, False, True, 0.0),
            (0, "a", True, True, True, 5.0),
            (0, "a", True, False, True, 0.0),
            (0, "a", False, None, None, None),
            (1, "b", True, False, True, 0.0),
            (1, "b", False, None, None, None),
        ]
    )
    rs = prepare_reward_sites(sites)
    assert rs["rewards_before"].tolist() == [0, 1, 1, 1, 2, 2, 0, 0]
    assert rs["failures_before"].tolist() == [0, 0, 1, 2, 0, 1, 0, 1]


def test_leave_probability_masks_sparse_bins():
    sites = make_sites([(i, "a", i % 2 == 0, False, True, 0.0) for i in range(10)])
    table = leave_probability(prepare_reward_sites(sites), min_visits=5)
    assert table.loc[0, 0] == pytest.approx(0.5)  # 10 first-site visits, half skipped
    table_strict = leave_probability(prepare_reward_sites(sites), min_visits=11)
    assert table_strict.isna().all().all()


def test_per_patch_table_counts_only_delivered_water():
    sites = make_sites(
        [(0, "a", True, True, True, 5.0), (0, "a", True, False, True, 0.0), (1, "b", False, None, None, None)]
    )
    table = per_patch_table(prepare_reward_sites(sites)).set_index("patch_label")
    assert table.loc["a", "water_ul"] == 5.0
    assert table.loc["b", "water_ul"] == 0.0
    assert table.loc["a", "stop_rate"] == 1.0


def test_align_to_events_interpolates_and_pads_with_nan():
    ts = np.arange(0.0, 10.0, 0.5)
    grid, traces = align_to_events(ts, ts * 2, np.array([1.0, 9.0]), window=(-2.0, 2.0), step=1.0)
    np.testing.assert_allclose(grid, [-2, -1, 0, 1, 2])
    assert np.isnan(traces[0, 0])  # t = -1 is before the recording
    np.testing.assert_allclose(traces[0, 1:], [0, 2, 4, 6])
    assert np.isnan(traces[1, -1])  # t = 11 is after it


def test_patch_colors_are_stable_and_never_cycle():
    assert patch_colors(["b", "a", "b"]) == patch_colors(["a", "b"])
    with pytest.raises(ValueError, match="palette slots"):
        patch_colors([str(i) for i in range(9)])
