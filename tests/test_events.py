"""Tests for gunfire-event clustering.

The regression test that matters most is ``test_time_actually_separates_events``.
The first implementation divided ``astype("int64")`` by 1e9 assuming nanosecond
timestamps, but these columns are ``datetime64[ms]``. That compressed nineteen
years into ten minutes, time stopped separating anything, and DBSCAN chained
101,303 of 108,800 records into a single "event" spanning 19 km. Nothing about
the output looked wrong except the numbers.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from src import config
from src.join import events as ev


def _shots(rows) -> gpd.GeoDataFrame:
    """Build a Shots_Fired-shaped frame from (x, y, iso_ts, detection) tuples."""
    df = pd.DataFrame(rows, columns=["x", "y", "ts", "detection"])
    # ISO8601 so fixtures may mix "12:00" and "12:09:50" precision.
    df["ts"] = (
        pd.to_datetime(df["ts"], format="ISO8601")
        .dt.tz_localize("America/Chicago")
        .astype("datetime64[ms, America/Chicago]")
    )
    df["is_sensor"] = df["detection"].eq("sensor")
    df["is_human_report"] = df["detection"].isin({"human_call", "human_report"})
    return gpd.GeoDataFrame(
        df,
        geometry=[Point(x, y) for x, y in zip(df["x"], df["y"])],
        crs=config.CRS_PROJECTED,
    )


class TestProjectionGuard:
    def test_geographic_crs_is_rejected(self):
        s = _shots([(480000, 4980000, "2024-01-01 00:00", "sensor")])
        with pytest.raises(ValueError, match="metres"):
            ev.build_events(s.to_crs(config.CRS_GEOGRAPHIC))


class TestTimeHandling:
    def test_time_actually_separates_events(self):
        """Same location, one year apart: two events, never one.

        Guards the millisecond/nanosecond unit bug.
        """
        s = _shots([
            (480000, 4980000, "2024-01-01 12:00", "sensor"),
            (480000, 4980000, "2025-01-01 12:00", "sensor"),
        ])
        _, rep = ev.build_events(s, radius_m=150, window_min=10)
        assert rep.events == 2

    def test_millisecond_and_nanosecond_inputs_agree(self):
        """Clustering must not depend on the timestamp's storage unit."""
        rows = [
            (480000, 4980000, "2024-01-01 12:00", "sensor"),
            (480050, 4980000, "2024-01-01 12:05", "human_call"),
            (480000, 4980000, "2024-06-01 12:00", "sensor"),
        ]
        ms = _shots(rows)
        ns = ms.copy()
        ns["ts"] = ns["ts"].astype("datetime64[ns, America/Chicago]")

        _, rep_ms = ev.build_events(ms)
        _, rep_ns = ev.build_events(ns)
        assert rep_ms.events == rep_ns.events == 2

    def test_records_outside_window_do_not_merge(self):
        s = _shots([
            (480000, 4980000, "2024-01-01 12:00", "sensor"),
            (480000, 4980000, "2024-01-01 12:30", "human_call"),
        ])
        _, rep = ev.build_events(s, radius_m=150, window_min=10)
        assert rep.events == 2


class TestClustering:
    def test_colocated_records_in_window_merge(self):
        s = _shots([
            (480000, 4980000, "2024-01-01 12:00", "sensor"),
            (480050, 4980000, "2024-01-01 12:02", "human_call"),
        ])
        _, rep = ev.build_events(s, radius_m=150, window_min=10)
        assert rep.events == 1
        assert rep.largest_event == 2

    def test_distant_records_do_not_merge(self):
        s = _shots([
            (480000, 4980000, "2024-01-01 12:00", "sensor"),
            (485000, 4980000, "2024-01-01 12:02", "human_call"),
        ])
        _, rep = ev.build_events(s, radius_m=150, window_min=10)
        assert rep.events == 2

    def test_metric_is_a_ball_not_a_box(self):
        """150 m AND 10 min apart is sqrt(150^2+150^2)=212 m away, so no merge.

        The joint constraint is deliberately stricter than each axis alone.
        """
        s = _shots([
            (480000, 4980000, "2024-01-01 12:00", "sensor"),
            (480149, 4980000, "2024-01-01 12:09:50", "human_call"),
        ])
        _, rep = ev.build_events(s, radius_m=150, window_min=10)
        assert rep.events == 2

    def test_every_record_is_assigned(self):
        s = _shots([
            (480000, 4980000, "2024-01-01 12:00", "sensor"),
            (480050, 4980000, "2024-01-01 12:01", "human_call"),
            (485000, 4985000, "2024-03-01 03:00", "human_call"),
        ])
        clustered, rep = ev.build_events(s)
        assert clustered["event_id"].notna().all()
        assert rep.records == 3

    def test_records_without_geometry_become_singletons(self):
        s = _shots([
            (480000, 4980000, "2024-01-01 12:00", "sensor"),
            (480050, 4980000, "2024-01-01 12:01", "human_call"),
        ])
        s.loc[1, "geometry"] = None
        clustered, rep = ev.build_events(s)
        assert rep.records == 2
        assert rep.events == 2
        assert clustered["event_id"].nunique() == 2


class TestSummarise:
    def test_detection_modes(self):
        s = _shots([
            # event A: sensor + human -> both
            (480000, 4980000, "2024-01-01 12:00", "sensor"),
            (480050, 4980000, "2024-01-01 12:01", "human_call"),
            # event B: sensor alone
            (485000, 4985000, "2024-02-01 12:00", "sensor"),
            # event C: human alone
            (470000, 4975000, "2024-03-01 12:00", "human_call"),
        ])
        clustered, _ = ev.build_events(s)
        summary = ev.summarise_events(clustered)
        assert sorted(summary["detection_mode"]) == [
            "both", "human_only", "sensor_only"
        ]

    def test_centroid_is_the_coordinate_mean(self):
        s = _shots([
            (480000, 4980000, "2024-01-01 12:00", "sensor"),
            (480100, 4980000, "2024-01-01 12:01", "human_call"),
        ])
        clustered, _ = ev.build_events(s)
        summary = ev.summarise_events(clustered)
        assert summary.geometry.iloc[0].x == pytest.approx(480050)

    def test_duration_reflects_real_span(self):
        s = _shots([
            (480000, 4980000, "2024-01-01 12:00", "sensor"),
            (480050, 4980000, "2024-01-01 12:06", "human_call"),
        ])
        clustered, _ = ev.build_events(s)
        summary = ev.summarise_events(clustered)
        assert summary["duration_min"].iloc[0] == pytest.approx(6.0)


class TestChainingReport:
    def test_singletons_are_excluded(self):
        s = _shots([
            (480000, 4980000, "2024-01-01 12:00", "sensor"),
            (485000, 4985000, "2024-02-01 12:00", "sensor"),
        ])
        clustered, _ = ev.build_events(s)
        assert ev.chaining_report(clustered, radius_m=150).empty

    def test_chained_cluster_is_flagged_as_overstretched(self):
        """A string of records each inside eps merges into one sprawling event.

        Single-linkage chaining is the failure mode this report exists to
        surface, so a chain well beyond the radius must be flagged.
        """
        rows = [
            (480000 + 140 * i, 4980000, "2024-01-01 12:00", "sensor")
            for i in range(12)
        ]
        clustered, rep = ev.build_events(_shots(rows), radius_m=150, window_min=10)
        assert rep.events == 1, "fixture should chain into one cluster"

        chain = ev.chaining_report(clustered, radius_m=150)
        assert len(chain) == 1
        assert chain.loc[0, "span_m"] == pytest.approx(140 * 11, abs=1)
        assert bool(chain.loc[0, "overstretched"]) is True
