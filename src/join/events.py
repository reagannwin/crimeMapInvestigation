"""Collapse gunfire records into distinct real-world events.

The problem
-----------
``Shots_Fired`` assigns a unique ``Master_Incident_Number`` to every row --
108,800 rows, 108,800 ids -- so a single burst of gunfire that trips a sensor
*and* draws three 911 calls appears as four separate records. Counting rows
therefore counts *reporting*, not gunfire. Neighbourhoods that call more would
look like they have more gunfire, which inverts the signal the project exists
to measure.

The approach
------------
DBSCAN over space and time jointly. Time is rescaled into spatial units so
that ``window_min`` minutes costs the same as ``radius_m`` metres:

    t_scaled = t_minutes * (radius_m / window_min)

With ``eps = radius_m`` in that space, two records cluster when they are close
in both dimensions. Note this is a Euclidean ball, not a box: records 150 m
*and* 10 min apart sit at sqrt(150^2 + 150^2) = 212 m and do not merge. The
constraint is deliberately stricter than "within 150 m and within 10 min".

``min_samples=1`` is correct here and is not the usual DBSCAN default. It makes
every record a core point, so the algorithm performs pure transitive grouping
and an isolated record becomes its own single-record event rather than being
labelled noise. Dedup has no notion of an outlier to discard.

The risk this creates
---------------------
Transitive grouping is single-linkage, so it chains: a string of records each
140 m apart can merge into one sprawling "event" spanning far more than the
radius. That failure is silent and inflates nothing visible in the row count,
so :func:`chaining_report` measures the actual spatial and temporal extent of
every cluster and flags those that exceed what the parameters should permit.
Check it before trusting an event count.
"""
from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from src import config

SECONDS_PER_MINUTE = 60.0


@dataclass(frozen=True)
class EventReport:
    """What the clustering did to the record count."""

    records: int
    events: int
    radius_m: int
    window_min: int
    multi_record_events: int
    largest_event: int

    @property
    def compression(self) -> float:
        return self.records / self.events if self.events else float("nan")

    def __str__(self) -> str:
        return (
            f"{self.records:,} records -> {self.events:,} events "
            f"({self.compression:.2f} records/event) | "
            f"{self.multi_record_events:,} events merged >1 record, "
            f"largest merged {self.largest_event}"
        )


def build_events(
    shots: gpd.GeoDataFrame,
    *,
    radius_m: int = config.EVENT_CLUSTER_RADIUS_M,
    window_min: int = config.EVENT_CLUSTER_WINDOW_MIN,
) -> tuple[gpd.GeoDataFrame, EventReport]:
    """Assign an ``event_id`` to every gunfire record.

    Records missing geometry or timestamp cannot be clustered and are given
    their own singleton event, so the record count is always preserved.
    """
    if shots.crs is None or shots.crs.to_string() != config.CRS_PROJECTED:
        raise ValueError(
            f"shots must be in {config.CRS_PROJECTED} (metres), got {shots.crs}. "
            "The time-to-space rescaling assumes metres."
        )

    out = shots.copy().reset_index(drop=True)
    usable = out.geometry.notna() & out["ts"].notna()

    out["event_id"] = pd.Series(pd.NA, index=out.index, dtype="Int64")

    if usable.any():
        sub = out.loc[usable]
        # Unit-agnostic, and relative to the first record so the magnitudes stay
        # small. Do NOT use astype("int64") / 1e9: these columns are
        # datetime64[ms], not [ns], and that arithmetic silently compresses
        # nineteen years into ten minutes -- after which time separates nothing
        # and DBSCAN chains the whole city into one event.
        minutes = (sub["ts"] - sub["ts"].min()).dt.total_seconds() / SECONDS_PER_MINUTE
        scale = radius_m / window_min

        coords = np.column_stack([
            sub.geometry.x.to_numpy(),
            sub.geometry.y.to_numpy(),
            minutes.to_numpy() * scale,
        ])
        labels = DBSCAN(
            eps=radius_m, min_samples=1, algorithm="ball_tree"
        ).fit_predict(coords)
        out.loc[usable, "event_id"] = labels.astype("int64")

    # Unclusterable records become singleton events after the real ones.
    n_clustered = int(out["event_id"].max()) + 1 if usable.any() else 0
    orphans = out["event_id"].isna()
    if orphans.any():
        out.loc[orphans, "event_id"] = np.arange(
            n_clustered, n_clustered + int(orphans.sum())
        )
    out["event_id"] = out["event_id"].astype("int64")

    sizes = out.groupby("event_id").size()
    report = EventReport(
        records=len(out),
        events=int(out["event_id"].nunique()),
        radius_m=radius_m,
        window_min=window_min,
        multi_record_events=int((sizes > 1).sum()),
        largest_event=int(sizes.max()),
    )
    return out, report


def summarise_events(clustered: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """One row per event, describing how it was detected.

    ``detection_mode`` is the analytically important column:

        both        sensor fired AND a human called
        sensor_only sensor fired, nobody called
        human_only  somebody called, no sensor fired
    """
    work = clustered.assign(
        _x=clustered.geometry.x,
        _y=clustered.geometry.y,
        _confirmed=clustered["detection"].eq("confirmed_shooting"),
    )
    g = work.groupby("event_id")

    summary = pd.DataFrame({
        "n_records": g.size(),
        "ts_start": g["ts"].min(),
        "ts_end": g["ts"].max(),
        "n_sensor": g["is_sensor"].sum(),
        "n_human": g["is_human_report"].sum(),
        "n_confirmed": g["_confirmed"].sum(),
    })
    summary["duration_min"] = (
        (summary["ts_end"] - summary["ts_start"]).dt.total_seconds()
        / SECONDS_PER_MINUTE
    )
    summary["has_sensor"] = summary["n_sensor"] > 0
    summary["has_human"] = summary["n_human"] > 0
    summary["detection_mode"] = np.select(
        [summary["has_sensor"] & summary["has_human"],
         summary["has_sensor"] & ~summary["has_human"]],
        ["both", "sensor_only"],
        default="human_only",
    )

    # For point inputs the centroid is the coordinate mean, so this avoids a
    # per-group union_all() that dominates runtime at ~100k groups.
    centre = g[["_x", "_y"]].mean()
    summary = gpd.GeoDataFrame(
        summary,
        geometry=gpd.points_from_xy(centre["_x"], centre["_y"], crs=clustered.crs),
    )
    return summary.reset_index()


def chaining_report(clustered: gpd.GeoDataFrame, *, radius_m: int) -> pd.DataFrame:
    """Measure how far single-linkage chaining actually stretched each cluster.

    ``span_m`` is the diameter of the cluster's bounding box. A cluster built
    from an eps of ``radius_m`` may legitimately exceed it through chaining;
    what matters is whether it exceeds it by enough to have merged genuinely
    distinct events. ``overstretched`` flags spans beyond 3x the radius.
    """
    multi = clustered[clustered.groupby("event_id")["event_id"].transform("size") > 1]
    if multi.empty:
        return pd.DataFrame(
            columns=["event_id", "n_records", "span_m", "duration_min", "overstretched"]
        )

    work = multi.assign(_x=multi.geometry.x, _y=multi.geometry.y)
    g = work.groupby("event_id")
    extent = g.agg(
        n_records=("_x", "size"),
        xmin=("_x", "min"), xmax=("_x", "max"),
        ymin=("_y", "min"), ymax=("_y", "max"),
        ts_min=("ts", "min"), ts_max=("ts", "max"),
    )

    rep = pd.DataFrame({
        "n_records": extent["n_records"],
        "span_m": np.hypot(
            extent["xmax"] - extent["xmin"], extent["ymax"] - extent["ymin"]
        ).round(1),
        "duration_min": (
            (extent["ts_max"] - extent["ts_min"]).dt.total_seconds()
            / SECONDS_PER_MINUTE
        ).round(1),
    }).reset_index()
    rep["overstretched"] = rep["span_m"] > 3 * radius_m
    return rep.sort_values("span_m", ascending=False, ignore_index=True)


def null_match_rate(
    shots: gpd.GeoDataFrame,
    *,
    radius_m: int = config.EVENT_CLUSTER_RADIUS_M,
    window_min: int = config.EVENT_CLUSTER_WINDOW_MIN,
    n_iter: int = 20,
    max_shift_days: int = 180,
    seed: int = 0,
) -> pd.DataFrame:
    """Estimate how often sensor and human records co-cluster purely by chance.

    Matching two dense point processes in a small city will pair some records
    that have nothing to do with each other, and the observed rate alone cannot
    tell you how many. This shifts every human report by a large random offset
    -- breaking any true pairing while preserving each record's location, the
    overall temporal density, and time-of-day structure -- then re-runs the
    clustering. The gap between observed and null is the real co-detection
    signal.

    Sensor records keep their true timestamps, so the null differs from the
    observation in exactly one respect: whether human reports are aligned with
    the gunfire the sensors heard.
    """
    rng = np.random.default_rng(seed)
    human = shots["is_human_report"].to_numpy()

    observed = _detection_counts(shots, radius_m=radius_m, window_min=window_min)
    rows = [{"iteration": "observed", **observed}]

    for i in range(n_iter):
        shifted = shots.copy()
        offset = rng.integers(-max_shift_days, max_shift_days + 1)
        # Avoid a near-zero shift, which would not break the true pairing.
        if abs(offset) < 7:
            offset = 7 * (1 if offset >= 0 else -1)
        shifted.loc[human, "ts"] = shifted.loc[human, "ts"] + pd.Timedelta(days=int(offset))
        rows.append({
            "iteration": f"null_{i}",
            **_detection_counts(shifted, radius_m=radius_m, window_min=window_min),
        })

    out = pd.DataFrame(rows)
    nulls = out[out["iteration"].str.startswith("null")]["pct_sensor_with_human"]
    obs = out.loc[0, "pct_sensor_with_human"]
    out.attrs["observed"] = obs
    out.attrs["null_mean"] = float(nulls.mean())
    out.attrs["null_sd"] = float(nulls.std())
    out.attrs["excess"] = float(obs - nulls.mean())
    return out


def _detection_counts(
    shots: gpd.GeoDataFrame, *, radius_m: int, window_min: int
) -> dict[str, float]:
    clustered, _ = build_events(shots, radius_m=radius_m, window_min=window_min)
    g = clustered.groupby("event_id")
    has_s = g["is_sensor"].any()
    has_h = g["is_human_report"].any()
    both = int((has_s & has_h).sum())
    sensor_only = int((has_s & ~has_h).sum())
    return {
        "both": both,
        "sensor_only": sensor_only,
        "pct_sensor_with_human": round(100 * both / (both + sensor_only), 2),
    }


def sweep_event_params(
    shots: gpd.GeoDataFrame,
    *,
    radii: tuple[int, ...] = (100, 150, 250),
    windows: tuple[int, ...] = (5, 10, 20),
) -> pd.DataFrame:
    """Vary the clustering parameters and report how the event count moves."""
    rows = []
    for r in radii:
        for w in windows:
            clustered, rep = build_events(shots, radius_m=r, window_min=w)
            chain = chaining_report(clustered, radius_m=r)
            summary = summarise_events(clustered)
            modes = summary["detection_mode"].value_counts()
            rows.append({
                "radius_m": r,
                "window_min": w,
                "events": rep.events,
                "compression": round(rep.compression, 3),
                "largest_event": rep.largest_event,
                "overstretched": int(chain["overstretched"].sum()) if len(chain) else 0,
                "both": int(modes.get("both", 0)),
                "sensor_only": int(modes.get("sensor_only", 0)),
                "human_only": int(modes.get("human_only", 0)),
            })
    return pd.DataFrame(rows)


def load_shots() -> gpd.GeoDataFrame:
    return gpd.read_parquet(config.DATA_INTERIM / "shots_fired.parquet")
