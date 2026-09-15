"""Phase 5a: collapse gunfire records into events and measure reporting rates.

    python -m src.join.build_events

Writes to ``data/processed``:

    gunfire_events.parquet          one row per event, with detection mode
    gunfire_reporting_rates.parquet per-neighbourhood reporting rate
    gunfire_event_sweep.csv         parameter sensitivity

Uses ``EVENT_LINKAGE_RADIUS_M`` (400 m), not the tighter dedup radius: pairing
a sensor fix to the 911 call about the same gunfire has to absorb the gap
between an estimated acoustic position and a caller's reported address.
"""
from __future__ import annotations

import geopandas as gpd
import pandas as pd

from src import config
from src.join import events as ev


def main() -> int:
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    shots = ev.load_shots()
    radius = config.EVENT_LINKAGE_RADIUS_M
    window = config.EVENT_CLUSTER_WINDOW_MIN

    clustered, report = ev.build_events(shots, radius_m=radius, window_min=window)
    print(f">> clustering at {radius} m / {window} min")
    print(f"   {report}")

    chain = ev.chaining_report(clustered, radius_m=radius)
    over = int(chain["overstretched"].sum())
    print(f"   chaining: max span {chain['span_m'].max():.0f} m, "
          f"p99 {chain['span_m'].quantile(0.99):.0f} m, overstretched {over}")
    if over:
        print(f"   WARNING: {over} clusters exceed 3x the radius; inspect before use")

    summary = ev.summarise_events(clustered)

    polys = gpd.read_parquet(
        config.DATA_INTERIM / "neighborhood_polygons.parquet"
    )[["neighborhood", "geometry"]]
    coverage = pd.read_parquet(
        config.DATA_PROCESSED / "sensor_coverage.parquet"
    )[["neighborhood", "coverage_class"]]

    summary = (
        gpd.sjoin(summary, polys, how="left", predicate="within")
        .drop(columns=["index_right"])
        .merge(coverage, on="neighborhood", how="left")
    )
    summary.to_parquet(config.DATA_PROCESSED / "gunfire_events.parquet", index=False)

    print("\n>> detection mode")
    print(summary["detection_mode"].value_counts().to_string())

    # Reporting rate is only meaningful where sensors actually exist.
    covered = summary[
        (summary["coverage_class"] == "covered") & summary["has_sensor"]
    ]
    rates = (
        covered.groupby("neighborhood")
        .agg(sensor_events=("has_sensor", "sum"), with_call=("has_human", "sum"))
    )
    rates = rates[rates["sensor_events"] >= config.MIN_CELL_EVENTS]
    rates["report_rate"] = (rates["with_call"] / rates["sensor_events"]).round(4)
    rates = rates.sort_values("report_rate").reset_index()
    rates.to_parquet(
        config.DATA_PROCESSED / "gunfire_reporting_rates.parquet", index=False
    )

    print(f"\n>> reporting rate, {len(rates)} covered neighbourhoods "
          f"(>= {config.MIN_CELL_EVENTS} sensor events)")
    print(f"   overall: {covered['has_human'].sum():,}/{len(covered):,} "
          f"({100 * covered['has_human'].mean():.1f}%) of sensor-detected "
          "gunfire also drew a 911 call")
    print(rates.to_string(index=False))

    print("\n>> null model: how much of that is coincidence?")
    null = ev.null_match_rate(shots, radius_m=radius, window_min=window, n_iter=10)
    a = null.attrs
    print(f"   observed {a['observed']:.2f}%  null {a['null_mean']:.2f}% "
          f"(+/- {a['null_sd']:.2f})  excess {a['excess']:.2f} pp")

    sweep = ev.sweep_event_params(shots)
    sweep.to_csv(config.DATA_PROCESSED / "gunfire_event_sweep.csv", index=False)
    print(f"\n   wrote sweep ({len(sweep)} settings)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
