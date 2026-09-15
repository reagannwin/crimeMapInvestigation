"""Phase 7: render every figure to ``reports/figures``.

    python -m src.viz.build_figures

Each figure states its finding in the subtitle rather than only labelling what
is plotted, so it can be read without the surrounding prose.
"""
from __future__ import annotations

import argparse
import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from src import config
from src.viz import charts, maps, style

FIGURES = config.PROJECT_ROOT / "reports" / "figures"


def _save(fig, name: str) -> str:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    size_kb = path.stat().st_size / 1024
    print(f"   {path.relative_to(config.PROJECT_ROOT)}  ({size_kb:.0f} KB)")
    return str(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--imputed-initiation", action="store_true")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    style.apply()
    P = config.DATA_PROCESSED
    suffix = "_imputed" if args.imputed_initiation else ""

    polys = maps.load_geo()
    print(">> rendering")

    # 1. Sensor coverage -- the gate on every sensor-vs-human comparison.
    coverage = pd.read_parquet(P / "sensor_coverage.parquet")
    footprint = gpd.read_parquet(P / "sensor_footprint.parquet").geometry.iloc[0]
    _save(maps.coverage_map(polys, coverage, footprint), "01_sensor_coverage")

    # 2. Reporting propensity, inside the footprint only.
    rates = pd.read_parquet(P / "gunfire_reporting_rates.parquet")
    _save(charts.reporting_rate_chart(rates), "02_reporting_rate")
    _save(
        maps.choropleth(
            polys, rates, "report_rate",
            title="Gunfire reporting rate, sensor-covered neighbourhoods",
            subtitle="Share of sensor-detected gunfire that also drew a 911 "
                     "call. Grey is outside the sensor footprint.",
            legend_label="Calls per sensor-detected event",
            fmt="{:.0%}",
            note="400 m / 10 min linkage; 98% of matches exceed the null model",
        ),
        "03_reporting_rate_map",
    )

    # 3. Enforcement trend.
    citywide = pd.read_csv(P / f"enforcement_citywide{suffix}.csv", index_col="year")
    _save(
        charts.enforcement_trend(citywide, imputed=args.imputed_initiation),
        "04_enforcement_trend",
    )

    # 4. Stops per crime, most recent full year.
    annual = pd.read_parquet(P / f"enforcement_annual{suffix}.parquet")
    latest = int(annual["year"].max())
    recent = annual[annual["year"] == latest]
    _save(
        maps.choropleth(
            polys, recent, "stops_per_crime_eb",
            title=f"Stops per recorded crime, {latest}",
            subtitle="Empirical-Bayes smoothed, so thin cells cannot top the "
                     "scale. Above 1 means more stops than recorded offences.",
            legend_label="Stops per crime",
            cmap=style.RAMP_NEUTRAL, fmt="{:.2f}",
            note=f"{latest} is the most recent fully covered year",
        ),
        "05_stops_per_crime",
    )

    # 5. Hit rate -- the outcome test and its stratification.
    hit_main = pd.read_parquet(P / "hit_rate_main.parquet")
    recent_hit = (
        hit_main[hit_main["window"] == "2024-2026"]
        .set_index("race_recorded")
        .drop(columns=["window"], errors="ignore")
    )
    by_init = pd.read_csv(P / "hit_rate_by_initiation.csv", index_col=0)
    _save(charts.hit_rate_chart(recent_hit, by_init), "06_hit_rate")

    # 6. Spatial structure.
    diagnostics = pd.read_csv(P / "spatial_diagnostics.csv")
    _save(charts.elasticity_chart(diagnostics), "07_crime_elasticity")
    lisa = pd.read_parquet(P / "spatial_lisa.parquet")
    _save(maps.lisa_map(polys, lisa, year=int(lisa["year"].iloc[0])), "08_lisa_clusters")

    print(f"\n>> 8 figures in {FIGURES.relative_to(config.PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
