"""Phase 4: derive and persist the ShotSpotter coverage mask.

    python -m src.bias.build_footprint

Writes three artefacts to ``data/processed``:

    sensor_footprint.parquet        the mask geometry
    sensor_coverage.parquet         per-neighbourhood coverage + class
    sensor_footprint_sweep.csv      parameter sensitivity

Every sensor-versus-human comparison downstream must filter to
``coverage_class == "covered"``. The mask is the gate, not a decoration.
"""
from __future__ import annotations

import geopandas as gpd

from src import config
from src.bias import sensor_footprint as sf


def main() -> int:
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    acts = sf.load_activations()
    polys = sf.load_polygons()
    city_km2 = polys.geometry.area.sum() / 1e6

    print(f"sensor activations: {len(acts):,}")
    fp = sf.build_footprint(acts, clip_to=polys)
    cov = sf.coverage_by_neighborhood(fp, polys, acts)

    print(f"\nmask: {fp}")
    print(f"  {fp.area_km2:.1f} km2 = {100 * fp.area_km2 / city_km2:.1f}% of the city")
    print("\n" + cov["coverage_class"].value_counts().to_string())

    covered = cov[cov["coverage_class"] == "covered"]
    print(f"\ncovered: {len(covered)} neighbourhoods holding "
          f"{covered['activations'].sum():,} activations "
          f"({100 * covered['activations'].sum() / len(acts):.1f}% of all)")
    downgraded = cov[cov["evidence_downgraded"]]
    if len(downgraded):
        print(f"downgraded for thin evidence: {len(downgraded)} "
              f"({', '.join(downgraded['neighborhood'])})")

    mask_gdf = gpd.GeoDataFrame(
        {"cell_m": [fp.cell_m], "min_points": [fp.min_points],
         "closing_m": [fp.closing_m], "area_km2": [fp.area_km2]},
        geometry=[fp.geometry], crs=config.CRS_PROJECTED,
    )
    mask_path = config.DATA_PROCESSED / "sensor_footprint.parquet"
    cov_path = config.DATA_PROCESSED / "sensor_coverage.parquet"
    mask_gdf.to_parquet(mask_path, index=False)
    cov.to_parquet(cov_path, index=False)

    print("\nrunning parameter sweep...")
    sweep = sf.sweep_parameters(acts, polys)
    sweep_path = config.DATA_PROCESSED / "sensor_footprint_sweep.csv"
    sweep.to_csv(sweep_path, index=False)
    print(f"  n_covered across 27 settings: "
          f"{sweep['n_covered'].min()}-{sweep['n_covered'].max()}")
    print(f"  n_uncovered:                  "
          f"{sweep['n_uncovered'].min()}-{sweep['n_uncovered'].max()} "
          "(the exclusion set is what must be stable)")

    for p in (mask_path, cov_path, sweep_path):
        print(f"  wrote {p.relative_to(config.PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
