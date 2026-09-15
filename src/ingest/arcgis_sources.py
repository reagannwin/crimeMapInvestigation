"""Loaders for the two layers only available from ArcGIS.

``Shots_Fired`` is the centrepiece and the one most easily misused. Its
``Problem`` column mixes an acoustic sensor feed with ordinary 911 calls:

    ShotSpotter Activation (P)   37,896   sensor
    Sound of Shots Fired (P)     63,254   human 911 call
    Shooting (PFE)                5,507   confirmed shooting
    Shooting Report Only (P)      2,143   delayed / walk-in report

Only the first is machine-generated. Treating the layer as a single
"objective gunfire" measure correlates community reporting with itself, so the
loader splits it into ``detection`` classes and refuses to hand back an
undifferentiated frame.

``Minneapolis_Neighborhoods`` supplies the polygon geometry the delivered
attribute-table CSV lacks. It is the spatial-join target for every point layer
and the basis for the ShotSpotter coverage mask in Phase 4.
"""
from __future__ import annotations

import pandas as pd

from src import config
from src.clean import neighborhoods as nb
from src.ingest.arcgis import fetch_geoframe
from src.ingest.csv_sources import LOCAL_TZ, LoadResult

# Shots_Fired.Problem -> detection class. Keys are exhaustive for the layer.
DETECTION_CLASS = {
    "ShotSpotter Activation (P)": "sensor",
    "Sound of Shots Fired (P)": "human_call",
    "Shooting Report Only (P)": "human_report",
    "Shooting (PFE)": "confirmed_shooting",
}


def load_shots_fired(*, verbose: bool = True) -> LoadResult:
    """108,800 gunfire records, 2007-2026, as a projected GeoDataFrame."""
    gdf = fetch_geoframe(config.LAYERS["shots_fired"], verbose=verbose)

    gdf = gdf.rename(columns={
        "Master_Incident_Number": "incident_id",
        "Problem": "problem",
        "Response_Date": "ts_epoch",
        "Jurisdiction": "jurisdiction",
        "Division": "division",
    })

    # Response_Date arrives as epoch milliseconds UTC in the GeoJSON payload.
    gdf["ts"] = (
        pd.to_datetime(gdf["ts_epoch"], unit="ms", utc=True).dt.tz_convert(LOCAL_TZ)
    )
    gdf["incident_id"] = gdf["incident_id"].astype("string").str.strip().str.upper()

    unknown = set(gdf["problem"].dropna().unique()) - set(DETECTION_CLASS)
    if unknown:
        raise ValueError(f"unmapped Shots_Fired problem value(s): {sorted(unknown)}")

    gdf["detection"] = gdf["problem"].map(DETECTION_CLASS).astype("string")
    gdf["is_sensor"] = gdf["detection"].eq("sensor")
    gdf["is_human_report"] = gdf["detection"].isin({"human_call", "human_report"})

    gdf["year"] = gdf["ts"].dt.year.astype("Int16")
    gdf["month_start"] = (
        gdf["ts"].dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()
    )
    gdf["hour"] = gdf["ts"].dt.hour.astype("Int8")
    gdf["dow"] = gdf["ts"].dt.dayofweek.astype("Int8")

    counts = gdf["detection"].value_counts()
    sensor_n = int(counts.get("sensor", 0))
    return LoadResult(
        "shots_fired", gdf,
        notes=[
            "detection split: " + ", ".join(f"{k}={v:,}" for k, v in counts.items()),
            f"only {sensor_n:,} of {len(gdf):,} rows ({100 * sensor_n / len(gdf):.1f}%) "
            "are sensor-generated; the rest are human reports",
            "no neighbourhood field on this layer -- assign by spatial join",
        ],
    )


def load_neighborhood_polygons(*, verbose: bool = True) -> LoadResult:
    """87 neighbourhood polygons, projected to EPSG:26915."""
    gdf = fetch_geoframe(config.LAYERS["neighborhoods"], verbose=verbose)

    gdf = gdf.rename(columns={"BDNAME": "neighborhood_raw", "BDNUM": "neighborhood_num"})
    gdf["neighborhood"] = nb.canonicalise(gdf["neighborhood_raw"])

    if gdf["neighborhood"].nunique() != 87:
        raise ValueError(
            f"expected 87 neighbourhoods, got {gdf['neighborhood'].nunique()}"
        )
    if gdf["neighborhood"].duplicated().any():
        dupes = gdf.loc[gdf["neighborhood"].duplicated(), "neighborhood"].tolist()
        raise ValueError(f"duplicate neighbourhood polygons: {dupes}")

    invalid = int((~gdf.geometry.is_valid).sum())
    if invalid:
        gdf["geometry"] = gdf.geometry.make_valid()

    # Area in km^2 from the projected geometry, for population density and
    # per-area rates. Shape__Area in the source is in Web Mercator units and
    # is badly inflated at this latitude -- do not use it.
    gdf["area_km2"] = gdf.geometry.area / 1e6

    return LoadResult(
        "neighborhood_polygons", gdf,
        notes=[
            f"{len(gdf)} polygons, CRS {gdf.crs.to_string()}",
            f"{invalid} invalid geometries repaired" if invalid
            else "all geometries valid",
            f"total area {gdf['area_km2'].sum():.1f} km2",
            "source Shape__Area is Web Mercator and inflated; use area_km2",
        ],
    )


ARCGIS_LOADERS = {
    "shots_fired": load_shots_fired,
    "neighborhood_polygons": load_neighborhood_polygons,
}
