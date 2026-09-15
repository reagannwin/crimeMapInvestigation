"""Coordinate validation and null-sentinel handling.

The city encodes a missing geocode as a literal ``0.0 / 0.0`` rather than
NULL. Those are valid floats: they survive every spatial join, land in the
Gulf of Guinea, and drop out of neighbourhood aggregates without raising.
Counts observed in the delivered extracts:

    Incidents_Reported_911.csv   3,062 rows at 0,0
    Police_Stop_Data.csv         2,368 rows at 0,0
    Police_Use_of_Force.csv      2,344 rows at 0,0
    Police_Incidents_2026.csv        0 rows at 0,0

They must become NA at ingest, and the loss must be reported rather than
absorbed silently.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config


@dataclass(frozen=True)
class CoordReport:
    """Outcome of cleaning one frame's coordinates."""

    total: int
    null_island: int
    out_of_bbox: int
    unparseable: int

    @property
    def usable(self) -> int:
        return self.total - self.null_island - self.out_of_bbox - self.unparseable

    @property
    def loss_pct(self) -> float:
        return 100.0 * (self.total - self.usable) / self.total if self.total else 0.0

    def __str__(self) -> str:
        return (
            f"{self.usable:,}/{self.total:,} usable ({100 - self.loss_pct:.2f}%) | "
            f"null-island={self.null_island:,} out-of-bbox={self.out_of_bbox:,} "
            f"unparseable={self.unparseable:,}"
        )


def clean_latlon(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    *,
    lat_out: str = "latitude",
    lon_out: str = "longitude",
) -> tuple[pd.DataFrame, CoordReport]:
    """Return a copy with validated ``lat_out``/``lon_out`` columns, plus a report.

    Invalid coordinates become NA; rows are never dropped here so callers can
    decide whether a row without geometry is still usable (it often is, because
    the city supplies a neighbourhood label independently of the point).
    """
    out = df.copy()
    lat = pd.to_numeric(out[lat_col], errors="coerce")
    lon = pd.to_numeric(out[lon_col], errors="coerce")

    unparseable = (lat.isna() | lon.isna()).sum()

    tol = config.NULL_ISLAND_TOL
    null_island = (lat.abs() < tol) & (lon.abs() < tol)

    lo_lat, hi_lat = config.BBOX_LAT
    lo_lon, hi_lon = config.BBOX_LON
    in_bbox = lat.between(lo_lat, hi_lat) & lon.between(lo_lon, hi_lon)
    out_of_bbox = ~in_bbox & ~null_island & lat.notna() & lon.notna()

    bad = null_island | out_of_bbox
    out[lat_out] = lat.where(~bad, np.nan)
    out[lon_out] = lon.where(~bad, np.nan)

    report = CoordReport(
        total=len(out),
        null_island=int(null_island.sum()),
        out_of_bbox=int(out_of_bbox.sum()),
        unparseable=int(unparseable),
    )
    return out, report


def to_geoframe(df: pd.DataFrame, lat: str = "latitude", lon: str = "longitude"):
    """Build a projected GeoDataFrame (EPSG:26915) from cleaned coordinates.

    Rows with NA coordinates are kept with null geometry so that record counts
    stay reconcilable with the source frame.
    """
    import geopandas as gpd  # imported lazily; not all pipeline steps need it

    geom = gpd.points_from_xy(df[lon], df[lat], crs=config.CRS_GEOGRAPHIC)
    gdf = gpd.GeoDataFrame(df.copy(), geometry=geom, crs=config.CRS_GEOGRAPHIC)
    return gdf.to_crs(config.CRS_PROJECTED)
