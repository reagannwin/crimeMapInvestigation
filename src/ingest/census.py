"""ACS demographics, interpolated from census block groups to neighbourhoods.

Why not the city's own demographic layer
----------------------------------------
``City Council Wards with Demographics`` on the Minneapolis portal carries
2010 Census *age bands only* -- no race, no income, no tenure -- across 13
wards. It is sixteen years stale and far too coarse to serve as a denominator
for 87 neighbourhoods. ACS block groups are the right unit.

Geography does not nest
-----------------------
Minneapolis neighbourhoods are not built from census geography, so block
groups straddle neighbourhood boundaries and the join must interpolate rather
than look up. Two kinds of variable are handled differently:

  * **Extensive** (counts: population, race counts, tenure counts) are split
    by the share of each block group's area falling in each neighbourhood.
    This assumes population is uniform within a block group -- the standard
    assumption, and the main source of error. Block groups are small enough
    that it is usually tolerable, and worst in the ones that mix a dense core
    with a park or industrial strip.
  * **Intensive** (median household income) cannot be summed or area-weighted.
    It is combined as a population-weighted mean of block group medians, which
    is an approximation of a median-of-medians and is flagged as such. Do not
    report it as "the neighbourhood's median income" without that caveat.

Credentials
-----------
The Census API now requires a key: unkeyed requests 302 to
``/data/missing_key.html``. Get a free one at
https://api.census.gov/data/key_signup.html and expose it as ``CENSUS_API_KEY``
(or pass ``api_key=``). The key is never written to disk by this module.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

from src import config

ACS_YEAR = 2023  # ACS 5-year 2019-2023
STATE_FIPS = "27"       # Minnesota
COUNTY_FIPS = "053"     # Hennepin (contains all of Minneapolis)

API_BASE = "https://api.census.gov/data"
TIGER_BG = (
    "https://www2.census.gov/geo/tiger/TIGER{year}/BG/tl_{year}_{state}_bg.zip"
)

# ACS "jam values": sentinels in the median-income field meaning the estimate
# is unavailable, not a real dollar amount. They are large negatives and will
# quietly destroy any mean that does not exclude them.
JAM_THRESHOLD = -666666

ACS_VARIABLES = {
    "B01003_001E": "pop_total",
    "B03002_003E": "pop_white_nh",
    "B03002_004E": "pop_black_nh",
    "B03002_005E": "pop_aian_nh",
    "B03002_006E": "pop_asian_nh",
    "B03002_012E": "pop_hispanic",
    "B19013_001E": "median_hh_income",
    "B25003_001E": "hh_occupied",
    "B25003_002E": "hh_owner",
    "B25003_003E": "hh_renter",
}

EXTENSIVE = [
    "pop_total", "pop_white_nh", "pop_black_nh", "pop_aian_nh",
    "pop_asian_nh", "pop_hispanic", "hh_occupied", "hh_owner", "hh_renter",
]
INTENSIVE = ["median_hh_income"]


class CensusAuthError(RuntimeError):
    """No API key available, or the key was rejected."""


@dataclass(frozen=True)
class InterpolationReport:
    block_groups: int
    neighborhoods: int
    pop_source: float
    pop_allocated: float

    @property
    def pop_retained_pct(self) -> float:
        return 100.0 * self.pop_allocated / self.pop_source if self.pop_source else 0.0

    def __str__(self) -> str:
        return (
            f"{self.block_groups} block groups -> {self.neighborhoods} neighbourhoods | "
            f"population {self.pop_allocated:,.0f}/{self.pop_source:,.0f} "
            f"({self.pop_retained_pct:.1f}% retained; the remainder lies in "
            f"Hennepin County outside Minneapolis)"
        )


def _resolve_key(api_key: str | None) -> str:
    key = api_key or os.environ.get("CENSUS_API_KEY", "").strip()
    if not key:
        raise CensusAuthError(
            "No Census API key. Get a free key at "
            "https://api.census.gov/data/key_signup.html then set CENSUS_API_KEY."
        )
    return key


def fetch_acs(
    *, year: int = ACS_YEAR, api_key: str | None = None, verbose: bool = True
) -> pd.DataFrame:
    """Pull ACS 5-year block-group estimates for Hennepin County."""
    key = _resolve_key(api_key)
    url = f"{API_BASE}/{year}/acs/acs5"
    params = {
        "get": "NAME," + ",".join(ACS_VARIABLES),
        "for": "block group:*",
        "in": f"state:{STATE_FIPS} county:{COUNTY_FIPS}",
        "key": key,
    }
    resp = requests.get(url, params=params, timeout=90, allow_redirects=False)
    if resp.status_code in (301, 302):
        raise CensusAuthError(
            f"Census API redirected to {resp.headers.get('Location')} -- "
            "the key is missing or invalid."
        )
    resp.raise_for_status()

    rows = resp.json()
    df = pd.DataFrame(rows[1:], columns=rows[0]).rename(columns=ACS_VARIABLES)

    for col in EXTENSIVE + INTENSIVE:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Blank out jam values so they never reach an average.
    df.loc[df["median_hh_income"] < JAM_THRESHOLD, "median_hh_income"] = np.nan

    df["GEOID"] = (
        df["state"] + df["county"] + df["tract"] + df["block group"]
    )
    if verbose:
        print(f"   ACS {year}: {len(df):,} block groups, "
              f"{df['pop_total'].sum():,.0f} people")
    return df[["GEOID", "NAME", *EXTENSIVE, *INTENSIVE]]


def fetch_block_group_geometry(
    *, year: int = ACS_YEAR, verbose: bool = True
) -> gpd.GeoDataFrame:
    """TIGER/Line block group polygons for Minnesota. No API key required."""
    url = TIGER_BG.format(year=year, state=STATE_FIPS)
    if verbose:
        print(f"   TIGER {year} block groups: {url}")
    gdf = gpd.read_file(url)
    gdf = gdf[gdf["COUNTYFP"] == COUNTY_FIPS].copy()
    gdf = gdf[["GEOID", "ALAND", "geometry"]].to_crs(config.CRS_PROJECTED)
    if verbose:
        print(f"   {len(gdf):,} Hennepin County block groups")
    return gdf


def interpolate_to_neighborhoods(
    acs: pd.DataFrame,
    block_groups: gpd.GeoDataFrame,
    neighborhoods: gpd.GeoDataFrame,
) -> tuple[pd.DataFrame, InterpolationReport]:
    """Area-weighted interpolation of block-group estimates to neighbourhoods."""
    if block_groups.crs.to_string() != config.CRS_PROJECTED:
        raise ValueError("block_groups must be projected to metres")
    if neighborhoods.crs.to_string() != config.CRS_PROJECTED:
        raise ValueError("neighborhoods must be projected to metres")

    bg = block_groups.merge(acs, on="GEOID", how="inner")
    bg["bg_area"] = bg.geometry.area

    pieces = gpd.overlay(
        bg, neighborhoods[["neighborhood", "geometry"]], how="intersection",
        keep_geom_type=True,
    )
    pieces["piece_area"] = pieces.geometry.area
    # Share of the block group's area contributed to this neighbourhood.
    pieces["w"] = pieces["piece_area"] / pieces["bg_area"]

    for col in EXTENSIVE:
        pieces[col] = pieces[col] * pieces["w"]

    agg = pieces.groupby("neighborhood", as_index=False)[EXTENSIVE].sum()

    # Intensive: population-weighted mean of block-group medians.
    inc = pieces.dropna(subset=["median_hh_income"]).copy()
    inc["wpop"] = inc["pop_total"]
    weighted = (
        inc.groupby("neighborhood")
        .apply(
            lambda g: np.average(g["median_hh_income"], weights=g["wpop"])
            if g["wpop"].sum() > 0 else np.nan,
            include_groups=False,
        )
        .rename("median_hh_income_approx")
        .reset_index()
    )
    out = agg.merge(weighted, on="neighborhood", how="left")

    # Shares, computed after aggregation so they are internally consistent.
    total = out["pop_total"].replace(0, np.nan)
    out["share_black_nh"] = out["pop_black_nh"] / total
    out["share_white_nh"] = out["pop_white_nh"] / total
    out["share_hispanic"] = out["pop_hispanic"] / total
    out["share_poc"] = 1 - out["share_white_nh"]
    hh = out["hh_occupied"].replace(0, np.nan)
    out["share_renter"] = out["hh_renter"] / hh

    area = (
        neighborhoods.assign(area_km2=neighborhoods.geometry.area / 1e6)
        [["neighborhood", "area_km2"]]
    )
    out = out.merge(area, on="neighborhood", how="left")
    out["pop_density_km2"] = out["pop_total"] / out["area_km2"]

    report = InterpolationReport(
        block_groups=len(bg),
        neighborhoods=out["neighborhood"].nunique(),
        pop_source=float(bg["pop_total"].sum()),
        pop_allocated=float(out["pop_total"].sum()),
    )
    return out, report
