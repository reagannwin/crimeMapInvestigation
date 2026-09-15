"""Phase 1c: pull ACS demographics and interpolate to neighbourhoods.

    set CENSUS_API_KEY=...        # Windows
    export CENSUS_API_KEY=...     # bash
    python -m src.ingest.build_census

Writes ``data/processed/neighborhood_demographics.parquet``.

The block-group geometry half needs no key and is verified working. Only the
ACS variable request is gated: unkeyed requests 302 to a "Missing Key" page.
Free key: https://api.census.gov/data/key_signup.html
"""
from __future__ import annotations

import argparse
import sys

import geopandas as gpd

from src import config
from src.ingest import census


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", type=int, default=census.ACS_YEAR)
    ap.add_argument("--api-key", default=None,
                    help="overrides CENSUS_API_KEY; not persisted")
    args = ap.parse_args()

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    print(">> ACS estimates")
    try:
        acs = census.fetch_acs(year=args.year, api_key=args.api_key)
    except census.CensusAuthError as exc:
        print(f"\nBLOCKED: {exc}", file=sys.stderr)
        return 2

    print("\n>> block group geometry")
    bg = census.fetch_block_group_geometry(year=args.year)

    print("\n>> interpolating to neighbourhoods")
    polys = gpd.read_parquet(config.DATA_INTERIM / "neighborhood_polygons.parquet")
    demo, report = census.interpolate_to_neighborhoods(acs, bg, polys)
    print(f"   {report}")

    out = config.DATA_PROCESSED / "neighborhood_demographics.parquet"
    demo.to_parquet(out, index=False)
    print(f"   wrote {out.relative_to(config.PROJECT_ROOT)}")

    print("\n=== sanity check: most and least white neighbourhoods ===")
    cols = ["neighborhood", "pop_total", "share_black_nh", "share_white_nh",
            "share_renter", "median_hh_income_approx"]
    ranked = demo.sort_values("share_white_nh")
    print(ranked[cols].head(5).round(3).to_string(index=False))
    print("   ...")
    print(ranked[cols].tail(5).round(3).to_string(index=False))
    print(f"\n   city population from interpolation: {demo['pop_total'].sum():,.0f} "
          "(Minneapolis is roughly 425,000)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
