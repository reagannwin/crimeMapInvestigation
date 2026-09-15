"""Phase 1b: pull the ArcGIS layers and persist them as GeoParquet.

    python -m src.ingest.build_geo
"""
from __future__ import annotations

import argparse
import sys
import time

from src import config
from src.ingest.arcgis_sources import ARCGIS_LOADERS


def build(names: list[str] | None = None) -> int:
    targets = names or list(ARCGIS_LOADERS)
    config.DATA_INTERIM.mkdir(parents=True, exist_ok=True)

    failures = 0
    for name in targets:
        print(f"\n{'=' * 72}\n>> {name}", flush=True)
        t0 = time.perf_counter()
        try:
            result = ARCGIS_LOADERS[name]()
        except Exception as exc:
            print(f"   FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            failures += 1
            continue

        out = config.DATA_INTERIM / f"{name}.parquet"
        frame = result.frame.drop(columns=["ts_epoch"], errors="ignore")
        frame.to_parquet(out, index=False, compression="zstd")
        print(result.summary())
        print(f"   wrote:  {out.name} ({out.stat().st_size / 1e6:.1f} MB) "
              f"in {time.perf_counter() - t0:.1f}s")

    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sources", nargs="*", help="subset to build (default: all)")
    args = ap.parse_args()

    failures = build(args.sources or None)
    print(f"\n{'=' * 72}")
    print("DONE" if not failures else f"DONE with {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
