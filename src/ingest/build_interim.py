"""Phase 1: parse every source CSV once and persist typed Parquet.

Run once, then never touch the CSVs again. Re-parsing the 344 MB 911 extract
costs roughly a minute; reading the Parquet costs about a second.

    python -m src.ingest.build_interim
"""
from __future__ import annotations

import argparse
import sys
import time

from src import config
from src.ingest.csv_sources import LOADERS


def build(names: list[str] | None = None, *, strict_names: bool = True) -> int:
    targets = names or list(LOADERS)
    config.DATA_INTERIM.mkdir(parents=True, exist_ok=True)

    failures = 0
    for name in targets:
        loader = LOADERS[name]
        print(f"\n{'=' * 72}\n>> {name}", flush=True)
        t0 = time.perf_counter()
        try:
            result = loader(strict_names=strict_names)
        except Exception as exc:  # surface, do not swallow
            print(f"   FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            failures += 1
            continue

        out = config.DATA_INTERIM / f"{name}.parquet"
        result.frame.to_parquet(out, index=False, compression="zstd")
        size_mb = out.stat().st_size / 1e6
        print(result.summary())
        print(f"   wrote:  {out.name} ({size_mb:.1f} MB) in {time.perf_counter() - t0:.1f}s")

    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sources", nargs="*", choices=list(LOADERS) + [], default=None,
                    help="subset to build (default: all)")
    ap.add_argument("--lenient-names", action="store_true",
                    help="map unrecognised neighbourhood spellings to NA "
                         "instead of raising")
    args = ap.parse_args()

    failures = build(args.sources or None, strict_names=not args.lenient_names)
    print(f"\n{'=' * 72}")
    print("DONE" if not failures else f"DONE with {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
