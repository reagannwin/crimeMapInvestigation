"""Phase 5c: build the neighbourhood x month enforcement panel.

    python -m src.features.build_panel

Writes to ``data/processed``:

    enforcement_panel.parquet        neighbourhood x month
    enforcement_annual.parquet       neighbourhood x year, smoothed rates
    enforcement_citywide.csv         citywide series

Answers the original question -- does enforcement track measured crime? --
without needing the ACS denominators, because every metric here is a ratio of
two police-activity or crime measures rather than a per-capita rate.
"""
from __future__ import annotations

import pandas as pd

from src import config
from src.features import panel as pn


def main() -> int:
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)

    sources = pn.load_sources()
    panel, report = pn.build_panel(**sources)
    print(f">> {report}")

    panel.to_parquet(config.DATA_PROCESSED / "enforcement_panel.parquet", index=False)

    annual = pn.annual_summary(panel)
    annual.to_parquet(
        config.DATA_PROCESSED / "enforcement_annual.parquet", index=False
    )
    years = sorted(int(y) for y in annual["year"].unique())
    print(f"   annual summary: {len(annual):,} neighbourhood-years, {years}")
    print("   (a year is kept only when all 12 months are covered by stops, "
          "calls AND crime)")

    citywide = annual.groupby("year")[
        ["stops_total", "stops_proactive", "stops_reactive", "calls", "crimes"]
    ].sum()
    citywide["stops_per_crime"] = (
        citywide["stops_total"] / citywide["crimes"]
    ).round(3)
    citywide["proactive_share"] = (
        citywide["stops_proactive"] / citywide["stops_total"]
    ).round(3)
    citywide["proactive_per_reactive"] = (
        citywide["stops_proactive"] / citywide["stops_reactive"]
    ).round(3)
    citywide.to_csv(config.DATA_PROCESSED / "enforcement_citywide.csv")

    print("\n>> citywide")
    print(citywide.to_string())

    first, last = years[0], years[-1]
    lo, hi = citywide.loc[first], citywide.loc[last]

    def pct(a, b):
        return 100 * (b - a) / a

    print(f"\n>> {first} to {last}")
    for label, key in (
        ("reported crime", "crimes"),
        ("911 calls (police)", "calls"),
        ("stops, total", "stops_total"),
        ("stops, proactive", "stops_proactive"),
        ("stops, reactive", "stops_reactive"),
        ("stops per crime", "stops_per_crime"),
    ):
        print(f"   {label:<22} {pct(lo[key], hi[key]):+7.1f}%")

    print("\n>> neighbourhoods: largest rise in proactive stops per crime")
    wide = annual.pivot(
        index="neighborhood", columns="year", values="stops_per_crime_eb"
    )
    if first in wide.columns and last in wide.columns:
        change = (
            pd.DataFrame({
                "first": wide[first], "last": wide[last],
                "change": wide[last] - wide[first],
            })
            .dropna()
            .sort_values("change", ascending=False)
        )
        print("   (empirical-Bayes smoothed, so thin cells cannot top the list)")
        print(change.head(10).round(3).to_string())

    print("\n>> caveats")
    for line in (
        "use of force has TWO recording breaks (2023-01 level, 2024-01 "
        "granularity) and must not be compared across them -- see force_regime",
        "`initiation` is the observed field: 26.9% of stops are blank, so "
        "proactive + reactive < total. Rebuild with use_imputed_initiation=True "
        "to use the 911-join recovery instead",
        "2021 is the first fully covered year because the 911 extract begins "
        "2020-07",
    ):
        print(f"   - {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
