"""Outcome (hit-rate) test for discrimination in discretionary searches.

The logic
---------
Searching requires a judgement that evidence is likely to be found. If officers
apply a *lower* evidentiary bar to one group, they will search more members of
that group who turn out to have nothing -- so that group's searches yield a
*lower* hit rate. The comparison of hit rates across groups is therefore a test
of the threshold being applied, and unlike a search-rate comparison it needs no
population denominator and no assumption about who is on the street. That is
why it is the most defensible discrimination test available in this data.

Choosing the race variable
--------------------------
``preRace`` (perceived before contact) is theoretically the right variable,
since an officer can only act on what they perceive. In this dataset it is
empirically unusable:

    year   preRace "Unknown"   preRace "Other"
    2018              68.2%               1.0%
    2021              74.4%               0.7%
    2024               7.6%              41.6%
    2026              21.7%              37.4%

The 2024 recording change cut "Unknown" by moving the mass into "Other", which
is not a group. ``race`` (recorded after contact) is far better: "Unknown"
falls to 0.0% from 2024 and "Other" to roughly 5% by 2025. This module
therefore uses ``race_recorded`` and reports ``race_perceived`` only as a
sensitivity check.

What counts as a hit
--------------------
``BKG-Booking`` (arrest) is the primary measure. A citation is a weak proxy --
a traffic citation says nothing about whether a search found anything -- so it
is reported separately rather than pooled.

Limitations that must travel with any result
--------------------------------------------
1. **Arrest may precede the search.** A search incident to arrest inverts the
   causal order this test assumes. Nothing in the data distinguishes the two,
   and if the arrest-first share differs by race the comparison is biased. This
   is the single largest threat to the finding.
2. **Infra-marginality.** Equal hit rates do not prove equal thresholds. If the
   distribution of actual wrongdoing differs across groups, identical
   thresholds can produce different hit rates and vice versa. The outcome test
   detects a *sign*, not a magnitude; a threshold test would be needed for the
   latter.
3. **Aggregation.** Rates pooled across officers, places and stop types can
   reverse under stratification (Simpson's paradox). :func:`hit_rate_by_stratum`
   exists to check that.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config

REFERENCE_RACE = "White"
# Groups that are not people, plus the residual codes that are not a group.
NON_GROUP = frozenset(config.NON_PERSON_RACE | {"Unknown", "Other"})
# Minimum searches before a hit rate is reported at all.
MIN_SEARCHES = 50

HIT_DEFINITIONS = {
    "booking": lambda d: d["disposition"].eq("BKG-Booking"),
    "citation": lambda d: d["citation"].fillna(False),
    "booking_or_citation": lambda d: (
        d["disposition"].eq("BKG-Booking") | d["citation"].fillna(False)
    ),
}

SEARCH_DEFINITIONS = {
    "person": lambda d: d["search_person"],
    "vehicle": lambda d: d["search_vehicle"],
    "either": lambda d: (
        d["search_person"].fillna(False) | d["search_vehicle"].fillna(False)
    ),
}


@dataclass(frozen=True)
class HitRateResult:
    table: pd.DataFrame
    race_col: str
    hit: str
    search: str
    year_range: tuple[int, int]

    def __str__(self) -> str:
        lo, hi = self.year_range
        return (
            f"{self.race_col} | search={self.search} | hit={self.hit} | {lo}-{hi}"
        )


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval.

    Preferred over the normal approximation because hit rates are computed on
    small denominators for smaller groups, where the normal interval can run
    outside [0, 1] and understates uncertainty.
    """
    if trials == 0:
        return (np.nan, np.nan)
    p = successes / trials
    denom = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denom
    half = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def two_proportion_test(s1: int, n1: int, s2: int, n2: int) -> tuple[float, float]:
    """Pooled two-proportion z-test. Returns (z, two-sided p)."""
    from scipy import stats

    if min(n1, n2) == 0:
        return (np.nan, np.nan)
    p_pool = (s1 + s2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return (np.nan, np.nan)
    z = (s1 / n1 - s2 / n2) / se
    return (z, 2 * (1 - stats.norm.cdf(abs(z))))


def prepare(
    stops: pd.DataFrame,
    *,
    race_col: str = "race_recorded",
    year_min: int = 2024,
    year_max: int = 2026,
) -> pd.DataFrame:
    """Filter to stops where the test is meaningful.

    Drops pre-``usable_window`` years, records with no person contacted, and
    residual race codes that do not name a group.
    """
    person_flag = (
        "person_recorded" if race_col == "race_recorded" else "person_perceived"
    )
    d = stops[
        stops["usable_window"]
        & stops[person_flag]
        & stops["year"].between(year_min, year_max)
    ].copy()
    return d[~d[race_col].isin(NON_GROUP)]


def hit_rate_table(
    stops: pd.DataFrame,
    *,
    race_col: str = "race_recorded",
    hit: str = "booking",
    search: str = "person",
    year_min: int = 2024,
    year_max: int = 2026,
    min_searches: int = MIN_SEARCHES,
) -> HitRateResult:
    """Search rate and hit rate by race, with intervals and tests vs White."""
    d = prepare(stops, race_col=race_col, year_min=year_min, year_max=year_max)
    searched_mask = SEARCH_DEFINITIONS[search](d).fillna(False)
    d = d.assign(_searched=searched_mask, _hit=HIT_DEFINITIONS[hit](d))

    grouped = d.groupby(race_col, observed=True)
    table = pd.DataFrame({
        "stops": grouped.size(),
        "searches": grouped["_searched"].sum(),
    })
    searched = d[d["_searched"]]
    hits = searched.groupby(race_col, observed=True)["_hit"].agg(["size", "sum"])
    table = table.join(hits.rename(columns={"size": "n_searched", "sum": "n_hit"}))
    table = table.fillna(0).astype({"n_searched": int, "n_hit": int})

    table["search_rate"] = table["searches"] / table["stops"]
    table["hit_rate"] = np.where(
        table["n_searched"] > 0, table["n_hit"] / table["n_searched"], np.nan
    )

    intervals = [
        wilson_interval(int(r.n_hit), int(r.n_searched)) for r in table.itertuples()
    ]
    table["hit_lo"] = [lo for lo, _ in intervals]
    table["hit_hi"] = [hi for _, hi in intervals]

    table = table[table["n_searched"] >= min_searches]

    if REFERENCE_RACE in table.index:
        ref = table.loc[REFERENCE_RACE]
        table["search_rate_ratio"] = table["search_rate"] / ref["search_rate"]
        stats_rows = [
            two_proportion_test(
                int(r.n_hit), int(r.n_searched), int(ref.n_hit), int(ref.n_searched)
            )
            for r in table.itertuples()
        ]
        table["hit_gap_pp"] = (table["hit_rate"] - ref["hit_rate"]) * 100
        table["z"] = [z for z, _ in stats_rows]
        table["p_value"] = [p for _, p in stats_rows]

    table = table.sort_values("search_rate", ascending=False)
    return HitRateResult(
        table=table.round(4), race_col=race_col, hit=hit, search=search,
        year_range=(year_min, year_max),
    )


def hit_rate_by_stratum(
    stops: pd.DataFrame,
    stratum: str,
    *,
    race_col: str = "race_recorded",
    hit: str = "booking",
    search: str = "person",
    year_min: int = 2024,
    year_max: int = 2026,
    min_searches: int = MIN_SEARCHES,
) -> pd.DataFrame:
    """Hit rate split by a stratum, to test whether the pooled gap survives.

    A pooled difference that vanishes or reverses within strata is a Simpson's
    paradox artefact, not evidence about thresholds.
    """
    d = prepare(stops, race_col=race_col, year_min=year_min, year_max=year_max)
    d = d.assign(
        _searched=SEARCH_DEFINITIONS[search](d).fillna(False),
        _hit=HIT_DEFINITIONS[hit](d),
    )
    searched = d[d["_searched"]]

    out = (
        searched.groupby([stratum, race_col], observed=True)["_hit"]
        .agg(n_searched="size", n_hit="sum")
        .reset_index()
    )
    out = out[out["n_searched"] >= min_searches]
    out["hit_rate"] = (out["n_hit"] / out["n_searched"]).round(4)

    wide = out.pivot(index=stratum, columns=race_col, values="hit_rate")
    if REFERENCE_RACE in wide.columns and "Black" in wide.columns:
        wide["black_white_gap_pp"] = (
            (wide["Black"] - wide[REFERENCE_RACE]) * 100
        ).round(1)
    return wide


def sensitivity(stops: pd.DataFrame) -> pd.DataFrame:
    """Black-White hit-rate gap under every reasonable analytic choice.

    If the sign of the gap depends on the definition, there is no finding.
    """
    rows = []
    windows = [(2018, 2023), (2024, 2026), (2025, 2026), (2018, 2026)]
    for race_col in ("race_recorded", "race_perceived"):
        for hit in HIT_DEFINITIONS:
            for search in SEARCH_DEFINITIONS:
                for lo, hi in windows:
                    res = hit_rate_table(
                        stops, race_col=race_col, hit=hit, search=search,
                        year_min=lo, year_max=hi,
                    )
                    t = res.table
                    if "Black" not in t.index or REFERENCE_RACE not in t.index:
                        continue
                    rows.append({
                        "race_col": race_col, "hit": hit, "search": search,
                        "years": f"{lo}-{hi}",
                        "black_searched": int(t.loc["Black", "n_searched"]),
                        "black_hit_rate": t.loc["Black", "hit_rate"],
                        "white_hit_rate": t.loc[REFERENCE_RACE, "hit_rate"],
                        "gap_pp": round(t.loc["Black", "hit_gap_pp"], 2),
                        "search_rate_ratio": round(
                            t.loc["Black", "search_rate_ratio"], 2
                        ),
                        "p_value": t.loc["Black", "p_value"],
                    })
    return pd.DataFrame(rows)


def load_stops() -> pd.DataFrame:
    return pd.read_parquet(config.DATA_INTERIM / "stops.parquet")
