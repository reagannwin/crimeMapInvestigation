"""Neighbourhood x month enforcement panel.

The question this exists to answer: does police activity track measured crime,
or diverge from it?

Design decisions that matter
----------------------------
**Partial months are trimmed, not carried.** Every extract ends mid-month, and
a half-month looks exactly like a collapse in activity:

    stops        2026-06    1,037  against ~2,400 the month before
    calls        2026-07   20,504  against ~26,800
    use of force 2026-09      154  against ~8,200

Left in, those become a spurious downward trend at the end of every series.
:func:`last_complete_month` finds each source's final whole month and the panel
is cut to it.

**Zero is not missing.** A neighbourhood-month with no stops genuinely had no
stops, and must count as 0 in any mean. A neighbourhood-month *before its
source begins* is unknown and must stay NA. The panel is built on a complete
neighbourhood x month index so the first case is filled with zeros, and
per-source coverage flags keep the second as NA. Conflating them biases every
early average downward.

**Sources start at different times**, so the panel spans the widest window any
metric supports and marks per-metric validity rather than truncating
everything to the narrowest overlap:

    crime      2017-08    stops  2016-10    calls  2020-07    force  2021-12

**Rates on small denominators are smoothed.** A neighbourhood with three
offences in a month produces a stops-per-crime figure that is mostly noise and
will top any ranked list by accident. :func:`empirical_bayes_rate` shrinks
towards the citywide rate in proportion to how little evidence a cell carries.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config
from src.clean import neighborhoods as nb

# Community-initiated demand: police calls only. Fire and BCR are a different
# service and would swamp the denominator.
POLICE_ONLY = True

# Use-of-force carries two recording breaks, and neither is a change in
# behaviour. Force must not be compared across them.
#
#   2023-01  LEVEL break.       Monthly incidents fall from 499 (Dec 2022) to
#                               153 (Jan 2023) overnight and stay there. Annual
#                               incidents drop 6,846 -> 1,547.
#   2024-01  GRANULARITY break. Rows per incident jump 4.3 -> 28.8, so every
#                               force action and officer is now logged
#                               separately. Incident counts are comparable
#                               across this break; row counts are not.
FORCE_REGIME_BREAKS = ("2023-01-01", "2024-01-01")


@dataclass(frozen=True)
class PanelReport:
    neighborhoods: int
    months: int
    rows: int
    window: tuple[pd.Timestamp, pd.Timestamp]
    trimmed: dict[str, pd.Timestamp]

    def __str__(self) -> str:
        lo, hi = self.window
        trims = ", ".join(
            f"{k}<={v:%Y-%m}" for k, v in sorted(self.trimmed.items())
        )
        return (
            f"{self.rows:,} rows = {self.neighborhoods} neighbourhoods x "
            f"{self.months} months ({lo:%Y-%m} to {hi:%Y-%m}) | trimmed: {trims}"
        )


def last_complete_month(ts: pd.Series) -> pd.Timestamp:
    """The final month for which a source has whole-month coverage.

    A source whose last record falls before its month's end is partial, so the
    preceding month is the last complete one.
    """
    last = ts.max()
    if getattr(last, "tzinfo", None) is not None:
        # Drop the zone before Period arithmetic: month boundaries are being
        # compared, not instants, and keeping it only emits a warning.
        last = last.tz_localize(None)
    period = last.to_period("M")
    # Compare on date only; a record at 08:00 on the last day still leaves that
    # day incomplete, but month-level aggregates tolerate that.
    if last.date() < period.to_timestamp("M").date():
        return (period - 1).to_timestamp()
    return period.to_timestamp()


def empirical_bayes_rate(
    events: pd.Series, exposure: pd.Series, *, prior_strength: float | None = None
) -> pd.Series:
    """Poisson-Gamma shrinkage of ``events / exposure`` towards the global rate.

    ``prior_strength`` is the pseudo-exposure carried by the prior. When None it
    is estimated by method of moments from the observed dispersion, so cells
    with plenty of exposure barely move and thin cells shrink hard.
    """
    events = pd.to_numeric(events, errors="coerce").fillna(0.0)
    exposure = pd.to_numeric(exposure, errors="coerce")

    valid = exposure > 0
    if not valid.any():
        return pd.Series(np.nan, index=events.index)

    global_rate = events[valid].sum() / exposure[valid].sum()

    if prior_strength is None:
        raw = events[valid] / exposure[valid]
        var = raw.var()
        # Poisson component of the variance; the excess is real between-area
        # spread, and the ratio sets how much evidence the prior is worth.
        poisson_var = global_rate / exposure[valid].mean()
        excess = max(var - poisson_var, 1e-12)
        prior_strength = float(global_rate / excess)
        prior_strength = float(np.clip(prior_strength, 0.1, 1e6))

    alpha = global_rate * prior_strength
    out = (events + alpha) / (exposure + prior_strength)
    return out.where(valid)


def _monthly_counts(
    df: pd.DataFrame,
    *,
    name: str,
    mask: pd.Series | None = None,
    distinct: str | None = None,
) -> pd.DataFrame:
    """Monthly counts per neighbourhood.

    ``distinct`` counts unique values of a column instead of rows. Use-of-force
    carries one row per force action per subject per officer -- 256,489 rows
    across 14,958 incidents -- so counting rows would make force look roughly
    seventeen times more common than it is.
    """
    d = df if mask is None else df[mask]
    d = d.dropna(subset=["neighborhood", "month_start"])
    grouped = d.groupby(["neighborhood", "month_start"], observed=True)
    counts = grouped[distinct].nunique() if distinct else grouped.size()
    return counts.rename(name).reset_index()


def build_panel(
    *,
    stops: pd.DataFrame,
    calls: pd.DataFrame,
    crime: pd.DataFrame,
    force: pd.DataFrame,
    use_imputed_initiation: bool = False,
) -> tuple[pd.DataFrame, PanelReport]:
    """Assemble the neighbourhood x month panel."""
    trimmed = {
        "stops": last_complete_month(stops["ts"]),
        "calls": last_complete_month(calls["ts"]),
        "force": last_complete_month(force["ts"]),
        # crime_stats is already monthly aggregate, so its last month is whole.
        "crimes": crime["month_start"].max(),
    }
    stops = stops[stops["month_start"] <= trimmed["stops"]]
    calls = calls[calls["month_start"] <= trimmed["calls"]]
    force = force[force["month_start"] <= trimmed["force"]]

    init_col = "initiation_filled" if use_imputed_initiation else "initiation"
    if init_col not in stops.columns:
        raise KeyError(
            f"{init_col!r} not present; run src.join.validate_keys to add it"
        )

    # Complete index: every neighbourhood in every month of the overall span.
    names = list(nb.canonical_names())
    lo = min(stops["month_start"].min(), crime["month_start"].min())
    hi = max(trimmed.values())
    months = pd.date_range(lo, hi, freq="MS")
    index = pd.MultiIndex.from_product(
        [names, months], names=["neighborhood", "month_start"]
    )
    panel = pd.DataFrame(index=index).reset_index()

    parts = [
        _monthly_counts(stops, name="stops_total"),
        _monthly_counts(stops, name="stops_proactive",
                        mask=stops[init_col].eq("proactive")),
        _monthly_counts(stops, name="stops_reactive",
                        mask=stops[init_col].eq("reactive")),
        _monthly_counts(stops, name="stops_searched",
                        mask=stops["search_person"].fillna(False)),
        _monthly_counts(force, name="force_incidents",
                        distinct="incident_id"),
    ]
    if POLICE_ONLY:
        parts.append(_monthly_counts(calls, name="calls", mask=calls["is_police"]))
        parts.append(
            _monthly_counts(calls, name="calls_discretionary",
                            mask=calls["is_police"] & calls["is_discretionary"])
        )
    for part in parts:
        panel = panel.merge(part, on=["neighborhood", "month_start"], how="left")

    crime_wide = (
        crime.dropna(subset=["neighborhood"])
        .groupby(["neighborhood", "month_start"], observed=True)["count_zerofilled"]
        .sum()
        .rename("crimes")
        .reset_index()
    )
    panel = panel.merge(crime_wide, on=["neighborhood", "month_start"], how="left")

    # Coverage: inside a source's window a blank means zero; outside it means
    # unknown and must stay NA.
    starts = {
        "stops": stops["month_start"].min(),
        "calls": calls["month_start"].min(),
        "force": force["month_start"].min(),
        "crimes": crime["month_start"].min(),
    }
    groups = {
        "stops": ["stops_total", "stops_proactive", "stops_reactive", "stops_searched"],
        "calls": ["calls", "calls_discretionary"],
        "force": ["force_incidents"],
        "crimes": ["crimes"],
    }
    for src, cols in groups.items():
        covered = panel["month_start"].between(starts[src], trimmed[src])
        panel[f"has_{src}"] = covered
        for col in cols:
            if col in panel.columns:
                panel[col] = panel[col].fillna(0).where(covered)

    b1, b2 = (pd.Timestamp(b) for b in FORCE_REGIME_BREAKS)
    panel["force_regime"] = np.select(
        [panel["month_start"] < b1, panel["month_start"] < b2],
        ["pre_2023", "2023"],
        default="post_2024",
    )

    panel = _add_metrics(panel)
    report = PanelReport(
        neighborhoods=panel["neighborhood"].nunique(),
        months=panel["month_start"].nunique(),
        rows=len(panel),
        window=(months.min(), months.max()),
        trimmed=trimmed,
    )
    return panel, report


def _add_metrics(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy()

    def safe(num: str, den: str) -> pd.Series:
        d = p[den].replace(0, np.nan)
        return p[num] / d

    # Self-normalising: controls for baseline demand without a population
    # denominator, so it is usable before the ACS data lands.
    p["proactive_share"] = safe("stops_proactive", "stops_total")
    p["proactive_per_reactive"] = safe("stops_proactive", "stops_reactive")

    # Does enforcement track measured crime?
    p["stops_per_crime"] = safe("stops_total", "crimes")
    p["proactive_per_crime"] = safe("stops_proactive", "crimes")

    # Enforcement per unit of community-initiated demand.
    p["stops_per_call"] = safe("stops_total", "calls")
    p["search_rate"] = safe("stops_searched", "stops_total")
    p["force_per_stop"] = safe("force_incidents", "stops_total")
    p["discretionary_call_share"] = safe("calls_discretionary", "calls")
    return p


def annual_summary(panel: pd.DataFrame, *, min_crimes: int = 24) -> pd.DataFrame:
    """Collapse to neighbourhood-year with smoothed rates.

    Aggregating counts first, then dividing, avoids averaging monthly ratios
    whose denominators differ -- which would weight a quiet month as heavily as
    a busy one.
    """
    p = panel.copy()
    p["year"] = p["month_start"].dt.year

    # A complete calendar year is not enough: the year must also be fully
    # COVERED by every source used. Crime data begins in 2017-08, so a naive
    # 12-month check keeps 2017 and then sums only five months of offences --
    # inflating stops_per_crime roughly two-and-a-half fold for that year.
    required = ["has_stops", "has_calls", "has_crimes"]
    covered_months = (
        p.assign(_all=p[required].all(axis=1))
        .groupby("year")["_all"]
        .transform(lambda s: s.groupby(p.loc[s.index, "month_start"]).first().sum())
    )
    p = p[(p.groupby("year")["month_start"].transform("nunique").eq(12))
          & (covered_months == 12)]

    agg = (
        p.groupby(["neighborhood", "year"], as_index=False)
        .agg({
            "stops_total": "sum", "stops_proactive": "sum",
            "stops_reactive": "sum", "stops_searched": "sum",
            "calls": "sum", "crimes": "sum", "force_incidents": "sum",
        })
    )
    agg["stops_per_crime"] = np.where(
        agg["crimes"] >= min_crimes, agg["stops_total"] / agg["crimes"], np.nan
    )
    agg["proactive_share"] = np.where(
        agg["stops_total"] > 0, agg["stops_proactive"] / agg["stops_total"], np.nan
    )
    agg["stops_per_crime_eb"] = empirical_bayes_rate(
        agg["stops_total"], agg["crimes"]
    )
    return agg


def load_sources() -> dict[str, pd.DataFrame]:
    P = config.DATA_INTERIM
    return {
        "stops": pd.read_parquet(P / "stops.parquet"),
        "calls": pd.read_parquet(
            P / "calls_911.parquet",
            columns=["ts", "month_start", "neighborhood", "is_police",
                     "is_discretionary"],
        ),
        "crime": pd.read_parquet(P / "crime_stats.parquet"),
        "force": pd.read_parquet(
            P / "use_of_force.parquet", columns=["ts", "month_start", "neighborhood", "incident_id"]
        ),
    }
