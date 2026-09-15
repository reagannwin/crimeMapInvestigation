"""Validate the Tier-1 join key and recover missing stop-initiation labels.

Two findings drive this module.

1. ``Master_Incident_Number`` is a real shared key, but only where a shared
   record should exist. Measured over the overlapping window:

        Citizen / 9-1-1 stops -> 911 extract   94.7%
        Investigative                          19.1%
        Equipment Violation                     2.0%
        Moving Violation                        1.3%

   That gradient is the expected one. ``Incidents_Reported_911`` is a
   911-only extract (``is_911_Call`` is "Yes" on every row), so officer-
   initiated stops are correctly absent from it. A low overall match rate is
   therefore evidence the key works, not evidence it is broken.

2. Because presence-in-911 tracks initiation so sharply, the join doubles as
   an instrument for the 26.9% of stops whose ``reason`` is blank. A blank-
   reason stop that matches a 911 call is almost certainly reactive. This
   recovers a large share of an otherwise unusable field -- but it is an
   imputation, so it is written to a separate column and flagged, never
   merged into the observed one.

Two id shapes exist in the 911 extract and split cleanly by agency:
``YY-NNNNNN`` for POLICE/BCR and ``YY-NNNNNNN`` for FIRE. They do not collide,
so no re-padding is required.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src import config

# Empirical match rates by initiation class, used to set the imputation prior.
REACTIVE_MATCH_RATE = 0.947
PROACTIVE_MATCH_RATE = 0.087


@dataclass(frozen=True)
class KeyReport:
    source: str
    window_rows: int
    matched: int

    @property
    def pct(self) -> float:
        return 100.0 * self.matched / self.window_rows if self.window_rows else 0.0

    def __str__(self) -> str:
        return f"{self.source}: {self.matched:,}/{self.window_rows:,} ({self.pct:.1f}%)"


def normalise_incident_id(series: pd.Series) -> pd.Series:
    """Uppercase, strip, and drop ids that do not match ``YY-NNNNNN(N)``.

    One malformed stop id (``99999999``, no hyphen) exists in the extract.
    """
    s = series.str.strip().str.upper()
    return s.where(s.str.fullmatch(r"\d{2}-\d{6,7}"), pd.NA)


def match_rate_by_initiation(
    stops: pd.DataFrame, calls: pd.DataFrame
) -> pd.DataFrame:
    """Per-initiation-class match rate, restricted to the overlapping window."""
    lo, hi = calls["ts"].min(), calls["ts"].max()
    win = stops[stops["ts"].between(lo, hi)].copy()
    win["matched"] = win["incident_id"].isin(set(calls["incident_id"].dropna()))

    out = (
        win.groupby(win["initiation"].fillna("(blank reason)"), dropna=False)
        .agg(stops=("matched", "size"), matched=("matched", "sum"))
    )
    out["pct"] = (100 * out["matched"] / out["stops"]).round(1)
    return out.reset_index(names="initiation")


def impute_initiation(stops: pd.DataFrame, calls: pd.DataFrame) -> pd.DataFrame:
    """Add ``initiation_filled`` and ``initiation_imputed`` columns.

    Observed labels are never overwritten. Blank-reason stops are labelled
    ``reactive`` when they match a 911 record and ``proactive`` when they do
    not, with ``initiation_imputed`` marking every row where this fired so
    downstream analysis can drop or sensitivity-test them.

    The imputation is confined to the window the 911 extract covers, and that
    guard is essential rather than defensive. "No match" only means "not
    community-initiated" where a match was possible in the first place. The
    extract begins 2020-07, so applying the rule earlier labels *every*
    blank-reason stop proactive: 45,104 stops, 100% of them, with no
    possibility of the other answer. Outside the window the label stays NA.
    """
    out = stops.copy()
    call_ids = set(calls["incident_id"].dropna())
    lo, hi = calls["ts"].min(), calls["ts"].max()

    in_window = out["ts"].between(lo, hi)
    blank = out["initiation"].isna()
    matched = out["incident_id"].isin(call_ids)
    impute = blank & in_window

    out["initiation_filled"] = out["initiation"].astype("string")
    out.loc[impute & matched, "initiation_filled"] = "reactive"
    out.loc[impute & ~matched, "initiation_filled"] = "proactive"
    out["initiation_imputed"] = impute
    out["initiation_unresolved"] = blank & ~in_window

    return out


def run(verbose: bool = True) -> dict[str, object]:
    """Load interim Parquet, validate keys, and report."""
    P = config.DATA_INTERIM
    calls = pd.read_parquet(P / "calls_911.parquet", columns=["incident_id", "ts", "agency"])
    stops = pd.read_parquet(
        P / "stops.parquet", columns=["incident_id", "ts", "initiation", "reason"]
    )
    uof = pd.read_parquet(P / "use_of_force.parquet", columns=["incident_id", "ts"])

    for df in (calls, stops, uof):
        df["incident_id"] = normalise_incident_id(df["incident_id"])

    rates = match_rate_by_initiation(stops, calls)
    filled = impute_initiation(stops, calls)

    if verbose:
        print("=== stop -> 911 match rate by initiation ===")
        print(rates.to_string(index=False))
        print("\n=== initiation coverage before / after imputation ===")
        before = stops["initiation"].notna().mean()
        after = filled["initiation_filled"].notna().mean()
        print(f"  observed: {100 * before:.1f}%   after imputation: {100 * after:.1f}%")
        print(f"  imputed rows:    {int(filled['initiation_imputed'].sum()):,}")
        print(f"  left unresolved: {int(filled['initiation_unresolved'].sum()):,} "
              "(blank reason, outside the 911 window)")

    return {"rates": rates, "stops_filled": filled}


if __name__ == "__main__":
    run()
