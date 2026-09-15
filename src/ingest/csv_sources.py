"""Load the delivered city CSV extracts into validated, typed frames.

Each loader is responsible for exactly three things: correct parsing, correct
types, and an honest report of what was lost. Nothing is silently dropped.

Parsing notes drawn from profiling the delivered files:

  * All five extracts are UTF-8 with a BOM -> ``encoding="utf-8-sig"``.
  * All parse cleanly with a real CSV reader (0 ragged rows). Naive
    ``split(",")`` does NOT work: ``Call_Disposition`` is multi-valued and
    1,491 rows carry an embedded comma inside quotes, e.g.
    ``"CNL-Cancel,INF-Information"``.
  * Timestamps are ``YYYY/MM/DD HH:MM:SS+00`` and are UTC, not local. The city
    documents this explicitly. Converting to America/Chicago matters for any
    hour-of-day or veil-of-darkness analysis.
  * ``X``/``Y`` columns are Web Mercator (EPSG:3857), NOT longitude/latitude.
    Use the named lat/long columns instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src import config
from src.clean import neighborhoods as nb
from src.clean.coordinates import CoordReport, clean_latlon

LOCAL_TZ = "America/Chicago"
TS_FORMAT = "%Y/%m/%d %H:%M:%S%z"


@dataclass
class LoadResult:
    """A loaded frame plus everything a reviewer needs to trust it."""

    name: str
    frame: pd.DataFrame
    coords: CoordReport | None = None
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"{self.name}: {len(self.frame):,} rows x {self.frame.shape[1]} cols"]
        if self.coords:
            lines.append(f"   coords: {self.coords}")
        lines.extend(f"   note:   {n}" for n in self.notes)
        return "\n".join(lines)


def _src(key: str) -> Path:
    path = config.SOURCE_DIR / config.SOURCE_FILES[key]
    if not path.exists():
        raise FileNotFoundError(f"missing source file for {key!r}: {path}")
    return path


def _to_local(series: pd.Series) -> pd.Series:
    """Parse a UTC city timestamp and convert to Minneapolis local time."""
    parsed = pd.to_datetime(series, format=TS_FORMAT, utc=True, errors="coerce")
    return parsed.dt.tz_convert(LOCAL_TZ)


def _add_time_parts(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    ts = df[ts_col]
    return df.assign(
        year=ts.dt.year.astype("Int16"),
        month=ts.dt.month.astype("Int8"),
        month_start=ts.dt.tz_localize(None).dt.to_period("M").dt.to_timestamp(),
        hour=ts.dt.hour.astype("Int8"),
        dow=ts.dt.dayofweek.astype("Int8"),
    )


def _yes_no(series: pd.Series) -> pd.Series:
    return series.str.strip().str.upper().map({"YES": True, "NO": False}).astype("boolean")


# --------------------------------------------------------------------------
# 911 calls for service
# --------------------------------------------------------------------------
def load_calls_911(*, strict_names: bool = True) -> LoadResult:
    """1.64M CAD calls, 2020-07-24 onward.

    ``is_911_Call`` is ``"Yes"`` on every row, so this extract cannot
    distinguish community-initiated from officer-initiated activity. Use
    ``load_stops`` for that. The column is dropped rather than carried as a
    constant that invites misuse.
    """
    usecols = [
        "Master_Incident_Number", "Agency_Type", "Initial_Problem",
        "Problem__Final_", "Initial_Priority", "Final_Priority", "Division",
        "Jurisdiction", "Response_Date", "Call_Disposition", "Latitude",
        "Longitude", "Ward__GIS_", "Neighborhood__GIS_",
    ]
    df = pd.read_csv(
        _src("calls_911"),
        encoding="utf-8-sig",
        usecols=usecols,
        dtype={
            "Master_Incident_Number": "string",
            "Agency_Type": "category",
            "Initial_Problem": "category",
            "Problem__Final_": "category",
            "Division": "category",
            "Jurisdiction": "category",
            "Call_Disposition": "string",
            "Neighborhood__GIS_": "string",
            "Ward__GIS_": "string",
        },
    )
    df = df.rename(columns={
        "Master_Incident_Number": "incident_id",
        "Agency_Type": "agency",
        "Initial_Problem": "problem_initial",
        "Problem__Final_": "problem_final",
        "Initial_Priority": "priority_initial",
        "Final_Priority": "priority_final",
        "Response_Date": "ts",
        "Call_Disposition": "disposition_raw",
        "Ward__GIS_": "ward",
        "Neighborhood__GIS_": "neighborhood_raw",
    })

    df["ts"] = _to_local(df["ts"])
    df, coords = clean_latlon(df, "Latitude", "Longitude")
    df = df.drop(columns=["Latitude", "Longitude"])
    df["neighborhood"] = nb.canonicalise(df["neighborhood_raw"], strict=strict_names)
    df["incident_id"] = df["incident_id"].str.strip().str.upper()

    # Disposition is multi-valued; keep the raw string and a list for exploding.
    df["disposition"] = (
        df["disposition_raw"].fillna("").str.split(",").map(
            lambda xs: [x.strip() for x in xs if x.strip()]
        )
    )
    multi = int((df["disposition"].str.len() > 1).sum())

    df = _add_time_parts(df, "ts")
    df["is_police"] = df["agency"].eq("POLICE")
    df["is_discretionary"] = df["problem_initial"].isin(config.DISCRETIONARY_CALL_TYPES)
    # Initial != final label: a proxy for how often the reported concern is
    # not what responders actually found.
    df["reclassified"] = (
        df["problem_initial"].astype("string") != df["problem_final"].astype("string")
    )

    return LoadResult(
        "calls_911", df, coords,
        notes=[
            "is_911_Call dropped: constant 'Yes' on all rows, carries no signal",
            f"{multi:,} rows have a multi-valued Call_Disposition",
            "timestamps converted UTC -> America/Chicago",
        ],
    )


# --------------------------------------------------------------------------
# Police stops
# --------------------------------------------------------------------------
def load_stops(*, strict_names: bool = True) -> LoadResult:
    """298,936 stops, 2016-10-31 onward. The only source of officer-initiated activity.

    Two regime breaks are flagged as columns rather than silently applied:
      * ``usable_window``  - 2016/2017 have 100%/64% blank ``citationIssued``.
      * ``prerace_regime`` - ``preRace == "Unknown"`` falls 38.7% -> 3.9%
        between 2023 and 2024, a recording-system change. Perceived-race
        series must not be compared across it.
    """
    df = pd.read_csv(
        _src("stops"),
        encoding="utf-8-sig",
        dtype={
            "masterIncidentNumber": "string",
            "reason": "string",
            "problem": "category",
            "callDisposition": "string",
            "citationIssued": "string",
            "personSearch": "string",
            "vehicleSearch": "string",
            "preRace": "string",
            "race": "string",
            "gender": "string",
            "policePrecinct": "string",
            "neighborhood": "string",
        },
    ).rename(columns={
        "masterIncidentNumber": "incident_id",
        "responseDate": "ts",
        "callDisposition": "disposition",
        "citationIssued": "citation",
        "personSearch": "search_person",
        "vehicleSearch": "search_vehicle",
        "preRace": "race_perceived",
        "race": "race_recorded",
        "policePrecinct": "precinct",
        "neighborhood": "neighborhood_raw",
        "lat": "lat_src",
        "long": "lon_src",
    })

    df["ts"] = _to_local(df["ts"])
    df, coords = clean_latlon(df, "lat_src", "lon_src")
    df = df.drop(columns=["lat_src", "lon_src", "x", "y"], errors="ignore")
    df["neighborhood"] = nb.canonicalise(df["neighborhood_raw"], strict=strict_names)
    df["incident_id"] = df["incident_id"].str.strip().str.upper()
    df = _add_time_parts(df, "ts")

    df["initiation"] = (
        df["reason"].str.strip().map(config.STOP_REASON_CLASS).astype("string")
    )
    df["is_proactive"] = df["initiation"].eq("proactive")

    for col in ("citation", "search_person", "search_vehicle"):
        df[col] = _yes_no(df[col])

    # "NO CONTACT" / "UNOCCUPIED VEHICLE" are not people and must be excluded
    # from race denominators rather than folded into "Unknown".
    df["person_perceived"] = (
        ~df["race_perceived"].isin(config.NON_PERSON_RACE) & df["race_perceived"].notna()
    )
    df["person_recorded"] = (
        ~df["race_recorded"].isin(config.NON_PERSON_RACE) & df["race_recorded"].notna()
    )

    usable_from = pd.Timestamp(config.STOPS_USABLE_START, tz=LOCAL_TZ)
    regime_break = pd.Timestamp(config.PRERACE_REGIME_BREAK, tz=LOCAL_TZ)
    df["usable_window"] = df["ts"] >= usable_from
    df["prerace_regime"] = pd.Series(
        ["post_2024" if x else "pre_2024" for x in (df["ts"] >= regime_break)],
        index=df.index, dtype="string",
    ).where(df["ts"].notna())

    blank_reason = int((df["reason"].fillna("").str.strip() == "").sum())
    return LoadResult(
        "stops", df, coords,
        notes=[
            f"{blank_reason:,} rows ({100 * blank_reason / len(df):.1f}%) have blank "
            "`reason` -> initiation unknown; rate rises 16.9% (2018) to 31.2% (2025)",
            "pre-2018 flagged via `usable_window` (2016 reason/citation 100% blank)",
            "`prerace_regime` marks the 2023/2024 preRace recording break",
        ],
    )


# --------------------------------------------------------------------------
# Use of force
# --------------------------------------------------------------------------
def load_use_of_force(*, strict_names: bool = True) -> LoadResult:
    """256,489 force records. Joins to calls/stops on case number."""
    df = pd.read_csv(
        _src("use_of_force"),
        encoding="utf-8-sig",
        dtype={
            "caseNumber": "string",
            "problem": "category",
            "force": "category",
            "forceType": "category",
            "sex": "string",
            "race": "string",
            "ethnicity": "string",
            "precinct": "string",
            "ward": "string",
            "neighborhood": "string",
            "address": "string",
        },
    ).rename(columns={
        "caseNumber": "incident_id",
        "reportedDate": "ts",
        "race": "race_recorded",
        "forceType": "force_type",
        "eventAge": "age",
        "neighborhood": "neighborhood_raw",
        "centerLatitude": "lat_src",
        "centerLongitude": "lon_src",
    })

    df["ts"] = _to_local(df["ts"])
    df, coords = clean_latlon(df, "lat_src", "lon_src")
    df = df.drop(columns=["lat_src", "lon_src", "X", "Y"], errors="ignore")
    df["neighborhood"] = nb.canonicalise(df["neighborhood_raw"], strict=strict_names)
    df["incident_id"] = df["incident_id"].str.strip().str.upper()
    df = _add_time_parts(df, "ts")

    return LoadResult(
        "use_of_force", df, coords,
        notes=[
            "one row per force action, so a single incident appears repeatedly; "
            "de-duplicate on (incident_id, useOfForceID) before counting people",
        ],
    )


# --------------------------------------------------------------------------
# Neighbourhood crime statistics (pre-aggregated panel)
# --------------------------------------------------------------------------
def load_crime_stats(*, strict_names: bool = True) -> LoadResult:
    """Balanced neighbourhood x month x UCR-offence panel, 2017-2026.

    This is the crime denominator, and it arrives pre-aggregated, which removes
    the need to stitch together ten annual Police_Incidents services.

    The name vocabulary changes mid-panel: ALL-CAPS in 2017-2018, proper case
    from 2018 on, with an overlap during 2018. Grouping on the raw string
    splits that year across two spellings; ``canonicalise`` collapses them.
    """
    df = pd.read_csv(
        _src("crime_stats"),
        encoding="utf-8-sig",
        dtype={
            "neighborhoodCrimeStatisticsID": "int64",
            "neighborhood": "string",
            "ucrDescription": "category",
            # 3,290 of 74,624 `number` cells (4.4%) are blank. Nullable Int32,
            # never plain int32: a blank is not the same fact as a zero.
            "number": "Int32",
            "reportMonth": "int8",
            "reportYear": "int16",
        },
    ).rename(columns={
        "neighborhoodCrimeStatisticsID": "stat_id",
        "neighborhood": "neighborhood_raw",
        "ucrDescription": "offense",
        "number": "count",
        "reportMonth": "month",
        "reportYear": "year",
    })

    df["neighborhood"] = nb.canonicalise(df["neighborhood_raw"], strict=strict_names)
    df["month_start"] = pd.to_datetime(
        dict(year=df["year"].astype(int), month=df["month"].astype(int), day=1)
    )

    # A blank `number` in a balanced panel almost certainly means "no offences
    # that month", but "almost certainly" is not a basis for silently writing
    # zeros. Keep `count` nullable, expose the zero-filled variant explicitly,
    # and let the feature layer choose which it wants.
    df["count_missing"] = df["count"].isna()
    df["count_zerofilled"] = df["count"].fillna(0).astype("int32")

    raw_variants = df["neighborhood_raw"].nunique()
    unassigned_rows = int(df["neighborhood"].isna().sum())
    unassigned_offences = int(
        df.loc[df["neighborhood"].isna(), "count_zerofilled"].sum()
    )
    blank_counts = int(df["count_missing"].sum())

    return LoadResult(
        "crime_stats", df,
        notes=[
            f"{raw_variants} raw name variants collapsed to "
            f"{df['neighborhood'].nunique()} canonical neighbourhoods",
            f"{unassigned_rows:,} rows are NOT ASSIGNED "
            f"({unassigned_offences:,} offences with no neighbourhood)",
            f"{blank_counts:,} rows ({100 * blank_counts / len(df):.1f}%) have a blank "
            "`number`; kept NA in `count`, zeros in `count_zerofilled`",
            "2017 and 2026 are partial years (5 months each)",
        ],
    )


def load_incidents_2026(*, strict_names: bool = True) -> LoadResult:
    """16,309 incident-level offences for 2026 only. Use crime_stats for series."""
    df = pd.read_csv(
        _src("incidents_2026"),
        encoding="utf-8-sig",
        dtype={
            "caseNumber": "string",
            "precinct": "string",
            "offense": "category",
            "description": "category",
            "UCRCode": "string",
            "neighborhood": "string",
            "publicaddress": "string",
        },
    ).rename(columns={
        "caseNumber": "incident_id",
        "reportedDateTime": "ts",
        "neighborhood": "neighborhood_raw",
        "centerLat": "lat_src",
        "centerLong": "lon_src",
    })

    df["ts"] = _to_local(df["ts"])
    df, coords = clean_latlon(df, "lat_src", "lon_src")
    df = df.drop(
        columns=["lat_src", "lon_src", "X", "Y", "centerX", "centerY"], errors="ignore"
    )
    df["neighborhood"] = nb.canonicalise(df["neighborhood_raw"], strict=strict_names)
    df["incident_id"] = df["incident_id"].str.strip().str.upper()
    df = _add_time_parts(df, "ts")

    return LoadResult(
        "incidents_2026", df, coords,
        notes=["partial year; not comparable to full-year counts"],
    )


LOADERS = {
    "calls_911": load_calls_911,
    "stops": load_stops,
    "use_of_force": load_use_of_force,
    "crime_stats": load_crime_stats,
    "incidents_2026": load_incidents_2026,
}
