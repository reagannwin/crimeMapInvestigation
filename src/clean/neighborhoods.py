"""Canonicalise Minneapolis neighbourhood names across sources.

Every source spells neighbourhoods differently. Observed in the delivered data:

  * NEIGHBORHOOD_CRIME_STATS.csv carries 177 distinct values for 87 real
    neighbourhoods: proper-case names (2018+) AND an ALL-CAPS vocabulary
    used in 2017-2018. The two spellings OVERLAP during 2018, so grouping on
    the raw string silently splits that year's counts across two variants.
  * Three neighbourhoods were renamed and appear under both names:
        CARAG        -> South Uptown
        ECCO         -> East Bde Maka Ska
        West Calhoun -> West Maka Ska
  * Punctuation drifts: "Steven's Square" vs "STEVENS SQUARE",
    "St. Anthony East" vs "ST. ANTHONY EAST".
  * Unassigned sentinels appear as "** NOT ASSIGNED **" and
    "Z_** NOT ASSIGNED **".

`canonicalise` maps any of these onto the 87 authoritative BDNAME values from
the neighbourhood polygon layer, and raises on anything it does not recognise
so a new spelling fails loudly instead of quietly dropping rows.
"""
from __future__ import annotations

import csv
import re
from functools import lru_cache

import pandas as pd

from src import config

# Sentinels meaning "no neighbourhood assigned". Mapped to NA, never to a name.
UNASSIGNED = frozenset({"** NOT ASSIGNED **", "Z_** NOT ASSIGNED **", ""})

# Renamed neighbourhoods, keyed by normalised form.
RENAMES = {
    "CARAG": "South Uptown",
    "ECCO": "East Bde Maka Ska",
    "WEST CALHOUN": "West Maka Ska",
    # Defensive: pre-rename spellings that may appear in older extracts.
    "EAST CALHOUN": "East Bde Maka Ska",
    "CALHOUN AREA RESIDENTS ACTION GROUP": "South Uptown",
}

_PUNCT = re.compile(r"[.'’]")
_SPACE = re.compile(r"\s+")
_HYPHEN = re.compile(r"\s*-\s*")


def normalise_key(name: str | None) -> str:
    """Reduce a neighbourhood string to a comparison key.

    Upper-cases, strips ``.'’``, collapses whitespace, and normalises spacing
    around hyphens so ``Bryn - Mawr``, ``BRYN-MAWR`` and ``Bryn  -  Mawr``
    all converge.
    """
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    s = str(name).strip()
    if not s:
        return ""
    s = _PUNCT.sub("", s.upper())
    s = _HYPHEN.sub(" - ", s)
    return _SPACE.sub(" ", s).strip()


@lru_cache(maxsize=1)
def canonical_names() -> tuple[str, ...]:
    """The 87 authoritative BDNAME values, read from the polygon attribute table."""
    path = config.SOURCE_DIR / config.SOURCE_FILES["neighborhoods"]
    with path.open(encoding="utf-8-sig", newline="") as fh:
        names = {r["BDNAME"].strip() for r in csv.DictReader(fh) if r["BDNAME"].strip()}
    if len(names) != 87:
        raise ValueError(f"expected 87 canonical neighbourhoods, found {len(names)}")
    return tuple(sorted(names))


@lru_cache(maxsize=1)
def _lookup() -> dict[str, str]:
    """Normalised key -> canonical name, including rename aliases."""
    table = {normalise_key(n): n for n in canonical_names()}
    for alias, target in RENAMES.items():
        key = normalise_key(alias)
        if target not in table.values():
            raise ValueError(f"rename target {target!r} is not a canonical name")
        table[key] = target
    return table


def canonicalise(
    values: pd.Series,
    *,
    strict: bool = True,
) -> pd.Series:
    """Map a Series of raw neighbourhood strings onto canonical names.

    Unassigned sentinels and blanks become ``pd.NA``. With ``strict=True`` an
    unrecognised spelling raises; with ``strict=False`` it becomes ``pd.NA``.
    """
    table = _lookup()
    unassigned_keys = {normalise_key(u) for u in UNASSIGNED}

    # Materialise NA as "" before mapping: Series.map propagates NA rather than
    # passing it to the callable, which would otherwise make every missing
    # neighbourhood look like an unrecognised spelling.
    keys = values.fillna("").astype(str).map(normalise_key)
    out = keys.map(table)

    unmapped = out.isna() & ~keys.isin(unassigned_keys)
    if unmapped.any():
        offenders = sorted(values[unmapped].dropna().astype(str).unique())
        if strict:
            raise ValueError(
                f"{int(unmapped.sum())} rows with {len(offenders)} unrecognised "
                f"neighbourhood spelling(s): {offenders[:10]}"
            )

    return out.astype("string")


def coverage_report(values: pd.Series) -> pd.DataFrame:
    """Per-raw-spelling diagnostic: what it maps to and how many rows."""
    table = _lookup()
    unassigned_keys = {normalise_key(u) for u in UNASSIGNED}
    raw = values.fillna("").astype(str)
    keys = raw.map(normalise_key)  # raw already has NA materialised as ""

    def classify(k: str) -> str:
        if k in unassigned_keys:
            return "unassigned"
        if k in table:
            return "mapped"
        return "UNRECOGNISED"

    return (
        pd.DataFrame({"raw": raw, "key": keys})
        .assign(
            canonical=lambda d: d["key"].map(table),
            status=lambda d: d["key"].map(classify),
        )
        .groupby(["raw", "canonical", "status"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values("rows", ascending=False, ignore_index=True)
    )
