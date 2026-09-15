# crimeMapInvestigation

Spatial-temporal analysis of Minneapolis policing: community-initiated 911 calls
and gunfire reports layered against officer-initiated enforcement, to test
whether police activity tracks measured crime or diverges from it.

## Status

| Phase | State |
|---|---|
| 0 — Scaffold, config, environment | done |
| 1 — Ingest all sources to Parquet | done |
| 2 — Clean: names, coordinates, timezones | done |
| 3 — Joins: key validation + initiation recovery | done |
| 4 — ShotSpotter coverage mask | done |
| 5 — Feature panel | blocked on ACS demographics (needs a Census API key) |
| 6 — Bias diagnostics | not started |
| 7 — Models and maps | not started |

## Quickstart

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m src.ingest.build_interim
.venv/Scripts/python.exe -m pytest tests/ -q
```

`build_interim` reads the CSVs from `~/Downloads` (see `src/config.py`) and
writes typed Parquet to `data/interim/`. Re-parsing the 344 MB 911 extract takes
about 40 seconds; reading the Parquet takes about one.

## Data sources

| Source | Rows | Coverage | Role |
|---|---:|---|---|
| `Incidents_Reported_911.csv` | 1,643,836 | 2020-07 → 2026-07 | Community-initiated demand |
| `Police_Stop_Data.csv` | 298,936 | 2016-10 → 2026-06 | Officer-initiated enforcement |
| `Police_Use_of_Force.csv` | 256,489 | 2021-12 → 2026-09 | Severity outcome |
| `NEIGHBORHOOD_CRIME_STATS.csv` | 74,624 | 2017 → 2026 | Crime denominator (pre-aggregated) |
| `Police_Incidents_2026.csv` | 16,309 | 2026 only | Incident-level detail |
| `Minneapolis_Neighborhoods…csv` | 87 | — | Canonical neighbourhood names |
| `Shots_Fired` (ArcGIS) | 108,800 | 2007 → 2026 | Gunfire: sensor + human |
| `Minneapolis_Neighborhoods` (ArcGIS) | 87 | — | Polygon geometry, EPSG:26915 |
| ACS 5-year block groups | 428 | 2019–2023 | Demographic denominators |

**ACS requires a free Census API key.** Unkeyed requests now 302 to a
"Missing Key" page. Get one at https://api.census.gov/data/key_signup.html and
set `CENSUS_API_KEY`, then run `python -m src.ingest.build_census`. The
block-group geometry half needs no key and is verified working.

## Findings that constrain the analysis

**`Shots_Fired` is not objective sensor data.** Only 37,896 of its 108,800 rows
are `ShotSpotter Activation`; 63,254 are `Sound of Shots Fired`, i.e. human 911
calls. Treating the layer as a whole as "objective" correlates community
reporting with itself. The sensor/human split is the analysis, not a nuisance.

**`Incidents_Reported_911` contains only community-initiated activity.**
`is_911_Call` is `"Yes"` on all 1,643,836 rows. Proactive policing cannot be
measured from this file; it requires `Police_Stop_Data.reason`.

**The join key is real, and its failures are informative.** Stops match into the
911 extract at 94.7% when `reason == "Citizen / 9-1-1"` and 1.3% for
`Moving Violation`. That gradient confirms both the key and the point above.

**The 911 join recovers the missing `reason` field.** 26.9% of stops have a
blank `reason`. Because presence-in-911 tracks initiation so sharply, matching
recovers a label for all 80,307 of them — written to `initiation_filled` and
flagged by `initiation_imputed`, never merged into the observed column.

**Two regime breaks in the stop data.** 2016–2017 have 100%/64% blank
`citationIssued` and are excluded via `usable_window`. Separately,
`preRace == "Unknown"` falls from 38.7% (2023) to 3.9% (2024) — a recording
system change, not a behaviour change. Perceived-race series must not cross it.

**Neighbourhood names need canonicalising.** `NEIGHBORHOOD_CRIME_STATS` carries
177 spellings for 87 neighbourhoods: ALL-CAPS in 2017–2018, proper case from
2018, overlapping during 2018. Grouping on the raw string silently splits that
year. Three neighbourhoods were also renamed (CARAG → South Uptown, ECCO → East
Bde Maka Ska, West Calhoun → West Maka Ska).

**Missing geocodes are encoded as `0.0, 0.0`.** Valid floats that survive every
spatial join. Nulled at ingest: 1,582 rows in 911, 2,368 in stops, 2,344 in
use of force.

**The ShotSpotter footprint covers 21.9% of the city.** Derived from the
activation points themselves: 20 of 87 neighbourhoods qualify as covered,
holding 97.6% of all activations. Only those 20 are valid for sensor-versus-
human comparison. Across 27 parameter settings the excluded set stays at 64–67
neighbourhoods, which is the stability that licenses using one setting.

**Geometric coverage alone over-promotes neighbourhoods.** Morphological
closing connects sparse activations, so a neighbourhood with almost no sensors
can read as covered — Downtown West reaches 22% coverage on 2 activations in
nineteen years. `coverage_by_neighborhood` therefore cross-checks against a
direct activation count and downgrades anything below 50, flagged in
`evidence_downgraded`.

## Layout

```
src/
  config.py              CRS, windows, thresholds, taxonomies — no magic numbers elsewhere
  clean/
    neighborhoods.py     87 canonical names; strict by default so new spellings fail loudly
    coordinates.py       null-island handling, bbox validation, projection to EPSG:26915
  ingest/
    csv_sources.py       one typed loader per source, each returning a LoadResult
    build_interim.py     Phase 1  CSV -> Parquet, run once
    arcgis.py            paginated Feature Service client
    arcgis_sources.py    Shots_Fired + neighbourhood polygons
    build_geo.py         Phase 1b ArcGIS -> GeoParquet
    census.py            ACS pull + areal interpolation to neighbourhoods
    build_census.py      Phase 1c runner (needs CENSUS_API_KEY)
  join/
    validate_keys.py     Phase 3  key validation + initiation recovery
  bias/
    sensor_footprint.py  coverage mask, evidence guard, parameter sweep
    build_footprint.py   Phase 4 runner
```

## Conventions

- Analysis CRS is **EPSG:26915** (UTM 15N, metres). Distance work in degrees
  distorts by ~30% at this latitude. `X`/`Y` columns in the city exports are Web
  Mercator, not lat/long — use the named lat/long columns.
- City timestamps are **UTC**; loaders convert to `America/Chicago`.
- Imputed values always live in a separate column beside an `_imputed` flag.
