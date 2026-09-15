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
| 5a — Gunfire event de-duplication | done |
| 5b — Search hit-rate (outcome) test | done |
| 5c — Enforcement panel | done |
| 5d — Per-capita rates | blocked on ACS demographics (needs a Census API key) |
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

**Gunfire records are not gunfire events.** Every row in `Shots_Fired` carries
a unique incident number, so one burst that trips a sensor and draws three 911
calls appears as four records. Counting rows counts *reporting*, not gunfire —
which inverts the signal. `src/join/events.py` clusters records in space and
time: 108,800 records collapse to 95,826 events.

**Sensor-to-human linkage needs a wider radius than de-duplication.** Widening
the time window 8x moves co-detection only 6.0% -> 6.6%, while widening the
radius 150 m -> 400 m moves it 6.0% -> 11.3%. Space is the binding constraint,
because ShotSpotter reports an estimated acoustic position while a caller
reports their own address. Hence two radii in config: 150 m for dedup, 400 m
for cross-source linkage.

**The matches are real, not coincidence.** A permutation test shifting every
human report by a random offset gives a null co-detection rate of 0.22%
against an observed 11.33% — about 98% of matches are genuine.

**11.4% of sensor-detected gunfire draws a 911 call, and the rate varies 4x
across covered neighbourhoods** — from 6.0% in Lind - Bohanon to 23.9% in
Powderhorn Park. This is the project's core signal: the same objective
denominator, the same sensor footprint, very different propensity to call.

**`preRace` is unusable despite being the theoretically correct variable.**
Perceived race is 68–74% `Unknown` from 2018–2022. The 2024 recording change
cut that to 7.6% but moved the mass into `Other` (41.6%), which is not a group.
Recorded `race` is far better — `Unknown` falls to 0.0% from 2024 — so the hit-
rate work uses recorded race and reports perceived race only as sensitivity.

**Black drivers are searched ~2.4x as often and those searches yield arrests
less often.** 2024–2026: search rate 16.7% vs 7.6%, arrest-given-search 44.2%
vs 51.7% (−7.5 pp, p=0.003). Under the outcome test a lower yield from more
searching implies a lower evidentiary bar. The sign holds in **all 24**
arrest-based specifications (−5.3 to −11.8 pp).

**The deficit is confined to discretionary stops.** Stratifying by initiation:
−9.4 pp in officer-initiated stops, **+3.6 pp in 911-dispatched stops**. When a
call determines the stop, the gap reverses. That isolates discretion as the
mechanism — a difference in underlying offending would move both alike. It is
also why the pooled number alone should not be quoted.

**Citation-based hit rates do not replicate the arrest-based result**
(median +1.1 pp). A traffic citation is not evidence a search succeeded, so
booking is the defensible measure, but the divergence is reported rather than
buried.

**Enforcement rose sharply while crime fell and demand stayed flat.** Citywide,
2021 to 2025:

| | change |
|---|---:|
| reported crime | **−5.4%** |
| 911 calls (police) | +1.7% |
| stops, total | +36.2% |
| **stops, proactive** | **+82.3%** |
| stops, reactive | −14.0% |
| **stops per crime** | **+44.0%** |

Officer-initiated stops rose 82% while dispatched stops fell 14%, so the growth
is discretionary rather than demand-driven. `proactive_per_reactive` went 0.60
to 1.28.

**That holds under both initiation definitions.** 23-31% of stops carry a blank
`reason`; recovering them via the 911 join (`--imputed-initiation`) strengthens
the result rather than weakening it — proactive **+105.8%** against +82.3%
observed, and the proactive share reaches **58.9%** of all stops by 2025.

| 2021→2025 | observed | imputed |
|---|---:|---:|
| stops, total | +36.2% | +36.2% |
| stops, proactive | +82.3% | **+105.8%** |
| stops, reactive | −14.0% | −8.3% |
| proactive per reactive | +111.9% | +124.5% |

**The imputation is confined to the 911 window.** "No match" only means "not
community-initiated" where a match was possible. The extract begins 2020-07, so
applying the rule earlier labelled **45,104 stops proactive — 100% of them**,
with no possibility of the other answer. Outside the window the label stays NA.

**Use of force has two recording breaks and must not be compared across them.**
Monthly incidents fall 499 (Dec 2022) to 153 (Jan 2023) overnight — a level
break — and rows per incident jump 4.3 to 28.8 in 2024, a granularity change.
Incident counts survive the second break; row counts do not. Flagged as
`force_regime`.

**Partial months are trimmed.** Every extract ends mid-month and a half month
reads as a collapse (stops 2026-06: 1,037 against ~2,400; force 2026-09: 154
against ~8,200). Left in, each becomes a fake downward trend.

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
    events.py            spatio-temporal clustering, chaining + null-model checks
    build_events.py      Phase 5a runner
  features/
    panel.py             neighbourhood x month panel, EB smoothing, trimming
    build_panel.py       Phase 5c runner
    hit_rate.py          outcome test: Wilson CIs, two-proportion tests, strata
    build_hit_rate.py    Phase 5b runner
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
