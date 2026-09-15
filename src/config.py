"""Central configuration. No magic numbers anywhere else in the codebase."""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# Source CSVs as delivered (Downloads). Copied into data/raw on first ingest.
SOURCE_DIR = Path.home() / "Downloads"

SOURCE_FILES = {
    "calls_911": "Incidents_Reported_911.csv",
    "stops": "Police_Stop_Data.csv",
    "use_of_force": "Police_Use_of_Force.csv",
    "incidents_2026": "Police_Incidents_2026.csv",
    "crime_stats": "NEIGHBORHOOD_CRIME_STATS.csv",
    "neighborhoods": "Minneapolis_Neighborhoods_-3137758889740763984.csv",
}

# --------------------------------------------------------------------------
# Coordinate reference systems
# --------------------------------------------------------------------------
# Storage/interchange CRS (lat/long as published).
CRS_GEOGRAPHIC = "EPSG:4326"
# Analysis CRS: UTM 15N, metres. ALL distance/area/buffer work happens here.
# Doing radius math in degrees distorts distance ~30% at this latitude.
CRS_PROJECTED = "EPSG:26915"
# The X/Y columns in the city exports are Web Mercator, NOT lat/long.
CRS_WEB_MERCATOR = "EPSG:3857"

# Minneapolis bounding box (geographic) used to reject bad geocodes.
BBOX_LAT = (44.88, 45.06)
BBOX_LON = (-93.34, -93.19)

# --------------------------------------------------------------------------
# Null-coordinate sentinel
# --------------------------------------------------------------------------
# The city encodes missing geocodes as a literal 0.0 / 0.0 rather than NULL.
# These are valid floats and will silently survive every spatial join,
# landing in the Gulf of Guinea. They must be nulled at ingest.
NULL_ISLAND_TOL = 1e-6

# --------------------------------------------------------------------------
# Analysis windows
# --------------------------------------------------------------------------
# 911 CSV export begins 2020-07-24; stops begin 2016-10-31; crime stats 2017.
ANALYSIS_START = "2020-07-01"

# Stop data before 2018 is structurally unusable:
#   2016 -> `reason` and `citationIssued` 100% blank
#   2017 -> `reason` 36% blank, `citationIssued` 64% blank
STOPS_USABLE_START = "2018-01-01"

# `preRace == "Unknown"` collapses from 38.7% (2023) to 3.9% (2024): a data
# system change, not a behaviour change. Perceived-race series must not be
# compared across this boundary.
PRERACE_REGIME_BREAK = "2024-01-01"

# --------------------------------------------------------------------------
# Spatio-temporal matching parameters (swept, never assumed)
# --------------------------------------------------------------------------
MATCH_RADII_M = (150, 300, 500)
# Asymmetric: a human dials AFTER the sensor triggers.
MATCH_WINDOW_MIN = (-5, 15)
MATCH_WINDOW_SWEEP = ((-5, 5), (-5, 15), (-10, 30))

# Gunfire-event de-duplication (DBSCAN over projected coords + time).
EVENT_CLUSTER_RADIUS_M = 150
EVENT_CLUSTER_WINDOW_MIN = 10

# --------------------------------------------------------------------------
# Small-number handling
# --------------------------------------------------------------------------
MIN_CELL_EVENTS = 20  # suppress rates computed on fewer events than this

# --------------------------------------------------------------------------
# ArcGIS REST (for layers not supplied as CSV, e.g. polygon geometry)
# --------------------------------------------------------------------------
ARCGIS_ORG = "https://services.arcgis.com/afSMGVsC7QlRK1kZ/arcgis/rest/services"
ARCGIS_PAGE_SIZE = 2000  # server-side cap

LAYERS = {
    "shots_fired": f"{ARCGIS_ORG}/Shots_Fired/FeatureServer/0",
    "neighborhoods": f"{ARCGIS_ORG}/Minneapolis_Neighborhoods/FeatureServer/0",
    "police_sectors": f"{ARCGIS_ORG}/Police_Sectors/FeatureServer/0",
}

# --------------------------------------------------------------------------
# Call-type taxonomy
# --------------------------------------------------------------------------
# Shots_Fired.Problem mixes sensor detections with human 911 calls. Only
# SENSOR_PROBLEMS is machine-generated; treating the whole layer as
# "objective" correlates community reporting with itself.
SENSOR_PROBLEMS = frozenset({"ShotSpotter Activation (P)"})
HUMAN_SHOTS_PROBLEMS = frozenset({
    "Sound of Shots Fired (P)",
    "Shooting Report Only (P)",
})
CONFIRMED_SHOOTING_PROBLEMS = frozenset({"Shooting (PFE)"})

# High-discretion call types: officer/caller judgement drives the label.
DISCRETIONARY_CALL_TYPES = frozenset({
    "Suspicious Person (P)",
    "Suspicious Vehicle (P)",
    "Unwanted Person (P)",
    "Suspicious Activity (P)",
    "Narcotics (Drug) Activity (P)",
})

# Low-discretion anchors: medical emergencies, used as a null comparison.
ANCHOR_CALL_TYPES = frozenset({
    "Shortness of Breath (FE)",
    "Chest Pain (FE)",
    "Unconscious Person (FE)",
})

# Police_Stop_Data.reason -> initiation class.
STOP_REASON_CLASS = {
    "Citizen / 9-1-1": "reactive",
    "Moving Violation": "proactive",
    "Equipment Violation": "proactive",
    "Investigative": "proactive",
}

# Values that mean "no person was contacted" and must be excluded from any
# per-person race denominator rather than folded into "Unknown".
NON_PERSON_RACE = frozenset({"NO CONTACT", "UNOCCUPIED VEHICLE"})
