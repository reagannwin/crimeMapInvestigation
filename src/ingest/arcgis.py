"""Paginated ArcGIS Feature Service client.

Used for the two layers that were not supplied as CSV: ``Shots_Fired``
(108,800 points) and ``Minneapolis_Neighborhoods`` (87 polygons, the geometry
the supplied attribute-table CSV lacks).

Three details the service imposes:

  * ``maxRecordCount`` is per-layer (16,000 for Shots_Fired, 1,000 for
    Minneapolis_Neighborhoods) and the server silently truncates to it, so the
    page size is read from the layer rather than assumed.
  * Paging without ``orderByFields`` is not stable -- rows can repeat or
    vanish between pages. Every request orders by the layer's OID field.
  * ``f=geojson`` always returns WGS84 regardless of the layer's native SRS
    (Shots_Fired is EPSG:4269, Minneapolis_Neighborhoods is EPSG:3857), which
    is why the loaders reproject explicitly rather than trusting the source.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Iterator

import requests

from src import config

USER_AGENT = "crimeMapInvestigation/0.1 (research; contact via repo)"
TIMEOUT = 60
MAX_RETRIES = 4
BACKOFF_SECONDS = 2.0


class ArcGISError(RuntimeError):
    """The service returned an error payload or malformed JSON."""


@dataclass(frozen=True)
class LayerInfo:
    name: str
    oid_field: str
    max_record_count: int
    geometry_type: str | None
    count: int


def _get(url: str, params: dict[str, Any]) -> dict[str, Any]:
    """GET with retry. Raises ArcGISError on a service-level error payload.

    ArcGIS returns HTTP 200 with an ``{"error": ...}`` body on failure, so the
    status code alone is not a success signal.
    """
    last: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(
                url, params=params, timeout=TIMEOUT,
                headers={"User-Agent": USER_AGENT},
            )
            resp.raise_for_status()
            payload = resp.json()
            if isinstance(payload, dict) and "error" in payload:
                raise ArcGISError(f"{url}: {payload['error']}")
            return payload
        except (requests.RequestException, json.JSONDecodeError) as exc:
            last = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_SECONDS * (2 ** attempt))
    raise ArcGISError(f"{url}: failed after {MAX_RETRIES} attempts: {last}")


def layer_info(url: str, where: str = "1=1") -> LayerInfo:
    meta = _get(url, {"f": "json"})
    # Count with the caller's filter, or a filtered pull is compared
    # against the whole layer and trips a false mismatch warning.
    count = _get(url + "/query", {"where": where, "returnCountOnly": "true", "f": "json"})
    # Hosted FeatureServers expose objectIdField; MapServers such as TIGERweb
    # often do not, and name the OID column "OID". Read it from the schema
    # rather than assuming, or orderByFields points at a column that is absent.
    oid = meta.get("objectIdField") or next(
        (f["name"] for f in meta.get("fields", [])
         if f.get("type") == "esriFieldTypeOID"),
        "OBJECTID",
    )
    return LayerInfo(
        name=meta.get("name", "?"),
        oid_field=oid,
        max_record_count=int(meta.get("maxRecordCount", 1000)),
        geometry_type=meta.get("geometryType"),
        count=int(count.get("count", 0)),
    )


def iter_geojson_pages(
    url: str,
    *,
    where: str = "1=1",
    out_fields: str = "*",
    page_size: int | None = None,
    out_sr: int = 4326,
    verbose: bool = True,
) -> Iterator[dict[str, Any]]:
    """Yield GeoJSON FeatureCollection pages until the layer is exhausted."""
    info = layer_info(url, where)
    size = min(page_size or info.max_record_count, info.max_record_count)
    if verbose:
        print(f"   {info.name}: {info.count:,} features, page size {size:,}")

    offset = 0
    seen = 0
    while True:
        page = _get(url + "/query", {
            "where": where,
            "outFields": out_fields,
            "returnGeometry": "true",
            "outSR": out_sr,
            "orderByFields": info.oid_field,  # stable paging
            "resultOffset": offset,
            "resultRecordCount": size,
            "f": "geojson",
        })
        feats = page.get("features") or []
        if not feats:
            break

        seen += len(feats)
        if verbose:
            print(f"      {seen:,}/{info.count:,}", end="\r", flush=True)
        yield page

        if len(feats) < size:
            break
        offset += size

        if seen > info.count * 2:  # runaway guard
            raise ArcGISError(f"{url}: paging exceeded twice the reported count")

    if verbose:
        print(f"      {seen:,}/{info.count:,} fetched")
    if info.count and seen != info.count:
        print(f"   WARNING: fetched {seen:,} but layer reports {info.count:,}")


def fetch_geoframe(url: str, *, where: str = "1=1", out_fields: str = "*",
                   verbose: bool = True):
    """Fetch a whole layer into a GeoDataFrame projected to the analysis CRS."""
    import geopandas as gpd
    import pandas as pd

    frames = [
        gpd.GeoDataFrame.from_features(page["features"], crs=config.CRS_GEOGRAPHIC)
        for page in iter_geojson_pages(url, where=where, out_fields=out_fields,
                                       verbose=verbose)
    ]
    if not frames:
        raise ArcGISError(f"{url}: no features returned")

    gdf = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True), crs=config.CRS_GEOGRAPHIC
    )
    return gdf.to_crs(config.CRS_PROJECTED)
