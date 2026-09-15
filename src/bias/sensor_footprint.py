"""Derive the ShotSpotter coverage footprint empirically from activation points.

Why this gates everything downstream
------------------------------------
ShotSpotter covers roughly a quarter of Minneapolis. Outside that area an
absence of activations means an absence of *sensors*, not an absence of
gunfire. Downtown West makes the point: 293 confirmed shootings (5th highest
in the city) and 2 activations in nineteen years, against 9-15 activations per
confirmed shooting in comparably violent neighbourhoods. Any citywide
comparison that reads sensor counts as gunfire will conclude downtown is
quiet, which is false by a factor of roughly a thousand.

So every sensor-versus-human comparison must be restricted to the footprint,
and the footprint has to be derived from the data rather than assumed.

What this mask is, and is not
-----------------------------
It is a *detected-coverage* footprint: the area where sensors have actually
fired. Because activations cluster where gunfire happens, quiet blocks inside
the real sensor zone may contain no activation and fall outside a naive mask.
The footprint is therefore a **lower bound** on true sensor coverage.

Two things mitigate that, both explicit parameters rather than hidden choices:

  * ``closing_m`` applies a morphological closing (buffer out, then in) that
    fills interior gaps without inflating the outer edge.
  * ``min_points`` sets how much evidence a cell needs, so isolated strays -
    a single activation far outside the zone - do not drag the boundary out.

Both are swept by :func:`sweep_parameters`. If a downstream finding only
survives at one setting, it is not a finding.
"""
from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box
from shapely.ops import unary_union

from src import config

# Defaults chosen to be defensible, not magic: a 250 m cell is roughly a city
# block, and 500 m of closing bridges a few blocks of quiet interior without
# reaching across the river or into an adjacent uncovered district.
DEFAULT_CELL_M = 250
# min_points=1 is NOT the safe default it looks like. A single stray activation
# creates a cell, the closing then inflates it, and neighbourhoods with almost
# no sensor evidence get classified as covered: at min_points=1 the mask calls
# Camden Industrial (1 activation) and Regina (3) "covered". Requiring three
# activations in a cell costs little real coverage and removes that failure.
DEFAULT_MIN_POINTS = 3
DEFAULT_CLOSING_M = 500
# Below this share of its area inside the mask, a neighbourhood is not usable
# for sensor-vs-human comparison.
COVERED_THRESHOLD = 0.60
PARTIAL_THRESHOLD = 0.20
# Geometric coverage alone is not enough. Closing deliberately reaches across
# quiet ground, so it can bleed into a neighbour that has no sensors at all --
# Downtown West lands at 22% covered on the strength of 2 activations in
# nineteen years. A neighbourhood must also carry its own direct evidence.
MIN_NEIGHBORHOOD_ACTIVATIONS = 50


@dataclass(frozen=True)
class Footprint:
    """A derived coverage mask plus the parameters that produced it."""

    geometry: object  # shapely (Multi)Polygon in CRS_PROJECTED
    cell_m: int
    min_points: int
    closing_m: int
    n_points: int
    n_cells: int

    @property
    def area_km2(self) -> float:
        return self.geometry.area / 1e6

    def __str__(self) -> str:
        return (
            f"cell={self.cell_m}m min_pts={self.min_points} close={self.closing_m}m "
            f"-> {self.area_km2:.1f} km2 from {self.n_cells:,} cells"
        )


def _require_projected(gdf: gpd.GeoDataFrame, what: str) -> None:
    if gdf.crs is None or gdf.crs.to_string() != config.CRS_PROJECTED:
        raise ValueError(
            f"{what} must be in {config.CRS_PROJECTED} (metres), got {gdf.crs}. "
            "Grid and buffer distances are metres; degrees would be meaningless."
        )


def build_footprint(
    activations: gpd.GeoDataFrame,
    *,
    cell_m: int = DEFAULT_CELL_M,
    min_points: int = DEFAULT_MIN_POINTS,
    closing_m: int = DEFAULT_CLOSING_M,
    clip_to: gpd.GeoDataFrame | None = None,
) -> Footprint:
    """Grid-bin activation points and dissolve occupied cells into a mask.

    Binning is done on integer cell indices rather than by building 38k
    polygons and unioning them, which is orders of magnitude faster and gives
    the same result.
    """
    _require_projected(activations, "activations")

    pts = activations.geometry.dropna()
    if pts.empty:
        raise ValueError("no activation geometries supplied")

    x = pts.x.to_numpy()
    y = pts.y.to_numpy()
    x0 = np.floor(x.min() / cell_m) * cell_m
    y0 = np.floor(y.min() / cell_m) * cell_m

    ix = ((x - x0) // cell_m).astype(np.int64)
    iy = ((y - y0) // cell_m).astype(np.int64)

    cells, counts = np.unique(np.stack([ix, iy], axis=1), axis=0, return_counts=True)
    keep = cells[counts >= min_points]
    if not len(keep):
        raise ValueError(f"no cell reached min_points={min_points}")

    boxes = [
        box(x0 + cx * cell_m, y0 + cy * cell_m,
            x0 + (cx + 1) * cell_m, y0 + (cy + 1) * cell_m)
        for cx, cy in keep
    ]
    mask = unary_union(boxes)

    if closing_m:
        # Morphological closing: fills interior holes, restores the outer edge.
        mask = mask.buffer(closing_m).buffer(-closing_m)

    if clip_to is not None:
        _require_projected(clip_to, "clip_to")
        mask = mask.intersection(unary_union(clip_to.geometry.to_list()))

    return Footprint(
        geometry=mask, cell_m=cell_m, min_points=min_points,
        closing_m=closing_m, n_points=len(pts), n_cells=len(keep),
    )


def coverage_by_neighborhood(
    footprint: Footprint,
    polygons: gpd.GeoDataFrame,
    activations: gpd.GeoDataFrame | None = None,
    *,
    min_activations: int = MIN_NEIGHBORHOOD_ACTIVATIONS,
) -> pd.DataFrame:
    """Share of each neighbourhood's area inside the mask, plus a class label.

    When ``activations`` is supplied, geometric coverage is cross-checked
    against direct evidence: a neighbourhood with fewer than
    ``min_activations`` of its own is forced to ``uncovered`` regardless of how
    much of its area the closing operation reached into, and flagged in
    ``evidence_downgraded``. Without that guard the mask silently promotes
    neighbourhoods that have no sensors.
    """
    _require_projected(polygons, "polygons")

    inside = polygons.geometry.intersection(footprint.geometry).area
    frac = (inside / polygons.geometry.area).fillna(0.0)

    out = pd.DataFrame({
        "neighborhood": polygons["neighborhood"].to_numpy(),
        "area_km2": (polygons.geometry.area / 1e6).to_numpy(),
        "covered_km2": (inside / 1e6).to_numpy(),
        "covered_frac": frac.to_numpy(),
    })
    out["coverage_class"] = np.select(
        [out["covered_frac"] >= COVERED_THRESHOLD,
         out["covered_frac"] >= PARTIAL_THRESHOLD],
        ["covered", "partial"],
        default="uncovered",
    )
    out["evidence_downgraded"] = False

    if activations is not None:
        _require_projected(activations, "activations")
        counts = (
            gpd.sjoin(
                activations[["geometry"]],
                polygons[["neighborhood", "geometry"]],
                how="inner", predicate="within",
            )
            .groupby("neighborhood").size()
        )
        out["activations"] = (
            out["neighborhood"].map(counts).fillna(0).astype(int)
        )
        thin = (out["activations"] < min_activations) & (
            out["coverage_class"] != "uncovered"
        )
        out.loc[thin, "evidence_downgraded"] = True
        out.loc[thin, "coverage_class"] = "uncovered"

    return out.sort_values("covered_frac", ascending=False, ignore_index=True)


def sweep_parameters(
    activations: gpd.GeoDataFrame,
    polygons: gpd.GeoDataFrame,
    *,
    cells: tuple[int, ...] = (150, 250, 400),
    min_points: tuple[int, ...] = (1, 3, 5),
    closings: tuple[int, ...] = (250, 500, 750),
) -> pd.DataFrame:
    """Vary every mask parameter and report how much the footprint moves.

    Stability here is what licenses using a single setting downstream.
    """
    rows = []
    for c in cells:
        for m in min_points:
            for k in closings:
                fp = build_footprint(
                    activations, cell_m=c, min_points=m, closing_m=k,
                    clip_to=polygons,
                )
                cov = coverage_by_neighborhood(fp, polygons, activations)
                rows.append({
                    "cell_m": c, "min_points": m, "closing_m": k,
                    "area_km2": round(fp.area_km2, 1),
                    "n_covered": int((cov["coverage_class"] == "covered").sum()),
                    "n_partial": int((cov["coverage_class"] == "partial").sum()),
                    "n_uncovered": int((cov["coverage_class"] == "uncovered").sum()),
                    "n_downgraded": int(cov["evidence_downgraded"].sum()),
                })
    return pd.DataFrame(rows)


def load_activations() -> gpd.GeoDataFrame:
    """Sensor activations only. Human 911 reports are not sensor evidence."""
    shots = gpd.read_parquet(config.DATA_INTERIM / "shots_fired.parquet")
    return shots[shots["is_sensor"]].copy()


def load_polygons() -> gpd.GeoDataFrame:
    return gpd.read_parquet(config.DATA_INTERIM / "neighborhood_polygons.parquet")
