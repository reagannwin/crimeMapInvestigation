"""Tests for the ShotSpotter coverage mask.

The mask gates every sensor-versus-human comparison, so its failure modes
matter more than its happy path. Two are pinned here: silent acceptance of
unprojected input (metres vs degrees), and closing-operation bleed promoting
a neighbourhood that has no sensors.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box

from src import config
from src.bias import sensor_footprint as sf


def _points(coords) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        geometry=[Point(x, y) for x, y in coords], crs=config.CRS_PROJECTED
    )


def _square(x0, y0, size, name) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"neighborhood": [name]},
        geometry=[box(x0, y0, x0 + size, y0 + size)],
        crs=config.CRS_PROJECTED,
    )


@pytest.fixture
def dense_cluster():
    """400 points packed into a 200 m square -- unambiguous coverage."""
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 200, size=(400, 2)) + np.array([1000.0, 1000.0])
    return _points(xy)


class TestProjectionGuard:
    def test_geographic_crs_is_rejected(self, dense_cluster):
        """Grid and buffer distances are metres; degrees are meaningless."""
        degrees = dense_cluster.to_crs(config.CRS_GEOGRAPHIC)
        with pytest.raises(ValueError, match="metres"):
            sf.build_footprint(degrees)

    def test_missing_crs_is_rejected(self, dense_cluster):
        # set_crs(None, allow_override=True) is required: constructing with
        # crs=None inherits the CRS already carried by the GeometryArray.
        naive = dense_cluster.set_crs(None, allow_override=True)
        assert naive.crs is None
        with pytest.raises(ValueError, match="metres"):
            sf.build_footprint(naive)


class TestBuildFootprint:
    def test_dense_cluster_produces_a_mask(self, dense_cluster):
        fp = sf.build_footprint(dense_cluster, closing_m=0)
        assert fp.area_km2 > 0
        assert fp.n_points == 400

    def test_min_points_excludes_isolated_strays(self, dense_cluster):
        """A lone activation 5 km away must not extend the footprint."""
        with_stray = gpd.GeoDataFrame(
            geometry=list(dense_cluster.geometry) + [Point(6000, 6000)],
            crs=config.CRS_PROJECTED,
        )
        loose = sf.build_footprint(with_stray, min_points=1, closing_m=0)
        strict = sf.build_footprint(with_stray, min_points=3, closing_m=0)
        assert loose.n_cells > strict.n_cells
        assert not strict.geometry.contains(Point(6000, 6000))

    def test_closing_bridges_a_gap_between_clusters(self):
        """Two clusters 400 m apart: closing should join them into one mask.

        Tested on a real gap rather than a single blob, where buffer-out /
        buffer-in is area-neutral to within circle-approximation error and so
        proves nothing.
        """
        rng = np.random.default_rng(1)
        a = rng.uniform(0, 200, size=(200, 2)) + np.array([1000.0, 1000.0])
        b = rng.uniform(0, 200, size=(200, 2)) + np.array([1600.0, 1000.0])
        pts = _points(np.vstack([a, b]))

        open_ = sf.build_footprint(pts, closing_m=0)
        closed = sf.build_footprint(pts, closing_m=400)

        assert open_.geometry.geom_type == "MultiPolygon"
        assert closed.geometry.geom_type == "Polygon"
        assert closed.area_km2 > open_.area_km2

    def test_clip_keeps_mask_inside_boundary(self, dense_cluster):
        boundary = _square(1000, 1000, 100, "small")
        fp = sf.build_footprint(dense_cluster, closing_m=0, clip_to=boundary)
        assert fp.area_km2 <= boundary.geometry.area.sum() / 1e6 + 1e-9

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="no activation"):
            sf.build_footprint(_points([]))

    def test_unreachable_min_points_raises(self, dense_cluster):
        with pytest.raises(ValueError, match="min_points"):
            sf.build_footprint(dense_cluster, min_points=10_000)


class TestCoverageByNeighborhood:
    def test_fully_covered_neighbourhood(self, dense_cluster):
        poly = _square(900, 900, 400, "Core")
        fp = sf.build_footprint(dense_cluster, closing_m=0)
        cov = sf.coverage_by_neighborhood(fp, poly)
        assert cov.loc[0, "covered_frac"] > 0
        assert cov.loc[0, "neighborhood"] == "Core"

    def test_distant_neighbourhood_is_uncovered(self, dense_cluster):
        poly = _square(50_000, 50_000, 400, "Far")
        fp = sf.build_footprint(dense_cluster, closing_m=0)
        cov = sf.coverage_by_neighborhood(fp, poly)
        assert cov.loc[0, "coverage_class"] == "uncovered"

    def test_evidence_guard_downgrades_thinly_evidenced_neighbourhood(self):
        """A handful of stray activations must not buy a coverage label.

        This is the Downtown West failure. Closing does not expand a mask
        outward; what it does is *connect sparse points*. A neighbourhood with
        a couple of activations sitting near a genuinely covered area gets
        swept into one blob, and its area then reads as covered. Downtown West
        lands at 22% covered on the strength of 2 activations in nineteen
        years. Geometry alone must not promote it.
        """
        rng = np.random.default_rng(2)
        dense = rng.uniform(0, 200, size=(400, 2)) + np.array([1000.0, 1000.0])
        # Four strays scattered across the neighbour, enough for closing to
        # sweep the whole square in once they are connected to the dense core.
        strays = np.array([
            [1500.0, 1000.0], [1700.0, 1100.0],
            [1500.0, 1300.0], [1700.0, 1300.0],
        ])
        pts = _points(np.vstack([dense, strays]))

        core = _square(900, 900, 400, "Core")
        neighbour = _square(1450, 950, 400, "Thin")
        polys = gpd.GeoDataFrame(
            pd.concat([core, neighbour], ignore_index=True),
            crs=config.CRS_PROJECTED,
        )
        fp = sf.build_footprint(pts, min_points=1, closing_m=400)

        naive = sf.coverage_by_neighborhood(fp, polys)
        thin_naive = naive.set_index("neighborhood").loc["Thin"]
        assert thin_naive["covered_frac"] > 0, "fixture should produce bleed"

        guarded = sf.coverage_by_neighborhood(
            fp, polys, pts, min_activations=50
        ).set_index("neighborhood").loc["Thin"]

        assert guarded["activations"] == 4
        assert guarded["coverage_class"] == "uncovered"
        assert bool(guarded["evidence_downgraded"]) is True

    def test_evidence_guard_leaves_real_coverage_alone(self, dense_cluster):
        core = _square(900, 900, 400, "Core")
        fp = sf.build_footprint(dense_cluster, closing_m=0)
        cov = sf.coverage_by_neighborhood(
            fp, core, dense_cluster, min_activations=50
        ).set_index("neighborhood").loc["Core"]
        assert cov["activations"] == 400
        assert bool(cov["evidence_downgraded"]) is False

