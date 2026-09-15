"""Tests for spatial weights, Moran's I and the elasticity test.

Moran's I is checked against constructed patterns with a known sign rather
than against a hard-coded statistic, so the tests assert the property that
matters (clustered positive, dispersed negative) without pinning an
implementation detail of the estimator.
"""
from __future__ import annotations

import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from src import config
from src.bias import spatial as sp

warnings.filterwarnings("ignore")

GRID = 5  # 5x5 lattice -> 25 units, enough for stable permutation inference


@pytest.fixture(scope="module")
def lattice() -> gpd.GeoDataFrame:
    """Regular grid of square 'neighbourhoods' with predictable adjacency."""
    cells, names = [], []
    for row in range(GRID):
        for col in range(GRID):
            cells.append(box(col, row, col + 1, row + 1))
            names.append(f"n_{row}_{col}")
    return gpd.GeoDataFrame(
        {"neighborhood": names}, geometry=cells, crs=config.CRS_PROJECTED
    )


@pytest.fixture(scope="module")
def weights(lattice):
    return sp.build_weights(lattice)


class TestBuildWeights:
    def test_queen_lattice_is_fully_connected(self, weights):
        assert weights.n == GRID * GRID
        assert weights.islands == []
        assert weights.n_components == 1

    def test_weights_are_row_standardised(self, weights):
        for neighbours in weights.weights.values():
            assert sum(neighbours) == pytest.approx(1.0)

    def test_corner_has_three_queen_neighbours(self, weights):
        assert weights.cardinalities["n_0_0"] == 3

    def test_interior_has_eight_queen_neighbours(self, weights):
        assert weights.cardinalities["n_2_2"] == 8

    def test_knn_gives_every_unit_k_neighbours(self, lattice):
        w = sp.build_weights(lattice, kind="knn", k=4)
        assert set(w.cardinalities.values()) == {4}

    def test_unknown_kind_raises(self, lattice):
        with pytest.raises(ValueError, match="unknown weights kind"):
            sp.build_weights(lattice, kind="voronoi")

    def test_islands_are_rejected_not_silently_dropped(self):
        """A unit with no neighbours cannot enter a spatial model."""
        gdf = gpd.GeoDataFrame(
            {"neighborhood": ["a", "b", "far"]},
            geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(50, 50, 51, 51)],
            crs=config.CRS_PROJECTED,
        )
        with pytest.raises(ValueError, match="island"):
            sp.build_weights(gdf)


class TestMoransI:
    def test_clustered_values_give_positive_i(self, lattice, weights):
        """Left half low, right half high -> strong positive autocorrelation."""
        col = np.array([int(n.split("_")[2]) for n in lattice["neighborhood"]])
        values = np.where(col < GRID / 2, 0.0, 10.0)
        i, p = sp.morans_i(values, weights)
        assert i > 0.3
        assert p < 0.05

    def test_checkerboard_gives_negative_i(self, lattice, weights):
        parts = [n.split("_") for n in lattice["neighborhood"]]
        values = np.array([(int(r) + int(c)) % 2 for _, r, c in parts], dtype=float)
        i, _ = sp.morans_i(values, weights)
        assert i < 0

    def test_random_values_give_i_near_zero(self, weights):
        rng = np.random.default_rng(7)
        i, p = sp.morans_i(rng.normal(size=weights.n), weights)
        assert abs(i) < 0.35
        assert p > 0.05


class TestPrepareCrossSection:
    def test_selects_year_and_logs_variables(self):
        annual = pd.DataFrame({
            "neighborhood": ["a", "b", "a"],
            "year": [2024, 2024, 2025],
            "stops_total": [10.0, 20.0, 30.0],
            "crimes": [5.0, 10.0, 15.0],
            "calls": [100.0, 200.0, 300.0],
        })
        out = sp.prepare_cross_section(annual, year=2024)
        assert len(out) == 2
        assert out["log_stops_total"].iloc[0] == pytest.approx(np.log(11.0))

    def test_zero_counts_survive_the_log(self):
        """Structural zeros are meaningful and must not become -inf or drop."""
        annual = pd.DataFrame({
            "neighborhood": ["a"], "year": [2024],
            "stops_total": [0.0], "crimes": [0.0], "calls": [0.0],
        })
        out = sp.prepare_cross_section(annual, year=2024)
        assert np.isfinite(out[["log_stops_total", "log_crimes", "log_calls"]]).all().all()


class TestElasticityTest:
    def _fit(self, beta_crime: float, n: int = 25, seed: int = 0):
        """Synthesise a cross-section with a known crime elasticity."""
        rng = np.random.default_rng(seed)
        log_crimes = rng.normal(4, 0.6, n)
        log_calls = rng.normal(6, 0.4, n)
        log_area = rng.normal(0.5, 0.3, n)
        log_stops = (
            1.0 + beta_crime * log_crimes + 0.1 * log_calls + 0.05 * log_area
            + rng.normal(0, 0.05, n)
        )
        return pd.DataFrame({
            "neighborhood": [f"n_{i}" for i in range(n)],
            "log_stops_total": log_stops, "log_crimes": log_crimes,
            "log_calls": log_calls, "log_area_km2": log_area,
        })

    def test_recovers_a_proportional_elasticity(self, lattice, weights):
        df = self._fit(1.0).assign(neighborhood=lattice["neighborhood"].values)
        fit = sp.fit(df.sort_values("neighborhood").reset_index(drop=True), weights)
        ols = fit.elasticity_test().set_index("model").loc["ols"]
        assert ols["beta"] == pytest.approx(1.0, abs=0.1)
        assert bool(ols["proportional_at_05"])

    def test_detects_a_sub_proportional_elasticity(self, lattice, weights):
        df = self._fit(0.5).assign(neighborhood=lattice["neighborhood"].values)
        fit = sp.fit(df.sort_values("neighborhood").reset_index(drop=True), weights)
        ols = fit.elasticity_test().set_index("model").loc["ols"]
        assert ols["beta"] == pytest.approx(0.5, abs=0.1)
        assert not bool(ols["proportional_at_05"])
        assert ols["p_vs_1"] < 0.05

    def test_tests_against_one_not_zero(self, lattice, weights):
        """A beta of 1 is highly significant against 0 but proportional
        against 1. Testing the wrong null answers the wrong question."""
        df = self._fit(1.0).assign(neighborhood=lattice["neighborhood"].values)
        fit = sp.fit(df.sort_values("neighborhood").reset_index(drop=True), weights)
        ols = fit.elasticity_test().set_index("model").loc["ols"]
        assert ols["p_vs_1"] > 0.05
        z_vs_zero = ols["beta"] / ols["se"]
        assert abs(z_vs_zero) > 5
