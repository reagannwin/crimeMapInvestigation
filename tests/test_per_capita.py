"""Tests for per-resident rates and the source-directory override."""
from __future__ import annotations

import importlib

import pandas as pd
import pytest

from src.features import panel as pn


def _annual() -> pd.DataFrame:
    return pd.DataFrame({
        "neighborhood": ["A", "Downtown West"],
        "year": [2025, 2025],
        "stops_total": [50.0, 200.0],
        "stops_proactive": [20.0, 90.0],
        "stops_reactive": [20.0, 60.0],
        "crimes": [40.0, 150.0],
        "calls": [500.0, 1500.0],
    })


def _demo(names=("A", "Downtown West")) -> pd.DataFrame:
    return pd.DataFrame({
        "neighborhood": list(names),
        "pop_total": [2000.0, 1000.0][: len(names)],
        "share_black_nh": [0.3, 0.1][: len(names)],
        "share_white_nh": [0.5, 0.7][: len(names)],
        "share_renter": [0.4, 0.9][: len(names)],
        "median_hh_income_approx": [60000.0, 90000.0][: len(names)],
    })


class TestAddPerCapita:
    def test_rates_are_per_thousand_residents(self):
        out = pn.add_per_capita(_annual(), _demo()).set_index("neighborhood")
        assert out.loc["A", "stops_total_per_1k"] == pytest.approx(25.0)
        assert out.loc["A", "crimes_per_1k"] == pytest.approx(20.0)

    def test_downtown_core_is_flagged(self):
        """Daytime population makes per-resident rates there incomparable."""
        out = pn.add_per_capita(_annual(), _demo()).set_index("neighborhood")
        assert bool(out.loc["Downtown West", "daytime_core"])
        assert not bool(out.loc["A", "daytime_core"])

    def test_missing_population_raises_rather_than_dividing_by_nan(self):
        with pytest.raises(ValueError, match="no population"):
            pn.add_per_capita(_annual(), _demo(names=("A",)))

    def test_row_count_is_preserved(self):
        assert len(pn.add_per_capita(_annual(), _demo())) == 2


class TestSourceDirOverride:
    def test_env_var_overrides_downloads(self, monkeypatch, tmp_path):
        """The Docker image points CRIME_SOURCE_DIR at a read-only mount."""
        from src import config

        monkeypatch.setenv("CRIME_SOURCE_DIR", str(tmp_path))
        try:
            reloaded = importlib.reload(config)
            assert reloaded.SOURCE_DIR == tmp_path
        finally:
            monkeypatch.delenv("CRIME_SOURCE_DIR")
            importlib.reload(config)
