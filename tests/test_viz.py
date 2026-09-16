"""Tests for the figure layer.

Figures are judged by eye, but a few properties are worth pinning because
breaking them produces a plausible-looking chart that misleads: no-data
polygons rendered as if they were low values, and a colourbar in different
units from the labels on the same map.
"""
from __future__ import annotations

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

matplotlib.use("Agg")

from src import config  # noqa: E402
from src.viz import charts, maps, style  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _style():
    style.apply()


@pytest.fixture
def polys() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"neighborhood": [f"n{i}" for i in range(6)]},
        geometry=[box(i, 0, i + 1, 1) for i in range(6)],
        crs=config.CRS_PROJECTED,
    )


class TestStyle:
    def test_fit_bounds_crops_to_the_data(self, polys):
        fig, ax = style.figure()
        style.fit_bounds(ax, polys, pad=0.0)
        assert ax.get_xlim() == pytest.approx((0.0, 6.0))
        assert ax.get_ylim() == pytest.approx((0.0, 1.0))

    def test_fit_bounds_pads_proportionally(self, polys):
        fig, ax = style.figure()
        style.fit_bounds(ax, polys, pad=0.1)
        lo, hi = ax.get_xlim()
        assert lo < 0 and hi > 6

    def test_lisa_and_no_data_share_the_grey(self):
        """One grey, one meaning: 'nothing to report here'."""
        assert style.LISA_COLORS["ns"] == style.NO_DATA

    def test_palette_has_no_duplicate_colours(self):
        assert len(set(style.CATEGORICAL)) == len(style.CATEGORICAL)


class TestChoropleth:
    def _values(self, n: int = 4) -> pd.DataFrame:
        return pd.DataFrame({
            "neighborhood": [f"n{i}" for i in range(n)],
            "rate": np.linspace(0.05, 0.25, n),
        })

    def test_missing_polygons_are_hatched_not_blank(self, polys):
        """A reader must not mistake 'not measured' for 'low value'."""
        fig = maps.choropleth(
            polys, self._values(4), "rate", title="t", legend_label="l"
        )
        ax = fig.axes[0]
        hatched = [c for c in ax.collections if c.get_hatch()]
        assert hatched, "polygons without a value must carry a hatch"

    def test_all_polygons_are_drawn(self, polys):
        fig = maps.choropleth(
            polys, self._values(4), "rate", title="t", legend_label="l"
        )
        drawn = sum(len(c.get_paths()) for c in fig.axes[0].collections
                    if hasattr(c, "get_paths"))
        assert drawn >= len(polys)

    def test_percent_format_propagates_to_the_colourbar(self, polys):
        """Colourbar and polygon labels must share units."""
        fig = maps.choropleth(
            polys, self._values(4), "rate", title="t", legend_label="l",
            fmt="{:.0%}",
        )
        cbar_ax = fig.axes[-1]
        labels = [t.get_text() for t in cbar_ax.get_yticklabels()]
        assert any("%" in t for t in labels)

    def test_no_data_legend_appears_only_when_needed(self, polys):
        complete = pd.DataFrame({
            "neighborhood": polys["neighborhood"],
            "rate": np.linspace(0.1, 0.4, len(polys)),
        })
        fig = maps.choropleth(
            polys, complete, "rate", title="t", legend_label="l"
        )
        assert fig.axes[0].get_legend() is None


class TestCharts:
    def test_trend_is_indexed_to_the_base_year(self):
        citywide = pd.DataFrame(
            {"stops_total": [100, 150], "stops_proactive": [50, 100],
             "stops_reactive": [50, 40], "calls": [1000, 1020],
             "crimes": [200, 190]},
            index=pd.Index([2021, 2025], name="year"),
        )
        fig = charts.enforcement_trend(citywide)
        ax = fig.axes[0]
        first_points = [line.get_ydata()[0] for line in ax.lines
                        if len(line.get_ydata()) == 2]
        assert all(y == pytest.approx(100.0) for y in first_points)

    def test_reporting_chart_orders_by_rate(self):
        rates = pd.DataFrame({
            "neighborhood": ["a", "b", "c"],
            "report_rate": [0.2, 0.05, 0.12],
            "sensor_events": [100, 100, 100],
            "with_call": [20, 5, 12],
        })
        fig = charts.reporting_rate_chart(rates)
        labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
        assert labels == ["b", "c", "a"]


class TestDemographicCharts:
    def _diagnostics(self) -> pd.DataFrame:
        rows = []
        for year in (2024, 2025):
            for variant in ("full", "full + income", "full, KNN weights",
                            "full, no downtown", "calls (retracted)"):
                rows.append({
                    "year": year, "variant": variant,
                    "crime_beta_error": 1.0, "crime_se_error": 0.1,
                    "black_beta": 0.8, "black_se": 0.3,
                })
        return pd.DataFrame(rows)

    def test_elasticity_chart_keeps_the_retracted_line_visible(self):
        """The retraction is shown, not hidden, so the reader sees its size."""
        fig = charts.elasticity_chart(self._diagnostics())
        labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert any("Retracted" in label for label in labels)

    def test_elasticity_chart_does_not_overclaim_a_final_year_dip(self):
        """An interval that still reaches 1 must be said to reach 1."""
        fig = charts.elasticity_chart(self._diagnostics())
        texts = " ".join(t.get_text() for t in fig.axes[0].texts)
        assert "interval still reaches 1" in texts

    def test_black_share_chart_plots_every_check(self):
        fig = charts.black_share_chart(self._diagnostics())
        labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert len(labels) == 4

    def _bands_annual(self, top_crime: float) -> pd.DataFrame:
        n = 8
        return pd.DataFrame({
            "neighborhood": [f"n{i}" for i in range(n)],
            "year": 2025,
            "daytime_core": False,
            "share_black_nh": np.linspace(0.01, 0.6, n),
            "pop_total": 1000.0,
            "stops_total": [20, 22, 40, 42, 60, 60, 84, 84],
            "crimes": [30, 30, 44, 44, 57, 57, top_crime, top_crime],
        })

    def test_bands_title_is_computed_not_asserted(self):
        fig = charts.per_capita_bands_chart(self._bands_annual(57), year=2025)
        assert "Same crime rate, 40% more stops" in fig.axes[0].get_title(loc="left")

    def test_bands_title_changes_when_crime_differs(self):
        """The 'same crime rate' headline must not survive data that breaks it."""
        fig = charts.per_capita_bands_chart(self._bands_annual(90), year=2025)
        title = fig.axes[0].get_title(loc="left")
        assert title, "read the left-aligned title; an empty string passes vacuously"
        assert "Same crime rate" not in title
