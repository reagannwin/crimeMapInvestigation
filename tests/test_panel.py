"""Tests for the enforcement panel.

Three failure modes are pinned here because each is silent and each biases a
headline number:

  * a partial final month read as a collapse in activity,
  * a structural zero conflated with an unobserved month,
  * an annual total summed over a year the source only partly covers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import panel as pn


def _ts(values) -> pd.Series:
    return pd.to_datetime(pd.Series(values)).dt.tz_localize("America/Chicago")


class TestLastCompleteMonth:
    def test_partial_final_month_is_excluded(self):
        ts = _ts(["2026-05-20", "2026-06-08"])
        assert pn.last_complete_month(ts) == pd.Timestamp("2026-05-01")

    def test_full_final_month_is_kept(self):
        ts = _ts(["2026-05-02", "2026-05-31"])
        assert pn.last_complete_month(ts) == pd.Timestamp("2026-05-01")

    def test_handles_naive_timestamps(self):
        ts = pd.to_datetime(pd.Series(["2026-05-20", "2026-06-08"]))
        assert pn.last_complete_month(ts) == pd.Timestamp("2026-05-01")

    def test_february_is_not_special_cased_wrongly(self):
        assert pn.last_complete_month(_ts(["2024-02-29"])) == pd.Timestamp("2024-02-01")
        assert pn.last_complete_month(_ts(["2023-02-28"])) == pd.Timestamp("2023-02-01")


class TestEmpiricalBayes:
    # Rates must differ from the global rate, or there is nothing to shrink
    # toward and every assertion here passes vacuously.
    THIN_EVENTS, THIN_EXPOSURE = 1.0, 10.0      # rate 0.10
    THICK_EVENTS, THICK_EXPOSURE = 100.0, 200.0  # rate 0.50

    def _cells(self):
        return (
            pd.Series([self.THIN_EVENTS, self.THICK_EVENTS]),
            pd.Series([self.THIN_EXPOSURE, self.THICK_EXPOSURE]),
        )

    def test_thin_cells_shrink_more_than_thick_ones(self):
        events, exposure = self._cells()
        out = pn.empirical_bayes_rate(events, exposure, prior_strength=50.0)
        raw = events / exposure
        assert abs(out[0] - raw[0]) > abs(out[1] - raw[1])

    def test_estimate_stays_between_raw_and_global(self):
        events = pd.Series([10.0, 1.0, 50.0])
        exposure = pd.Series([100.0, 2.0, 100.0])
        global_rate = events.sum() / exposure.sum()
        out = pn.empirical_bayes_rate(events, exposure, prior_strength=20.0)
        for i in range(len(events)):
            raw = events[i] / exposure[i]
            assert min(raw, global_rate) - 1e-9 <= out[i] <= max(raw, global_rate) + 1e-9

    def test_zero_exposure_is_na_not_infinity(self):
        out = pn.empirical_bayes_rate(pd.Series([5.0]), pd.Series([0.0]))
        assert out.isna().all()

    def test_stronger_prior_shrinks_harder(self):
        events, exposure = self._cells()
        weak = pn.empirical_bayes_rate(events, exposure, prior_strength=1.0)
        strong = pn.empirical_bayes_rate(events, exposure, prior_strength=500.0)
        global_rate = events.sum() / exposure.sum()
        assert abs(strong[0] - global_rate) < abs(weak[0] - global_rate)


class TestMonthlyCounts:
    def test_distinct_counts_incidents_not_rows(self):
        """Use-of-force logs one row per action per officer.

        Counting rows overstates force roughly seventeen-fold.
        """
        df = pd.DataFrame({
            "neighborhood": ["Jordan"] * 5,
            "month_start": [pd.Timestamp("2025-01-01")] * 5,
            "incident_id": ["A", "A", "A", "B", "B"],
        })
        rows = pn._monthly_counts(df, name="n")
        incidents = pn._monthly_counts(df, name="n", distinct="incident_id")
        assert rows["n"].iloc[0] == 5
        assert incidents["n"].iloc[0] == 2

    def test_rows_without_neighborhood_are_dropped(self):
        df = pd.DataFrame({
            "neighborhood": ["Jordan", None],
            "month_start": [pd.Timestamp("2025-01-01")] * 2,
            "incident_id": ["A", "B"],
        })
        assert pn._monthly_counts(df, name="n")["n"].iloc[0] == 1


class TestAnnualSummary:
    def _panel(self, years, covered_from=None):
        months = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-01", freq="MS")
        p = pd.DataFrame({
            "neighborhood": "Jordan",
            "month_start": months,
            "stops_total": 10.0,
            "stops_proactive": 4.0,
            "stops_reactive": 3.0,
            "stops_searched": 1.0,
            "calls": 100.0,
            "crimes": 20.0,
            "force_incidents": 1.0,
        })
        covered = (
            p["month_start"] >= pd.Timestamp(covered_from)
            if covered_from else pd.Series(True, index=p.index)
        )
        p["has_stops"] = True
        p["has_calls"] = True
        p["has_crimes"] = covered
        p.loc[~covered, "crimes"] = np.nan
        return p

    def test_fully_covered_year_is_kept(self):
        out = pn.annual_summary(self._panel([2024]))
        assert list(out["year"]) == [2024]
        assert out["crimes"].iloc[0] == pytest.approx(240.0)

    def test_partially_covered_year_is_dropped(self):
        """Crime data begins 2017-08.

        A naive 12-month check keeps 2017 and then sums five months of
        offences, inflating stops_per_crime roughly two-and-a-half fold.
        """
        out = pn.annual_summary(self._panel([2017], covered_from="2017-08-01"))
        assert out.empty

    def test_mixed_years_keep_only_the_covered_one(self):
        out = pn.annual_summary(self._panel([2017, 2018], covered_from="2017-08-01"))
        assert list(out["year"]) == [2018]

    def test_rates_divide_totals_not_average_ratios(self):
        """Summing counts then dividing weights months by their exposure."""
        p = self._panel([2024])
        p.loc[p["month_start"].dt.month == 1, ["stops_total", "crimes"]] = [100.0, 1.0]
        out = pn.annual_summary(p)
        expected = out["stops_total"].iloc[0] / out["crimes"].iloc[0]
        assert out["stops_per_crime"].iloc[0] == pytest.approx(expected)
