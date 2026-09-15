"""Tests for the outcome (hit-rate) test.

These pin the statistical primitives against known values and the filtering
rules against the specific ways this dataset can mislead: residual race codes
that are not groups, non-person contacts, and unusable early years.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.features import hit_rate as hr


def _stops(rows) -> pd.DataFrame:
    """Build a stops-shaped frame from
    (year, race, searched, booked, cited, initiation) tuples."""
    df = pd.DataFrame(
        rows,
        columns=["year", "race_recorded", "searched", "booked", "cited", "initiation"],
    )
    df["race_perceived"] = df["race_recorded"]
    df["search_person"] = df["searched"].astype("boolean")
    df["search_vehicle"] = df["searched"].astype("boolean")
    df["citation"] = df["cited"].astype("boolean")
    df["disposition"] = np.where(df["booked"], "BKG-Booking", "ADV-Advised")
    df["usable_window"] = df["year"] >= 2018
    df["person_recorded"] = ~df["race_recorded"].isin(config.NON_PERSON_RACE)
    df["person_perceived"] = df["person_recorded"]
    return df


def _balanced(race: str, n_searched: int, n_hit: int, year: int = 2024,
              initiation: str = "proactive"):
    """n_searched searched stops of which n_hit end in arrest."""
    return (
        [(year, race, True, True, False, initiation) for _ in range(n_hit)]
        + [(year, race, True, False, False, initiation)
           for _ in range(n_searched - n_hit)]
    )


class TestWilsonInterval:
    def test_brackets_the_point_estimate(self):
        lo, hi = hr.wilson_interval(50, 100)
        assert lo < 0.5 < hi

    def test_stays_within_unit_interval_at_extremes(self):
        lo, hi = hr.wilson_interval(0, 10)
        assert lo == 0.0 and 0 < hi < 1
        lo, hi = hr.wilson_interval(10, 10)
        assert hi == 1.0 and 0 < lo < 1

    def test_narrows_as_sample_grows(self):
        small = hr.wilson_interval(5, 10)
        large = hr.wilson_interval(500, 1000)
        assert (large[1] - large[0]) < (small[1] - small[0])

    def test_zero_trials_is_nan(self):
        lo, hi = hr.wilson_interval(0, 0)
        assert np.isnan(lo) and np.isnan(hi)


class TestTwoProportionTest:
    def test_identical_proportions_give_no_effect(self):
        z, p = hr.two_proportion_test(50, 100, 50, 100)
        assert z == pytest.approx(0.0)
        assert p == pytest.approx(1.0)

    def test_large_clear_difference_is_significant(self):
        z, p = hr.two_proportion_test(20, 100, 60, 100)
        assert p < 0.001
        assert z < 0

    def test_sign_follows_first_argument(self):
        z_lower, _ = hr.two_proportion_test(10, 100, 50, 100)
        z_higher, _ = hr.two_proportion_test(50, 100, 10, 100)
        assert z_lower < 0 < z_higher


class TestPrepare:
    def test_drops_non_group_race_codes(self):
        """Unknown and Other are residual codes, not groups."""
        df = _stops([
            (2024, "Black", True, True, False, "proactive"),
            (2024, "Unknown", True, True, False, "proactive"),
            (2024, "Other", True, True, False, "proactive"),
            (2024, "NO CONTACT", True, True, False, "proactive"),
        ])
        out = hr.prepare(df)
        assert set(out["race_recorded"]) == {"Black"}

    def test_drops_years_outside_window(self):
        df = _stops([
            (2016, "Black", True, True, False, "proactive"),
            (2024, "Black", True, True, False, "proactive"),
        ])
        assert len(hr.prepare(df, year_min=2018, year_max=2026)) == 1


class TestHitRateTable:
    def test_computes_rates_and_gap(self):
        rows = _balanced("Black", 200, 80) + _balanced("White", 200, 100)
        res = hr.hit_rate_table(_stops(rows), min_searches=50)
        t = res.table
        assert t.loc["Black", "hit_rate"] == pytest.approx(0.40)
        assert t.loc["White", "hit_rate"] == pytest.approx(0.50)
        assert t.loc["Black", "hit_gap_pp"] == pytest.approx(-10.0)
        assert t.loc["Black", "p_value"] < 0.05

    def test_reference_group_has_zero_gap(self):
        rows = _balanced("Black", 200, 80) + _balanced("White", 200, 100)
        t = hr.hit_rate_table(_stops(rows), min_searches=50).table
        assert t.loc["White", "hit_gap_pp"] == pytest.approx(0.0)
        assert t.loc["White", "search_rate_ratio"] == pytest.approx(1.0)

    def test_groups_below_min_searches_are_dropped(self):
        rows = _balanced("Black", 200, 80) + _balanced("White", 200, 100) \
            + _balanced("Asian", 10, 5)
        t = hr.hit_rate_table(_stops(rows), min_searches=50).table
        assert "Asian" not in t.index

    def test_citation_and_booking_are_distinct_measures(self):
        """A stop can be cited without arrest; pooling them changes the answer."""
        rows = (
            [(2024, "Black", True, False, True, "proactive")] * 100
            + [(2024, "White", True, True, False, "proactive")] * 100
        )
        booking = hr.hit_rate_table(_stops(rows), hit="booking", min_searches=50).table
        citation = hr.hit_rate_table(_stops(rows), hit="citation", min_searches=50).table
        assert booking.loc["Black", "hit_rate"] == 0.0
        assert citation.loc["Black", "hit_rate"] == 1.0


class TestStratification:
    def test_pooled_gap_can_reverse_within_strata(self):
        """Simpson's paradox is the reason hit_rate_by_stratum exists.

        Mirrors the real result: a deficit in proactive stops sitting beside a
        surplus in reactive ones.
        """
        rows = (
            _balanced("Black", 200, 60, initiation="proactive")
            + _balanced("White", 200, 100, initiation="proactive")
            + _balanced("Black", 200, 100, initiation="reactive")
            + _balanced("White", 200, 70, initiation="reactive")
        )
        df = _stops(rows)

        pooled = hr.hit_rate_table(df, min_searches=50).table
        assert pooled.loc["Black", "hit_gap_pp"] < 0

        strata = hr.hit_rate_by_stratum(df, "initiation", min_searches=50)
        assert strata.loc["proactive", "black_white_gap_pp"] < 0
        assert strata.loc["reactive", "black_white_gap_pp"] > 0
