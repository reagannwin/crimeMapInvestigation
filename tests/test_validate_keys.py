"""Tests for join-key normalisation and initiation recovery.

The load-bearing test is ``test_no_imputation_outside_the_calls_window``. The
imputation rule reads "no 911 match" as "not community-initiated", which only
holds where a match was possible. The 911 extract begins 2020-07, so applying
the rule to earlier stops labelled 45,104 of them proactive -- 100%, with no
possibility of the other answer -- and would have manufactured exactly the
trend the panel exists to measure.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.join import validate_keys as vk

WINDOW_START = pd.Timestamp("2020-07-01", tz="America/Chicago")
WINDOW_END = pd.Timestamp("2026-06-01", tz="America/Chicago")


def _calls(ids, times) -> pd.DataFrame:
    return pd.DataFrame({
        "incident_id": pd.Series(ids, dtype="string"),
        "ts": pd.to_datetime(pd.Series(times)).dt.tz_localize("America/Chicago"),
    })


def _stops(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["incident_id", "ts", "initiation"])
    df["incident_id"] = df["incident_id"].astype("string")
    df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize("America/Chicago")
    df["initiation"] = df["initiation"].astype("string")
    return df


class TestNormaliseIncidentId:
    def test_uppercases_and_strips(self):
        out = vk.normalise_incident_id(pd.Series([" 20-0026589 "], dtype="string"))
        assert out.iloc[0] == "20-0026589"

    @pytest.mark.parametrize("value", ["20-123456", "20-1234567"])
    def test_accepts_both_real_id_shapes(self, value):
        """POLICE/BCR use six digits, FIRE seven. They do not collide."""
        assert vk.normalise_incident_id(pd.Series([value], dtype="string")).iloc[0] == value

    @pytest.mark.parametrize("value", ["99999999", "abc", "20-12345", "", "20_123456"])
    def test_rejects_malformed_ids(self, value):
        assert vk.normalise_incident_id(pd.Series([value], dtype="string")).isna().all()


class TestImputeInitiation:
    def test_observed_labels_are_never_overwritten(self):
        calls = _calls(["20-100001"], ["2021-01-01"])
        stops = _stops([("20-100001", "2021-01-01", "proactive")])
        out = vk.impute_initiation(stops, calls)
        assert out["initiation_filled"].iloc[0] == "proactive"
        assert not out["initiation_imputed"].iloc[0]

    def test_blank_with_a_match_becomes_reactive(self):
        calls = _calls(["20-100001"], ["2021-01-01"])
        stops = _stops([("20-100001", "2021-01-01", None)])
        out = vk.impute_initiation(stops, calls)
        assert out["initiation_filled"].iloc[0] == "reactive"
        assert bool(out["initiation_imputed"].iloc[0])

    def test_blank_without_a_match_becomes_proactive(self):
        calls = _calls(["20-100001"], ["2021-01-01"])
        stops = _stops([("20-999999", "2021-01-01", None)])
        out = vk.impute_initiation(stops, calls)
        assert out["initiation_filled"].iloc[0] == "proactive"

    def test_no_imputation_outside_the_calls_window(self):
        """Before the 911 extract begins, no stop can match, so "no match"
        carries no information and must not be read as proactive."""
        calls = _calls(["20-100001"], ["2021-01-01"])
        stops = _stops([("18-500000", "2018-03-01", None)])
        out = vk.impute_initiation(stops, calls)
        assert pd.isna(out["initiation_filled"].iloc[0])
        assert not out["initiation_imputed"].iloc[0]
        assert bool(out["initiation_unresolved"].iloc[0])

    def test_no_imputation_after_the_calls_window(self):
        calls = _calls(["20-100001"], ["2021-01-01"])
        stops = _stops([("26-500000", "2026-09-01", None)])
        out = vk.impute_initiation(stops, calls)
        assert pd.isna(out["initiation_filled"].iloc[0])
        assert bool(out["initiation_unresolved"].iloc[0])

    def test_out_of_window_stops_are_not_all_one_label(self):
        """Regression on the original defect: every pre-window blank became
        proactive, which manufactures a proactive trend out of nothing."""
        calls = _calls(["20-100001"], ["2021-01-01"])
        stops = _stops([
            (f"18-50000{i}", "2018-03-01", None) for i in range(5)
        ])
        out = vk.impute_initiation(stops, calls)
        assert out["initiation_filled"].isna().all()
        assert int(out["initiation_imputed"].sum()) == 0

    def test_observed_labels_survive_outside_the_window(self):
        calls = _calls(["20-100001"], ["2021-01-01"])
        stops = _stops([("18-500000", "2018-03-01", "reactive")])
        out = vk.impute_initiation(stops, calls)
        assert out["initiation_filled"].iloc[0] == "reactive"
        assert not out["initiation_unresolved"].iloc[0]
