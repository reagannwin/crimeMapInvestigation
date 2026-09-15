"""Tests for neighbourhood canonicalisation.

This module is the linchpin of every downstream join: if it silently drops or
mis-maps a spelling, neighbourhood-level rates are wrong in ways that look
plausible. These tests pin the behaviour observed in the delivered data.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.clean import neighborhoods as nb


class TestNormaliseKey:
    def test_case_is_folded(self):
        assert nb.normalise_key("ARMATAGE") == nb.normalise_key("Armatage")

    def test_apostrophes_are_stripped(self):
        # "Steven's Square - Loring Heights" vs "STEVENS SQUARE - LORING HEIGHTS"
        assert nb.normalise_key("Steven's Square - Loring Heights") == \
               nb.normalise_key("STEVENS SQUARE - LORING HEIGHTS")

    def test_periods_are_stripped(self):
        assert nb.normalise_key("St. Anthony East") == nb.normalise_key("ST ANTHONY EAST")

    def test_hyphen_spacing_is_normalised(self):
        keys = {
            nb.normalise_key("Bryn - Mawr"),
            nb.normalise_key("Bryn-Mawr"),
            nb.normalise_key("BRYN  -  MAWR"),
        }
        assert len(keys) == 1

    def test_blank_and_none_give_empty_key(self):
        assert nb.normalise_key(None) == ""
        assert nb.normalise_key("   ") == ""
        assert nb.normalise_key(float("nan")) == ""


class TestCanonicalNames:
    def test_exactly_87_neighbourhoods(self):
        assert len(nb.canonical_names()) == 87

    def test_renamed_targets_are_canonical(self):
        names = set(nb.canonical_names())
        assert {"South Uptown", "East Bde Maka Ska", "West Maka Ska"} <= names

    def test_legacy_names_are_not_canonical(self):
        names = set(nb.canonical_names())
        assert not ({"CARAG", "ECCO", "West Calhoun"} & names)


class TestCanonicalise:
    def test_uppercase_and_propercase_converge(self):
        """The 2018 overlap bug: both spellings must land on one name."""
        out = nb.canonicalise(pd.Series(["ARMATAGE", "Armatage"]))
        assert out.nunique() == 1
        assert out.iloc[0] == "Armatage"

    @pytest.mark.parametrize(
        "legacy,expected",
        [
            ("CARAG", "South Uptown"),
            ("ECCO", "East Bde Maka Ska"),
            ("West Calhoun", "West Maka Ska"),
            ("WEST CALHOUN", "West Maka Ska"),
        ],
    )
    def test_renames_are_applied(self, legacy, expected):
        assert nb.canonicalise(pd.Series([legacy])).iloc[0] == expected

    @pytest.mark.parametrize(
        "sentinel", ["** NOT ASSIGNED **", "Z_** NOT ASSIGNED **", "", "   "]
    )
    def test_unassigned_becomes_na(self, sentinel):
        assert nb.canonicalise(pd.Series([sentinel])).isna().all()

    def test_unrecognised_raises_in_strict_mode(self):
        with pytest.raises(ValueError, match="unrecognised"):
            nb.canonicalise(pd.Series(["Sometown"]))

    def test_unrecognised_becomes_na_when_not_strict(self):
        out = nb.canonicalise(pd.Series(["Sometown"]), strict=False)
        assert out.isna().all()

    def test_non_neighbourhood_tokens_are_rejected(self):
        """Guard against a mis-parsed CSV shifting ward numbers ('1'..'13') or
        response flags ('N','Y') into the neighbourhood column. The delivered
        files parse cleanly, so this is a regression guard, not a known defect."""
        with pytest.raises(ValueError):
            nb.canonicalise(pd.Series(["7", "N", "Y"]))


class TestCoverageReport:
    def test_flags_unrecognised_without_raising(self):
        rep = nb.coverage_report(pd.Series(["Armatage", "Sometown", "CARAG"]))
        statuses = dict(zip(rep["raw"], rep["status"]))
        assert statuses["Sometown"] == "UNRECOGNISED"
        assert statuses["Armatage"] == "mapped"
        assert statuses["CARAG"] == "mapped"

    def test_reports_rename_target(self):
        rep = nb.coverage_report(pd.Series(["CARAG"]))
        assert rep.loc[rep["raw"] == "CARAG", "canonical"].iloc[0] == "South Uptown"
