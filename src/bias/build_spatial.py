"""Phase 6: spatial dependence diagnostics and models.

    python -m src.bias.build_spatial

Writes to ``data/processed``:

    spatial_diagnostics.csv    every spec x year x variant: Moran's I, LM
                               choice, crime elasticity vs 1, share_black
    spatial_coefficients.csv   OLS vs lag vs error, model of record
    spatial_lisa.parquet       LISA cluster labels per neighbourhood

Needs ``neighborhood_demographics.parquet`` (src.ingest.build_census).
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from src import config
from src.bias import spatial as sp

YEARS = (2021, 2022, 2023, 2024, 2025)

# (spec, weights, drop downtown core)
VARIANTS = {
    "calls (retracted)": ("calls", "queen", False),
    "pop": ("pop", "queen", False),
    "full": ("full", "queen", False),
    "full + income": ("full_income", "queen", False),
    "full, KNN weights": ("full", "knn", False),
    "full, no downtown": ("full", "queen", True),
}


def _row(fit: sp.SpatialFit, *, year: int, variant: str, spec: str) -> dict:
    et = fit.elasticity_test().set_index("model").loc["error"]
    row = {
        "year": year, "variant": variant, "spec": spec, "n": fit.n,
        "moran_resid": fit.moran_resid[0], "moran_resid_p": fit.moran_resid[1],
        "recommended": fit.recommended, "ols_r2": fit.ols.r2,
        "condition_number": fit.ols.mulColli,
        "crime_beta_ols": float(fit.ols.betas[1][0]),
        "crime_beta_error": et["beta"], "crime_se_error": et["se"],
        "crime_p_vs_1_error": et["p_vs_1"],
    }
    table = fit.coefficient_table().set_index("variable")
    if "share_black_nh" in table.index:
        b = table.loc["share_black_nh", "error_beta"]
        se = table.loc["share_black_nh", "error_se"]
        row.update({
            "black_beta": b, "black_se": se, "black_z": b / se,
            "black_pct_per_10pp": 100 * (np.exp(0.1 * b) - 1),
        })
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--imputed-initiation", action="store_true")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 250)
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    annual = sp.load_annual(imputed=args.imputed_initiation)
    polys = sp.load_polygons()
    core = list(sp.DOWNTOWN_CORE)

    rows, coefficients = [], []
    for year in YEARS:
        full_d = sp.prepare_cross_section(annual, year=year)
        for variant, (spec, kind, drop_core) in VARIANTS.items():
            d, p = full_d, polys
            if drop_core:
                d = full_d[~full_d["neighborhood"].isin(core)].reset_index(drop=True)
                p = polys[~polys["neighborhood"].isin(core)]
            w = sp.build_weights(p, kind=kind)
            fit = sp.fit(d, w, x_names=sp.SPECS[spec], year=year)
            rows.append(_row(fit, year=year, variant=variant, spec=spec))
            if variant == "full":
                tbl = fit.coefficient_table()
                tbl.insert(0, "year", year)
                coefficients.append(tbl)

    diag = pd.DataFrame(rows)
    diag.to_csv(config.DATA_PROCESSED / "spatial_diagnostics.csv", index=False)
    pd.concat(coefficients, ignore_index=True).to_csv(
        config.DATA_PROCESSED / "spatial_coefficients.csv", index=False
    )

    record = diag[diag["variant"] == "full"]
    print(">> model of record: log stops ~ log crimes + log pop + share Black "
          "+ share renter + log area")
    print(record[["year", "moran_resid", "moran_resid_p", "recommended",
                  "ols_r2"]].round(3).to_string(index=False))

    print("\n>> does enforcement scale with crime? (elasticity; 1 = proportional)")
    wide = diag.pivot(index="year", columns="variant", values="crime_beta_error")
    print(wide[["calls (retracted)", "pop", "full"]].round(3).to_string())
    print("   With population in place of 911 calls the elasticity sits near 1.\n"
          "   The calls specification is a bad control -- calls are 0.92-correlated\n"
          "   with crime and largely caused by it -- and its sub-proportional\n"
          "   result is retracted.")

    print("\n>> share Black: extra stops per 10-point higher Black population share")
    pct = diag.pivot(index="year", columns="variant", values="black_pct_per_10pp")
    z = diag.pivot(index="year", columns="variant", values="black_z")
    cols = ["full", "full + income", "full, KNN weights", "full, no downtown"]
    shown = pct[cols].round(1).astype(str) + "% (z " + z[cols].round(1).astype(str) + ")"
    print(shown.to_string())
    robust = [
        y for y in YEARS
        if (z.loc[y, cols].abs() >= 1.96).all()
    ]
    print(f"\n   significant under every check in: {robust or 'no year'}")

    print("\n>> LISA clusters, 2025 (log stops)")
    w = sp.build_weights(polys)
    d = sp.prepare_cross_section(annual, year=2025)
    _, labels = sp.local_morans(d["log_stops_total"].to_numpy(), w)
    d["lisa"] = labels
    d[["neighborhood", "year", "stops_total", "crimes", "lisa"]].to_parquet(
        config.DATA_PROCESSED / "spatial_lisa.parquet", index=False
    )
    print(pd.Series(labels).value_counts().to_string())

    print("\n>> caveats")
    for line in (
        "cross-sectional associations, not causal effects",
        "demographics are ACS 2019-2023 estimates held fixed across years",
        "share Black and income correlate at -0.57; the '+ income' column is the "
        "check on whether race is standing in for income",
        f"downtown core ({', '.join(core)}) has a daytime population residents "
        "do not measure; the 'no downtown' column drops it",
        "2025 residuals remain strongly autocorrelated, so the 2025 fit is the "
        "weakest of the five years",
    ):
        print(f"   - {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
