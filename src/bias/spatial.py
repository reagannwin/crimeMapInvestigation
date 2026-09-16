"""Spatial dependence diagnostics and models for the neighbourhood panel.

Why this is not optional
-----------------------
Every result in this project so far is a neighbourhood-level statistic, and
neighbourhoods are not independent observations. Enforcement, crime and
reporting all spill across boundaries, so OLS residuals are correlated in
space and the model is misspecified.

The usual warning is that OLS standard errors will be too small. That is the
textbook consequence of an omitted spatial **lag**, and it is worth checking --
but these data do not reliably show it: the spatial error model's standard
errors are often *smaller* than OLS. The reason to model dependence here is
that residual Moran's I is significant in most specifications, so OLS is
misspecified and its point estimates cannot be taken at face value.

The question this module answers is therefore not "is there a spatial pattern"
-- of course there is -- but "does the finding survive being modelled
properly, and by how much does it move".

What it does
------------
1. Queen contiguity weights over the 87 neighbourhood polygons. The graph is
   fully connected (one component, no islands, 2-9 neighbours each), so no
   island correction is needed.
2. Global Moran's I on the outcome and, more importantly, on OLS *residuals*.
   Autocorrelated residuals are the diagnostic that OLS is inadequate.
3. Lagrange Multiplier tests to choose between a spatial **lag** and a spatial
   **error** specification rather than picking one by taste. The robust
   variants (RLMlag, RLMerr) are the ones to read when both simple tests are
   significant, which they usually are.
4. Both models fitted, with standard errors placed beside the OLS ones so the
   inflation is visible rather than asserted.

Specification
-------------
Log-log in the counts, so the crime coefficient reads as an elasticity:

    log(stops) ~ log(crimes) + log(pop) + share_black + share_renter + log(area)

``b_crime = 1`` means enforcement scales proportionally with recorded crime.
``share_black`` asks the disproportionality question directly: conditional on
crime, population, tenure and size, do neighbourhoods with a larger Black
population receive more stops? Its coefficient is a semi-elasticity; a value
of ``b`` means roughly ``exp(0.1 * b) - 1`` more stops per 10-point higher
Black share.

Why 911 calls are NOT a control
-------------------------------
An earlier version, written before the ACS data was available, used
``log(calls)`` in place of population. That was a bad control. Calls are
largely a *consequence* of crime and correlate with it at 0.92, so the model
attributed most of crime's effect to calls and reported the leftover as the
crime elasticity: 0.34-0.78, read as "enforcement is sub-proportional to
crime". Replacing calls with population puts the elasticity at 0.97-1.07 in
every year 2021-2024. Enforcement tracks recorded crime proportionally; the
earlier finding was an artefact and is retracted. The calls specification is
kept in :data:`SPECS` only so the artefact stays reproducible.

Remaining limits: demographics are ACS 2019-2023 estimates held fixed across
years; share_black and income correlate at -0.57, so the ``full_income``
specification is the check on whether race is proxying for income; and the
downtown core has a daytime population that residents do not measure, which
the ``drop_core`` option tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
import pandas as pd

from src import config

DEFAULT_YEAR = 2025
# Log transform needs a floor; every count here is a rate numerator where a
# structural zero is meaningful, so shift rather than drop.
LOG_OFFSET = 1.0
PERMUTATIONS = 999
SEED = 0

# Named specifications. ``full`` is the model of record.
SPECS: dict[str, tuple[str, ...]] = {
    # Retracted: calls are a bad control (see module docstring).
    "calls": ("log_crimes", "log_calls", "log_area_km2"),
    "pop": ("log_crimes", "log_pop", "log_area_km2"),
    "full": ("log_crimes", "log_pop", "share_black_nh", "share_renter",
             "log_area_km2"),
    # Checks whether share_black is standing in for income.
    "full_income": ("log_crimes", "log_pop", "share_black_nh", "share_renter",
                    "log_area_km2", "log_income"),
}
MODEL_OF_RECORD = "full"

# Commercial core: residents badly measure the population policed here.
DOWNTOWN_CORE = ("Downtown West", "Downtown East", "North Loop")


@dataclass
class SpatialFit:
    """OLS alongside its spatial alternatives, with the diagnostics that chose them."""

    year: int
    n: int
    y_name: str
    x_names: list[str]
    ols: object
    lag: object | None
    error: object | None
    moran_y: tuple[float, float]
    moran_resid: tuple[float, float]
    lm_tests: dict[str, tuple[float, float]] = field(default_factory=dict)

    @property
    def recommended(self) -> str:
        """Which specification the robust LM tests point to."""
        rlag = self.lm_tests.get("robust_lm_lag", (np.nan, 1.0))
        rerr = self.lm_tests.get("robust_lm_error", (np.nan, 1.0))
        if rlag[1] >= 0.05 and rerr[1] >= 0.05:
            return "ols"
        return "lag" if rlag[1] < rerr[1] else "error"

    def elasticity_test(self, variable: str = "log_crimes") -> pd.DataFrame:
        """Test each model's elasticity against 1, not against 0.

        The interesting null here is proportionality. "Enforcement rises with
        crime" (beta > 0) is uncontroversial and tells us nothing; the question
        is whether it rises *proportionally* (beta = 1), less than
        proportionally (beta < 1, enforcement spread more evenly than crime) or
        more (beta > 1, concentrated beyond crime). Reporting only the
        beta = 0 test would answer a question nobody asked.
        """
        from scipy import stats

        idx = (["CONSTANT"] + self.x_names).index(variable)
        rows = []
        for label, model in (("ols", self.ols), ("lag", self.lag), ("error", self.error)):
            if model is None:
                continue
            beta = float(model.betas[idx][0])
            se = float(np.sqrt(model.vm.diagonal()[idx]))
            z = (beta - 1.0) / se
            rows.append({
                "model": label,
                "beta": beta,
                "se": se,
                "ci_lo": beta - 1.96 * se,
                "ci_hi": beta + 1.96 * se,
                "z_vs_1": z,
                "p_vs_1": 2 * (1 - stats.norm.cdf(abs(z))),
                "proportional_at_05": abs(z) < 1.96,
            })
        return pd.DataFrame(rows)

    def coefficient_table(self) -> pd.DataFrame:
        """OLS and spatial coefficients side by side, with SE inflation."""
        rows = []
        names = ["CONSTANT"] + self.x_names
        for i, name in enumerate(names):
            row = {
                "variable": name,
                "ols_beta": float(self.ols.betas[i][0]),
                "ols_se": float(np.sqrt(self.ols.vm.diagonal()[i])),
            }
            for label, model in (("lag", self.lag), ("error", self.error)):
                if model is None:
                    continue
                row[f"{label}_beta"] = float(model.betas[i][0])
                row[f"{label}_se"] = float(np.sqrt(model.vm.diagonal()[i]))
            rows.append(row)

        out = pd.DataFrame(rows)
        chosen = self.recommended
        if chosen != "ols" and f"{chosen}_se" in out.columns:
            out["se_inflation"] = out[f"{chosen}_se"] / out["ols_se"]
        return out


def build_weights(polys: gpd.GeoDataFrame, *, kind: str = "queen", k: int = 5):
    """Row-standardised spatial weights over neighbourhood polygons.

    Queen contiguity is the default because neighbourhood adjacency is the
    relevant notion of "nearby" here. KNN is offered as a robustness check:
    a finding that only holds under one weights definition is a finding about
    the weights.
    """
    from libpysal.weights import KNN, Queen

    polys = polys.sort_values("neighborhood").reset_index(drop=True)
    if kind == "queen":
        w = Queen.from_dataframe(
            polys, ids=polys["neighborhood"].tolist(), use_index=False
        )
    elif kind == "knn":
        w = KNN.from_dataframe(polys, k=k, ids=polys["neighborhood"].tolist())
    else:
        raise ValueError(f"unknown weights kind: {kind!r}")

    if w.islands:
        raise ValueError(
            f"{len(w.islands)} island(s) with no neighbours: {w.islands}. "
            "Spatial models cannot use them; switch to KNN weights."
        )
    w.transform = "r"
    return w


def morans_i(values: np.ndarray, w, *, permutations: int = PERMUTATIONS) -> tuple[float, float]:
    """Global Moran's I with a permutation p-value.

    The permutation inference is used rather than the normal approximation
    because n is 87 and the analytic p-value is unreliable at that size.
    """
    from esda.moran import Moran

    mi = Moran(np.asarray(values, dtype=float), w, permutations=permutations)
    return float(mi.I), float(mi.p_sim)


def local_morans(values: np.ndarray, w, *, permutations: int = PERMUTATIONS):
    """LISA cluster labels: which neighbourhoods drive the global statistic."""
    from esda.moran import Moran_Local

    lisa = Moran_Local(
        np.asarray(values, dtype=float), w, permutations=permutations, seed=SEED
    )
    labels = np.array(["ns", "HH", "LH", "LL", "HL"])[lisa.q * (lisa.p_sim < 0.05)]
    return lisa, labels


def prepare_cross_section(
    annual: pd.DataFrame, *, year: int = DEFAULT_YEAR
) -> pd.DataFrame:
    """One row per neighbourhood for a single year, with logged variables."""
    d = annual[annual["year"] == year].copy()
    for col in ("stops_total", "crimes", "calls"):
        d[f"log_{col}"] = np.log(d[col].astype(float) + LOG_OFFSET)
    return d.sort_values("neighborhood").reset_index(drop=True)


def fit(
    df: pd.DataFrame,
    w,
    *,
    y_name: str = "log_stops_total",
    x_names: tuple[str, ...] = SPECS[MODEL_OF_RECORD],
    year: int = DEFAULT_YEAR,
) -> SpatialFit:
    """Fit OLS, run spatial diagnostics, then fit lag and error models."""
    from spreg import GM_Error_Het, GM_Lag, OLS

    y = df[[y_name]].to_numpy(dtype=float)
    X = df[list(x_names)].to_numpy(dtype=float)

    ols = OLS(
        y, X, w=w, spat_diag=True, moran=True,
        name_y=y_name, name_x=list(x_names), name_w="queen",
    )
    lm_tests = {
        "lm_lag": tuple(ols.lm_lag),
        "lm_error": tuple(ols.lm_error),
        "robust_lm_lag": tuple(ols.rlm_lag),
        "robust_lm_error": tuple(ols.rlm_error),
    }

    lag = GM_Lag(y, X, w=w, name_y=y_name, name_x=list(x_names), name_w="queen")
    error = GM_Error_Het(
        y, X, w=w, name_y=y_name, name_x=list(x_names), name_w="queen"
    )

    return SpatialFit(
        year=year, n=len(df), y_name=y_name, x_names=list(x_names),
        ols=ols, lag=lag, error=error,
        moran_y=morans_i(df[y_name].to_numpy(), w),
        moran_resid=morans_i(ols.u.flatten(), w),
        lm_tests=lm_tests,
    )


DEMOGRAPHIC_COLS = (
    "pop_total", "share_black_nh", "share_white_nh", "share_renter",
    "median_hh_income_approx",
)


def load_annual(*, imputed: bool = False) -> pd.DataFrame:
    """Annual panel joined to polygon area and ACS demographics.

    The demographics are ACS 2019-2023 estimates, so the same values attach to
    every year of the panel.
    """
    suffix = "_imputed" if imputed else ""
    annual = pd.read_parquet(
        config.DATA_PROCESSED / f"enforcement_annual{suffix}.parquet"
    )
    polys = gpd.read_parquet(config.DATA_INTERIM / "neighborhood_polygons.parquet")
    area = polys.assign(area_km2=polys.geometry.area / 1e6)[
        ["neighborhood", "area_km2"]
    ]
    out = annual.merge(area, on="neighborhood", how="left")
    out["log_area_km2"] = np.log(out["area_km2"])

    demo_path = config.DATA_PROCESSED / "neighborhood_demographics.parquet"
    if not demo_path.exists():
        raise FileNotFoundError(
            f"{demo_path.name} is missing; run src.ingest.build_census first"
        )
    demo = pd.read_parquet(demo_path, columns=["neighborhood", *DEMOGRAPHIC_COLS])
    # The annual panel may already carry these (build_panel adds them); take
    # them from the demographics file only, never merge them twice.
    out = out.drop(columns=[c for c in DEMOGRAPHIC_COLS if c in out.columns])
    out = out.merge(demo, on="neighborhood", how="left")
    if out["pop_total"].isna().any():
        missing = sorted(out.loc[out["pop_total"].isna(), "neighborhood"].unique())
        raise ValueError(f"no demographics for: {missing}")
    out["log_pop"] = np.log(out["pop_total"])
    out["log_income"] = np.log(out["median_hh_income_approx"])
    return out


def load_polygons() -> gpd.GeoDataFrame:
    return gpd.read_parquet(config.DATA_INTERIM / "neighborhood_polygons.parquet")
