"""Spatial dependence diagnostics and models for the neighbourhood panel.

Why this is not optional
-----------------------
Every result in this project so far is a neighbourhood-level statistic, and
neighbourhoods are not independent observations. Enforcement, crime and
reporting all spill across boundaries, so OLS residuals are correlated in
space and the model is misspecified.

The usual warning is that OLS standard errors will be too small. That is the
textbook consequence of an omitted spatial **lag**, and it is worth checking --
but it is not what happens here. Fitting these data, the spatial error model's
standard errors come out *smaller* than OLS, not larger. The damage shows up
instead in the point estimates: the crime elasticity falls from 1.09 under OLS
to 0.66 once spatial dependence is modelled. So the reason to do this is not a
reliable direction of bias in the standard errors; it is that OLS is answering
the wrong question when residual Moran's I is 0.62.

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
Log-log, so coefficients read as elasticities:

    log(stops) ~ log(crimes) + log(calls) + log(area)

``b_crime = 1`` means enforcement scales proportionally with recorded crime.
Below 1 means enforcement is spread more evenly than crime; above 1 means it is
concentrated more than crime. That is the project's question stated as a
parameter.

Population is deliberately absent -- the ACS pull is still blocked -- so
``calls`` carries the demand signal and ``area`` absorbs size. Read the crime
elasticity as "conditional on demand and size", not as a causal effect.
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
    x_names: tuple[str, ...] = ("log_crimes", "log_calls", "log_area_km2"),
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


def load_annual(*, imputed: bool = False) -> pd.DataFrame:
    """Annual panel joined to polygon area, ready for the cross-section."""
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
    return out


def load_polygons() -> gpd.GeoDataFrame:
    return gpd.read_parquet(config.DATA_INTERIM / "neighborhood_polygons.parquet")
