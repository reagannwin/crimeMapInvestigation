"""Choropleths and the ShotSpotter coverage map."""
from __future__ import annotations

import geopandas as gpd
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src import config
from src.viz import style

# Neighbourhoods worth naming on a map. Labelling all 87 is unreadable; these
# are the ones the findings actually turn on.
LANDMARKS = (
    "Jordan", "Hawthorne", "Near - North", "Downtown West", "Whittier",
    "Powderhorn Park", "Lind - Bohanon", "Ventura Village",
)


def _label(ax, gdf: gpd.GeoDataFrame, names, *, size=7.5, color=None):
    for _, row in gdf[gdf["neighborhood"].isin(names)].iterrows():
        point = row.geometry.representative_point()
        ax.annotate(
            row["neighborhood"], (point.x, point.y),
            ha="center", va="center", fontsize=size,
            color=color or style.INK,
            path_effects=None,
            bbox=dict(boxstyle="round,pad=0.18", fc=style.PAPER,
                      ec="none", alpha=0.72),
        )


def coverage_map(polys: gpd.GeoDataFrame, coverage: pd.DataFrame, footprint):
    """Where ShotSpotter actually is, and what that excludes.

    The point of the figure is the exclusion: three quarters of the city
    cannot enter any sensor-versus-human comparison, and one neighbourhood with
    the fifth-highest confirmed shooting count has essentially no coverage.
    """
    gdf = polys.merge(coverage, on="neighborhood", how="left")
    fig, ax = style.figure(7.4, 9.6)

    fills = {
        "covered": style.CATEGORICAL[0],
        "partial": "#8fb3d1",
        "uncovered": style.NO_DATA,
    }
    for cls, colour in fills.items():
        subset = gdf[gdf["coverage_class"] == cls]
        if len(subset):
            subset.plot(ax=ax, color=colour, edgecolor="white", linewidth=0.6)

    downgraded = gdf[gdf["evidence_downgraded"].fillna(False)]
    if len(downgraded):
        downgraded.plot(ax=ax, color=style.NO_DATA, edgecolor=style.ACCENT,
                        linewidth=1.1, hatch="///", alpha=0.95)

    gpd.GeoSeries([footprint], crs=polys.crs).boundary.plot(
        ax=ax, color=style.ACCENT, linewidth=1.8, linestyle="--", zorder=5
    )

    style.strip_axes(ax)
    style.fit_bounds(ax, gdf)
    style.fig_titled(
        fig, "ShotSpotter covers a fifth of Minneapolis",
        "20 of 87 neighbourhoods hold 97.6% of activations.\n"
        "Only those are valid for sensor-vs-community comparison.",
    )
    _label(ax, gdf, LANDMARKS)

    dw = gdf[gdf["neighborhood"] == "Downtown West"]
    if len(dw):
        point = dw.geometry.iloc[0].representative_point()
        ax.annotate(
            "Downtown West\n293 confirmed shootings\n2 activations in 19 years",
            xy=(point.x, point.y), xytext=(point.x + 2900, point.y - 3400),
            fontsize=8.5, color=style.ACCENT, ha="left",
            arrowprops=dict(arrowstyle="-", color=style.ACCENT, lw=1.0),
            bbox=dict(boxstyle="round,pad=0.35", fc=style.PAPER,
                      ec=style.ACCENT, lw=0.8),
        )

    handles = [
        mpatches.Patch(facecolor=fills["covered"], label="Covered (>=60% of area)"),
        mpatches.Patch(facecolor=fills["partial"], label="Partial"),
        mpatches.Patch(facecolor=style.NO_DATA, label="Uncovered"),
        mpatches.Patch(facecolor=style.NO_DATA, hatch="///", edgecolor=style.ACCENT,
                       label="Downgraded: <50 activations"),
        Line2D([0], [0], color=style.ACCENT, lw=1.8, ls="--",
               label="Derived sensor footprint"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.04, 0.045),
              ncol=2, columnspacing=1.1, handlelength=1.6)
    fig.subplots_adjust(top=0.90, bottom=0.10)
    style.source_note(fig, "Footprint derived from 37,896 activations, 2007-2026")
    return fig


def choropleth(
    polys: gpd.GeoDataFrame,
    values: pd.DataFrame,
    column: str,
    *,
    title: str,
    subtitle: str | None = None,
    legend_label: str = "",
    cmap: str = style.RAMP_RATE,
    note: str | None = None,
    label_extremes: int = 3,
    fmt: str = "{:.0%}",
):
    """Generic neighbourhood choropleth with explicit no-data treatment.

    Polygons without a value are hatched rather than left white, so a reader
    cannot mistake "not measured here" for "low value here" -- the single most
    common misreading of a choropleth.
    """
    gdf = polys.merge(values[["neighborhood", column]], on="neighborhood", how="left")
    fig, ax = style.figure(7.4, 9.6)

    missing = gdf[gdf[column].isna()]
    if len(missing):
        missing.plot(ax=ax, color=style.NO_DATA, edgecolor="white", linewidth=0.6,
                     hatch=style.NO_DATA_HATCH)

    present = gdf[gdf[column].notna()]
    present.plot(
        ax=ax, column=column, cmap=cmap, edgecolor="white", linewidth=0.6,
        legend=True,
        legend_kwds={"label": legend_label, "shrink": 0.5, "pad": 0.01,
                     "orientation": "vertical"},
    )
    # Keep the colourbar in the same units as the polygon labels: showing
    # "0.24" on the bar next to "24%" on the polygon reads as two quantities.
    if fmt.endswith("%}") and fig.axes:
        from matplotlib.ticker import FuncFormatter

        fig.axes[-1].yaxis.set_major_formatter(
            FuncFormatter(lambda v, _: fmt.format(v))
        )

    style.strip_axes(ax)
    style.fit_bounds(ax, gdf)
    style.fig_titled(fig, title, subtitle)

    if label_extremes and len(present) >= 2 * label_extremes:
        ordered = present.sort_values(column)
        for _, row in pd.concat(
            [ordered.head(label_extremes), ordered.tail(label_extremes)]
        ).iterrows():
            point = row.geometry.representative_point()
            ax.annotate(
                f"{row['neighborhood']}\n{fmt.format(row[column])}",
                (point.x, point.y), ha="center", va="center", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.2", fc=style.PAPER,
                          ec="none", alpha=0.78),
            )

    if len(missing):
        ax.legend(
            handles=[mpatches.Patch(facecolor=style.NO_DATA,
                                    hatch=style.NO_DATA_HATCH,
                                    label="No data / outside scope")],
            loc="upper left", bbox_to_anchor=(-0.04, 0.045),
        )
    fig.subplots_adjust(top=0.90, bottom=0.08)
    style.source_note(fig, note)
    return fig


def lisa_map(polys: gpd.GeoDataFrame, lisa: pd.DataFrame, *, year: int):
    """Local Moran clusters: where enforcement clusters with its neighbours."""
    gdf = polys.merge(lisa[["neighborhood", "lisa"]], on="neighborhood", how="left")
    gdf["lisa"] = gdf["lisa"].fillna("ns")

    fig, ax = style.figure(7.4, 9.6)
    for tag, colour in style.LISA_COLORS.items():
        subset = gdf[gdf["lisa"] == tag]
        if len(subset):
            subset.plot(ax=ax, color=colour, edgecolor="white", linewidth=0.6)

    style.strip_axes(ax)
    style.fit_bounds(ax, gdf)
    style.fig_titled(
        fig, f"Enforcement clusters in space, {year}",
        "Local Moran's I on log stops.\n"
        "Clustering is why OLS is misspecified for these data.",
    )
    labels = {
        "HH": "High, among high (21)",
        "LL": "Low, among low (10)",
        "LH": "Low, among high",
        "HL": "High, among low",
        "ns": "Not significant",
    }
    present = [t for t in ("HH", "LL", "LH", "HL", "ns") if (gdf["lisa"] == t).any()]
    ax.legend(
        handles=[mpatches.Patch(facecolor=style.LISA_COLORS[t], label=labels[t])
                 for t in present],
        loc="upper left", bbox_to_anchor=(-0.04, 0.05), ncol=2,
        columnspacing=1.1, title="Cluster type",
    )
    fig.subplots_adjust(top=0.90, bottom=0.09)
    style.source_note(fig, "Queen contiguity, 999 permutations, p < 0.05")
    return fig


def load_geo() -> gpd.GeoDataFrame:
    return gpd.read_parquet(
        config.DATA_INTERIM / "neighborhood_polygons.parquet"
    )[["neighborhood", "geometry"]]
