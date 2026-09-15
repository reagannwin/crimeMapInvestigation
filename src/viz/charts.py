"""Non-map figures: trends, hit rates, elasticities."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.viz import style


def enforcement_trend(citywide: pd.DataFrame, *, imputed: bool = False):
    """Indexed series: crime down, demand flat, discretionary enforcement up.

    Indexed to the base year because the raw quantities differ by an order of
    magnitude (210,000 calls against 21,000 crimes); on a shared axis the
    smaller series would be a flat line at the bottom regardless of what it did.
    """
    base = citywide.index.min()
    series = {
        "Proactive stops": ("stops_proactive", style.ACCENT, 2.4),
        "Total stops": ("stops_total", style.CATEGORICAL[0], 1.8),
        "911 calls": ("calls", style.CATEGORICAL[5], 1.6),
        "Reported crime": ("crimes", style.CATEGORICAL[2], 1.8),
        "Reactive stops": ("stops_reactive", style.CATEGORICAL[1], 1.8),
    }

    fig, ax = style.figure(9.5, 6.0)
    ax.axhline(100, color=style.MUTED, lw=1.0, ls=":", zorder=1)

    for label, (col, colour, lw) in series.items():
        indexed = 100 * citywide[col] / citywide.loc[base, col]
        ax.plot(citywide.index, indexed, color=colour, lw=lw,
                marker="o", ms=4.5, label=label, zorder=3)
        ax.annotate(
            f"{label}  {indexed.iloc[-1]:.0f}",
            (citywide.index[-1], indexed.iloc[-1]),
            xytext=(8, 0), textcoords="offset points",
            fontsize=style.LABEL_SIZE, color=colour, va="center", weight="600",
        )

    ax.set_xticks(list(citywide.index))
    ax.set_xlim(citywide.index.min() - 0.15, citywide.index.max() + 1.35)
    ax.set_ylabel(f"Index ({base} = 100)")
    style.despine(ax)
    style.titled(
        ax, "Enforcement rose as crime fell",
        f"Officer-initiated stops {'+105.8' if imputed else '+82.3'}% while "
        "reported crime fell 5.4% and 911 calls stayed flat.",
    )
    style.source_note(
        fig,
        f"Initiation: {'imputed via 911 join' if imputed else 'observed field only'}",
    )
    return fig


def hit_rate_chart(table: pd.DataFrame, by_initiation: pd.DataFrame):
    """The outcome test, and the stratification that locates its mechanism."""
    fig, axes = style.figure(11.0, 5.6)
    fig.clf()
    ax1, ax2 = fig.subplots(1, 2, width_ratios=[1.25, 1])

    # Left: hit rate with Wilson intervals, ordered by search rate.
    d = table.sort_values("search_rate")
    y = np.arange(len(d))
    colours = [style.ACCENT if r == "Black" else style.CATEGORICAL[0]
               for r in d.index]

    ax1.hlines(y, d["hit_lo"] * 100, d["hit_hi"] * 100,
               color=colours, lw=2.4, alpha=0.45)
    ax1.scatter(d["hit_rate"] * 100, y, color=colours, s=62, zorder=3)
    for i, (name, row) in enumerate(d.iterrows()):
        ax1.annotate(f"{row['search_rate']:.1%} searched",
                     (row["hit_rate"] * 100, i), xytext=(0, 11),
                     textcoords="offset points", ha="center",
                     fontsize=7.8, color=style.MUTED)

    ax1.set_yticks(y)
    ax1.set_yticklabels(d.index)
    ax1.set_xlabel("Arrests per 100 searches")
    ax1.grid(axis="y", visible=False)
    style.despine(ax1)
    style.titled(
        ax1, "Searched more, found less",
        "Arrest rate given a search, 2024-2026, with 95% Wilson intervals.",
    )

    # Right: the stratification.
    strata = by_initiation.loc[["proactive", "reactive"], ["Black", "White"]] * 100
    x = np.arange(len(strata))
    width = 0.34
    ax2.bar(x - width / 2, strata["Black"], width,
            color=style.ACCENT, label="Black")
    ax2.bar(x + width / 2, strata["White"], width,
            color=style.CATEGORICAL[0], label="White")
    for i, idx in enumerate(strata.index):
        gap = strata.loc[idx, "Black"] - strata.loc[idx, "White"]
        ax2.annotate(
            f"{gap:+.1f} pp", (i, max(strata.loc[idx]) + 2.2),
            ha="center", fontsize=style.LABEL_SIZE, weight="600",
            color=style.ACCENT if gap < 0 else style.CATEGORICAL[2],
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(["Officer-initiated", "911-dispatched"])
    ax2.set_ylabel("Arrests per 100 searches")
    ax2.set_ylim(0, max(strata.max()) + 12)
    ax2.grid(axis="x", visible=False)
    # Inside the axes: an outside legend is clipped by the tight bbox.
    ax2.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.97))
    style.despine(ax2)
    style.titled(
        ax2, "The gap is in discretionary stops",
        "It reverses where a 911 call sent the officer.",
    )

    fig.subplots_adjust(wspace=0.30, top=0.84, bottom=0.14)
    style.source_note(fig, "Hit = booking. Recorded race. Person searches.")
    return fig


def elasticity_chart(diagnostics: pd.DataFrame):
    """Crime elasticity by year against the proportionality line."""
    d = diagnostics[diagnostics["weights"] == "queen"].set_index("year")

    fig, ax = style.figure(9.0, 5.6)
    ax.axhline(1.0, color=style.MUTED, lw=1.2, ls="--", zorder=1)
    ax.annotate("Proportional (elasticity = 1)", (d.index.min() - 0.12, 1.03),
                fontsize=style.LABEL_SIZE, color=style.MUTED, va="bottom")

    ax.plot(d.index, d["crime_beta_ols"], color=style.CATEGORICAL[5],
            lw=1.6, marker="s", ms=6, ls=":", label="OLS (misspecified)")
    ax.plot(d.index, d["crime_beta_error"], color=style.CATEGORICAL[0],
            lw=2.4, marker="o", ms=7, label="Spatial error model")

    outlier = d.index.max()
    ax.scatter([outlier], [d.loc[outlier, "crime_beta_ols"]],
               s=190, facecolor="none", edgecolor=style.ACCENT, lw=1.8, zorder=4)
    ax.annotate(
        "2025: model breaks down\nR² 0.46, residual Moran's I 0.62",
        (outlier, d.loc[outlier, "crime_beta_ols"]),
        xytext=(-14, 26), textcoords="offset points", ha="right",
        fontsize=8.5, color=style.ACCENT,
        arrowprops=dict(arrowstyle="-", color=style.ACCENT, lw=1.0),
    )

    ax.set_xticks(list(d.index))
    ax.set_ylabel("Crime elasticity of stops")
    ax.set_ylim(0, 1.45)
    style.despine(ax)
    style.titled(
        ax, "Enforcement does not scale with crime",
        "Below 1 means a neighbourhood with twice the crime gets less than "
        "twice the enforcement.",
    )
    ax.legend(loc="lower right")
    style.source_note(fig, "log stops ~ log crimes + log calls + log area; Queen weights")
    return fig


def reporting_rate_chart(rates: pd.DataFrame):
    """Reporting propensity across the covered neighbourhoods."""
    d = rates.sort_values("report_rate")
    fig, ax = style.figure(8.6, 7.4)

    y = np.arange(len(d))
    colours = [
        style.ACCENT if v <= d["report_rate"].quantile(0.25)
        else style.CATEGORICAL[0] for v in d["report_rate"]
    ]
    ax.barh(y, d["report_rate"] * 100, color=colours, height=0.68)
    for i, (_, row) in enumerate(d.iterrows()):
        ax.annotate(f"{row['report_rate']:.1%}",
                    (row["report_rate"] * 100, i), xytext=(5, 0),
                    textcoords="offset points", va="center",
                    fontsize=8.2, color=style.MUTED)

    ax.set_yticks(y)
    ax.set_yticklabels(d["neighborhood"])
    ax.set_xlabel("Share of sensor-detected gunfire that also drew a 911 call")
    ax.set_xlim(0, d["report_rate"].max() * 100 + 5)
    ax.grid(axis="y", visible=False)
    style.despine(ax)
    style.titled(
        ax, "Who calls when the sensors hear gunfire",
        "Same objective denominator, same footprint: a 4x spread in "
        "propensity to report.",
    )
    style.source_note(fig, "Sensor-covered neighbourhoods with >=20 sensor events")
    return fig
