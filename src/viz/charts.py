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
    """Crime elasticity by year, model of record against the retracted spec.

    The retracted line stays on the chart deliberately: it shows the reader
    exactly how far a bad control moved the estimate.
    """
    d = diagnostics.set_index(["variant", "year"])
    full = d.loc["full"]
    calls = d.loc["calls (retracted)"]

    fig, ax = style.figure(9.0, 5.6)
    ax.axhline(1.0, color=style.MUTED, lw=1.2, ls="--", zorder=1)
    # Right end, where the model-of-record line has dropped below 1 and the
    # label cannot collide with it.
    ax.annotate("Proportional (elasticity = 1)", (full.index.max() + 0.15, 1.02),
                fontsize=style.LABEL_SIZE, color=style.MUTED,
                ha="right", va="bottom")

    lo = full["crime_beta_error"] - 1.96 * full["crime_se_error"]
    hi = full["crime_beta_error"] + 1.96 * full["crime_se_error"]
    ax.fill_between(full.index, lo, hi, color=style.CATEGORICAL[0], alpha=0.12,
                    lw=0, label="95% interval")
    ax.plot(full.index, full["crime_beta_error"], color=style.CATEGORICAL[0],
            lw=2.4, marker="o", ms=7,
            label="Controls: population, race, tenure, area")
    ax.plot(calls.index, calls["crime_beta_error"], color=style.MUTED,
            lw=1.4, marker="s", ms=5, ls=":",
            label="Retracted: 911 calls as a control")

    last = full.index.max()
    last_hi = hi.loc[last]
    # Say whether the final-year dip is distinguishable from 1, rather than
    # letting a falling line imply a break the interval does not support.
    verdict = "interval still reaches 1" if last_hi >= 1 else "below 1"
    ax.annotate(
        f"{last}: {full.loc[last, 'crime_beta_error']:.2f}, {verdict}",
        (last, full.loc[last, "crime_beta_error"]),
        xytext=(-70, 26), textcoords="offset points", ha="right",
        fontsize=8.5, color=style.ACCENT,
        arrowprops=dict(arrowstyle="-", color=style.ACCENT, lw=1.0),
    )

    ax.set_xticks(list(full.index))
    ax.set_ylabel("Crime elasticity of stops")
    ax.set_ylim(0, 1.6)
    style.despine(ax)
    style.titled(
        ax, "Enforcement tracked recorded crime through 2024",
        "Elasticity near 1. The dotted line is an earlier model that used 911 "
        "calls as a control and was retracted.",
    )
    ax.legend(loc="lower left")
    style.source_note(fig, "Spatial error model, Queen weights")
    return fig


def black_share_chart(diagnostics: pd.DataFrame):
    """The Black-share effect by year under each robustness check."""
    variants = {
        "full": ("Model of record", style.CATEGORICAL[0], "o"),
        "full + income": ("+ income", style.CATEGORICAL[1], "s"),
        "full, KNN weights": ("KNN weights", style.CATEGORICAL[2], "^"),
        "full, no downtown": ("Downtown excluded", style.CATEGORICAL[3], "D"),
    }
    fig, ax = style.figure(9.5, 5.8)
    ax.axhline(0, color=style.MUTED, lw=1.0)

    offsets = np.linspace(-0.24, 0.24, len(variants))
    for offset, (key, (label, colour, marker)) in zip(offsets, variants.items()):
        d = diagnostics[diagnostics["variant"] == key].set_index("year")
        pct = 100 * (np.exp(0.1 * d["black_beta"]) - 1)
        lo = 100 * (np.exp(0.1 * (d["black_beta"] - 1.96 * d["black_se"])) - 1)
        hi = 100 * (np.exp(0.1 * (d["black_beta"] + 1.96 * d["black_se"])) - 1)
        x = d.index + offset
        ax.vlines(x, lo, hi, color=colour, lw=2.0, alpha=0.55)
        ax.scatter(x, pct, color=colour, marker=marker, s=46, zorder=3,
                   label=label)

    years = sorted(diagnostics["year"].unique())
    ax.set_xticks(years)
    ax.set_ylabel("Extra stops per 10-point higher Black share (%)")
    style.despine(ax)
    style.titled(
        ax, "More stops where more Black residents live",
        "Holding recorded crime, population, renter share and area fixed. "
        "Significant under every check only in 2025.",
    )
    ax.legend(loc="upper left", ncol=2)
    style.source_note(fig, "Spatial error model; bars are 95% intervals")
    return fig


def per_capita_bands_chart(annual: pd.DataFrame, *, year: int):
    """Stops and crimes per resident across Black-share quarters."""
    d = annual[(annual["year"] == year) & ~annual["daytime_core"]].copy()
    d["band"] = pd.qcut(d["share_black_nh"], 4,
                        labels=["Lowest", "Second", "Third", "Highest"])
    g = d.groupby("band", observed=True).agg(
        share=("share_black_nh", "mean"), pop=("pop_total", "sum"),
        stops=("stops_total", "sum"), crimes=("crimes", "sum"),
    )
    g["stops_1k"] = 1000 * g["stops"] / g["pop"]
    g["crimes_1k"] = 1000 * g["crimes"] / g["pop"]

    fig, ax = style.figure(9.0, 5.6)
    x = np.arange(len(g))
    width = 0.36
    ax.bar(x - width / 2, g["crimes_1k"], width, color=style.CATEGORICAL[2],
           label="Recorded crimes")
    ax.bar(x + width / 2, g["stops_1k"], width, color=style.ACCENT,
           label="Police stops")
    for i, (_, row) in enumerate(g.iterrows()):
        ax.annotate(f"{row['stops_1k']:.0f}", (i + width / 2, row["stops_1k"]),
                    xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=style.LABEL_SIZE, color=style.ACCENT, weight="600")
        ax.annotate(f"{row['crimes_1k']:.0f}", (i - width / 2, row["crimes_1k"]),
                    xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=style.LABEL_SIZE, color=style.CATEGORICAL[2])

    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n{s:.0%} Black" for b, s in zip(g.index, g["share"])])
    ax.set_ylabel("Per 1,000 residents")
    ax.set_ylim(0, g[["stops_1k", "crimes_1k"]].to_numpy().max() * 1.18)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    style.despine(ax)
    # Compute the headline from the data rather than asserting it, so the
    # title stays true if the year or the inputs change.
    top, third = g.iloc[-1], g.iloc[-2]
    extra_stops = 100 * (top["stops_1k"] / third["stops_1k"] - 1)
    crime_gap = 100 * abs(top["crimes_1k"] / third["crimes_1k"] - 1)
    if crime_gap < 5:
        title = f"Same crime rate, {extra_stops:.0f}% more stops ({year})"
        subtitle = ("Neighbourhoods grouped into quarters by Black population "
                    "share. The top two quarters record the same crime per "
                    "resident.")
    else:
        title = f"Stops and crime per resident by Black population share ({year})"
        subtitle = (f"Top quarter vs third: {extra_stops:+.0f}% stops, "
                    f"{crime_gap:.0f}% difference in recorded crime.")
    style.titled(ax, title, subtitle)
    style.source_note(fig, "Downtown core excluded; ACS 2019-2023 population")
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
