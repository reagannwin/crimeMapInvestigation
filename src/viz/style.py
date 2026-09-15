"""Shared visual language for every figure in the project.

One palette, one type scale, one set of map conventions, so a reader moving
between the coverage map and the enforcement chart is not relearning the
encoding each time.

Colour choices are deliberate rather than defaults:

  * Sequential ramps are perceptually uniform and colour-vision-safe, because
    several figures encode a rate that readers will compare across polygons.
  * The categorical palette avoids red/green opposition for the same reason.
  * Grey is reserved. On the coverage map it means "no sensor evidence", and
    on the LISA map "not significant". A reader should be able to learn that
    once.
  * Red is reserved for the thing that needs attention -- an anomaly call-out,
    or the high end of an enforcement ramp -- and is not used decoratively.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

INK = "#1d2021"
MUTED = "#6b6f72"
HAIRLINE = "#d6d3ce"
PAPER = "#faf8f5"
GRID = "#e8e4df"

# Categorical: distinguishable in greyscale and under common colour-vision
# deficiencies. Ordered by how often they are needed.
CATEGORICAL = [
    "#1f4e79",  # deep blue
    "#c0532b",  # burnt orange
    "#4a7c59",  # forest
    "#8b5a9f",  # violet
    "#b8912f",  # ochre
    "#4a6d8c",  # slate
]

ACCENT = "#c0392b"      # anomalies and call-outs only
POSITIVE = "#1f4e79"
NEGATIVE = "#c0532b"

# Map ramps.
RAMP_RATE = "YlOrRd"        # rates where high means more
RAMP_NEUTRAL = "viridis"
RAMP_DIVERGING = "RdBu_r"   # change from a meaningful midpoint

NO_DATA = "#e5e1db"
NO_DATA_HATCH = "///"

# LISA cluster colours. Shared with the coverage map's grey convention.
LISA_COLORS = {
    "HH": "#b2182b",
    "LL": "#2166ac",
    "HL": "#ef8a62",
    "LH": "#67a9cf",
    "ns": NO_DATA,
}

TITLE_SIZE = 15
SUBTITLE_SIZE = 10.5
LABEL_SIZE = 9.5
TICK_SIZE = 9
NOTE_SIZE = 8

SOURCE_NOTE = (
    "Source: City of Minneapolis Open Data (CC0). "
    "Analysis: crimeMapInvestigation."
)


def apply() -> None:
    """Install the project style. Call once before building figures."""
    mpl.rcParams.update({
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "text.color": INK,
        "axes.edgecolor": HAIRLINE,
        "axes.labelcolor": INK,
        "axes.labelsize": LABEL_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.titleweight": "600",
        "axes.titlelocation": "left",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.frameon": False,
        "legend.fontsize": LABEL_SIZE,
        "figure.titlesize": TITLE_SIZE,
    })


def titled(ax, title: str, subtitle: str | None = None) -> None:
    """Title plus an optional subtitle carrying the interpretation.

    The subtitle is where the finding goes. A title that says only what is
    plotted makes the reader derive the point themselves.
    """
    ax.set_title(title, pad=18 if subtitle else 10)
    if subtitle:
        ax.text(
            0.0, 1.02, subtitle, transform=ax.transAxes,
            fontsize=SUBTITLE_SIZE, color=MUTED, va="bottom", ha="left",
        )


def source_note(fig, extra: str | None = None) -> None:
    text = SOURCE_NOTE if extra is None else f"{extra}  |  {SOURCE_NOTE}"
    fig.text(0.01, 0.005, text, fontsize=NOTE_SIZE, color=MUTED, ha="left")


def strip_axes(ax) -> None:
    """Maps carry no meaningful axes; coordinates are not the message."""
    ax.set_axis_off()
    ax.set_aspect("equal")


def fig_titled(fig, title: str, subtitle: str | None = None) -> None:
    """Title a *map* at figure level rather than on the axes.

    An equal-aspect map of a tall, narrow city fills only the middle of its
    axes box, so ``ax.set_title`` and an axes-relative subtitle both land at
    the top of the box and collide. Anchoring to the figure keeps them stacked
    and left-aligned with the frame instead.
    """
    fig.text(0.02, 0.975, title, fontsize=TITLE_SIZE, weight="600",
             color=INK, ha="left", va="top")
    if subtitle:
        fig.text(0.02, 0.945, subtitle, fontsize=SUBTITLE_SIZE,
                 color=MUTED, ha="left", va="top")


def fit_bounds(ax, gdf, *, pad: float = 0.04) -> None:
    """Crop to the data with a small margin, so the city fills the frame."""
    minx, miny, maxx, maxy = gdf.total_bounds
    dx, dy = (maxx - minx) * pad, (maxy - miny) * pad
    ax.set_xlim(minx - dx, maxx + dx)
    ax.set_ylim(miny - dy, maxy + dy)


def despine(ax, keep=("left", "bottom")) -> None:
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)


def figure(width: float = 9.0, height: float = 7.0):
    fig, ax = plt.subplots(figsize=(width, height))
    return fig, ax
