"""Shared style for the thesis figures.

The principle is to draw at the final size: a figure produced at 15 inches
and included at \\textwidth is shrunk by a factor 0.42, and a 9 pt font ends
up at 3.8 pt on the page. Here the native width coincides with the inclusion
width, so the scale factor is 1 and the fonts stay as set.

Two widths:
    FULL    multi-panel figures, to be included with width=\\textwidth
    SINGLE  single-panel figures, to be included with width=0.7\\textwidth

The rcParams are the same for both, so the text has the same size everywhere
in the document.

Typical use:

    from thesis_plots import style as ts
    ts.apply()
    fig, axes = ts.panels(3)
    ...
    ts.legend_above(fig, axes[0], ncol=5)
    ts.save(fig, "thesis_images/.../figure_name")
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Geometry. TEXTWIDTH should be checked in the document with \the\textwidth:
# 6.27 in corresponds to A4 with one-inch margins.
# --------------------------------------------------------------------------
FULL = 6.27
SINGLE = 0.7 * FULL          # ~4.39 in

# Default height by number of side-by-side panels. It comes from wanting to
# keep every panel close to a 4:3 ratio once axes and labels are removed.
_HEIGHT = {1: 3.0, 2: 2.6, 3: 2.1, 4: 1.9}

# --------------------------------------------------------------------------
# Typography. One place for every script.
# --------------------------------------------------------------------------
FONT_BASE = 9                # axis labels and generic text
FONT_TICK = 8
FONT_LEGEND = 8
LINEWIDTH = 0.8
MARKERSIZE = 2.5
MARKEVERY = 5               # one marker every 5 points
BAND_ALPHA = 0.15


def apply():
    """Apply the shared rcParams. To be called once per script."""
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": FONT_BASE,
        "axes.labelsize": FONT_BASE,
        "axes.titlesize": FONT_BASE,
        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,
        "legend.fontsize": FONT_LEGEND,
        "lines.linewidth": LINEWIDTH,
        "lines.markersize": MARKERSIZE,
        "axes.grid": False,
    })


def panels(n=3, height=None, width=None, **kwargs):
    """Figure with n side-by-side panels, as wide as the text.

    kwargs go to plt.subplots (sharex, sharey, ...). The height follows the
    number of panels when not specified.
    """
    w = FULL if width is None else width
    h = height if height is not None else _HEIGHT.get(n, 2.15)
    fig, axes = plt.subplots(1, n, figsize=(w, h),
                             layout="constrained", **kwargs)
    _tick_in(axes)
    return fig, axes


def single(height=None, **kwargs):
    """Single-panel figure, to be included with width=0.7\\textwidth."""
    h = height if height is not None else 3.0
    fig, ax = plt.subplots(figsize=(SINGLE, h),
                           layout="constrained", **kwargs)
    _tick_in(ax)
    return fig, ax


def grid(rows, cols, height=None, **kwargs):
    """Grid of panels as wide as the text, for the 2x2 layouts."""
    h = height if height is not None else 2.2 * rows
    fig, axes = plt.subplots(rows, cols, figsize=(FULL, h),
                             layout="constrained", **kwargs)
    _tick_in(axes)
    return fig, axes


def _tick_in(axes):
    for ax in (axes.ravel() if hasattr(axes, "ravel") else
               (axes if isinstance(axes, (list, tuple)) else [axes])):
        ax.tick_params(direction="in")


def legend_above(fig, source, ncol=None, handles=None, labels=None):
    """Single legend above the figure, with the handles of the first panel."""
    if handles is None:
        handles, labels = source.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=FONT_LEGEND,
               ncol=ncol or len(handles), loc="outside upper center",
               columnspacing=1.2, handlelength=1.6)


def save(fig, path, tight_top=None, dpi=300):
    """Save PDF and PNG with the same name.

    The room for the legend is handled by the constrained layout set when
    the figure is created, so there is no need to touch the margins here.
    The tight_top parameter is still accepted for compatibility with the
    existing calls, but it is ignored.
    """
    fig.savefig(f"{path}.pdf", bbox_inches="tight")
    fig.savefig(f"{path}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}.pdf / .png")
