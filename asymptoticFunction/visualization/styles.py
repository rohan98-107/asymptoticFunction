"""
Visualization styles.

TikZ / LaTeX-inspired matplotlib style with safe fallbacks
and post-plot axis helpers.
"""

import matplotlib as mpl


# ---------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------

def use_tikz_style():
    """
    Configure matplotlib to mimic TikZ / LaTeX aesthetics.

    Call once at the beginning of a script or notebook.
    """

    mpl.rcParams.update({
        # Fonts (safe fallback chain)
        "font.family": "serif",
        "font.serif": [
            "Computer Modern Roman",
            "CMU Serif",
            "Latin Modern Roman",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,

        # Font sizes
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,

        # Lines
        "lines.linewidth": 1.0,
        "lines.markersize": 4.0,

        # Axes
        "axes.linewidth": 0.8,
        "axes.edgecolor": "black",
        "axes.facecolor": "white",
        "axes.grid": False,

        # Ticks
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.pad": 2,
        "ytick.major.pad": 2,

        # Legend
        "legend.frameon": False,
        "legend.handlelength": 1.4,

        # Figure / export
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


# ---------------------------------------------------------------------
# Axis post-processing helpers
# ---------------------------------------------------------------------

def style_axes_2d(ax):
    """
    Post-process a 2D axis to reduce clutter (TikZ-like).
    """
    ax.grid(False)

    xt = ax.get_xticks()
    yt = ax.get_yticks()

    if len(xt) > 1:
        ax.set_xticks(xt[::2])
    if len(yt) > 1:
        ax.set_yticks(yt[::2])


def style_axes_3d(ax):
    """
    Post-process a 3D axis to reduce clutter and remove grid/panes.
    """
    ax.grid(False)

    # Make panes transparent / clean
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor("black")

    xt = ax.get_xticks()
    yt = ax.get_yticks()
    zt = ax.get_zticks()

    if len(xt) > 1:
        ax.set_xticks(xt[::2])
    if len(yt) > 1:
        ax.set_yticks(yt[::2])
    if len(zt) > 1:
        ax.set_zticks(zt[::2])
