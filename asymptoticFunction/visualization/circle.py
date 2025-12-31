import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def plot_directions(
    directions,
    dim=None,
    ax=None,
    title=None,
    show=True,
    rays=True,
    geodesic_radius=0.035,
    max_dirs=200,
    color="steelblue",
    fill_connected=False,
):
    directions = np.asarray(directions, dtype=float)

    if directions.ndim != 2:
        raise ValueError("directions must be a 2D array")

    n, d = directions.shape

    if dim is None:
        dim = d
    if dim != 2:
        raise NotImplementedError("circle plotting only supports 2D")

    if ax is None:
        _, ax = plt.subplots()

    if n == 0 or np.all(np.linalg.norm(directions, axis=1) == 0):
        ax.set_aspect("equal")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)

        ax.scatter([0.0], [0.0], s=18, color=color, zorder=5)

        if title is not None:
            ax.set_title(title)

        if show:
            plt.show()

        return ax

    X = _normalize_rows(directions[:, :2])

    if len(X) > max_dirs:
        idx = np.linspace(0, len(X) - 1, max_dirs, dtype=int)
        X = X[idx]

    angles = np.arctan2(X[:, 1], X[:, 0])

    if geodesic_radius is not None:
        if fill_connected:
            arcs = _merge_overlapping_arcs(angles, geodesic_radius)
            for a_min, a_max in arcs:
                _draw_arc(ax, a_min, a_max, color)
                _fill_arc_sector(ax, a_min, a_max, color)
                if rays:
                    _draw_radial_edges(ax, a_min, a_max, color)
        else:
            for a in angles:
                _draw_arc(ax, a - geodesic_radius, a + geodesic_radius, color)
            if rays:
                _draw_unit_rays(ax, X, color)

    elif rays:
        _draw_unit_rays(ax, X, color)

    ax.set_aspect("equal")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)

    if title is not None:
        ax.set_title(title)

    if show:
        plt.show()

    return ax


def _merge_overlapping_arcs(angles, radius):
    arcs = np.sort(
        np.column_stack((angles - radius, angles + radius)),
        axis=0,
    )

    merged = []
    cur_min, cur_max = arcs[0]

    for a_min, a_max in arcs[1:]:
        if a_min <= cur_max:
            cur_max = max(cur_max, a_max)
        else:
            merged.append((cur_min, cur_max))
            cur_min, cur_max = a_min, a_max

    merged.append((cur_min, cur_max))
    return merged


def _draw_arc(ax, a_min, a_max, color, n_pts=60, alpha=0.7):
    t = np.linspace(a_min, a_max, n_pts)
    ax.plot(
        np.cos(t),
        np.sin(t),
        color=color,
        linewidth=2.0,
        alpha=alpha,
    )


def _fill_arc_sector(ax, a_min, a_max, color, n_pts=80, alpha=0.18):
    t = np.linspace(a_min, a_max, n_pts)
    x = np.cos(t)
    y = np.sin(t)

    ax.fill(
        np.concatenate([[0.0], x, [0.0]]),
        np.concatenate([[0.0], y, [0.0]]),
        color=color,
        alpha=alpha,
        linewidth=0.0,
    )


def _draw_radial_edges(ax, a_min, a_max, color, width=1.8):
    ax.plot(
        [0.0, np.cos(a_min)],
        [0.0, np.sin(a_min)],
        color=color,
        linewidth=width,
    )
    ax.plot(
        [0.0, np.cos(a_max)],
        [0.0, np.sin(a_max)],
        color=color,
        linewidth=width,
    )


def _draw_unit_rays(ax, X, color, alpha=0.6):
    segments = np.zeros((len(X), 2, 2))
    segments[:, 1, :] = X

    lc = LineCollection(
        segments,
        linewidths=0.6,
        alpha=alpha,
        color=color,
    )
    ax.add_collection(lc)


def _normalize_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n
