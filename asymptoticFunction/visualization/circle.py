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
    color="black",
    cluster_fill=False,
    cluster_gap=0.12
):
    directions = np.asarray(directions)

    if directions.ndim != 2:
        raise ValueError("directions must be a 2D array")

    n, d = directions.shape

    if dim is None:
        dim = d
    if dim != 2:
        raise NotImplementedError("circle plotting only supports 2D")

    if ax is None:
        fig, ax = plt.subplots()

    X = _normalize_rows(directions[:, :2])

    if len(X) > max_dirs:
        idx = np.random.choice(len(X), max_dirs, replace=False)
        X = X[idx]

    angles = np.arctan2(X[:, 1], X[:, 0])

    if cluster_fill:
        _draw_clustered_wedges(ax, angles, cluster_gap, color)
    else:
        if rays:
            _draw_unit_rays(ax, X, color)
        if geodesic_radius is not None:
            _draw_boundary_arcs(ax, angles, geodesic_radius, color)

    ax.set_aspect("equal")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)

    if title is not None:
        ax.set_title(title)

    if show:
        plt.show()

    return ax


def _draw_unit_rays(ax, X, color, alpha=0.6):
    segments = np.zeros((len(X), 2, 2))
    segments[:, 1, :] = X

    lc = LineCollection(
        segments,
        linewidths=0.6,
        alpha=alpha,
        color=color
    )
    ax.add_collection(lc)


def _draw_boundary_arcs(ax, angles, geodesic_radius, color, n_pts=20, alpha=0.6):
    for a in angles:
        t = np.linspace(a - geodesic_radius, a + geodesic_radius, n_pts)
        ax.plot(
            np.cos(t),
            np.sin(t),
            color=color,
            linewidth=0.9,
            alpha=alpha
        )


def _draw_clustered_wedges(ax, angles, gap, color):
    clusters = _cluster_angles(angles, gap)
    for c in clusters:
        _draw_cluster_wedge(ax, c.min(), c.max(), color, color)


def _cluster_angles(angles, gap):
    angles = np.sort(angles)
    clusters = [[angles[0]]]

    for a in angles[1:]:
        if a - clusters[-1][-1] <= gap:
            clusters[-1].append(a)
        else:
            clusters.append([a])

    return [np.array(c) for c in clusters]


def _draw_cluster_wedge(
    ax,
    theta_min,
    theta_max,
    fill_color,
    edge_color,
    fill_alpha=0.18,
    edge_width=2.0,
    arc_width=2.5,
    n_arc_pts=80
):
    t = np.linspace(theta_min, theta_max, n_arc_pts)
    arc_x = np.cos(t)
    arc_y = np.sin(t)

    ax.fill(
        np.concatenate([[0.0], arc_x, [0.0]]),
        np.concatenate([[0.0], arc_y, [0.0]]),
        color=fill_color,
        alpha=fill_alpha,
        linewidth=0.0
    )

    ax.plot(
        arc_x,
        arc_y,
        color=edge_color,
        linewidth=arc_width
    )

    ax.plot(
        [0.0, np.cos(theta_min)],
        [0.0, np.sin(theta_min)],
        color=edge_color,
        linewidth=edge_width
    )

    ax.plot(
        [0.0, np.cos(theta_max)],
        [0.0, np.sin(theta_max)],
        color=edge_color,
        linewidth=edge_width
    )


def _normalize_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n
