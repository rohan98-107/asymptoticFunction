import numpy as np
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection

try:
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
except Exception:
    Line3DCollection = None
    Poly3DCollection = None


def plot_directions(
    directions,
    masks=None,
    dim=None,
    ax=None,
    title=None,
    show=True,
    rays=True,
    ray_length=1.0,
    geodesic_radius=0.04,
):
    directions = np.asarray(directions, dtype=float)

    if directions.ndim != 2:
        raise ValueError("directions must be a 2D array")

    n, d = directions.shape

    if dim is None:
        dim = d
    if dim != 3:
        raise NotImplementedError("sphere plotting only supports 3D")

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    if n == 0 or np.all(np.linalg.norm(directions, axis=1) == 0):
        ax.scatter([0.0], [0.0], [0.0], s=24, color="steelblue")

        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_zlim(-1.1, 1.1)

        try:
            ax.set_box_aspect((1, 1, 1))
        except TypeError:
            pass

        if title is not None:
            ax.set_title(title)

        if show:
            plt.show()

        return ax

    X = _normalize_rows(directions[:, :3])

    if masks is None:
        _draw_group(
            ax,
            X,
            rays=rays,
            ray_length=ray_length,
            geodesic_radius=geodesic_radius,
        )
    else:
        for name, mask in masks.items():
            Xm = X[np.asarray(mask)]
            if len(Xm) == 0:
                continue
            _draw_group(
                ax,
                Xm,
                label=name,
                rays=rays,
                ray_length=ray_length,
                geodesic_radius=geodesic_radius,
            )
        ax.legend()

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(-1.1, 1.1)

    try:
        ax.set_box_aspect((1, 1, 1))
    except TypeError:
        pass

    if title is not None:
        ax.set_title(title)

    if show:
        plt.show()

    return ax


def _draw_group(
    ax,
    X,
    label=None,
    rays=True,
    ray_length=1.0,
    geodesic_radius=0.04,
):
    if rays:
        _draw_rays(ax, X, label=label, ray_length=ray_length)

    if geodesic_radius is not None:
        _draw_geodesic_caps(
            ax,
            X,
            geodesic_radius=geodesic_radius,
        )


def _draw_rays(ax, X, label=None, ray_length=1.0, alpha=0.6):
    if Line3DCollection is None:
        return

    segments = np.zeros((len(X), 2, 3))
    segments[:, 1, :] = X * ray_length

    lc = Line3DCollection(
        segments,
        linewidths=0.4,
        alpha=alpha,
        label=label,
    )
    ax.add_collection3d(lc)


def _draw_geodesic_caps(
    ax,
    X,
    geodesic_radius,
    max_caps=300,
    n_pts=20,
    alpha_fill=0.16,
    alpha_edge=0.35,
):
    if Poly3DCollection is None:
        return

    if len(X) > max_caps:
        idx = np.linspace(0, len(X) - 1, max_caps, dtype=int)
        X = X[idx]

    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    ct = np.cos(theta)
    st = np.sin(theta)

    c = np.cos(geodesic_radius)
    s = np.sin(geodesic_radius)

    polys = []

    for d in X:
        u, v = _tangent_basis(d)
        boundary = c * d + s * (ct[:, None] * u + st[:, None] * v)
        polys.append(boundary)

        if len(X) <= 80:
            ax.plot(
                boundary[:, 0],
                boundary[:, 1],
                boundary[:, 2],
                linewidth=0.4,
                alpha=alpha_edge,
            )

    poly = Poly3DCollection(polys, alpha=alpha_fill, linewidths=0.0)
    ax.add_collection3d(poly)


def _tangent_basis(d):
    if abs(d[2]) < 0.9:
        u = np.cross(d, [0.0, 0.0, 1.0])
    else:
        u = np.cross(d, [0.0, 1.0, 0.0])

    u = u / np.linalg.norm(u)
    v = np.cross(d, u)
    return u, v


def _normalize_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n
