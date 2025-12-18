import numpy as np
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection

try:
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
except Exception:
    Line3DCollection = None
    Poly3DCollection = None

try:
    from scipy.spatial import ConvexHull, cKDTree
except Exception:
    ConvexHull = None
    cKDTree = None


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
        hull=False,
):
    directions = np.asarray(directions)

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

    X = _normalize_rows(directions[:, :3])

    if hull:
        _draw_clustered_hulls_filled( # can replace with _draw_clustered_hulls_boundary()
            ax,
            X,
            geodesic_radius=geodesic_radius,
        )
    else:
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
    segments = np.zeros((len(X), 2, 3))
    segments[:, 1, :] = X * ray_length

    if Line3DCollection is None:
        return

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
        idx = np.random.choice(len(X), max_caps, replace=False)
        X = X[idx]

    polys = []
    edges = []

    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    ct = np.cos(theta)
    st = np.sin(theta)

    c = np.cos(geodesic_radius)
    s = np.sin(geodesic_radius)

    for d in X:
        u, v = _tangent_basis(d)
        boundary = c * d + s * (ct[:, None] * u + st[:, None] * v)
        polys.append(boundary)
        edges.append(boundary)

    poly = Poly3DCollection(polys, alpha=alpha_fill, linewidths=0.0)
    ax.add_collection3d(poly)

    if len(edges) <= 80:
        for b in edges:
            ax.plot(b[:, 0], b[:, 1], b[:, 2], linewidth=0.4, alpha=alpha_edge)


def _draw_clustered_hulls_boundary(
        ax,
        X,
        geodesic_radius,
        cluster_angle=None,
):
    if ConvexHull is None or cKDTree is None:
        return

    if cluster_angle is None:
        cluster_angle = 3.0 * geodesic_radius

    clusters = _cluster_by_angle(X, cluster_angle)
    hull_color = "black"

    for idx in clusters:
        P = X[idx]

        if len(P) < 3:
            P = _augment_points_in_caps(P, geodesic_radius)

        if len(P) < 3:
            continue

        center = P.mean(axis=0)
        center /= np.linalg.norm(center)

        u, v = _tangent_basis(center)
        coords = np.stack([P @ u, P @ v], axis=1)

        try:
            hull = ConvexHull(coords)
        except Exception:
            continue

        boundary = P[hull.vertices]

        ax.plot(
            np.append(boundary[:, 0], boundary[0, 0]),
            np.append(boundary[:, 1], boundary[0, 1]),
            np.append(boundary[:, 2], boundary[0, 2]),
            linewidth=1.2,
            alpha=0.8,
            color=hull_color
        )


def _draw_clustered_hulls_filled(
        ax,
        X,
        geodesic_radius,
        cluster_angle=None
):
    if Poly3DCollection is None:
        return
    if ConvexHull is None or cKDTree is None:
        return

    X = np.asarray(X)
    if len(X) == 0:
        return

    if cluster_angle is None:
        cluster_angle = 3.0 * geodesic_radius

    clusters = _cluster_by_angle(X, cluster_angle)

    tris = []

    for idx in clusters:
        P = X[idx]

        if len(P) < 3:
            P = _augment_points_in_caps(P, geodesic_radius)

        if len(P) < 3:
            continue

        center = P.mean(axis=0)
        n = np.linalg.norm(center)
        if n == 0:
            continue
        center = center / n

        u, v = _tangent_basis(center)
        coords = np.stack([P @ u, P @ v], axis=1)

        try:
            hull = ConvexHull(coords)
        except Exception:
            continue

        boundary = P[hull.vertices]

        m = len(boundary)
        for i in range(m):
            a = boundary[i]
            b = boundary[(i + 1) % m]
            tris.append([center, a, b])

    if len(tris) == 0:
        return

    poly = Poly3DCollection(
        tris,
        alpha=0.15,
        linewidths=0.0
    )
    ax.add_collection3d(poly)


def _augment_points_in_caps(P, geodesic_radius):
    if len(P) == 1:
        return np.vstack([P, _sample_cap(P[0], geodesic_radius, 2)])

    if len(P) == 2:
        mid = _normalize_vec(P[0] + P[1])
        return np.vstack([P, mid])

    return P


def _sample_cap(d, geodesic_radius, k):
    u, v = _tangent_basis(d)
    angles = np.random.uniform(0, 2 * np.pi, size=k)
    radii = np.random.uniform(0, geodesic_radius, size=k)

    pts = (
            np.cos(radii)[:, None] * d
            + np.sin(radii)[:, None]
            * (np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v)
    )
    return pts


def _cluster_by_angle(X, angle):
    r = 2.0 * np.sin(0.5 * angle)
    tree = cKDTree(X)

    visited = np.zeros(len(X), dtype=bool)
    clusters = []

    for i in range(len(X)):
        if visited[i]:
            continue

        stack = [i]
        visited[i] = True
        comp = []

        while stack:
            j = stack.pop()
            comp.append(j)
            for k in tree.query_ball_point(X[j], r):
                if not visited[k]:
                    visited[k] = True
                    stack.append(k)

        clusters.append(np.array(comp))

    return clusters


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


def _normalize_vec(x):
    n = np.linalg.norm(x)
    if n == 0:
        return x
    return x / n
