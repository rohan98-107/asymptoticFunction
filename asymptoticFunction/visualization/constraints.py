import numpy as np
import matplotlib.pyplot as plt

from .circle import plot_directions as plot_circle
from .sphere import plot_directions as plot_sphere


def _as_callable_list(funcs):
    if funcs is None:
        return []
    out = []
    for f in funcs:
        if not callable(f):
            raise TypeError("All constraints must be callable")
        out.append(f)
    return out


def _as_direction_array(directions, dim):
    if directions is None:
        return np.empty((0, dim), dtype=float)
    D = np.asarray(directions, dtype=float)
    if D.ndim != 2 or D.shape[1] != dim:
        raise ValueError("directions must have shape (N, dim)")
    return D


def _validate_view(view, dim):
    v = tuple(view)
    if dim == 2:
        if len(v) != 4:
            raise ValueError("view must be (xmin, xmax, ymin, ymax) for dim=2")
        return v
    if dim == 3:
        if len(v) != 6:
            raise ValueError("view must be (xmin, xmax, ymin, ymax, zmin, zmax) for dim=3")
        return v
    raise ValueError("dim must be 2 or 3")


def plot_constraint_set_2d(
    ax,
    X,
    view,
    resolution=400,
    fill_alpha=0.35,
    boundary_width=1.0
):
    xmin, xmax, ymin, ymax = _validate_view(view, 2)
    xs = np.linspace(xmin, xmax, resolution)
    ys = np.linspace(ymin, ymax, resolution)
    XX, YY = np.meshgrid(xs, ys)

    feas = np.ones_like(XX, dtype=bool)

    for fi in _as_callable_list(X):
        Z = np.vectorize(lambda x, y: fi(np.array([x, y], dtype=float)))(XX, YY)
        feas &= Z <= 0.0

        ax.contour(
            XX,
            YY,
            Z,
            levels=[0.0],
            colors="black",
            linewidths=boundary_width
        )

    ax.contourf(
        XX,
        YY,
        feas.astype(float),
        levels=[0.5, 1.5],
        colors=["0.7"],
        alpha=fill_alpha
    )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")


def plot_constraint_set_3d(
    ax,
    X,
    view,
    resolution=45,
    max_points=40000,
    fill_alpha=0.08,
    boundary_alpha=0.6
):
    xmin, xmax, ymin, ymax, zmin, zmax = _validate_view(view, 3)
    xs = np.linspace(xmin, xmax, resolution)
    ys = np.linspace(ymin, ymax, resolution)
    zs = np.linspace(zmin, zmax, resolution)

    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="ij")
    P = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)

    feas = np.ones(len(P), dtype=bool)
    maxf = np.full(len(P), -np.inf)

    for fi in _as_callable_list(X):
        vals = np.array([fi(p) for p in P], dtype=float)
        feas &= vals <= 0.0
        maxf = np.maximum(maxf, vals)

    Q = P[feas]
    if len(Q) > max_points:
        idx = np.random.choice(len(Q), max_points, replace=False)
        Q = Q[idx]

    ax.scatter(
        Q[:, 0],
        Q[:, 1],
        Q[:, 2],
        s=2.0,
        color="0.6",
        alpha=fill_alpha
    )

    bd = np.abs(maxf) <= 1e-3
    B = P[bd]
    if len(B) > max_points:
        idx = np.random.choice(len(B), max_points, replace=False)
        B = B[idx]

    ax.scatter(
        B[:, 0],
        B[:, 1],
        B[:, 2],
        s=4.0,
        color="black",
        alpha=boundary_alpha
    )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)

    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def plot_constraints_and_asymptotics(
    X,
    dim,
    view,
    cone_directions,
    axL=None,
    axR=None,
    title_left=None,
    title_right=None,
    show=True,
    circle_kwargs=None,
    sphere_kwargs=None
):
    if dim not in (2, 3):
        raise ValueError("dim must be 2 or 3")

    _validate_view(view, dim)
    D = _as_direction_array(cone_directions, dim)
    X_list = _as_callable_list(X)

    created = False
    if axL is None or axR is None:
        created = True
        fig = plt.figure(figsize=(11, 4.5))
        if dim == 2:
            axL = fig.add_subplot(1, 2, 1)
            axR = fig.add_subplot(1, 2, 2)
        else:
            axL = fig.add_subplot(1, 2, 1, projection="3d")
            axR = fig.add_subplot(1, 2, 2, projection="3d")
    else:
        fig = axL.figure

    if dim == 2:
        plot_constraint_set_2d(axL, X_list, view)
        if title_left is not None:
            axL.set_title(title_left)
        kw = {} if circle_kwargs is None else dict(circle_kwargs)
        plot_circle(D, dim=2, ax=axR, show=False, title=title_right, **kw)
    else:
        plot_constraint_set_3d(axL, X_list, view)
        if title_left is not None:
            axL.set_title(title_left)
        kw = {} if sphere_kwargs is None else dict(sphere_kwargs)
        plot_sphere(D, dim=3, ax=axR, show=False, title=title_right, **kw)

    if show and created:
        plt.show()

    return fig, axL, axR
