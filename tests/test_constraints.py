import os
import numpy as np

os.environ["MPLBACKEND"] = "Agg"

from asymptoticFunction.visualization.constraints import (
    plot_constraints_and_asymptotics,
    plot_constraint_set_2d,
    plot_constraint_set_3d
)


def test_invalid_dim_raises():
    X = [lambda x: x[0]]
    D = np.zeros((10, 2))
    try:
        plot_constraints_and_asymptotics(X, 4, (-1, 1, -1, 1), D, show=False)
        assert False, "Expected ValueError"
    except ValueError:
        assert True


def test_invalid_view_2d_raises():
    X = [lambda x: x[0]]
    D = np.zeros((10, 2))
    try:
        plot_constraints_and_asymptotics(X, 2, (-1, 1, -1), D, show=False)
        assert False, "Expected ValueError"
    except ValueError:
        assert True


def test_invalid_dirs_shape_raises():
    X = [lambda x: x[0]]
    D = np.zeros((10, 3))
    try:
        plot_constraints_and_asymptotics(X, 2, (-1, 1, -1, 1), D, show=False)
        assert False, "Expected ValueError"
    except ValueError:
        assert True


def test_noncallable_constraint_raises():
    X = [3.0]
    D = np.zeros((10, 2))
    try:
        plot_constraints_and_asymptotics(X, 2, (-1, 1, -1, 1), D, show=False)
        assert False, "Expected TypeError"
    except TypeError:
        assert True


def test_empty_dirs_ok_2d():
    X = [lambda x: x[0] ** 2 + x[1] ** 2 - 1.0]
    D = np.empty((0, 2))
    fig, axL, axR = plot_constraints_and_asymptotics(
        X,
        2,
        (-1.5, 1.5, -1.5, 1.5),
        D,
        show=False
    )
    assert fig is not None
    assert axL is not None
    assert axR is not None


def test_basic_2d_side_by_side_runs():
    X = [lambda x: x[0] ** 2 + x[1] ** 2 - 1.0]
    rng = np.random.default_rng(0)
    D = rng.normal(size=(250, 2))
    fig, axL, axR = plot_constraints_and_asymptotics(
        X,
        2,
        (-1.5, 1.5, -1.5, 1.5),
        D,
        show=False,
        circle_kwargs={"rays": False, "geodesic_radius": 0.05, "max_dirs": 200}
    )
    assert fig is not None


def test_basic_3d_side_by_side_runs():
    X = [lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2 - 1.0]
    rng = np.random.default_rng(1)
    D = rng.normal(size=(300, 3))
    fig, axL, axR = plot_constraints_and_asymptotics(
        X,
        3,
        (-1.2, 1.2, -1.2, 1.2, -1.2, 1.2),
        D,
        show=False,
        sphere_kwargs={"rays": False, "geodesic_radius": 0.15}
    )
    assert fig is not None


def test_bad_sphere_kwargs_bubble_typeerror():
    X = [lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2 - 1.0]
    D = np.zeros((5, 3))
    try:
        plot_constraints_and_asymptotics(X, 3, (-1, 1, -1, 1, -1, 1), D, show=False, sphere_kwargs={"max_dirs": 10})
        assert False, "Expected TypeError"
    except TypeError:
        assert True


def test_plot_constraint_set_helpers_do_not_crash():
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    X = [lambda x: x[0] ** 2 + x[1] ** 2 - 1.0]
    plot_constraint_set_2d(ax, X, (-1.5, 1.5, -1.5, 1.5), resolution=80)

    fig = plt.figure()
    ax3 = fig.add_subplot(1, 1, 1, projection="3d")
    X3 = [lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2 - 1.0]
    plot_constraint_set_3d(ax3, X3, (-1.2, 1.2, -1.2, 1.2, -1.2, 1.2), resolution=12, max_points=2000)