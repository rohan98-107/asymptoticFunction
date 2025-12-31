import numpy as np
import matplotlib.pyplot as plt

from asymptoticFunction.core.asymptotic_cone import AsymptoticCone
from asymptoticFunction.core.types import CallableFunction
from asymptoticFunction.heuristics.polynomial_heuristics import polynomial_empirical_asymptotic
from asymptoticFunction.visualization.constraints import plot_constraints_and_directions
from asymptoticFunction.visualization.styles import (
    use_tikz_style,
    style_axes_2d,
    style_axes_3d,
)


def prune_polynomial(dirs, X, tol=1e-6):
    dirs = np.asarray(dirs, dtype=float)

    if dirs.size == 0:
        return np.empty((0, dirs.shape[1]), dtype=float)

    keep = []
    for d in dirs:
        ok = True
        for cf in X:
            val = polynomial_empirical_asymptotic(cf.f, d)
            if val > tol:
                ok = False
                break
        if ok:
            keep.append(d)

    if len(keep) == 0:
        return np.empty((0, dirs.shape[1]), dtype=float)

    return np.asarray(keep, dtype=float)


def print_stats(label, n_samples, raw, pruned):
    print("-" * 70)
    print(label)
    print(f"sampled directions        : {n_samples}")
    print(f"cone directions (raw)     : {len(raw)}")
    print(f"cone directions (pruned)  : {len(pruned)}")
    ratio = 0.0 if len(raw) == 0 else len(pruned) / len(raw)
    print(f"retention ratio           : {ratio:.6f}")
    print("-" * 70)
    print()


# ------------------------------------------------------------
# 2D TESTS
# ------------------------------------------------------------

def demo_2d():
    tests = [
        (
            "2D — parabolic wedge + equality",
            [
                CallableFunction(lambda x: x[1] ** 2 - x[0]),
                CallableFunction(lambda x: -x[0]),
                CallableFunction(lambda x: x[1])
            ],
            "{ d : d2 = 0, d1 >= 0 }",
            (-2, 6, -4, 4)
        ),
        (
            "2D — quartic horn",
            [
                CallableFunction(lambda x: x[1] ** 4 - x[0]),
                CallableFunction(lambda x: -x[0])
            ],
            "{ d : d2 = 0, d1 >= 0 }",
            (-2, 6, -4, 4)
        ),
        (
            "2D — polynomial / exponential mix",
            [
                CallableFunction(lambda x: x[1] ** 2 - x[0]),
                CallableFunction(lambda x: np.exp(x[1]) - x[0])
            ],
            "{ d : d2 <= 0, d1 >= 0 }",
            (-2, 6, -4, 4)
        ),
        (
            "2D — curved strip",
            [
                CallableFunction(lambda x: (x[1] - x[0]) ** 2 - x[0]),
                CallableFunction(lambda x: -x[0])
            ],
            "{ d : d1 = d2 >= 0 }",
            (-2, 6, -4, 4)
        )
    ]

    for name, X, cone_desc, view in tests:
        print("\nTRUE ASYMPTOTIC CONE (2D):", cone_desc)

        cone = AsymptoticCone(X, dim=2)
        raw = cone.compute(mode="numerical", n_samples=4096)
        pruned = prune_polynomial(raw, X)

        fig, axL, axR = plot_constraints_and_directions(
            X,
            dim=2,
            view=view,
            directions=raw,
            title_left=name,
            title_right="raw numerical",
            show=False
        )

        style_axes_2d(axL)
        style_axes_2d(axR)
        plt.show()

        fig, axL, axR = plot_constraints_and_directions(
            X,
            dim=2,
            view=view,
            directions=pruned,
            title_left=name,
            title_right="after polynomial heuristics",
            show=False
        )

        style_axes_2d(axL)
        style_axes_2d(axR)
        plt.show()

        print_stats(name, 4096, raw, pruned)


# ------------------------------------------------------------
# 3D TESTS
# ------------------------------------------------------------

def demo_3d():
    tests = [
        (
            "3D — parabolic cylinder",
            [
                CallableFunction(lambda x: x[0] ** 2 - x[2]),
                CallableFunction(lambda x: -x[2])
            ],
            "{ d : d3 >= 0, d1 = 0 }",
            (-4, 4, -4, 4, -1, 6)
        ),
        (
            "3D — quartic cone",
            [
                CallableFunction(lambda x: x[0] ** 4 + x[1] ** 4 - x[2] ** 2),
                CallableFunction(lambda x: -x[2])
            ],
            "{ d : d3 >= |d1|, |d2| }",
            (-4, 4, -4, 4, -1, 6)
        ),
        (
            "3D — saddle with equality",
            [
                CallableFunction(lambda x: x[0] * x[1] - x[2]),
                CallableFunction(lambda x: -x[2]),
                CallableFunction(lambda x: x[0] - x[1])
            ],
            "{ d : d1 = d2, d3 >= 0 }",
            (-4, 4, -4, 4, -1, 6)
        ),
        (
            "3D — polynomial / exponential mix",
            [
                CallableFunction(lambda x: x[0] ** 2 + x[1] ** 2 - x[2]),
                CallableFunction(lambda x: np.exp(x[0]) - x[2])
            ],
            "{ d : d1 <= 0, d3 >= 0 }",
            (-4, 4, -4, 4, -1, 6)
        )
    ]

    for name, X, cone_desc, view in tests:
        print("\nTRUE ASYMPTOTIC CONE (3D):", cone_desc)

        cone = AsymptoticCone(X, dim=3)
        raw = cone.compute(mode="numerical", n_samples=8192)
        pruned = prune_polynomial(raw, X)

        fig, axL, axR = plot_constraints_and_directions(
            X,
            dim=3,
            view=view,
            directions=raw,
            title_left=name,
            title_right="raw numerical",
            show=False
        )

        style_axes_3d(axL)
        style_axes_3d(axR)
        plt.show()

        fig, axL, axR = plot_constraints_and_directions(
            X,
            dim=3,
            view=view,
            directions=pruned,
            title_left=name,
            title_right="after polynomial heuristics",
            show=False
        )

        style_axes_3d(axL)
        style_axes_3d(axR)
        plt.show()

        print_stats(name, 8192, raw, pruned)


# ------------------------------------------------------------
# run
# ------------------------------------------------------------

if __name__ == "__main__":
    use_tikz_style()
    np.set_printoptions(precision=4, suppress=True)
    demo_2d()
    demo_3d()
