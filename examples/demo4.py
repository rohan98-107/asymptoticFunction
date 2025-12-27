import numpy as np

from asymptoticFunction.core.types import CallableFunction
from asymptoticFunction.core.asymptotic_cone import AsymptoticCone
from asymptoticFunction.numerical.approximation import approximateAsymptoticFunc
from asymptoticFunction.known.registry import get_analytical_form
from asymptoticFunction.visualization.constraints import (
    plot_constraints_and_asymptotics
)


# ============================================================
# Utility: pick representative directions
# ============================================================

def pick_directions(dirs, k=6):
    assert len(dirs) > 0
    idx = np.linspace(0, len(dirs) - 1, k, dtype=int)
    return dirs[idx]


# ============================================================
# Utility: explicit diagnostics
# ============================================================

def diagnostics(constraints, directions, label):
    print("\n" + "=" * 70)
    print(f"ASYMPTOTIC DIAGNOSTICS — {label}")
    print("=" * 70)

    for j, d in enumerate(directions):
        print(f"\nd{j+1} = {np.array2string(d, precision=4)}")
        print(f"{'constraint':<12} {'numerical f∞':>18} {'analytical f∞':>18}")
        print("-" * 52)

        for i, cf in enumerate(constraints):
            f = cf.f
            kind = cf.kind
            params = cf.params

            num = approximateAsymptoticFunc(f, d)

            ana = "—"
            if kind is not None:
                try:
                    ana_func = get_analytical_form(kind)
                    ana = ana_func(f, d, **params)
                except Exception:
                    pass

            print(f"f{i+1:<10} {num:>18.6g} {str(ana):>18}")


# ============================================================
# 2D DEMO — 3 CONSTRAINTS
# ============================================================

def demo_2d():
    X = [
        CallableFunction(lambda x: x[1]**2 - x[0], kind="polynomial"),
        CallableFunction(lambda x: -x[0], kind="polynomial"),
        CallableFunction(lambda x: x[1]**4 - x[0]**2, kind="polynomial"),
    ]

    cone = AsymptoticCone(X, dim=2)
    dirs = cone.compute(n_samples=4096)

    assert len(dirs) > 0, "2D cone should be nonempty"

    test_dirs = pick_directions(dirs, 6)
    diagnostics(X, test_dirs, "2D — THREE CONSTRAINT PARABOLIC SET")

    plot_constraints_and_asymptotics(
        X,
        dim=2,
        view=(-2, 6, -4, 4),
        cone_directions=dirs,
        title_left="2D constraint set X",
        title_right="Asymptotic cone of X",
        show=True
    )


# ============================================================
# 3D DEMO — 3 CONSTRAINTS
# ============================================================

def demo_3d():
    X = [
        CallableFunction(lambda x: x[0]**2 + x[1]**2 - x[2]**2, kind="polynomial"),
        CallableFunction(lambda x: -x[2], kind="polynomial"),
        CallableFunction(lambda x: x[0]**4 - x[2]**2, kind="polynomial"),
    ]

    cone = AsymptoticCone(X, dim=3)
    dirs = cone.compute(n_samples=8192)

    assert len(dirs) > 0, "3D cone should be nonempty"

    test_dirs = pick_directions(dirs, 6)
    diagnostics(X, test_dirs, "3D — THREE CONSTRAINT QUADRATIC SET")

    plot_constraints_and_asymptotics(
        X,
        dim=3,
        view=(-4, 4, -4, 4, -1, 6),
        cone_directions=dirs,
        title_left="3D constraint set X",
        title_right="Asymptotic cone of X",
        show=True
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    demo_2d()
    demo_3d()
