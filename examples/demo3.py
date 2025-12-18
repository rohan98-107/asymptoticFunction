import numpy as np

from asymptoticFunction.core.asymptotic_cone import AsymptoticCone
from asymptoticFunction.visualization.sphere import plot_directions


# ============================================================
# Constraint factories
# ============================================================

def pyramid_constraints():
    f = [
        lambda x: -x[2],
        lambda x: -x[2] + x[0] + x[1],
        lambda x: -x[2] + x[0] - x[1],
        lambda x: -x[2] - x[0] + x[1],
        lambda x: -x[2] - x[0] - x[1],
    ]
    return f, []


def double_cone_constraints():
    f = [
        lambda x: x[0] ** 2 + x[1] ** 2 - x[2] ** 2,
    ]
    return f, []


def diagonal_escape_constraints():
    f = [
        lambda x: x[0] ** 2 + x[1] ** 2 - x[2],
    ]
    return f, []


def near_isotropic_constraints():
    f = [
        lambda x: x[0] ** 4 + x[1] ** 4 + x[2] ** 4 - 1.0,
    ]
    return f, []


# ============================================================
# Test runner
# ============================================================

def run_case(
    name,
    constraint_factory,
    n_samples=3000,
    geodesic_radius=0.04,
):
    print(f"\n=== {name} ===")

    f_list, g_list = constraint_factory()

    cone = AsymptoticCone(
        f_list=f_list,
        g_list=g_list,
        dim=3,
    )

    cone.x0 = np.zeros(3)

    directions = cone.compute(
        mode="outer_union",
        n_samples=n_samples,
        sampling="sobol",
        tol=1e-6,
    )

    if directions is None or len(directions) == 0:
        print("No directions found.")
        return

    print(f"Found {len(directions)} directions")

    plot_directions(
        directions,
        rays=True,
        geodesic_radius=geodesic_radius,
        title=f"{name} — exploration",
    )

    plot_directions(
        directions,
        hull=True,
        geodesic_radius=geodesic_radius,
        title=f"{name} — hull summary",
    )


# ============================================================
# Main
# ============================================================

def main():
    test_cases = [
        ("Pyramid cone", pyramid_constraints, 3000),
        ("Double cone", double_cone_constraints, 3000),
        ("Diagonal escape", diagonal_escape_constraints, 2500),
        ("Near isotropic", near_isotropic_constraints, 1800),
    ]

    for name, factory, n_samples in test_cases:
        run_case(
            name,
            factory,
            n_samples=n_samples,
            geodesic_radius=0.04,
        )


if __name__ == "__main__":
    main()
