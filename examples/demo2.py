import numpy as np
import matplotlib.pyplot as plt

from asymptoticFunction.core.asymptotic_cone import AsymptoticCone
from asymptoticFunction.visualization.circle import plot_directions


def make_constraints(case):
    if case == "A":
        f_list = [lambda x: x[1] - x[0]]
        xlim, ylim = (-2, 2), (-2, 2)
    elif case == "B":
        f_list = [lambda x: x[0], lambda x: -x[1]]
        xlim, ylim = (-2, 2), (-2, 2)
    elif case == "C":
        f_list = [lambda x: -x[1] + x[0] ** 2]
        xlim, ylim = (-2, 2), (-1, 4)
    else:
        raise ValueError("Unknown case")
    return f_list, xlim, ylim


def make_dirs_from_angles(angles):
    return np.column_stack([np.cos(angles), np.sin(angles)])


def run_algorithm_suite(n_samples=1024, sampling="sobol", tol=1e-9):
    cases = ["A", "B", "C"]
    modes = [
        ("Outer-union", "outer_union", "tab:red"),
        ("Intersection", "intersection", "tab:blue"),
        ("Numerical", "numerical", "tab:green")
    ]

    fig, axes = plt.subplots(3, 3, figsize=(14, 12), constrained_layout=True)

    for i, case in enumerate(cases):
        f_list, xlim, ylim = make_constraints(case)

        cone = AsymptoticCone(f_list=f_list, g_list=[], dim=2)
        cone.x0 = np.zeros(2)

        results = {}
        for _, mode, _ in modes:
            results[mode] = cone.compute(mode=mode, n_samples=n_samples, sampling=sampling, tol=tol)

        for j, (label, mode, color) in enumerate(modes):
            ax = axes[i, j]
            ax.axhline(0, color="black", linewidth=0.6)
            ax.axvline(0, color="black", linewidth=0.6)
            ax.plot(0, 0, "ko", markersize=4)
            ax.set_aspect("equal")
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-1.05, 1.05)

            plot_directions(
                results[mode],
                dim=2,
                ax=ax,
                show=False,
                rays=True,
                geodesic_radius=0.035,
                max_dirs=220,
                color=color,
                cluster_fill=True,
                cluster_gap=0.12,
                title=f"{label} ({case})"
            )

    plt.show()


def run_visual_hull_tests():
    tests = [
        ("Single connected", make_dirs_from_angles(np.linspace(-0.5, 0.5, 300)), "tab:gray"),
        ("Disconnected", make_dirs_from_angles(np.concatenate([np.linspace(-0.3, -0.1, 120), np.linspace(1.6, 1.9, 120)])), "tab:gray"),
        ("Very narrow", make_dirs_from_angles(np.linspace(0.15, 0.2, 80)), "tab:gray"),
        ("Almost full", make_dirs_from_angles(np.linspace(-np.pi + 0.15, np.pi - 0.15, 500)), "tab:gray"),
        ("Noisy cluster", make_dirs_from_angles(np.linspace(-0.6, 0.6, 300) + 0.02 * np.random.randn(300)), "tab:gray")
    ]

    fig, axes = plt.subplots(1, len(tests), figsize=(18, 4), constrained_layout=True)

    for ax, (name, dirs, color) in zip(axes, tests):
        ax.axhline(0, color="black", linewidth=0.6)
        ax.axvline(0, color="black", linewidth=0.6)
        ax.plot(0, 0, "ko", markersize=4)
        ax.set_aspect("equal")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)

        plot_directions(
            dirs,
            dim=2,
            ax=ax,
            show=False,
            rays=True,
            geodesic_radius=0.035,
            max_dirs=240,
            color=color,
            cluster_fill=True,
            cluster_gap=0.12,
            title=name
        )

    plt.show()


def main():
    run_algorithm_suite(n_samples=1024, sampling="sobol", tol=1e-9)
    run_visual_hull_tests()


if __name__ == "__main__":
    main()
