import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D


FIG_DIR = Path("../figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


# Figure 1: Asymptotic direction
def figure_asymptotic_direction():
    k = np.arange(1, 35)

    d = np.array([1.0, 0.0])
    n = np.array([0.0, 1.0])

    amplitude = 0.15 * k
    omega = 0.6
    oscillation = np.sin(omega * k)

    points = np.outer(k, d) + np.outer(amplitude * oscillation, n)
    x = points[:, 0]
    y = points[:, 1]

    x_min = -2.0
    x_max = x[-1] * 1.05
    y_min = -x[-1] * 0.5
    y_max =  x[-1] * 0.5

    plt.figure(figsize=(4.8, 4.8))

    axis_arrow = dict(
        width=0.0,
        head_width=0.25,
        head_length=0.35,
        length_includes_head=True,
        color="black",
        zorder=1
    )

    plt.arrow(x_min, 0.0, x_max - x_min, 0.0, **axis_arrow)
    plt.arrow(0.0, y_min, 0.0, y_max - y_min, **axis_arrow)

    plt.scatter(x, y, s=30, color="black", zorder=3)

    plt.arrow(
        0.0, 0.0,
        x[-1] * 0.95, 0.0,
        width=0.12,
        head_width=0.6,
        head_length=0.45,
        length_includes_head=True,
        color="red",
        zorder=4
    )

    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "asymptotic_direction.svg")
    plt.close()


# Figure 2: Asymptotic function (surface + discontinuous limit)
def figure_asymptotic_function():
    x = np.linspace(-2.0, 2.0, 40)
    y = np.linspace(-2.0, 2.0, 80)
    X, Y = np.meshgrid(x, y)
    Z = Y**3 - Y

    fig = plt.figure(figsize=(9, 4))

    # ---- Left: surface ----
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(
        X, Y, Z,
        color="lightgray",
        edgecolor="black",
        linewidth=0.3,
        rstride=3,
        cstride=3,
        alpha=0.9
    )
    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$y$")
    ax1.set_zlabel(r"$f(x,y)$")
    ax1.view_init(elev=25, azim=-60)

    # ---- Right: asymptotic function (DISCONTINUOUS) ----
    ax2 = fig.add_subplot(1, 2, 2)

    theta = np.linspace(-np.pi, np.pi, 400)
    dy = np.sin(theta)

    theta_pos = theta[dy > 0]
    theta_neg = theta[dy < 0]

    ax2.plot(theta_pos, np.ones_like(theta_pos), color="black", linewidth=2)
    ax2.plot(theta_neg, -np.ones_like(theta_neg), color="black", linewidth=2)

    ax2.plot(
        [0], [1],
        marker="o",
        markersize=7,
        markerfacecolor="white",
        markeredgecolor="black"
    )
    ax2.plot(
        [0], [-1],
        marker="o",
        markersize=7,
        markerfacecolor="white",
        markeredgecolor="black"
    )
    ax2.plot([0], [0], marker="o", markersize=7, color="black")

    ax2.axhline(0.0, color="black", linewidth=0.5)

    ax2.set_yticks([-1, 0, 1])
    ax2.set_yticklabels([r"$-\infty$", r"$0$", r"$+\infty$"])
    ax2.set_xlabel(r"direction angle $\theta$")
    ax2.set_ylabel(r"$f_\infty(d)$")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "asymptotic_function.svg")
    plt.close(fig)


# Figure 3: Asymptotic cone
def figure_asymptotic_cone():
    x_min, x_max = -3.2, 3.2
    y_min, y_max = -1.0, 6.2

    x = np.linspace(x_min, x_max, 800)
    y = x**2

    plt.figure(figsize=(4.8, 4.8))

    plt.fill_between(
        x,
        np.minimum(y, y_max),
        y_max,
        where=y <= y_max,
        color="gray",
        alpha=0.25,
        zorder=1
    )

    mask = y <= y_max
    plt.plot(x[mask], y[mask], color="black", linewidth=2, zorder=3)

    axis_arrow = dict(
        width=0.0,
        head_width=0.18,
        head_length=0.22,
        length_includes_head=True,
        color="black",
        zorder=4
    )

    plt.arrow(x_min, 0.0, x_max - x_min, 0.0, **axis_arrow)
    plt.arrow(0.0, y_min, 0.0, y_max - y_min, **axis_arrow)

    plt.arrow(
        0.0, 0.0,
        0.0, y_max - 0.6,
        width=0.10,
        head_width=0.55,
        head_length=0.35,
        length_includes_head=True,
        color="red",
        zorder=5
    )

    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "asymptotic_cone.svg")
    plt.close()


if __name__ == "__main__":
    figure_asymptotic_direction()
    figure_asymptotic_function()
    figure_asymptotic_cone()
