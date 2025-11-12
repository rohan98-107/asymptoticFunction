import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from asymptoticFunction.core.asymptotic_cone import AsymptoticCone


def make_constraints(case):
    if case == "A":
        # Halfspace: x2 <= x1
        f = [lambda x: x[1] - x[0]]
        g = []
        xlim, ylim = (-2, 2), (-2, 2)
    elif case == "B":
        # Quadrant: x1 <= 0, x2 >= 0
        f = [lambda x: x[0], lambda x: -x[1]]
        g = []
        xlim, ylim = (-2, 2), (-2, 2)
    elif case == "C":
        # Epigraph of parabola: x2 >= x1^2
        f = [lambda x: -x[1] + x[0] ** 2]
        g = []
        xlim, ylim = (-2, 2), (-1, 4)
    else:
        raise ValueError("Unknown case.")
    return f, g, xlim, ylim


def add_axes(fig, lim, row, col):
    axis_line = dict(color="black", width=1)
    fig.add_trace(go.Scatter(x=[-lim, lim], y=[0, 0],
                             mode="lines", line=axis_line,
                             hoverinfo="skip", showlegend=False),
                  row=row, col=col)
    fig.add_trace(go.Scatter(x=[0, 0], y=[-lim, lim],
                             mode="lines", line=axis_line,
                             hoverinfo="skip", showlegend=False),
                  row=row, col=col)
    fig.add_trace(go.Scatter(x=[0], y=[0],
                             mode="markers",
                             marker=dict(color="black", size=10),
                             hoverinfo="skip", showlegend=False),
                  row=row, col=col)


def shade_set(fig, case, xlim, ylim, row, col):
    light_gray = "rgb(180,180,180)"

    if case == "A":
        fig.add_trace(go.Scatter(
            x=[xlim[0], xlim[1], xlim[1]],
            y=[ylim[0], ylim[0], xlim[1]],
            fill="toself",
            fillcolor=light_gray,
            line=dict(color="rgba(0,0,0,0)", width=0),
            hoverinfo="skip",
            showlegend=False,
        ), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=[xlim[0], xlim[1]],
            y=[xlim[0], xlim[1]],
            mode="lines",
            line=dict(color="black", width=3),
            hoverinfo="skip",
            showlegend=False,
        ), row=row, col=col)

    elif case == "B":
        # fill the full NW quadrant: x in [xlim[0], 0], y in [0, ylim[1]]
        fig.add_trace(go.Scatter(
            x=[xlim[0], 0, 0, xlim[0]],
            y=[0, 0, ylim[1], ylim[1]],
            fill="toself",
            fillcolor=light_gray,
            line=dict(color="rgba(0,0,0,0)", width=0),  # no outer box edges
            hoverinfo="skip",
            showlegend=False,
        ), row=row, col=col)

        # redraw the two actual boundaries on top so they're visible
        # negative x-axis: y = 0, xlim[0] → 0
        fig.add_trace(go.Scatter(
            x=[xlim[0], 0],
            y=[0, 0],
            mode="lines",
            line=dict(color="black", width=3),
            hoverinfo="skip",
            showlegend=False,
        ), row=row, col=col)

        # positive y-axis: x = 0, 0 → ylim[1]
        fig.add_trace(go.Scatter(
            x=[0, 0],
            y=[0, ylim[1]],
            mode="lines",
            line=dict(color="black", width=3),
            hoverinfo="skip",
            showlegend=False,
        ), row=row, col=col)

    elif case == "C":
        x = np.linspace(*xlim, 400)
        y_parab = x ** 2
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([y_parab, np.full_like(x, ylim[1])]),
            fill="toself",
            fillcolor=light_gray,
            line=dict(color="rgba(0,0,0,0)", width=0),
            hoverinfo="skip",
            showlegend=False,
        ), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=x,
            y=y_parab,
            mode="lines",
            line=dict(color="black", width=3),
            hoverinfo="skip",
            showlegend=False,
        ), row=row, col=col)


def cone_slices_traces(dirs, line_color, fill_color,
                       angle_gap=np.deg2rad(8), scale=1.7):
    """
    Group contiguous angles into continuous cone wedges.
    Lightly fill cone interiors with same hue as boundary.
    """
    if dirs is None or len(dirs) == 0:
        return []

    dirs = np.array([d / np.linalg.norm(d) for d in dirs])
    angles = np.arctan2(dirs[:, 1], dirs[:, 0])
    order = np.argsort(angles)
    dirs, angles = dirs[order], angles[order]

    clusters, current = [], [0]
    for i in range(1, len(angles)):
        if angles[i] - angles[i - 1] > angle_gap:
            clusters.append(current)
            current = [i]
        else:
            current.append(i)
    clusters.append(current)

    traces = []
    for idxs in clusters:
        if len(idxs) < 2:
            continue
        pts = dirs[idxs] * scale
        xs = np.concatenate([[0], pts[:, 0], [0]])
        ys = np.concatenate([[0], pts[:, 1], [0]])

        traces.append(go.Scatter(
            x=xs, y=ys,
            fill="toself",
            fillcolor=fill_color,  # translucent cone interior
            line=dict(color=line_color, width=3),
            mode="lines",
            hoverinfo="skip",
            showlegend=False,
        ))
    return traces


def main(show=True, save_path=None):
    cases = ["A", "B", "C"]
    cone_modes = [
        ("Outer-union", "outer_union", "rgba(213,0,0,1)", "rgba(213,0,0,0.25)"),
        ("Intersection", "intersection", "rgba(0,114,178,1)", "rgba(0,114,178,0.25)"),
        ("Numerical", "numerical", "rgba(0,158,115,1)", "rgba(0,158,115,0.35)"),
    ]

    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=[f"{label} ({case})"
                        for case in cases for label, *_ in cone_modes],
        horizontal_spacing=0.03,
        vertical_spacing=0.08,
    )

    for i, case in enumerate(cases, start=1):
        f, g, xlim, ylim = make_constraints(case)
        cone = AsymptoticCone(f_list=f, g_list=g, dim=2)
        cone.x0 = np.zeros(2)

        results = {
            "outer_union": cone.compute(mode="outer_union", n_samples=800, sampling="sobol", tol=1e-9),
            "intersection": cone.compute(mode="intersection", n_samples=800, sampling="sobol", tol=1e-9),
            "numerical": cone.compute(mode="numerical", n_samples=800, sampling="sobol", tol=1e-9),
        }

        for j, (label, mode, line_c, fill_c) in enumerate(cone_modes, start=1):
            add_axes(fig, lim=max(abs(xlim[0]), xlim[1], abs(ylim[0]), ylim[1]) + 0.3,
                     row=i, col=j)
            shade_set(fig, case, xlim, ylim, row=i, col=j)
            dirs = results[mode]
            for tr in cone_slices_traces(dirs, line_c, fill_c):
                fig.add_trace(tr, row=i, col=j)

    fig.update_layout(
        width=1800, height=1900,
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        title=dict(
            text="Asymptotic Cones of Closed Sets — Polyhedral & Epigraph Examples",
            x=0.5, xanchor="center",
            font=dict(size=22, family="Times New Roman"),
        ),
        margin=dict(l=20, r=20, t=100, b=40),
    )

    if save_path:
        fig.write_image(save_path, scale=2)
        print(f"Figure saved to: {save_path}")
    if show:
        fig.show()


if __name__ == "__main__":
    main(show=False, save_path="../tests/cones_demo.png")
