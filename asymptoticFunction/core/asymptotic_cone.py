import numpy as np

from ..numerical.approximation import approximateAsymptoticFunc
from ..numerical.sampling import sample_sphere
from ..core.types import CallableFunction


def negate(func):
    return lambda x: -func(x)


class AsymptoticCone:

    def __init__(self, f_list=None, g_list=None, x0=None, dim=None):
        self.f_list = f_list or []
        self.g_list = g_list or []
        self.x0 = np.zeros(dim) if x0 is None else np.asarray(x0, float)
        self.dim = dim if dim is not None else len(self.x0)

        self.directions = None
        self.F_inf = None
        self.G_inf = None
        self.results = {}

    def _empty_dirs(self):
        return np.empty((0, self.dim), dtype=float)

    def _ensure_2d(self, arr):
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1:
            return arr.reshape(1, self.dim)
        return arr

    def _resolve_constraint(self, spec):
        if isinstance(spec, CallableFunction):
            return spec

        if isinstance(spec, dict):
            return CallableFunction(
                spec.get("func"),
                kind=spec.get("kind"),
                params={k: v for k, v in spec.items() if k not in ("func", "kind")}
            )

        if callable(spec):
            return CallableFunction(spec)

        raise TypeError(f"Constraint must be callable or CallableFunction, got {type(spec).__name__}")

    def _compute_asymptotic_values(self, dirs):
        n_dirs = len(dirs)
        n_f = len(self.f_list)
        n_g = len(self.g_list)

        F_inf = np.full((n_f, n_dirs), np.nan) if n_f > 0 else np.empty((0, n_dirs))
        G_inf = np.full((2 * n_g, n_dirs), np.nan) if n_g > 0 else np.empty((0, n_dirs))

        for i, f_spec in enumerate(self.f_list):
            cf = self._resolve_constraint(f_spec)
            for k, d in enumerate(dirs):
                F_inf[i, k] = approximateAsymptoticFunc(cf.f, d)

        for j, g_spec in enumerate(self.g_list):
            cf = self._resolve_constraint(g_spec)
            g_neg = negate(cf.f)
            for k, d in enumerate(dirs):
                G_inf[2 * j, k] = approximateAsymptoticFunc(cf.f, d)
                G_inf[2 * j + 1, k] = approximateAsymptoticFunc(g_neg, d)

        return F_inf, G_inf

    def compute(self, mode="intersection", n_samples=500, tol=1e-6, sampling="sobol", t_grid=None):
        dirs = sample_sphere(n_samples, self.dim, method=sampling)
        self.directions = dirs

        if mode == "numerical":
            if t_grid is None:
                t_grid = np.geomspace(1, 1e3, 8)

            feasible = []

            for d in dirs:
                feas = True
                for spec in self.f_list:
                    f = spec["func"] if isinstance(spec, dict) else spec
                    for t in t_grid:
                        if f(self.x0 + t * d) > tol:
                            feas = False
                            break
                    if not feas:
                        break

                eqs = True
                for spec in self.g_list:
                    g = spec["func"] if isinstance(spec, dict) else spec
                    for t in t_grid:
                        if abs(g(self.x0 + t * d)) > tol:
                            eqs = False
                            break
                    if not eqs:
                        break

                if feas and eqs:
                    feasible.append(d)

            if len(feasible) == 0:
                keep = self._empty_dirs()
            else:
                keep = self._ensure_2d(feasible)

            self.results[mode] = keep
            return keep

        if mode != "intersection" and mode != "outer_union":
            raise ValueError(f"Unknown mode '{mode}'")

        F_inf, G_inf = self._compute_asymptotic_values(dirs)
        self.F_inf = F_inf
        self.G_inf = G_inf

        if mode == "intersection":
            mask = np.all(F_inf <= tol, axis=0) & np.all(np.abs(G_inf) <= tol, axis=0)
        else:
            mask = np.all(F_inf <= tol, axis=0) & np.all(G_inf <= tol, axis=0)

        keep = dirs[mask]

        if keep.size == 0:
            keep = self._empty_dirs()
        else:
            keep = self._ensure_2d(keep)

        self.results[mode] = keep
        return keep

    def get(self, mode):
        return self.results.get(mode)
