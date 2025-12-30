import numpy as np

from ..numerical.approximation import approximateAsymptoticFunc
from ..numerical.sampling import sample_sphere
from ..core.types import CallableFunction


def negate(func):
    return lambda x: -func(x)


class AsymptoticCone:

    def __init__(self, f_list=None, g_list=None, x0=None, dim=None):
        f_list = f_list or []
        g_list = g_list or []

        if x0 is None and dim is None:
            raise ValueError("Provide either x0 or dim.")

        self.x0 = np.zeros(dim) if x0 is None else np.asarray(x0, float)
        self.dim = dim if dim is not None else len(self.x0)

        self.f_list = [self._ensure_callable_function(s) for s in f_list]
        self.g_list = [self._ensure_callable_function(s) for s in g_list]

        self.directions = None
        self.F_inf = None
        self.G_inf = None
        self.results = {}

    def _ensure_callable_function(self, spec):
        if isinstance(spec, CallableFunction):
            return spec
        if callable(spec):
            return CallableFunction(spec)
        raise TypeError(f"Constraint must be callable or CallableFunction, got {type(spec).__name__}")

    def _empty_dirs(self):
        return np.empty((0, self.dim), dtype=float)

    def _ensure_2d(self, arr):
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1:
            return arr.reshape(1, self.dim)
        return arr

    def _compute_asymptotic_values(self, dirs):
        n_dirs = len(dirs)
        n_f = len(self.f_list)
        n_g = len(self.g_list)

        F_inf = np.full((n_f, n_dirs), np.nan) if n_f > 0 else np.empty((0, n_dirs))
        G_inf = np.full((2 * n_g, n_dirs), np.nan) if n_g > 0 else np.empty((0, n_dirs))

        for i, cf in enumerate(self.f_list):
            for k, d in enumerate(dirs):
                F_inf[i, k] = approximateAsymptoticFunc(cf.f, d)

        for j, cf in enumerate(self.g_list):
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
                for cf in self.f_list:
                    for t in t_grid:
                        if cf(self.x0 + t * d) > tol:
                            feas = False
                            break
                    if not feas:
                        break

                eqs = True
                for cf in self.g_list:
                    for t in t_grid:
                        if abs(cf(self.x0 + t * d)) > tol:
                            eqs = False
                            break
                    if not eqs:
                        break

                if feas and eqs:
                    feasible.append(d)

            keep = self._empty_dirs() if len(feasible) == 0 else self._ensure_2d(feasible)
            self.results[mode] = keep
            return keep

        if mode != "intersection":
            raise ValueError(f"Unknown mode '{mode}'")

        F_inf, G_inf = self._compute_asymptotic_values(dirs)
        self.F_inf = F_inf
        self.G_inf = G_inf

        mask = np.all(F_inf <= tol, axis=0) & np.all(G_inf <= tol, axis=0)
        keep = dirs[mask]
        keep = self._empty_dirs() if keep.size == 0 else self._ensure_2d(keep)

        self.results[mode] = keep
        return keep

    def get(self, mode):
        return self.results.get(mode)
