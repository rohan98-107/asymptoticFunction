import numpy as np
import warnings

from ..numerical.approximation import approximateAsymptoticFunc
from ..numerical.sampling import sample_sphere
from ..known.registry import get_analytical_form
from ..core.types import CallableFunction


def negate(func):
    return lambda x: -func(x)


class AsymptoticCone:
    """
    Compute and store the asymptotic cone of a constraint-defined set:

        X = { x | f_i(x) <= 0,  g_j(x) = 0 }.

    Modes:
        - 'intersection' : analytical/theoretical cone (all constraints active)
        - 'outer_union'  : outer approximation (any constraint active)
        - 'numerical'    : empirical escape cone (ray simulation)
    """

    def __init__(self, f_list=None, g_list=None, x0=None, dim=None):
        self.f_list = f_list or []
        self.g_list = g_list or []
        self.x0 = np.zeros(dim) if x0 is None else np.asarray(x0, float)
        self.dim = dim if dim is not None else len(self.x0)

        self.directions = None
        self.F_inf = None
        self.G_inf = None
        self.results = {}

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
        """
        Returns
        -------
        F_inf : (len(f_list), n_dirs)
        G_inf : (2 * len(g_list), n_dirs)
        """
        n_dirs = len(dirs)
        n_f = len(self.f_list)
        n_g = len(self.g_list)

        F_inf = np.full((n_f, n_dirs), np.nan) if n_f > 0 else np.empty((0, n_dirs))
        G_inf = np.full((2 * n_g, n_dirs), np.nan) if n_g > 0 else np.empty((0, n_dirs))

        registry_cache = {}

        for i, f_spec in enumerate(self.f_list):
            cf = self._resolve_constraint(f_spec)
            analytical = None
            kind = cf.kind

            if kind is not None:
                if kind in registry_cache:
                    analytical = registry_cache[kind]
                else:
                    try:
                        analytical = get_analytical_form(kind)
                        if callable(analytical):
                            registry_cache[kind] = analytical
                        else:
                            warnings.warn(f"Registry entry for kind='{kind}' not callable. Using numerical.")
                            analytical = None
                    except Exception:
                        warnings.warn(f"kind='{kind}' not found in registry. Using numerical.")
                        analytical = None

            for k, d in enumerate(dirs):
                val = analytical(cf.f, d, **cf.params) if analytical else approximateAsymptoticFunc(cf.f, d)
                F_inf[i, k] = val

        for j, g_spec in enumerate(self.g_list):
            cf = self._resolve_constraint(g_spec)
            analytical = None
            kind = cf.kind

            if kind is not None:
                if kind in registry_cache:
                    analytical = registry_cache[kind]
                else:
                    try:
                        analytical = get_analytical_form(kind)
                        if callable(analytical):
                            registry_cache[kind] = analytical
                        else:
                            warnings.warn(f"Registry entry for kind='{kind}' not callable. Using numerical.")
                            analytical = None
                    except Exception:
                        warnings.warn(f"kind='{kind}' not found in registry. Using numerical.")
                        analytical = None

            for k, d in enumerate(dirs):
                val_pos = analytical(cf.f, d, **cf.params) if analytical else approximateAsymptoticFunc(cf.f, d)
                g_neg = negate(cf.f)
                val_neg = analytical(g_neg, d, **cf.params) if analytical else approximateAsymptoticFunc(g_neg, d)

                G_inf[2 * j, k] = val_pos
                G_inf[2 * j + 1, k] = val_neg

        return F_inf, G_inf

    # ------------------------------------------------------------
    def compute(self, mode="intersection", n_samples=500, tol=1e-6,
                sampling="sobol", t_grid=None):
        """
        sampling = ['sobol', 'fibonacci', 'normal']
        mode = ['intersection', 'outer_union', 'numerical']
        """
        dirs = sample_sphere(n_samples, self.dim, method=sampling)
        self.directions = dirs

        if mode in ("intersection", "outer_union"):
            F_inf, G_inf = self._compute_asymptotic_values(dirs)
            self.F_inf, self.G_inf = F_inf, G_inf

            if mode == "intersection":
                mask = np.all(F_inf <= tol, axis=0) & np.all(np.abs(G_inf) <= tol, axis=0)
            else:
                mask = np.all(F_inf <= tol, axis=0) & np.all(G_inf <= tol, axis=0)

            keep = dirs[mask]

        elif mode == "numerical":
            if t_grid is None:
                t_grid = np.geomspace(1, 1e3, 8)

            feasible = []
            for d in dirs:
                feas = all(
                    all((spec["func"](self.x0 + t * d) if isinstance(spec, dict)
                         else spec(self.x0 + t * d)) <= tol for t in t_grid)
                    for spec in self.f_list
                )
                eqs = all(
                    all(abs(spec["func"](self.x0 + t * d) if isinstance(spec, dict)
                            else spec(self.x0 + t * d)) <= tol for t in t_grid)
                    for spec in self.g_list
                )
                if feas and eqs:
                    feasible.append(d)
            keep = np.array(feasible)

        else:
            raise ValueError(f"Unknown mode '{mode}'")

        self.results[mode] = keep
        return keep

    def get(self, mode):
        """Return precomputed cone directions if available."""
        return self.results.get(mode)
