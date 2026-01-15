# asymptoticFunction/heuristics/polynomial_heuristics.py

import numpy as np
from typing import Sequence, Callable


def _as_vector(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        arr = arr[None]
    elif arr.ndim > 1 and 1 in arr.shape:
        arr = arr.reshape(-1)
    elif arr.ndim > 1:
        raise ValueError(f"Expected 1D vector, got shape {arr.shape}")
    return arr


def detect_polynomial_growth(f, d, *, n_steps=20, t0=1.0, growth_factor=2.0):
    d = np.asarray(d, dtype=float)

    t = t0
    ratios = []

    for _ in range(n_steps):
        val = f(t * d)
        val = float(np.asarray(val).squeeze())
        ratios.append(val / t)
        t *= growth_factor

    ratios = np.asarray(ratios)
    tail = ratios[-5:]

    if np.all(tail > 0.0) and np.all(np.diff(tail) > 0.0):
        return "pos_inf", None

    if np.all(tail < 0.0) and np.all(np.diff(tail) < 0.0):
        return "neg_inf", None

    if np.max(np.abs(np.diff(tail))) < 1e-6:
        return "linear", float(np.mean(tail))

    if np.max(np.abs(tail)) < 1e-6:
        return "zero", 0.0

    return "undetermined", tail[-1]


def polynomial_empirical_asymptotic(f, d, **kw):
    regime, value = detect_polynomial_growth(f, d, **kw)

    if regime == "pos_inf":
        return np.inf

    if regime == "neg_inf":
        return -np.inf

    if regime == "linear":
        return value

    if regime == "zero":
        return 0.0

    return value


def polynomial_thickened_asymptotic(f, d, *, phi: Sequence[Callable], tol=0.0, **kw):
    d = _as_vector(d)

    mu = None
    mu_val = 0.0

    for i in range(len(phi) - 1, 0, -1):
        val = float(phi[i](d))
        if abs(val) > tol:
            mu = i
            mu_val = val

    if mu is None:
        return 0.0

    if mu == 1:
        return float(phi[1](d))

    return np.inf if mu_val > 0.0 else -np.inf
