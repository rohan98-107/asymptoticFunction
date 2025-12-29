# asymptoticFunction/heuristics/polynomial_heuristics.py

import numpy as np


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
