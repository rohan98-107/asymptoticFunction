from ..core.config import INFTY, EPS
import numpy as np


def normalize_direction(d):
    d = np.asarray(d, dtype=float)

    if np.any(np.isnan(d)):
        return np.full_like(d, np.nan)

    has_pos_inf = np.any(np.isposinf(d))

    has_neg_inf = np.any(np.isneginf(d))

    if has_pos_inf and has_neg_inf:

        return np.full_like(d, np.nan)

    elif has_pos_inf:

        return np.full_like(d, +INFTY)

    elif has_neg_inf:

        return np.full_like(d, -INFTY)

    norm = np.linalg.norm(d)

    if norm < EPS:
        return d  # leave zero vector as is

    return d / norm
