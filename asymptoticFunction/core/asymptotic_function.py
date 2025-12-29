# asymptoticFunction/core/asymptotic_function.py

import numpy as np

from ..core.types import AsymptoticResult, CallableFunction
from ..numerical.approximation import approximateAsymptoticFunc

__all__ = ["asymptotic_function", "asymptoticFunction"]


def asymptotic_function(f, d, *, params=None):
    """
    Numerically approximate the asymptotic growth of f along direction d.

    This routine is intentionally:
        - structure-agnostic
        - heuristic-free
        - visualization-neutral

    It performs a numerical probe of the form:
        f_infty(d) ≈ lim f(t d) / t

    Parameters
    ----------
    f : callable or CallableFunction
        Function whose asymptotic behavior is evaluated.
    d : array_like
        Direction vector.
    params : dict, optional
        Passed to the numerical approximation routine.

    Returns
    -------
    AsymptoticResult
        value  : numerical asymptotic estimate
        method : "numerical"
        kind   : None
    """

    if params is None:
        params = {}

    d = np.asarray(d, dtype=float)
    if d.ndim != 1:
        raise ValueError("Direction d must be a 1D vector.")

    if np.any(np.isnan(d)):
        raise ValueError("Direction d contains NaNs.")

    func = f if isinstance(f, CallableFunction) else CallableFunction(f)

    value = approximateAsymptoticFunc(func.f, d, **params)

    return AsymptoticResult(
        value=value,
        method="numerical",
        kind=None,
        params=params
    )


# backwards compatibility alias
asymptoticFunction = asymptotic_function
