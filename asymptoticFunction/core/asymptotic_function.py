# core/asymptotic_function.py

import numpy as np
from ..core.types import AsymptoticResult, CallableFunction
from ..numerical.approximation import approximateAsymptoticFunc
from ..known.registry import get_analytical_form

__all__ = ["asymptotic_function", "asymptoticFunction"]


def asymptotic_function(f, d, kind=None, params=None, strict_kind=False):
    """
    Parameters
    ----------
    f : callable
        Function whose asymptotic behavior is to be evaluated.
    d : array_like
        Direction at which to compute f∞(d).
    kind : str, optional
        Name of a known analytical form. If provided, uses the registry.
    params : dict, optional
        Extra parameters for analytical or numerical routines.
    strict_kind : bool, default False
        If True, raises an error if the analytical form is not found or fails.

    Returns
    -------
    AsymptoticResult
        Object containing:
        - value : the computed asymptotic value
        - method : "analytical" or "numerical"
        - kind : analytical kind (if any)
    """

    if params is None:
        params = {}

    d = np.asarray(d, dtype=float)
    if np.any(np.isnan(d)):
        raise ValueError("Direction 'd' contains NaNs.")

    func = CallableFunction(f)

    # Try analytical form first
    if kind is not None:
        try:
            form = get_analytical_form(kind)
            if form is None:
                if strict_kind:
                    raise ValueError("No analytical form found for kind=%r." % kind)
            else:
                value = form(func.f, d, **params)
                return AsymptoticResult(value, method="analytical", kind=kind)
        except Exception as err:
            if strict_kind:
                raise ValueError("Analytical form %r failed: %s" % (kind, err))

    # Default: numerical approximation
    value = approximateAsymptoticFunc(func.f, d)
    return AsymptoticResult(value, method="numerical")


asymptoticFunction = asymptotic_function
