import numpy as np

from ..core.types import CallableFunction, AsymptoticResult
from ..numerical.approximation import approximateAsymptoticFunc

__all__ = ["asymptotic_function"]


def asymptotic_function(f, d) -> AsymptoticResult:
    """
    Parameters
    ----------
    f : callable or CallableFunction
    d : array_like

    Returns
    -------
    AsymptoticResult
    """

    func = f if isinstance(f, CallableFunction) else CallableFunction(f)
    d = np.asarray(d, dtype=float)

    if np.any(np.isnan(d)):
        raise ValueError("Direction contains NaNs.")

    value = approximateAsymptoticFunc(func.f, d)
    return AsymptoticResult(value)
