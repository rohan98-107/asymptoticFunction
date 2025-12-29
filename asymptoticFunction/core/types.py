# asymptoticFunction/core/types.py
"""
Core function containers.

Provides:
- CallableFunction: lightweight wrapper around f(x) with optional
  asymptotic behavior classification ("kind").
- AsymptoticResult: structured container for asymptotic evaluations
  used in visualization and diagnostics.
"""

from __future__ import annotations
import numpy as np
from typing import Callable, Optional


class CallableFunction:
    """
    Wrapper for a scalar-valued callable f(x).

    Parameters
    ----------
    f : callable
        Function mapping R^n -> R
    kind : str, optional
        Known asymptotic behavior class (e.g. 'polynomial').
        Interpreted by the visualization / heuristic registry.
    """

    def __init__(self, f: Callable, *, kind: Optional[str] = None):
        if not callable(f):
            raise TypeError("Expected a callable f(x).")
        self.f = f
        self.kind = kind

    def __call__(self, x) -> float:
        arr = np.asarray(x, dtype=float)
        val = self.f(arr)

        if isinstance(val, (float, int, np.floating)):
            return float(val)

        val = np.asarray(val, dtype=float)
        if val.size == 1:
            return float(val)

        raise ValueError(
            f"f(x) returned non-scalar value with shape {val.shape}."
        )

    def __repr__(self) -> str:
        if self.kind is None:
            return "CallableFunction(kind=None)"
        return f"CallableFunction(kind='{self.kind}')"


class AsymptoticResult:
    """
    Container for an asymptotic evaluation result.

    This is a *visual / heuristic* asymptotic value, not an exact
    mathematical asymptotic function.
    """

    def __init__(
        self,
        value: float,
        *,
        method: str,
        kind: Optional[str] = None
    ):
        self.value = float(value)
        self.method = method  # 'heuristics' or 'numerical'
        self.kind = kind

    def __repr__(self) -> str:
        tag = f"kind='{self.kind}', " if self.kind else ""
        return (
            f"AsymptoticResult({tag}"
            f"value={self.value:.6g}, method='{self.method}')"
        )

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "method": self.method,
            "kind": self.kind
        }


# ------------------------------------------------------------
# Convenience
# ------------------------------------------------------------

def as_vector(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        return arr[None]
    if arr.ndim > 1:
        return arr.reshape(-1)
    return arr
