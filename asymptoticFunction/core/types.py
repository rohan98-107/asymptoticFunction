from __future__ import annotations
import numpy as np
from typing import Callable


class CallableFunction:
    """
    Lightweight wrapper around a scalar-valued callable f(x).

    Responsibilities:
    - ensure scalar output
    - normalize input to ndarray
    - nothing else
    """

    def __init__(self, f: Callable):
        if not callable(f):
            raise TypeError("Expected a callable f(x).")
        self.f = f

    def __call__(self, x) -> float:
        arr = np.asarray(x, dtype=float)
        val = self.f(arr)

        if isinstance(val, (float, int, np.floating)):
            return float(val)

        val = np.asarray(val, dtype=float)
        if val.size == 1:
            return float(val)

        raise ValueError(
            f"f(x) returned non-scalar value with shape {val.shape}"
        )

    def __repr__(self) -> str:
        return "CallableFunction()"


class AsymptoticResult:
    """
    Container for a numerical asymptotic evaluation.

    NOTE:
    This represents a numerical approximation of f_∞(d),
    not an exact mathematical asymptotic function.
    """

    def __init__(self, value: float):
        self.value = float(value)

    def __repr__(self) -> str:
        return f"AsymptoticResult(value={self.value:.6g})"

    def to_dict(self) -> dict:
        return {
            "value": self.value
        }


def as_vector(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        return arr[None]
    if arr.ndim > 1:
        return arr.reshape(-1)
    return arr
