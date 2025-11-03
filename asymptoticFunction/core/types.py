# Define CallableFunction and AsymptoticResult containers here
# asymptoticFunction/core/types.py
"""
Provides:
- CallableFunction: small wrapper ensuring any f(x) -> float behaves safely
- AsymptoticResult: structured return type storing value + metadata
- as_vector(): helper for consistent 1-D coercion
"""

from __future__ import annotations
import numpy as np
from typing import Any, Callable, Optional


class CallableFunction:

    def __init__(self, f):
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
        raise ValueError(f"f(x) returned non-scalar value with shape {val.shape}.")


class AsymptoticResult:

    def __init__(self, value, *, method, kind=None, params=None):
        self.value = float(value)
        self.method = method  # "analytical" or "numerical"
        self.kind = kind  # e.g., "linear", "quadratic"
        self.params = params or {}

    def __repr__(self) -> str:
        tag = f"kind='{self.kind}', " if self.kind else ""
        return f"AsymptoticResult({tag}value={self.value:.6g}, method='{self.method}')"

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "method": self.method,
            "kind": self.kind,
            "params": self.params,
        }


# convenience

def as_vector(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        arr = arr[None]
    elif arr.ndim > 1:
        arr = arr.reshape(-1)
    return arr
