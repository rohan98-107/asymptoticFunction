# asymptoticFunction/heuristics/exact_forms.py

# These functions implement exact asymptotic definitions.
# They are NOT stable under perturbations and NOT suitable
# for visualization or sampling-based geometry.

from __future__ import annotations
import numpy as np
from typing import Callable, Sequence, Optional


def _as_vector(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        arr = arr[None]
    elif arr.ndim > 1 and 1 in arr.shape:
        arr = arr.reshape(-1)
    elif arr.ndim > 1:
        raise ValueError(f"Expected 1D vector, got shape {arr.shape}")
    return arr


def _as_matrix(M) -> np.ndarray:
    A = np.asarray(M, dtype=float)
    if A.ndim == 1:
        A = A.reshape(1, -1)
    elif A.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {A.shape}")
    return A


# ============================================================
# Linear / affine / norm families
# ============================================================

def linear_asymptotic(f, d, *, a, **kw):
    a = _as_vector(a)
    d = _as_vector(d)
    if a.shape != d.shape:
        raise ValueError("Shape mismatch in linear_asymptotic")
    return float(np.dot(a, d))


def affine_asymptotic(f, d, *, a, **kw):
    return linear_asymptotic(f, d, a=a)


def norm_asymptotic(f, d, *, p: Optional[float] = 2, **kw):
    d = _as_vector(d)
    return float(np.linalg.norm(d, ord=p))


def weighted_norm_asymptotic(f, d, *, W, p: float = 2.0, **kw):
    W = _as_matrix(W)
    d = _as_vector(d)
    return float(np.linalg.norm(W @ d, ord=p))


def affine_plus_norm_asymptotic(f, d, *, a, c=1.0, **kw):
    a = _as_vector(a)
    d = _as_vector(d)
    return float(np.dot(a, d) + c * np.linalg.norm(d))


def abs_linear_asymptotic(f, d, *, a, **kw):
    a = _as_vector(a)
    d = _as_vector(d)
    return float(abs(np.dot(a, d)))


def max_affine_asymptotic(f, d, *, A, **kw):
    A = _as_matrix(A)
    d = _as_vector(d)
    vals = A @ d
    return float(np.max(vals)) if vals.size else -np.inf


# ============================================================
# Indicator functions
# ============================================================

def indicator_halfspace_asymptotic(f, d, *, a, **kw):
    a = _as_vector(a)
    d = _as_vector(d)
    return 0.0 if np.dot(a, d) <= 0.0 else np.inf


def indicator_hyperplane_asymptotic(f, d, *, a, **kw):
    a = _as_vector(a)
    d = _as_vector(d)
    return 0.0 if np.dot(a, d) == 0.0 else np.inf


def indicator_affine_subspace_asymptotic(f, d, *, A, **kw):
    A = _as_matrix(A)
    d = _as_vector(d)
    return 0.0 if np.all(A @ d == 0.0) else np.inf


def indicator_polyhedron_asymptotic(f, d, *, A, E=None, **kw):
    A = _as_matrix(A)
    d = _as_vector(d)
    if np.any(A @ d > 0.0):
        return np.inf
    if E is not None:
        E = _as_matrix(E)
        if np.any(E @ d != 0.0):
            return np.inf
    return 0.0


def indicator_box_asymptotic(f, d, *, lower=None, upper=None, **kw):
    d = _as_vector(d)
    return 0.0 if np.all(d == 0.0) else np.inf


# ============================================================
# Quadratic
# ============================================================

def quadratic_asymptotic(f, d, *, Q, b=None, **kw):
    Q = _as_matrix(Q)
    d = _as_vector(d)
    q = float(d @ Q @ d)
    if q > 0.0:
        return np.inf
    if q < 0.0:
        return -np.inf
    if b is None:
        return 0.0
    b = _as_vector(b)
    return float(np.dot(b, d))


# ============================================================
# Exact polynomial (decomposed)
# ============================================================

def polynomial_decomposed_asymptotic(f, d, *, phi: Sequence[Callable], **kw):
    d = _as_vector(d)

    mu = None
    for i in range(len(phi) - 1, 0, -1):
        val = phi[i](d)
        if val != 0.0:
            mu = i
            break

    if mu is None:
        return 0.0

    if mu == 1:
        return float(phi[1](d))

    return np.inf if phi[mu](d) > 0.0 else -np.inf
