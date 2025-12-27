# asymptoticFunction/known/analytical_forms.py
"""
Analytical (closed-form) asymptotic functions f_∞ for standard classes
using definition  f_∞(d) = liminf_{d'→d, t→∞} f(t d') / t.
"""

from __future__ import annotations
import numpy as np
import sympy as sp
from typing import Callable, Optional


# =====================================================================
# Utilities
# =====================================================================

def _as_vector(x) -> np.ndarray:
    """Coerce any 1D-like input to (n,) float array."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        arr = arr[None]
    elif arr.ndim > 1 and 1 in arr.shape:
        arr = arr.reshape(-1)
    elif arr.ndim > 1:
        raise ValueError(f"Expected 1D vector, got shape {arr.shape}.")
    return arr


def _as_matrix(M) -> np.ndarray:
    """Coerce input to 2D matrix float array."""
    A = np.asarray(M, dtype=float)
    if A.ndim == 1:
        A = A.reshape(1, -1)
    elif A.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {A.shape}.")
    return A


def _dot_safe(a, d):
    """Compute dot products robustly for d possibly stacked."""
    if d.ndim == 1:
        return float(np.dot(a, d))
    return np.dot(d, a)  # shape (k,)


def _decompose_polynomial(f, d, max_deg=6):
    d = np.asarray(d, dtype=float)

    t_vals = [1.0, 2.0, 4.0, 8.0]
    vals = []

    for t in t_vals:
        v = f(t * d)
        v = float(np.asarray(v).squeeze())
        vals.append(v)

    vals = np.array(vals, dtype=float)

    if not np.any(np.abs(vals) > 1e-12):
        return 0, 0

    abs_vals = np.abs(vals)

    for k in range(max_deg, 1, -1):
        scaled = abs_vals / (np.array(t_vals) ** k)
        if np.all(scaled > 1e-8):
            sign = np.sign(vals[-1])
            return k, int(sign)

    ratios = vals[-1] / t_vals[-1]
    if abs(ratios) > 1e-12:
        return 1, int(np.sign(ratios))

    return 0, 0


# =====================================================================
# 1. Linear / affine / norm families
# =====================================================================

def linear_asymptotic(f, d, *, a, **kw):
    a = _as_vector(a)
    d = _as_vector(d)
    if a.shape != d.shape:
        raise ValueError(f"linear_asymptotic: a{a.shape} vs d{d.shape}")
    return _dot_safe(a, d)


def affine_asymptotic(f, d, *, a, **kw):
    return linear_asymptotic(f, d, a=a)


def norm_asymptotic(f, d, *, p: Optional[float] = 2, **kw):
    d = np.asarray(d, dtype=float)
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


# =====================================================================
# 2. Indicator functions for common convex sets
# =====================================================================

def indicator_halfspace_asymptotic(f, d, *, a, **kw):
    a = _as_vector(a)
    d = _as_vector(d)
    return 0.0 if np.dot(a, d) <= 0 else np.inf


def indicator_hyperplane_asymptotic(f, d, *, a, **kw):
    a = _as_vector(a)
    d = _as_vector(d)
    return 0.0 if abs(np.dot(a, d)) <= 1e-12 else np.inf


def indicator_affine_subspace_asymptotic(f, d, *, A, **kw):
    A = _as_matrix(A)
    d = _as_vector(d)
    return 0.0 if np.allclose(A @ d, 0.0) else np.inf


def indicator_polyhedron_asymptotic(f, d, *, A, E=None, **kw):
    A = _as_matrix(A)
    d = _as_vector(d)
    cond1 = np.all(A @ d <= 1e-12)
    cond2 = True
    if E is not None:
        E = _as_matrix(E)
        cond2 = np.allclose(E @ d, 0.0)
    return 0.0 if (cond1 and cond2) else np.inf


def indicator_box_asymptotic(f, d, *, lower=None, upper=None, **kw):
    d = _as_vector(d)
    if np.allclose(d, 0.0):
        return 0.0
    return np.inf


def indicator_cone_asymptotic(f, d, *, cone_membership: Callable, **kw):
    """
    need to implement
    """
    return None


# =====================================================================
# 3. Distance and support-type functions
# =====================================================================

def support_function_asymptotic(f, d, *, C_points, **kw):
    """σ_C(x)=max_z xᵀz ⇒ σ_C,∞(d)=σ_C(d)."""
    V = _as_matrix(C_points)
    d = _as_vector(d)
    vals = V @ d
    return float(np.max(vals)) if vals.size else -np.inf


def distance_cone_asymptotic(f, d, *, proj_K: Callable, **kw):
    """dist(x,K), K a cone ⇒ dist_∞(d)=||d-P_K(d)||."""
    d = _as_vector(d)
    p = np.asarray(proj_K(d), dtype=float)
    if p.shape != d.shape:
        raise ValueError("Projection shape mismatch.")
    return float(np.linalg.norm(d - p, 2))


# =====================================================================
# 4. Quadratic / polynomial families
# =====================================================================

def quadratic_asymptotic(f, d, *, Q, b=None, **kw):
    Q = np.asarray(Q, dtype=float)
    d = _as_vector(d)
    q = float(d @ Q @ d)
    b = np.zeros_like(d) if b is None else _as_vector(b)
    if q > 0:
        return np.inf
    if q < 0:
        return -np.inf
    return float(np.dot(b, d))


def polynomial_asymptotic(f, d, max_deg=6):
    mu, sign = _decompose_polynomial(f, d, max_deg=max_deg)

    if mu == 0:
        return 0.0
    if mu == 1:
        return float(f(d))
    if sign > 0:
        return np.inf
    if sign < 0:
        return -np.inf
    return 0.0


# =====================================================================
# 5. ML losses (hinge, logistic, huber, exponential)
# =====================================================================

def hinge_sum_asymptotic(f, d, *, A, y, **kw):
    A = _as_matrix(A)
    y = _as_vector(y)
    d = _as_vector(d)
    z = A @ d
    return float(np.sum(np.maximum(-y * z, 0.0)))


def logistic_sum_asymptotic(f, d, *, A, y=None, **kw):
    A = _as_matrix(A)
    d = _as_vector(d)
    z = A @ d
    if y is None:
        return float(np.sum(np.maximum(z, 0.0)))
    y = _as_vector(y)
    return float(np.sum(np.maximum(-y * z, 0.0)))


def huber_sum_asymptotic(f, d, *, A, delta=1.0, **kw):
    A = _as_matrix(A);
    d = _as_vector(d)
    z = A @ d
    return float(delta * np.sum(np.abs(z)))


def exponential_sum_asymptotic(f, d, *, A, y, **kw):
    A = _as_matrix(A);
    y = _as_vector(y);
    d = _as_vector(d)
    margins = y * (A @ d)
    return np.inf if np.any(margins < 0.0) else 0.0


# =====================================================================
# 6. Sum of exponentials / log-sum-exp
# =====================================================================

def sum_exp_asymptotic(f, d, *, C, w=None, b=None, **kw):
    """
    f(x)=∑ w_i exp(c_i^T x + b_i).
    With nonnegative weights ⇒ f_∞(d)=+∞ if max_i c_i^T d>0 else 0.
    """
    C = _as_matrix(C)
    d = _as_vector(d)
    vals = C @ d
    if w is not None:
        w = np.asarray(w, dtype=float)
        if w.shape[0] != vals.shape[0]:
            raise ValueError("Weights length mismatch.")
    if np.max(vals) > 0:
        return np.inf
    return 0.0


def log_sum_exp_asymptotic(f, d, *, C, **kw):
    """f(x)=log∑exp(c_i^T x + b_i) ⇒ f_∞(d)=max_i c_i^T d."""
    C = _as_matrix(C)
    d = _as_vector(d)
    vals = C @ d
    return float(np.max(vals))
