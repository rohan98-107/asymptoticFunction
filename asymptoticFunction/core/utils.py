from ..core.config import INFTY, SEED
import numpy as np


def is_proper(f, x0, n_checks=100):
    rng = np.random.default_rng(SEED)

    x0 = np.asarray(x0, dtype=float)

    val0 = f(x0)

    if val0 <= -INFTY:
        return False

    if np.isfinite(val0):
        return True

    n = x0.size if x0.ndim > 0 else 1

    for _ in range(n_checks):

        x = rng.normal(size=n)

        val = f(x)

        if np.isfinite(val):
            return True

        if val == -np.inf:
            return False

    return False


def approximateLimInf(sequence):
    """Approximate the liminf of a numerical sequence (ignoring NaNs)."""

    seq = np.asarray(sequence, dtype=float)

    seq = seq[np.isfinite(seq) | np.isinf(seq)]  # drop only NaNs

    if seq.size == 0:
        return np.nan

    infima_of_tails = [np.nanmin(seq[i:]) for i in range(len(seq))]

    return np.nanmax(infima_of_tails)


def safe_eval(f, x, magnitude_factor=1e40):

    x = np.asarray(x, dtype=float)
    dim = x.shape[-1] if x.ndim > 0 else 1
    tol = magnitude_factor * (10 ** (dim / 2))

    try:
        # Try vectorized evaluation first
        val = f(x)

        # --- Minimal, universal fallback for batched inputs ---
        if x.ndim == 2:
            # if f returned scalar or wrong-length array → evaluate row-wise
            if (not isinstance(val, np.ndarray)) or (val.ndim == 0) or (val.shape[0] != x.shape[0]):
                val = np.array([f(xi) for xi in x], dtype=float)

    except ZeroDivisionError:
        shape = x.shape[:-1] if x.ndim > 1 else ()
        return np.full(shape, np.inf)

    except OverflowError:
        shape = x.shape[:-1] if x.ndim > 1 else ()
        return np.full(shape, np.inf)

    except (FloatingPointError, ValueError, TypeError):
        shape = x.shape[:-1] if x.ndim > 1 else ()
        return np.full(shape, np.nan)

    except Exception:
        if isinstance(x, np.ndarray) and x.ndim == 2:
            out = []
            for xi in x:
                try:
                    out.append(safe_eval(f, xi, magnitude_factor))
                except ZeroDivisionError:
                    out.append(np.inf)
                except OverflowError:
                    out.append(np.inf)
                except Exception:
                    out.append(np.nan)
            return np.array(out, dtype=float)
        raise

    val = np.asarray(val, dtype=float)
    val[np.isnan(val)] = np.nan

    mask = np.isfinite(val) & (np.abs(val) > tol)
    if np.any(mask):
        val[mask] = np.sign(val[mask]) * np.inf

    if val.ndim == 0:
        return float(val)
    return val


def make_vectorization_safe(f):
    """
    Example
    -------
        >>> a = np.array([1., 2.])
        >>> f = lambda x: np.dot(a, x) + np.linalg.norm(x)
        >>> f_safe = make_vectorization_safe(f)
        >>> X = np.array([[1,2], [3,4]])
        >>> f_safe(X)
    array([5.236..., 10.650...])
    """

    def wrapped(X):
        X = np.atleast_2d(X)
        out = [f(xi) for xi in X]
        return np.array(out, dtype=float)

    return wrapped
