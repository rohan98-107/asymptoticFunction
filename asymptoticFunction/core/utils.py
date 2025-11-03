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

        if x.ndim == 2 and val.ndim == 0:
            val = np.full(x.shape[0], val)
        elif x.ndim == 2 and val.ndim == 1 and val.shape[0] != x.shape[0]:
            # shape mismatch: try row-wise evaluation
            out = []
            for xi in x:
                try:
                    out.append(f(xi))
                except Exception:
                    out.append(np.nan)
            val = np.array(out, dtype=float)


    except ZeroDivisionError:
        # True divide-by-zero → inf
        shape = x.shape[:-1] if x.ndim > 1 else ()
        return np.full(shape, np.inf)

    except OverflowError:
        # Numeric overflow → inf
        shape = x.shape[:-1] if x.ndim > 1 else ()
        return np.full(shape, np.inf)

    except (FloatingPointError, ValueError, TypeError):
        # Generic numeric or dtype issue → nan
        shape = x.shape[:-1] if x.ndim > 1 else ()
        return np.full(shape, np.nan)

    except Exception:
        # Function not vectorized → evaluate row-wise
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
        raise  # re-raise unknown programming errors

    val = np.asarray(val, dtype=float)
    val[np.isnan(val)] = np.nan

    # Promote extremely large finite values to ±inf
    mask = np.isfinite(val) & (np.abs(val) > tol)
    if np.any(mask):
        val[mask] = np.sign(val[mask]) * np.inf

    # Scalar normalization
    if val.ndim == 0:
        return float(val)
    return val
