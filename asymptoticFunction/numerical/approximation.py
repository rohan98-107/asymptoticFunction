from ..core.normalization import normalize_direction
from ..core.utils import *
from ..numerical.sampling import sign_preserving_jitter


def approximateAsymptoticFunc(f, d, verbose=False, tol=1e-3, max_cycles=5):
    d = np.asarray(d, dtype=float)
    d = normalize_direction(d)

    # --- Case 1: Undefined direction ---
    if np.any(np.isnan(d)):
        if verbose:
            print("→ Regime: undefined direction (mixed ±inf or NaN)")
        return np.nan

    # --- Case 2: Infinite direction (+∞ or -∞) ---
    if np.any(np.isinf(d)):
        sign = np.sign(d[np.isinf(d)][0])
        if verbose:
            print(f"→ Regime: {'+∞' if sign > 0 else '-∞'} direction")
        try:
            probe = 1e6 * sign
            val = safe_eval(f, np.full_like(d, probe))
        except Exception:
            val = np.nan

        if np.isposinf(val) and sign > 0:
            return +np.inf
        elif np.isneginf(val) and sign < 0:
            return -np.inf
        else:
            return 0.0

    # --- Case 3: Zero direction ---
    if np.allclose(d, 0):
        if verbose:
            print("→ Regime: zero direction")
        return 0 if is_proper(f, d) else -INFTY

    # --- Case 4: Finite direction (main computation) ---
    if verbose:
        print("→ Regime: finite direction (computing numerical liminf)")

    n = d.size
    magnitude = 10 ** -(n / 4)
    t_min, t_max = 1.0, 1e2
    prev_tail = None
    vals = np.empty
    for cycle in range(max_cycles):
        grid = np.geomspace(t_min, t_max, 100 * n)
        d_prime = sign_preserving_jitter(d, len(grid), magnitude)

        # --- Vectorized evaluation is now default ---
        X = grid[:, None] * d_prime
        vals_raw = safe_eval(f, X)  # shape (len(grid),)
        vals = np.where(np.isfinite(vals_raw), vals_raw / grid, vals_raw)

        # Compute tail statistics
        m = max(1, int(0.2 * len(vals)))
        tail_vals = vals[-m:]
        finite_tail = tail_vals[np.isfinite(tail_vals)]
        if finite_tail.size > 0:
            cur_tail = np.nanmean(finite_tail)
        elif np.any(np.isposinf(tail_vals)):
            cur_tail = +np.inf
        elif np.any(np.isneginf(tail_vals)):
            cur_tail = -np.inf
        else:
            cur_tail = np.nan

        if verbose:
            print(f"Cycle {cycle + 1}: t_max={t_max:.1e}, tail≈{cur_tail:.4g}")

        # Divergence detection
        if np.all(np.isposinf(vals)):
            return +np.inf
        if np.all(np.isneginf(vals)):
            return -np.inf

        # Convergence detection
        if prev_tail is not None and np.isfinite(prev_tail) and np.isfinite(cur_tail):
            rel_change = abs(cur_tail - prev_tail) / (abs(prev_tail) + 1e-12)
            if rel_change < tol:
                break

        prev_tail = cur_tail
        t_min, t_max = t_max, t_max * 10  # grow t-range

    return approximateLimInf(vals)
