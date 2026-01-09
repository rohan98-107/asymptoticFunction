from ..core.normalization import normalize_direction
from ..core.utils import *
from ..numerical.sampling import sign_preserving_jitter


def approximateAsymptoticFunc(f, d, verbose=False, tol=1e-3, max_cycles=5):
    # Convert the input direction to a NumPy float array.
    d = np.asarray(d, dtype=float)

    # Store the original Euclidean norm of the direction.
    orig_norm = np.linalg.norm(d)

    # Normalize the direction according to package conventions.
    # WARNING: This may introduce NaNs or ±inf for degenerate or unbounded inputs.
    d = normalize_direction(d)

    # --- Case 1: Undefined direction ---
    # If normalization produced NaNs, the direction is not interpretable
    if np.any(np.isnan(d)):
        if verbose:
            print("→ Undefined direction (mixed ±inf or NaN)")
        # Return NaN to explicitly signal an undefined asymptotic value.
        return np.nan

    # --- Case 2: Infinite direction (+∞ or -∞) ---
    # If any component of the direction is infinite, the usual limit is not well-defined numerically.
    if np.any(np.isinf(d)):
        # Extract the sign of the first infinite component.
        sign = np.sign(d[np.isinf(d)][0])

        if verbose:
            # Sign determines whether we are probing a +∞ or −∞ direction.
            print(f"→ {'+∞' if sign > 0 else '-∞'} direction")

        try:
            # Construct a large finite proxy with the same sign pattern.
            probe = 1e6 * sign

            # np.full_like(d, probe) creates a vector aligned with d's shape.
            temp_vec = np.full_like(d, probe)
            # Evaluate the function at a large-magnitude point.
            val = safe_eval(f, temp_vec)
        except Exception:
            # If evaluation fails catastrophically, fall back to NaN.
            val = np.nan

        # Classify the observed behavior at large scale.
        if np.isposinf(val) and sign > 0:
            return +np.inf
        elif np.isneginf(val) and sign < 0:
            return -np.inf
        else:
            # If neither clear blow-up nor clear decay is observed treat the behavior as asymptotically flat.
            return 0.0

    # --- Case 3: Zero direction ---
    # If the normalized direction is (numerically) zero, there is no meaningful ray along which to probe.
    if np.allclose(d, 0):
        if verbose:
            print("→ Zero direction")

        # For proper functions, return 0 (no linear growth along zero).
        # For improper functions, return a large negative value.
        return 0 if is_proper(f, d) else -INFTY

    # --- Case 4: Finite direction (main computation) ---
    # This is the primary numerical estimator for finite, nonzero directions.
    if verbose:
        print("→ Finite direction (computing numerical liminf)")

    # Dimension of the ambient space.
    n = d.size

    # Magnitude of directional jitter.
    magnitude = 10 ** -(n / 4)

    # Initial lower and upper bounds for the geometric t-grid.
    t_min, t_max = 1.0, 1e2

    # Previous tail statistic, used to detect convergence.
    prev_tail = None

    # Placeholder for the scaled function values f(t d') / t.
    vals = np.empty

    # Iterate over expanding t-ranges to probe larger and larger scales.
    for cycle in range(max_cycles):

        # Construct a geometric grid of t-values.
        # NOTE: the grid size scales with dimension
        grid = np.geomspace(t_min, t_max, 100 * n)

        # Generate sign-preserving jittered directions d' → d.
        # Each row corresponds to a perturbed direction.
        d_prime = sign_preserving_jitter(d, len(grid), magnitude)

        # --- Vectorized evaluation is now default ---

        # Form the batch of evaluation points X = t * d' ==> Shape: (len(grid), n)
        X = grid[:, None] * d_prime

        # Evaluate the function on the entire batch.
        # safe_eval handles non-vectorized functions and exceptions.
        vals_raw = safe_eval(f, X)  # shape (len(grid),)

        # Scale by 1 / t where finite. Non-finite values are kept as is.
        vals = np.where(np.isfinite(vals_raw), vals_raw / grid, vals_raw)

        # --- Tail statistics ---
        # Use the last 20% of values to approximate asymptotic behavior.
        m = max(1, int(0.2 * len(vals)))
        tail_vals = vals[-m:]

        # Extract finite values from the tail.
        finite_tail = tail_vals[np.isfinite(tail_vals)]

        if finite_tail.size > 0:
            # If finite values exist, use their mean as a tail statistic.
            cur_tail = np.nanmean(finite_tail)
        elif np.any(np.isposinf(tail_vals)):
            # If the tail contains +inf, treat the tail as diverging upward.
            cur_tail = +np.inf
        elif np.any(np.isneginf(tail_vals)):
            # If the tail contains -inf, treat the tail as diverging downward.
            cur_tail = -np.inf
        else:
            # Otherwise, the tail provides no usable information.
            cur_tail = np.nan

        if verbose:
            # Helpful print statements for debugging
            print(f"Cycle {cycle + 1}: t_max={t_max:.1e}, tail≈{cur_tail:.4g}")

        # --- Divergence detection ---
        # If all scaled values are +inf, declare divergence upward.
        if np.all(np.isposinf(vals)):
            return +np.inf

        # If all scaled values are -inf, declare divergence downward.
        if np.all(np.isneginf(vals)):
            return -np.inf

        # --- Convergence detection ---
        # Compare the current tail statistic to the previous one.
        if prev_tail is not None and np.isfinite(prev_tail) and np.isfinite(cur_tail):

            # Relative change with small denominator regularization to protect against DivideByZero
            rel_change = abs(cur_tail - prev_tail) / (abs(prev_tail) + 1e-12)

            # If the change is small enough, stop expanding the scale.
            if rel_change < tol:
                break

        # Update the previous tail statistic.
        prev_tail = cur_tail

        # Expand the t-range to probe larger scales.
        t_min, t_max = t_max, t_max * 10

    # Multiply by the original norm to undo normalization and approximate f_\infty(d).
    return orig_norm * approximateLimInf(vals)
