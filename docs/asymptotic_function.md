# Asymptotic Functions

This document describes how **asymptotic functions** are represented and computed in this package. This is a key reference point and underlies all downstream 
computational and visualization behaviors within the `asymptoticFunction` package. We will concentrate specifically on 
numerical estimation of the asymptotic function which is defined formally in [mathematical_background.md](mathematical_background.md). We will liberally use the term 'asymptotic direction' 
to describe normalized input to the methods described here. Knowledge of [asymptotic_direction.md](asymptotic_direction.md) is assumed. 

## What is being computed? 

The **asymptotic_function** routine **approximates** the value $f_\infty(d)$ given a callable function $f$ and a normalized asymptotic direction $d$. 
The routine's algorithmic flow is given below: 

![asymptotic_function_flow](figures/asymptotic_function_flow.svg)

## CallableFunction

It is the convention of the `asymptoticFunction` package that all functions are required to be callable. We enforce that the input direction is indeed a `NumPy`
array and the input is Callable through a thin wrapper class called `CallableFunction`. 

```python
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
```
This class ensures input type-safety for later computations which require arithmetic between asymptotic functions. 

## Core Approximate Computation 

The `approximateAsymptoticFunc()` method is the backbone of the `asymptoticFunction` package. The above flowchart represents 
its core logic however much care has been taken in commenting this method so as to make it seamless to follow: 

```python
from asymptoticFunction.core.normalization import normalize_direction
from asymptoticFunction.core.utils import *
from asymptoticFunction.numerical.sampling import sign_preserving_jitter


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

```

## AsymptoticResult 

Similar to `CallableFunction` we have implemented a wrapper class which houses the output of 'approximateAsymptoticFunction' 
and returns it to the user via `asymptotic_function()`. The intention of this class is to provide metadata, detailed failure modes,
raw/unscaled vector data for indeterminate outputs and other metrics/parameters. In its current state we have only provided 
basic console output functionality. 

```python
from __future__ import annotations
import numpy as np

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
```

## Helper Functions and Numerical Utilities 

For more details on the nonstandard methods used in **approximateAsymptoticFunction** please see future documentation or see the code
directly in `asymptoticFunction.core.utils`. In 
this section we give a high-level summary of the relevant subroutines and their respective roles in the overall computation.  

- **normalize_direction** - see [asymptotic_direction.md](asymptotic_direction.md) for detailed implementation and explanation 
- **safe_eval** - key method handling "infinite arithmetic", numerical exception handling and sample-noise correction 
- **is_proper** - method to determine whether or not a function is 'proper' in the mathematical sense
- **sign_preserving_jitter** - another key method that evaluates $f_\infty(d)$ upon a series of perturbed directions $d'$ which are constructed so as not to cross any standard axes lines in the ambient space 
- **make_vectorization_safe** - shape handling utility for batch evaluation 