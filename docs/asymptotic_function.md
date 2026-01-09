# Asymptotic Functions

This document describes how **asymptotic function** are represented and computed in this package. This is a key reference point and underlies all downstream 
computational and visualization behaviors within the 'asymptoticFunction' package. We will concentrate specifically on 
numerical estimation of the asymptotic function which is defined formally in [mathematical_background.md](mathematical_background.md). We will liberally use the term 'asymptotic direction' 
to describe normalized input to the asymptotic_function methods. Knowledge of [asymptotic_direction.md](asymptotic_direction.md) is assumed. 

## What is being computed? 

The **asymptotic_function** routine **approximates** the value $f_\infty(d)$ given a callable function $f$ and a normalized asymptotic direction $d$. 
The routine's algorithmic flow is given below: 

![asymptotic_function_flow](figures/asymptotic_function_flow.svg)

## Callable Functions 

It is the convention of the 'asymptoticFunction' package that all functions are required to be callable. We enforce that the input direction is indeed a NumPy
array and the input is Callable through a thin wrapper class called 'CallableFunction'. 

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

## Approximation    

The **approximateAsymptoticFunc()** method is the backbone of the 'asymptoticFunction' package. 
