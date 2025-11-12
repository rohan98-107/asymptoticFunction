import numpy as np
import warnings
from scipy.stats import qmc, norm


def sign_preserving_jitter(d, n_samples, magnitude):
    """
    Generate n_samples of small perturbations that do not change
    the sign pattern of d (keeps within same orthant).
    """
    rng = np.random.default_rng(42)
    n = d.size
    jitters = rng.normal(size=(n_samples, n))
    jitters /= np.linalg.norm(jitters, axis=1, keepdims=True)
    jitters *= magnitude

    d_prime = np.empty_like(jitters)
    for k, j in enumerate(jitters):
        pert = d + j

        # enforce sign consistency
        for i in range(n):
            if d[i] < 0:
                pert[i] = min(pert[i], 0.0)
            elif d[i] > 0:
                pert[i] = max(pert[i], 0.0)
            else:
                # for boundary zeros, only allow nonpositive moves
                pert[i] = min(pert[i], 0.0)

        d_prime[k] = pert / np.linalg.norm(pert) if np.linalg.norm(pert) > 0 else pert

    return d_prime


def sample_sphere(n_samples=1000, dim=2, method="sobol", seed=None):
    """
    Generate quasi-uniform samples on the unit sphere S^{dim-1}.
    """
    rng = np.random.default_rng(seed)

    if method == "normal":
        X = rng.normal(size=(n_samples, dim))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X

    elif method == "sobol":
        sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
        U = sampler.random(n_samples)
        Z = norm.ppf(U)
        Z[np.isinf(Z)] = 0.0
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)
        return Z

    elif method == "fibonacci":
        if dim != 3:
            warnings.warn("Fibonacci is only defined for dim=3; using random fallback.")
            return sample_sphere(n_samples, dim, method="random", seed=seed)

        k = np.arange(n_samples, dtype=float)
        phi = (1 + np.sqrt(5)) / 2
        theta = 2 * np.pi * k / phi
        z = 1 - (2 * k + 1) / n_samples
        r = np.sqrt(1 - z**2)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.stack((x, y, z), axis=1)

    else:
        raise ValueError(f"Unknown sampling method '{method}'.")
