import numpy as np


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
