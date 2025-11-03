# examples/demo1.py

import numpy as np
import sympy as sp
from asymptoticFunction.core.asymptotic_function import asymptotic_function


def _fmt(val):
    if isinstance(val, float):
        if np.isnan(val):
            return "nan"
        if np.isinf(val):
            return "∞" if val > 0 else "−∞"
        return f"{val:.4f}"
    return str(val)


def _fmt_d(d):
    return "[" + ", ".join(f"{x:.2g}" for x in d) + "]"


def run_case(name, f_num, d, kind=None, params=None, f_for_analytical=None):
    if f_for_analytical is None:
        f_for_analytical = f_num

    # analytical
    try:
        res_a = asymptotic_function(f_for_analytical, d, kind=kind, params=params or {})
        val_a = _fmt(res_a.value)
    except Exception:
        val_a = "err"

    # numerical (never pass params here)
    try:
        res_n = asymptotic_function(f_num, d)
        val_n = _fmt(res_n.value)
    except Exception:
        val_n = "err"

    return name, _fmt_d(d), val_a, val_n


def main():
    rows = []

    # 1. linear
    f = lambda x: 3 * x[0] + 2 * x[1] + 1
    d = np.array([1.0, 2.0])
    rows.append(run_case("Linear",
                         f, d,
                         kind="linear",
                         params={"a": np.array([3., 2.])}))

    # 2. norm
    f = lambda x: np.linalg.norm(x, 2)
    d = np.array([-3.0, 4.0])
    rows.append(run_case("Norm (L2)",
                         f, d,
                         kind="norm",
                         params={"p": 2}))

    # 3. weighted norm
    W = np.array([[2.0, 0.0], [0.0, 1.0]])
    f = lambda x: np.linalg.norm(W @ x, 2)
    d = np.array([1.0, -2.0])
    rows.append(run_case("Weighted Norm",
                         f, d,
                         kind="weighted_norm",
                         params={"W": W}))

    # 4. affine + norm
    a = np.array([1.0, 2.0])
    f = lambda x: np.dot(a, x) + np.linalg.norm(x)
    d = np.array([-1.0, 1.0])
    rows.append(run_case("Affine + Norm",
                         f, d,
                         kind="affine_plus_norm",
                         params={"a": a, "c": 1.0}))

    # 5. abs linear
    a = np.array([1.0, -2.0])
    f = lambda x: abs(np.dot(a, x))
    d = np.array([1.0, 2.0])
    rows.append(run_case("Abs Linear",
                         f, d,
                         kind="abs_linear",
                         params={"a": a}))

    # 6. max affine
    A = np.array([[1.0, 0.0],
                  [0.0, 2.0],
                  [-1.0, -1.0]])
    f = lambda x: np.max(A @ x)
    d = np.array([1.0, -1.0])
    rows.append(run_case("Max-Affine",
                         f, d,
                         kind="max_affine",
                         params={"A": A}))

    # 7. indicator halfspace
    a = np.array([1.0, 1.0])
    f = lambda x: 0.0 if np.dot(a, x) <= 0 else np.inf
    d = np.array([-1.0, 0.5])
    rows.append(run_case("Indicator (Halfspace)",
                         f, d,
                         kind="indicator_halfspace",
                         params={"a": a}))

    # 8. indicator hyperplane
    a = np.array([1.0, -1.0])
    f = lambda x: 0.0 if abs(np.dot(a, x)) <= 1e-12 else np.inf
    d = np.array([1.0, 1.0])
    rows.append(run_case("Indicator (Hyperplane)",
                         f, d,
                         kind="indicator_hyperplane",
                         params={"a": a}))

    # 9. support function
    C_points = np.array([[1.0, 0.0],
                         [0.0, 1.0]])
    f = lambda x: np.max(C_points @ x)
    d = np.array([1.0, 2.0])
    rows.append(run_case("Support Function",
                         f, d,
                         kind="support_function",
                         params={"C_points": C_points}))

    # 10. distance to cone
    proj_K = lambda x: np.maximum(x, 0.0)
    f = lambda x: np.linalg.norm(x - proj_K(x))
    d = np.array([-1.0, 2.0])
    rows.append(run_case("Distance to Cone",
                         f, d,
                         kind="distance_cone",
                         params={"proj_K": proj_K}))

    # 11. quadratic
    Q = np.array([[2.0, 0.0],
                  [0.0, -1.0]])
    b = np.array([1.0, 0.0])
    f = lambda x: x.T @ Q @ x + b @ x
    d = np.array([1.0, 1.0])
    rows.append(run_case("Quadratic",
                         f, d,
                         kind="quadratic",
                         params={"Q": Q, "b": b}))

    # 12. polynomial: analytical = sympy expr, numerical = lambdified
    f = lambda x: x[0]**3 - 2*x[0]*x[1] + 5*x[0] + 1
    d = np.array([1.0, 2.0])
    rows.append(run_case("Polynomial",
                         f, d,
                         kind="polynomial"))

    # 13. hinge
    A = np.array([[1.0, -1.0],
                  [0.5, 2.0]])
    y = np.array([1.0, -1.0])
    f = lambda x: np.sum(np.maximum(0.0, 1 - y * (A @ x)))
    d = np.array([1.0, 0.5])
    rows.append(run_case("Hinge Loss",
                         f, d,
                         kind="hinge_sum",
                         params={"A": A, "y": y}))

    # 14. logistic
    f = lambda x: np.sum(np.log(1 + np.exp(-y * (A @ x))))
    d = np.array([1.0, 0.5])
    rows.append(run_case("Logistic Loss",
                         f, d,
                         kind="logistic_sum",
                         params={"A": A, "y": y}))

    # 15. huber
    f = lambda x: np.sum(np.abs(A @ x))  # simple version, matches your asymptotic
    d = np.array([1.0, 0.5])
    rows.append(run_case("Huber Loss",
                         f, d,
                         kind="huber_sum",
                         params={"A": A, "delta": 1.0}))

    # 16. exponential loss
    f = lambda x: np.sum(np.exp(-y * (A @ x)))
    d = np.array([1.0, 0.5])
    rows.append(run_case("Exponential Loss",
                         f, d,
                         kind="exponential_sum",
                         params={"A": A, "y": y}))

    # 17. sum exp
    C = np.array([[1.0, 0.0],
                  [-1.0, 2.0],
                  [0.0, -1.0]])
    f = lambda x: np.sum(np.exp(C @ x))
    d = np.array([1.0, 0.5])
    rows.append(run_case("Sum of Exponentials",
                         f, d,
                         kind="sum_exp",
                         params={"C": C}))

    # 18. log-sum-exp
    f = lambda x: np.log(np.sum(np.exp(C @ x)))
    rows.append(run_case("Log-sum-exp",
                         f, d,
                         kind="log_sum_exp",
                         params={"C": C}))

    header = f"{'Function Type':<28} | {'d':<12} | {'Analytical f∞(d)':<18} | {'Numerical f∞(d)':<18}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for r in rows:
        print(f"{r[0]:<28} | {r[1]:<12} | {r[2]:<18} | {r[3]:<18}")
    print("=" * len(header))


if __name__ == "__main__":
    main()
