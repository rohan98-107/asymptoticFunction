from .analytical_forms import (
    linear_asymptotic,
    affine_asymptotic,
    norm_asymptotic,
    weighted_norm_asymptotic,
    affine_plus_norm_asymptotic,
    abs_linear_asymptotic,
    max_affine_asymptotic,
    indicator_halfspace_asymptotic,
    indicator_hyperplane_asymptotic,
    indicator_affine_subspace_asymptotic,
    indicator_polyhedron_asymptotic,
    indicator_box_asymptotic,
    indicator_cone_asymptotic,
    support_function_asymptotic,
    distance_cone_asymptotic,
    quadratic_asymptotic,
    polynomial_asymptotic,
    hinge_sum_asymptotic,
    logistic_sum_asymptotic,
    huber_sum_asymptotic,
    exponential_sum_asymptotic,
    sum_exp_asymptotic,
    log_sum_exp_asymptotic,
)

analytical_registry = {
    "linear": linear_asymptotic,
    "affine": affine_asymptotic,
    "norm": norm_asymptotic,
    "weighted_norm": weighted_norm_asymptotic,
    "affine_plus_norm": affine_plus_norm_asymptotic,
    "abs_linear": abs_linear_asymptotic,
    "max_affine": max_affine_asymptotic,
    "indicator_halfspace": indicator_halfspace_asymptotic,
    "indicator_hyperplane": indicator_hyperplane_asymptotic,
    "indicator_affine_subspace": indicator_affine_subspace_asymptotic,
    "indicator_polyhedron": indicator_polyhedron_asymptotic,
    "indicator_box": indicator_box_asymptotic,
    "indicator_cone": indicator_cone_asymptotic,
    "support_function": support_function_asymptotic,
    "distance_cone": distance_cone_asymptotic,
    "quadratic": quadratic_asymptotic,
    "polynomial": polynomial_asymptotic,
    "hinge_sum": hinge_sum_asymptotic,
    "logistic_sum": logistic_sum_asymptotic,
    "huber_sum": huber_sum_asymptotic,
    "exponential_sum": exponential_sum_asymptotic,
    "sum_exp": sum_exp_asymptotic,
    "log_sum_exp": log_sum_exp_asymptotic,
}


def get_analytical_form(kind: str):
    func = analytical_registry.get(kind)
    if func is None:
        raise ValueError(f"Unknown analytical kind '{kind}'.")
    return func
