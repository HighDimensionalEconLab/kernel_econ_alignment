import time
from typing import List, Optional

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
import unopy
from jax import config

from kernels import integrated_matern_kernel_matrices
from neoclassical_growth_benchmark import neoclassical_growth_benchmark
from rkhs import rkhs_norm_squared

config.update("jax_enable_x64", True)

NLP_OPTIONS = dict(preset="ipopt")
KINKED_NLP_OPTIONS = dict(preset="ipopt", max_iterations=25, time_limit=0.15)
ACCEPTED_SOLUTION_STATUSES = {"FEASIBLE_KKT_POINT", "FEASIBLE_SMALL_STEP"}
TRAIN_RESIDUAL_TOL = 1e-5
VALIDATION_RESIDUAL_TOL = 5e-3
MPK_BOUND_TOL = 1e-3
DOMAIN_EPS = 1e-6


def neoclassical_growth_matern(
    a: float = 1 / 3,
    delta: float = 0.1,
    rho_hat: float = 0.11,
    k_0: float = 1.0,  # k_0 is the state variable initial condition.
    A: Optional[float] = None,
    b_1: Optional[float] = None,
    b_2: Optional[float] = None,
    nu: float = 0.5,
    sigma: float = 1.0,
    rho: float = 10,
    train_T: float = 40.0,
    train_points: int = 41,
    test_T: float = 50.0,
    test_points: int = 100,
    benchmark_T: float = 60.0,
    benchmark_points: int = 300,
    train_points_list: Optional[List[float]] = None,
    verbose: bool = False,
):
    kinked_parameter_count = sum(parameter is not None for parameter in (A, b_1, b_2))
    if kinked_parameter_count not in {0, 3}:
        raise ValueError("Set all of A, b_1, and b_2, or set none of them.")
    use_kinked_production = kinked_parameter_count == 3
    A_value = 1.0 if A is None else float(A)
    b_1_value = 1.0 if b_1 is None else float(b_1)
    b_2_value = 0.0 if b_2 is None else float(b_2)

    if train_points_list is None:
        train_data = jnp.linspace(0, train_T, train_points)
    else:
        train_data = jnp.array(train_points_list)
    test_data = jnp.linspace(0, test_T, test_points)
    validation_points = max(test_points, benchmark_points)
    validation_data = jnp.linspace(0, train_T, validation_points)
    benchmark_grid = jnp.linspace(0, benchmark_T, benchmark_points)

    n_train = len(train_data)
    K, K_tilde = integrated_matern_kernel_matrices(
        train_data, train_data, nu, sigma, rho
    )
    K = np.asarray((K + K.T) / 2)
    K_tilde = np.asarray(K_tilde)
    K_jax = jnp.asarray(K)
    K_tilde_jax = jnp.asarray(K_tilde)

    n_variables = 3 * n_train + 2
    n_constraints = 6 * n_train

    def initial_production(k_value):
        if use_kinked_production:
            return A_value * max(k_value**a, b_1_value * k_value**a - b_2_value)
        return k_value**a

    c_flow_guess = max(initial_production(k_0) - delta * k_0, DOMAIN_EPS)
    x_initial = np.zeros(n_variables, dtype=np.float64)
    x_initial[3 * n_train] = c_flow_guess
    x_initial[3 * n_train + 1] = 1.0 / c_flow_guess

    def unpack(x):
        alpha_mu = x[:n_train]
        alpha_c = x[n_train : 2 * n_train]
        alpha_k = x[2 * n_train : 3 * n_train]
        c_0 = x[3 * n_train]
        mu_0 = x[3 * n_train + 1]
        return alpha_mu, alpha_c, alpha_k, c_0, mu_0

    def path_values(x, K_eval, K_tilde_eval):
        alpha_mu, alpha_c, alpha_k, c_0, mu_0 = unpack(x)
        mu = mu_0 + K_tilde_eval @ alpha_mu
        c = c_0 + K_tilde_eval @ alpha_c
        k = k_0 + K_tilde_eval @ alpha_k
        dmu_dt = K_eval @ alpha_mu
        dk_dt = K_eval @ alpha_k
        return k, c, mu, dk_dt, dmu_dt

    def production_scalar(k_scalar):
        k_positive = jnp.maximum(k_scalar, DOMAIN_EPS)
        z = k_positive**a
        if use_kinked_production:
            return A_value * jnp.maximum(z, b_1_value * z - b_2_value)
        return z

    production = jax.vmap(production_scalar)
    marginal_product = jax.vmap(jax.grad(production_scalar))

    def objective(x):
        alpha_mu, alpha_c, alpha_k, _, _ = unpack(x)
        return (
            alpha_mu @ K_jax @ alpha_mu
            + alpha_c @ K_jax @ alpha_c
            + alpha_k @ K_jax @ alpha_k
        )

    def constraints(x):
        k, c, mu, dk_dt, dmu_dt = path_values(x, K_jax, K_tilde_jax)
        output = production(k)
        mpk = marginal_product(k)
        resource = dk_dt - (output - delta * k - c)
        euler = dmu_dt + mu * (mpk - delta - rho_hat)
        shadow_price = mu * c - 1.0
        return jnp.concatenate([resource, euler, shadow_price, k, c, mu])

    def lagrangian(x, objective_multiplier, multipliers):
        return objective_multiplier * objective(x) + jnp.dot(multipliers, constraints(x))

    objective_value = jax.jit(objective)
    objective_gradient = jax.jit(jax.grad(objective))
    constraint_values = jax.jit(constraints)
    constraint_jacobian = jax.jit(jax.jacfwd(constraints))
    lagrangian_hessian = jax.jit(jax.hessian(lagrangian, argnums=0))

    x_initial_jax = jnp.asarray(x_initial)
    zero_multipliers = jnp.zeros(n_constraints, dtype=jnp.float64)
    jax.block_until_ready(objective_value(x_initial_jax))
    jax.block_until_ready(objective_gradient(x_initial_jax))
    jax.block_until_ready(constraint_values(x_initial_jax))
    jax.block_until_ready(constraint_jacobian(x_initial_jax))
    jax.block_until_ready(lagrangian_hessian(x_initial_jax, 1.0, zero_multipliers))

    variable_lower_bounds = np.full(n_variables, -np.inf, dtype=np.float64)
    variable_upper_bounds = np.full(n_variables, np.inf, dtype=np.float64)
    variable_lower_bounds[3 * n_train :] = DOMAIN_EPS
    constraint_lower_bounds = np.concatenate(
        [
            np.zeros(3 * n_train, dtype=np.float64),
            np.full(3 * n_train, DOMAIN_EPS, dtype=np.float64),
        ]
    )
    constraint_upper_bounds = np.concatenate(
        [
            np.zeros(3 * n_train, dtype=np.float64),
            np.full(3 * n_train, np.inf, dtype=np.float64),
        ]
    )

    model = unopy.Model(
        unopy.PROBLEM_NONLINEAR,
        n_variables,
        variable_lower_bounds,
        variable_upper_bounds,
        unopy.ZERO_BASED_INDEXING,
    )

    def objective_callback(x):
        return float(objective_value(jnp.asarray(x)))

    def objective_gradient_callback(x, gradient):
        gradient[:] = np.asarray(objective_gradient(jnp.asarray(x)))

    model.set_objective(
        unopy.MINIMIZE, objective_callback, objective_gradient_callback
    )

    jacobian_rows = np.repeat(np.arange(n_constraints, dtype=np.int32), n_variables)
    jacobian_columns = np.tile(np.arange(n_variables, dtype=np.int32), n_constraints)

    def constraints_callback(x, constraint_output):
        constraint_output[:] = np.asarray(constraint_values(jnp.asarray(x)))

    def jacobian_callback(x, jacobian_output):
        jacobian_output[:] = np.asarray(
            constraint_jacobian(jnp.asarray(x))
        ).reshape(-1)

    model.set_constraints(
        n_constraints,
        constraints_callback,
        constraint_lower_bounds,
        constraint_upper_bounds,
        len(jacobian_rows),
        jacobian_rows,
        jacobian_columns,
        jacobian_callback,
    )

    hessian_rows, hessian_columns = np.tril_indices(n_variables)
    hessian_rows = hessian_rows.astype(np.int32)
    hessian_columns = hessian_columns.astype(np.int32)

    def hessian_callback(x, objective_multiplier, multipliers, hessian_output):
        hessian = np.asarray(
            lagrangian_hessian(
                jnp.asarray(x),
                float(objective_multiplier),
                jnp.asarray(multipliers),
            )
        )
        hessian_output[:] = hessian[hessian_rows, hessian_columns]

    model.set_lagrangian_hessian(
        len(hessian_rows),
        unopy.LOWER_TRIANGLE,
        hessian_rows,
        hessian_columns,
        hessian_callback,
    )
    model.set_lagrangian_sign_convention(unopy.MULTIPLIER_POSITIVE)
    model.set_initial_primal_iterate(x_initial)

    solver = unopy.UnoSolver()
    options = dict(KINKED_NLP_OPTIONS if use_kinked_production else NLP_OPTIONS)
    solver.set_preset(options.pop("preset"))
    if not verbose:
        solver.set_option("logger", "SILENT")
        solver.set_option("print_solution", False)
    for option_name, option_value in options.items():
        solver.set_option(option_name, option_value)

    start = time.perf_counter()
    result = solver.optimize(model)
    elapsed = time.perf_counter() - start

    x = jnp.asarray(np.array(result.primal_solution, dtype=np.float64))
    alpha_mu, alpha_c, alpha_k, c_0, mu_0 = unpack(x)
    rkhs_norms = {
        "k": rkhs_norm_squared(alpha_k, K),
        "c": rkhs_norm_squared(alpha_c, K),
        "mu": rkhs_norm_squared(alpha_mu, K),
    }

    def evaluate_grid(points_data):
        K_eval, K_tilde_eval = integrated_matern_kernel_matrices(
            points_data, train_data, nu, sigma, rho
        )
        k_values, c_values, mu_values, dk_values, dmu_values = path_values(
            x, K_eval, K_tilde_eval
        )
        output_values = production(k_values)
        mpk_values = marginal_product(k_values)
        k_positive = jnp.maximum(k_values, DOMAIN_EPS)
        z_values = k_positive**a
        if use_kinked_production:
            branch_low_values = A_value * z_values
            branch_high_values = A_value * (b_1_value * z_values - b_2_value)
            branch_gap = branch_low_values - branch_high_values
            m1_values = A_value * a * k_positive ** (a - 1.0)
            m2_values = b_1_value * m1_values
        else:
            branch_low_values = z_values
            branch_high_values = z_values
            branch_gap = jnp.full_like(k_values, jnp.inf)
            m1_values = mpk_values
            m2_values = mpk_values
        implied_mpk = delta + rho_hat - dmu_values / mu_values
        mpk_lower_bound = jnp.minimum(m1_values, m2_values)
        mpk_upper_bound = jnp.maximum(m1_values, m2_values)
        return {
            "k": k_values,
            "c": c_values,
            "mu": mu_values,
            "z": z_values,
            "Y": output_values,
            "P": implied_mpk,
            "branch_low": branch_low_values,
            "branch_high": branch_high_values,
            "resource": dk_values - (output_values - delta * k_values - c_values),
            "euler": dmu_values + mu_values * (mpk_values - delta - rho_hat),
            "shadow_price": mu_values * c_values - 1.0,
            "branch_gap": branch_gap,
            "mpk_lower_violation": jnp.maximum(mpk_lower_bound - implied_mpk, 0.0),
            "mpk_upper_violation": jnp.maximum(implied_mpk - mpk_upper_bound, 0.0),
        }

    train_eval = evaluate_grid(train_data)
    validation_eval = evaluate_grid(validation_data)
    test_eval = evaluate_grid(test_data)

    train_residuals = {
        "resource": train_eval["resource"],
        "euler": train_eval["euler"],
        "shadow_price": train_eval["shadow_price"],
    }
    validation_residuals = {
        "resource_validation": validation_eval["resource"],
        "euler_validation": validation_eval["euler"],
        "shadow_price_validation": validation_eval["shadow_price"],
    }
    helper_residuals = {**train_residuals, **validation_residuals}
    max_train_residual = max(
        float(jnp.max(jnp.abs(residual))) for residual in train_residuals.values()
    )
    max_validation_residual = max(
        float(jnp.max(jnp.abs(residual))) for residual in validation_residuals.values()
    )
    max_helper_residual = max(max_train_residual, max_validation_residual)
    p_lower_violation = max(
        float(jnp.max(train_eval["mpk_lower_violation"])),
        float(jnp.max(validation_eval["mpk_lower_violation"])),
    )
    p_upper_violation = max(
        float(jnp.max(train_eval["mpk_upper_violation"])),
        float(jnp.max(validation_eval["mpk_upper_violation"])),
    )
    active_branch_switch_train = bool(
        use_kinked_production
        and jnp.min(train_eval["branch_gap"]) < 0.0
        and jnp.max(train_eval["branch_gap"]) > 0.0
    )
    min_branch_gap_train = float(jnp.min(jnp.abs(train_eval["branch_gap"])))

    @jax.jit
    def kernel_solution(test_points_data):
        K_test, K_tilde_test = integrated_matern_kernel_matrices(
            test_points_data, train_data, nu, sigma, rho
        )
        k_test, c_test, _, _, _ = path_values(x, K_test, K_tilde_test)
        return k_test, c_test

    finite_positive = bool(
        jnp.all(jnp.isfinite(train_eval["k"]))
        and jnp.all(jnp.isfinite(train_eval["c"]))
        and jnp.all(jnp.isfinite(train_eval["mu"]))
        and jnp.all(jnp.isfinite(validation_eval["k"]))
        and jnp.all(jnp.isfinite(validation_eval["c"]))
        and jnp.all(jnp.isfinite(validation_eval["mu"]))
        and jnp.all(jnp.isfinite(test_eval["k"]))
        and jnp.all(jnp.isfinite(test_eval["c"]))
        and jnp.min(train_eval["k"]) > 0.0
        and jnp.min(train_eval["c"]) > 0.0
        and jnp.min(train_eval["mu"]) > 0.0
        and jnp.min(validation_eval["k"]) > 0.0
        and jnp.min(validation_eval["c"]) > 0.0
        and jnp.min(validation_eval["mu"]) > 0.0
        and jnp.min(test_eval["k"]) > 0.0
        and jnp.min(test_eval["c"]) > 0.0
    )

    solve_time = result.cpu_time
    if solve_time is None:
        solve_time = elapsed
    solver_status = str(result.optimization_status).split(".")[-1]
    solution_status = str(result.solution_status).split(".")[-1]
    rejection_reasons = []
    if solver_status != "SUCCESS":
        rejection_reasons.append("solver_status")
    if solution_status not in ACCEPTED_SOLUTION_STATUSES:
        rejection_reasons.append("solution_status")
    if not finite_positive:
        rejection_reasons.append("nonfinite_or_nonpositive")
    if max_train_residual > TRAIN_RESIDUAL_TOL:
        rejection_reasons.append("train_residual")
    if use_kinked_production and max_validation_residual > VALIDATION_RESIDUAL_TOL:
        rejection_reasons.append("validation_residual")
    if use_kinked_production and (
        p_lower_violation > MPK_BOUND_TOL or p_upper_violation > MPK_BOUND_TOL
    ):
        rejection_reasons.append("p_bound")
    valid_solution = not rejection_reasons

    benchmark_solution = None
    k_benchmark = jnp.full_like(test_data, jnp.nan)
    c_benchmark = jnp.full_like(test_data, jnp.nan)
    k_rel_error = jnp.full_like(test_data, jnp.nan)
    c_rel_error = jnp.full_like(test_data, jnp.nan)
    if not use_kinked_production:
        benchmark_solution = neoclassical_growth_benchmark(
            a, delta, rho_hat, 1.0, k_0, benchmark_grid
        )
        k_benchmark, c_benchmark = benchmark_solution(test_data)
        k_rel_error = jnp.abs(k_benchmark - test_eval["k"]) / k_benchmark
        c_rel_error = jnp.abs(c_benchmark - test_eval["c"]) / c_benchmark

    if use_kinked_production:
        print(f"solve_time(s) = {solve_time}")
    else:
        print(
            f"solve_time(s) = {solve_time}, E(|rel_error(k)|) = {k_rel_error.mean()}, E(|rel_error(c)|) = {c_rel_error.mean()}"
        )
    return {
        "t_train": train_data,
        "t_test": test_data,
        "k_test": test_eval["k"],
        "c_test": test_eval["c"],
        "k_benchmark": k_benchmark,
        "c_benchmark": c_benchmark,
        "k_rel_error": k_rel_error,
        "c_rel_error": c_rel_error,
        "alpha_m": alpha_mu,
        "alpha_mu": alpha_mu,
        "alpha_c": alpha_c,
        "alpha_k": alpha_k,
        "c_0": float(c_0),
        "mu_0": float(mu_0),
        "rkhs_norms": rkhs_norms,
        "helper_residuals": helper_residuals,
        "train_residuals": train_residuals,
        "validation_residuals": validation_residuals,
        "k_train": train_eval["k"],
        "mu_train": train_eval["mu"],
        "c_train": train_eval["c"],
        "z_train": train_eval["z"],
        "Y_train": train_eval["Y"],
        "P_train": train_eval["P"],
        "branch_low_train": train_eval["branch_low"],
        "branch_high_train": train_eval["branch_high"],
        "solve_time": solve_time,
        "wall_time": elapsed,
        "solver_status": solver_status,
        "solution_status": solution_status,
        "stationarity": result.solution_stationarity,
        "primal_feasibility": result.solution_primal_feasibility,
        "valid_solution": valid_solution,
        "rejection_reason": "accepted" if valid_solution else ",".join(rejection_reasons),
        "candidate_branch": "jax_max" if use_kinked_production else "smooth",
        "candidate_diagnostics": [
            {
                "branch": "jax_max" if use_kinked_production else "smooth",
                "valid_solution": valid_solution,
                "rejection_reason": "accepted"
                if valid_solution
                else ",".join(rejection_reasons),
                "solver_status": solver_status,
                "solution_status": solution_status,
                "solve_time": solve_time,
                "wall_time": elapsed,
                "max_helper_residual": max_helper_residual,
                "max_train_residual": max_train_residual,
                "max_validation_residual": max_validation_residual,
                "p_lower_violation": p_lower_violation,
                "p_upper_violation": p_upper_violation,
                "objective": float(
                    rkhs_norms["k"] + rkhs_norms["c"] + rkhs_norms["mu"]
                ),
            }
        ],
        "max_helper_residual": max_helper_residual,
        "max_train_residual": max_train_residual,
        "max_validation_residual": max_validation_residual,
        "p_lower_violation": p_lower_violation,
        "p_upper_violation": p_upper_violation,
        "active_branch_switch_train": active_branch_switch_train,
        "min_branch_gap_train": min_branch_gap_train,
        "use_kinked_production": use_kinked_production,
        "production_parameters": {
            "A": A_value,
            "b_1": b_1_value,
            "b_2": b_2_value,
        },
        "kernel_solution": kernel_solution,
        "benchmark_solution": benchmark_solution,
    }


if __name__ == "__main__":
    jsonargparse.CLI(neoclassical_growth_matern)
