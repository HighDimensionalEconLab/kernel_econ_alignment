import time
from typing import List, Optional

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
import unopy
from jax import config
from nlls_gram import UnderdeterminedLevenbergMarquardt

from kernels import integrated_matern_kernel_matrices
from rkhs import rkhs_norm_squared

config.update("jax_enable_x64", True)

NLP_OPTIONS = dict(preset="filtersqp", time_limit=5.0)
ACCEPTED_SOLUTION_STATUSES = {"FEASIBLE_KKT_POINT", "FEASIBLE_SMALL_STEP"}
DOMAIN_EPS = 1e-8


def _human_capital_initial_residual(params, batch):
    a_k, a_h, delta_k, delta_h, k_0 = batch
    h_0 = jnp.exp(params["log_h"])
    c_0 = jnp.exp(params["log_c"])
    f = (k_0**a_k) * (h_0**a_h)
    f_k = a_k * (k_0 ** (a_k - 1.0)) * (h_0**a_h)
    f_h = a_h * (k_0**a_k) * (h_0 ** (a_h - 1.0))
    return jnp.array(
        [
            f_h - delta_h - (f_k - delta_k),
            c_0 + delta_h * h_0 + delta_k * k_0 - f,
        ],
        dtype=h_0.dtype,
    )


def human_capital_initial_conditions(
    a_k: float = 1 / 3,
    a_h: float = 1 / 4,
    delta_k: float = 0.1,
    delta_h: float = 0.05,
    k_0: float = 1.5,
    iterations: int = 20,
):
    dtype = jnp.float64
    batch = tuple(
        jnp.asarray(value, dtype=dtype) for value in (a_k, a_h, delta_k, delta_h, k_0)
    )
    k_0_jax = batch[-1]
    c_guess = (
        (k_0_jax**batch[0]) * (k_0_jax**batch[1])
        - batch[3] * k_0_jax
        - batch[2] * k_0_jax
    )
    c_guess = jnp.maximum(c_guess, jnp.asarray(1e-12, dtype=dtype))
    params = {
        "log_h": jnp.log(k_0_jax),
        "log_c": jnp.log(c_guess),
    }
    solver = UnderdeterminedLevenbergMarquardt(
        _human_capital_initial_residual, init_damping=1e-4
    )
    state = solver.init(dtype=dtype)

    @jax.jit
    def step(params, state):
        return solver.update(params, state, batch)

    info = None
    for _ in range(iterations):
        params, state, info = step(params, state)

    h_0 = jnp.exp(params["log_h"])
    c_0 = jnp.exp(params["log_c"])
    residual = _human_capital_initial_residual(params, batch)
    return h_0, c_0, residual, info


def human_capital_matern(
    a_k: float = 1 / 3,
    a_h: float = 1 / 4,
    delta_k: float = 0.1,
    delta_h: float = 0.05,
    rho_hat: float = 0.11,
    k_0: float = 1.5,
    nu: float = 0.5,
    sigma: float = 1.0,
    rho: float = 10,
    train_T: float = 80.0,
    train_points: int = 61,
    test_T: float = 100.0,
    test_points: int = 100,
    benchmark_T: float = 60.0,
    benchmark_points: int = 300,
    train_points_list: Optional[List[float]] = None,
    verbose: bool = False,
):
    _ = benchmark_T
    if train_points_list is None:
        train_data = jnp.linspace(0, train_T, train_points)
    else:
        train_data = jnp.array(train_points_list)
    test_data = jnp.linspace(0, test_T, test_points)
    validation_points = max(test_points, benchmark_points)
    validation_data = jnp.linspace(0, train_T, validation_points)

    n_train = len(train_data)
    K, K_tilde = integrated_matern_kernel_matrices(
        train_data, train_data, nu, sigma, rho
    )
    K = np.asarray((K + K.T) / 2)
    K_tilde = np.asarray(K_tilde)
    K_jax = jnp.asarray(K)
    K_tilde_jax = jnp.asarray(K_tilde)

    h_0_jax, c_0_init_jax, initial_residual, _ = human_capital_initial_conditions(
        a_k, a_h, delta_k, delta_h, k_0
    )
    h_0 = float(h_0_jax)
    c_0_init = float(c_0_init_jax)

    n_functions = 7
    n_scalars = 5
    n_variables = n_functions * n_train + n_scalars
    n_equalities = 7 * n_train
    n_constraints = 14 * n_train
    x_initial = np.zeros(n_variables, dtype=np.float64)
    scalar_start = n_functions * n_train
    x_initial[scalar_start] = delta_k * k_0
    x_initial[scalar_start + 1] = delta_h * h_0
    x_initial[scalar_start + 2] = c_0_init
    x_initial[scalar_start + 3] = 1.0 / c_0_init
    x_initial[scalar_start + 4] = 1.0 / c_0_init

    def unpack(x):
        alpha_k = x[:n_train]
        alpha_h = x[n_train : 2 * n_train]
        alpha_i_k = x[2 * n_train : 3 * n_train]
        alpha_i_h = x[3 * n_train : 4 * n_train]
        alpha_c = x[4 * n_train : 5 * n_train]
        alpha_mu_k = x[5 * n_train : 6 * n_train]
        alpha_mu_h = x[6 * n_train : 7 * n_train]
        i_k_0 = x[7 * n_train]
        i_h_0 = x[7 * n_train + 1]
        c_0 = x[7 * n_train + 2]
        mu_k_0 = x[7 * n_train + 3]
        mu_h_0 = x[7 * n_train + 4]
        return (
            alpha_k,
            alpha_h,
            alpha_i_k,
            alpha_i_h,
            alpha_c,
            alpha_mu_k,
            alpha_mu_h,
            i_k_0,
            i_h_0,
            c_0,
            mu_k_0,
            mu_h_0,
        )

    def path_values(x, K_eval, K_tilde_eval):
        (
            alpha_k,
            alpha_h,
            alpha_i_k,
            alpha_i_h,
            alpha_c,
            alpha_mu_k,
            alpha_mu_h,
            i_k_0,
            i_h_0,
            c_0,
            mu_k_0,
            mu_h_0,
        ) = unpack(x)
        k = k_0 + K_tilde_eval @ alpha_k
        h = h_0 + K_tilde_eval @ alpha_h
        i_k = i_k_0 + K_tilde_eval @ alpha_i_k
        i_h = i_h_0 + K_tilde_eval @ alpha_i_h
        c = c_0 + K_tilde_eval @ alpha_c
        mu_k = mu_k_0 + K_tilde_eval @ alpha_mu_k
        mu_h = mu_h_0 + K_tilde_eval @ alpha_mu_h
        dk_dt = K_eval @ alpha_k
        dh_dt = K_eval @ alpha_h
        dmu_k_dt = K_eval @ alpha_mu_k
        dmu_h_dt = K_eval @ alpha_mu_h
        return k, h, i_k, i_h, c, mu_k, mu_h, dk_dt, dh_dt, dmu_k_dt, dmu_h_dt

    def production_terms(k, h):
        k_positive = jnp.maximum(k, DOMAIN_EPS)
        h_positive = jnp.maximum(h, DOMAIN_EPS)
        f = (k_positive**a_k) * (h_positive**a_h)
        f_k = a_k * (k_positive ** (a_k - 1.0)) * (h_positive**a_h)
        f_h = a_h * (k_positive**a_k) * (h_positive ** (a_h - 1.0))
        return f, f_k, f_h

    def objective(x):
        (
            alpha_k,
            alpha_h,
            alpha_i_k,
            alpha_i_h,
            alpha_c,
            alpha_mu_k,
            alpha_mu_h,
            *_,
        ) = unpack(x)
        return (
            alpha_k @ K_jax @ alpha_k
            + alpha_h @ K_jax @ alpha_h
            + alpha_i_k @ K_jax @ alpha_i_k
            + alpha_i_h @ K_jax @ alpha_i_h
            + alpha_c @ K_jax @ alpha_c
            + alpha_mu_k @ K_jax @ alpha_mu_k
            + alpha_mu_h @ K_jax @ alpha_mu_h
        )

    def constraints(x):
        (
            k,
            h,
            i_k,
            i_h,
            c,
            mu_k,
            mu_h,
            dk_dt,
            dh_dt,
            dmu_k_dt,
            dmu_h_dt,
        ) = path_values(x, K_jax, K_tilde_jax)
        f, f_k, f_h = production_terms(k, h)
        equalities = jnp.concatenate(
            [
                dk_dt - (i_k - delta_k * k),
                dh_dt - (i_h - delta_h * h),
                dmu_k_dt + mu_k * (f_k - delta_k - rho_hat),
                dmu_h_dt + mu_h * (f_h - delta_h - rho_hat),
                c + i_h + i_k - f,
                mu_k * c - 1.0,
                mu_k - mu_h,
            ]
        )
        positive_domains = jnp.concatenate([k, h, i_k, i_h, c, mu_k, mu_h])
        return jnp.concatenate([equalities, positive_domains])

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
    variable_lower_bounds[scalar_start:] = DOMAIN_EPS
    constraint_lower_bounds = np.concatenate(
        [
            np.zeros(n_equalities, dtype=np.float64),
            np.full(7 * n_train, DOMAIN_EPS, dtype=np.float64),
        ]
    )
    constraint_upper_bounds = np.concatenate(
        [
            np.zeros(n_equalities, dtype=np.float64),
            np.full(7 * n_train, np.inf, dtype=np.float64),
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
    options = dict(NLP_OPTIONS)
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
    (
        alpha_k,
        alpha_h,
        alpha_i_k,
        alpha_i_h,
        alpha_c,
        alpha_mu_k,
        alpha_mu_h,
        i_k_0,
        i_h_0,
        c_0,
        mu_k_0,
        mu_h_0,
    ) = unpack(x)
    rkhs_norms = {
        "k": rkhs_norm_squared(alpha_k, K),
        "h": rkhs_norm_squared(alpha_h, K),
        "i_k": rkhs_norm_squared(alpha_i_k, K),
        "i_h": rkhs_norm_squared(alpha_i_h, K),
        "c": rkhs_norm_squared(alpha_c, K),
        "mu_k": rkhs_norm_squared(alpha_mu_k, K),
        "mu_h": rkhs_norm_squared(alpha_mu_h, K),
    }

    def evaluate_grid(points_data):
        K_eval, K_tilde_eval = integrated_matern_kernel_matrices(
            points_data, train_data, nu, sigma, rho
        )
        (
            k_values,
            h_values,
            i_k_values,
            i_h_values,
            c_values,
            mu_k_values,
            mu_h_values,
            dk_values,
            dh_values,
            dmu_k_values,
            dmu_h_values,
        ) = path_values(x, K_eval, K_tilde_eval)
        f_values, f_k_values, f_h_values = production_terms(k_values, h_values)
        return {
            "k": k_values,
            "h": h_values,
            "i_k": i_k_values,
            "i_h": i_h_values,
            "c": c_values,
            "mu_k": mu_k_values,
            "mu_h": mu_h_values,
            "physical_accumulation": dk_values
            - (i_k_values - delta_k * k_values),
            "human_accumulation": dh_values - (i_h_values - delta_h * h_values),
            "physical_euler": dmu_k_values
            + mu_k_values * (f_k_values - delta_k - rho_hat),
            "human_euler": dmu_h_values
            + mu_h_values * (f_h_values - delta_h - rho_hat),
            "feasibility": c_values + i_h_values + i_k_values - f_values,
            "shadow_price": mu_k_values * c_values - 1.0,
            "costate_gap": mu_k_values - mu_h_values,
            "hidden_dae_residual": (f_k_values - delta_k)
            - (f_h_values - delta_h),
        }

    train_eval = evaluate_grid(train_data)
    validation_eval = evaluate_grid(validation_data)
    test_eval = evaluate_grid(test_data)

    residual_names = [
        "physical_accumulation",
        "human_accumulation",
        "physical_euler",
        "human_euler",
        "feasibility",
        "shadow_price",
        "costate_gap",
    ]
    train_residuals = {name: train_eval[name] for name in residual_names}
    validation_residuals = {
        f"{name}_validation": validation_eval[name] for name in residual_names
    }
    max_train_residual = max(
        float(jnp.max(jnp.abs(residual))) for residual in train_residuals.values()
    )
    max_validation_residual = max(
        float(jnp.max(jnp.abs(residual))) for residual in validation_residuals.values()
    )
    finite_positive = bool(
        all(
            jnp.all(jnp.isfinite(test_eval[name]))
            and jnp.all(jnp.isfinite(validation_eval[name]))
            and jnp.min(test_eval[name]) > 0.0
            and jnp.min(validation_eval[name]) > 0.0
            for name in ["k", "h", "i_k", "i_h", "c", "mu_k", "mu_h"]
        )
    )

    @jax.jit
    def kernel_solution(test_points_data):
        K_test, K_tilde_test = integrated_matern_kernel_matrices(
            test_points_data, train_data, nu, sigma, rho
        )
        (
            k_test,
            h_test,
            i_k_test,
            i_h_test,
            c_test,
            mu_k_test,
            mu_h_test,
            *_,
        ) = path_values(x, K_test, K_tilde_test)
        f_test, _, _ = production_terms(k_test, h_test)
        feasibility_test = c_test + i_h_test + i_k_test - f_test
        return (
            k_test,
            h_test,
            c_test,
            i_k_test,
            i_h_test,
            mu_k_test,
            mu_h_test,
            feasibility_test,
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
    if max_train_residual > 1e-5:
        rejection_reasons.append("train_residual")
    valid_solution = not rejection_reasons

    print(f"solve_time(s) = {solve_time}")
    return {
        "t_train": train_data,
        "t_test": test_data,
        "k_test": test_eval["k"],
        "h_test": test_eval["h"],
        "c_test": test_eval["c"],
        "i_k_test": test_eval["i_k"],
        "i_h_test": test_eval["i_h"],
        "mu_k_test": test_eval["mu_k"],
        "mu_h_test": test_eval["mu_h"],
        "feasibility_test": test_eval["feasibility"],
        "hidden_dae_residual_test": test_eval["hidden_dae_residual"],
        "alpha_c": alpha_c,
        "alpha_k": alpha_k,
        "alpha_h": alpha_h,
        "alpha_i_k": alpha_i_k,
        "alpha_i_h": alpha_i_h,
        "alpha_mu_k": alpha_mu_k,
        "alpha_mu_h": alpha_mu_h,
        "c_0": float(c_0),
        "h_0": h_0,
        "i_k_0": float(i_k_0),
        "i_h_0": float(i_h_0),
        "mu_k_0": float(mu_k_0),
        "mu_h_0": float(mu_h_0),
        "initial_condition_residual": initial_residual,
        "rkhs_norms": rkhs_norms,
        "train_residuals": train_residuals,
        "validation_residuals": validation_residuals,
        "max_train_residual": max_train_residual,
        "max_validation_residual": max_validation_residual,
        "solve_time": solve_time,
        "wall_time": elapsed,
        "solver_status": solver_status,
        "solution_status": solution_status,
        "stationarity": result.solution_stationarity,
        "primal_feasibility": result.solution_primal_feasibility,
        "valid_solution": valid_solution,
        "rejection_reason": "accepted" if valid_solution else ",".join(rejection_reasons),
        "kernel_solution": kernel_solution,
    }


if __name__ == "__main__":
    jsonargparse.CLI(human_capital_matern)
