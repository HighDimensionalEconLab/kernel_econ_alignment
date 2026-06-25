import time
from typing import List, Optional

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
import unopy
from jax import config

from asset_pricing_benchmark import mu_f_array
from kernels import integrated_matern_kernel_matrices
from rkhs import rkhs_norm_squared

config.update("jax_enable_x64", True)

NLP_OPTIONS = dict(preset="ipopt")


def asset_pricing_matern(
    r: float = 0.1,
    c: float = 0.02,
    g: float = -0.2,
    x_0: float = 0.01,
    nu: float = 0.5,
    sigma: float = 1.0,
    rho: float = 10,
    train_T: float = 40.0,
    train_points: int = 41,
    test_T: float = 50.0,
    test_points: int = 100,
    train_points_list: Optional[List[float]] = None,
    verbose: bool = False,
):
    if train_points_list is None:
        train_data = jnp.linspace(0, train_T, train_points)
    else:
        train_data = jnp.array(train_points_list)
    test_data = jnp.linspace(0, test_T, test_points)

    n_train = len(train_data)
    K, K_tilde = integrated_matern_kernel_matrices(
        train_data, train_data, nu, sigma, rho
    )
    K = np.asarray((K + K.T) / 2)
    K_tilde = np.asarray(K_tilde)
    K_jax = jnp.asarray(K)
    K_tilde_jax = jnp.asarray(K_tilde)
    x_train = (x_0 + c / g) * jnp.exp(g * train_data) - c / g

    n_variables = n_train + 1
    n_constraints = n_train
    p_0_guess = float(mu_f_array(jnp.array([0.0]), c, g, r, x_0)[0])
    x_initial = np.zeros(n_variables, dtype=np.float64)
    x_initial[n_train] = max(p_0_guess, 0.0)

    def unpack(z):
        alpha = z[:n_train]
        p_0 = z[n_train]
        return alpha, p_0

    def path_values(z, K_eval, K_tilde_eval):
        alpha, p_0 = unpack(z)
        p = p_0 + K_tilde_eval @ alpha
        dp_dt = K_eval @ alpha
        return p, dp_dt

    def objective(z):
        alpha, _ = unpack(z)
        return alpha @ K_jax @ alpha

    def constraints(z):
        p, dp_dt = path_values(z, K_jax, K_tilde_jax)
        return dp_dt - (r * p - x_train)

    def lagrangian(z, objective_multiplier, multipliers):
        return objective_multiplier * objective(z) + jnp.dot(multipliers, constraints(z))

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
    variable_lower_bounds[n_train] = 0.0
    constraint_lower_bounds = np.zeros(n_constraints, dtype=np.float64)
    constraint_upper_bounds = np.zeros(n_constraints, dtype=np.float64)

    model = unopy.Model(
        unopy.PROBLEM_QUADRATIC,
        n_variables,
        variable_lower_bounds,
        variable_upper_bounds,
        unopy.ZERO_BASED_INDEXING,
    )

    def objective_callback(z):
        return float(objective_value(jnp.asarray(z)))

    def objective_gradient_callback(z, gradient):
        gradient[:] = np.asarray(objective_gradient(jnp.asarray(z)))

    model.set_objective(
        unopy.MINIMIZE, objective_callback, objective_gradient_callback
    )

    jacobian_rows = np.repeat(np.arange(n_constraints, dtype=np.int32), n_variables)
    jacobian_columns = np.tile(np.arange(n_variables, dtype=np.int32), n_constraints)

    def constraints_callback(z, constraint_output):
        constraint_output[:] = np.asarray(constraint_values(jnp.asarray(z)))

    def jacobian_callback(z, jacobian_output):
        jacobian_output[:] = np.asarray(
            constraint_jacobian(jnp.asarray(z))
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

    def hessian_callback(z, objective_multiplier, multipliers, hessian_output):
        hessian = np.asarray(
            lagrangian_hessian(
                jnp.asarray(z),
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

    z = jnp.asarray(np.array(result.primal_solution, dtype=np.float64))
    alpha, p_0 = unpack(z)
    rkhs_norms = {"p": rkhs_norm_squared(alpha, K)}
    train_residual = constraints(z)
    max_train_residual = float(jnp.max(jnp.abs(train_residual)))

    @jax.jit
    def kernel_solution(test_points_data):
        _, K_tilde_test = integrated_matern_kernel_matrices(
            test_points_data, train_data, nu, sigma, rho
        )
        return p_0 + K_tilde_test @ alpha

    p_benchmark = mu_f_array(test_data, c, g, r, x_0)
    p_test = kernel_solution(test_data)
    p_rel_error = jnp.abs(p_benchmark - p_test) / p_benchmark

    solve_time = result.cpu_time
    if solve_time is None:
        solve_time = elapsed
    solver_status = str(result.optimization_status).split(".")[-1]
    solution_status = str(result.solution_status).split(".")[-1]
    print(
        f"solve_time(s) = {solve_time}, E(|rel_error(p)|) = {p_rel_error.mean()}"
    )
    return {
        "t_train": train_data,
        "t_test": test_data,
        "p_test": p_test,
        "p_benchmark": p_benchmark,
        "p_rel_error": p_rel_error,
        "alpha": alpha,
        "p_0": float(p_0),
        "rkhs_norms": rkhs_norms,
        "train_residuals": {"asset_pricing": train_residual},
        "max_train_residual": max_train_residual,
        "solve_time": solve_time,
        "wall_time": elapsed,
        "solver_status": solver_status,
        "solution_status": solution_status,
        "stationarity": result.solution_stationarity,
        "primal_feasibility": result.solution_primal_feasibility,
        "kernel_solution": kernel_solution,
    }


if __name__ == "__main__":
    jsonargparse.CLI(asset_pricing_matern)
