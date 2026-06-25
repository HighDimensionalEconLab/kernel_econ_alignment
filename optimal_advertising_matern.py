import time
from typing import List, Optional

import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
import unopy
from jax import config

from kernels import integrated_matern_kernel_matrices
from rkhs import rkhs_norm_squared

config.update("jax_enable_x64", True)

NLP_OPTIONS = dict(preset="ipopt")
DOMAIN_EPS = 1e-8


def optimal_advertising_matern(
    rho_hat: float = 0.11,
    c: float = 0.5,
    beta: float = 0.05,
    kappa: float = 0.5,
    x_0: float = 0.4,
    nu: float = 0.5,
    sigma: float = 1.0,
    rho: float = 15,
    train_T: float = 40.0,
    train_points: int = 41,
    test_T: float = 50.0,
    test_points: int = 100,
    benchmark_T: float = 60.0,
    benchmark_points: int = 300,
    train_points_list: Optional[List[float]] = None,
    verbose: bool = False,
):
    _ = (benchmark_T, benchmark_points)
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

    n_variables = 3 * n_train + 2
    n_constraints = 4 * n_train
    gamma = (beta + rho_hat) / c
    control_power = (1.0 - kappa) / kappa
    u_0_guess = max(beta * x_0 / max(1.0 - x_0, DOMAIN_EPS), DOMAIN_EPS)
    mu_0_guess = max(gamma / (rho_hat + beta + u_0_guess), DOMAIN_EPS)
    x_initial = np.zeros(n_variables, dtype=np.float64)
    x_initial[3 * n_train] = mu_0_guess
    x_initial[3 * n_train + 1] = u_0_guess

    def unpack(z):
        alpha_x = z[:n_train]
        alpha_mu = z[n_train : 2 * n_train]
        alpha_u = z[2 * n_train : 3 * n_train]
        mu_0 = z[3 * n_train]
        u_0 = z[3 * n_train + 1]
        return alpha_x, alpha_mu, alpha_u, mu_0, u_0

    def path_values(z, K_eval, K_tilde_eval):
        alpha_x, alpha_mu, alpha_u, mu_0, u_0 = unpack(z)
        x = x_0 + K_tilde_eval @ alpha_x
        mu = mu_0 + K_tilde_eval @ alpha_mu
        u = u_0 + K_tilde_eval @ alpha_u
        dx_dt = K_eval @ alpha_x
        dmu_dt = K_eval @ alpha_mu
        return x, mu, u, dx_dt, dmu_dt

    def objective(z):
        alpha_x, alpha_mu, alpha_u, _, _ = unpack(z)
        return (
            alpha_x @ K_jax @ alpha_x
            + alpha_mu @ K_jax @ alpha_mu
            + alpha_u @ K_jax @ alpha_u
        )

    def constraints(z):
        x, mu, u, dx_dt, dmu_dt = path_values(z, K_jax, K_tilde_jax)
        state = dx_dt - ((1.0 - x) * u - beta * x)
        costate = dmu_dt - (-gamma + (rho_hat + beta) * mu + mu * u)
        shadow_price = u**control_power - kappa * mu * (1.0 - x)
        return jnp.concatenate([state, costate, shadow_price, u])

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
    variable_lower_bounds[3 * n_train] = 0.0
    variable_lower_bounds[3 * n_train + 1] = DOMAIN_EPS
    constraint_lower_bounds = np.concatenate(
        [
            np.zeros(3 * n_train, dtype=np.float64),
            np.full(n_train, DOMAIN_EPS, dtype=np.float64),
        ]
    )
    constraint_upper_bounds = np.concatenate(
        [
            np.zeros(3 * n_train, dtype=np.float64),
            np.full(n_train, np.inf, dtype=np.float64),
        ]
    )

    model = unopy.Model(
        unopy.PROBLEM_NONLINEAR,
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
    alpha_x, alpha_mu, alpha_u, mu_0, u_0 = unpack(z)
    rkhs_norms = {
        "x": rkhs_norm_squared(alpha_x, K),
        "mu": rkhs_norm_squared(alpha_mu, K),
        "u": rkhs_norm_squared(alpha_u, K),
    }

    train_constraints = constraints(z)
    train_residuals = {
        "state": train_constraints[:n_train],
        "costate": train_constraints[n_train : 2 * n_train],
        "shadow_price": train_constraints[2 * n_train : 3 * n_train],
    }
    max_train_residual = max(
        float(jnp.max(jnp.abs(residual))) for residual in train_residuals.values()
    )

    @jax.jit
    def kernel_solution(test_points_data):
        K_test, K_tilde_test = integrated_matern_kernel_matrices(
            test_points_data, train_data, nu, sigma, rho
        )
        x_test, mu_test, u_test, _, _ = path_values(z, K_test, K_tilde_test)
        return x_test, mu_test, u_test

    x_test, mu_test, u_test = kernel_solution(test_data)
    solve_time = result.cpu_time
    if solve_time is None:
        solve_time = elapsed
    solver_status = str(result.optimization_status).split(".")[-1]
    solution_status = str(result.solution_status).split(".")[-1]
    print(f"solve_time(s) = {solve_time}")
    return {
        "t_train": train_data,
        "t_test": test_data,
        "x_test": x_test,
        "mu_test": mu_test,
        "u_test": u_test,
        "alpha_mu": alpha_mu,
        "alpha_x": alpha_x,
        "alpha_u": alpha_u,
        "mu_0": float(mu_0),
        "u_0": float(u_0),
        "rkhs_norms": rkhs_norms,
        "train_residuals": train_residuals,
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
    jsonargparse.CLI(optimal_advertising_matern)
