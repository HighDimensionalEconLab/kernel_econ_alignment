import jax
import jax.numpy as jnp
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition
import jsonargparse
from jax import config
from kernels import integrated_matern_kernel_matrices
from typing import List, Optional

config.update("jax_enable_x64", True)


def optimal_advertising_matern(
    rho_hat: float = 0.11,
    c: float = 0.5,
    beta: float = 0.05,
    kappa: float = 0.5,
    x_0: float = 0.4,
    nu: float = 0.5,
    sigma: float = 1.0,
    rho: float = 15,
    solver_type: str = "ipopt",
    train_T: float = 40.0,
    train_points: int = 81,
    test_T: float = 50.0,
    test_points: int = 100,
    benchmark_T: float = 60.0,
    benchmark_points: int = 300,
    train_points_list: Optional[List[float]] = None,
    verbose: bool = False,
):
    # if passing in `train_points` then doesn't us a grid.  Otherwise, uses linspace
    if train_points_list is None:
        train_data = jnp.linspace(0, train_T, train_points)
    else:
        train_data = jnp.array(train_points_list)
    test_data = jnp.linspace(0, test_T, test_points)
    benchmark_grid = jnp.linspace(0, benchmark_T, benchmark_points)

    # Construct kernel matrices
    N = len(train_data)
    K, K_tilde = integrated_matern_kernel_matrices(
        train_data, train_data, nu, sigma, rho
    )
    K = np.array(K)  # pyomo doesn't support jax arrays
    K_tilde = np.array(K_tilde)

    # Create pyomo model and variables
    m = pyo.ConcreteModel()
    m.I = range(N)
    m.alpha_x = pyo.Var(m.I, within=pyo.Reals, initialize=0.0)
    m.alpha_mu = pyo.Var(m.I, within=pyo.Reals, initialize=0.0)
    m.alpha_b = pyo.Var(m.I, within=pyo.Reals, initialize=0.0)
    m.mu_0 = pyo.Var(within=pyo.NonNegativeReals, initialize=0.0)
    m.b_0 = pyo.Var(within=pyo.NonNegativeReals, initialize=0.0)

    # Map kernels to variables. Pyomo doesn't support mu_0 + K_tilde @ m.alpha_mu
    def mu(m, i):
        return m.mu_0 + sum(K_tilde[i, j] * m.alpha_mu[j] for j in m.I)

    def x(m, i):
        return x_0 + sum(K_tilde[i, j] * m.alpha_x[j] for j in m.I)

    def dmu_dt(m, i):
        return sum(K[i, j] * m.alpha_mu[j] for j in m.I)

    def dx_dt(m, i):
        return sum(K[i, j] * m.alpha_x[j] for j in m.I)

    def b(m, i):
        return m.b_0 + sum(K_tilde[i, j] * m.alpha_b[j] for j in m.I)
    
    # Define constraints and objective for model and solve
    @m.Constraint(m.I)  # for each index in m.I
    def dx_dt_constraint(m, i):
        return dx_dt(m, i) == ((1 - x(m, i))**(1 / (1 - kappa))) * ((kappa * mu(m, i))**(kappa / (1 - kappa))) - beta * x(m, i)

    gamma = (beta + rho_hat) / c
    @m.Constraint(m.I)  # for each index in m.I
    def dmu_dt_constraint(m, i):
        return dmu_dt(m, i) == -gamma + (rho_hat + beta)*mu(m, i) + (mu(m, i)**(1 / (1 - kappa)))*((kappa * (1 - x(m, i)))**(kappa / (1 - kappa)))

    @m.Constraint(m.I)  # for each index in m.I
    def b_constraint(m, i):
        return mu(m, i) * x(m, i) == b(m, i)
    
    @m.Objective(sense=pyo.minimize)
    def min_norm(m):  # alpha @ K @ alpha not supported by pyomo
        return sum(K[i, j] * m.alpha_b[i] * m.alpha_b[j] for i in m.I for j in m.I) 

    solver = pyo.SolverFactory(solver_type)
    options = {
        "tol": 1e-6,  # Tighten the tolerance for optimality
        "dual_inf_tol": 1e-6,  # Tighten the dual infeasibility tolerance
        "constr_viol_tol": 1e-6,  # Tighten the constraint violation tolerance
        "max_iter": 1000,  # Adjust the maximum number of iterations if needed
    } 
    results = solver.solve(m, tee=verbose, options=options)
    if not results.solver.termination_condition == TerminationCondition.optimal:
        print(str(results.solver))  # raise exception?

    alpha_mu = jnp.array([pyo.value(m.alpha_mu[i]) for i in m.I])
    alpha_x = jnp.array([pyo.value(m.alpha_x[i]) for i in m.I])
    mu_0 = pyo.value(m.mu_0)

    # Interpolator using training data
    @jax.jit
    def kernel_solution(test_data):
        # pointwise comparison test_data to train_data
        K_test, K_tilde_test = integrated_matern_kernel_matrices(
            test_data, train_data, nu, sigma, rho
        )
        mu_test = mu_0 + K_tilde_test @ alpha_mu
        x_test = x_0 + K_tilde_test @ alpha_x
        return x_test, mu_test

    # Generate test_data and compare to the benchmark
    x_test, mu_test = kernel_solution(test_data)

    print(f"solve_time(s) = {results.solver.Time}")
    return {
        "t_train": train_data,
        "t_test": test_data,
        "x_test": x_test,
        "mu_test": mu_test,
        "alpha_mu": alpha_mu,
        "alpha_x": alpha_x,
        "mu_0": mu_0,
        "solve_time": results.solver.Time,
        "kernel_solution": kernel_solution,  # interpolator
    }


if __name__ == "__main__":
    jsonargparse.CLI(optimal_advertising_matern)
