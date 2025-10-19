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


def neoclassical_growth_concave_convex_matern(
    a: float = 1 / 3,
    delta: float = 0.1,
    rho_hat: float = 0.11,
    A: float = 0.5,
    b_1: float = 3.0,
    b_2: float = 2.5,
    k_0: float = 1.0,
    nu: float = 0.5,
    sigma: float = 1.0,
    rho: float = 10,
    solver_type: str = "ipopt",
    train_T: float = 30.0,
    train_points: int = 31,
    test_T: float = 40.0,
    test_points: int = 100,
    benchmark_T: float = 60.0,
    benchmark_points: int = 300,
    train_points_list: Optional[List[float]] = None,
    lambda_k: float = 1.0,
    lambda_c: float = 1.0,
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
    m.alpha_c = pyo.Var(m.I, within=pyo.Reals, initialize=0.0)
    m.alpha_k = pyo.Var(m.I, within=pyo.Reals, initialize=0.0)
    m.c_0 = pyo.Var(within=pyo.NonNegativeReals, initialize=k_0**a - delta * k_0)

    # Map kernels to variables. Pyomo doesn't support c_0 + K_tilde @ m.alpha_c
    def c(m, i):
        return m.c_0 + sum(K_tilde[i, j] * m.alpha_c[j] for j in m.I)

    def k(m, i):
        return k_0 + sum(K_tilde[i, j] * m.alpha_k[j] for j in m.I)

    def dc_dt(m, i):
        return sum(K[i, j] * m.alpha_c[j] for j in m.I)

    def dk_dt(m, i):
        return sum(K[i, j] * m.alpha_k[j] for j in m.I)

    # Production function
    base = b_2 / (b_1 - 1)
    exponent = 1 / a
    k_bar = base**exponent

    def f(k):
        return A * pyo.Expr_if(k < k_bar, k**a, b_1 * k**a - b_2)

    def f_prime(k):
        return pyo.Expr_if(
            k < k_bar, A * a * (k ** (a - 1)), A * a * b_1 * (k ** (a - 1))
        )

    # Define constraints and objective for model and solve
    @m.Constraint(m.I)  # for each index in m.I
    def resource_constraint(m, i):
        return dk_dt(m, i) == f(k(m, i)) - delta * k(m, i) - c(m, i)

    @m.Constraint(m.I)  # for each index in m.I
    def euler(m, i):
        return dc_dt(m, i) == c(m, i) * (f_prime(k(m, i)) - delta - rho_hat)

    @m.Objective(sense=pyo.minimize)
    def min_norm(m):  # alpha @ K @ alpha not supported by pyomo
        return lambda_k * sum(K[i, j] * m.alpha_k[i] * m.alpha_k[j] for i in m.I for j in m.I) + lambda_c * sum(K[i, j] * m.alpha_c[i] * m.alpha_c[j] for i in m.I for j in m.I)

    solver = pyo.SolverFactory(solver_type)
    options = (
        {}
    )  # can add options here.   See https://coin-or.github.io/Ipopt/OPTIONS.html#OPTIONS_AMPL
    results = solver.solve(m, tee=verbose, options=options)
    if not results.solver.termination_condition == TerminationCondition.optimal:
        print(str(results.solver))  # raise exception?

    alpha_c = jnp.array([pyo.value(m.alpha_c[i]) for i in m.I])
    alpha_k = jnp.array([pyo.value(m.alpha_k[i]) for i in m.I])
    c_0 = pyo.value(m.c_0)

    # Interpolator using training data
    @jax.jit
    def kernel_solution(test_data):
        # pointwise comparison test_data to train_data
        K_test, K_tilde_test = integrated_matern_kernel_matrices(
            test_data, train_data, nu, sigma, rho
        )
        c_test = c_0 + K_tilde_test @ alpha_c
        k_test = k_0 + K_tilde_test @ alpha_k
        return k_test, c_test

    # Generate test_data and compare to the benchmark
    k_test, c_test = kernel_solution(test_data)

    print(f"solve_time(s) = {results.solver.Time}")
    return {
        "t_train": train_data,
        "t_test": test_data,
        "k_test": k_test,
        "c_test": c_test,
        "alpha_c": alpha_c,
        "alpha_k": alpha_k,
        "c_0": c_0,
        "solve_time": results.solver.Time,
        "kernel_solution": kernel_solution,  # interpolator
    }


if __name__ == "__main__":
    jsonargparse.CLI(neoclassical_growth_concave_convex_matern)
