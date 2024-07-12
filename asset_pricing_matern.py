import jax
import jax.numpy as jnp
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition
import jsonargparse
from jax import config
from kernels import integrated_matern_kernel_matrices
from asset_pricing_benchmark import p_f_array
from typing import List, Optional

config.update("jax_enable_x64", True)


def asset_pricing_matern(
    r: float = 0.1,
    c: float = 0.02,
    g: float = -0.2,
    y_0: float = 0.01,
    nu: float = 0.5,
    sigma: float = 1.0,
    rho: float = 10,
    solver_type: str = "ipopt",
    train_T: float = 30.0,
    train_points: int = 31,
    test_T: float = 40.0,
    test_points: int = 100,
    train_points_list: Optional[List[float]] = None,
    verbose: bool = False,
):
    # if passing in `train_points` then doesn't us a grid.  Otherwise, uses linspace
    if train_points_list is None:
        train_data = jnp.linspace(0, train_T, train_points)
    else:
        train_data = jnp.array(train_points_list)
    test_data = jnp.linspace(0, test_T, test_points)

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
    m.alpha = pyo.Var(m.I, within=pyo.Reals, initialize=0.0)
    m.p_0 = pyo.Var(within=pyo.NonNegativeReals, initialize=0.0)

    # Map kernels to variables. Pyomo doesn't support p_0 + K_tilde @ m.alpha
    def p(m, i):
        return m.p_0 + sum(K_tilde[i, j] * m.alpha[j] for j in m.I)

    def dp_dt(m, i):
        return sum(K[i, j] * m.alpha[j] for j in m.I)

    def y(i):
        return (y_0 + (c / g)) * np.exp(g * i) - (c / g)

    # Define constraints and objective for model and solve
    @m.Constraint(m.I)  # for each index in m.I
    def dp_dt_constraint(m, i):
        return dp_dt(m, i) == r * p(m, i) - y(i)

    @m.Objective(sense=pyo.minimize)
    def min_norm(m):  # alpha @ K @ alpha not supported by pyomo
        return sum(K[i, j] * m.alpha[i] * m.alpha[j] for i in m.I for j in m.I)

    solver = pyo.SolverFactory(solver_type)
    options = (
        {}
    )  # can add options here.   See https://coin-or.github.io/Ipopt/OPTIONS.html#OPTIONS_AMPL
    results = solver.solve(m, tee=verbose, options=options)
    if not results.solver.termination_condition == TerminationCondition.optimal:
        print(str(results.solver))  # raise exception?

    alpha = jnp.array([pyo.value(m.alpha[i]) for i in m.I])
    p_0 = pyo.value(m.p_0)

    # Interpolator using training data
    @jax.jit
    def kernel_solution(test_data):
        # pointwise comparison test_data to train_data
        K_test, K_tilde_test = integrated_matern_kernel_matrices(
            test_data, train_data, nu, sigma, rho
        )
        p_test = p_0 + K_tilde_test @ alpha
        return p_test

    # Generate test_data and compare to the benchmark
    p_benchmark = p_f_array(test_data, c, g, r, y_0)
    p_test = kernel_solution(test_data)

    p_rel_error = jnp.abs(p_benchmark - p_test) / p_benchmark
    print(
        f"solve_time(s) = {results.solver.Time}, E(|rel_error(p)|) = {p_rel_error.mean()}"
    )
    return {
        "t_train": train_data,
        "t_test": test_data,
        "p_test": p_test,
        "p_benchmark": p_benchmark,
        "p_rel_error": p_rel_error,
        "alpha": alpha,
        "p_0": p_0,
        "solve_time": results.solver.Time,
        "kernel_solution": kernel_solution,  # interpolator
    }


if __name__ == "__main__":
    jsonargparse.CLI(asset_pricing_matern)
