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

from scipy.optimize import fsolve

def human_capital_matern(
    a_k: float = 1 / 3,
    a_h: float = 1 / 4,
    delta_k: float = 0.1,
    delta_h: float = 0.05,
    rho_hat: float = 0.11,
    k_0: float = 1.5, #3.0024724187979452, #1.5,
    #h_0: float = 2.9555587872542275, #1.3,
    nu: float = 0.5,
    sigma: float = 1.0,
    rho: float = 20,
    lambda_k: float = 1.0,
    lambda_h: float = 1.0,
    lambda_c: float = 1.0,
    lambda_i_h: float = 1.0,
    lambda_i_k: float = 1.0,
    solver_type: str = "ipopt",
    train_T: float = 80.0,
    train_points: int = 60,
    test_T: float = 100.0,
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

    # Production function
    def f(k, h):
        return (k**a_k) * (h**a_h)
    
    def f_k(k, h):
        return (a_k * k**(a_k - 1)) * (h**a_h)
    
    def f_h(k, h):
        return (k**a_k) * (a_h * h**(a_h - 1))
    
    def no_arbitrage_constraint(h):
        return f_h(k_0, h) - f_k(k_0, h) - delta_h + delta_k
    
    initial_guess = [k_0]
    result = fsolve(no_arbitrage_constraint, initial_guess)
    h_0 = result[0]

    # Create pyomo model and variables
    m = pyo.ConcreteModel()
    m.I = range(N)
    m.alpha_c = pyo.Var(m.I, within=pyo.Reals, initialize=0.0)
    m.alpha_k = pyo.Var(m.I, within=pyo.Reals, initialize=0.0)
    m.alpha_h = pyo.Var(m.I, within=pyo.Reals, initialize=0.0)
    m.alpha_i_k = pyo.Var(m.I, within=pyo.Reals, initialize=0.0)
    m.alpha_i_h = pyo.Var(m.I, within=pyo.Reals, initialize=0.0)
    m.c_0 = pyo.Var(within=pyo.NonNegativeReals, initialize=f(k_0, h_0) - delta_h * h_0 - delta_k * k_0)
    m.i_k_0 = pyo.Var(within=pyo.NonNegativeReals, initialize=delta_k * k_0)
    m.i_h_0 = pyo.Var(within=pyo.NonNegativeReals, initialize=delta_h * h_0)

    # Map kernels to variables. Pyomo doesn't support c_0 + K_tilde @ m.alpha_c
    def c(m, i):
        return m.c_0 + sum(K_tilde[i, j] * m.alpha_c[j] for j in m.I)

    def k(m, i):
        return k_0 + sum(K_tilde[i, j] * m.alpha_k[j] for j in m.I)
    
    def h(m, i):
        return h_0 + sum(K_tilde[i, j] * m.alpha_h[j] for j in m.I)
    
    def i_k(m, i):
        return m.i_k_0 + sum(K_tilde[i, j] * m.alpha_i_k[j] for j in m.I)
    
    def i_h(m, i):
        return m.i_h_0 + sum(K_tilde[i, j] * m.alpha_i_h[j] for j in m.I)

    def dc_dt(m, i):
        return sum(K[i, j] * m.alpha_c[j] for j in m.I)

    def dk_dt(m, i):
        return sum(K[i, j] * m.alpha_k[j] for j in m.I)
    
    def dh_dt(m, i):
        return sum(K[i, j] * m.alpha_h[j] for j in m.I)

    # Define constraints and objective for model and solve
    @m.Constraint(m.I)  # for each index in m.I
    def dk_dt_constraint(m, i):
        return dk_dt(m, i) == i_k(m, i) - delta_k * k(m, i)
    
    @m.Constraint(m.I)  # for each index in m.I
    def dh_dt_constraint(m, i):
        return dh_dt(m, i) == i_h(m, i) - delta_h * h(m, i)

    @m.Constraint(m.I)  # for each index in m.I
    def dc_dt_constraint(m, i):
        return dc_dt(m, i) == c(m, i) * (f_k(k(m, i), h(m, i)) - delta_k - rho_hat)
    
    @m.Constraint(m.I)  # for each index in m.I
    def feasibility(m, i):
        return 0.0 == c(m, i) + i_h(m, i) + i_k(m, i) - f(k(m, i), h(m, i))
    
    @m.Constraint(m.I)  # for each index in m.I
    def no_arbitrage(m, i):
        return 0.0 == f_h(k(m, i), h(m, i)) - f_k(k(m, i), h(m, i)) - delta_h + delta_k

    @m.Objective(sense=pyo.minimize)
    def min_norm(m):  # alpha @ K @ alpha not supported by pyomo
        return lambda_k * sum(K[i, j] * m.alpha_k[i] * m.alpha_k[j] for i in m.I for j in m.I) + lambda_h * sum(K[i, j] * m.alpha_h[i] * m.alpha_h[j] for i in m.I for j in m.I)+ lambda_c*sum(K[i, j] * m.alpha_c[i] * m.alpha_c[j] for i in m.I for j in m.I) + lambda_i_h*sum(K[i, j] * m.alpha_i_h[i] * m.alpha_i_h[j] for i in m.I for j in m.I) + lambda_i_k*sum(K[i, j] * m.alpha_i_k[i] * m.alpha_i_k[j] for i in m.I for j in m.I)

    solver = pyo.SolverFactory(solver_type)
    options = (
        {
            'bound_relax_factor': 0,
        }
    )  # can add options here.   See https://coin-or.github.io/Ipopt/OPTIONS.html#OPTIONS_AMPL
    results = solver.solve(m, tee=verbose, options=options)
    if not results.solver.termination_condition == TerminationCondition.optimal:
        print(str(results.solver))  # raise exception?

    alpha_c = jnp.array([pyo.value(m.alpha_c[i]) for i in m.I])
    alpha_k = jnp.array([pyo.value(m.alpha_k[i]) for i in m.I])
    alpha_h = jnp.array([pyo.value(m.alpha_h[i]) for i in m.I])
    alpha_i_k = jnp.array([pyo.value(m.alpha_i_k[i]) for i in m.I])
    alpha_i_h = jnp.array([pyo.value(m.alpha_i_h[i]) for i in m.I])
    c_0 = pyo.value(m.c_0)
    i_k_0 = pyo.value(m.i_k_0)
    i_h_0 = pyo.value(m.i_h_0)

    # Interpolator using training data
    #@jax.jit
    def kernel_solution(test_data):
        # pointwise comparison test_data to train_data
        K_test, K_tilde_test = integrated_matern_kernel_matrices(
            test_data, train_data, nu, sigma, rho
        )
        c_test = c_0 + K_tilde_test @ alpha_c
        k_test = k_0 + K_tilde_test @ alpha_k
        h_test = h_0 + K_tilde_test @ alpha_h
        i_k_test = i_k_0 + K_tilde_test @ alpha_i_k
        i_h_test = i_h_0 + K_tilde_test @ alpha_i_h
        feasibility_test = c_test + i_h_test + i_k_test - f(k_test, h_test)
        return k_test, h_test, c_test, i_k_test, i_h_test, feasibility_test

    # Generate test_data
    k_test, h_test, c_test, i_k_test, i_h_test, feasibility_test = kernel_solution(test_data)

    print(
        f"solve_time(s) = {results.solver.Time}"
    )
    return {
        "t_train": train_data,
        "t_test": test_data,
        "k_test": k_test,
        "h_test": h_test,
        "c_test": c_test,
        "i_k_test": i_k_test,
        "i_h_test": i_h_test,
        "feasibility_test": feasibility_test,
        "alpha_c": alpha_c,
        "alpha_k": alpha_k,
        "alpha_h": alpha_h,
        "alpha_i_k": alpha_i_k,
        "alpha_i_h": alpha_i_h,
        "c_0": c_0,
        "i_k_0": i_k_0,
        "i_h_0": i_h_0,
        "solve_time": results.solver.Time,
        "kernel_solution": kernel_solution,  # interpolator
    }


if __name__ == "__main__":
    jsonargparse.CLI(human_capital_matern)
