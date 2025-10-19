import jax
import jax.numpy as jnp
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition
import jsonargparse
from jax import config
from kernels import integrated_matern_kernel_matrices
from neoclassical_growth_benchmark import neoclassical_growth_benchmark
from typing import List, Optional

config.update("jax_enable_x64", True)


def neoclassical_growth_matern(
    a: float = 1 / 3,
    delta: float = 0.1,
    rho_hat: float = 0.11,
    k_0: float = 1.0,#k_0 is the state variable initial conditions here, i.e., x_0
    nu: float = 0.5,
    sigma: float = 1.0,
    rho: float = 10,
    solver_type: str = "ipopt",
    train_T: float = 40.0,
    train_points: int = 41,
    test_T: float = 50,
    test_points: int = 100,
    benchmark_T: float = 60.0,
    benchmark_points: int = 300,
    train_points_list: Optional[List[float]] = None,
    lambda_p: float = 0.0, #Smooting penalty for the optimizer, This is purely because of the DAE term mu*c = 1
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
    m.alpha_mu = pyo.Var(m.I, within=pyo.Reals, initialize=0.0)
    m.alpha_c = pyo.Var(m.I, within=pyo.Reals, initialize=0.0)
    m.alpha_k = pyo.Var(m.I, within=pyo.Reals, initialize=0.0) #k is the state variable here, i.e., x
    m.c_0 = pyo.Var(within=pyo.NonNegativeReals, initialize=k_0**a - delta * k_0)
    m.mu_0 = pyo.Var(within=pyo.NonNegativeReals, initialize=k_0**a - delta * k_0)

    # Map kernels to variables. Pyomo doesn't support c_0 + K_tilde @ m.alpha_c
    def mu(m, i):
        return m.mu_0 + sum(K_tilde[i, j] * m.alpha_mu[j] for j in m.I)

    def c(m, i):
        return m.c_0 + sum(K_tilde[i, j] * m.alpha_c[j] for j in m.I)

    def k(m, i): #k is the state variable here, i.e., x
        return k_0 + sum(K_tilde[i, j] * m.alpha_k[j] for j in m.I)

    def dmu_dt(m, i):
        return sum(K[i, j] * m.alpha_mu[j] for j in m.I)

    def dk_dt(m, i): #dk_dt is the state variable's derivative here, i.e., dx_dt
        return sum(K[i, j] * m.alpha_k[j] for j in m.I)

    # Define constraints and objective for model and solve
    @m.Constraint(m.I)  # for each index in m.I
    def resource_constraint(m, i):
        return dk_dt(m, i) == k(m, i) ** a - delta * k(m, i) - c(m, i)

    @m.Constraint(m.I)  # for each index in m.I
    def euler(m, i):
        return dmu_dt(m, i) == -mu(m, i) * (a * k(m, i) ** (a - 1) - delta - rho_hat)

    @m.Constraint(m.I)  # for each index in m.I
    def shadow_price(m, i):
        return mu(m, i) * c(m, i) == 1.0


    @m.Objective(sense=pyo.minimize)
    def min_norm(m):  # alpha @ K @ alpha not supported by pyomo
        return sum(K[i, j] * m.alpha_mu[i] * m.alpha_mu[j] for i in m.I for j in m.I)+sum(K[i, j] * m.alpha_k[i] * m.alpha_k[j] for i in m.I for j in m.I) + lambda_p*(sum(K[i, j] * m.alpha_c[i] * m.alpha_c[j] for i in m.I for j in m.I)
        + sum(K[i, j] * m.alpha_k[i] * m.alpha_k[j] for i in m.I for j in m.I)+ sum(K[i, j] * m.alpha_mu[i] * m.alpha_mu[j] for i in m.I for j in m.I)
        )
        #lambda_p makes sure the optimizer returns smooth (non-wiggly solutions in the extrapolation), we set it to zero
    solver = pyo.SolverFactory(solver_type)
    options = {
        "tol": 1e-8,  # Tighten the tolerance for optimality
        "dual_inf_tol": 1e-8,  # Tighten the dual infeasibility tolerance
        "constr_viol_tol": 1e-8,  # Tighten the constraint violation tolerance
        "max_iter": 2000,  # Adjust the maximum number of iterations if needed
    }  # See https://coin-or.github.io/Ipopt/OPTIONS.html for more details # can add options here.   See https://coin-or.github.io/Ipopt/OPTIONS.html#OPTIONS_AMPL
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

    sol_benchmark = neoclassical_growth_benchmark(
        a, delta, rho_hat, 1.0, k_0, benchmark_grid
    )

    # Generate test_data and compare to the benchmark
    k_benchmark, c_benchmark = sol_benchmark(test_data)
    k_test, c_test = kernel_solution(test_data)

    k_rel_error = jnp.abs(k_benchmark - k_test) / k_benchmark
    c_rel_error = jnp.abs(c_benchmark - c_test) / c_benchmark
    print(
        f"solve_time(s) = {results.solver.Time}, E(|rel_error(k)|) = {k_rel_error.mean()}, E(|rel_error(c)|) = {c_rel_error.mean()}"
    )
    return {
        "t_train": train_data,
        "t_test": test_data,
        "k_test": k_test,
        "c_test": c_test,
        "k_benchmark": k_benchmark,
        "c_benchmark": c_benchmark,
        "k_rel_error": k_rel_error,
        "c_rel_error": c_rel_error,
        "alpha_c": alpha_c,
        "alpha_k": alpha_k,
        "c_0": c_0,
        "solve_time": results.solver.Time,
        "kernel_solution": kernel_solution,  # interpolator
        "benchmark_solution": sol_benchmark,  # interpolator
    }


if __name__ == "__main__":
    jsonargparse.CLI(neoclassical_growth_matern)
