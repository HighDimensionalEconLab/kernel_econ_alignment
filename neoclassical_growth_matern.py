import time
import jax
import jax.numpy as jnp
import numpy as np
import cvxpy as cp
import jsonargparse
from jax import config
from kernels import integrated_matern_kernel_matrices
from neoclassical_growth_benchmark import neoclassical_growth_benchmark
from rkhs import rkhs_norm_squared
from typing import List, Optional

config.update("jax_enable_x64", True)

# CVXPY DNLP implementation. The optimal-growth FOC collocation is genuinely
# nonconvex (bilinear shadow-price DAE mu*c == 1 and k**a production), so
# CVXPY hands the smooth nonlinear program to UNO.
NLP_OPTIONS = dict(preset="filtersqp")


def neoclassical_growth_matern(
    a: float = 1 / 3,
    delta: float = 0.1,
    rho_hat: float = 0.11,
    k_0: float = 1.0,  # k_0 is the state variable initial conditions here, i.e., x_0
    nu: float = 0.5,
    sigma: float = 1.0,
    rho: float = 10,
    train_T: float = 40.0,
    train_points: int = 41,
    test_T: float = 50,
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

    # Construct kernel matrices. CVXPY needs numpy arrays at the solver boundary.
    N = len(train_data)
    K, K_tilde = integrated_matern_kernel_matrices(
        train_data, train_data, nu, sigma, rho
    )
    K = np.asarray((K + K.T) / 2)  # symmetrize -> exactly PSD for quad_form
    K_tilde = np.asarray(K_tilde)

    # Decision variables: zero kernel coefficients and the flat k_0 flow guess
    # k_0**a - delta*k_0 for the scalar initial conditions c_0, mu_0 >= 0.
    alpha_mu, alpha_c, alpha_k = cp.Variable(N), cp.Variable(N), cp.Variable(N)
    c_0 = cp.Variable(nonneg=True)
    mu_0 = cp.Variable(nonneg=True)
    alpha_mu.value = np.zeros(N)
    alpha_c.value = np.zeros(N)
    alpha_k.value = np.zeros(N)
    c_0.value = mu_0.value = k_0**a - delta * k_0

    # Affine kernel expansions. k is the state x; mu the costate; c the control.
    mu = mu_0 + K_tilde @ alpha_mu
    c = c_0 + K_tilde @ alpha_c
    k = k_0 + K_tilde @ alpha_k
    dmu_dt = K @ alpha_mu
    dk_dt = K @ alpha_k

    rkhs_objective = (
        cp.quad_form(alpha_k, cp.psd_wrap(K))
        + cp.quad_form(alpha_c, cp.psd_wrap(K))
        + cp.quad_form(alpha_mu, cp.psd_wrap(K))
    )
    prob = cp.Problem(
        cp.Minimize(rkhs_objective),
        [
            k >= 1e-6, c >= 1e-6, mu >= 1e-6,  # keep fractional powers in domain
            dk_dt == cp.power(k, a) - delta * k - c,  # resource
            dmu_dt
            == -cp.multiply(mu, a * cp.power(k, a - 1) - delta - rho_hat),  # Euler
            cp.multiply(mu, c) == 1.0,  # shadow price (DAE)
        ],
    )
    assert prob.is_dnlp()
    options = dict(NLP_OPTIONS)
    if not verbose:
        options["logger"] = "SILENT"  # mute UNO's C-level iteration table
    start = time.perf_counter()
    prob.solve(nlp=True, solver=cp.UNO, verbose=verbose, **options)
    elapsed = time.perf_counter() - start
    print(f"elapsed solve(s) = {elapsed}")
    if prob.status not in ("optimal", "optimal_inaccurate"):
        print(f"solver status: {prob.status}")

    alpha_c = jnp.array(alpha_c.value)
    alpha_k = jnp.array(alpha_k.value)
    alpha_mu = jnp.array(alpha_mu.value)
    c_0 = float(c_0.value)
    mu_0 = float(mu_0.value)
    rkhs_norms = {
        "k": rkhs_norm_squared(alpha_k, K),
        "c": rkhs_norm_squared(alpha_c, K),
        "mu": rkhs_norm_squared(alpha_mu, K),
    }

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
    # Fall back to the wall-clock timer if solver_stats.solve_time is unavailable.
    solve_time = prob.solver_stats.solve_time
    if solve_time is None:
        solve_time = elapsed
    print(
        f"solve_time(s) = {solve_time}, E(|rel_error(k)|) = {k_rel_error.mean()}, E(|rel_error(c)|) = {c_rel_error.mean()}"
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
        "alpha_mu": alpha_mu,
        "c_0": c_0,
        "mu_0": mu_0,
        "rkhs_norms": rkhs_norms,
        "solve_time": solve_time,
        "kernel_solution": kernel_solution,  # interpolator
        "benchmark_solution": sol_benchmark,  # interpolator
    }


if __name__ == "__main__":
    jsonargparse.CLI(neoclassical_growth_matern)
