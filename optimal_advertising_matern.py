import time
import jax
import jax.numpy as jnp
import numpy as np
import cvxpy as cp
import jsonargparse
from jax import config
from kernels import integrated_matern_kernel_matrices
from rkhs import rkhs_norm_squared
from typing import List, Optional

config.update("jax_enable_x64", True)

# CVXPY DNLP implementation. The advertising-capital FOC collocation is
# nonconvex (the bilinear (1-x)*u and mu*u terms, and the
# u**((1-kappa)/kappa) marginal-cost relation), so CVXPY hands the smooth
# nonlinear program to UNO.
NLP_OPTIONS = dict(preset="filtersqp")


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

    # Decision variables, initialized with zero kernel coefficients and
    # mu_0 = u_0 = 0.
    alpha_x, alpha_mu, alpha_u = cp.Variable(N), cp.Variable(N), cp.Variable(N)
    mu_0 = cp.Variable(nonneg=True)
    u_0 = cp.Variable(nonneg=True)
    alpha_x.value = np.zeros(N)
    alpha_mu.value = np.zeros(N)
    alpha_u.value = np.zeros(N)
    mu_0.value = u_0.value = 0.0

    # Affine kernel expansions. x is the market-share state, mu the costate,
    # u the advertising control.
    mu = mu_0 + K_tilde @ alpha_mu
    x = x_0 + K_tilde @ alpha_x
    u = u_0 + K_tilde @ alpha_u
    dmu_dt = K @ alpha_mu
    dx_dt = K @ alpha_x

    gamma = (beta + rho_hat) / c
    prob = cp.Problem(
        cp.Minimize(
            cp.quad_form(alpha_x, cp.psd_wrap(K))
            + cp.quad_form(alpha_mu, cp.psd_wrap(K))
            + cp.quad_form(alpha_u, cp.psd_wrap(K))
        ),
        [
            u >= 1e-8,
            dx_dt == cp.multiply(1 - x, u) - beta * x,  # market-share dynamics
            dmu_dt == -gamma + (rho_hat + beta) * mu + cp.multiply(mu, u),  # costate
            cp.power(u, (1.0 - kappa) / kappa) - kappa * cp.multiply(mu, 1 - x)
            == 0.0,  # shadow price (marginal cost of advertising)
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

    alpha_mu = jnp.array(alpha_mu.value)
    alpha_x = jnp.array(alpha_x.value)
    alpha_u = jnp.array(alpha_u.value)
    u_0 = float(u_0.value)
    mu_0 = float(mu_0.value)
    rkhs_norms = {
        "x": rkhs_norm_squared(alpha_x, K),
        "mu": rkhs_norm_squared(alpha_mu, K),
        "u": rkhs_norm_squared(alpha_u, K),
    }

    # Interpolator using training data
    @jax.jit
    def kernel_solution(test_data):
        # pointwise comparison test_data to train_data
        K_test, K_tilde_test = integrated_matern_kernel_matrices(
            test_data, train_data, nu, sigma, rho
        )
        mu_test = mu_0 + K_tilde_test @ alpha_mu
        x_test = x_0 + K_tilde_test @ alpha_x
        u_test = u_0 + K_tilde_test @ alpha_u
        return x_test, mu_test, u_test

    # Generate test_data and compare to the benchmark
    x_test, mu_test, u_test = kernel_solution(test_data)

    solve_time = prob.solver_stats.solve_time
    if solve_time is None:
        solve_time = elapsed
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
        "mu_0": mu_0,
        "u_0": u_0,
        "rkhs_norms": rkhs_norms,
        "solve_time": solve_time,
        "kernel_solution": kernel_solution,  # interpolator
    }


if __name__ == "__main__":
    jsonargparse.CLI(optimal_advertising_matern)
