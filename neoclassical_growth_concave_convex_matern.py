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

# CVXPY DNLP implementation. With z = k**a the concave-convex envelope
# A*max(k**a, b_1*k**a - b_2) is convex-PWL and bounds output Y from below;
# two complementarity equalities pin the marginal product P to the active branch
# and force Y to bind -- an exact reformulation, no smoothing.
NLP_OPTIONS = dict(preset="filtersqp")


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
    train_T: float = 40.0,
    train_points: int = 41,
    test_T: float = 50,
    test_points: int = 41,
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

    # Consumption is represented explicitly with c*mu = 1. z = k**a linearizes the
    # production branches; Y is output, P the marginal product of capital.  k and
    # mu carry lower bounds so trial points stay in the domain of k**a and 1/mu.
    alpha_mu, alpha_k = cp.Variable(N), cp.Variable(N)
    mu_0 = cp.Variable(nonneg=True)
    c, z, Y, P = cp.Variable(N), cp.Variable(N), cp.Variable(N), cp.Variable(N)
    k, mu = cp.Variable(N), cp.Variable(N)

    # Flat warm start at k_0 (no ramp toward a steady state): consumption held at
    # the capital-stationary level, with z, Y, P on the active production branch
    # (m1 below the kink k_bar, m2 above) so complementarity holds at the start.
    k_bar = (b_2 / (b_1 - 1)) ** (1 / a)
    f_0 = A * max(k_0**a, b_1 * k_0**a - b_2)
    c_0 = f_0 - delta * k_0
    m1_0 = A * a * k_0 ** (a - 1)
    alpha_mu.value = np.zeros(N)
    alpha_k.value = np.zeros(N)
    mu_0.value = 1.0 / c_0
    k.value = np.full(N, k_0)
    mu.value = np.full(N, 1.0 / c_0)
    c.value = np.full(N, c_0)
    z.value = np.full(N, k_0**a)
    Y.value = np.full(N, f_0)
    P.value = np.full(N, b_1 * m1_0 if k_0 >= k_bar else m1_0)

    dmu_dt = K @ alpha_mu
    dk_dt = K @ alpha_k
    m1 = A * a * cp.power(k, a - 1)
    m2 = b_1 * m1
    branch_low = A * z
    branch_high = A * (b_1 * z - b_2)
    prob = cp.Problem(
        cp.Minimize(
            cp.quad_form(alpha_mu, cp.psd_wrap(K))
            + cp.quad_form(alpha_k, cp.psd_wrap(K))
        ),
        [
            k == k_0 + K_tilde @ alpha_k,  # state/costate from the kernel expansion
            mu == mu_0 + K_tilde @ alpha_mu,
            k >= 1e-4, mu >= 1e-4, c >= 1e-4, z >= 1e-6,  # domain bounds
            z == cp.power(k, a),
            cp.maximum(branch_low, branch_high) <= Y,  # production envelope
            cp.multiply(Y - branch_low, P - m2) == 0,  # pin P, bind Y
            cp.multiply(Y - branch_high, P - m1) == 0,
            dk_dt == Y - delta * k - c,  # resource
            cp.multiply(c, mu) == 1.0,  # shadow price
            dmu_dt == -cp.multiply(mu, P - delta - rho_hat),  # Euler (MPK = P)
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
    alpha_k = jnp.array(alpha_k.value)
    mu_0 = float(mu_0.value)
    rkhs_norms = {
        "k": rkhs_norm_squared(alpha_k, K),
        "mu": rkhs_norm_squared(alpha_mu, K),
    }
    k_train = jnp.array(k.value)
    mu_train = jnp.array(mu.value)
    c_train = jnp.array(c.value)
    z_train = jnp.array(z.value)
    Y_train = jnp.array(Y.value)
    P_train = jnp.array(P.value)
    m1_train = A * a * k_train ** (a - 1)
    m2_train = b_1 * m1_train
    branch_low_train = A * z_train
    branch_high_train = A * (b_1 * z_train - b_2)
    helper_residuals = {
        "z_power": z_train - k_train**a,
        "shadow_price": c_train * mu_train - 1.0,
        "output_binding": Y_train - jnp.maximum(branch_low_train, branch_high_train),
        "complementarity_low": (Y_train - branch_low_train) * (P_train - m2_train),
        "complementarity_high": (Y_train - branch_high_train) * (P_train - m1_train),
        "marginal_product_lower_violation": jnp.maximum(m1_train - P_train, 0.0),
        "marginal_product_upper_violation": jnp.maximum(P_train - m2_train, 0.0),
    }

    # Interpolator using training data
    @jax.jit
    def kernel_solution(test_data):
        # pointwise comparison test_data to train_data
        K_test, K_tilde_test = integrated_matern_kernel_matrices(
            test_data, train_data, nu, sigma, rho
        )
        mu_test = mu_0 + K_tilde_test @ alpha_mu
        k_test = k_0 + K_tilde_test @ alpha_k
        c_test = 1.0 / mu_test
        return k_test, c_test

    # Generate test_data and compare to the benchmark
    k_test, c_test = kernel_solution(test_data)

    solve_time = prob.solver_stats.solve_time
    if solve_time is None:
        solve_time = elapsed
    print(f"solve_time(s) = {solve_time}")
    return {
        "t_train": train_data,
        "t_test": test_data,
        "k_test": k_test,
        "c_test": c_test,
        "alpha_m": alpha_mu,
        "alpha_mu": alpha_mu,
        "alpha_k": alpha_k,
        "mu_0": mu_0,
        "rkhs_norms": rkhs_norms,
        "helper_residuals": helper_residuals,
        "k_train": k_train,
        "mu_train": mu_train,
        "c_train": c_train,
        "z_train": z_train,
        "Y_train": Y_train,
        "P_train": P_train,
        "solve_time": solve_time,
        "kernel_solution": kernel_solution,  # interpolator
    }


if __name__ == "__main__":
    jsonargparse.CLI(neoclassical_growth_concave_convex_matern)
