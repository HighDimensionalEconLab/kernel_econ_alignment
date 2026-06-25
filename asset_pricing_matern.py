import time
import jax
import jax.numpy as jnp
import numpy as np
import cvxpy as cp
import jsonargparse
from jax import config
from kernels import integrated_matern_kernel_matrices
from asset_pricing_benchmark import mu_f_array
from rkhs import rkhs_norm_squared
from typing import List, Optional

config.update("jax_enable_x64", True)

# CVXPY DCP implementation. The model is a convex QP:
#   minimize  alpha' K alpha
#   s.t.      K alpha == r (mu_0 + K_tilde alpha) - x,   mu_0 >= 0
# solver_type selects an open-source cvxpy backend. OSQP (native QP) is the
# default; CLARABEL is the more robust choice for ill-conditioned kernels
# (large nu/rho/N), where OSQP's ADMM can report infeasibility.
SOLVER_OPTIONS = {
    "OSQP": (cp.OSQP, dict(eps_abs=1e-12, eps_rel=1e-12, max_iter=5000)),
    "CLARABEL": (
        cp.CLARABEL,
        dict(tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12),
    ),
    "SCS": (cp.SCS, dict(eps=1e-9, max_iters=20000)),
    "HIGHS": (cp.HIGHS, dict(primal_feasibility_tolerance=1e-4)),
}


def asset_pricing_matern(
    r: float = 0.1,
    c: float = 0.02,
    g: float = -0.2,
    x_0: float = 0.01,
    nu: float = 0.5,
    sigma: float = 1.0,
    rho: float = 10,
    solver_type: str = "OSQP",
    train_T: float = 40.0,
    train_points: int = 41,
    test_T: float = 50.0,
    test_points: int = 100,
    train_points_list: Optional[List[float]] = None,
    verbose: bool = False,
):
    # if passing in `train_points_list` then doesn't use a grid.  Otherwise, uses linspace
    if train_points_list is None:
        train_data = jnp.linspace(0, train_T, train_points)
    else:
        train_data = jnp.array(train_points_list)
    test_data = jnp.linspace(0, test_T, test_points)

    # Construct kernel matrices. CVXPY needs numpy arrays at the solver boundary.
    N = len(train_data)
    K, K_tilde = integrated_matern_kernel_matrices(
        train_data, train_data, nu, sigma, rho
    )
    K = np.asarray((K + K.T) / 2)  # symmetrize -> exactly PSD for quad_form
    K_tilde = np.asarray(K_tilde)
    x = (x_0 + c / g) * np.exp(g * np.asarray(train_data)) - c / g

    # Solve the QP.  psd_wrap asserts the (provably PSD) Gram matrix K so cvxpy
    # skips its O(N^3) PSD certification, which both dominates canonicalization
    # time and fails to converge for ill-conditioned K.
    alpha_mu = cp.Variable(N)
    mu_0 = cp.Variable(nonneg=True)
    prob = cp.Problem(
        cp.Minimize(cp.quad_form(alpha_mu, cp.psd_wrap(K))),
        [K @ alpha_mu == r * (mu_0 + K_tilde @ alpha_mu) - x],
    )
    solver, options = SOLVER_OPTIONS[solver_type]
    start = time.perf_counter()
    prob.solve(solver=solver, verbose=verbose, **options)
    print(f"elapsed solve(s) = {time.perf_counter() - start}")
    if prob.status not in ("optimal", "optimal_inaccurate"):
        print(f"solver status: {prob.status}")

    alpha_mu = jnp.array(alpha_mu.value)
    mu_0 = float(mu_0.value)
    rkhs_norms = {"p": rkhs_norm_squared(alpha_mu, K)}

    # Interpolator using training data
    @jax.jit
    def kernel_solution(test_data):
        # pointwise comparison test_data to train_data
        _, K_tilde_test = integrated_matern_kernel_matrices(
            test_data, train_data, nu, sigma, rho
        )
        mu_test = mu_0 + K_tilde_test @ alpha_mu
        return mu_test

    # Generate test_data and compare to the benchmark
    mu_benchmark = mu_f_array(test_data, c, g, r, x_0)
    mu_test = kernel_solution(test_data)

    mu_rel_error = jnp.abs(mu_benchmark - mu_test) / mu_benchmark
    print(
        f"solve_time(s) = {prob.solver_stats.solve_time}, E(|rel_error(p)|) = {mu_rel_error.mean()}"
    )
    return {
        "t_train": train_data,
        "t_test": test_data,
        "p_test": mu_test,
        "p_benchmark": mu_benchmark,
        "p_rel_error": mu_rel_error,
        "alpha": alpha_mu,
        "p_0": mu_0,
        "rkhs_norms": rkhs_norms,
        "solve_time": prob.solver_stats.solve_time,
        "kernel_solution": kernel_solution,  # interpolator
    }


if __name__ == "__main__":
    jsonargparse.CLI(asset_pricing_matern)
