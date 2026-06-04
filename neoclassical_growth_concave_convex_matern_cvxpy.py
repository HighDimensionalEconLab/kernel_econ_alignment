import time
import jax
import jax.numpy as jnp
import numpy as np
import cvxpy as cp
import jsonargparse
from jax import config
from kernels import integrated_matern_kernel_matrices
from typing import List, Optional

config.update("jax_enable_x64", True)

# Pyomo-free DNLP port of neoclassical_growth_concave_convex_matern. The
# concave-convex production is the upper envelope A*max(k**a, b_1*k**a - b_2)
# (the two branches cross at k_bar), written in pyomo as Expr_if(k < k_bar, ...).
# cp.maximum is non-smooth, but cvxpy's DNLP rules keep it usable in an L-convex
# position (convex <= affine): substituting z = k**a makes both branches affine
# in z, so max(A*z, A*(b_1*z - b_2)) is convex-PWL and may bound an output
# variable Y from below (cvxpy expands the epigraph into two smooth inequalities).
# Two complementarity equalities then pin the marginal product P to the active
# branch and force Y to bind to the envelope -- an exact reformulation, no
# smoothing.  Where it converges it matches the pyomo solution, but the resulting
# complementarity (MPCC) collocation is fragile for x_0 in a wide band just above
# the threshold k_bar (the high-steady-state approach trajectories): there the KKT
# system is degenerate and neither backend is reliable, so the bistable threshold
# figure is generated with pyomo instead.  This model defaults to IPOPT (cyipopt)
# rather than UNO: on those degenerate solves UNO's filterSQP drops into an
# uninterruptible restoration loop that ignores its iteration budget, whereas
# IPOPT's interior point respects max_iter and fails fast and cleanly.  UNO stays
# selectable (it is a touch more accurate when it does converge).
NLP_SOLVERS = {
    "IPOPT": (
        cp.IPOPT,
        dict(tol=1e-8, dual_inf_tol=1e-8, constr_viol_tol=1e-8, max_iter=1000),
    ),
    "UNO": (cp.UNO, dict(preset="filtersqp")),
}


def neoclassical_growth_concave_convex_matern_cvxpy(
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
    solver_type: str = "IPOPT",
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

    # Construct kernel matrices.  cvxpy needs numpy (not jax) arrays.
    N = len(train_data)
    K, K_tilde = integrated_matern_kernel_matrices(
        train_data, train_data, nu, sigma, rho
    )
    K = np.asarray((K + K.T) / 2)  # symmetrize -> exactly PSD for quad_form
    K_tilde = np.asarray(K_tilde)

    # Decision variables.  Consumption is substituted out via c = 1/mu, so the
    # costate mu and state k are the dynamic unknowns.  z = k**a linearizes the
    # production branches; Y is output (= the envelope at the optimum); P is the
    # marginal product of capital carried into the Euler equation.
    alpha_mu, alpha_k = cp.Variable(N), cp.Variable(N)
    mu_0 = cp.Variable(nonneg=True)
    z, Y, P = cp.Variable(N), cp.Variable(N), cp.Variable(N)

    # Warm start.  The cold flat init k(t) = k_0 sits at a degenerate stationary
    # point, so seed k(t) with a crude straight-line ramp toward the steady state
    # on the correct side of the production threshold k_bar -- all closed-form, no
    # solve needed.  k_bar is where the branches cross; k_star solves f'(k)=delta+
    # rho_hat on the active branch.  The ramp only picks the basin; the solver then
    # converges to the exact branch.  K_tilde is rank-deficient, so pinv gives the
    # least-norm alpha_k reproducing the ramp approximately (close, not exact).
    k_bar = (b_2 / (b_1 - 1)) ** (1 / a)
    if k_0 < k_bar:
        k_star = ((delta + rho_hat) / (A * a)) ** (1 / (a - 1))
        c_star = A * k_star**a - delta * k_star
    else:
        k_star = ((delta + rho_hat) / (A * a * b_1)) ** (1 / (a - 1))
        c_star = A * (b_1 * k_star**a - b_2) - delta * k_star
    k_ramp = k_0 + (k_star - k_0) * (np.asarray(train_data) / float(train_data[-1]))
    alpha_mu.value = np.zeros(N)
    alpha_k.value = np.linalg.pinv(K_tilde) @ (k_ramp - k_0)
    mu_0.value = 1.0 / c_star
    z.value = k_ramp**a
    Y.value = np.maximum(A * z.value, A * (b_1 * z.value - b_2))
    P.value = A * a * k_ramp ** (a - 1)

    # Affine kernel expansions (the pyomo mu/k/dmu_dt/dk_dt helpers inline).
    mu = mu_0 + K_tilde @ alpha_mu
    k = k_0 + K_tilde @ alpha_k
    dmu_dt = K @ alpha_mu
    dk_dt = K @ alpha_k

    # Exact concave-convex production.  With z = k**a the branches are affine, so
    # max(A*z, A*(b_1*z - b_2)) is convex-PWL and bounds output Y from below (an
    # L-convex epigraph cvxpy expands into two smooth inequalities).  m1/m2 are the
    # two branch marginal products; the complementarity equalities pin P to the
    # active branch and force Y to bind (were Y above both branches, the two
    # products could not both vanish for a single P).
    m1 = A * a * cp.power(k, a - 1)
    m2 = b_1 * m1
    prob = cp.Problem(
        cp.Minimize(
            cp.quad_form(alpha_mu, cp.psd_wrap(K)) + cp.quad_form(alpha_k, cp.psd_wrap(K))
        ),
        [
            z == cp.power(k, a),
            cp.maximum(A * z, A * (b_1 * z - b_2)) <= Y,  # production envelope
            cp.multiply(Y - A * z, P - m2) == 0,  # complementarity: pin P, bind Y
            cp.multiply(Y - A * (b_1 * z - b_2), P - m1) == 0,
            dk_dt == Y - delta * k - cp.power(mu, -1),  # resource (c = 1/mu)
            dmu_dt == -cp.multiply(mu, P - delta - rho_hat),  # Euler (MPK = P)
        ],
    )
    assert prob.is_dnlp()
    solver, options = NLP_SOLVERS[solver_type]
    start = time.perf_counter()
    prob.solve(nlp=True, solver=solver, verbose=verbose, **options)
    elapsed = time.perf_counter() - start
    print(f"elapsed solve(s) = {elapsed}")
    if prob.status not in ("optimal", "optimal_inaccurate"):
        print(f"solver status: {prob.status}")

    alpha_mu = jnp.array(alpha_mu.value)
    alpha_k = jnp.array(alpha_k.value)
    mu_0 = float(mu_0.value)

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
        "alpha_k": alpha_k,
        "mu_0": mu_0,
        "solve_time": solve_time,
        "kernel_solution": kernel_solution,  # interpolator
    }


if __name__ == "__main__":
    jsonargparse.CLI(neoclassical_growth_concave_convex_matern_cvxpy)
