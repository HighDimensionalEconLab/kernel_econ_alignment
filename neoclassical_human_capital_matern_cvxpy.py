import time
from typing import List, Optional

import jax.numpy as jnp
import jsonargparse
import numpy as np
import cvxpy as cp
from jax import config
from scipy.optimize import fsolve

from kernels import integrated_matern_kernel_matrices

config.update("jax_enable_x64", True)

# Pyomo-free DNLP port of neoclassical_human_capital_matern. The two-capital
# (physical + human) FOC collocation is nonconvex (Cobb-Douglas f = k**a_k h**a_h,
# the bilinear costate/feasibility/shadow-price relations), so it is solved
# through cvxpy's DNLP interface (prob.solve(nlp=True, ...)).  solver_type defaults
# to UNO (unopy, in-process, filtersqp preset); IPOPT (cyipopt) stays available as
# an alternative, with options mirroring the pyomo tolerances.
NLP_SOLVERS = {
    "UNO": (cp.UNO, dict(preset="filtersqp")),
    "IPOPT": (
        cp.IPOPT,
        dict(
            tol=1e-8,
            dual_inf_tol=1e-8,
            constr_viol_tol=1e-8,
            acceptable_tol=1e-6,
            mu_strategy="adaptive",
            max_iter=4000,
        ),
    ),
}


def human_capital_matern_cvxpy(
    a_k: float = 1 / 3,
    a_h: float = 1 / 4,
    delta_k: float = 0.1,
    delta_h: float = 0.05,
    rho_hat: float = 0.11,  # discount rate
    k_0: float = 1.5,
    nu: float = 0.5,
    sigma: float = 1.0,
    rho: float = 10,
    solver_type: str = "UNO",
    train_T: float = 80.0,
    train_points: int = 61,
    test_T: float = 100.0,
    test_points: int = 100,
    benchmark_T: float = 60.0,
    benchmark_points: int = 300,
    train_points_list: Optional[List[float]] = None,
    lambda_p: float = 5e-3,  # small smoothing penalty to stabilize the solve
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

    # Solve the no-arbitrage condition for the initial human capital h_0, given
    # k_0, so the two marginal products net of depreciation coincide at t=0.
    h_0 = fsolve(
        lambda h: (k_0**a_k) * (a_h * h ** (a_h - 1))
        - (a_k * k_0 ** (a_k - 1)) * (h**a_h)
        - delta_h
        + delta_k,
        [k_0],
    )[0]
    c_0_init = (k_0**a_k) * (h_0**a_h) - delta_h * h_0 - delta_k * k_0

    # Decision variables (7 coefficient vectors, 5 scalars), initialized at the
    # pyomo starting point so the local NLP solve begins identically.
    alpha_k, alpha_h = cp.Variable(N), cp.Variable(N)
    alpha_mu_k, alpha_mu_h = cp.Variable(N), cp.Variable(N)
    alpha_i_k, alpha_i_h, alpha_c = cp.Variable(N), cp.Variable(N), cp.Variable(N)
    i_k_0 = cp.Variable(nonneg=True)
    i_h_0 = cp.Variable(nonneg=True)
    c_0 = cp.Variable(nonneg=True)
    mu_k_0 = cp.Variable(nonneg=True)
    mu_h_0 = cp.Variable(nonneg=True)
    for v in (alpha_k, alpha_h, alpha_mu_k, alpha_mu_h, alpha_i_k, alpha_i_h, alpha_c):
        v.value = np.zeros(N)
    i_k_0.value = delta_k * k_0
    i_h_0.value = delta_h * h_0
    c_0.value = c_0_init
    mu_k_0.value = mu_h_0.value = 1 / c_0_init

    # Affine kernel expansions (the pyomo helper functions inline).  k/h are the
    # physical/human capital states, i_k/i_h investments, c consumption, mu_* costates.
    k = k_0 + K_tilde @ alpha_k
    h = h_0 + K_tilde @ alpha_h
    i_k = i_k_0 + K_tilde @ alpha_i_k
    i_h = i_h_0 + K_tilde @ alpha_i_h
    c = c_0 + K_tilde @ alpha_c
    mu_k = mu_k_0 + K_tilde @ alpha_mu_k
    mu_h = mu_h_0 + K_tilde @ alpha_mu_h
    dk_dt = K @ alpha_k
    dh_dt = K @ alpha_h
    dmu_k_dt = K @ alpha_mu_k
    dmu_h_dt = K @ alpha_mu_h

    # Cobb-Douglas production and its marginal products as cvxpy expressions.
    f = cp.multiply(cp.power(k, a_k), cp.power(h, a_h))
    f_k = a_k * cp.multiply(cp.power(k, a_k - 1), cp.power(h, a_h))
    f_h = a_h * cp.multiply(cp.power(k, a_k), cp.power(h, a_h - 1))

    # Core RKHS norms on the state/costate coefficients; small smoothing reg on the
    # coefficients that only enter via constraints (investments and consumption).
    core = (
        cp.quad_form(alpha_mu_k, cp.psd_wrap(K))
        + cp.quad_form(alpha_k, cp.psd_wrap(K))
        + cp.quad_form(alpha_mu_h, cp.psd_wrap(K))
        + cp.quad_form(alpha_h, cp.psd_wrap(K))
    )
    reg = (
        cp.quad_form(alpha_i_k, cp.psd_wrap(K))
        + cp.quad_form(alpha_i_h, cp.psd_wrap(K))
        + cp.quad_form(alpha_c, cp.psd_wrap(K))
    )
    prob = cp.Problem(
        cp.Minimize(core + lambda_p * reg),
        [
            dk_dt == i_k - delta_k * k,  # physical capital accumulation
            dh_dt == i_h - delta_h * h,  # human capital accumulation
            dmu_k_dt == -cp.multiply(mu_k, f_k - delta_k - rho_hat),  # physical Euler
            dmu_h_dt == -cp.multiply(mu_h, f_h - delta_h - rho_hat),  # human Euler
            c + i_h + i_k - f == 0.0,  # resource feasibility
            cp.multiply(mu_k, c) == 1.0,  # shadow price
            mu_k - mu_h == 0.0,  # both capitals priced equally
        ],
    )
    assert prob.is_dnlp()
    solver, options = NLP_SOLVERS[solver_type]
    start = time.perf_counter()
    try:
        prob.solve(nlp=True, solver=solver, verbose=verbose, **options)
        assert prob.status in ("optimal", "optimal_inaccurate")
    except Exception:
        # retry once with relaxed tolerances (mirrors the pyomo IPOPT fallback)
        relaxed = {**options}
        if solver_type == "IPOPT":
            relaxed.update(
                tol=1e-6, dual_inf_tol=1e-6, constr_viol_tol=1e-6, acceptable_tol=1e-4
            )
        prob.solve(nlp=True, solver=solver, verbose=verbose, **relaxed)
    elapsed = time.perf_counter() - start
    print(f"elapsed solve(s) = {elapsed}")
    if prob.status not in ("optimal", "optimal_inaccurate"):
        print(f"solver status: {prob.status}")

    alpha_c = jnp.array(alpha_c.value)
    alpha_k = jnp.array(alpha_k.value)
    alpha_h = jnp.array(alpha_h.value)
    alpha_i_k = jnp.array(alpha_i_k.value)
    alpha_i_h = jnp.array(alpha_i_h.value)
    alpha_mu_k = jnp.array(alpha_mu_k.value)
    alpha_mu_h = jnp.array(alpha_mu_h.value)
    c_0 = float(c_0.value)
    i_k_0 = float(i_k_0.value)
    i_h_0 = float(i_h_0.value)
    mu_k_0 = float(mu_k_0.value)
    mu_h_0 = float(mu_h_0.value)

    # Interpolator using training data
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
        mu_k_test = mu_k_0 + K_tilde_test @ alpha_mu_k
        mu_h_test = mu_h_0 + K_tilde_test @ alpha_mu_h
        feasibility_test = (
            c_test + i_h_test + i_k_test - (k_test**a_k) * (h_test**a_h)
        )
        return (
            k_test,
            h_test,
            c_test,
            i_k_test,
            i_h_test,
            mu_k_test,
            mu_h_test,
            feasibility_test,
        )

    # Generate test_data
    (
        k_test,
        h_test,
        c_test,
        i_k_test,
        i_h_test,
        mu_k_test,
        mu_h_test,
        feasibility_test,
    ) = kernel_solution(test_data)

    solve_time = prob.solver_stats.solve_time
    if solve_time is None:
        solve_time = elapsed
    print(f"solve_time(s) = {solve_time}")
    return {
        "t_train": train_data,
        "t_test": test_data,
        "k_test": k_test,
        "h_test": h_test,
        "c_test": c_test,
        "i_k_test": i_k_test,
        "i_h_test": i_h_test,
        "mu_k_test": mu_k_test,
        "mu_h_test": mu_h_test,
        "feasibility_test": feasibility_test,
        "alpha_c": alpha_c,
        "alpha_k": alpha_k,
        "alpha_h": alpha_h,
        "alpha_i_k": alpha_i_k,
        "alpha_i_h": alpha_i_h,
        "c_0": c_0,
        "i_k_0": i_k_0,
        "i_h_0": i_h_0,
        "solve_time": solve_time,
        "kernel_solution": kernel_solution,  # interpolator
    }


if __name__ == "__main__":
    jsonargparse.CLI(human_capital_matern_cvxpy)
