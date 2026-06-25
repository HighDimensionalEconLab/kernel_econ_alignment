import time
from typing import List, Optional

import cvxpy as cp
import jax
import jax.numpy as jnp
import jsonargparse
import numpy as np
from jax import config

from kernels import integrated_matern_kernel_matrices
from rkhs import rkhs_norm_squared

config.update("jax_enable_x64", True)

# CVXPY DNLP implementation. The production function is
# A*max(k**a, b_1*k**a - b_2).  We enumerate the two smooth active-branch
# candidates and accept only candidates that validate against the original max
# problem. This uses no steady-state information.
NLP_OPTIONS = dict(preset="ipopt", max_iterations=2000, time_limit=2.5)
ACCEPTED_STATUSES = {"optimal", "optimal_inaccurate"}
TRAIN_HELPER_TOL = 1e-5
VALIDATION_HELPER_TOL = 5e-3
MPK_BOUND_TOL = 1e-3
DOMAIN_EPS = 1e-4


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
    _ = benchmark_T
    if train_points_list is None:
        train_data = jnp.linspace(0, train_T, train_points)
    else:
        train_data = jnp.array(train_points_list)
    test_data = jnp.linspace(0, test_T, test_points)
    validation_points = max(test_points, benchmark_points)
    validation_data = jnp.linspace(0, train_T, validation_points)

    N = len(train_data)
    K, K_tilde = integrated_matern_kernel_matrices(
        train_data, train_data, nu, sigma, rho
    )
    K = np.asarray((K + K.T) / 2)
    K_tilde = np.asarray(K_tilde)

    output_0 = A * max(k_0**a, b_1 * k_0**a - b_2)
    c_0_guess = max(output_0 - delta * k_0, DOMAIN_EPS)
    mu_0_guess = 1.0 / c_0_guess
    branches = (("low", 1.0, 0.0), ("high", b_1, b_2))

    def solve_candidate(branch_name, q, b):
        alpha_mu, alpha_k = cp.Variable(N), cp.Variable(N)
        mu_0 = cp.Variable(nonneg=True)
        c = cp.Variable(N)

        alpha_mu.value = np.zeros(N)
        alpha_k.value = np.zeros(N)
        mu_0.value = mu_0_guess
        c.value = np.full(N, c_0_guess)

        k = k_0 + K_tilde @ alpha_k
        mu = mu_0 + K_tilde @ alpha_mu
        dk_dt = K @ alpha_k
        dmu_dt = K @ alpha_mu
        branch_output = A * (q * cp.power(k, a) - b)
        branch_mpk = A * q * a * cp.power(k, a - 1.0)
        objective = cp.quad_form(alpha_mu, cp.psd_wrap(K)) + cp.quad_form(
            alpha_k, cp.psd_wrap(K)
        )
        prob = cp.Problem(
            cp.Minimize(objective),
            [
                k >= DOMAIN_EPS,
                mu >= DOMAIN_EPS,
                c >= DOMAIN_EPS,
                dk_dt == branch_output - delta * k - c,
                cp.multiply(c, mu) == 1.0,
                dmu_dt == -cp.multiply(mu, branch_mpk - delta - rho_hat),
            ],
        )
        assert prob.is_dnlp()

        options = dict(NLP_OPTIONS)
        if not verbose:
            options["logger"] = "SILENT"
        start = time.perf_counter()
        solver_error = None
        try:
            prob.solve(nlp=True, solver=cp.UNO, verbose=verbose, **options)
        except Exception as exc:
            solver_error = str(exc)
        wall_time = time.perf_counter() - start
        solve_time = wall_time
        if prob.solver_stats is not None and prob.solver_stats.solve_time is not None:
            solve_time = prob.solver_stats.solve_time

        if solver_error is not None or alpha_mu.value is None or alpha_k.value is None:
            return {
                "branch": branch_name,
                "valid_solution": False,
                "rejection_reason": "solver_error",
                "solver_status": prob.status,
                "solver_error": solver_error,
                "solve_time": solve_time,
                "wall_time": wall_time,
                "max_helper_residual": float("inf"),
                "max_train_residual": float("inf"),
                "max_validation_residual": float("inf"),
                "p_lower_violation": float("inf"),
                "p_upper_violation": float("inf"),
                "objective": float("inf"),
            }

        alpha_mu_value = jnp.array(alpha_mu.value)
        alpha_k_value = jnp.array(alpha_k.value)
        mu_0_value = float(mu_0.value)
        def evaluate_grid(points_data, c_values=None):
            K_eval, K_tilde_eval = integrated_matern_kernel_matrices(
                points_data, train_data, nu, sigma, rho
            )
            k_values = k_0 + K_tilde_eval @ alpha_k_value
            mu_values = mu_0_value + K_tilde_eval @ alpha_mu_value
            dk_values = K_eval @ alpha_k_value
            dmu_values = K_eval @ alpha_mu_value
            if c_values is None:
                c_values = 1.0 / mu_values
            z_values = k_values**a
            branch_low_values = A * z_values
            branch_high_values = A * (b_1 * z_values - b_2)
            output_values = jnp.maximum(branch_low_values, branch_high_values)
            m1_values = A * a * k_values ** (a - 1.0)
            m2_values = b_1 * m1_values
            if branch_name == "low":
                active_output = branch_low_values
                active_mpk = m1_values
            else:
                active_output = branch_high_values
                active_mpk = m2_values
            implied_mpk = delta + rho_hat - dmu_values / mu_values
            return {
                "k": k_values,
                "mu": mu_values,
                "c": c_values,
                "z": z_values,
                "Y": active_output,
                "P": implied_mpk,
                "branch_gap": branch_low_values - branch_high_values,
                "resource": dk_values - (output_values - delta * k_values - c_values),
                "euler": dmu_values + mu_values * (active_mpk - delta - rho_hat),
                "output_binding": active_output - output_values,
                "mpk_lower_violation": jnp.maximum(m1_values - implied_mpk, 0.0),
                "mpk_upper_violation": jnp.maximum(implied_mpk - m2_values, 0.0),
            }

        train_eval = evaluate_grid(train_data, jnp.array(c.value))
        validation_eval = evaluate_grid(validation_data)
        test_eval = evaluate_grid(test_data)

        train_residuals = {
            "shadow_price": train_eval["c"] * train_eval["mu"] - 1.0,
            "resource": train_eval["resource"],
            "euler": train_eval["euler"],
            "output_binding": train_eval["output_binding"],
        }
        validation_residuals = {
            "resource_validation": validation_eval["resource"],
            "euler_validation": validation_eval["euler"],
            "output_binding_validation": validation_eval["output_binding"],
        }
        helper_residuals = {**train_residuals, **validation_residuals}
        max_train_residual = max(
            float(jnp.max(jnp.abs(residual))) for residual in train_residuals.values()
        )
        max_validation_residual = max(
            float(jnp.max(jnp.abs(residual)))
            for residual in validation_residuals.values()
        )
        max_helper_residual = max(
            max_train_residual,
            max_validation_residual,
        )
        p_lower_violation = max(
            float(jnp.max(train_eval["mpk_lower_violation"])),
            float(jnp.max(validation_eval["mpk_lower_violation"])),
        )
        p_upper_violation = max(
            float(jnp.max(train_eval["mpk_upper_violation"])),
            float(jnp.max(validation_eval["mpk_upper_violation"])),
        )

        branch_gap_train = train_eval["branch_gap"]
        active_branch_switch_train = bool(
            jnp.min(branch_gap_train) < 0.0 and jnp.max(branch_gap_train) > 0.0
        )
        min_branch_gap_train = float(jnp.min(jnp.abs(branch_gap_train)))

        @jax.jit
        def kernel_solution(test_points_data):
            _, K_tilde_eval = integrated_matern_kernel_matrices(
                test_points_data, train_data, nu, sigma, rho
            )
            mu_values = mu_0_value + K_tilde_eval @ alpha_mu_value
            k_values = k_0 + K_tilde_eval @ alpha_k_value
            c_values = 1.0 / mu_values
            return k_values, c_values

        finite_positive = bool(
            jnp.all(jnp.isfinite(train_eval["k"]))
            and jnp.all(jnp.isfinite(train_eval["c"]))
            and jnp.all(jnp.isfinite(train_eval["mu"]))
            and jnp.all(jnp.isfinite(validation_eval["k"]))
            and jnp.all(jnp.isfinite(validation_eval["c"]))
            and jnp.all(jnp.isfinite(validation_eval["mu"]))
            and jnp.all(jnp.isfinite(test_eval["k"]))
            and jnp.all(jnp.isfinite(test_eval["c"]))
            and jnp.min(train_eval["k"]) > 0.0
            and jnp.min(train_eval["c"]) > 0.0
            and jnp.min(train_eval["mu"]) > 0.0
            and jnp.min(validation_eval["k"]) > 0.0
            and jnp.min(validation_eval["c"]) > 0.0
            and jnp.min(validation_eval["mu"]) > 0.0
            and jnp.min(test_eval["k"]) > 0.0
            and jnp.min(test_eval["c"]) > 0.0
        )

        rejection_reasons = []
        if prob.status not in ACCEPTED_STATUSES:
            rejection_reasons.append("solver_status")
        if wall_time > options["time_limit"]:
            rejection_reasons.append("time_limit")
        if not finite_positive:
            rejection_reasons.append("nonfinite_or_nonpositive")
        if (
            max_train_residual > TRAIN_HELPER_TOL
            or max_validation_residual > VALIDATION_HELPER_TOL
        ):
            rejection_reasons.append("helper_residual")
        if p_lower_violation > MPK_BOUND_TOL or p_upper_violation > MPK_BOUND_TOL:
            rejection_reasons.append("p_bound")

        rkhs_norms = {
            "k": rkhs_norm_squared(alpha_k_value, K),
            "mu": rkhs_norm_squared(alpha_mu_value, K),
        }
        valid_solution = not rejection_reasons
        return {
            "branch": branch_name,
            "valid_solution": valid_solution,
            "rejection_reason": "accepted"
            if valid_solution
            else ",".join(rejection_reasons),
            "solver_status": prob.status,
            "solver_error": solver_error,
            "solve_time": solve_time,
            "wall_time": wall_time,
            "max_helper_residual": max_helper_residual,
            "max_train_residual": max_train_residual,
            "max_validation_residual": max_validation_residual,
            "p_lower_violation": p_lower_violation,
            "p_upper_violation": p_upper_violation,
            "objective": float(rkhs_norms["k"] + rkhs_norms["mu"]),
            "alpha_mu": alpha_mu_value,
            "alpha_k": alpha_k_value,
            "mu_0": mu_0_value,
            "rkhs_norms": rkhs_norms,
            "helper_residuals": helper_residuals,
            "k_train": train_eval["k"],
            "mu_train": train_eval["mu"],
            "c_train": train_eval["c"],
            "z_train": train_eval["z"],
            "Y_train": train_eval["Y"],
            "P_train": train_eval["P"],
            "k_test": test_eval["k"],
            "c_test": test_eval["c"],
            "active_branch_switch_train": active_branch_switch_train,
            "min_branch_gap_train": min_branch_gap_train,
            "kernel_solution": kernel_solution,
        }

    candidates = [solve_candidate(*branch) for branch in branches]
    valid_candidates = [candidate for candidate in candidates if candidate["valid_solution"]]
    solve_time = sum(candidate["solve_time"] for candidate in candidates)
    print(f"solve_time(s) = {solve_time}")

    if valid_candidates:
        selected = min(valid_candidates, key=lambda candidate: candidate["objective"])
    else:
        flat = jnp.full_like(test_data, k_0)
        c_flat = jnp.full_like(test_data, c_0_guess)
        train_flat = jnp.full_like(train_data, k_0)
        c_train_flat = jnp.full_like(train_data, c_0_guess)
        empty_residuals = {
            name: jnp.full_like(train_data, jnp.inf)
            for name in [
                "shadow_price",
                "resource",
                "euler",
                "output_binding",
                "resource_validation",
                "euler_validation",
                "output_binding_validation",
            ]
        }
        def fallback_kernel_solution(points_data):
            return (
                jnp.full_like(points_data, k_0),
                jnp.full_like(points_data, c_0_guess),
            )

        return {
            "t_train": train_data,
            "t_test": test_data,
            "k_test": flat,
            "c_test": c_flat,
            "alpha_m": jnp.zeros_like(train_data),
            "alpha_mu": jnp.zeros_like(train_data),
            "alpha_k": jnp.zeros_like(train_data),
            "mu_0": mu_0_guess,
            "rkhs_norms": {"k": 0.0, "mu": 0.0},
            "helper_residuals": empty_residuals,
            "k_train": train_flat,
            "mu_train": jnp.full_like(train_data, mu_0_guess),
            "c_train": c_train_flat,
            "z_train": train_flat**a,
            "Y_train": jnp.maximum(A * train_flat**a, A * (b_1 * train_flat**a - b_2)),
            "P_train": jnp.full_like(train_data, jnp.nan),
            "solve_time": solve_time,
            "solver_status": "rejected",
            "valid_solution": False,
            "rejection_reason": "no_valid_branch",
            "candidate_branch": "none",
            "candidate_diagnostics": [
                {
                    "branch": candidate["branch"],
                    "valid_solution": candidate["valid_solution"],
                    "rejection_reason": candidate["rejection_reason"],
                    "solver_status": candidate["solver_status"],
                    "solve_time": candidate["solve_time"],
                    "wall_time": candidate["wall_time"],
                    "max_helper_residual": candidate["max_helper_residual"],
                    "max_train_residual": candidate["max_train_residual"],
                    "max_validation_residual": candidate["max_validation_residual"],
                    "p_lower_violation": candidate["p_lower_violation"],
                    "p_upper_violation": candidate["p_upper_violation"],
                    "objective": candidate["objective"],
                }
                for candidate in candidates
            ],
            "max_helper_residual": float("inf"),
            "max_train_residual": float("inf"),
            "max_validation_residual": float("inf"),
            "p_lower_violation": float("inf"),
            "p_upper_violation": float("inf"),
            "active_branch_switch_train": True,
            "min_branch_gap_train": 0.0,
            "kernel_solution": fallback_kernel_solution,
        }

    return {
        "t_train": train_data,
        "t_test": test_data,
        "k_test": selected["k_test"],
        "c_test": selected["c_test"],
        "alpha_m": selected["alpha_mu"],
        "alpha_mu": selected["alpha_mu"],
        "alpha_k": selected["alpha_k"],
        "mu_0": selected["mu_0"],
        "rkhs_norms": selected["rkhs_norms"],
        "helper_residuals": selected["helper_residuals"],
        "k_train": selected["k_train"],
        "mu_train": selected["mu_train"],
        "c_train": selected["c_train"],
        "z_train": selected["z_train"],
        "Y_train": selected["Y_train"],
        "P_train": selected["P_train"],
        "solve_time": solve_time,
        "solver_status": selected["solver_status"],
        "valid_solution": selected["valid_solution"],
        "rejection_reason": selected["rejection_reason"],
        "candidate_branch": selected["branch"],
        "candidate_diagnostics": [
            {
                "branch": candidate["branch"],
                "valid_solution": candidate["valid_solution"],
                "rejection_reason": candidate["rejection_reason"],
                "solver_status": candidate["solver_status"],
                "solve_time": candidate["solve_time"],
                "wall_time": candidate["wall_time"],
                "max_helper_residual": candidate["max_helper_residual"],
                "max_train_residual": candidate["max_train_residual"],
                "max_validation_residual": candidate["max_validation_residual"],
                "p_lower_violation": candidate["p_lower_violation"],
                "p_upper_violation": candidate["p_upper_violation"],
                "objective": candidate["objective"],
            }
            for candidate in candidates
        ],
        "max_helper_residual": selected["max_helper_residual"],
        "max_train_residual": selected["max_train_residual"],
        "max_validation_residual": selected["max_validation_residual"],
        "p_lower_violation": selected["p_lower_violation"],
        "p_upper_violation": selected["p_upper_violation"],
        "active_branch_switch_train": selected["active_branch_switch_train"],
        "min_branch_gap_train": selected["min_branch_gap_train"],
        "kernel_solution": selected["kernel_solution"],
    }


if __name__ == "__main__":
    jsonargparse.CLI(neoclassical_growth_concave_convex_matern)
