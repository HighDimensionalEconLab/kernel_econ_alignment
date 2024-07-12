import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_bvp


# Neoclassical Growth Benchmark solution
def neoclassical_growth_benchmark(a, delta, r, sigma_crra, k_0, t_grid, perturb_k=1e-4):
    k_ss = ((delta + r) / a) ** (1 / (a - 1))
    c_ss = a * k_ss**a - -delta * k_ss
    # perturb the final value of the capital at T to help convergence
    k_T = k_ss - perturb_k

    def ODE(t, y):
        k = y[0]
        c = y[1]
        return jnp.vstack(
            (
                k**a - c - delta * k,
                (c / sigma_crra) * (a * k ** (a - 1) - r - delta),
            )
        )

    def bc(ya, yb):
        return jnp.array([ya[0] - k_0, yb[0] - k_T])

    iv = 1 * jnp.ones((2, t_grid.size))
    solution = solve_bvp(ODE, bc, t_grid, iv)

    # the "solution" is an interpolator already, can just unpack
    T_max = t_grid[-1]

    def interpolate_solution(t_grid):
        if t_grid[-1] > T_max:
            raise ValueError("Extrapolation not supported")
        val = solution.sol(t_grid)
        return val[0], val[1]

    return interpolate_solution
