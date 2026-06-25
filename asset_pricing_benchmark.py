import jax.numpy as jnp


def mu_f_array(t, c, g, r, x_0):
    a = x_0 + c / g
    horizon = 2000.0
    exp_gt = jnp.exp(g * t)
    finite_horizon = jnp.exp(r * t) * (
        a * jnp.exp((g - r) * horizon) / (g - r)
        + (c / (g * r)) * jnp.exp(-r * horizon)
    )
    return finite_horizon - a * exp_gt / (g - r) - c / (g * r)
