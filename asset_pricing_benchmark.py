import jax.numpy as jnp
from scipy.integrate import quad


def mu_f_array(t, c, g, r, x_0):

    def x(s, c, g, x_0):
        return (x_0 + (c / g)) * jnp.exp(g * s) - (c / g)

    def discount_x(s):
        return jnp.exp(-r * s) * x(s, c, g, x_0)

    result = jnp.zeros_like(t)
    for i, t_value in enumerate(t):
        integral, err = quad(discount_x, t_value, 2000)
        result = result.at[i].set(jnp.exp(r * t_value) * integral)
    return result
