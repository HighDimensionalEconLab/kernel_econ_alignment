import jax.numpy as jnp
from scipy.integrate import quad


def p_f_array(t, c, g, r, y_0):

    def y(s, c, g, y_0):
        return (y_0 + (c / g)) * jnp.exp(g * s) - (c / g)

    def discount_y(s):
        return jnp.exp(-r * s) * y(s, c, g, y_0)

    result = jnp.zeros_like(t)
    for i, t_value in enumerate(t):
        integral, err = quad(discount_y, t_value, 2000)
        result = result.at[i].set(jnp.exp(r * t_value) * integral)
    return result
