import jax
import jax.numpy as jnp
from quadax import quadgk
from functools import partial

# Defininig the kernels
#------------------------------------------------------------------
# jit compatible for kernels
def matern_kernel_0p5(t_i, t_j, sigma, rho):
    d = jnp.abs(t_i - t_j)
    return sigma**2 * jnp.exp(-d / rho)


def matern_kernel_1p5(t_i, t_j, sigma, rho):
    d = jnp.abs(t_i - t_j)
    exponent = jnp.sqrt(3) * d / rho
    return sigma**2 * (1 + exponent) * jnp.exp(-exponent)


def matern_kernel_2p5(t_i, t_j, sigma, rho):
    d = jnp.abs(t_i - t_j)
    exponent = jnp.sqrt(5) * d / rho
    term = (5 * d**2) / (3 * rho**2)
    return sigma**2 * (1 + exponent + term) * jnp.exp(-exponent)

def matern_kernel_inf(t_i,t_j, sigma, rho): # limit of matern kernel as $\nu$ -> \infty 
    d = jnp.abs(t_i - t_j)
    exponent = - (1/2)*(d**2/rho**2) 
    return (sigma**2)*jnp.exp(exponent)


# Defining the integrals of the kernels
#---------------------------------------------------------------------------
def integrated_matern_kernel_0p5(t_i, t_j, sigma, rho):
    s = t_i - t_j
    d = jnp.abs(s)

    def if_s_neg():
        return rho * (sigma**2) * (jnp.exp(-d / rho) - jnp.exp(-t_j / rho))

    def if_s_non_neg():
        return rho * (sigma**2) * (2 - jnp.exp(-d / rho) - jnp.exp(-t_j / rho))

    return jax.lax.cond(s < 0, if_s_neg, if_s_non_neg)


def integrated_matern_kernel_1p5(t_i, t_j, sigma, rho, quad_tol=1e-7):
    def matern_integrand(t):
        return matern_kernel_1p5(t, t_j, sigma, rho)

    integral, info = quadgk(
        matern_integrand, [0, t_i], epsabs=quad_tol, epsrel=quad_tol
    )
    return jnp.where(t_i == 0, 0, integral)


def integrated_matern_kernel_2p5(t_i, t_j, sigma, rho, quad_tol=1e-7):
    def matern_integrand(t):
        return matern_kernel_2p5(t, t_j, sigma, rho)

    integral, info = quadgk(
        matern_integrand, [0, t_i], epsabs=quad_tol, epsrel=quad_tol
    )
    return jnp.where(t_i == 0, 0, integral)

def integrated_matern_kernel_inf(t_i, t_j, sigma, rho, quad_tol=1e-7):
    def matern_integrand(t):
        return matern_kernel_inf(t, t_j, sigma, rho)

    integral, info = quadgk(
        matern_integrand, [0, t_i], epsabs=quad_tol, epsrel=quad_tol
    )
    return jnp.where(t_i == 0, 0, integral)

#--------------------------------------------------------------------------------

# Cannot jit the 'if' unless a static argument.  jax.lax.switch tricky to use
@partial(jax.jit, static_argnums=(2,))
def integrated_matern_kernel_matrices(t_i, t_j, nu, sigma, rho):
    if nu == 0.5:
        K = jax.vmap(
            jax.vmap(matern_kernel_0p5, in_axes=(None, 0, None, None)),
            in_axes=(0, None, None, None),
        )(t_i, t_j, sigma, rho)
        K_tilde = jax.vmap(
            jax.vmap(integrated_matern_kernel_0p5, in_axes=(None, 0, None, None)),
            in_axes=(0, None, None, None),
        )(t_i, t_j, sigma, rho)
        return K, K_tilde
    elif nu == 1.5:
        K = jax.vmap(
            jax.vmap(matern_kernel_1p5, in_axes=(None, 0, None, None)),
            in_axes=(0, None, None, None),
        )(t_i, t_j, sigma, rho)
        K_tilde = jax.vmap(
            jax.vmap(integrated_matern_kernel_1p5, in_axes=(None, 0, None, None)),
            in_axes=(0, None, None, None),
        )(t_i, t_j, sigma, rho)
        return K, K_tilde
    elif nu == 2.5:
        K = jax.vmap(
            jax.vmap(matern_kernel_2p5, in_axes=(None, 0, None, None)),
            in_axes=(0, None, None, None),
        )(t_i, t_j, sigma, rho)
        K_tilde = jax.vmap(
            jax.vmap(integrated_matern_kernel_2p5, in_axes=(None, 0, None, None)),
            in_axes=(0, None, None, None),
        )(t_i, t_j, sigma, rho)
        return K, K_tilde
    elif nu == "inf":
        K = jax.vmap(
            jax.vmap(matern_kernel_inf, in_axes=(None, 0, None, None)),
            in_axes=(0, None, None, None),
        )(t_i, t_j, sigma, rho)
        K_tilde = jax.vmap(
            jax.vmap(integrated_matern_kernel_inf, in_axes=(None, 0, None, None)),
            in_axes=(0, None, None, None),
        )(t_i, t_j, sigma, rho)
        return K, K_tilde
    else:
        print("nu not supported")
        return jnp.full((t_i.shape[0], t_j.shape[0]), jnp.nan), jnp.full(
            (t_i.shape[0], t_j.shape[0]), jnp.nan
        )


# There is a discontinuity in the derivative which we need a special case for.
def make_safe_at_equality(func):
    def safe_func(t_i, t_j, *args, **kwargs):
        # Define the safe derivative function with a condition
        return jax.lax.cond(
            t_i == t_j,
            lambda _: 0.0,  # Return 0 if inputs are equal
            lambda _: func(
                t_i, t_j, *args, **kwargs
            ),  # Otherwise, compute the gradient
            operand=None,
        )

    return safe_func

# The final output
@partial(jax.jit, static_argnums=(2,))
def differentiated_matern_kernel_matrices(t_i, t_j, nu, sigma, rho):
    if nu == 0.5:
        K = jax.vmap(
            jax.vmap(matern_kernel_0p5, in_axes=(None, 0, None, None)),
            in_axes=(0, None, None, None),
        )(t_i, t_j, sigma, rho)
        diff_kernel = make_safe_at_equality(jax.grad(matern_kernel_0p5, argnums=0))
        K_dot = jax.vmap(
            jax.vmap(diff_kernel, in_axes=(None, 0, None, None)),
            in_axes=(0, None, None, None),
        )(t_i, t_j, sigma, rho)
        return K, K_dot
    elif nu == 1.5:
        K = jax.vmap(
            jax.vmap(matern_kernel_1p5, in_axes=(None, 0, None, None)),
            in_axes=(0, None, None, None),
        )(t_i, t_j, sigma, rho)
        diff_kernel = make_safe_at_equality(jax.grad(matern_kernel_1p5, argnums=0))
        K_dot = jax.vmap(
            jax.vmap(diff_kernel, in_axes=(None, 0, None, None)),
            in_axes=(0, None, None, None),
        )(t_i, t_j, sigma, rho)
        return K, K_dot
    elif nu == 2.5:
        K = jax.vmap(
            jax.vmap(matern_kernel_2p5, in_axes=(None, 0, None, None)),
            in_axes=(0, None, None, None),
        )(t_i, t_j, sigma, rho)
        diff_kernel = make_safe_at_equality(jax.grad(matern_kernel_2p5, argnums=0))
        K_dot = jax.vmap(
            jax.vmap(diff_kernel, in_axes=(None, 0, None, None)),
            in_axes=(0, None, None, None),
        )(t_i, t_j, sigma, rho)
        return K, K_dot
    else:
        print("nu not supported")
        return jnp.full((t_i.shape[0], t_j.shape[0]), jnp.nan), jnp.full(
            (t_i.shape[0], t_j.shape[0]), jnp.nan
        )
