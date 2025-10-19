import jax.numpy as jnp
from scipy.integrate import solve_ivp
from scipy.integrate import solve_bvp
import numpy as np
import matplotlib.pyplot as plt
import os
from neoclassical_growth_benchmark import neoclassical_growth_benchmark
from neoclassical_growth_matern import neoclassical_growth_matern

from mpl_toolkits.axes_grid1.inset_locator import (
    zoomed_inset_axes,
    mark_inset,
    inset_axes,
)

fontsize = 17
ticksize = 16
figsize = (15, 10)
params = {
    "font.family": "serif",
    "figure.figsize": figsize,
    "figure.dpi": 80,
    "figure.edgecolor": "k",
    "figure.constrained_layout.use": True,  # Adjust layout to prevent overlap
    "font.size": fontsize,
    "axes.labelsize": fontsize,
    "axes.titlesize": fontsize,
    "xtick.labelsize": ticksize,
    "ytick.labelsize": ticksize,
}
plt.rcParams.update(params)


sol_matern = neoclassical_growth_matern()
output_path = "figures/shooting_vs_kernel_neoclassical_growth_model.pdf"


t = sol_matern["t_test"]
T = sol_matern["t_train"].max()
c_hat_matern = sol_matern["c_test"]
k_hat_matern = sol_matern["k_test"]
c_benchmark = sol_matern["c_benchmark"]
k_benchmark = sol_matern["k_benchmark"]
k_rel_error_matern = sol_matern["k_rel_error"]
c_rel_error_matern = sol_matern["c_rel_error"]

# To get the bechmark right we have to solve the BVP for T=40


# Parameters to solve the ODE
a = neoclassical_growth_matern.__defaults__[0]
delta = neoclassical_growth_matern.__defaults__[1]
r = neoclassical_growth_matern.__defaults__[2]# thats the rho_hat in the code
k_0 = neoclassical_growth_matern.__defaults__[3]

# Defining the ODE system 

#functions to be used
def f(k):
    return k**a

def f_p(k):
    return a*(k**(a-1))
#solving the ODE Iinital value probelm

def neoclassical_ode(t,y):
    k, c = y
    dkdt = f(k)- c - delta*k
    dcdt = c*(f_p(k)-r-delta)
    return [dkdt, dcdt]

# Solving the BVP over the train data to get c_0 of the train data
'''
perturb_k = 1e-4
k_ss = ((delta + r) / a) ** (1 / (a - 1))
c_ss = a * k_ss**a - -delta * k_ss
# perturb the final value of the capital at T to help convergence
k_T = k_ss - perturb_k

def bc(ya, yb):
    return np.array([ya[0] - k_0, yb[0] - k_T]) #boundary condition, k(0) = k_0, k(T) = k_T

t_grid = np.linspace(0, T.item(), 300)

iv = 1 * np.ones((2, t_grid.size))
solution_bvp = solve_bvp(neoclassical_ode, bc, t_grid, iv)
k_bvp_t = solution_bvp.y[0]
c_bvp_t = solution_bvp.y[1]
'''

#these are for the ODE Solver
#c_0 = c_bvp_t[0]
#c_0 = c_benchmark[0]
c_0 = sol_matern["c_0"]





# Initial conditions
z0 = [k_0, c_0]  

# Time span
t_span = (0, t.max().item())


# Time points where we want the solution
t_eval = np.linspace(t_span[0], t_span[1], 100)

# Solve the ODE
solution_ivp = solve_ivp(neoclassical_ode, t_span, z0, t_eval=t_eval)

k_t_ode = solution_ivp.y[0]
c_t_ode = solution_ivp.y[1]

k_ode_rel_error = np.abs((k_t_ode-k_benchmark)/k_benchmark)
c_ode_rel_error = np.abs((c_t_ode-c_benchmark)/c_benchmark)


plt.figure(figsize=(15, 10))

ax_capital = plt.subplot(2, 2, 1)

plt.plot(t, k_hat_matern, color="k", label=r"$\hat{x}(t)$: Kernel Approximation")# Matérn
plt.plot(t_eval, k_t_ode, color="gray", label=r"$\hat{x}(t)$: Shooting Method")# Matérn
#plt.plot(t, k_benchmark, linestyle="--", color="k", label=r"$x(t)$: Benchmark Solution")
plt.axvline(x=T, color="k", linestyle=":", label="Extrapolation/Interpolation")

plt.ylabel("Capital: $x(t)$")
plt.xlabel("Time")
plt.legend()  # Show legend with labels


ax_rel_k = plt.subplot(2, 2, 2)
k_rel_error_ylim = (1e-6, 2 * 1e+1)

plt.plot(
    t,
    k_rel_error_matern,
    color="k",
    label=r"$\varepsilon_x(t)$: Kernel Method",
)
plt.plot(
    t,
    k_ode_rel_error,
    color="gray",
    label=r"$\varepsilon_x(t)$: Shooting Method",
)
plt.axvline(x=T, color="k", linestyle=":", label="Extrapolation/Interpolation")
plt.yscale("log")  # Set y-scale to logarithmic
plt.ylim(k_rel_error_ylim[0], k_rel_error_ylim[1])
plt.xlabel("Time")
plt.legend() 

ax_consumption = plt.subplot(2, 2, 3)
c_rel_error_ylim = (1e-7, 2 * 1e+1)

plt.plot(t, c_hat_matern, color="b", label=r"$\hat{y}(t)$: Kernel Approximation") #Matérn
plt.plot(t_eval, c_t_ode, color="c", label=r"$\hat{y}(t)$: Shooting Method")# Matérn
#plt.plot(t, c_benchmark, linestyle="--", color="b", label=r"$y(t)$: Benchmark Solution")
plt.axvline(x=T, color="k", linestyle=":", label="Extrapolation/Interpolation")

plt.ylabel("Consumption: $y(t)$")
plt.xlabel("Time")
plt.legend()  # Show legend with labels

ax_rel_c = plt.subplot(2, 2, 4)

plt.plot(
    t,
    c_rel_error_matern,
    color="b",
    label=r"$\varepsilon_y(t)$: Kernel Method",
)
plt.plot(
    t,
    c_ode_rel_error,
    color="c",
    label=r"$\varepsilon_y(t)$: Shooting Method",
)

plt.axvline(x=T, color="k", linestyle=":", label="Extrapolation/Interpolation")
plt.yscale("log")  # Set y-scale to logarithmic
plt.ylim(c_rel_error_ylim[0], c_rel_error_ylim[1])
plt.xlabel("Time")
plt.legend()  # Show legend with labels

plt.savefig(output_path, format="pdf")
