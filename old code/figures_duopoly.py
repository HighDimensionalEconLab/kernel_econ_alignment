import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
from new_duopoly import duopoly_matern
from mpl_toolkits.axes_grid1.inset_locator import (
    zoomed_inset_axes,
    mark_inset,
    inset_axes,
)

fontsize = 14
ticksize = 14
figsize = (15, 15)
params = {
    "font.family": "serif",
    "figure.figsize": figsize,
    "figure.dpi": 80,
    "figure.edgecolor": "k",
    "font.size": fontsize,
    "axes.labelsize": fontsize,
    "axes.titlesize": fontsize,
    "xtick.labelsize": ticksize,
    "ytick.labelsize": ticksize,
}
plt.rcParams.update(params)


## Plot given solution
sol = duopoly_matern()
#output_path = "figures/neoclassical_human_capital.pdf"

t = sol["t_test"]
T = sol["t_train"].max()
u_1_test = sol["u_1_test"]
u_2_test = sol["u_2_test"]

# Plotting

ax_physical_capital = plt.subplot(1, 2, 1)

plt.plot(t, u_1_test, color="k", label=r"$\hat{x}_1(t)$")
#plt.axhline(y=sol["k_ss"], linestyle="-.", color="k", label=r"$k^*$:Steady-State")
plt.axvline(x=T, color="k", linestyle=":", label="Extrapolation/Interpolation")
plt.ylabel("Firm 1: $x_1(t)$")
plt.xlabel("Time")
plt.legend()  # Show legend with labels

ax_human_capital = plt.subplot(1, 2, 2)

plt.plot(t, u_2_test, color="k", label=r"$\hat{x}_2(t)$")
#plt.axhline(y=sol["h_ss"], linestyle="-.", color="grey", label=r"$h^*$: Steady-State")
plt.axvline(x=T, color="k", linestyle=":", label="Extrapolation/Interpolation")
plt.ylabel("Firm 2: $x_1(t)$")
plt.xlabel("Time")
plt.legend()  # Show legend with labels


plt.tight_layout()  # Adjust layout to prevent overlap

