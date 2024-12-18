import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
from new_neoclassical_human_capital_matern import human_capital_matern

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
sol = human_capital_matern()
output_path = "figures/neoclassical_human_capital.pdf"

t = sol["t_test"]
T = sol["t_train"].max()
c_hat = sol["c_test"]
k_hat = sol["k_test"]
h_hat = sol["h_test"]
i_k_hat = sol["i_k_test"]
i_h_hat = sol["i_h_test"]
mu_hat = sol["mu_test"]
# Plotting

ax_physical_capital = plt.subplot(3, 2, 1)

plt.plot(t, k_hat, color="k", label=r"$\hat{x}_k(t)$")
#plt.axhline(y=sol["k_ss"], linestyle="-.", color="k", label=r"$k^*$:Steady-State")
plt.axvline(x=T, color="k", linestyle=":", label="Extrapolation/Interpolation")
plt.ylabel("Physical Capital: $x_k(t)$")
plt.xlabel("Time")
plt.legend()  # Show legend with labels

ax_human_capital = plt.subplot(3, 2, 2)

plt.plot(t, h_hat, color="k", label=r"$\hat{x}_h(t)$")
#plt.axhline(y=sol["h_ss"], linestyle="-.", color="grey", label=r"$h^*$: Steady-State")
plt.axvline(x=T, color="k", linestyle=":", label="Extrapolation/Interpolation")
plt.ylabel("Human Capital: $x_h(t)$")
plt.xlabel("Time")
plt.legend()  # Show legend with labels

ax_consumption = plt.subplot(3, 2, 3)

plt.plot(t, c_hat, color="b", label=r"$\hat{y}_c(t)$")
#plt.axhline(y=sol["c_ss"], linestyle="-.", color="b", label=r"$c^*$: Steady-State")
plt.axvline(x=T, color="b", linestyle=":", label="Extrapolation/Interpolation")
plt.ylabel("Consumption: $y_c(t)$")
plt.xlabel("Time")
plt.legend()  # Show legend with labels

ax_investment_k = plt.subplot(3, 2, 4)

plt.plot(t, i_k_hat, color="b", label=r"$\hat{y}_k(t)$")
#plt.axhline(y=sol["i_k_ss"], linestyle="-.", color="k", label=r"$i_k^*$: Steady-State")
plt.axvline(x=T, color="b", linestyle=":", label="Extrapolation/Interpolation")
plt.ylabel("Physical Capital Investment: $y_k(t)$")
plt.xlabel("Time")
plt.legend()  # Show legend with labels

ax_investment_h = plt.subplot(3, 2, 5)

plt.plot(t, i_h_hat, color="b", label=r"$\hat{y}_h(t)$")
#plt.axhline(y=sol["i_h_ss"], linestyle="-.", color="grey", label=r"$i_h^*$: Steady-State")
plt.axvline(x=T, color="b", linestyle=":", label="Extrapolation/Interpolation")
plt.ylabel("Human Capital Investment: $y_h(t)$")
plt.xlabel("Time")
plt.legend()  # Show legend with labels

ax_investment_h = plt.subplot(3, 2, 6)

plt.plot(t, mu_hat, color="grey", label=r"$\hat{\mu}(t)$")
#plt.axhline(y=sol["i_h_ss"], linestyle="-.", color="grey", label=r"$i_h^*$: Steady-State")
plt.axvline(x=T, color="grey", linestyle=":", label="Extrapolation/Interpolation")
plt.ylabel("Co-state Variable: $\mu(t)$")
plt.xlabel("Time")
plt.legend()  # Show legend with labels


plt.tight_layout()  # Adjust layout to prevent overlap

plt.savefig(output_path, format="pdf")
