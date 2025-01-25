import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import os
from new_optimal_advertising_matern import (
    optimal_advertising_matern,
)

from mpl_toolkits.axes_grid1.inset_locator import (
    zoomed_inset_axes,
    mark_inset,
    inset_axes,
)

fontsize = 17
ticksize = 16
figsize = (15, 7)
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


## Plot for optimal advertising
sol = optimal_advertising_matern()
output_path = "figures/optimal_advertising.pdf"

plt.figure(figsize=(15, 7))

x_hat = sol["x_test"]
mu_hat = sol["mu_test"]
u_hat = sol["u_test"]

T = sol["t_train"].max()
t = sol["t_test"]

ax_market_share = plt.subplot(1, 2, 1)
plt.plot(t, x_hat, color="k", label=r"$\hat{x}(t)$")
#plt.axhline(y=sol["x_ss"], linestyle="-.", color="k", label=r"$x^*$: Steady-State")
plt.axvline(x=T, color="k", linestyle=":", label="Extrapolation/Interpolation")
plt.ylabel("Market Share: $x(t)$")
plt.xlabel("Time")
plt.legend()  # Show legend with labels

ax_mu = plt.subplot(1, 2, 2)
plt.plot(t, mu_hat, color="blue", label=r"$\hat{\mu}(t)$")
#plt.axhline(y=sol["mu_ss"], linestyle="-.", color="b", label=r"$\mu^*$: Steady-State")
plt.axvline(x=T, color="k", linestyle=":", label="Extrapolation/Interpolation")
plt.ylabel("Co-state Variable: $\mu(t)$")
plt.xlabel("Time")
plt.legend()  # Show legend with labels
plt.tight_layout()

plt.savefig(output_path, format="pdf")
