import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import os
from new_neoclassical_growth_concave_convex_matern  import neoclassical_growth_concave_convex_matern

from mpl_toolkits.axes_grid1.inset_locator import (
    zoomed_inset_axes,
    mark_inset,
    inset_axes,
)

fontsize = 17
ticksize = 16
figsize = (15, 8)
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


## Plots for concave-convex production function
sol_1 = neoclassical_growth_concave_convex_matern(k_0=0.5, train_points=20)
sol_2 = neoclassical_growth_concave_convex_matern(k_0=1.0, train_points=20)
sol_3 = neoclassical_growth_concave_convex_matern(k_0=3.0, train_points=20)
sol_4 = neoclassical_growth_concave_convex_matern(k_0=4.0, train_points=20)
output_path = "figures/neoclassical_growth_model_concave_convex.pdf"

plt.figure(figsize=(15, 8))

k_hat_1 = sol_1["k_test"]
k_hat_2 = sol_2["k_test"]
k_hat_3 = sol_3["k_test"]
k_hat_4 = sol_4["k_test"]

c_hat_1 = sol_1["c_test"]
c_hat_2 = sol_2["c_test"]
c_hat_3 = sol_3["c_test"]
c_hat_4 = sol_4["c_test"]

T = sol_1["t_train"].max()
t = sol_1["t_test"]

ax_capital = plt.subplot(1, 2, 1)
plt.plot(t, k_hat_1, color="b", label=r"$\hat{x}(t): x_0 = 0.5$")
plt.plot(t, k_hat_2, color="gray", label=r"$\hat{x}(t): x_0 = 1$")
plt.plot(t, k_hat_3, color="r", label=r"$\hat{x}(t): x_0 = 3$")
plt.plot(t, k_hat_4, color="c", label=r"$\hat{x}(t): x_0 = 4$")
# plt.axhline(y=sol_1["k_ss_low"], linestyle="-.", color="k", label=r"$x_1^*$: Steady-State")
# plt.axhline(y=sol_1["k_ss_high"], linestyle="dashed", color="k", label=r"$x_2^*$: Steady-State")
plt.axvline(x=T, color="k", linestyle=":", label="Extrapolation/Interpolation")
plt.ylabel("Capital: $x(t)$")
plt.xlabel("Time")
plt.legend()  # Show legend with labels

ax_consumption = plt.subplot(1, 2, 2)
plt.plot(t, c_hat_1, color="b", label=r"$\hat{y}(t): x_0 = 0.5$")
plt.plot(t, c_hat_2, color="gray", label=r"$\hat{y}(t): x_0 = 1$")
plt.plot(t, c_hat_3, color="r", label=r"$\hat{y}(t): x_0 = 3$")
plt.plot(t, c_hat_4, color="c", label=r"$\hat{y}(t): x_0 = 4$")
# plt.axhline(y=sol_1["c_ss_low"], linestyle="-.", color="k", label=r"$y_1^*$: Steady-State")
# plt.axhline(y=sol_1["c_ss_high"], linestyle="dashed", color="k", label=r"$y_2^*$: Steady-State")
plt.axvline(x=T, color="k", linestyle=":", label="Extrapolation/Interpolation")
plt.ylabel("Consumption: $y(t)$")
plt.xlabel("Time")
plt.legend()  # Show legend with labels

plt.savefig(output_path, format="pdf")
sols = [
    neoclassical_growth_concave_convex_matern(k_0=k_0, train_points=20)
    for k_0 in np.linspace(0.5, 4.0, 70)
]

output_path = "figures/neoclassical_growth_model_concave_convex_threshold.pdf"

plt.figure(figsize=(15,8))

T = sols[0]["t_train"].max()
t = sols[0]["t_test"]

ax_capital = plt.subplot(1, 2, 1)
for sol in sols:
    plt.plot(t, sol["k_test"], color="gray")

# plt.axhline(
#    y=sols[0]["k_ss_low"], linestyle="-.", color="k", label=r"$k_1^*$: Steady-State"
# )
# plt.axhline(
#    y=sols[0]["k_ss_high"],
#    linestyle="dashed",
#    color="k",
#    label=r"$k_2^*$: Steady-State",
# )
plt.axvline(x=T, color="k", linestyle=":", label="Extrapolation/Interpolation")
plt.ylabel("Capital: $x(t)$")
plt.xlabel("Time")
plt.legend()  # Show legend with labels

ax_consumption = plt.subplot(1, 2, 2)
for sol in sols:
    plt.plot(t, sol["c_test"], color="b")

# plt.axhline(
#    y=sols[0]["c_ss_low"], linestyle="-.", color="k", label=r"$c_1^*$: Steady-State"
# )
# plt.axhline(
#    y=sols[0]["c_ss_high"],
#    linestyle="dashed",
#    color="k",
#    label=r"$c_2^*$: Steady-State",
# )
plt.axvline(x=T, color="k", linestyle=":", label="Extrapolation/Interpolation")
plt.ylabel("Consumption: $y(t)$")
plt.xlabel("Time")
plt.legend()  # Show legend with labels

plt.savefig(output_path, format="pdf")
