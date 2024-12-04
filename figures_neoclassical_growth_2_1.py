import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
from new_neoclassical_growth_matern import neoclassical_growth_matern
from neoclassical_growth_neural import neoclassical_growth_neural

from mpl_toolkits.axes_grid1.inset_locator import (
    zoomed_inset_axes,
    mark_inset,
    inset_axes,
)

fontsize = 14
ticksize = 14
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


## Plot given solution

sol_matern = neoclassical_growth_matern()
sol_neural = neoclassical_growth_neural()
output_path = "figures/neoclassical_growth_model_2_by_1.pdf"

zoom = True
zoom_loc = [90, 99]

t = sol_matern["t_test"]
T = sol_matern["t_train"].max()
c_hat_matern = sol_matern["c_test"]
k_hat_matern = sol_matern["k_test"]
c_benchmark = sol_matern["c_benchmark"]
k_benchmark = sol_matern["k_benchmark"]
k_rel_error_matern = sol_matern["k_rel_error"]
c_rel_error_matern = sol_matern["c_rel_error"]

c_hat_neural = sol_neural["c_test"]
k_hat_neural = sol_neural["k_test"]
k_rel_error_neural = sol_neural["k_rel_error"]
c_rel_error_neural = sol_neural["c_rel_error"]

# Plotting
plt.figure(figsize=(15, 8))

ax_capital = plt.subplot(1, 2, 1)

plt.plot(t, k_hat_matern, color="k", label=r"$\hat{x}(t)$: Matérn Kernel Approximation")
plt.plot(
    t,
    k_hat_neural,
    color="b",
    label=r"$\hat{x}(t)$: Neural Network Approximation",
)
plt.plot(t, k_benchmark, linestyle="--", color="k", label=r"$x(t)$: Benchmark Solution")
#plt.axvline(x=T, color="k", linestyle=":", label="Extrapolation/Interpolation")

plt.ylabel("Capital: $x(t)$")
plt.xlabel("Time")
plt.legend()  # Show legend with labels


ax_consumption = plt.subplot(1, 2, 2)

plt.plot(t, c_hat_matern, color="k", label=r"$\hat{y}(t)$: Matérn Kernel Approximation")
plt.plot(
    t,
    c_hat_neural,
    color="b",
    label=r"$\hat{y}(t)$: Neural Network Approximation",
)
plt.plot(t, c_benchmark, linestyle="--", color="k", label=r"$y(t)$: Benchmark Solution")
#plt.axvline(x=T, color="k", linestyle=":", label="Extrapolation/Interpolation")

plt.ylabel("Consumption: $y(t)$")
plt.xlabel("Time")
plt.legend()  # Show legend with labels


# Zoom in part of the plot
if zoom == True:
    time_window = (
        zoom_loc  # Indices: The window on the x-axis that want to be zoomed in
    )
    ave_value = 0.5 * (
        k_benchmark[time_window[0]] + k_benchmark[time_window[1]]
    )  # The average on the y-axis that want to be zoomed in
    window_width = 0.01 * ave_value
    axins = zoomed_inset_axes(
        ax_capital,
        3,
        loc="center",
        bbox_to_anchor=(0.5, 0.7, -0.3, -0.3),
        bbox_transform=ax_capital.transAxes,
    )

    axins.plot(
        t[time_window[0] - 1 : time_window[1] + 1],
        k_hat_matern[time_window[0] - 1 : time_window[1] + 1],
        color="k",
    )
    axins.plot(
        t[time_window[0] - 1 : time_window[1] + 1],
        k_hat_neural[time_window[0] - 1 : time_window[1] + 1],
        color="b",
    )
    axins.plot(
        t[time_window[0] - 1 : time_window[1] + 1],
        k_benchmark[time_window[0] - 1 : time_window[1] + 1],
        linestyle="--",
        color="k",
    )

    x1, x2, y1, y2 = (
        t[time_window[0]],
        t[time_window[1]],
        ave_value - window_width,
        ave_value + window_width,
    )
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    plt.xticks(fontsize=8, visible=False)
    plt.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    plt.yticks(fontsize=8)
    mark_inset(ax_capital, axins, loc1=1, loc2=3, linewidth="0.7", ls="--", ec="0.5")

    time_window = (
        zoom_loc  # Indices: The window on the x-axis that want to be zoomed in
    )
    ave_value = 0.5 * (
        c_benchmark[time_window[0]] + c_benchmark[time_window[1]]
    )  # The average on the y-axis that want to be zoomed in
    window_width = 0.01 * ave_value
    axins = zoomed_inset_axes(
        ax_consumption,
        3,
        loc="center",
        bbox_to_anchor=(0.5, 0.7, -0.3, -0.3),
        bbox_transform=ax_consumption.transAxes,
    )

    axins.plot(
        t[time_window[0] - 1 : time_window[1] + 1],
        c_hat_matern[time_window[0] - 1 : time_window[1] + 1],
        color="k",
    )
    axins.plot(
        t[time_window[0] - 1 : time_window[1] + 1],
        c_hat_neural[time_window[0] - 1 : time_window[1] + 1],
        color="b",
    )
    axins.plot(
        t[time_window[0] - 1 : time_window[1] + 1],
        c_benchmark[time_window[0] - 1 : time_window[1] + 1],
        linestyle="--",
        color="k",
    )

    x1, x2, y1, y2 = (
        t[time_window[0]],
        t[time_window[1]],
        ave_value - window_width,
        ave_value + window_width,
    )
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    plt.xticks(fontsize=8, visible=False)
    plt.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    plt.yticks(fontsize=8)
    mark_inset(
        ax_consumption, axins, loc1=1, loc2=3, linewidth="0.7", ls="--", ec="0.5"
    )
plt.savefig(output_path, format="pdf")
