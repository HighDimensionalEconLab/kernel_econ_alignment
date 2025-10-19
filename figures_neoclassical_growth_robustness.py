import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
from neoclassical_growth_matern import neoclassical_growth_matern

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
    "font.size": fontsize,
    "axes.labelsize": fontsize,
    "axes.titlesize": fontsize,
    "xtick.labelsize": ticksize,
    "ytick.labelsize": ticksize,
}
plt.rcParams.update(params)


## Plot given solution
def plot_neoclassical_growth(
    sol,
    output_path,
    k_rel_error_ylim=(1e-6, 2 * 1e-2),
    c_rel_error_ylim=(1e-6, 2 * 1e-2),
    zoom=True,
    zoom_loc=[50, 60],
):
    t = sol["t_test"]
    T = sol["t_train"].max()
    c_hat = sol["c_test"]
    k_hat = sol["k_test"]
    c_benchmark = sol["c_benchmark"]
    k_benchmark = sol["k_benchmark"]
    k_rel_error = sol["k_rel_error"]
    c_rel_error = sol["c_rel_error"]
    # Plotting
    plt.figure(figsize=(15, 10))

    ax_capital = plt.subplot(2, 2, 1)

    plt.plot(t, k_hat, color="k", label=r"$\hat{x}(t)$: Kernel Approximation") #Matérn
    plt.plot(
        t, k_benchmark, linestyle="--", color="k", label=r"$x(t)$: Benchmark Solution"
    )
    plt.axvline(x=T, color="k", linestyle=":", label="Extrapolation/Interpolation")

    plt.ylabel("Capital: $x(t)$")
    plt.xlabel("Time")
    plt.legend()  # Show legend with labels

    ax_rel_k = plt.subplot(2, 2, 2)

    plt.plot(
        t,
        k_rel_error,
        color="k",
        label=r"$\varepsilon_x(t)$: Relative Errors for $x(t)$",
    )
    plt.axvline(x=T, color="k", linestyle=":", label="Extrapolation/Interpolation")
    plt.yscale("log")  # Set y-scale to logarithmic
    plt.ylim(k_rel_error_ylim[0], k_rel_error_ylim[1])
    plt.xlabel("Time")
    plt.legend()  # Show legend with labels

    ax_consumption = plt.subplot(2, 2, 3)

    plt.plot(t, c_hat, color="b", label=r"$\hat{y}(t)$: Kernel Approximation") #Matérn
    plt.plot(
        t, c_benchmark, linestyle="--", color="b", label=r"$y(t)$: Benchmark Solution"
    )
    plt.axvline(x=T, color="k", linestyle=":", label="Extrapolation/Interpolation")

    plt.ylabel("Consumption: $y(t)$")
    plt.xlabel("Time")
    plt.legend()  # Show legend with labels

    ax_rel_c = plt.subplot(2, 2, 4)

    plt.plot(
        t,
        c_rel_error,
        color="b",
        label=r"$\varepsilon_y(t)$: Reletaive Errors for $y(t)$",
    )
    plt.axvline(x=T, color="k", linestyle=":", label="Extrapolation/Interpolation")
    plt.yscale("log")  # Set y-scale to logarithmic
    plt.ylim(c_rel_error_ylim[0], c_rel_error_ylim[1])
    plt.xlabel("Time")
    plt.legend()  # Show legend with labels

    plt.tight_layout()  # Adjust layout to prevent overlap

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
            k_hat[time_window[0] - 1 : time_window[1] + 1],
            color="k",
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
        plt.tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )
        plt.yticks(fontsize=8)
        mark_inset(
            ax_capital, axins, loc1=1, loc2=3, linewidth="0.7", ls="--", ec="0.5"
        )

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
            c_hat[time_window[0] - 1 : time_window[1] + 1],
            color="b",
        )
        axins.plot(
            t[time_window[0] - 1 : time_window[1] + 1],
            c_benchmark[time_window[0] - 1 : time_window[1] + 1],
            linestyle="--",
            color="b",
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
        plt.tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )
        plt.yticks(fontsize=8)
        mark_inset(
            ax_consumption, axins, loc1=1, loc2=3, linewidth="0.7", ls="--", ec="0.5"
        )
    plt.savefig(output_path, format="pdf")


# Plots with various parameters
sol = neoclassical_growth_matern(
    train_points_list=[0.0, 1.0, 3.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 38.0, 40.0], lambda_p = 1e-6
)
plot_neoclassical_growth(
    sol,
    "figures/neoclassical_growth_model_sparse.pdf",
    c_rel_error_ylim=(1e-7, 2 * 1e-2),
    zoom=True,
    zoom_loc=[10, 20],
)

sol = neoclassical_growth_matern(train_T=10.0, train_points=11, test_T=15.0)
plot_neoclassical_growth(
    sol,
    "figures/neoclassical_growth_model_far_steady_state.pdf",
    k_rel_error_ylim=(1e-4, 1e-1),
    c_rel_error_ylim=(1e-4, 1e-1),
    zoom=False,
)
'''
sol = neoclassical_growth_matern(nu=1.5)
plot_neoclassical_growth(
    sol,
    "figures/neoclassical_growth_model_nu_1_5.pdf",
    k_rel_error_ylim=(1e-7, 1e-2),
    c_rel_error_ylim=(1e-7, 1e-2),
    zoom=True,
    zoom_loc=[75, 85],
)


sol = neoclassical_growth_matern(nu=2.5)
plot_neoclassical_growth(
    sol,
    "figures/neoclassical_growth_model_nu_2_5.pdf",
    k_rel_error_ylim=(1e-7, 1e-2),
    c_rel_error_ylim=(1e-8, 1e-2),
    zoom=True,
    zoom_loc=[75, 85],
)


sol = neoclassical_growth_matern(rho=2)
plot_neoclassical_growth(
    sol,
    "figures/neoclassical_growth_model_rho_2.pdf",
    k_rel_error_ylim=(1e-7, 1e-2),
    c_rel_error_ylim=(1e-7, 1e-2),
    zoom=True,
    zoom_loc=[75, 85],
)


sol = neoclassical_growth_matern(rho=20)
plot_neoclassical_growth(
    sol,
    "figures/neoclassical_growth_model_rho_20.pdf",
    k_rel_error_ylim=(1e-7, 1e-2),
    c_rel_error_ylim=(1e-7, 1e-2),
    zoom=True,
    zoom_loc=[75, 85],
)
'''
