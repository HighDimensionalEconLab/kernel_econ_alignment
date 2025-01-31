import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
from new_asset_pricing_matern import asset_pricing_matern

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
    "figure.constrained_layout.use": True,  # Adjust layout to prevent overlap
    "font.size": fontsize,
    "axes.labelsize": fontsize,
    "axes.titlesize": fontsize,
    "xtick.labelsize": ticksize,
    "ytick.labelsize": ticksize,
}
plt.rcParams.update(params)


## Plot given solution
def plot_asset_pricing(
    sol_matern,
    output_path,
    p_rel_error_ylim=(1e-5, 2 * 1e-2),
    zoom=True,
    zoom_loc=[85, 95],
):
    t = sol_matern["t_test"]
    T = sol_matern["t_train"].max()
    p_hat_matern = sol_matern["p_test"]
    p_benchmark = sol_matern["p_benchmark"]
    p_rel_error_matern = sol_matern["p_rel_error"]


    # Plotting
    plt.figure(figsize=(15, 7))

    ax_prices = plt.subplot(1, 2, 1)

    plt.plot(
        t, p_hat_matern, color="k", label=r"$\hat{\mu}(t)$: Kernel Approximation"#Mtérn 
    )
    plt.plot(
        t,
        p_benchmark,
        linestyle="--",
        color="k",
        label=r"$\mu_f(t)$: Closed-Form Solution",
    )
    plt.axvline(x=T, color="k", linestyle=":", label="Extrapolation/Interpolation")

    plt.ylabel(r"Price: $\mu(t)$")
    plt.xlabel("Time")
    plt.legend()  # Show legend with labels

    ax_rel = plt.subplot(1, 2, 2)

    plt.plot(
        t,
        p_rel_error_matern,
        color="k",
        label=r"$\varepsilon_{\mu}(t)$: Relative Errors",
    )#, Matérn Kernel Approx.
   
    plt.axvline(x=T, color="k", linestyle=":", label="Extrapolation/Interpolation")
    plt.yscale("log")  # Set y-scale to logarithmic
    plt.ylim(p_rel_error_ylim[0], p_rel_error_ylim[1])
    plt.xlabel("Time")
    plt.legend()  # Show legend with labels

    # Zoom in part of the plot
    if zoom == True:
        time_window = (
            zoom_loc  # Indices: The window on the x-axis that want to be zoomed in
        )
        ave_value = 0.5 * (
            p_benchmark[time_window[0]] + p_benchmark[time_window[1]]
        )  # The average on the y-axis that want to be zoomed in
        window_width = 0.01 * ave_value
        axins = zoomed_inset_axes(
            ax_prices,
            3,
            loc="center",
            bbox_to_anchor=(0.5, 0.7, -0.3, -0.3),
            bbox_transform=ax_prices.transAxes,
        )

        axins.plot(
            t[time_window[0] - 1 : time_window[1] + 1],
            p_hat_matern[time_window[0] - 1 : time_window[1] + 1],
            color="k",
        )

        axins.plot(
            t[time_window[0] - 1 : time_window[1] + 1],
            p_benchmark[time_window[0] - 1 : time_window[1] + 1],
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
        mark_inset(ax_prices, axins, loc1=1, loc2=3, linewidth="0.7", ls="--", ec="0.5")

    plt.savefig(output_path, format="pdf")


# Plots with various parameters
sol_matern = asset_pricing_matern()
plot_asset_pricing(
    sol_matern, "figures/asset_pricing_contiguous.pdf"
)
