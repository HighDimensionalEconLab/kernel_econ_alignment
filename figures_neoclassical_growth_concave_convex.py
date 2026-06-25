import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import jsonargparse
import subprocess
import sys
from neoclassical_growth_concave_convex_matern import neoclassical_growth_concave_convex_matern

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


def parse_threshold_payload(stdout: str):
    for line in reversed(stdout.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        return [
            {key: jnp.asarray(value) for key, value in item.items() if key != "k_0"}
            for item in payload
        ]
    return []


def solve_threshold_points(k_0_values, timeout_seconds: float):
    code = """
import json
import sys

import numpy as np

from neoclassical_growth_concave_convex_matern import neoclassical_growth_concave_convex_matern

payload = []
for raw_k_0 in sys.argv[1:]:
    k_0 = float(raw_k_0)
    try:
        sol = neoclassical_growth_concave_convex_matern(k_0=k_0, train_points=20)
    except Exception:
        continue
    payload.append({
        "k_0": k_0,
        "t_train": np.asarray(sol["t_train"]).tolist(),
        "t_test": np.asarray(sol["t_test"]).tolist(),
        "k_test": np.asarray(sol["k_test"]).tolist(),
        "c_test": np.asarray(sol["c_test"]).tolist(),
    })
print(json.dumps(payload))
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code, *[str(float(k_0)) for k_0 in k_0_values]],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return []
    if result.returncode != 0:
        return []

    return parse_threshold_payload(result.stdout)


def solve_threshold_group(k_0_values):
    sols = solve_threshold_points(k_0_values, timeout_seconds=3.0)
    if len(sols) == len(k_0_values):
        return sols

    fallback = []
    for k_0 in k_0_values:
        fallback.extend(solve_threshold_points([k_0], timeout_seconds=3.0))
    return fallback


def main():
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

    #plt.savefig(output_path, format="pdf")

    # Sweep x_0 across both basins, skipping failed, non-physical, or stalled solves.
    sols = []
    threshold_grid = np.linspace(0.5, 4.0, 40)
    for i in range(0, len(threshold_grid), 8):
        for sol in solve_threshold_group(threshold_grid[i : i + 8]):
            k = np.asarray(sol["k_test"])
            if np.all(np.isfinite(k)) and k.min() > 0 and k.max() < 10:
                sols.append(sol)

    if not sols:
        raise RuntimeError("No finite concave-convex threshold trajectories solved.")

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


if __name__ == "__main__":
    jsonargparse.CLI(main)
