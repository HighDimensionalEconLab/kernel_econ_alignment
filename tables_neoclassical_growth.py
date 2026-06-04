import pandas as pd
import jsonargparse

from neoclassical_growth_matern import neoclassical_growth_matern
from neoclassical_growth_matern_cvxpy import neoclassical_growth_matern_cvxpy


# implementation selects the solve backend: "cvxpy" (default, DNLP via UNO)
# or "pyomo" (Ipopt binary).
def main(implementation: str = "cvxpy"):
    solve = {
        "cvxpy": neoclassical_growth_matern_cvxpy,
        "pyomo": neoclassical_growth_matern,
    }[implementation]

    sol_default = solve()
    sol_nu_1_5 = solve(nu=1.5)
    sol_nu_2_5 = solve(nu=2.5, lambda_p=1e-4)
    sol_rho_2 = solve(rho=2)
    sol_rho_20 = solve(rho=20)

    k_rel_error = sol_default["k_rel_error"]
    c_rel_error = sol_default["c_rel_error"]

    sols = [sol_default, sol_nu_1_5, sol_nu_2_5, sol_rho_2, sol_rho_20]

    df = pd.DataFrame(
        {
            r"$\nu$": [r"$1/2$", r"$3/2$", r"$5/2$", r"$1/2$", r"$1/2$"],
            r"$\ell$": [10, 10, 10, 2, 20],
            r"Max of Rel. Error: $\hat{x}(t)$": [
                sol["k_rel_error"].max().item() for sol in sols
            ],
            r"Max of Rel. Error: $\hat{y}(t)$": [
                sol["c_rel_error"].max().item() for sol in sols
            ],
            r"Min of Rel. Error: $\hat{x}(t)$": [
                sol["k_rel_error"][1:].min().item() for sol in sols
            ],
            r"Min of Rel. Error: $\hat{y}(t)$": [
                sol["c_rel_error"].min().item() for sol in sols
            ],
        }
    )

    with open("figures/neoclassical_growth_model_nu_rho.tex", "w") as f:
        f.write(df.to_latex(index=False, float_format="%.1e"))


if __name__ == "__main__":
    jsonargparse.CLI(main)


# r"Avg. of Rel. Error: $\hat{k}(t)$": [
# sol["k_rel_error"].mean().item() for sol in sols
# ],
# r"Avg. of Rel. Error: $\hat{c}(t)$": [
# sol["c_rel_error"].mean().item() for sol in sols
# ],
