import pandas as pd

from new_neoclassical_growth_matern import neoclassical_growth_matern

sol_default = neoclassical_growth_matern()
sol_nu_1_5 = neoclassical_growth_matern(nu=1.5)
sol_nu_2_5 = neoclassical_growth_matern(nu=2.5,lambda_p = 1e-4)
sol_rho_2 = neoclassical_growth_matern(rho=2)
sol_rho_20 = neoclassical_growth_matern(rho=20)

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


# r"Avg. of Rel. Error: $\hat{k}(t)$": [
# sol["k_rel_error"].mean().item() for sol in sols
# ],
# r"Avg. of Rel. Error: $\hat{c}(t)$": [
# sol["c_rel_error"].mean().item() for sol in sols
# ],
