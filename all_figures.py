import os

output_folder = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
)
os.makedirs(output_folder, exist_ok=True)

# run all the figures scripts
print("Executing figures_asset_pricing.py")
os.system("python figures_asset_pricing.py")

print("Executing figures_neoclassical_growth_concave_convex.py")
os.system("python figures_neoclassical_growth_concave_convex.py")

print("Executing figures_neoclassical_growth_baseline.py")
os.system("python figures_neoclassical_growth_baseline.py")

print("Executing figures_neoclassical_growth_robustness.py")
os.system("python figures_neoclassical_growth_robustness.py")

print("Executing figures_optimal_advertising.py")
os.system("python figures_optimal_advertising.py")

print("Executing tables_neoclassical_growth.py")
os.system("python tables_neoclassical_growth.py")

print("Executing figures_neoclassical_human_capital.py")
os.system("python figures_neoclassical_human_capital.py")
