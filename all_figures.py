import os
import subprocess
import sys

output_folder = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
)
os.makedirs(output_folder, exist_ok=True)

SCRIPTS = [
    "figures_asset_pricing.py",
    "figures_neoclassical_growth_concave_convex.py",
    "figures_neoclassical_growth_baseline.py",
    "figures_neoclassical_growth_robustness.py",
    "figures_optimal_advertising.py",
    "tables_neoclassical_growth.py",
    "figures_neoclassical_human_capital.py",
]


for script in SCRIPTS:
    print(f"Executing {script}")
    subprocess.run([sys.executable, script], check=True)
