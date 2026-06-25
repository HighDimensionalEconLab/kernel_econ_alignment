# Solving Models of Economic Dynamics with Ridgeless Kernel Regressions

This repository contains replication code and extra Python examples for
[Solving Models of Economic Dynamics with Ridgeless Kernel Regressions](https://arxiv.org/pdf/2406.01898)
by Mahdi Ebrahimi Kahou, Jesse Perla, and Geoff Pleiss.

## Setup

Use `uv` for all Python environment management.

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
   - Linux/macOS: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Windows PowerShell: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
   - Windows winget: `winget install --id=astral-sh.uv -e`
2. Synchronize the environment:

```bash
uv sync
```

The project solves the RKHS collocation problems through direct JAX callbacks
to UNO via the `unopy` wheel. No optimization DSL, external optimizer
executable, or conda environment is needed.

The human-capital example uses `nlls_gram` for its small JAX float64
initial-condition solve.

## Figure Replication

Generate the paper and appendix figures with:

```bash
uv run python all_figures.py
```

Outputs are written to `./figures`.

## Example Usage

Individual scripts support CLI arguments through `jsonargparse`:

```bash
uv run python neoclassical_growth_matern.py
uv run python neoclassical_growth_matern.py --train_points=5
uv run python neoclassical_growth_matern.py --rho=5.0
uv run python neoclassical_growth_matern.py --train_points_list="[0.0,2.0,5.0,10.0,20.0]"
uv run python neoclassical_growth_matern.py --train_points=20 --train_T=10.0 --test_T=10.0 --k_0=0.5
```

The functions can also be imported directly:

```python
from neoclassical_growth_matern import neoclassical_growth_matern

sol = neoclassical_growth_matern(rho=10.0)
print(sol["c_rel_error"].mean())
```

## Models

- `asset_pricing_matern.py`: asset-pricing QP.
- `neoclassical_growth_matern.py`: baseline and optional kinked-production neoclassical growth DNLP.
- `neoclassical_growth_concave_convex_matern.py`: concave-convex growth wrapper.
- `neoclassical_human_capital_matern.py`: two-capital human-capital DNLP.
- `optimal_advertising_matern.py`: optimal-advertising DNLP.

## Tests

Run the Python smoke and initialization checks with:

```bash
uv run python -m unittest discover -s tests
```
