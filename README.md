# Solving Models of Economic Dynamics with Ridgeless Kernel Regressions
This repository contains the replication code and extra examples for [Solving Models of Economic Dynamics with Ridgeless Kernel Regressions](https://arxiv.org/pdf/2406.01898) by Mahdi Ebrahimi Kahou, Jesse Perla, and Geoff Pleiss

Full replication code is provided for Python, and additional Julia code is in progress.

# Python Setup
We recommend installation with `uv` (on all platforms, but especially on MacOS and Linux).  If you wish to use `conda` see notes at the end of these instructions.

## Windows Ipopt Installation
Ipopt binary installation is tricky, and we find the most reliable method is to use Anaconda (even if you will otherwise use `uv`).  To do so, install ipopt in your base Anaconda with

```bash
conda install -n base -c conda-forge ipopt=3.11.1 pkg-config
```
  - For Windows, version `3.11.1` seems to be essential.
  - If you have clutter in your base Anaconda which prevents this from correctly installing, then you may need to reinstall Anaconda.
  - If you run into issues the binary dependencies in Julia are bullet-proof and seamless.

## Setup with uv
`uv` is a much faster alternative to Conda, even if it has incomplete support for challenging binary dependencies.

1. Install [uv](https://github.com/astral-sh/uv#installation). This is a usually a one-line installation:
     - On Linux/MacOS: `curl -LsSf https://astral.sh/uv/install.sh | sh`
     - On Windows: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"` or with winget `winget install --id=astral-sh.uv  -e`
     - If you have any installation issues, [see the docs](https://docs.astral.sh/uv/getting-started/installation/) for troubleshooting
2. Install optimizer dependencies.
     - On MacOS: `brew install ipopt pkg-config`
     - Linux: `sudo apt-get install coinor-libipopt-dev pkg-config`
     - Windows: See note above if you have not previously installed Ipopt or JAX
3. Synchronize the environment
```bash
  uv sync
```
4. Finally, you will want to make sure you activate the python environment each time you use it.
   - In VS Code you can activate the default environment with `>Python: Select Interpreter` to be the `.venv` local to the directory
   - If the debugger isn't working in that case, sometimes setting the vscode `terminal.integrated.shellIntegration.enabled: true` in the settings can help
   - Outside of vscode, a simple, platform specific CLI line will [activate .venv](https://docs.python.org/3/tutorial/venv.html#creating-virtual-environments) in your terminal.

**Troubleshooting**:
- If you receive JAX errors about DLL load failures, you may need to update [https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170#visual-studio-2015-2017-2019-and-2022](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170#visual-studio-2015-2017-2019-and-2022)
- See notes above on Windows Ipopt installation challenges

## Figure Replication
After installation, to generate all of the figures in the paper and the appendices,

```bash
python all_figures.py
```

The output will be in the `./figures` folder

## Example Usage
The individual files support CLI arguments.  To pick specific points rather than the linspace grid, pass in `--train_points_list` as below

```bash
python neoclassical_growth_matern.py
python neoclassical_growth_matern.py --train_points=5
python neoclassical_growth_matern.py --rho=5.0
python neoclassical_growth_matern.py --train_points_list="[0.0,2.0,5.0,10.0,20.0]"
python neoclassical_growth_matern.py --train_points=20 --train_T=10.0 --test_T=10.0 --k_0=0.5
```

These functions can also be imported and called directly, for example,

```python
from neoclassical_growth_matern import neoclassical_growth_matern
sol = neoclassical_growth_matern(rho=10.0)
print(sol["c_rel_error"].mean())~~~~
```

## Python Conda Installation
If you prefer to use `conda` for your entire environment management, install `uv` as above, then generate a requirements file and a conda environment

```bash
uv pip freeze > requirements.txt
conda create -n kernels python=3.11
conda activate kernels
pip install -r requirements.txt
conda install -c conda-forge ipopt=3.11.1
```


# Julia Setup
Julia is more straightforward and cross-platform
1. If required, install Julia from [juliaup](https://github.com/JuliaLang/juliaup).  This can be done in a one-line terminal command:
   - MacOS/Linux: `curl -fsSL https://install.julialang.org | sh`
   - Windows: `winget install --name Julia --id 9NJNWW8PVKMN -e -s msstore`
2. Install the [VS Code Julia extension](https://marketplace.visualstudio.com/items?itemName=julialang.language-julia).  See [here](https://julia.quantecon.org/getting_started_julia/getting_started.html#setting-up-git-and-vs-code) for more details.
3. Then activate and instantiate the environment to get the required packages.  See [here](https://julia.quantecon.org/getting_started_julia/getting_started.html#installing-packages) for more.
     - If you start a Julia REPL is VS Code, then you just need: `] instantiate`