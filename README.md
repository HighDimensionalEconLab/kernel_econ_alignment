# kernel_econ_alignment

## Setup with uv
`uv` is a much faster alternative to conda, but is sometimes more challenging for binary dependencies.  To use it here,

1. Install [uv](https://github.com/astral-sh/uv#installation) to install `uv`.  Summary:
  - On linux/macos: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - On windows: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
2. Install optimizer dependencies.  On linux/macos with homebrew run
  - On macos: `brew install ipopt pkg-config`, linux likely: `sudo apt-get install coinor-libipopt-dev pkg-config`
  - Windows: `conda install -c conda-forge ipopt pkg-config`  This will only use conda for the binaries.  Make sure not to run it in a conda environment?
3. Synchronize the environment
```bash
  uv sync
```
- Finally, in VS Code you can activate the default environment with `>Python: Select Interpreter` to be the `.venv` local to the directory 
- If the debugger isn't working in that case, sometimes setting the vscode `terminal.integrated.shellIntegration.enabled: true` in the settings can help
- Outside of vscode, you will need to [activate .venv](https://docs.python.org/3/tutorial/venv.html#creating-virtual-environments) in your terminal 

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
print(sol["c_rel_error"].mean())
```