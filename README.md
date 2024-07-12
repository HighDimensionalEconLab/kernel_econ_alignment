# kernel_econ_alignment

## Setup
```bash
conda create -n kernels python=3.11
conda activate kernels
pip install -r requirements.txt
conda install -c conda-forge ipopt=3.11.1
```

If you have trouble with that, try `conda install -c conda-forge ipopt=3.11.1`

- Then you will need to activate it with `conda activate kernels`.
- Then when using vscode, consider `>Python: Select Interpreter` to select the `kernels` environment.
- If the debugger isn't working in that case, sometimes setting the vscode `terminal.integrated.shellIntegration.enabled: true` in the settings can help

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