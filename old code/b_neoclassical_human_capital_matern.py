import jax
import jax.numpy as jnp
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition
import jsonargparse
from jax import config
from kernels import integrated_matern_kernel_matrices
from typing import List, Optional

config.update("jax_enable_x64", True)

from scipy.optimize import fsolve

def human_capital_matern(
    a_k: float = 1 / 3,
    a_h: float = 1 / 4,
    delta_k: float = 0.1,
    delta_h: float = 0.05,
    rho_hat: float = 0.11, #discount rate
    k_0: float = 1.5, #3.0024724187979452, #1.5,
    nu: float = 0.5,
    sigma: float = 1.0,
    rho: float = 20,
    lambda_b_k: float = 1.0,
    lambda_b_h: float = 1.0,
    solver_type: str = "ipopt",
    train_T: float = 80.0,
    train_points: int = 60,
    test_T: float = 100.0,
    test_points: int = 100,
    benchmark_T: float = 60.0,
    benchmark_points: int = 300,
    train_points_list: Optional[List[float]] = None,
    verbose: bool = False,
):
    # if passing in `train_points` then doesn't us a grid.  Otherwise, uses linspace
    if train_points_list is None:
        train_data = jnp.linspace(0, train_T, train_points)
    else:
        train_data = jnp.array(train_points_list)
    test_data = jnp.linspace(0, test_T, test_points)
    benchmark_grid = jnp.linspace(0, benchmark_T, benchmark_points)

    # Construct kernel matrices
    N = len(train_data)
    K, K_tilde = integrated_matern_kernel_matrices(
        train_data, train_data, nu, sigma, rho
    )
    K = np.array(K)  # pyomo doesn't support jax arrays
    K_tilde = np.array(K_tilde)

    # Production function
    def f(k, h):
        return (k**a_k) * (h**a_h)
    
    def f_k(k, h):
        return (a_k * k**(a_k - 1)) * (h**a_h)
    
    def f_h(k, h):
        return (k**a_k) * (a_h * h**(a_h - 1))
    
    def no_arbitrage_constraint(h):
        return f_h(k_0, h) - f_k(k_0, h) - delta_h + delta_k
    
    initial_guess = [k_0]
    result = fsolve(no_arbitrage_constraint, initial_guess)
    h_0 = result[0]

    # Create pyomo model and variables
    m = pyo.ConcreteModel()
    m.I = range(N)
    # 8 variables
    m.alpha_k = pyo.Var(m.I, within=pyo.Reals, initialize = 0.0) # coeffs for x_k(t): we call it in the code k(t)
    m.alpha_h = pyo.Var(m.I, within=pyo.Reals, initialize = 0.0) # coeffs for x_h(t): we call it in the code h(t)
    m.alpha_mu = pyo.Var(m.I, within=pyo.Reals, initialize = 0.0)  # coeffs for \mu(t)
    m.alpha_i_k = pyo.Var(m.I, within=pyo.Reals, initialize = 0.0) # coeff for y_k(t): we call it i_k(t) in the code
    m.alpha_i_h = pyo.Var(m.I, within=pyo.Reals, initialize = 0.0) # coeff for y_h(t): we call it i_h(t) in the code
    m.alpha_c = pyo.Var(m.I, within=pyo.Reals, initialize = 0.0) # coeff for for y_c(t):  we call it c(t) in the code
    m.alpha_b_k = pyo.Var(m.I, within=pyo.Reals, initialize=0.0) # coeff for b_k(t) = \mu(t)x_k(t)
    m.alpha_b_h = pyo.Var(m.I, within=pyo.Reals, initialize=0.0) # coeff for b_k(t) = \mu(t)x_h(t)

    #initializations of variables at 0 for 6 variables: h_0 and k_0 are given to us from the economic setup
    m.i_k_0 = pyo.Var(within=pyo.NonNegativeReals, initialize=delta_k * k_0) #y_k(0)
    m.i_h_0 = pyo.Var(within=pyo.NonNegativeReals, initialize=delta_h * h_0)#y_h(0)
    m.c_0 = pyo.Var(within=pyo.NonNegativeReals, initialize = f(k_0, h_0) - delta_h * h_0 - delta_k * k_0) #y_c(0)
    m.mu_0 = pyo.Var(within=pyo.NonNegativeReals, initialize = 1/(f(k_0, h_0) - delta_h * h_0 - delta_k * k_0)) #\mu(0) = 1/c(0) 
    m.b_k_0 = pyo.Var(within=pyo.NonNegativeReals, initialize=k_0/(f(k_0, h_0) - delta_h * h_0 - delta_k * k_0)) # b_k_0 = m(0)k(0)
    m.b_h_0 = pyo.Var(within=pyo.NonNegativeReals, initialize=h_0/(f(k_0, h_0) - delta_h * h_0 - delta_k * k_0)) #b_h_0 = m(0)k(0)


    # defining the integrated kernel functional approximation
    def k(m, i): #x_k(t): physical capital k(t)
        return k_0 + sum(K_tilde[i, j] * m.alpha_k[j] for j in m.I)
    
    def h(m, i): #x_h(t): human capital h(t)
        return h_0 + sum(K_tilde[i, j] * m.alpha_h[j] for j in m.I)
    
    def i_k(m, i): # y_k(t): investment in physical capital i_k(t)
        return m.i_k_0 + sum(K_tilde[i, j] * m.alpha_i_k[j] for j in m.I)
    
    def i_h(m, i): # y_h(t): investment in human capital i_h(t)
        return m.i_h_0 + sum(K_tilde[i, j] * m.alpha_i_h[j] for j in m.I)
    
    def c(m, i): #y_c(t): consumption, c(t)
        return m.c_0 + sum(K_tilde[i, j] * m.alpha_c[j] for j in m.I)
    
    def mu(m, i): #\mu(t): co-state variable
        return m.mu_0 + sum(K_tilde[i, j] * m.alpha_mu[j] for j in m.I)

    def b_k(m, i): #the transversality variable for physical capital
        return m.b_k_0 + sum(K_tilde[i, j] * m.alpha_b_k[j] for j in m.I)
    
    def b_h(m, i): #the transversality variable for human capital
        return m.b_h_0 + sum(K_tilde[i, j] * m.alpha_b_h[j] for j in m.I)

    # defining the derivatives of the variables
    def dk_dt(m, i):  # \dot{x}_k(t) : derivative of the physical capital
        return sum(K[i, j] * m.alpha_k[j] for j in m.I)
    
    def dh_dt(m, i): # \dot{x}_h(t) : derivative of the human capital
        return sum(K[i, j] * m.alpha_h[j] for j in m.I)
    
    def dmu_dt(m, i):
        return sum(K[i, j] * m.alpha_mu[j] for j in m.I)

    

    # defining  constraints and objective for model and solve
    @m.Constraint(m.I)  # for each index in m.I
    def dk_dt_constraint(m, i):
        return dk_dt(m, i) == i_k(m, i) - delta_k * k(m, i)
    
    @m.Constraint(m.I)  # for each index in m.I
    def dh_dt_constraint(m, i):
        return dh_dt(m, i) == i_h(m, i) - delta_h * h(m, i)

    @m.Constraint(m.I)  # for each index in m.I
    def dmu_dt_constraint(m, i):
        return dmu_dt(m, i) == -mu(m, i) * (f_k(k(m, i), h(m, i)) - delta_k - rho_hat)
    
    @m.Constraint(m.I)  # for each index in m.I
    def no_arbitrage(m, i):
        return 0.0 == f_h(k(m, i), h(m, i)) - f_k(k(m, i), h(m, i)) - delta_h + delta_k
    
    @m.Constraint(m.I)  # for each index in m.I
    def feasibility(m, i):
        return 0.0 == c(m, i) + i_h(m, i) + i_k(m, i) - f(k(m, i), h(m, i))
    
    @m.Constraint(m.I)  # for each index in m.I
    def shadow_price(m, i):
        return mu(m, i) * c(m, i) -1.0 == 0.0
    
    @m.Constraint(m.I)  # for each index in m.I
    def b_k_constraint(m, i):
        return mu(m, i) * k(m, i) == b_k(m, i)
    
    @m.Constraint(m.I)  # for each index in m.I
    def b_h_constraint(m, i):
        return mu(m, i) * h(m, i) == b_h(m, i)

    @m.Objective(sense=pyo.minimize)
    def min_norm(m):  # alpha @ K @ alpha not supported by pyomo
        return lambda_b_k * sum(K[i, j] * m.alpha_b_k[i] * m.alpha_b_k[j] for i in m.I for j in m.I) + lambda_b_h * sum(K[i, j] * m.alpha_b_h[i] * m.alpha_b_h[j] for i in m.I for j in m.I)

    solver = pyo.SolverFactory(solver_type)
    options = (
        {
            'bound_relax_factor': 0,
        }
    )  # can add options here.   See https://coin-or.github.io/Ipopt/OPTIONS.html#OPTIONS_AMPL
    results = solver.solve(m, tee=verbose, options=options)
    if not results.solver.termination_condition == TerminationCondition.optimal:
        print(str(results.solver))  # raise exception?

    alpha_c = jnp.array([pyo.value(m.alpha_c[i]) for i in m.I])
    alpha_k = jnp.array([pyo.value(m.alpha_k[i]) for i in m.I])
    alpha_h = jnp.array([pyo.value(m.alpha_h[i]) for i in m.I])
    alpha_i_k = jnp.array([pyo.value(m.alpha_i_k[i]) for i in m.I])
    alpha_i_h = jnp.array([pyo.value(m.alpha_i_h[i]) for i in m.I])
    alpha_mu = jnp.array([pyo.value(m.alpha_mu[i]) for i in m.I])

    c_0 = pyo.value(m.c_0)
    i_k_0 = pyo.value(m.i_k_0)
    i_h_0 = pyo.value(m.i_h_0)
    mu_0 = pyo.value(m.mu_0)
    # Interpolator using training data
    #@jax.jit
    def kernel_solution(test_data):
        # pointwise comparison test_data to train_data
        K_test, K_tilde_test = integrated_matern_kernel_matrices(
            test_data, train_data, nu, sigma, rho
        )
        c_test = c_0 + K_tilde_test @ alpha_c
        k_test = k_0 + K_tilde_test @ alpha_k
        h_test = h_0 + K_tilde_test @ alpha_h
        i_k_test = i_k_0 + K_tilde_test @ alpha_i_k
        i_h_test = i_h_0 + K_tilde_test @ alpha_i_h
        mu_test = mu_0 + K_tilde_test @ alpha_mu
        feasibility_test = c_test + i_h_test + i_k_test - f(k_test, h_test)
        return k_test, h_test, c_test, i_k_test, i_h_test, mu_test, feasibility_test

    # Generate test_data
    k_test, h_test, c_test, i_k_test, i_h_test, mu_test, feasibility_test = kernel_solution(test_data)

    print(
        f"solve_time(s) = {results.solver.Time}"
    )
    return {
        "t_train": train_data,
        "t_test": test_data,
        "k_test": k_test,
        "h_test": h_test,
        "c_test": c_test,
        "i_k_test": i_k_test,
        "i_h_test": i_h_test,
        "mu_test": mu_test,
        "feasibility_test": feasibility_test,
        "alpha_c": alpha_c,
        "alpha_k": alpha_k,
        "alpha_h": alpha_h,
        "alpha_i_k": alpha_i_k,
        "alpha_i_h": alpha_i_h,
        "c_0": c_0,
        "i_k_0": i_k_0,
        "i_h_0": i_h_0,
        "solve_time": results.solver.Time,
        "kernel_solution": kernel_solution,  # interpolator
    }


if __name__ == "__main__":
    jsonargparse.CLI(human_capital_matern)
