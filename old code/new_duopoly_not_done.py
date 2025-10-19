import numpy as np
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition
import jsonargparse
from kernels import integrated_matern_kernel_matrices
from typing import List, Optional



def duopoly_matern(
    A: float = 0.05, #adjustment cost coeff
    a: float = 1.0,
    b: float = 0.0, # the coefficient of quadratic cost
    c: float = 0.1, # marginal cost
    rho_hat: float = 0.11, # discount rate
    u_1_0: float = 0.01, # the initial condition for the first firm production
    u_2_0: float = 0.02, # the initial condition for the first firm production
    nu: float = 0.5,
    sigma: float = 1.0,
    rho: float = 20,
    solver_type: str = "ipopt",
    train_T: float = 40.0,
    train_points: int = 81,
    test_T: float = 50.0,
    test_points: int = 100,
    train_points_list: Optional[List[float]] = None,
    verbose: bool = False,
):
    # if passing in `train_points` then doesn't us a grid.  Otherwise, uses linspace
    if train_points_list is None:
        train_data = np.linspace(0, train_T, train_points)
    else:
        train_data = np.array(train_points_list)
    test_data = np.linspace(0, test_T, test_points)

    # Construct kernel matrices
    N = len(train_data)
    K, K_tilde = integrated_matern_kernel_matrices(
        train_data, train_data, nu, sigma, rho
    )
    K = np.array(K)  
    K_tilde = np.array(K_tilde)

    # Create pyomo model and variables
    m = pyo.ConcreteModel()
    m.I = range(N)
    # 8 variables
    m.alpha_mu_1_2 = pyo.Var(m.I, within=pyo.Reals, initialize = 0.0) # coeffs for $\mu_{12}$
    m.alpha_mu_2_2 = pyo.Var(m.I, within=pyo.Reals, initialize = 0.0) # coeffs for $\mu_{22}$
    m.alpha_u_1 = pyo.Var(m.I, within=pyo.Reals, initialize = 0.0) # coeffs for $x_{1}$: production of firm one 
    m.alpha_u_2 = pyo.Var(m.I, within=pyo.Reals, initialize = 0.0) # coeffs for $y_{2}$: production of firm two 

# Initialization of the variables
    m.alpha_mu_1_2_0 = pyo.Var(within=pyo.Reals, initialize = 0.0) # $\mu_{12}(0)$
    m.alpha_mu_2_2_0 = pyo.Var(within=pyo.Reals, initialize = 0.0) # $\mu_{22}(0)$

    # coeffiecients of the assumed policy 
    m.q_0 = pyo.Var(within=pyo.Reals, initialize = 0.1) # $q_0$
    m.q_1 = pyo.Var(within=pyo.Reals, initialize = -0.1) # $q_1$
    m.q_2 = pyo.Var(within=pyo.Reals, initialize = 0.01) # $q_2$



    # defining the integrated kernel functional approximation

    #def mu_11(m, i): #\mu_{11}(t): co-state variable
    #    return m.alpha_mu_11_0 + sum(K_tilde[i, j] * m.alpha_mu_11[j] for j in m.I)
    
    def mu_1_2(m, i): #\mu_{12}(t): co-state variable
        return m.alpha_mu_1_2_0 + sum(K_tilde[i, j] * m.alpha_mu_1_2[j] for j in m.I)


    def mu_2_2(m, i): #\mu_{12}(t): co-state variable
        return m.alpha_mu_2_2_0 + sum(K_tilde[i, j] * m.alpha_mu_2_2[j] for j in m.I)


    def u_1(m, i): #y_1(t): adjustment, derivative of x_1
        return u_1_0 + sum(K_tilde[i, j] * m.alpha_u_1[j] for j in m.I)
    
    def u_2(m, i): #y_2(t): adjustment, derivative of x_2
        return u_2_0 + sum(K_tilde[i, j] * m.alpha_u_2[j] for j in m.I)
    
    


    
    def dmu_1_2_dt(m, i):  # \dot{\mu_{12}} : derivative of $\mu_{12}(t)$
        return sum(K[i, j] * m.alpha_mu_1_2[j] for j in m.I)
    
    
    def dmu_2_2_dt(m, i):  # \dot{\mu_{22}} : derivative of $\mu_{22}(t)$
        return sum(K[i, j] * m.alpha_mu_2_2[j] for j in m.I)

    def du_1_dt(m, i): # \dot{x_1} : derivative of the firm  one production
        return sum(K[i, j] * m.alpha_u_1[j] for j in m.I)
    
    def du_2_dt(m, i): # \dot{x_2} : derivative of the firm  two production
        return sum(K[i, j] * m.alpha_u_2[j] for j in m.I)
    



    

    # defining  constraints and objective for model and solve
    #Agent 1 constraints
    
    @m.Constraint(m.I)  #2
    def dx_1_dt_constraint(m, i):
        return A*(m.q_1*du_1_dt(m, i) + m.q_2*du_2_dt(m, i)) == rho_hat*A*(m.q_0 +  m.q_1*u_1(m, i) + m.q_2*u_2(m, i)) - a + 2*u_1(m, i) + u_2(m, i) + c + b*u_1(m, i) - m.q_2*mu_1_2(m, i)

    @m.Constraint(m.I)  # 5
    def dmu_1_2_dt_constraint(m, i):
        return dmu_1_2_dt(m, i) == rho_hat*mu_1_2(m, i) + u_1(m, i) - m.q_1*mu_1_2(m, i)
    

    #Agent 2 constraints
    @m.Constraint(m.I)  # 7
    def dx_2_dt_constraint(m, i):
        return A*(m.q_1*du_2_dt(m, i) + m.q_2*du_1_dt(m, i)) == A*rho_hat*(m.q_0+ m.q_1*u_2(m, i) + m.q_2*u_1(m, i)) - a + u_1(m, i) + 2*u_2(m, i) + c + b*u_2(m, i) - m.q_2*mu_2_2(m, i)

    @m.Constraint(m.I)  # for each index in m.I
    def dmu_2_2_dt_constraint(m, i):
        return dmu_2_2_dt(m, i) == rho_hat*mu_2_2(m, i) + u_2(m, i) - m.q_1*mu_2_2(m, i)
    
    
   

    @m.Objective(sense=pyo.minimize)
    def min_norm(m):  # alpha @ K @ alpha not supported by pyomo
        return (sum(K[i, j] * m.alpha_mu_1_2[i] * m.alpha_mu_1_2[j] for i in m.I for j in m.I) + 
        sum(K[i, j] * m.alpha_mu_2_2[i] * m.alpha_mu_2_2[j] for i in m.I for j in m.I)  + 0.0001*(sum(K[i, j] * m.alpha_u_1[i] * m.alpha_u_1[j] for i in m.I for j in m.I)+
        sum(K[i, j] * m.alpha_u_2[i] * m.alpha_u_2[j] for i in m.I for j in m.I)
        )
        ) 
    solver = pyo.SolverFactory(solver_type)
    options = (
        {
            'bound_relax_factor': 0,
        }
    )  # can add options here.   See https://coin-or.github.io/Ipopt/OPTIONS.html#OPTIONS_AMPL
    results = solver.solve(m, tee=verbose, options=options)
    if not results.solver.termination_condition == TerminationCondition.optimal:
        print(str(results.solver))  # raise exception?

    alpha_u_1 = np.array([pyo.value(m.alpha_u_1[i]) for i in m.I])
    alpha_u_2 = np.array([pyo.value(m.alpha_u_2[i]) for i in m.I])
    q_0 = pyo.value(m.q_0)
    q_1 = pyo.value(m.q_1)
    q_2 = pyo.value(m.q_2)



 
    def kernel_solution(test_data):
        # pointwise comparison test_data to train_data
        K_test, K_tilde_test = integrated_matern_kernel_matrices(
            test_data, train_data, nu, sigma, rho
        )
        u_1_test = u_1_0 + K_tilde_test @ alpha_u_1
        u_2_test = u_2_0 + K_tilde_test @ alpha_u_2
        x_1_test = q_0 + q_1*u_1_test + q_2*u_2_test
        x_2_test = q_0 + q_2*u_1_test + q_1*u_2_test
        mu_1_1_test = A*x_1_test
        mu_2_1_test = A*x_1_test
        return u_1_test, u_2_test
    # Generate test_data
    u_1_test, u_2_test = kernel_solution(test_data)

    print(
        f"solve_time(s) = {results.solver.Time}"
    )
    return {
        "t_train": train_data,
        "t_test": test_data,
        "u_1_test": u_1_test,
        "u_2_test": u_2_test,
        "alpha_u_1": alpha_u_1,
        "alpha_u_1": alpha_u_2,
        "kernel_solution": kernel_solution,  # interpolator
    }


if __name__ == "__main__":
    jsonargparse.CLI(duopoly_matern)
