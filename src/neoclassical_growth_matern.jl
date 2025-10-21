using JuMP
using Ipopt
using Statistics
# JuMP nonlinear optimization is incomplete

# Broadcasted code for nonlinear constraints doesn't work well with JuMPs nonlinear auto-diff implementation
# Objective: minimize alpha' * K * alpha 
# @objective(model, Min, 
#     alpha_mu' * K * alpha_mu + 
#     alpha_k' * K * alpha_k + 

# mu = mu_0 .+ K_tilde * alpha_mu
# c = c_0 .+ K_tilde * alpha_c
# k = k_0 .+ K_tilde * alpha_k
# dmu_dt = K * alpha_mu
# dk_dt = K * alpha_k

# # Resource constraint: dk/dt = k^a - delta*k - c
# @constraint(model, dk_dt .== k.^a .- delta .* k .- c)

# # Euler equation: dmu/dt = -mu * (a*k^(a-1) - delta - rho_hat)
# @constraint(model, dmu_dt .== -mu .* (a .* k.^(a-1) .- delta .- rho_hat))

# # Shadow price: mu * c = 1
# @constraint(model, mu .* c .== 1.0)       

function neoclassical_growth_matern(;
    a=1/3,
    delta=0.1,
    rho_hat=0.11,
    k_0=1.0,
    nu=0.5,
    sigma=1.0,
    rho=10.0,
    train_T=40.0,
    train_points=41,
    test_T=50.0,
    test_points=100,
    baseline_T=60.0,
    lambda_p=0.0,
    verbose=false,
    tol=1e-8,
    dual_inf_tol=1e-8,
    constr_viol_tol=1e-8,
    max_iter=2000
)
    # Setup training and test data
    train_data = range(0, train_T, length=train_points)
    test_data = range(0, test_T, length=test_points)
    
    # Construct kernel matrices using nu=0.5
    K, K_tilde = matrices_matern_kernel_0p5(train_data, train_data; sigma, rho)
    K = Symmetric((K + K')/2) # symmetric since train_data both arguments. K_tilde not symmetric
    N = length(train_data)
    
    # Create JuMP model with Ipopt (non-convex problem)
    model = Model(Ipopt.Optimizer)
    if !verbose
        set_silent(model)
    end
    set_attribute(model, "tol", tol)
    set_attribute(model, "dual_inf_tol", dual_inf_tol)
    set_attribute(model, "constr_viol_tol", constr_viol_tol)
    set_attribute(model, "max_iter", max_iter)

    @variable(model, alpha_mu[1:N])
    @variable(model, alpha_c[1:N])
    @variable(model, alpha_k[1:N])
    @variable(model, c_0 >= 0, start = k_0^a - delta * k_0)
    @variable(model, mu_0 >= 0, start = k_0^a - delta * k_0)

    @objective(model, Min,
        dot(alpha_mu, K * alpha_mu) +
        dot(alpha_k,  K * alpha_k)
    )    

    # Building blocks to preserve sparsity in JuMP's nonlinear model
    # Create auxiliary variables for linear transformations of alpha coefficients
    @variable(model, dk_dt[1:N])      # dk_dt = K * alpha_k (derivative via kernel)
    @variable(model, dmu_dt[1:N])     # dmu_dt = K * alpha_mu (derivative via kernel)
    @variable(model, k_tilde[1:N])    # k_tilde = K_tilde * alpha_k (integrated kernel)
    @variable(model, mu_tilde[1:N])   # mu_tilde = K_tilde * alpha_mu (integrated kernel)
    @variable(model, c_tilde[1:N])    # c_tilde = K_tilde * alpha_c (integrated kernel)
    
    # Linear equality constraints define these auxiliary variables
    @constraint(model, dk_dt .== K * alpha_k)
    @constraint(model, dmu_dt .== K * alpha_mu)
    @constraint(model, k_tilde .== K_tilde * alpha_k)
    @constraint(model, mu_tilde .== K_tilde * alpha_mu)
    @constraint(model, c_tilde .== K_tilde * alpha_c)

    # Nonlinear expressions for state and control variables
    @NLexpression(model, mu[i=1:N], mu_0 + mu_tilde[i])
    @NLexpression(model, c[i=1:N],  c_0 + c_tilde[i])
    @NLexpression(model, k[i=1:N],  k_0 + k_tilde[i])

    # System constraints (resource, Euler, shadow price)
    @NLconstraint(model, [i=1:N], dk_dt[i] == k[i]^a - delta * k[i] - c[i])
    @NLconstraint(model, [i=1:N], dmu_dt[i] == -mu[i] * (a * k[i]^(a-1) - delta - rho_hat))
    @NLconstraint(model, [i=1:N], mu[i] * c[i] == 1.0)

    println("Solving Neoclassical Growth Matern with N=$N")
    optimize!(model)
    
    alpha_c_val = value.(alpha_c)
    alpha_k_val = value.(alpha_k)
    c_0_val = value(c_0)
    solve_time_sec = solve_time(model)
    
    # Kernel solution interpolator
    function kernel_solution(t_test)
        _, K_tilde_test = matrices_matern_kernel_0p5(t_test, train_data; sigma, rho)
        c_test = c_0_val .+ K_tilde_test * alpha_c_val
        k_test = k_0 .+ K_tilde_test * alpha_k_val
        return k_test, c_test
    end
    
    # Baseline solution
    sol_baseline = neoclassical_growth_baseline(a, delta, rho_hat, 1.0, k_0, baseline_T)
    
    function baseline_solution(t_test)
        sol = sol_baseline.(t_test)
        k_baseline = [s[1] for s in sol]
        c_baseline = [s[2] for s in sol]
        return k_baseline, c_baseline
    end
    
    # Evaluate on test data
    k_baseline, c_baseline = baseline_solution(test_data)
    k_test, c_test = kernel_solution(test_data)
    
    k_rel_error = abs.(k_baseline .- k_test) ./ k_baseline
    c_rel_error = abs.(c_baseline .- c_test) ./ c_baseline
    
    println("solve_time(s) = $solve_time_sec, E(|rel_error(k)|) = $(mean(k_rel_error)), E(|rel_error(c)|) = $(mean(c_rel_error))")
    
    return (; t_train=train_data, t_test=test_data, k_test, c_test, 
              k_baseline, c_baseline, k_rel_error, c_rel_error,
              alpha_c=alpha_c_val, alpha_k=alpha_k_val, c_0=c_0_val,
              solve_time=solve_time_sec, kernel_solution, baseline_solution)
end
