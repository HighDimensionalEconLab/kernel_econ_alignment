using JuMP
using OSQP
using Statistics

function asset_pricing_matern(;
    r=0.1,
    c=0.02,
    g=-0.2,
    x_0=0.01,
    sigma=1.0,
    rho=10.0,
    train_T=40.0,
    train_points=41,
    test_T=50.0,
    test_points=100,
    verbose=false,
    eps_abs = 1e-12,
    eps_rel = 1e-12,
    max_iter = 5000
)
    # Setup training and test data
    train_data = range(0, train_T, length=train_points)
    test_data = range(0, test_T, length=test_points)
    
    # Construct kernel matrices using nu=0.5
    K, K_tilde = matrices_matern_kernel_0p5(train_data, train_data; sigma, rho)
    K = Symmetric((K + K')/2) 
    N = length(train_data)
    
    # x(t) solution
    x = (x_0 + c/g) * exp.(g * train_data) .- c/g
    
    # OSQP is ~35x faster than Ipopt for this convex QP
    # To use Ipopt instead: Model(Ipopt.Optimizer) with attributes:
    #   "tol", "dual_inf_tol", "constr_viol_tol", "max_iter"
    model = Model(OSQP.Optimizer)
    if !verbose
        set_silent(model)
    end
    set_attribute(model, "eps_abs", eps_abs)
    set_attribute(model, "eps_rel", eps_rel)
    set_attribute(model, "max_iter", max_iter)

    @variable(model, alpha_mu[1:N])
    @variable(model, mu_0 >= 0)
    
    # Objective: minimize alpha' * K * alpha (using dot for symmetric K)
    @objective(model, Min, dot(alpha_mu, K * alpha_mu))
    
    # Constraints: dp/dt = r*p - x(t)
    # where p(t) = mu_0 + K_tilde * alpha and dp/dt = K * alpha
    @constraint(model, K * alpha_mu .== r * (mu_0 .+ K_tilde * alpha_mu) .- x)
    
    optimize!(model)
    
    alpha = value.(alpha_mu)
    p_0 = value(mu_0)
    solve_time_sec = solve_time(model)
    
    # Kernel solution interpolator
    function kernel_solution(t_test)
        _, K_tilde_test = matrices_matern_kernel_0p5(t_test, train_data; sigma, rho)
        return p_0 .+ K_tilde_test * alpha
    end
    
    # Evaluate on test data
    p_baseline = asset_pricing_baseline(test_data, c, g, r, x_0)
    p_test = kernel_solution(test_data)
    p_rel_error = abs.(p_baseline .- p_test) ./ p_baseline
    
    println("solve_time(s) = $solve_time_sec, E(|rel_error(p)|) = $(mean(p_rel_error))")
    
    return (; t_train=train_data, t_test=test_data, p_test, p_baseline, p_rel_error, 
              alpha, p_0, solve_time=solve_time_sec, kernel_solution)
end
