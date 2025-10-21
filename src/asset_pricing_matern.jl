using JuMP
using Ipopt

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
    verbose=false
)
    # Setup training and test data
    train_data = range(0, train_T, length=train_points)
    test_data = range(0, test_T, length=test_points)
    
    # Construct kernel matrices using nu=0.5
    K, K_tilde = matrices_matern_kernel_0p5(train_data, train_data; sigma, rho)
    N = length(train_data)
    
    # x(t) solution
    x = (x_0 + c/g) * exp.(g * train_data) .- c/g
    
    model = Model(Ipopt.Optimizer)
    if !verbose
        set_silent(model)
    end
    
    set_attribute(model, "tol", 1e-14)
    set_attribute(model, "dual_inf_tol", 1e-14)
    set_attribute(model, "constr_viol_tol", 1e-14)
    set_attribute(model, "max_iter", 5000)
    
    @variable(model, alpha_mu[1:N])
    @variable(model, mu_0 >= 0)
    
    # Objective: minimize alpha' * K * alpha
    @objective(model, Min, alpha_mu' * K * alpha_mu)
    
    # Constraints: dp/dt = r*p - x(t)
    # where p(t) = mu_0 + K_tilde * alpha and dp/dt = K * alpha
    @constraint(model, K * alpha_mu .== r * (mu_0 .+ K_tilde * alpha_mu) .- x)
    
    optimize!(model)
    
    alpha_mu_sol = value.(alpha_mu)
    mu_0_sol = value(mu_0)
    solve_time_sec = solve_time(model)
    
    # Kernel solution interpolator
    function kernel_solution(t_test)
        _, K_tilde_test = matrices_matern_kernel_0p5(t_test, train_data; sigma, rho)
        return mu_0_sol .+ K_tilde_test * alpha_mu_sol
    end
    
    # Evaluate on test data
    mu_benchmark = asset_pricing_baseline(test_data, c, g, r, x_0)
    mu_test = kernel_solution(test_data)
    mu_rel_error = abs.(mu_benchmark .- mu_test) ./ mu_benchmark
    
    println("solve_time(s) = $solve_time_sec, E(|rel_error(p)|) = $(sum(mu_rel_error)/length(mu_rel_error))")
    
    return (
        t_train = collect(train_data),
        t_test = collect(test_data),
        p_test = mu_test,
        p_benchmark = mu_benchmark,
        p_rel_error = mu_rel_error,
        alpha = alpha_mu_sol,
        p_0 = mu_0_sol,
        solve_time = solve_time_sec,
        kernel_solution = kernel_solution
    )
end
