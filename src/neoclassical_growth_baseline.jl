using DifferentialEquations
using BoundaryValueDiffEq

function neoclassical_growth_baseline(a, delta, r, sigma_crra, k_0, T_max; perturb_k=1e-4, dt=0.1)
    k_ss = ((delta + r) / a)^(1 / (a - 1))
    c_ss = a * k_ss^a - delta * k_ss
    k_T = k_ss - perturb_k
    
    function ode!(dy, y, p, t)
        k, c = y
        dy[1] = k^a - c - delta * k
        dy[2] = (c / sigma_crra) * (a * k^(a - 1) - r - delta)
    end
    
    function bc!(residual, y, p, t)
        residual[1] = y[1][1] - k_0
        residual[2] = y[end][1] - k_T
    end
    
    tspan = (0.0, T_max)
    initial_guess = [k_ss, c_ss]
    bvp = BVProblem(ode!, bc!, initial_guess, tspan)
    
    return solve(bvp, MIRK4(); dt)
end
