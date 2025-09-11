function matern_kernel_0p5(t_i, t_j; sigma, rho)
    d = abs(t_i - t_j)
    return sigma^2 * exp(-d / rho)
end