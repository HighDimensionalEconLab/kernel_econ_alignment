function matern_kernel_0p5(t_i, t_j; sigma, rho)
    d = abs(t_i - t_j)
    return sigma^2 * exp(-d / rho)
end

function matern_kernel_1p5(t_i, t_j; sigma, rho)
    d = abs(t_i - t_j)
    exponent = sqrt(3) * d / rho
    return sigma^2 * (1 + exponent) * exp(-exponent)
end

function matern_kernel_2p5(t_i, t_j; sigma, rho)
    d = abs(t_i - t_j)
    exponent = sqrt(5) * d / rho
    term = (5 * d^2) / (3 * rho^2)
    return sigma^2 * (1 + exponent + term) * exp(-exponent)
end

function matern_kernel_inf(t_i, t_j; sigma, rho)
    d = abs(t_i - t_j)
    exponent = -0.5 * (d^2 / rho^2)
    return sigma^2 * exp(exponent)
end

#Integrated matern kernels
function integrated_matern_kernel_0p5(t_i, t_j; sigma, rho)
    s = t_i - t_j
    d = abs(s)
    base = rho * (sigma^2) 
    if s < 0
        return base * (exp(-d / rho) - exp(-t_j / rho))
    else
        return base * (2 - exp(-d / rho) - exp(-t_j / rho))
    end
end

function integrated_matern_kernel_1p5(t_i, t_j; sigma, rho)
    function matern_integrand(t)
        return matern_kernel_1p5(t, t_j; sigma=sigma, rho=rho)
    end
    integral, info = QuadGK.quadgk(matern_integrand, 0.0, t_i)
    return ifelse(iszero(t_i), zero(integral), integral)
end

function integrated_matern_kernel_2p5(t_i, t_j; sigma, rho)
    function matern_integrand(t)
        return matern_kernel_2p5(t, t_j; sigma=sigma, rho=rho)
    end
    integral, info = QuadGK.quadgk(matern_integrand, 0.0, t_i)
    return ifelse(iszero(t_i), zero(integral), integral)
end

function integrated_matern_kernel_inf(t_i, t_j; sigma, rho)
    function matern_integrand(t)
        return matern_kernel_inf(t, t_j; sigma=sigma, rho=rho)
    end
    integral, info = QuadGK.quadgk(matern_integrand, 0.0, t_i)
    return ifelse(iszero(t_i), zero(integral), integral)
end

function matrices_matern_kernel_0p5(t, s; sigma, rho)
    K = [matern_kernel_0p5(t[i], s[j]; sigma, rho) for i in 1:length(t), j in 1:length(s)]
    K_tilde = [integrated_matern_kernel_0p5(t[i], s[j]; sigma, rho) for i in 1:length(t), j in 1:length(s)]
    return K, K_tilde
end