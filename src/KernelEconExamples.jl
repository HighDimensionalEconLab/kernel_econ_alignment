module KernelEconExamples
using Distributions, QuadGK
using DifferentialEquations, BoundaryValueDiffEq
using JuMP, OSQP

include("kernels.jl")
include("neoclassical_growth_baseline.jl")
include("asset_pricing_baseline.jl")
include("asset_pricing_matern.jl")

export matern_kernel_0p5, 
    matern_kernel_1p5,
    matern_kernel_2p5,
    matern_kernel_inf,
    integrated_matern_kernel_0p5,
    integrated_matern_kernel_1p5,
    integrated_matern_kernel_2p5,
    integrated_matern_kernel_inf,
    matrices_matern_kernel_0p5,
    neoclassical_growth_baseline,
    asset_pricing_baseline,
    asset_pricing_matern
end
