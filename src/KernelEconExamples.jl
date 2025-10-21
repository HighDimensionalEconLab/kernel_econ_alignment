module KernelEconExamples
using Distributions, QuadGK
using DifferentialEquations, BoundaryValueDiffEq

include("kernels.jl")
include("neoclassical_growth_baseline.jl")
include("asset_pricing_baseline.jl")

export matern_kernel_0p5, 
    matern_kernel_1p5,
    matern_kernel_2p5,
    matern_kernel_inf,
    integrated_matern_kernel_0p5,
    integrated_matern_kernel_1p5,
    integrated_matern_kernel_2p5,
    integrated_matern_kernel_inf,
    neoclassical_growth_baseline,
    asset_pricing_baseline
end
