module KernelEconExamples
using Distributions, QuadGK

include("kernels.jl")
export matern_kernel_0p5, 
    matern_kernel_1p5,
    matern_kernel_2p5,
    matern_kernel_inf,
    integrated_matern_kernel_0p5,
    integrated_matern_kernel_1p5,
    integrated_matern_kernel_2p5,
    integrated_matern_kernel_inf
end
