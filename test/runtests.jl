using KernelEconExamples
using Test

@testset "matern_kernel_0p5" begin
    # AI Generated... not sure they are relevant tests
    @test matern_kernel_0p5(0.0, 1.0; sigma = 1.0, rho = 1.0) ≈ exp(-1)
    @test matern_kernel_0p5(1.0, 1.0; sigma = 2.0, rho = 1.0) ≈ 4.0
end
