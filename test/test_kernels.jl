using KernelEconExamples
using Test

@testset "matern_kernel_0p5" begin
    @test matern_kernel_0p5(25.0, 25.0; sigma = 1.5, rho = 11.0) ≈ 2.25 rtol=1e-6
    @test matern_kernel_0p5(0.0, 50.0; sigma = 1.0, rho = 10.0) ≈ 0.00673795 rtol=1e-6
    @test matern_kernel_0p5(1.0, 40.0; sigma = 1.5, rho = 11.0) ≈ 0.06492489 rtol=1e-6
end

@testset "matern_kernel_1p5" begin
    @test matern_kernel_1p5(25.0, 25.0; sigma = 1.5, rho = 11.0) ≈ 2.25 rtol=1e-6
    @test matern_kernel_1p5(0.0, 50.0; sigma = 1.0, rho = 10.0) ≈ 0.00167451 rtol=1e-6
    @test matern_kernel_1p5(1.0, 40.0; sigma = 1.5, rho = 11.0) ≈ 0.034591846 rtol=1e-6
end

@testset "matern_kernel_2p5" begin
    @test matern_kernel_2p5(25.0, 25.0; sigma = 1.0, rho = 15.0) ≈ 1.0 rtol=1e-6
    @test matern_kernel_2p5(0.0, 50.0; sigma = 1.0, rho = 15.0) ≈ 0.01562696 rtol=1e-6
    @test matern_kernel_2p5(1.0, 40.0; sigma = 1.5, rho = 11.0) ≈ 0.024238449 rtol=1e-6
end

@testset "matern_kernel_inf" begin
    @test matern_kernel_inf(25.0, 25.0; sigma = 1.0, rho = 15.0) ≈ 1.0 rtol=1e-6
    @test matern_kernel_inf(0.0, 50.0; sigma = 1.0, rho = 15.0) ≈ 0.00386592 rtol=1e-6
    @test matern_kernel_inf(1.0, 40.0; sigma = 1.5, rho = 11.0) ≈ 0.004193609 rtol=1e-6
end

@testset "integrated_matern_kernel_0p5" begin
    @test integrated_matern_kernel_0p5(25.0, 25.0; sigma = 1.0, rho = 15.0) ≈ 12.166864 rtol=1e-5
    @test integrated_matern_kernel_0p5(0.0, 50.0; sigma = 1.0, rho = 15.0) ≈ 0.0 
    @test integrated_matern_kernel_0p5(1.0, 40.0; sigma = 1.5, rho = 11.0) ≈ 0.06206109 rtol=1e-5
end

@testset "integrated_matern_kernel_1p5" begin
    @test integrated_matern_kernel_1p5(25.0, 25.0; sigma = 1.0, rho = 15.0) ≈ 14.96084 rtol=1e-5
    @test integrated_matern_kernel_1p5(0.0, 50.0; sigma = 1.0, rho = 15.0) ≈ 0.0 
    @test integrated_matern_kernel_1p5(1.0, 40.0; sigma = 1.5, rho = 11.0) ≈ 0.03234953 rtol=1e-6
end

@testset "integrated_matern_kernel_2p5" begin
    @test integrated_matern_kernel_2p5(25.0, 25.0; sigma = 1.0, rho = 15.0) ≈ 15.7074995 rtol=1e-5
    @test integrated_matern_kernel_2p5(0.0, 50.0; sigma = 1.0, rho = 15.0) ≈ 0.0 
    @test integrated_matern_kernel_2p5(1.0, 40.0; sigma = 1.5, rho = 11.0) ≈ 0.02238993 rtol=1e-6
end

@testset "integrated_matern_kernel_inf" begin
    @test integrated_matern_kernel_inf(25.0, 25.0; sigma = 1.0, rho = 15.0) ≈ 17.002823 rtol=1e-5
    @test integrated_matern_kernel_inf(0.0, 50.0; sigma = 1.0, rho = 15.0) ≈ 0.0 
    @test integrated_matern_kernel_inf(1.0, 40.0; sigma = 1.5, rho = 11.0) ≈ 0.00358036 rtol=1e-5
end

#matrices_matern_kernel tests
@testset "matrices_matern_kernel_0p5" begin
    t = [1.0, 5.0, 10.0]
    s = [10.0, 20.0, 30.0]
    K, K_tilde = matrices_matern_kernel_0p5(t, s; sigma=1.5, rho=11.0)
    @test K[1, 1] ≈ 0.9927746 rtol=1e-6
    @test K[2, 2] ≈ 0.57539064 rtol=1e-6

    @test K_tilde[1, 1] ≈ 0.9489851 rtol=1e-6             
    @test K_tilde[2, 2] ≈ 2.311862 rtol=1e-6
end

