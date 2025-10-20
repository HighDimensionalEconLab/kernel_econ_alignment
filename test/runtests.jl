using KernelEconExamples
using Test

@testset "matern_kernel_0p5" begin
    # AI Generated... not sure they are relevant tests
    #@test matern_kernel_0p5(0.0, 1.0; sigma = 1.0, rho = 1.0) ≈ exp(-1)
    #@test matern_kernel_0p5(1.0, 1.0; sigma = 2.0, rho = 1.0) ≈ 4.0
    @test matern_kernel_0p5(0.0, 50.0; sigma = 1.0, rho = 10.0) ≈ 0.00673795 rtol=1e-6
    @test matern_kernel_0p5(0.0, 40.0; sigma = 1.5, rho = 11.0) ≈ 0.05928295 rtol=1e-6
end

@testset "matern_kernel_1p5" begin
    @test matern_kernel_1p5(0.0, 50.0; sigma = 1.0, rho = 10.0) ≈ 0.00167451 rtol=1e-6
    @test matern_kernel_1p5(0.0, 40.0; sigma = 1.5, rho = 11.0) ≈ 0.03020387 rtol=1e-6
end

@testset "matern_kernel_2p5" begin
    @test matern_kernel_2p5(0.0, 50.0; sigma = 1.0, rho = 15.0) ≈ 0.01562696 rtol=1e-6
    @test matern_kernel_2p5(0.0, 40.0; sigma = 1.5, rho = 11.0) ≈ 0.02063476 rtol=1e-6
end

@testset "matern_kernel_inf" begin
    @test matern_kernel_inf(0.0, 50.0; sigma = 1.0, rho = 15.0) ≈ 0.00386592 rtol=1e-6
    @test matern_kernel_inf(0.0, 40.0; sigma = 1.5, rho = 11.0) ≈ 0.00302562 rtol=1e-6
end

@testset "integrated_matern_kernel_0p5" begin
    @test integrated_matern_kernel_0p5(1.0, 50.0; sigma = 1.0, rho = 15.0) ≈ 0.03688982 rtol=1e-5
    @test integrated_matern_kernel_0p5(1.0, 40.0; sigma = 1.5, rho = 11.0) ≈ 0.06206109 rtol=1e-5
end

@testset "integrated_matern_kernel_1p5" begin
    @test integrated_matern_kernel_1p5(1.0, 50.0; sigma = 1.0, rho = 15.0) ≈ 0.02212782 rtol=1e-6
    @test integrated_matern_kernel_1p5(1.0, 40.0; sigma = 1.5, rho = 11.0) ≈ 0.03234953 rtol=1e-6
end

@testset "integrated_matern_kernel_2p5" begin
    @test integrated_matern_kernel_2p5(1.0, 50.0; sigma = 1.0, rho = 15.0) ≈ 0.01656854 rtol=1e-6
    @test integrated_matern_kernel_2p5(1.0, 40.0; sigma = 1.5, rho = 11.0) ≈ 0.02238993 rtol=1e-6
end

@testset "integrated_matern_kernel_inf" begin
    @test integrated_matern_kernel_inf(1.0, 50.0; sigma = 1.0, rho = 15.0) ≈ 0.00432575 rtol=1e-5
    @test integrated_matern_kernel_inf(1.0, 40.0; sigma = 1.5, rho = 11.0) ≈ 0.00358036 rtol=1e-5
end

