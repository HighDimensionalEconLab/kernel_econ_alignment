using Test
using KernelEconExamples
using Statistics

@testset "Neoclassical Growth Matern" begin
    @testset "Exact Python Defaults" begin
        # Test with exact Python default parameters
        result = neoclassical_growth_matern(
            a=1/3,
            delta=0.1,
            rho_hat=0.11,
            k_0=1.0,
            nu=0.5,
            sigma=1.0,
            rho=10.0,
            train_T=40.0,
            train_points=41,
            test_T=50.0,
            test_points=100,
            baseline_T=60.0,
            baseline_points=300,
            lambda_p=0.0,
            verbose=false
        )
        
        # Test c_0 value (relaxed due to solver differences)
        @test result.c_0 ≈ 0.691 atol=0.01
        
        # Test mean relative errors (primary validation)
        @test mean(result.k_rel_error) < 0.001  # Should be around 0.0004
        @test mean(result.c_rel_error) < 0.005  # Should be around 0.003
        
        # Test that we have the correct number of points
        @test length(result.t_train) == 41
        @test length(result.t_test) == 100
        @test length(result.alpha_c) == 41
        @test length(result.alpha_k) == 41
        
        # Test that solve time is positive
        @test result.solve_time > 0
        
        # Test that kernel_solution is callable
        @test result.kernel_solution isa Function
        k_interp, c_interp = result.kernel_solution([0.0, 1.0])
        @test length(k_interp) == 2
        @test length(c_interp) == 2
    end
end
