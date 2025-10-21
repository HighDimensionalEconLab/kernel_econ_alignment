using KernelEconExamples
using Test
using Statistics

@testset "Asset Pricing Matern" begin
    @testset "Exact Python Defaults" begin
        # Use EXACT defaults from Python asset_pricing_matern.py:
        # r=0.1, c=0.02, g=-0.2, x_0=0.01, sigma=1.0, rho=10.0,
        # train_T=40.0, train_points=41, test_T=50.0, test_points=100
        result = asset_pricing_matern(
            r=0.1,
            c=0.02,
            g=-0.2,
            x_0=0.01,
            sigma=1.0,
            rho=10.0,
            train_T=40.0,
            train_points=41,
            test_T=50.0,
            test_points=100,
            verbose=false
        )
        
        # Verify p_0 matches Python: 0.699497563627038
        @test result.p_0 ≈ 0.699497563627038 atol=1e-10
        
        # Verify mean relative error matches Python: ~0.00015457791718911825
        @test mean(result.p_rel_error) < 0.0002
        
        # Check p_test values match Python output
        # Python: [0.69949756 0.72836592 0.75446368 0.7780531  0.79937546]
        @test result.p_test[1] ≈ 0.699497563627038 atol=1e-10
        @test result.p_test[2] ≈ 0.7283659243149404 atol=1e-9
        @test result.p_test[3] ≈ 0.7544636771840599 atol=1e-9
        @test result.p_test[4] ≈ 0.7780530968662679 atol=1e-9
        @test result.p_test[5] ≈ 0.7993754633594473 atol=1e-9
        
        # Check alpha values match Python output
        # Python: [ 0.08571768 -0.00738086 -0.00604294 -0.00494754 -0.0040507 ]
        @test result.alpha[1] ≈ 0.08571768381161791 atol=1e-10
        @test result.alpha[2] ≈ -0.007380863880940638 atol=1e-10
        @test result.alpha[3] ≈ -0.006042940243608704 atol=1e-10
        @test result.alpha[4] ≈ -0.004947541016455121 atol=1e-10
        @test result.alpha[5] ≈ -0.004050703982286589 atol=1e-10
        
        # Check dimensions
        @test length(result.t_train) == 41
        @test length(result.t_test) == 100
        @test length(result.alpha) == 41
        
        # Sanity checks
        @test result.p_0 >= 0
        @test all(isfinite.(result.alpha))
        @test all(isfinite.(result.p_test))
        @test all(isfinite.(result.p_baseline))
    end
    
    @testset "Different Parameters" begin
        # Test with different parameters to ensure robustness
        result = asset_pricing_matern(
            r=0.05,
            c=0.01,
            g=-0.1,
            x_0=0.02,
            sigma=1.0,
            rho=10.0,
            train_T=20.0,
            train_points=11,
            test_T=20.0,
            test_points=50,
            verbose=false
        )
        
        # Basic sanity checks
        @test result.p_0 >= 0
        @test length(result.alpha) == 11
        @test length(result.t_train) == 11
        @test length(result.t_test) == 50
        @test all(isfinite.(result.p_test))
        @test all(isfinite.(result.p_baseline))
        
        # Solution should be reasonably accurate (relaxed tolerance for sparser grid)
        @test mean(result.p_rel_error) < 0.25
    end
    
    @testset "Kernel Solution Interpolator" begin
        result = asset_pricing_matern(
            r=0.1,
            c=0.02,
            g=-0.2,
            x_0=0.01,
            sigma=1.0,
            rho=10.0,
            train_T=40.0,
            train_points=11,
            test_T=50.0,
            test_points=100,
            verbose=false
        )
        
        # Test interpolator at new points
        t_new = [5.0, 15.0, 25.0]
        p_new = result.kernel_solution(t_new)
        
        @test length(p_new) == 3
        @test all(isfinite.(p_new))
        @test all(p_new .> 0)  # Prices should be positive
    end
end
