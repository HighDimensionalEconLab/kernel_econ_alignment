using KernelEconExamples
using Test
using Statistics

@testset "Asset Pricing Baseline" begin
        c = 1.0
        g = 0.02
        r = 0.05
        x_0 = 1.0
        
        # Test vector input with values from Python
        t_values = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
        expected = [700.0, 1076.384765625, 1536.1019287109375, 2097.60205078125, 
                    2783.419677734375, 3621.0791015625]
        
        result = asset_pricing_baseline(t_values, c, g, r, x_0)
        
    @test all(isapprox.(result, expected; rtol=1e-5))
end

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
        

        # Sanity checks
        @test result.p_0 >= 0
        @test all(isfinite.(result.alpha))
        @test all(isfinite.(result.p_test))
        @test all(isfinite.(result.p_baseline))
    end

end
