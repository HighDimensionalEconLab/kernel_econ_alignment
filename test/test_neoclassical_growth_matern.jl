using Test
using KernelEconExamples
using Statistics

@testset "Neoclassical Growth Baseline" begin
    # Default parameters from neoclassical_growth_matern.py
    a = 1/3
    delta = 0.1
    rho_hat = 0.11
    sigma_crra = 1.0
    k_0 = 1.0
    T_max = 60.0
    
    @testset "Steady State Calculation" begin 
        k_ss = ((delta + rho_hat) / a)^(1 / (a - 1))
        c_ss = a * k_ss^a - delta * k_ss
        
        @test k_ss ≈ 1.999812026503847 rtol=1e-10
        @test c_ss ≈ 0.21997932291542321 rtol=1e-10
    end
    
    @testset "Baseline Solution" begin
        sol = neoclassical_growth_baseline(a, delta, rho_hat, sigma_crra, k_0, T_max;dt=0.01)
        
        # Test values generated from Python implementation
        test_cases = [
            (0.0,  1.000000000000000, 0.6933843919901543),
            (5.0,  1.657592269184338, 0.9428171663266427),
            (10.0, 1.8861852823518497, 1.0217645195263247),
            (20.0, 1.9874366901147922, 1.055808609459406),
        ]
        
        for (t, k_expected, c_expected) in test_cases
            val = sol(t)
            @test val[1] ≈ k_expected rtol=1e-3
            @test val[2] ≈ c_expected rtol=1e-3
        end
    end


end

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
            train_T=80.0,
            train_points=81,
            test_T=50.0,
            test_points=100,
            baseline_T=60.0,
            lambda_p=0.0,
            verbose=true
        )
        
        # Comparing mean relative errors to Python errors - Julia error strictly smaller
        @test mean(result.k_rel_error) < 1e-3 # 0.00040656029171102316 in Python
        @test mean(result.c_rel_error) < 1e-3  #  0.0019231719188908187 in Python

        # Testing relative errors at specific T's
        k_error_list = result.k_rel_error
        c_error_list = result.c_rel_error
        
        @test k_error_list[1] == 0.0
        @test k_error_list[40] < 1e-3
        @test k_error_list[80] < 1e-5
        @test k_error_list[100] < 1e-5

        @test c_error_list[1] < 1e-2
        @test c_error_list[40] < 1e-4
        @test c_error_list[80] < 1e-5
        @test c_error_list[100] < 1e-6
    

    end
end
