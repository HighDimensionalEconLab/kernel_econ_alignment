# Test file for neoclassical_growth_baseline.jl
using Test
using KernelEconExamples

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
        sol = neoclassical_growth_baseline(a, delta, rho_hat, sigma_crra, k_0, T_max)
        
        # Test values generated from Python implementation
        test_cases = [
            (0.0,  1.000000000000000, 0.693381490749002),
            (10.0, 1.886226753937708, 1.021760153254940),
            (20.0, 1.987526619089377, 1.055806946207147),
            (30.0, 1.998488713577121, 1.059459885006358),
            (40.0, 1.999669244228402, 1.059852986564493),
            (50.0, 1.999793048487918, 1.059896048245372),
            (60.0, 1.999712026723342, 1.059921736106915),
        ]
        
        for (t, k_expected, c_expected) in test_cases
            val = sol(t)
            @test val[1] ≈ k_expected rtol=1e-5
            @test val[2] ≈ c_expected rtol=1e-5
        end
    end
    
    @testset "Vector Evaluation" begin
        sol = neoclassical_growth_baseline(a, delta, rho_hat, sigma_crra, k_0, T_max)
        
        # Test evaluating at multiple points (evaluate individually to avoid deprecation)
        test_t = [0.0, 10.0, 20.0, 30.0]
        
        # Check initial condition
        val_0 = sol(test_t[1])
        @test val_0[1] ≈ 1.0 rtol=1e-10
        
        # Check monotonicity (capital should increase towards steady state)
        k_vals = [sol(t)[1] for t in test_t]
        @test all(k_vals[i+1] >= k_vals[i] for i in 1:length(k_vals)-1)
    end
    
    @testset "Different Parameters" begin
        a2 = 0.3
        delta2 = 0.05
        r2 = 0.04
        sigma_crra2 = 2.0
        k_0_2 = 0.5
        T_max2 = 30.0
        
        sol2 = neoclassical_growth_baseline(a2, delta2, r2, sigma_crra2, k_0_2, T_max2)
        
        # Basic sanity checks
        val_0 = sol2(0.0)
        @test val_0[1] ≈ k_0_2 rtol=1e-10
        @test val_0[2] > 0
        
        val_T = sol2(T_max2)
        k_ss2 = ((delta2 + r2) / a2)^(1 / (a2 - 1))
        @test abs(val_T[1] - k_ss2) < 0.01
    end
end
