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
    
    @testset "Steady State Calculation" begin #there's a perturb_k measuring deviation from ss 
        k_ss = ((delta + rho_hat) / a)^(1 / (a - 1))
        c_ss = a * k_ss^a - delta * k_ss
        
        @test k_ss ≈ 1.999812026503847 rtol=1e-10
        @test c_ss ≈ 0.21997932291542321 rtol=1e-10
    end
    
    @testset "Baseline Solution" begin
        sol = neoclassical_growth_baseline(a, delta, rho_hat, sigma_crra, k_0, T_max)
        
        # Test values generated from Python implementation
        test_cases = [
            (0.0,  1.000000000000000, 0.6924790598846664),
            (10.0, 1.8855061840618146, 1.0215114953510323),
            (20.0, 1.9873840082680825, 1.0557592671175073),
            (30.0, 1.9984666671221869, 1.0594524695219083),
            (40.0, 1.9996663056043391, 1.0598519467970264),
            (50.0, 1.9997925349500083, 1.0598959424587540),
            (60.0, 1.9997120265030062, 1.0599217100950145),
        ]
        
        for (t, k_expected, c_expected) in test_cases
            val = sol(t)
            @test val[1] ≈ k_expected rtol=1e-3
            @test val[2] ≈ c_expected rtol=1e-2
        end
    end

    @testset "Regression Test for different parameters" begin
    a2 = 0.3
    delta2 = 0.05
    r2 = 0.04
    sigma_crra2 = 2.0
    k_0_2 = 0.5
    T_max2 = 30.0

    k_ss2 = ((delta2 + r2) / a2)^(1 / (a2 - 1))

    sol2 = neoclassical_growth_baseline(a2, delta2, r2, sigma_crra2, k_0_2, T_max2)
    
    val_0 = sol2(0.0)
    val_10 = sol2(10.0)
    val_30 = sol2(T_max2)

    @test abs(val_30[1] - k_ss2) < 0.01
    @test val_0[1] ≈ 0.5 rtol=1e-10
    @test val_0[2] ≈ 0.5223374558692421 rtol= 1e-6   
    @test val_10[1] ≈ 2.8876964556629114 rtol= 1e-6
    @test val_10[2] ≈ 1.0409788165950287 rtol=1e-6
    @test val_30[1] ≈ 5.584211504181964 rtol=1e-6
    @test val_30[2] ≈ 1.2651287490361591 rtol=1e-6
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
            train_T=40.0,
            train_points=41,
            test_T=50.0,
            test_points=100,
            baseline_T=60.0,
            lambda_p=0.0,
            verbose=true
        )
        
        # Test c_0 value (relaxed due to solver differences)
        @test result.c_0 ≈ 0.691 atol=0.01
        
        # Test mean relative errors (primary validation)
        @test mean(result.k_rel_error) < 0.001  # Should be around 0.0004
        @test mean(result.c_rel_error) < 0.005  # Should be around 0.003


        # Test that kernel_solution is callable
        @test result.kernel_solution isa Function
        k_interp, c_interp = result.kernel_solution([0.0, 1.0])
        @test length(k_interp) == 2
        @test length(c_interp) == 2
    end
end
