using KernelEconExamples
using Test

@testset "Asset Pricing Baseline" begin
    @testset "Default Parameters" begin
        c = 1.0
        g = 0.02
        r = 0.05
        x_0 = 1.0
        
        # Test vector input with values from Python
        t_values = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
        expected = [700.0, 1076.384765625, 1536.1019287109375, 2097.60205078125, 
                    2783.419677734375, 3621.0791015625]
        
        result = asset_pricing_baseline(t_values, c, g, r, x_0)
        
        for (i, (res, exp)) in enumerate(zip(result, expected))
            @test res ≈ exp rtol=1e-5
        end
    end
    
    @testset "Different Parameters" begin
        c = 0.5
        g = 0.03
        r = 0.04
        x_0 = 2.0
        
        t_values = [0.0, 15.0, 25.0]
        expected = [1450.0, 2510.849609375, 3535.06640625]
        
        result = asset_pricing_baseline(t_values, c, g, r, x_0)
        
        for (res, exp) in zip(result, expected)
            @test res ≈ exp rtol=1e-5
        end
    end
    
    @testset "Single Time Point" begin
        c = 1.0
        g = 0.02
        r = 0.05
        x_0 = 1.0
        
        # Test single-element array
        t = [10.0]
        result = asset_pricing_baseline(t, c, g, r, x_0)
        
        @test length(result) == 1
        @test result[1] ≈ 1076.384765625 rtol=1e-5
    end
    
    @testset "Monotonicity" begin
        c = 1.0
        g = 0.02
        r = 0.05
        x_0 = 1.0
        
        t_values = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
        result = asset_pricing_baseline(t_values, c, g, r, x_0)
        
        # Result should be monotonically increasing
        @test all(result[i+1] > result[i] for i in 1:length(result)-1)
    end
end
