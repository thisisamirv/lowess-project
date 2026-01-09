#!/usr/bin/env julia
"""
Tests for fastLowess Julia bindings.

Comprehensive test suite covering:
- Basic smoothing functionality
- Full smooth() with all options
- Cross-validation
- Streaming and online adapters
- Error handling
- Edge cases

Run with: julia --project=bindings/julia/julia tests/julia/test_fastlowess.jl
"""

using Test
using Random

# Handle package loading - check if we're already in the fastLowess project
using Pkg
project_name = Pkg.project().name
if project_name != "fastLowess"
    # Not in the fastLowess project, need to develop it
    script_dir = @__DIR__
    project_root = dirname(dirname(script_dir))
    julia_pkg_dir = joinpath(project_root, "bindings", "julia", "julia")
    if !haskey(Pkg.project().dependencies, "fastLowess")
        Pkg.develop(path=julia_pkg_dir)
    end
end

using fastLowess

@testset "fastLowess Julia Bindings" begin

    @testset "Basic Smooth" begin
        @testset "default parameters" begin
            x = [1.0, 2.0, 3.0, 4.0, 5.0]
            y = [2.0, 4.1, 5.9, 8.2, 9.8]

            result = smooth(x, y, fraction=0.5)

            @test result isa LowessResult
            @test length(result.y) == length(x)
            @test length(result.x) == length(x)
            @test result.fraction_used ≈ 0.5
        end

        @testset "serial execution" begin
            x = [1.0, 2.0, 3.0, 4.0, 5.0]
            y = [2.0, 4.1, 5.9, 8.2, 9.8]

            result = smooth(x, y, fraction=0.5, parallel=false)

            @test result isa LowessResult
            @test length(result.y) == length(x)
        end

        @testset "with diagnostics" begin
            x = [1.0, 2.0, 3.0, 4.0, 5.0]
            y = [2.0, 4.1, 5.9, 8.2, 9.8]

            result = smooth(x, y, fraction=0.5, return_diagnostics=true)

            @test result.diagnostics !== nothing
            @test result.diagnostics isa Diagnostics
            @test result.diagnostics.rmse >= 0
            @test result.diagnostics.mae >= 0
            @test 0 <= result.diagnostics.r_squared <= 1
            @test result.diagnostics.residual_sd >= 0
        end

        @testset "with residuals" begin
            x = [1.0, 2.0, 3.0, 4.0, 5.0]
            y = [2.0, 4.1, 5.9, 8.2, 9.8]

            result = smooth(x, y, fraction=0.5, return_residuals=true)

            @test result.residuals !== nothing
            @test length(result.residuals) == length(x)
        end

        @testset "with robustness weights" begin
            x = [1.0, 2.0, 3.0, 4.0, 5.0]
            y = [2.0, 4.1, 100.0, 8.2, 9.8]  # Outlier

            result = smooth(x, y, fraction=0.7, iterations=3, return_robustness_weights=true)

            @test result.robustness_weights !== nothing
            @test length(result.robustness_weights) == length(x)
            @test all(result.robustness_weights .>= 0)
            @test all(result.robustness_weights .<= 1)
        end

        @testset "with confidence intervals" begin
            Random.seed!(42)
            x = collect(range(0, 10, length=20))
            y = 2 .* x .+ randn(20)

            result = smooth(x, y, fraction=0.5, confidence_intervals=0.95)

            @test result.confidence_lower !== nothing
            @test result.confidence_upper !== nothing
            @test length(result.confidence_lower) == length(x)
            @test length(result.confidence_upper) == length(x)
            @test all(result.confidence_lower .<= result.confidence_upper)
        end

        @testset "with prediction intervals" begin
            Random.seed!(42)
            x = collect(range(0, 10, length=20))
            y = 2 .* x .+ randn(20)

            result = smooth(x, y, fraction=0.5, prediction_intervals=0.95)

            @test result.prediction_lower !== nothing
            @test result.prediction_upper !== nothing
            @test length(result.prediction_lower) == length(x)
            @test length(result.prediction_upper) == length(x)
        end
    end

    @testset "Weight Functions" begin
        x = collect(range(0, 10, length=20))
        y = sin.(x)

        kernels = ["tricube", "epanechnikov", "gaussian", "uniform", "biweight", "triangle"]

        for kernel in kernels
            @testset "$kernel" begin
                result = smooth(x, y, fraction=0.5, weight_function=kernel)
                @test length(result.y) == length(x)
            end
        end
    end

    @testset "Robustness Methods" begin
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 100.0, 8.0, 10.0]  # Outlier

        methods = ["bisquare", "huber", "talwar"]

        for method in methods
            @testset "$method" begin
                result = smooth(x, y, fraction=0.7, iterations=3, robustness_method=method)
                @test length(result.y) == length(x)
            end
        end
    end

    @testset "Iterations" begin
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 100.0, 8.0, 10.0]

        for iterations in [0, 1, 3, 5]
            @testset "iterations=$iterations" begin
                result = smooth(x, y, fraction=0.7, iterations=iterations)
                @test length(result.y) == length(x)
            end
        end
    end

    @testset "Streaming Smooth" begin
        @testset "basic streaming" begin
            x = collect(range(0, 1000, length=2000))
            y = sin.(x ./ 100)

            result = smooth_streaming(x, y, fraction=0.1, chunk_size=1000)

            @test result isa LowessResult
            @test length(result.y) == length(x)
        end

        @testset "larger dataset" begin
            Random.seed!(42)
            x = collect(range(0, 1000, length=5000))
            y = sin.(x ./ 100) .+ randn(5000) .* 0.1

            result = smooth_streaming(x, y, fraction=0.05, chunk_size=1500)

            @test result isa LowessResult
            @test length(result.y) == length(x)
        end

        @testset "streaming accuracy" begin
            x = collect(range(0, 100, length=200))
            y = 2 .* x .+ 1  # Perfect linear

            result_stream = smooth_streaming(x, y, fraction=0.5, chunk_size=1000)
            result_batch = smooth(x, y, fraction=0.5)

            @test result_stream.y ≈ result_batch.y rtol = 1e-10
        end
    end

    @testset "Online Smooth" begin
        @testset "basic online" begin
            x = collect(Float64, 1:10)
            y = collect(Float64, 2:2:20)

            result = smooth_online(x, y, fraction=0.5, window_capacity=10, min_points=3)

            @test length(result.y) == length(x)
            @test result isa LowessResult
        end

        @testset "with noise" begin
            Random.seed!(42)
            x = collect(range(0, 20, length=50))
            y = 2 .* x .+ randn(50)

            result = smooth_online(x, y, fraction=0.3, window_capacity=20, min_points=5)

            @test length(result.y) == length(x)
        end

        @testset "update modes" begin
            x = collect(Float64, 0:99)
            y = 20.0 .+ 5.0 .* sin.(x .* 0.1)

            result_full = smooth_online(x, y, fraction=0.3, window_capacity=50, update_mode="full")
            result_inc = smooth_online(x, y, fraction=0.3, window_capacity=50, update_mode="incremental")

            @test length(result_full.y) == length(x)
            @test length(result_inc.y) == length(x)
        end
    end

    @testset "Result Fields" begin
        @testset "optional fields none" begin
            x = [1.0, 2.0, 3.0, 4.0, 5.0]
            y = [2.0, 4.0, 6.0, 8.0, 10.0]

            result = smooth(x, y, fraction=0.5)

            @test result.diagnostics === nothing
            @test result.residuals === nothing
            @test result.robustness_weights === nothing
            @test result.confidence_lower === nothing
            @test result.confidence_upper === nothing
            @test result.prediction_lower === nothing
            @test result.prediction_upper === nothing
        end
    end

    @testset "Diagnostics Values" begin
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]  # Perfect linear

        result = smooth(x, y, fraction=0.5, return_diagnostics=true)

        diag = result.diagnostics
        @test diag.rmse < 0.1
        @test diag.mae < 0.1
        @test diag.r_squared > 0.99
    end

    @testset "Edge Cases" begin
        @testset "two points" begin
            x = [1.0, 2.0]
            y = [2.0, 4.0]

            result = smooth(x, y, fraction=1.0)
            @test length(result.y) == 2
        end

        @testset "large dataset" begin
            Random.seed!(42)
            n = 1000
            x = collect(range(0, 100, length=n))
            y = sin.(x ./ 10) .+ randn(n) .* 0.1

            result = smooth(x, y, fraction=0.1)
            @test length(result.y) == n
        end

        @testset "unsorted input" begin
            x = [3.0, 1.0, 5.0, 2.0, 4.0]
            y = [6.0, 2.0, 10.0, 4.0, 8.0]

            result = smooth(x, y, fraction=0.7)
            @test length(result.y) == 5
        end

        @testset "duplicate x values" begin
            x = [1.0, 1.0, 2.0, 2.0, 3.0]
            y = [2.0, 2.1, 4.0, 3.9, 6.0]

            result = smooth(x, y, fraction=0.7)
            @test length(result.y) == 5
        end

        @testset "constant y values" begin
            x = [1.0, 2.0, 3.0, 4.0, 5.0]
            y = [5.0, 5.0, 5.0, 5.0, 5.0]

            result = smooth(x, y, fraction=0.5)
            @test result.y ≈ y rtol = 1e-10
        end
    end

    @testset "Cross-Validation" begin
        @testset "basic CV" begin
            x = collect(range(0, 10, length=50))
            y = 2 .* x .+ sin.(x)

            result = smooth(x, y, cv_fractions=[0.2, 0.3, 0.5, 0.7])

            @test result.fraction_used in [0.2, 0.3, 0.5, 0.7]
            @test length(result.y) == length(x)
        end

        @testset "k-fold CV" begin
            x = collect(range(0, 10, length=30))
            y = x .^ 2

            result = smooth(x, y, cv_fractions=[0.3, 0.5], cv_method="kfold", cv_k=5)

            @test result.fraction_used in [0.3, 0.5]
        end

        @testset "LOOCV" begin
            x = collect(range(0, 10, length=20))
            y = sin.(x)

            result = smooth(x, y, cv_fractions=[0.4, 0.6], cv_method="loocv")

            @test result.fraction_used in [0.4, 0.6]
        end
    end

    @testset "Error Handling" begin
        @testset "mismatched lengths" begin
            x = [1.0, 2.0, 3.0]
            y = [2.0, 4.0]

            @test_throws ArgumentError smooth(x, y, fraction=0.5)
        end

        @testset "invalid weight function" begin
            x = [1.0, 2.0, 3.0]
            y = [2.0, 4.0, 6.0]

            @test_throws ErrorException smooth(x, y, fraction=0.5, weight_function="invalid")
        end

        @testset "invalid robustness method" begin
            x = [1.0, 2.0, 3.0]
            y = [2.0, 4.0, 6.0]

            @test_throws ErrorException smooth(x, y, fraction=0.5, robustness_method="invalid")
        end
    end

end  # main testset

println("\n✓ All tests passed!")
