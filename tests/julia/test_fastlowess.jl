#!/usr/bin/env julia
"""
Tests for fastlowess Julia bindings.

Comprehensive test suite covering:
- Stateful Lowess class (batch smoothing)
- Reusability of Lowess instances
- StreamingLowess class
- OnlineLowess class
- Error handling
- Edge cases

Run with: julia --project=bindings/julia/julia tests/julia/test_fastlowess.jl
"""

using Test
using Random

# Handle package loading - check if we're already in the fastlowess project
using Pkg
project_name = Pkg.project().name
if project_name != "fastlowess"
    # Not in the fastlowess project, need to develop it
    script_dir = @__DIR__
    project_root = dirname(dirname(script_dir))
    julia_pkg_dir = joinpath(project_root, "bindings", "julia", "julia")
    if !haskey(Pkg.project().dependencies, "fastlowess")
        Pkg.develop(path = julia_pkg_dir)
    end
end

using fastlowess

@testset "fastlowess Julia Bindings" begin

    @testset "Lowess (Batch)" begin
        @testset "default parameters" begin
            x = [1.0, 2.0, 3.0, 4.0, 5.0]
            y = [2.0, 4.1, 5.9, 8.2, 9.8]

            model = Lowess(fraction = 0.5)
            result = fit(model, x, y)

            @test result isa LowessResult
            @test length(result.y) == length(x)
            @test length(result.x) == length(x)
            @test result.fraction_used ≈ 0.5
        end

        @testset "reuse Lowess instance" begin
            x1 = [1.0, 2.0, 3.0, 4.0, 5.0]
            y1 = [2.0, 4.1, 5.9, 8.2, 9.8]
            x2 = [1.0, 2.0, 3.0]
            y2 = [1.0, 2.0, 3.0]

            model = Lowess(fraction = 0.5)

            # First fit
            result1 = fit(model, x1, y1)
            @test length(result1.y) == length(x1)

            # Second fit with different data
            result2 = fit(model, x2, y2)
            @test length(result2.y) == length(x2)
        end

        @testset "serial execution" begin
            x = [1.0, 2.0, 3.0, 4.0, 5.0]
            y = [2.0, 4.1, 5.9, 8.2, 9.8]

            model = Lowess(fraction = 0.5, parallel = false)
            result = fit(model, x, y)

            @test result isa LowessResult
            @test length(result.y) == length(x)
        end

        @testset "with diagnostics" begin
            x = [1.0, 2.0, 3.0, 4.0, 5.0]
            y = [2.0, 4.1, 5.9, 8.2, 9.8]

            model = Lowess(fraction = 0.5, return_diagnostics = true)
            result = fit(model, x, y)

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

            model = Lowess(fraction = 0.5, return_residuals = true)
            result = fit(model, x, y)

            @test result.residuals !== nothing
            @test length(result.residuals) == length(x)
        end

        @testset "with robustness weights" begin
            x = [1.0, 2.0, 3.0, 4.0, 5.0]
            y = [2.0, 4.1, 100.0, 8.2, 9.8]  # Outlier

            model = Lowess(fraction = 0.7, iterations = 3, return_robustness_weights = true)
            result = fit(model, x, y)

            @test result.robustness_weights !== nothing
            @test length(result.robustness_weights) == length(x)
            @test all(result.robustness_weights .>= 0)
            @test all(result.robustness_weights .<= 1)
        end

        @testset "with confidence intervals" begin
            Random.seed!(42)
            x = collect(range(0, 10, length = 20))
            y = 2 .* x .+ randn(20)

            model = Lowess(fraction = 0.5, confidence_intervals = 0.95)
            result = fit(model, x, y)

            @test result.confidence_lower !== nothing
            @test result.confidence_upper !== nothing
            @test length(result.confidence_lower) == length(x)
            @test length(result.confidence_upper) == length(x)
            @test all(result.confidence_lower .<= result.confidence_upper)
        end

        @testset "with prediction intervals" begin
            Random.seed!(42)
            x = collect(range(0, 10, length = 20))
            y = 2 .* x .+ randn(20)

            model = Lowess(fraction = 0.5, prediction_intervals = 0.95)
            result = fit(model, x, y)

            @test result.prediction_lower !== nothing
            @test result.prediction_upper !== nothing
            @test length(result.prediction_lower) == length(x)
            @test length(result.prediction_upper) == length(x)
        end
    end

    @testset "Weight Functions" begin
        x = collect(range(0, 10, length = 20))
        y = sin.(x)

        kernels = ["tricube", "epanechnikov", "gaussian", "uniform", "biweight", "triangle"]

        for kernel in kernels
            @testset "$kernel" begin
                model = Lowess(fraction = 0.5, weight_function = kernel)
                result = fit(model, x, y)
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
                model = Lowess(fraction = 0.7, iterations = 3, robustness_method = method)
                result = fit(model, x, y)
                @test length(result.y) == length(x)
            end
        end
    end

    @testset "Iterations" begin
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 100.0, 8.0, 10.0]

        for iterations in [0, 1, 3, 5]
            @testset "iterations=$iterations" begin
                model = Lowess(fraction = 0.7, iterations = iterations)
                result = fit(model, x, y)
                @test length(result.y) == length(x)
            end
        end
    end

    @testset "StreamingLowess" begin
        @testset "basic streaming" begin
            x = collect(range(0, 1000, length = 2000))
            y = sin.(x ./ 100)

            stream = StreamingLowess(fraction = 0.1, chunk_size = 1000)

            # First chunk
            r1 = process_chunk(stream, x[1:1000], y[1:1000])
            @test r1 isa LowessResult # Partial results might be empty or not, but generally empty until finalized or enough data

            # Second chunk
            r2 = process_chunk(stream, x[1001:end], y[1001:end])

            r_final = finalize(stream)

            # Rust streaming usually returns results per chunk if possible, or buffered. 
            # The test just checks it runs and returns a LowessResult structure.
            @test r_final isa LowessResult
        end

        @testset "larger dataset streaming results" begin
            Random.seed!(42)
            x = collect(range(0, 1000, length = 5000))
            y = sin.(x ./ 100) .+ randn(5000) .* 0.1

            stream = StreamingLowess(fraction = 0.05, chunk_size = 1500)

            # We just verify it runs without error
            process_chunk(stream, x[1:2500], y[1:2500])
            process_chunk(stream, x[2501:end], y[2501:end])
            finalize(stream)
        end

        @testset "streaming accuracy" begin
            x = collect(range(0, 100, length = 200))
            y = 2 .* x .+ 1  # Perfect linear

            stream = StreamingLowess(fraction = 0.5, chunk_size = 1000)
            r1 = process_chunk(stream, x, y)
            r2 = finalize(stream)

            model_batch = Lowess(fraction = 0.5)
            result_batch = fit(model_batch, x, y)

            # Combine streaming results
            stream_y = vcat(r1.y, r2.y)

            @test stream_y ≈ result_batch.y rtol = 1e-10
        end
    end

    @testset "OnlineLowess" begin
        @testset "basic online" begin
            x = collect(Float64, 1:10)
            y = collect(Float64, 2:2:20)

            online = OnlineLowess(fraction = 0.5, window_capacity = 10, min_points = 3)
            result = add_points(online, x, y)

            @test length(result.y) == length(x)
            @test result isa LowessResult
        end

        @testset "with noise" begin
            Random.seed!(42)
            x = collect(range(0, 20, length = 50))
            y = 2 .* x .+ randn(50)

            online = OnlineLowess(fraction = 0.3, window_capacity = 20, min_points = 5)
            result = add_points(online, x, y)

            @test length(result.y) == length(x)
        end

        @testset "update modes" begin
            x = collect(Float64, 0:99)
            y = 20.0 .+ 5.0 .* sin.(x .* 0.1)

            o1 = OnlineLowess(fraction = 0.3, window_capacity = 50, update_mode = "full")
            result_full = add_points(o1, x, y)

            o2 = OnlineLowess(
                fraction = 0.3,
                window_capacity = 50,
                update_mode = "incremental",
            )
            result_inc = add_points(o2, x, y)

            @test length(result_full.y) == length(x)
            @test length(result_inc.y) == length(x)
        end
    end

    @testset "Result Fields" begin
        @testset "optional fields none" begin
            x = [1.0, 2.0, 3.0, 4.0, 5.0]
            y = [2.0, 4.0, 6.0, 8.0, 10.0]

            model = Lowess(fraction = 0.5)
            result = fit(model, x, y)

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

        model = Lowess(fraction = 0.5, return_diagnostics = true)
        result = fit(model, x, y)

        diag = result.diagnostics
        @test diag.rmse < 0.1
        @test diag.mae < 0.1
        @test diag.r_squared > 0.99
    end

    @testset "Edge Cases" begin
        @testset "two points" begin
            x = [1.0, 2.0]
            y = [2.0, 4.0]

            model = Lowess(fraction = 1.0)
            result = fit(model, x, y)
            @test length(result.y) == 2
        end

        @testset "large dataset" begin
            Random.seed!(42)
            n = 1000
            x = collect(range(0, 100, length = n))
            y = sin.(x ./ 10) .+ randn(n) .* 0.1

            model = Lowess(fraction = 0.1)
            result = fit(model, x, y)
            @test length(result.y) == n
        end

        @testset "unsorted input" begin
            x = [3.0, 1.0, 5.0, 2.0, 4.0]
            y = [6.0, 2.0, 10.0, 4.0, 8.0]

            model = Lowess(fraction = 0.7)
            result = fit(model, x, y)
            @test length(result.y) == 5
        end

        @testset "duplicate x values" begin
            x = [1.0, 1.0, 2.0, 2.0, 3.0]
            y = [2.0, 2.1, 4.0, 3.9, 6.0]

            model = Lowess(fraction = 0.7)
            result = fit(model, x, y)
            @test length(result.y) == 5
        end

        @testset "constant y values" begin
            x = [1.0, 2.0, 3.0, 4.0, 5.0]
            y = [5.0, 5.0, 5.0, 5.0, 5.0]

            model = Lowess(fraction = 0.5)
            result = fit(model, x, y)
            @test result.y ≈ y rtol = 1e-10
        end
    end

    @testset "Cross-Validation" begin
        @testset "basic CV" begin
            x = collect(range(0, 10, length = 50))
            y = 2 .* x .+ sin.(x)

            model = Lowess(cv_fractions = [0.2, 0.3, 0.5, 0.7])
            result = fit(model, x, y)

            @test result.fraction_used in [0.2, 0.3, 0.5, 0.7]
            @test length(result.y) == length(x)
        end

        @testset "k-fold CV" begin
            x = collect(range(0, 10, length = 30))
            y = x .^ 2

            model = Lowess(cv_fractions = [0.3, 0.5], cv_method = "kfold", cv_k = 5)
            result = fit(model, x, y)

            @test result.fraction_used in [0.3, 0.5]
        end

        @testset "LOOCV" begin
            x = collect(range(0, 10, length = 20))
            y = sin.(x)

            model = Lowess(cv_fractions = [0.4, 0.6], cv_method = "loocv")
            result = fit(model, x, y)

            @test result.fraction_used in [0.4, 0.6]
        end
    end

    @testset "Error Handling" begin
        @testset "mismatched lengths" begin
            x = [1.0, 2.0, 3.0]
            y = [2.0, 4.0]

            model = Lowess(fraction = 0.5)
            @test_throws ArgumentError fit(model, x, y)
        end

        @testset "invalid weight function" begin
            # Error happens at construction time now
            @test_throws ErrorException Lowess(fraction = 0.5, weight_function = "invalid")
        end

        @testset "invalid robustness method" begin
            @test_throws ErrorException Lowess(
                fraction = 0.5,
                robustness_method = "invalid",
            )
        end
    end

end  # main testset

println("\n✓ All tests passed!")
