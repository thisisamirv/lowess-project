# fastLowess Julia Basic Usage Example
#
# Run with: julia examples/basic_usage.jl
#
# Make sure to build the Rust library first:
#   cd bindings/julia && cargo build --release

using Pkg

# Add the local package if needed
script_dir = @__DIR__
julia_pkg_dir = joinpath(dirname(script_dir), "julia")
if !haskey(Pkg.project().dependencies, "fastLowess")
    Pkg.develop(path=julia_pkg_dir)
end

using fastLowess

println("fastLowess Julia Bindings Demo")
println("=" ^ 40)

# Generate sample data
println("\n1. Generating sample data...")
x = collect(1.0:0.1:10.0)
y = sin.(x) .+ 0.1 .* randn(length(x))
println("   Generated $(length(x)) data points")

# Basic smoothing
println("\n2. Basic LOWESS smoothing...")
result = smooth(x, y)
println("   Fraction used: $(result.fraction_used)")
println("   First 5 smoothed values: $(round.(result.y[1:5], digits=4))")

# Smoothing with diagnostics
println("\n3. LOWESS with diagnostics...")
result = smooth(x, y,
    fraction = 0.3,
    iterations = 3,
    return_diagnostics = true
)

if result.diagnostics !== nothing
    println("   R² = $(round(result.diagnostics.r_squared, digits=6))")
    println("   RMSE = $(round(result.diagnostics.rmse, digits=6))")
    println("   MAE = $(round(result.diagnostics.mae, digits=6))")
end

# Different weight functions
println("\n4. Comparing weight functions...")
for wf in ["tricube", "epanechnikov", "gaussian"]
    r = smooth(x, y, weight_function=wf, return_diagnostics=true)
    rmse = r.diagnostics !== nothing ? round(r.diagnostics.rmse, digits=6) : "N/A"
    println("   $wf: RMSE = $rmse")
end

# Online mode
println("\n5. Online LOWESS (sliding window)...")
result = smooth_online(x, y, window_capacity=20)
println("   Processed $(length(result.y)) points with sliding window")

println("\n" * "=" ^ 40)
println("Demo complete!")
