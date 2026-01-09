# Julia Examples

Complete code examples for the `fastLowess` Julia package.

## Basic Smoothing

```julia
using fastLowess
using Random

# Generate noisy data
Random.seed!(42)
x = collect(range(0, 10, length=100))
y = sin.(x) .+ randn(100) .* 0.3

# Smooth
result = smooth(x, y, fraction=0.3, iterations=3)

# Print first few results
println("First 5 smoothed values: ", result.y[1:5])
```

---

## With Confidence Intervals

```julia
using fastLowess
using Random
using Printf

x = collect(range(0, 10, length=100))
y = sin.(x) .+ randn(100) .* 0.3

result = smooth(
    x, y,
    fraction=0.3,
    iterations=3,
    confidence_intervals=0.95,
    prediction_intervals=0.95,
    return_diagnostics=true
)

# Print diagnostics
if result.diagnostics !== nothing
    diag = result.diagnostics
    @printf("R²: %.4f, RMSE: %.4f\n", diag.r_squared, diag.rmse)
end

# Show intervals for a few points
println("\nIndex | Lower (95% CI) | Smoothed | Upper (95% CI)")
for i in 1:5
    @printf("%4d | %14.4f | %8.4f | %14.4f\n", 
        i, result.confidence_lower[i], result.y[i], result.confidence_upper[i])
end
```

---

## Cross-Validation

```julia
using fastLowess

x = collect(range(0, 20, length=200))
y = x .* 0.5 .+ sin.(x) .+ randn(200)

result = smooth(
    x, y,
    cv_method="kfold",
    cv_k=5,
    cv_fractions=[0.1, 0.2, 0.3, 0.5, 0.7],
    cv_seed=42
)

println("Best fraction: ", result.fraction_used)
# Note: cv_scores field contains the scores for each fraction tested
```

---

## Outlier Detection

```julia
using fastLowess

x = collect(Float64, 1:50)
y = x .* 2.0
y[10] = 100.0  # Outlier
y[25] = -50.0  # Outlier
y[40] = 150.0  # Outlier

result = smooth(
    x, y,
    fraction=0.3,
    iterations=5,
    robustness_method="bisquare",
    return_robustness_weights=true
)

# Find outliers (low weight)
outliers = findall(w -> w < 0.5, result.robustness_weights)
println("Outliers detected at indices: ", outliers)
```

---

## Streaming Large Data

```julia
using fastLowess
using Random

# Large dataset
n = 1_000_000
x = collect(range(0, 100, length=n))
y = sin.(x ./ 10) .+ randn(n) .* 0.1

# Streaming mode handles chunking internally
# Uses multiple threads for processing chunks in parallel
result = smooth_streaming(
    x, y,
    fraction=0.01,
    chunk_size=50000,
    overlap=5000,
    merge_strategy="weighted",
    parallel=true
)

println("Processed $(length(result.y)) points")
```

---

## Online (Real-Time) Processing

```julia
using fastLowess
using Random

# Sensor simulation
times = collect(Float64, 0:99)
values = 20.0 .+ 5.0 .* sin.(times ./ 10) .+ randn(100)

result = smooth_online(
    times, values,
    fraction=0.3,
    window_capacity=25,
    min_points=5,
    update_mode="incremental"
)

# Each output corresponds to a time point after warm-up
println("Output values: ", length(result.y))
```

---

## Multiple Fractions Comparison

```julia
using fastLowess

x = collect(range(0, 10, length=100))
y = sin.(x) .+ randn(100) .* 0.3

fractions = [0.1, 0.3, 0.5, 0.7]

for f in fractions
    result = smooth(x, y, fraction=f)
    println("Fraction f=$(f): first smoothed value = $(result.y[1])")
end
```

---

## Kernel Comparison

```julia
using fastLowess

x = collect(range(0, 10, length=100))
y = sin.(x) .+ randn(100) .* 0.2

kernels = ["tricube", "epanechnikov", "gaussian", "uniform"]

for kernel in kernels
    result = smooth(x, y, fraction=0.3, weight_function=kernel)
    println("Kernel $(kernel): R² = $(smooth(x, y, fraction=0.3, weight_function=kernel, return_diagnostics=true).diagnostics.r_squared)")
end
```

---

## DataFrames Integration

```julia
using fastLowess
using DataFrames

# Create DataFrame
df = DataFrame(
    time = collect(1:100),
    value = sin.(collect(1:100) ./ 10) .+ randn(100) .* 0.2
)

# Smooth
result = smooth(df.time, df.value, fraction=0.3)

# Add to DataFrame
df.smoothed = result.y

println(first(df, 5))
```

---

## Error Handling

```julia
using fastLowess

try
    # Invalid fraction
    result = smooth([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], fraction=2.0)
catch e
    println("Caught expected error: ", e)
end

try
    # Mismatched lengths
    result = smooth([1.0, 2.0, 3.0], [1.0, 2.0])
catch e
    if e isa ArgumentError
        println("Caught expected ArgumentError: ", e.msg)
    end
end
```
