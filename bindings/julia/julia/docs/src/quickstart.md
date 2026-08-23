# Quick Start

Get up and running with LOWESS in minutes.

## Basic Smoothing

Smooth a noisy sine wave — the kind of signal where LOWESS shines. Each example recovers the underlying trend from 100 points of Gaussian noise.

```@example quickstart
using FastLOWESS, Random, Printf

# 100-point noisy sine wave
x = collect(range(0, 2π, length=100))
rng = MersenneTwister(42)
y = sin.(x) .+ randn(rng, 100) .* 0.3

model = Lowess(; fraction=0.3, iterations=3)
result = fit(model, x, y)

@printf "First smoothed: %.4f  (true: %.4f)\n" result.y[1] sin(x[1])
```

---

## With Confidence Intervals

```@example quickstart
using FastLOWESS
using Random, Statistics

rng = MersenneTwister(42)
x = collect(range(0, 2π, length=100))
y = sin.(x) .+ randn(rng, 100) .* 0.3

model = Lowess(;
    fraction=0.5,
    iterations=3,
    confidence_intervals=0.95,
    prediction_intervals=0.95,
    return_diagnostics=true
)
result = fit(model, x, y)

println("Smoothed: ", result.y)
println("CI Lower: ", result.confidence_lower)
println("CI Upper: ", result.confidence_upper)
println("R²: ", result.diagnostics.r_squared)
```

---

## Handling Outliers

LOWESS can robustly handle outliers through iterative reweighting:

```@example quickstart
using FastLOWESS
using Random, Statistics

rng = MersenneTwister(42)
x = collect(range(0, 2π, length=100))
y = sin.(x) .+ randn(rng, 100) .* 0.3

x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
y_with_outlier = [2.0, 4.0, 6.0, 50.0, 10.0, 12.0]

model = Lowess(;
    fraction=0.5,
    iterations=5,
    robustness_method="bisquare",
    return_robustness_weights=true
)
result = fit(model, x, y_with_outlier)

# Check which points were downweighted
for (i, w) in enumerate(result.robustness_weights)
    if w < 0.5
        println("Point $i is likely an outlier (weight: $(round(w, digits=3)))")
    end
end
```

---

## Streaming Mode

For datasets too large to fit in memory, stream them in fixed-size chunks with overlap.

```@example quickstart
using FastLOWESS, Random

x = collect(range(0, 10π, length=5000))
rng = MersenneTwister(42)
y = @. sin(x / π) * exp(-x / 30) + randn(rng) * 0.15

model = StreamingLowess(; fraction=0.2, chunk_size=1000, overlap=100,
                          merge_strategy="weighted_average")

chunk_size = 1000
for start in 1:chunk_size:4001
    stop = min(start + chunk_size - 1, length(x))
    process_chunk(model, x[start:stop], y[start:stop])
end
result = finalize(model)
println("Smoothed $(length(result.y)) points in streaming mode")
```

---

## Next Steps

| Topic | Link |
| --- | --- |
| How LOWESS works | [Concepts](concepts.md) |
| All parameters explained | [Parameters](parameters.md) |
| Batch vs Streaming vs Online | [Execution Modes](adapter-choice.md) |
| Edge handling | [Boundary](boundary.md) |
| Outlier handling in depth | [Robustness](robustness.md) |
| Full API per language | [API Reference](api.md) |
