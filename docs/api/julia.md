# FastLOWESS Julia API Reference

The Julia bindings provide a modern interface to the core Rust library, mirroring the Rust API structure.

> **StreamingLowess** and **OnlineLowess** are documented separately: [julia-streaming.md](julia-streaming.md), [julia-online.md](julia-online.md)

## Classes

### `Lowess`

The `Lowess` struct allows configuring the LOWESS parameters once and fitting multiple datasets using those parameters.

**Constructor:**

```julia
using FastLOWESS
using Random, Statistics

rng = MersenneTwister(42)
x = collect(range(0, 2π, length=100))
y = sin.(x) .+ randn(rng, 100) .* 0.3

model = Lowess()
```

* `kwargs`: Keyword arguments corresponding to `LowessOptions` fields.

**Methods:**

```julia
result = fit(model, x::Vector{Float64}, y::Vector{Float64};
             custom_weights::Union{Vector{Float64}, Nothing} = nothing) :: LowessResult
```

* Fits the model to the provided `x` and `y` data vectors.
* `custom_weights`: Optional per-observation weights. All values must be ≥ 0 and length must match `x`. Batch only.
* Returns a `LowessResult` struct containing the smoothed values and optional diagnostics.

See [julia-streaming.md](julia-streaming.md) for the `StreamingLowess` struct.

See [julia-online.md](julia-online.md) for the `OnlineLowess` struct.

## Options Structures

### `LowessOptions`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `fraction` | `Float64` | `0.67` | Smoothing fraction (bandwidth) |
| `iterations` | `Int` | `3` | Number of robustifying iterations |
| `delta` | `Float64` | `NaN` | Interpolation distance (NaN for auto) |
| `weight_function` | `String` | `"tricube"` | Weight function name |
| `robustness_method` | `String` | `"bisquare"` | Robustness method name |
| `scaling_method` | `String` | `"mad"` | Residual scaling method |
| `boundary_policy` | `String` | `"extend"` | Boundary handling policy |
| `zero_weight_fallback` | `String` | `"use_local_mean"` | Zero-weight handling strategy |
| `auto_converge` | `Float64` | `NaN` | Auto-convergence tolerance |
| `confidence_intervals` | `Float64` | `NaN` | Confidence level (e.g., 0.95) |
| `prediction_intervals` | `Float64` | `NaN` | Prediction level (e.g., 0.95) |
| `return_diagnostics` | `Bool` | `false` | Include diagnostics in result |
| `return_residuals` | `Bool` | `false` | Include residuals in result |
| `return_robustness_weights` | `Bool` | `false` | Include weights in result |
| `return_se` | `Bool` | `false` | Return standard errors |
| `parallel` | `Bool` | `true` | Enable parallel execution |
| `cv_method` | `String` | `"kfold"` | CV method (`"kfold"` or `"loocv"`) (Batch only) |
| `cv_k` | `Int` | `5` | Number of folds for k-fold CV (Batch only) |
| `cv_fractions` | `Vector{Float64}` | `Float64[]` | Fractions to test for cross-validation (Batch only) |
| `cv_seed` | `Union{Int, Nothing}` | `nothing` | Random seed for cross-validation shuffling (Batch only) |
| `custom_weights` | `Union{Vector{Float64}, Nothing}` | `nothing` | Per-observation case weights — passed to `fit()`, not the constructor (Batch only) |

See [julia-streaming.md](julia-streaming.md) for `StreamingOptions`.

See [julia-online.md](julia-online.md) for `OnlineOptions`.

## Result Structure

See [julia-online.md](julia-online.md) for `OnlineOutput`.

### `LowessResult`

| Field | Type | Description |
| --- | --- | --- |
| `x` | `Vector{Float64}` | Sorted x values |
| `y` | `Vector{Float64}` | Smoothed y values |
| `fraction_used` | `Float64` | Fraction used (set or selected by CV) |
| `iterations_used` | `Union{Int, Nothing}` | Robustness iterations actually performed |
| `standard_errors` | `Union{Vector{Float64}, Nothing}` | Per-point standard errors |
| `confidence_lower` | `Union{Vector{Float64}, Nothing}` | Lower confidence bounds |
| `confidence_upper` | `Union{Vector{Float64}, Nothing}` | Upper confidence bounds |
| `prediction_lower` | `Union{Vector{Float64}, Nothing}` | Lower prediction bounds |
| `prediction_upper` | `Union{Vector{Float64}, Nothing}` | Upper prediction bounds |
| `residuals` | `Union{Vector{Float64}, Nothing}` | Residuals (if `return_residuals`) |
| `robustness_weights` | `Union{Vector{Float64}, Nothing}` | Robustness weights (if `return_robustness_weights`) |
| `cv_scores` | `Union{Vector{Float64}, Nothing}` | CV score per tested fraction |
| `diagnostics` | `Union{Diagnostics, Nothing}` | Fit metrics (if `return_diagnostics`) |

### `Diagnostics`

| Field | Type | Description |
| --- | --- | --- |
| `rmse` | `Float64` | Root Mean Squared Error |
| `mae` | `Float64` | Mean Absolute Error |
| `r_squared` | `Float64` | R-squared |
| `residual_sd` | `Float64` | Residual standard deviation |
| `effective_df` | `Float64` | Effective degrees of freedom (NaN if not computed) |
| `aic` | `Float64` | AIC (NaN if not computed) |
| `aicc` | `Float64` | AICc (NaN if not computed) |

## Options

### weight_function

*See: [Weight Functions](../user-guide/kernels.md)*

* `"tricube"` (default)
* `"epanechnikov"`
* `"gaussian"`
* `"uniform"` (alias: `"boxcar"`)
* `"biweight"` (alias: `"bisquare"`)
* `"triangle"` (alias: `"triangular"`)
* `"cosine"`

### robustness_method

*See: [Robustness](../user-guide/robustness.md)*

* `"bisquare"` (default; alias: `"biweight"`)
* `"huber"`
* `"talwar"`

### boundary_policy

*See: [Boundary Handling](../user-guide/boundary.md)*

* `"extend"` (default; alias: `"pad"`)
* `"reflect"` (alias: `"mirror"`)
* `"zero"`
* `"noboundary"` (alias: `"none"`)

### scaling_method

*See: [Scaling Methods](../user-guide/scaling.md)*

* `"mad"` (default; alias: `"median_absolute_deviation"`)
* `"mar"` (alias: `"median_absolute_residual"`)
* `"mean"` (alias: `"mean_absolute_residual"`)

### zero_weight_fallback

*See: [Parameters](../user-guide/parameters.md)*

* `"use_local_mean"` (default; aliases: `"local_mean"`, `"mean"`)
* `"return_original"` (alias: `"original"`)
* `"return_none"` (alias: `"none"`)

### merge_strategy

See [julia-streaming.md](julia-streaming.md).

### update_mode

See [julia-online.md](julia-online.md).

## Example

```julia
using FastLOWESS

x = collect(range(0, 10, length=100))
y = sin.(x) .+ randn(100) .* 0.2

# Configure model
model = Lowess(fraction=0.5, iterations=3)

# Fit data (throws on error)
result = fit(model, x, y)

println("Smoothed Y: ", result.y)
```
