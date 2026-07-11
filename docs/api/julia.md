# FastLOWESS Julia API Reference

The Julia bindings provide a modern interface to the core Rust library, mirroring the Rust API structure.

## Classes

### `Lowess`

The `Lowess` struct allows configuring the LOWESS parameters once and fitting multiple datasets using those parameters.

**Constructor:**

```julia
model = Lowess(; kwargs...)
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

### `StreamingLowess`

The `StreamingLowess` struct processes data in chunks, suitable for very large datasets or streaming applications.

**Constructor:**

```julia
stream = StreamingLowess(; kwargs...)
```

* `kwargs`: Keyword arguments corresponding to `StreamingOptions` fields.

**Methods:**

```julia
partial_result = process_chunk(stream, x::Vector{Float64}, y::Vector{Float64}) :: LowessResult
```

* Processes a chunk of data. Returns partial results.

```julia
final_result = finalize(stream) :: LowessResult
```

* Finalizes the smoothing process and returns any remaining buffered results.

### `OnlineLowess`

The `OnlineLowess` struct updates the model incrementally with new data points.

**Constructor:**

```julia
online = OnlineLowess(; kwargs...)
```

* `kwargs`: Keyword arguments corresponding to `OnlineOptions` fields.

**Methods:**

```julia
result = add_point(online, x[1], y[1])  # returns OnlineOutput or nothing
```

* Adds a single point to the sliding window. Returns `nothing` while the window is still filling (fewer than `min_points` seen), and an `OnlineOutput` once smoothing begins.

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

### `StreamingOptions` (inherits `LowessOptions`)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `chunk_size` | `Int` | `5000` | Data chunk size |
| `overlap` | `Int` | `500` | Overlap between chunks |
| `merge_strategy` | `String` | `"weighted_average"` | Strategy for blending overlap regions |

### `OnlineOptions` (inherits `LowessOptions`)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `window_capacity` | `Int` | `1000` | Max points in sliding window |
| `min_points` | `Int` | `3` | Min points before smoothing starts |
| `update_mode` | `String` | `"full"` | Update mode (`"full"` or `"incremental"`) |
| `parallel` | `Bool` | `false` | Enable parallel execution (off by default; online LOWESS fits one point at a time) |

## Result Structure

### `OnlineOutput`

Returned by `add_point()` once the window has enough points (`nothing` until then).

| Field | Type | Description |
| --- | --- | --- |
| `smoothed` | `Float64` | Smoothed value for the latest point |
| `std_error` | `Union{Float64, Nothing}` | Standard error (if requested) |
| `residual` | `Union{Float64, Nothing}` | Residual y − smoothed (if requested) |
| `robustness_weight` | `Union{Float64, Nothing}` | Robustness weight (if requested) |
| `iterations_used` | `Union{Int, Nothing}` | Robustness iterations performed |

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

* `"tricube"` (default)
* `"epanechnikov"`
* `"gaussian"`
* `"uniform"` (alias: `"boxcar"`)
* `"biweight"` (alias: `"bisquare"`)
* `"triangle"` (alias: `"triangular"`)
* `"cosine"`

### robustness_method

* `"bisquare"` (default; alias: `"biweight"`)
* `"huber"`
* `"talwar"`

### boundary_policy

* `"extend"` (default; alias: `"pad"`)
* `"reflect"` (alias: `"mirror"`)
* `"zero"`
* `"noboundary"` (alias: `"none"`)

### scaling_method

* `"mad"` (default; alias: `"median_absolute_deviation"`)
* `"mar"` (alias: `"median_absolute_residual"`)
* `"mean"` (alias: `"mean_absolute_residual"`)

### zero_weight_fallback

* `"use_local_mean"` (default; aliases: `"local_mean"`, `"mean"`)
* `"return_original"` (alias: `"original"`)
* `"return_none"` (alias: `"none"`)

### merge_strategy

* `"weighted_average"` (default; alias: `"weighted"`)
* `"average"` (alias: `"mean"`)
* `"take_first"` (alias: `"first"`)
* `"take_last"` (alias: `"last"`)

### update_mode

* `"full"` (default; alias: `"resmooth"`)
* `"incremental"` (alias: `"single"`)

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
