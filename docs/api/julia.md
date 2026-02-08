# fastLowess Julia API Reference

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
result = fit(model, x::Vector{Float64}, y::Vector{Float64}) :: LowessResult
```

* Fits the model to the provided `x` and `y` data vectors.
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
result = add_points(online, x::Vector{Float64}, y::Vector{Float64}) :: LowessResult
```

* Adds new points to the model and returns the smoothed values (retrospective or prospective depending on mode).

## Options Structures

### `LowessOptions`

| Field                       | Type              | Default            | Description                           |
| --------------------------- | ----------------- | ------------------ | ------------------------------------- |
| `fraction`                  | `Float64`         | `0.67`             | Smoothing fraction (bandwidth)        |
| `iterations`                | `Int`             | `3`                | Number of robustifying iterations     |
| `delta`                     | `Float64`         | `NaN`              | Interpolation distance (NaN for auto) |
| `weight_function`           | `String`          | `"tricube"`        | Weight function name                  |
| `robustness_method`         | `String`          | `"bisquare"`       | Robustness method name                |
| `scaling_method`            | `String`          | `"mad"`            | Residual scaling method               |
| `boundary_policy`           | `String`          | `"extend"`         | Boundary handling policy              |
| `zero_weight_fallback`      | `String`          | `"use_local_mean"` | Zero-weight handling strategy         |
| `auto_converge`             | `Float64`         | `NaN`              | Auto-convergence tolerance            |
| `confidence_intervals`      | `Float64`         | `NaN`              | Confidence level (e.g., 0.95)         |
| `prediction_intervals`      | `Float64`         | `NaN`              | Prediction level (e.g., 0.95)         |
| `return_diagnostics`        | `Bool`            | `false`            | Include diagnostics in result         |
| `return_residuals`          | `Bool`            | `false`            | Include residuals in result           |
| `return_robustness_weights` | `Bool`            | `false`            | Include weights in result             |
| `parallel`                  | `Bool`            | `true`             | Enable parallel execution             |
| `cv_method`                 | `String`          | `""`               | Cross-validation method ("kfold")     |
| `cv_k`                      | `Int`             | `5`                | Number of CV folds                    |
| `cv_fractions`              | `Vector{Float64}` | `[]`               | Manual fractions for CV grid          |

### `StreamingOptions` (inherits `LowessOptions`)

| Field            | Type     | Default      | Description                |
| ---------------- | -------- | ------------ | -------------------------- |
| `chunk_size`     | `Int`    | `5000`       | Data chunk size            |
| `overlap`        | `Int`    | `500`        | Overlap size (-1 for auto) |
| `merge_strategy` | `String` | `"weighted"` | Merge strategy for overlap |

### `OnlineOptions` (inherits `LowessOptions`)

| Field             | Type     | Default         | Description                           |
| ----------------- | -------- | --------------- | ------------------------------------- |
| `window_capacity` | `Int`    | `1000`          | Max window size                       |
| `min_points`      | `Int`    | `2`             | Min points before smoothing           |
| `update_mode`     | `String` | `"incremental"` | Update mode ("full" or "incremental") |

## Result Structure

### `LowessResult`

| Field                | Type              | Description               |
| -------------------- | ----------------- | ------------------------- |
| `x`                  | `Vector{Float64}` | Smoothed X coordinates    |
| `y`                  | `Vector{Float64}` | Smoothed Y coordinates    |
| `valid`              | `Bool`            | True if result is valid   |
| `error`              | `String`          | Error message if failed   |
| `diagnostics`        | `Diagnostics`     | Diagnostic metrics struct |
| `residuals`          | `Vector{Float64}` | Residuals (if requested)  |
| `confidence_lower`   | `Vector{Float64}` | Lower CI bounds           |
| `confidence_upper`   | `Vector{Float64}` | Upper CI bounds           |
| `prediction_lower`   | `Vector{Float64}` | Lower PI bounds           |
| `prediction_upper`   | `Vector{Float64}` | Upper PI bounds           |
| `robustness_weights` | `Vector{Float64}` | Robustness weights        |

### `Diagnostics`

| Field          | Type      | Description                 |
| -------------- | --------- | --------------------------- |
| `rmse`         | `Float64` | Root Mean Squared Error     |
| `mae`          | `Float64` | Mean Absolute Error         |
| `r_squared`    | `Float64` | R-squared                   |
| `residual_sd`  | `Float64` | Residual standard deviation |
| `effective_df` | `Float64` | Effective degrees of freedom|
| `aic`          | `Float64` | AIC                         |
| `aicc`         | `Float64` | AICc                        |

## String Options

### Weight Functions

* `"tricube"` (default)
* `"epanechnikov"`
* `"gaussian"`
* `"uniform"`
* `"biweight"`
* `"triangle"`
* `"cosine"`

### Robustness Methods

* `"bisquare"` (default)
* `"huber"`
* `"talwar"`

### Boundary Policies

* `"extend"` (default - linear extrapolation)
* `"reflect"`
* `"zero"`
* `"noboundary"`

### Scaling Methods

* `"mad"` (default - Median Absolute Deviation)
* `"mar"` (Median Absolute Residual)
* `"mean"` (Mean Absolute Residual)

### Zero Weight Fallback

* `"use_local_mean"` (default)
* `"return_original"`
* `"return_none"`

### Merge Strategies (Streaming)

* `"weighted"` (default - weighted average of overlapping chunks)
* `"average"`
* `"left"`
* `"right"`

### Update Modes (Online)

* `"full"` (default - re-smooth entire window)
* `"incremental"` (O(1) update using existing fit)

## Example

```julia
using FastLOWESS

x = collect(range(0, 10, length=100))
y = sin.(x) .+ randn(100) .* 0.2

# Configure model
model = Lowess(fraction=0.3, iterations=3)

# Fit data
result = fit(model, x, y)

if !isempty(result.error)
    println("Error: ", result.error)
else
    println("Smoothed Y: ", result.y)
end
```
