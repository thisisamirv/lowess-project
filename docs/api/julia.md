# Julia API

API reference for the `fastLowess` Julia package.

## Installation

Install from the Julia General Registry:

```julia
using Pkg
Pkg.add("fastLowess")
```

---

## Functions

### smooth

Main function for batch smoothing.

```julia
smooth(
    x::Vector{Float64},
    y::Vector{Float64};
    fraction::Float64 = 0.67,
    iterations::Int = 3,
    delta::Float64 = NaN,
    parallel::Bool = true,
    weight_function::String = "tricube",
    robustness_method::String = "bisquare",
    scaling_method::String = "mad",
    zero_weight_fallback::String = "use_local_mean",
    boundary_policy::String = "extend",
    auto_converge::Float64 = NaN,
    return_residuals::Bool = false,
    return_diagnostics::Bool = false,
    return_robustness_weights::Bool = false,
    confidence_intervals::Float64 = NaN,
    prediction_intervals::Float64 = NaN,
    cv_method::String = "",
    cv_k::Int = 5,
    cv_fractions::Vector{Float64} = Float64[],
    cv_seed::Int = -1,
) -> LowessResult
```

**Parameters:**

| Parameter    | Type    | Default  | Description             |
|--------------|---------|----------|-------------------------|
| `x`          | Vector  | required | Independent variable    |
| `y`          | Vector  | required | Dependent variable      |
| `fraction`   | Float64 | 0.67     | Smoothing span (0, 1]   |
| `iterations` | Int     | 3        | Robustness iterations   |
| `delta`      | Float64 | NaN      | Interpolation threshold |
| `parallel`   | Bool    | true     | Enable parallelism      |

**Returns:** `LowessResult` struct with fields:

| Field                | Type                       | Description                         |
|----------------------|----------------------------|-------------------------------------|
| `x`                  | Vector{Float64}            | Input x values                      |
| `y`                  | Vector{Float64}            | Smoothed y values                   |
| `fraction_used`      | Float64                    | Actual fraction used                |
| `residuals`          | Union{Vector,Nothing}      | If `return_residuals=true`          |
| `confidence_lower`   | Union{Vector,Nothing}      | If `confidence_intervals` set       |
| `confidence_upper`   | Union{Vector,Nothing}      | If `confidence_intervals` set       |
| `prediction_lower`   | Union{Vector,Nothing}      | If `prediction_intervals` set       |
| `prediction_upper`   | Union{Vector,Nothing}      | If `prediction_intervals` set       |
| `robustness_weights` | Union{Vector,Nothing}      | If `return_robustness_weights=true` |
| `diagnostics`        | Union{Diagnostics,Nothing} | If `return_diagnostics=true`        |

**Example:**

```julia
using fastLowess

x = collect(range(0, 10, length=100))
y = sin.(x) .+ randn(100) .* 0.2

result = smooth(x, y, fraction=0.3, iterations=3)
println(result.y)
```

---

### smooth_streaming

Streaming mode for large datasets.

```julia
smooth_streaming(
    x::Vector{Float64},
    y::Vector{Float64};
    fraction::Float64 = 0.67,
    iterations::Int = 3,
    chunk_size::Int = 5000,
    overlap::Int = 500,
    merge_strategy::String = "average",
    parallel::Bool = true,
    # ... same kwargs as smooth()
) -> LowessResult
```

**Additional Parameters:**

| Parameter        | Type   | Default   | Description            |
|------------------|--------|-----------|------------------------|
| `chunk_size`     | Int    | 5000      | Points per chunk       |
| `overlap`        | Int    | 500       | Overlap between chunks |
| `merge_strategy` | String | "average" | How to merge overlaps  |

**Example:**

```julia
# Process 1 million points
x = collect(range(0, 1000, length=1_000_000))
y = sin.(x ./ 100) .+ randn(1_000_000) .* 0.1

result = smooth_streaming(x, y, chunk_size=10000, overlap=1000)
```

---

### smooth_online

Online mode for real-time data.

```julia
smooth_online(
    x::Vector{Float64},
    y::Vector{Float64};
    fraction::Float64 = 0.2,
    window_capacity::Int = 100,
    min_points::Int = 2,
    iterations::Int = 3,
    update_mode::String = "incremental",
    # ... same kwargs as smooth()
) -> LowessResult
```

**Additional Parameters:**

| Parameter         | Type   | Default       | Description          |
|-------------------|--------|---------------|----------------------|
| `window_capacity` | Int    | 100           | Max points in window |
| `min_points`      | Int    | 2             | Points before output |
| `update_mode`     | String | "incremental" | Update strategy      |

**Example:**

```julia
# Sensor data simulation
sensor_times = collect(Float64, 0:99)
sensor_values = 20 .+ 5 .* sin.(sensor_times ./ 10) .+ randn(100)

result = smooth_online(sensor_times, sensor_values, window_capacity=25)
```

---

## Types

### LowessResult

```julia
struct LowessResult
    x::Vector{Float64}
    y::Vector{Float64}
    fraction_used::Float64
    residuals::Union{Vector{Float64}, Nothing}
    standard_errors::Union{Vector{Float64}, Nothing}
    confidence_lower::Union{Vector{Float64}, Nothing}
    confidence_upper::Union{Vector{Float64}, Nothing}
    prediction_lower::Union{Vector{Float64}, Nothing}
    prediction_upper::Union{Vector{Float64}, Nothing}
    robustness_weights::Union{Vector{Float64}, Nothing}
    diagnostics::Union{Diagnostics, Nothing}
    iterations_used::Union{Int, Nothing}
end
```

### Diagnostics

```julia
struct Diagnostics
    rmse::Float64         # Root Mean Square Error
    mae::Float64          # Mean Absolute Error
    r_squared::Float64    # R² coefficient
    residual_sd::Float64  # Residual standard deviation
    effective_df::Float64 # Effective degrees of freedom
end
```

---

## String Options

### weight_function

- `"tricube"` (default)
- `"epanechnikov"`
- `"gaussian"`
- `"biweight"`
- `"cosine"`
- `"triangle"`
- `"uniform"`

### robustness_method

- `"bisquare"` (default)
- `"huber"`
- `"talwar"`

### boundary_policy

- `"extend"` (default)
- `"reflect"`
- `"zero"`
- `"no_boundary"`

### merge_strategy

- `"average"` (default)
- `"left"`
- `"right"`
- `"weighted"`

### update_mode

- `"incremental"` (default)
- `"full"`
