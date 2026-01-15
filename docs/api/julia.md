# Julia API

API reference for the `fastlowess` Julia package.

---

## Classes

### Lowess (Batch)

Stateful class for batch smoothing.

```julia
model = Lowess(;
    fraction::Float64 = 0.67,
    iterations::Int = 3,
    delta::Float64 = NaN,
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
    parallel::Bool = true
)
```

**Methods:**

#### fit

Fit the model to data.

```julia
result = fit(model, x::Vector{Float64}, y::Vector{Float64}) :: LowessResult
```

**Returns:** `LowessResult` struct (see below).

**Example:**

```julia
using fastlowess

x = collect(range(0, 10, length=100))
y = sin.(x) .+ randn(100) .* 0.2

model = Lowess(fraction=0.3, iterations=3)
result = fit(model, x, y)
println(result.y)
```

---

### StreamingLowess

Stateful class for streaming data processing.

```julia
stream = StreamingLowess(;
    fraction::Float64 = 0.67,
    chunk_size::Int = 5000,
    overlap::Int = 500,
    iterations::Int = 3,
    delta::Float64 = NaN,
    weight_function::String = "tricube",
    robustness_method::String = "bisquare",
    scaling_method::String = "mad",
    zero_weight_fallback::String = "use_local_mean",
    boundary_policy::String = "extend",
    auto_converge::Float64 = NaN,
    return_residuals::Bool = false,
    return_diagnostics::Bool = false,
    return_robustness_weights::Bool = false,
    parallel::Bool = true
)
```

**Methods:**

#### process_chunk

Process a chunk of data.

```julia
partial_result = process_chunk(stream, x::Vector{Float64}, y::Vector{Float64}) :: LowessResult
```

#### finalize

Finalize the stream and process remaining buffered data.

```julia
final_result = finalize(stream) :: LowessResult
```

**Example:**

```julia
stream = StreamingLowess(chunk_size=1000)
# ... process chunks ...
r1 = process_chunk(stream, x1, y1)
r2 = process_chunk(stream, x2, y2)
r_final = finalize(stream)
```

---

### OnlineLowess

Stateful class for online (real-time) data processing.

```julia
online = OnlineLowess(;
    fraction::Float64 = 0.2,
    window_capacity::Int = 100,
    min_points::Int = 2,
    iterations::Int = 3,
    update_mode::String = "incremental",
    delta::Float64 = NaN,
    weight_function::String = "tricube",
    robustness_method::String = "bisquare",
    scaling_method::String = "mad",
    zero_weight_fallback::String = "use_local_mean",
    boundary_policy::String = "extend",
    auto_converge::Float64 = NaN,
    return_robustness_weights::Bool = false,
    parallel::Bool = true
)
```

**Methods:**

#### add_points

Add new points to the online processor.

```julia
result = add_points(online, x::Vector{Float64}, y::Vector{Float64}) :: LowessResult
```

**Example:**

```julia
online = OnlineLowess(window_capacity=50)

for (new_x, new_y) in data_source
    res = add_points(online, [new_x], [new_y])
    if !isempty(res.y)
        println("Smoothed: ", res.y[1])
    end
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

### update_mode

- `"incremental"` (default)
- `"full"`

---

## Diagnostics

When `return_diagnostics=true`, the result includes:

```julia
result.diagnostics = Diagnostics(
    rmse::Float64,         # Root Mean Square Error
    mae::Float64,          # Mean Absolute Error
    r_squared::Float64,    # R² coefficient
    residual_sd::Float64,  # Residual standard deviation
    effective_df::Float64  # Effective degrees of freedom
    aic::Float64,          # AIC (NaN usually)
    aicc::Float64          # AICc (NaN usually)
)
```
