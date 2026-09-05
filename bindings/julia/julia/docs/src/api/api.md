# Batch Adapter

The Julia bindings provide a high-performance interface to the core Rust library, mirroring the Rust API structure.

> **StreamingLowess** and **OnlineLowess** are documented separately: [Streaming Adapter](api-streaming.md), [Online Adapter](api-online.md)

## When to Use Batch Adapter

- Dataset fits in memory
- Need intervals, cross-validation, or diagnostics
- Processing complete files

## Classes

### `Lowess`

The `Lowess` type allows configuring the LOWESS parameters once and fitting multiple datasets using those parameters.

**Constructor:**

```@example batch
using FastLOWESS

model = Lowess(; fraction=0.5, iterations=3)
println(typeof(model))
```

- Keyword arguments configure the `Lowess` model; see [Options Structures](#options-structures) below.

**Methods:**

```@example batch
using Random, Statistics

rng = MersenneTwister(42)
x = collect(range(0, 2π, length=100))
y = sin.(x) .+ randn(rng, 100) .* 0.3

result = fit(model, x, y)
println("First smoothed value: ", result.y[1])
```

- `fit(model, x, y; custom_weights=nothing)`: Fits the model to the provided `x` and `y` vectors.
- `custom_weights`: Optional `Vector{Float64}` of per-observation weights. All values must be ≥ 0 and length must match `x`.
- Returns a `LowessResult` containing the smoothed values and optional diagnostics.

## Options Structures

### `Lowess` keyword arguments

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `fraction` | `Float64` | `0.67` | Smoothing fraction (bandwidth) |
| `iterations` | `Int` | `3` | Number of robustifying iterations |
| `delta` | `Float64` | `NaN` | Interpolation distance (`NaN` auto-sets it to 1% of the x-range) |
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
| `return_sorted` | `Bool` | `false` | Return results sorted ascending by `x` instead of in original input order |
| `parallel` | `Bool` | `true` | Enable parallel execution |
| `backend` | `String` | `"cpu"` | Execution backend (`"cpu"` or `"gpu"`); GPU requires the library to be built with the `gpu` Cargo feature |
| `cv_method` | `String` | `"kfold"` | CV method (`"kfold"` fast or `"loocv"` slow, exhaustive) |
| `cv_k` | `Int` | `5` | Number of folds for k-fold CV |
| `cv_fractions` | `Vector{Float64}` | `Float64[]` | Fractions to test for cross-validation |
| `cv_seed` | `Union{Int, Nothing}` | `nothing` | Random seed for cross-validation shuffling |
| `custom_weights` | `Vector{Float64}` | `nothing` | Per-observation case weights — passed to `fit`, not the constructor |

## Options

### fraction

`fraction` is the most important parameter: it controls the size of the local neighbourhood used at each point.

| Range | Effect | Use case |
| --- | --- | --- |
| 0.1-0.3 | Fine detail | Rapidly changing signals |
| 0.3-0.5 | Balanced | General purpose |
| 0.5-0.7 | Heavy smoothing | Noisy data |
| 0.7-1.0 | Very smooth | Trend extraction |

### iterations

`iterations` controls robustness to outliers, at the cost of speed.

| Value | Effect | Performance |
| --- | --- | --- |
| 0 | No robustness | Fastest |
| 1-3 | Moderate | Recommended |
| 4-6 | Strong | Contaminated data |
| 7+ | Very strong | Heavy outliers |

### delta

Points within `delta` of each other on the x-axis share the same local fit instead of each computing its own regression — an interpolation shortcut that trades a small amount of accuracy for a large speedup on dense, evenly-spaced data. `NaN` (default) auto-sets it to 1% of the x-range. Set it to `0.0` explicitly to disable interpolation and fit every point exactly.

### weight_function

*See: [Weight Functions](../weighting/kernels.md)*

- `"tricube"` (default)
- `"epanechnikov"`
- `"gaussian"`
- `"uniform"` (alias: `"boxcar"`)
- `"biweight"` (alias: `"bisquare"`)
- `"triangle"` (alias: `"triangular"`)
- `"cosine"`

### robustness_method

*See: [Robustness](../weighting/robustness.md)*

- `"bisquare"` (default; alias: `"biweight"`)
- `"huber"`
- `"talwar"`

### scaling_method

*See: [Scaling Methods](../weighting/scaling.md)*

- `"mad"` (default; alias: `"median_absolute_deviation"`)
- `"mar"` (alias: `"median_absolute_residual"`)
- `"mean"` (alias: `"mean_absolute_residual"`)

### boundary_policy

*See: [Boundary Handling](../advanced/boundary.md)*

- `"extend"` (default; alias: `"pad"`)
- `"reflect"` (alias: `"mirror"`)
- `"zero"`
- `"noboundary"` (alias: `"none"`)

### zero_weight_fallback

Behavior when all neighborhood weights are zero:

| Option | Behavior |
| --- | --- |
| `"use_local_mean"` (default; aliases: `"local_mean"`, `"mean"`) | Use the mean of the neighborhood |
| `"return_original"` (alias: `"original"`) | Return the original y value |
| `"return_none"` (alias: `"none"`) | Return `NaN` |

### auto_converge

*See: [Robustness](../weighting/robustness.md#auto-convergence)*

Convergence tolerance for early stopping of robustness iterations. `NaN` (default) disables early stopping.

### confidence_intervals

*See: [Intervals](../guide/intervals.md)*

Confidence level for the confidence interval around the mean response (e.g. `0.95`). `NaN` (default) disables confidence intervals.

### prediction_intervals

*See: [Intervals](../guide/intervals.md)*

Confidence level for the prediction interval for new observations (e.g. `0.95`). `NaN` (default) disables prediction intervals.

### return_diagnostics

*See: [`Diagnostics`](#diagnostics)*

Include a `Diagnostics` object (RMSE, MAE, R2, AIC/AICc, effective degrees of freedom) in the result. AIC/AICc/`effective_df` additionally require `return_se=true` (or confidence/prediction intervals) to be populated, since they depend on hat-matrix statistics.

- `false` (default) — leaves `result.diagnostics` as `nothing`
- `true` — populates `result.diagnostics`

### return_residuals

Include per-point residuals (`y - fitted`) in the result.

- `false` (default) — leaves `result.residuals` as `nothing`
- `true` — populates `result.residuals`

### return_robustness_weights

Include the final per-point robustness weights (from the last robustness iteration) in the result.

- `false` (default) — leaves `result.robustness_weights` as `nothing`
- `true` — populates `result.robustness_weights`

### return_se

*See: [Intervals](../guide/intervals.md#standard-errors)*

Computes hat-matrix statistics (effective degrees of freedom, leverage, delta1/delta2) in addition to standard errors.

### return_sorted

When set to `true`, it reorders every result field (residuals, intervals, etc.) by `x` in an ascending manner, instead of in original input order.
To get both orderings, sort the default result client-side (e.g. `sortperm(result.x)`) instead of calling `fit` twice.

### parallel

Enable multi-threaded execution via Rayon.

- `true` (default) — parallelizes the local regression fits across CPU cores
- `false` — forces single-threaded execution (useful for benchmarking or deterministic profiling)

### backend

*See: [GPU Backend](../advanced/gpu-backend.md)*

The batch `Lowess` type can optionally run on a GPU-accelerated backend powered by `wgpu`, for high-throughput processing of large datasets (10k+ points).

- `"cpu"` (default)
- `"gpu"` — requires the library to be built with the `gpu` Cargo feature

### CV Options

*See: [Cross-Validation](../guide/cross-validation.md)*

- `cv_method`: `"kfold"` (default) — fast, evaluates each candidate fraction over `cv_k` folds; `"loocv"` — slow, exhaustive leave-one-out cross-validation
- `cv_k`: Number of folds for k-fold CV. Ignored when `cv_method="loocv"`.
- `cv_fractions`: Candidate fractions to evaluate. Cross-validation is disabled unless this is set.
- `cv_seed`: Seed for reproducible k-fold shuffling. `nothing` (default) uses a random seed.

### custom_weights

*See: [Custom Weights](../weighting/custom-weights.md)*

Per-observation weights, passed to `fit` rather than the constructor.

## Result Structure

### `LowessResult`

| Field | Type | Description |
| --- | --- | --- |
| `x` | `Vector{Float64}` | x values (same order as input) |
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
| `effective_df` | `Float64` | Effective degrees of freedom |
| `aic` | `Float64` | AIC |
| `aicc` | `Float64` | AICc |

## Example

```@example batch
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
    return_diagnostics=true,
    parallel=true
)
result = fit(model, x, y)
println("First smoothed value: ", result.y[1])
```

---
