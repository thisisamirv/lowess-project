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

#### `fit(model, x, y; custom_weights=nothing)`

Fits the model to the provided `x` and `y` vectors. `custom_weights` is an optional `Vector{Float64}` of per-observation weights — all values must be ≥ 0 and length must match `x`. Returns a `LowessResult` containing the smoothed values and optional diagnostics.

```@example batch
using Random, Statistics

rng = MersenneTwister(42)
x = collect(range(0, 2π, length=100))
y = sin.(x) .+ randn(rng, 100) .* 0.3

result = fit(model, x, y)
println("First smoothed value: ", result.y[1])
```

## Options Structures

### `Lowess` keyword arguments

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `fraction` | `Float64` | `0.67` | Smoothing fraction (bandwidth) |
| `iterations` | `Int` | `3` | Number of robustifying iterations |
| `weight_function` | `String` | `"tricube"` | Weight function name |
| `robustness_method` | `String` | `"bisquare"` | Robustness method name |
| `delta` | `Float64` | `NaN` | Interpolation distance (`NaN` auto-sets it to 1% of the x-range) |
| `zero_weight_fallback` | `String` | `"use_local_mean"` | Zero-weight handling strategy |
| `boundary_policy` | `String` | `"extend"` | Boundary handling policy |
| `scaling_method` | `String` | `"mad"` | Residual scaling method |
| `auto_converge` | `Float64` | `NaN` | Auto-convergence tolerance |
| `missing` | `String` | `"error"` | Policy for non-finite (NaN/Inf) values in input data |
| `parallel` | `Bool` | `true` | Enable parallel execution |
| `backend` | `String` | `"cpu"` | Execution backend (`"cpu"` or `"gpu"`); GPU requires the library to be built with the `gpu` Cargo feature |
| `outputs` | `Vector{String}` | `String[]` | Select `se`, `diagnostics`, `residuals`, `weights`, `derivative`, and/or `sorted` |
| `intervals` | `NamedTuple` | `nothing` | Grouped interval options: `confidence`, `prediction`, and `bootstrap` |
| `cv` | `NamedTuple` | `nothing` | Grouped CV options: `method`, `k`, and `fractions` |
| `seed` | `Union{Integer, Nothing}` | `nothing` | Shared CV/bootstrap seed; `0` is a valid seed |
| `retain_model` | `Bool` | `false` | Retain training data, enabling `predict(model, new_x; ...)` on the result |
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

### delta

Points within `delta` of each other on the x-axis share the same local fit instead of each computing its own regression — an interpolation shortcut that trades a small amount of accuracy for a large speedup on dense, evenly-spaced data. `NaN` (default) auto-sets it to 1% of the x-range. Set it to `0.0` explicitly to disable interpolation and fit every point exactly.

### zero_weight_fallback

Behavior when all neighborhood weights are zero:

| Option | Behavior |
| --- | --- |
| `"use_local_mean"` (default; aliases: `"local_mean"`, `"mean"`) | Use the mean of the neighborhood |
| `"return_original"` (alias: `"original"`) | Return the original y value |
| `"return_none"` (alias: `"none"`) | Return `NaN` |

### boundary_policy

*See: [Boundary Handling](../advanced/boundary.md)*

- `"extend"` (default; alias: `"pad"`)
- `"reflect"` (alias: `"mirror"`)
- `"zero"`
- `"noboundary"` (alias: `"none"`)

### scaling_method

*See: [Scaling Methods](../weighting/scaling.md)*

- `"mad"` (default; alias: `"median_absolute_deviation"`)
- `"mar"` (alias: `"median_absolute_residual"`)
- `"mean"` (alias: `"mean_absolute_residual"`)

### auto_converge

*See: [Robustness](../weighting/robustness.md#auto-convergence)*

Convergence tolerance for early stopping of robustness iterations. `NaN` (default) disables early stopping.

### missing

Policy for handling non-finite (NaN/Inf) values in `x`/`y` (and `custom_weights`):

| Option | Behavior |
| --- | --- |
| `"error"` (default) | Raise an error if any value is non-finite |
| `"drop"` | Silently remove observations where `x` or `y` is non-finite before fitting |

**Note:** A length mismatch between `x` and `y` always errors, even under `"drop"`.

### parallel

Enable multi-threaded execution via Rayon.

- `true` (default) — parallelizes the local regression fits across CPU cores
- `false` — forces single-threaded execution (useful for benchmarking or deterministic profiling)

### backend

*See: [GPU Backend](../advanced/gpu-backend.md)*

The batch `Lowess` type can optionally run on a GPU-accelerated backend powered by `wgpu`, for high-throughput processing of large datasets (10k+ points).

- `"cpu"` (default)
- `"gpu"` — requires the library to be built with the `gpu` Cargo feature

### outputs: se

*See: [Intervals](../guide/intervals.md#standard-errors)*

Select `"se"` to compute hat-matrix statistics (effective degrees of freedom, leverage, delta1/delta2) and standard errors.

### outputs: diagnostics

*See: [`Diagnostics`](#diagnostics)*

Select `"diagnostics"` to include a `Diagnostics` object (RMSE, MAE, R2, AIC/AICc, effective degrees of freedom) in the result. AIC/AICc/`effective_df` additionally require `"se"` (or confidence/prediction intervals) to be selected, since they depend on hat-matrix statistics.

### outputs: residuals

Include per-point residuals (`y - fitted`) in the result.

### outputs: weights

Include the final per-point robustness weights (from the last robustness iteration) in the result.

### outputs: derivative

Each point's local WLS fit already computes a slope internally; this exposes that per-point slope (rate of change of the smoothed curve) in `result.derivative`, enabling turning-point/rate-of-change analysis at effectively no extra computation cost.

### outputs: sorted

When set to `true`, it reorders every result field (residuals, intervals, etc.) by `x` in an ascending manner, instead of in original input order. To get both orderings, sort the default result client-side (e.g. `sortperm(result.x)`) instead of calling `fit` twice.

### intervals

*See: [Intervals](../guide/intervals.md)*

A `NamedTuple` such as `(confidence=0.90, prediction=0.99, bootstrap=200)`. Confidence and prediction coverage levels are independent: `confidence` bounds the mean response and `prediction` bounds a new observation; omitted levels disable that bound. `bootstrap` (at least `2`) replaces analytic intervals with residual-bootstrap standard errors and percentile bounds. `seed` controls fit-time bootstrap draws but does not enable bootstrap by itself.

### CV Options

*See: [Cross-Validation](../guide/cross-validation.md)*

A `NamedTuple` such as `(method="kfold", k=5, fractions=[0.2, 0.3, 0.5])`:

- `method`: `"kfold"` (default) — fast, evaluates each candidate fraction over `k` folds; `"loocv"` — slow, exhaustive leave-one-out cross-validation
- `k`: Number of folds for k-fold CV. Ignored when `method="loocv"`.
- `fractions`: Candidate fractions to evaluate. Required.

Seed k-fold shuffling with the outer `seed` keyword, not inside `cv`.

### seed

One seed shared by k-fold CV shuffling and fit-time residual bootstrap. It does not enable either feature by itself; `nothing` (default) uses each feature's default, and `0` is a valid seed.

### retain_model

*See: [Predict](../guide/predict.md)*

Retains the fitted model's training data, populating `result.predict_model` with a `PredictModel` usable to evaluate the fit at out-of-sample query points not in the training set. `false` (default) — no extra memory/copy cost unless requested.

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
| `residuals` | `Union{Vector{Float64}, Nothing}` | Residuals (if `"residuals"` was requested) |
| `robustness_weights` | `Union{Vector{Float64}, Nothing}` | Robustness weights (if `"weights"` was requested) |
| `cv_scores` | `Union{Vector{Float64}, Nothing}` | CV score per tested fraction |
| `diagnostics` | `Union{Diagnostics, Nothing}` | Fit metrics (if `"diagnostics"` was requested) |
| `derivative` | `Union{Vector{Float64}, Nothing}` | Per-point local fit derivative/slope (if `"derivative"` was requested) |

### `Diagnostics`

| Field | Type | Description |
| --- | --- | --- |
| `rmse` | `Float64` | Root Mean Squared Error |
| `mae` | `Float64` | Mean Absolute Error |
| `r_squared` | `Float64` | R-squared |
| `residual_sd` | `Float64` | Residual standard deviation |
| `effective_df` | `Union{Float64, Nothing}` | Effective degrees of freedom, or `nothing` if unavailable |
| `aic` | `Union{Float64, Nothing}` | AIC, or `nothing` if unavailable |
| `aicc` | `Union{Float64, Nothing}` | AICc, or `nothing` if unavailable |

## Example

```@example batch-example
using FastLOWESS
using Random, Statistics

rng = MersenneTwister(42)
x = collect(range(0, 2π, length=100))
y = sin.(x) .+ randn(rng, 100) .* 0.3

model = Lowess(;
    fraction=0.5,
    iterations=3,
    parallel=true,
    outputs=["diagnostics"],
    intervals=(confidence=0.90, prediction=0.99),
)
result = fit(model, x, y)
println("First smoothed value: ", result.y[1])
```

---
