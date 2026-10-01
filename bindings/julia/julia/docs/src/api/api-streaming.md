# Streaming Adapter

Process large datasets in chunks with configurable overlap.

See also: [Batch Adapter](api.md)

## When to Use

- Dataset >100,000 points
- Memory-constrained environments
- Batch processing pipelines

## Class

### `StreamingLowess`

The `StreamingLowess` type processes data in chunks, suitable for very large datasets or streaming applications.

**Constructor:**

```@example streaming
using FastLOWESS

model = StreamingLowess(; fraction=0.3, chunk_size=5000, overlap=500)
println(typeof(model))
```

- Keyword arguments configure the `StreamingLowess` model; see [Options Structure](#options-structure) below.

**Methods:**

#### `process_chunk(model, x, y)`

Feeds one chunk of data into the model. Each chunk is fit together with the trailing `overlap` points buffered from the previous call, then only the points that are fully resolved are returned — the tail of the chunk (the next `overlap` points) is held back internally, since it will be refit once the following chunk arrives and its estimate reconciled via `merge_strategy`. This is what lets the adapter process a dataset far larger than memory allows, one bounded-size chunk at a time, without ever materializing the whole dataset at once.

```@example streaming
using Random, Statistics

rng = MersenneTwister(42)
x = collect(range(0, 2π, length=100))
y = sin.(x) .+ randn(rng, 100) .* 0.3

process_chunk(model, x, y)
println("Called process_chunk")
```

#### `finalize(model)`

Flushes the overlap points still buffered from the last `process_chunk` call. Because each call withholds its tail until the next chunk arrives to resolve it, the final chunk's tail would never be emitted otherwise — always call `finalize` once after the last chunk to retrieve it.

```@example streaming
result = finalize(model)
println("First smoothed value: ", result.y[1])
```

## Options Structure

### `StreamingLowess` keyword arguments (mirrors `Lowess`)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `fraction` | `Float64` | `0.67` | Smoothing fraction (bandwidth) |
| `iterations` | `Int` | `3` | Number of robustifying iterations |
| `weight_function` | `String` | `"tricube"` | Weight function name |
| `robustness_method` | `String` | `"bisquare"` | Robustness method name |
| `delta` | `Float64` | `NaN` | Interpolation distance (`NaN` auto-sets it to 0.0 in Streaming, i.e. interpolation disabled) |
| `zero_weight_fallback` | `String` | `"use_local_mean"` | Zero-weight handling |
| `boundary_policy` | `String` | `"extend"` | Boundary handling policy |
| `scaling_method` | `String` | `"mad"` | Residual scaling method |
| `auto_converge` | `Float64` | `NaN` | Auto-convergence tolerance |
| `missing` | `String` | `"error"` | Policy for non-finite (NaN/Inf) values in each chunk |
| `chunk_size` | `Int` | `5000` | Points per chunk |
| `overlap` | `Int` | `chunk_size / 10` | Overlap between chunks |
| `merge_strategy` | `String` | `"weighted_average"` | Strategy for blending overlap regions |
| `parallel` | `Bool` | `true` | Enable parallel execution |
| `outputs` | `Vector{String}` | `String[]` | Select `se`, `diagnostics`, `residuals`, `weights`, and/or `derivative` |
| `intervals` | `NamedTuple` | `nothing` | Grouped `confidence`, `prediction`, and per-chunk `bootstrap` options |
| `seed` | `Union{Integer, Nothing}` | `nothing` | Reproducible bootstrap draws for each combined chunk |
| `return_se` | `Bool` | `false` | Populate `standard_errors` in the result |
| `return_diagnostics` | `Bool` | `false` | Include diagnostics in result |
| `return_residuals` | `Bool` | `false` | Include residuals in result |
| `return_robustness_weights` | `Bool` | `false` | Include weights in result |
| `return_derivative` | `Bool` | `false` | Include the per-point local fit derivative (slope) in result |

Cross-validation, GPU `backend`, `custom_weights`, and `return_sorted` are Batch-only and not available here; see [Batch Adapter](api.md) for those. Standard errors and confidence/prediction intervals are computed per combined chunk (including the previous overlap), then blended across overlap regions via `merge_strategy` like `y`/`derivative` are; they are local chunk intervals, not whole-stream intervals.

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

Points within `delta` of each other on the x-axis share the same local fit instead of each computing its own regression — an interpolation shortcut that trades a small amount of accuracy for a large speedup on dense, evenly-spaced data. `NaN` (default) auto-sets it to `0` in Streaming mode, i.e. interpolation is disabled and every point is fit exactly.

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

Policy for handling non-finite (NaN/Inf) values within each chunk:

| Option | Behavior |
| --- | --- |
| `"error"` (default) | Raise an error if any value in the chunk is non-finite |
| `"drop"` | Silently remove rows where `x` or `y` is non-finite before merging the chunk with the overlap buffer |

**Note:** A length mismatch between `x` and `y` always errors, even under `"drop"`.

### chunk_size

Number of points processed per chunk. Larger chunks reduce per-chunk overhead and give each local fit more surrounding context, at the cost of higher peak memory; smaller chunks bound memory tightly but increase the fraction of points that fall in overlap regions. A good starting point is balancing available memory against how much processing overhead per chunk is acceptable — match it to your file-read buffer or message-batch size to avoid unnecessary copying.

### overlap

Number of points retained from the previous chunk as context, so the neighbourhood at chunk boundaries isn't artificially truncated. Points inside the overlap zone are fitted twice (once by each chunk) and reconciled via `merge_strategy`. A good starting point is 10–20% of `chunk_size`: too little overlap causes visible boundary artefacts, while too much wastes computation refitting the same points twice.

- `-1` (default) — computes `chunk_size / 10`, clamped to at least 1 and less than `chunk_size`
- Any integer `>= 1` and `< chunk_size`

### merge_strategy

*See: [Merge Strategies](../advanced/merge.md)*

| Strategy | Alias | Behavior |
| --- | --- | --- |
| `"weighted_average"` (default) | `"weighted"` | Distance-weighted blend |
| `"average"` | `"mean"` | Average overlapping values |
| `"take_first"` | `"first"` | Keep left chunk values |
| `"take_last"` | `"last"` | Keep right chunk values |

### parallel

Enable multi-threaded execution via Rayon.

- `true` (default) — parallelizes the local regression fits across CPU cores
- `false` — forces single-threaded execution

### intervals

*See: [Intervals](../guide/intervals.md)*

A `NamedTuple` such as `(confidence=0.95, prediction=0.95, bootstrap=200)`, populating `result.confidence_lower`/`result.confidence_upper` and `result.prediction_lower`/`result.prediction_upper`. Computed per combined chunk and merged across overlap boundaries via `merge_strategy`. `bootstrap` (at least `2`) refits each combined chunk from resampled residuals; with `parallel=true`, refits run concurrently.

### seed

Seeds bootstrap draws. Each combined chunk restarts from the same seed. It does not enable bootstrap by itself; `0` is a valid seed.

### return_se

*See: [Intervals](../guide/intervals.md)*

Computes standard errors per chunk the same way Batch does, then merges the overlap region across chunk boundaries the same way `y`/`derivative` are, via `merge_strategy`.

- `false` (default) — leaves `result.standard_errors` as `nothing`
- `true` — populates it

### return_diagnostics

*See: [`Diagnostics`](#diagnostics)*

Include a `Diagnostics` object (RMSE, MAE, R2, residual_sd) in the result. `effective_df`/`aic`/`aicc` require per-chunk hat-matrix leverage to be threaded into the cumulative diagnostics computation across chunk boundaries, which isn't currently done (even with `return_se` or `intervals` set), so they're always `NaN` here.

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

### return_derivative

Each point's local WLS fit already computes a slope internally; this exposes that per-point slope (rate of change of the smoothed curve) in `result.derivative` at effectively no extra computation cost. Derivative values in the overlap region are merged across chunk boundaries the same way `y` is, via `merge_strategy`.

- `false` (default) — leaves `result.derivative` as `nothing`
- `true` — populates it

## Result Structure

### `LowessResult`

Returned by `process_chunk` and `finalize`.

| Field | Type | Description |
| --- | --- | --- |
| `x` | `Vector{Float64}` | x values (same order as input) |
| `y` | `Vector{Float64}` | Smoothed y values |
| `fraction_used` | `Float64` | Fraction used |
| `iterations_used` | `Union{Int, Nothing}` | Robustness iterations actually performed |
| `standard_errors` | `Union{Vector{Float64}, Nothing}` | Per-point standard errors (if `return_se` or any interval was set) |
| `confidence_lower` | `Union{Vector{Float64}, Nothing}` | Lower confidence bounds (if `intervals.confidence` was set) |
| `confidence_upper` | `Union{Vector{Float64}, Nothing}` | Upper confidence bounds (if `intervals.confidence` was set) |
| `prediction_lower` | `Union{Vector{Float64}, Nothing}` | Lower prediction bounds (if `intervals.prediction` was set) |
| `prediction_upper` | `Union{Vector{Float64}, Nothing}` | Upper prediction bounds (if `intervals.prediction` was set) |
| `residuals` | `Union{Vector{Float64}, Nothing}` | Residuals (if `return_residuals`) |
| `robustness_weights` | `Union{Vector{Float64}, Nothing}` | Robustness weights (if `return_robustness_weights`) |
| `cv_scores` | `Union{Vector{Float64}, Nothing}` | Always `nothing` (Batch only) |
| `diagnostics` | `Union{Diagnostics, Nothing}` | Fit metrics (if `return_diagnostics`) |
| `derivative` | `Union{Vector{Float64}, Nothing}` | Per-point local fit derivative/slope (if `return_derivative`) |

### `Diagnostics`

| Field | Type | Description |
| --- | --- | --- |
| `rmse` | `Float64` | Root Mean Squared Error |
| `mae` | `Float64` | Mean Absolute Error |
| `r_squared` | `Float64` | R-squared |
| `residual_sd` | `Float64` | Residual standard deviation |
| `effective_df` | `Union{Float64, Nothing}` | `nothing` (cumulative diagnostics don't integrate per-chunk leverage; Batch only) |
| `aic` | `Union{Float64, Nothing}` | `nothing` (requires `effective_df`; Batch only) |
| `aicc` | `Union{Float64, Nothing}` | `nothing` (requires `effective_df`; Batch only) |

## Example

```@example streaming
using FastLOWESS
using Random, Statistics

rng = MersenneTwister(42)
x = collect(range(0, 2π, length=100))
y = sin.(x) .+ randn(rng, 100) .* 0.3

model = StreamingLowess(;
    fraction=0.3,
    iterations=2,
    chunk_size=5000,
    overlap=500,
    merge_strategy="average"
)
process_chunk(model, x, y)
result = finalize(model)
println("First smoothed value: ", result.y[1])
```

---
