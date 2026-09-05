# StreamingLowess API

See also: [fastLowess](api.md)

## When to Use

- Dataset >100,000 points
- Memory-constrained environments
- Batch processing pipelines

## Class

### `StreamingLowess`

The `StreamingLowess` class processes data in chunks, suitable for very large datasets or streaming applications.

**Constructor:**

:::{jupyter-execute}
import fastlowess as fl

stream = fl.StreamingLowess(chunk_size=50, overlap=10)
:::

#### `process_chunk(x, y)`

Feeds one chunk of data into the model. Each chunk is fit together with the trailing `overlap` points buffered from the previous call, then only the points that are fully resolved are returned — the tail of the chunk (the next `overlap` points) is held back internally, since it will be refit once the following chunk arrives and its estimate reconciled via `merge_strategy`. This is what lets the adapter process a dataset far larger than memory allows, one bounded-size chunk at a time, without ever materializing the whole dataset at once.

:::{jupyter-execute}
import fastlowess as fl
import numpy as np

x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x) + 0.1

stream = fl.StreamingLowess(fraction=0.5, chunk_size=50, overlap=10)
partial_result = stream.process_chunk(x[:50], y[:50])
print(partial_result)
:::

#### `finalize()`

Flushes the overlap points still buffered from the last `process_chunk()` call. Because each call withholds its tail until the next chunk arrives to resolve it, the final chunk's tail would never be emitted otherwise — always call `finalize()` once after the last chunk to retrieve it.

:::{jupyter-execute}
import fastlowess as fl
import numpy as np

x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x) + 0.1

stream = fl.StreamingLowess(fraction=0.5, chunk_size=50, overlap=10)
stream.process_chunk(x[:50], y[:50])
stream.process_chunk(x[50:], y[50:])
final_result = stream.finalize()
print(final_result)
:::

## Options Structure

### `StreamingOptions` (inherits `LowessOptions`)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `fraction` | `float` | `0.67` | Smoothing fraction (bandwidth) |
| `iterations` | `int` | `3` | Number of robustifying iterations |
| `delta` | `float` | `None` | Interpolation distance (`None` auto-sets it to 0.0 in Streaming, i.e. interpolation disabled) |
| `weight_function` | `str` | `"tricube"` | Weight function name |
| `robustness_method` | `str` | `"bisquare"` | Robustness method name |
| `scaling_method` | `str` | `"mad"` | Residual scaling method |
| `boundary_policy` | `str` | `"extend"` | Boundary handling policy |
| `zero_weight_fallback` | `str` | `"use_local_mean"` | Zero-weight handling strategy |
| `auto_converge` | `float` | `None` | Auto-convergence tolerance |
| `return_diagnostics` | `bool` | `False` | Include diagnostics in result |
| `return_residuals` | `bool` | `False` | Include residuals in result |
| `return_robustness_weights` | `bool` | `False` | Include weights in result |
| `parallel` | `bool` | `True` | Enable parallel execution |
| `chunk_size` | `int` | `5000` | Data chunk size |
| `overlap` | `int` | `chunk_size / 10` | Overlap between chunks |
| `merge_strategy` | `str` | `"weighted_average"` | Strategy for blending overlap regions |

Confidence/prediction intervals, standard errors, cross-validation, GPU `backend`, `custom_weights`, and `return_sorted` are Batch-only and not available here; see [fastLowess](api.md) for those.

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

Points within `delta` of each other on the x-axis share the same local fit instead of each computing its own regression — an interpolation shortcut that trades a small amount of accuracy for a large speedup on dense, evenly-spaced data. `None` (default) auto-sets it to `0.0` in Streaming mode, i.e. interpolation is disabled and every point is fit exactly.

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

Convergence tolerance for early stopping of robustness iterations. `None` (default) disables early stopping.

### return_diagnostics

*See: [`Diagnostics`](#diagnostics)*

Include a `Diagnostics` object (RMSE, MAE, R², residual_sd) in the result. `effective_df`/`aic`/`aicc` require standard errors, which are Batch-only, so they're always `None` here.

- `False` (default) — leaves `result.diagnostics` as `None`
- `True` — populates `result.diagnostics`

### return_residuals

Include per-point residuals (`y - fitted`) in the result.

- `False` (default) — leaves `result.residuals` as `None`
- `True` — populates `result.residuals`

### return_robustness_weights

Include the final per-point robustness weights (from the last robustness iteration) in the result.

- `False` (default) — leaves `result.robustness_weights` as `None`
- `True` — populates `result.robustness_weights`

### parallel

Enable multi-threaded execution via Rayon.

- `True` (default) — parallelizes the local regression fits across CPU cores
- `False` — forces single-threaded execution (useful for benchmarking or deterministic profiling)

### chunk_size

Number of points processed per chunk. Larger chunks reduce per-chunk overhead and give each local fit more surrounding context, at the cost of higher peak memory; smaller chunks bound memory tightly but increase the fraction of points that fall in overlap regions. A good starting point is balancing available memory against how much processing overhead per chunk is acceptable — match it to your file-read buffer or message-batch size to avoid unnecessary copying.

### overlap

Number of points retained from the previous chunk as context, so the neighbourhood at chunk boundaries isn't artificially truncated. Points inside the overlap zone are fitted twice (once by each chunk) and reconciled via `merge_strategy`. A good starting point is 10–20% of `chunk_size`: too little overlap causes visible boundary artefacts, while too much wastes computation refitting the same points twice.

- `None` (default) — computes `chunk_size / 10`, clamped to at least 1 and less than `chunk_size`
- Any integer `>= 1` and `< chunk_size`

### merge_strategy

*See: [Merge Strategies](../advanced/merge.md)*

| Strategy | Alias | Behavior |
| --- | --- | --- |
| `"weighted_average"` (default) | `"weighted"` | Distance-weighted blend |
| `"average"` | `"mean"` | Average overlapping values |
| `"take_first"` | `"first"` | Keep left chunk values |
| `"take_last"` | `"last"` | Keep right chunk values |

## Result Structure

### `LowessResult`

Returned by `process_chunk()` and `finalize()`.

| Field | Type | Description |
| --- | --- | --- |
| `x` | `ndarray` | x values (same order as input) |
| `y` | `ndarray` | Smoothed y values |
| `fraction_used` | `float` | Fraction used |
| `iterations_used` | `int \| None` | Robustness iterations actually performed |
| `standard_errors` | `ndarray \| None` | Always `None` (Batch only) |
| `confidence_lower` | `ndarray \| None` | Always `None` (Batch only) |
| `confidence_upper` | `ndarray \| None` | Always `None` (Batch only) |
| `prediction_lower` | `ndarray \| None` | Always `None` (Batch only) |
| `prediction_upper` | `ndarray \| None` | Always `None` (Batch only) |
| `residuals` | `ndarray \| None` | Residuals (if `return_residuals`) |
| `robustness_weights` | `ndarray \| None` | Robustness weights (if `return_robustness_weights`) |
| `cv_scores` | `ndarray \| None` | Always `None` (Batch only) |
| `diagnostics` | `Diagnostics \| None` | Fit metrics (if `return_diagnostics`) |

### `Diagnostics`

| Field | Type | Description |
| --- | --- | --- |
| `rmse` | `float` | Root Mean Squared Error |
| `mae` | `float` | Mean Absolute Error |
| `r_squared` | `float` | R-squared |
| `residual_sd` | `float` | Residual standard deviation |
| `effective_df` | `float \| None` | Always `None` (requires standard errors, Batch only) |
| `aic` | `float \| None` | Always `None` (requires `effective_df`, Batch only) |
| `aicc` | `float \| None` | Always `None` (requires `effective_df`, Batch only) |
