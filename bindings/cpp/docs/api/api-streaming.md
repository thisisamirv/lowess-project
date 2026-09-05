\page api_streaming StreamingLowess API

# StreamingLowess API

See also: [fastLowess](api.md)

## When to Use

- Dataset >100,000 points
- Memory-constrained environments
- Batch processing pipelines

## Class

### fastlowess::StreamingLowess

The `StreamingLowess` class processes data in chunks, suitable for very large datasets or streaming applications.

**Constructor:**

```cpp
#include <fastlowess.hpp>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    const int n = 20;
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) { x[i] = i; y[i] = i + 0.1; }

    fastlowess::StreamingOptions opts;
    opts.chunk_size = 10;
    opts.overlap = 2;
    fastlowess::StreamingLowess model(opts);
    std::vector<double> x1(x.begin(), x.begin() + 10), y1(y.begin(), y.begin() + 10);
    std::vector<double> x2(x.begin() + 10, x.end()), y2(y.begin() + 10, y.end());
    model.process_chunk(x1, y1);
    model.process_chunk(x2, y2);
    auto result = model.finalize().value();

    std::cout << "y[0]: " << result.y_vector()[0] << "\n";
    return 0;
}
```

```output
y[0]: 17.6391
```

- `options`: A `StreamingOptions` struct (inherits from `LowessOptions`) with additional `chunk_size`, `overlap`, and `merge_strategy` parameters.

#### `process_chunk(x, y)`

Feeds one chunk of data into the model. Each chunk is fit together with the trailing `overlap` points buffered from the previous call, then only the points that are fully resolved are returned — the tail of the chunk (the next `overlap` points) is held back internally, since it will be refit once the following chunk arrives and its estimate reconciled via `merge_strategy`. This is what lets the adapter process a dataset far larger than memory allows, one bounded-size chunk at a time, without ever materializing the whole dataset at once.

```cpp
#include <fastlowess.hpp>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    const int n = 100;
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = i * 2 * M_PI / (n - 1);
        y[i] = std::sin(x[i]) + 0.1;
    }

    fastlowess::StreamingOptions opts;
    opts.fraction = 0.5;
    opts.chunk_size = 50;
    opts.overlap = 10;
    fastlowess::StreamingLowess model(opts);
    std::vector<double> x1(x.begin(), x.begin() + 50), y1(y.begin(), y.begin() + 50);
    auto partial = model.process_chunk(x1, y1).value();
    std::cout << partial.fraction_used() << std::endl;  // 0.5

    return 0;
}
```

```output
0.5
```

#### `finalize()`

Flushes the overlap points still buffered from the last `process_chunk()` call. Because each call withholds its tail until the next chunk arrives to resolve it, the final chunk's tail would never be emitted otherwise — always call `finalize()` once after the last chunk to retrieve it.

```cpp
#include <fastlowess.hpp>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    const int n = 100;
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = i * 2 * M_PI / (n - 1);
        y[i] = std::sin(x[i]) + 0.1;
    }

    fastlowess::StreamingOptions opts;
    opts.fraction = 0.5;
    opts.chunk_size = 50;
    opts.overlap = 10;
    fastlowess::StreamingLowess model(opts);
    std::vector<double> x1(x.begin(), x.begin() + 50), y1(y.begin(), y.begin() + 50);
    std::vector<double> x2(x.begin() + 50, x.end()), y2(y.begin() + 50, y.end());
    model.process_chunk(x1, y1);
    model.process_chunk(x2, y2);
    auto result = model.finalize().value();
    std::cout << result.fraction_used() << std::endl;  // 0.5

    return 0;
}
```

```output
0.5
```

## Options Structure

### StreamingOptions (inherits LowessOptions)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `fraction` | `double` | 0.67 | Smoothing fraction (bandwidth) |
| `iterations` | `int` | 3 | Number of robustifying iterations |
| `delta` | `double` | NaN | Interpolation distance (`NaN` auto-sets it to 0.0 in Streaming, i.e. interpolation disabled) |
| `weight_function` | `std::string` | "tricube" | Weight function name |
| `robustness_method` | `std::string` | "bisquare" | Robustness method name |
| `scaling_method` | `std::string` | "mad" | Residual scaling method |
| `boundary_policy` | `std::string` | "extend" | Boundary handling policy |
| `zero_weight_fallback` | `std::string` | "use_local_mean" | Zero-weight handling |
| `missing` | `std::string` | "error" | Policy for non-finite (NaN/Inf) values in each chunk |
| `auto_converge` | `double` | NaN | Auto-convergence tolerance |
| `return_diagnostics` | `bool` | false | Include diagnostics in result |
| `return_residuals` | `bool` | false | Include residuals in result |
| `return_robustness_weights` | `bool` | false | Include weights in result |
| `parallel` | `bool` | true | Enable parallel execution |
| `chunk_size` | `int` | 5000 | Data chunk size |
| `overlap` | `int` | chunk_size / 10 | Overlap between chunks |
| `merge_strategy` | `std::string` | "weighted_average" | Strategy for blending overlap regions |

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

Points within `delta` of each other on the x-axis share the same local fit instead of each computing its own regression — an interpolation shortcut that trades a small amount of accuracy for a large speedup on dense, evenly-spaced data. `NaN` (default) auto-sets it to `0` in Streaming mode, i.e. interpolation is disabled and every point is fit exactly.

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

### missing

Policy for handling non-finite (NaN/Inf) values within each chunk:

| Option | Behavior |
| --- | --- |
| `"error"` (default) | Return an error result if any value in the chunk is non-finite |
| `"drop"` | Silently remove rows where `x` or `y` is non-finite before merging the chunk with the overlap buffer |

**Note:** A length mismatch between `x` and `y` always errors, even under `"drop"`.

### auto_converge

*See: [Robustness](../weighting/robustness.md#auto-convergence)*

Convergence tolerance for early stopping of robustness iterations. `NaN` (default) disables early stopping.

### return_diagnostics

*See: [`Diagnostics`](#fastlowessdiagnostics)*

Include a `Diagnostics` object (RMSE, MAE, R², residual_sd) in the result. `effective_df`/`aic`/`aicc` require standard errors, which are Batch-only, so they're always empty/NaN here.

- `false` (default) — leaves `diagnostics()` empty
- `true` — populates `diagnostics()`

### return_residuals

Include per-point residuals (`y - fitted`) in the result.

- `false` (default) — leaves `residuals()` empty
- `true` — populates `residuals()`

### return_robustness_weights

Include the final per-point robustness weights (from the last robustness iteration) in the result.

- `false` (default) — leaves `robustness_weights()` empty
- `true` — populates `robustness_weights()`

### parallel

Enable multi-threaded execution via Rayon.

- `true` (default) — parallelizes the local regression fits across CPU cores
- `false` — forces single-threaded execution

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

## Result Structure

### fastlowess::LowessResult

Returned (inside `Expected`) by `process_chunk()` and `finalize()`.

| Method | Return Type | Description |
| --- | --- | --- |
| `x_vector()` | `std::vector<double>` | x values (same order as input) |
| `y_vector()` | `std::vector<double>` | Smoothed y values |
| `fraction_used()` | `double` | Fraction used |
| `iterations_used()` | `int` | Robustness iterations (-1 = N/A) |
| `standard_errors()` | `std::vector<double>` | Always empty (Batch only) |
| `confidence_lower()` | `std::vector<double>` | Always empty (Batch only) |
| `confidence_upper()` | `std::vector<double>` | Always empty (Batch only) |
| `prediction_lower()` | `std::vector<double>` | Always empty (Batch only) |
| `prediction_upper()` | `std::vector<double>` | Always empty (Batch only) |
| `residuals()` | `std::vector<double>` | Residuals (if `return_residuals`; empty if not) |
| `robustness_weights()` | `std::vector<double>` | Robustness weights (if `return_robustness_weights`; empty if not) |
| `cv_scores()` | `std::vector<double>` | Always empty (Batch only) |
| `diagnostics()` | `Diagnostics` | Fit metrics — check `has_value()` (if `return_diagnostics`) |

### fastlowess::Diagnostics

| Method | Return Type | Description |
| --- | --- | --- |
| `rmse()` | `double` | Root Mean Squared Error |
| `mae()` | `double` | Mean Absolute Error |
| `r_squared()` | `double` | R-squared |
| `residual_sd()` | `double` | Residual standard deviation |
| `effective_df()` | `double` | Always NaN (requires standard errors, Batch only) |
| `aic()` | `double` | Always NaN (requires `effective_df`, Batch only) |
| `aicc()` | `double` | Always NaN (requires `effective_df`, Batch only) |
