# fastLowess C++ API Reference

The C++ bindings provide a modern, object-oriented wrapper around the core Rust library, mirroring the Rust API structure.

## Classes

### `fastlowess::Lowess`

The `Lowess` class allows configuring the LOWESS parameters once and fitting multiple datasets using those parameters.

**Constructor:**

```cpp
explicit Lowess(const LowessOptions &options = {})
```

* `options`: A `LowessOptions` struct containing configuration parameters.

**Methods:**

```cpp
LowessResult fit(const std::vector<double> &x, const std::vector<double> &y)
LowessResult fit(const std::vector<double> &x, const std::vector<double> &y,
                 const std::vector<double> &custom_weights)
```

* Fits the model to the provided `x` and `y` data vectors.
* The second overload applies `custom_weights` — non-negative per-observation weights of length `n`. Batch only.
* Returns a `LowessResult` object containing the smoothed values and optional diagnostics.

### `fastlowess::StreamingLowess`

The `StreamingLowess` class processes data in chunks, suitable for very large datasets or streaming applications.

**Constructor:**

```cpp
explicit StreamingLowess(const StreamingOptions &options = {})
```

* `options`: A `StreamingOptions` struct (inherits from `LowessOptions`) with additional `chunk_size` and `overlap` parameters.

**Methods:**

```cpp
LowessResult process_chunk(const std::vector<double> &x, const std::vector<double> &y)
```

* Processes a chunk of data. Returns partial results.

```cpp
LowessResult finalize()
```

* Finalizes the smoothing process and returns any remaining buffered results.

### `fastlowess::OnlineLowess`

The `OnlineLowess` class updates the model incrementally with new data points.

**Constructor:**

```cpp
explicit OnlineLowess(const OnlineOptions &options = {})
```

* `options`: An `OnlineOptions` struct (inherits from `LowessOptions`) with `window_capacity`, `min_points`, and `update_mode`.

**Methods:**

```cpp
Expected<OnlineOutput> add_point(double x, double y)
```

* Adds a single point to the sliding window. Returns `Expected<OnlineOutput>` — check `has_value()` to see whether the window is ready.

## Options Structures

### `LowessOptions`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `fraction` | `double` | 0.67 | Smoothing fraction (bandwidth) |
| `iterations` | `int` | 3 | Number of robustifying iterations |
| `delta` | `double` | NaN | Interpolation distance (NaN for auto) |
| `weight_function` | `std::string` | "tricube" | Weight function name |
| `robustness_method` | `std::string` | "bisquare" | Robustness method name |
| `scaling_method` | `std::string` | "mad" | Residual scaling method |
| `boundary_policy` | `std::string` | "extend" | Boundary handling policy |
| `zero_weight_fallback` | `std::string` | "use_local_mean" | Zero-weight handling strategy |
| `auto_converge` | `double` | NaN | Auto-convergence tolerance |
| `confidence_intervals` | `double` | NaN | Confidence level (e.g., 0.95) |
| `prediction_intervals` | `double` | NaN | Prediction level (e.g., 0.95) |
| `return_diagnostics` | `bool` | false | Compute RMSE, MAE, R², AIC |
| `return_residuals` | `bool` | false | Include residuals in result |
| `return_robustness_weights` | `bool` | false | Include robustness weights in result |
| `parallel` | `bool` | true | Enable parallel execution |
| `cv_method` | `std::string` | "kfold" | CV method (`"kfold"` or `"loocv"`) (Batch only) |
| `cv_k` | `int` | 5 | Number of folds for k-fold CV (Batch only) |
| `cv_fractions` | `std::vector<double>` | `{}` | Fractions to test for cross-validation (Batch only) |
| `cv_seed` | `uint64_t` | `0` | Random seed for CV shuffling (Batch only; 0 = random) |
| `custom_weights` | `std::vector<double>` | `{}` | Per-observation case weights — passed to `fit()`, not the constructor (Batch only) |

### `StreamingOptions` (inherits `LowessOptions`)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `chunk_size` | `int` | 5000 | Data chunk size |
| `overlap` | `int` | 500 | Overlap between chunks |
| `merge_strategy` | `std::string` | "weighted_average" | Strategy for blending overlap regions |

### `OnlineOptions` (inherits `LowessOptions`)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `window_capacity` | `int` | 1000 | Max points in sliding window |
| `min_points` | `int` | 3 | Min points before smoothing starts |
| `update_mode` | `std::string` | "full" | Update mode (`"full"` or `"incremental"`) |
| `parallel` | `bool` | `false` | Enable parallel execution (off by default; online LOWESS fits one point at a time) |

## Result Structure

### `fastlowess::OnlineOutput`

Returned (inside `Expected`) by `add_point()`. Check `has_value()` before reading fields.

| Method | Return Type | Description |
| --- | --- | --- |
| `has_value()` | `bool` | `false` while window fills; `true` when output is ready |
| `smoothed()` | `double` | Smoothed value for the latest point |
| `std_error()` | `double` | Standard error (NaN if not computed) |
| `residual()` | `double` | Residual y − smoothed (NaN if not computed) |
| `robustness_weight()` | `double` | Robustness weight (NaN if not computed) |
| `iterations_used()` | `int` | Robustness iterations performed (−1 if N/A) |

### `fastlowess::LowessResult`

A RAII wrapper around the C result struct `fastlowess_CppLowessResult`.

| Method | Return Type | Description |
| --- | --- | --- |
| `x_vector()` | `std::vector<double>` | Sorted x values |
| `y_vector()` | `std::vector<double>` | Smoothed y values |
| `fraction_used()` | `double` | Fraction used (set or selected by CV) |
| `iterations_used()` | `int` | Robustness iterations actually performed (-1 = N/A) |
| `standard_errors()` | `std::vector<double>` | Per-point standard errors (empty if not computed) |
| `confidence_lower()` | `std::vector<double>` | Lower confidence bounds (empty if not computed) |
| `confidence_upper()` | `std::vector<double>` | Upper confidence bounds (empty if not computed) |
| `prediction_lower()` | `std::vector<double>` | Lower prediction bounds (empty if not computed) |
| `prediction_upper()` | `std::vector<double>` | Upper prediction bounds (empty if not computed) |
| `residuals()` | `std::vector<double>` | Residuals (if `return_residuals`; empty if not computed) |
| `robustness_weights()` | `std::vector<double>` | Robustness weights (if `return_robustness_weights`; empty if not computed) |
| `cv_scores()` | `std::vector<double>` | CV score per tested fraction (empty if CV not run) |
| `diagnostics()` | `Diagnostics` | Fit metrics — check `diagnostics().has_value()` before use (if `return_diagnostics`) |

### `fastlowess::Diagnostics`

All accessors are const methods (not public fields):

| Method | Return Type | Description |
| --- | --- | --- |
| `rmse()` | `double` | Root Mean Squared Error |
| `mae()` | `double` | Mean Absolute Error |
| `r_squared()` | `double` | R-squared |
| `residual_sd()` | `double` | Residual standard deviation |
| `effective_df()` | `double` | Effective degrees of freedom (NaN if not computed) |
| `aic()` | `double` | AIC (NaN if not computed) |
| `aicc()` | `double` | AICc (NaN if not computed) |

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

```cpp
#include "fastlowess.hpp"
#include <iostream>

int main() {
    std::vector<double> x = {1, 2, 3, 4, 5};
    std::vector<double> y = {2.1, 4.0, 6.2, 8.0, 10.1};

    fastlowess::LowessOptions opts;
    opts.fraction = 0.5;
    
    fastlowess::Lowess model(opts);
    auto expected = model.fit(x, y);

    if (expected.has_value()) {
        auto y_hat = expected.value().y_vector();
        for (double val : y_hat) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}
```
