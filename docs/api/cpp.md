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
```

* Fits the model to the provided `x` and `y` data vectors.
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
LowessResult add_points(const std::vector<double> &x, const std::vector<double> &y)
```

* Adds new points to the model and returns the smoothed values (retrospective or prospective depending on mode).

## Options Structures

### `LowessOptions`

| Field                       | Type                  | Default            | Description                           |
| --------------------------- | --------------------- | ------------------ | ------------------------------------- |
| `fraction`                  | `double`              | 0.67               | Smoothing fraction (bandwidth)        |
| `iterations`                | `int`                 | 3                  | Number of robustifying iterations     |
| `delta`                     | `double`              | NaN                | Interpolation distance (NaN for auto) |
| `weight_function`           | `std::string`         | "tricube"          | Weight function name                  |
| `robustness_method`         | `std::string`         | "bisquare"         | Robustness method name                |
| `scaling_method`            | `std::string`         | "mad"              | Residual scaling method               |
| `boundary_policy`           | `std::string`         | "extend"           | Boundary handling policy              |
| `zero_weight_fallback`      | `std::string`         | "use_local_mean"   | Zero-weight handling strategy         |
| `auto_converge`             | `double`              | NaN                | Auto-convergence tolerance            |
| `confidence_intervals`      | `double`              | NaN                | Confidence level (e.g., 0.95)         |
| `prediction_intervals`      | `double`              | NaN                | Prediction level (e.g., 0.95)         |
| `return_diagnostics`        | `bool`                | false              | Include diagnostics in result         |
| `return_residuals`          | `bool`                | false              | Include residuals in result           |
| `return_robustness_weights` | `bool`                | false              | Include weights in result             |
| `parallel`                  | `bool`                | false              | Enable parallel execution             |
| `cv_method`                 | `std::string`         | "kfold"            | Cross-validation method               |
| `cv_k`                      | `int`                 | 5                  | Number of CV folds                    |
| `cv_fractions`              | `std::vector<double>` | `{}`               | Manual fractions for CV grid          |

### `StreamingOptions` (inherits `LowessOptions`)

| Field            | Type          | Default    | Description                |
| ---------------- | ------------- | ---------- | -------------------------- |
| `chunk_size`     | `int`         | 5000       | Data chunk size            |
| `overlap`        | `int`         | -1         | Overlap size (-1 for auto) |
| `merge_strategy` | `std::string` | "weighted" | Merge strategy for overlap |

### `OnlineOptions` (inherits `LowessOptions`)

| Field             | Type          | Default | Description                           |
| ----------------- | ------------- | ------- | ------------------------------------- |
| `window_capacity` | `int`         | 1000    | Max window size                       |
| `min_points`      | `int`         | 2       | Min points before smoothing           |
| `update_mode`     | `std::string` | "full"  | Update mode ("full" or "incremental") |

## Result Structure

### `fastlowess::LowessResult`

A RAII wrapper around the C result struct `fastlowess_CppLowessResult`.

| Method                 | Return Type           | Description               |
| ---------------------- | --------------------- | ------------------------- |
| `x_vector()`           | `std::vector<double>` | Smoothed X coordinates    |
| `y_vector()`           | `std::vector<double>` | Smoothed Y coordinates    |
| `valid()`              | `bool`                | True if result is valid   |
| `error()`              | `std::string`         | Error message if failed   |
| `diagnostics()`        | `Diagnostics`         | Diagnostic metrics struct |
| `residuals()`          | `std::vector<double>` | Residuals (if requested)  |
| `confidence_lower()`   | `std::vector<double>` | Lower CI bounds           |
| `confidence_upper()`   | `std::vector<double>` | Upper CI bounds           |
| `prediction_lower()`   | `std::vector<double>` | Lower PI bounds           |
| `prediction_upper()`   | `std::vector<double>` | Upper PI bounds           |
| `robustness_weights()` | `std::vector<double>` | Robustness weights        |

### `fastlowess::Diagnostics`

| Field          | Type     | Description                 |
| -------------- | -------- | --------------------------- |
| `rmse`         | `double` | Root Mean Squared Error     |
| `mae`          | `double` | Mean Absolute Error         |
| `r_squared`    | `double` | R-squared                   |
| `residual_sd`  | `double` | Residual standard deviation |
| `effective_df` | `double` | Effective degrees of freedom|
| `aic`          | `double` | AIC                         |
| `aicc`         | `double` | AICc                        |

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

```cpp
#include "fastlowess.hpp"
#include <iostream>

int main() {
    std::vector<double> x = {1, 2, 3, 4, 5};
    std::vector<double> y = {2.1, 4.0, 6.2, 8.0, 10.1};

    fastlowess::LowessOptions opts;
    opts.fraction = 0.5;
    
    fastlowess::Lowess model(opts);
    auto result = model.fit(x, y);

    if (result.valid()) {
        auto y_hat = result.y_vector();
        for (double val : y_hat) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}
```
