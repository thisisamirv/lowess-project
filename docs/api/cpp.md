# fastLowess C++ API Reference

The C++ bindings provide a modern, object-oriented wrapper around the core Rust library, mirroring the Rust API structure.

> **StreamingLowess** and **OnlineLowess** are documented separately: [cpp-streaming.md](cpp-streaming.md), [cpp-online.md](cpp-online.md)

## Classes

### `fastlowess::Lowess`

The `Lowess` class allows configuring the LOWESS parameters once and fitting multiple datasets using those parameters.

**Constructor:**

```cpp
#include <fastlowess.hpp>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    fastlowess::LowessOptions opts;
    opts.fraction = 0.5;
    fastlowess::Lowess model(opts);

    return 0;
}
```

* `options`: A `LowessOptions` struct containing configuration parameters.

**Methods:**

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

    fastlowess::LowessOptions opts;
    opts.fraction = 0.5;
    fastlowess::Lowess model(opts);
    auto result = model.fit(x, y).value();
    std::cout << result.fraction_used() << std::endl;  // 0.5
    std::cout << result.iterations_used() << std::endl;  // -1
    // or with custom weights:
    std::vector<double> weights(x.size(), 1.0);
    auto resultW = model.fit(x, y, weights).value();

    return 0;
}
```

* Fits the model to the provided `x` and `y` data vectors.
* The second overload applies `custom_weights` — non-negative per-observation weights of length `n`. Batch only.
* Returns a `LowessResult` object containing the smoothed values and optional diagnostics.

See [cpp-streaming.md](cpp-streaming.md) for the `StreamingLowess` class.

See [cpp-online.md](cpp-online.md) for the `OnlineLowess` class.

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
| `return_se` | `bool` | false | Return standard errors |
| `parallel` | `bool` | true | Enable parallel execution |
| `cv_method` | `std::string` | "kfold" | CV method (`"kfold"` or `"loocv"`) (Batch only) |
| `cv_k` | `int` | 5 | Number of folds for k-fold CV (Batch only) |
| `cv_fractions` | `std::vector<double>` | `{}` | Fractions to test for cross-validation (Batch only) |
| `cv_seed` | `uint64_t` | `0` | Random seed for CV shuffling (Batch only; 0 = random) |
| `custom_weights` | `std::vector<double>` | `{}` | Per-observation case weights — passed to `fit()`, not the constructor (Batch only) |

See [cpp-streaming.md](cpp-streaming.md) for `StreamingOptions`.

See [cpp-online.md](cpp-online.md) for `OnlineOptions`.

## Result Structure

See [cpp-online.md](cpp-online.md) for `OnlineOutput`.

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

*See: [Weight Functions](../user-guide/kernels.md)*

* `"tricube"` (default)
* `"epanechnikov"`
* `"gaussian"`
* `"uniform"` (alias: `"boxcar"`)
* `"biweight"` (alias: `"bisquare"`)
* `"triangle"` (alias: `"triangular"`)
* `"cosine"`

### robustness_method

*See: [Robustness](../user-guide/robustness.md)*

* `"bisquare"` (default; alias: `"biweight"`)
* `"huber"`
* `"talwar"`

### boundary_policy

*See: [Boundary Handling](../user-guide/boundary.md)*

* `"extend"` (default; alias: `"pad"`)
* `"reflect"` (alias: `"mirror"`)
* `"zero"`
* `"noboundary"` (alias: `"none"`)

### scaling_method

*See: [Scaling Methods](../user-guide/scaling.md)*

* `"mad"` (default; alias: `"median_absolute_deviation"`)
* `"mar"` (alias: `"median_absolute_residual"`)
* `"mean"` (alias: `"mean_absolute_residual"`)

### zero_weight_fallback

*See: [Parameters](../user-guide/parameters.md)*

* `"use_local_mean"` (default; aliases: `"local_mean"`, `"mean"`)
* `"return_original"` (alias: `"original"`)
* `"return_none"` (alias: `"none"`)

### merge_strategy

See [cpp-streaming.md](cpp-streaming.md).

### update_mode

See [cpp-online.md](cpp-online.md).

## Example

```cpp
#include <fastlowess.hpp>
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
