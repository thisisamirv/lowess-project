# fastLowess R API Reference

The R bindings provide a high-performance interface to the core Rust library, mirroring the Rust API structure.

> **StreamingLowess** and **OnlineLowess** are documented separately: [r-streaming.md](r-streaming.md), [r-online.md](r-online.md)

## Classes

### `Lowess`

The `Lowess` class allows configuring the LOWESS parameters once and fitting multiple datasets using those parameters.

**Constructor:**

```r
library(rfastlowess)
set.seed(42)
x <- seq(0, 2 * pi, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)

library(rfastlowess)
model <- Lowess(fraction = 0.5, iterations = 3)
```

**Methods:**

```r
library(rfastlowess)
set.seed(42)
x <- seq(0, 2 * pi, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)

model <- Lowess(fraction = 0.5)
result <- fit(model, x, y, custom_weights = NULL)
```

* Fits the model to the provided `x` and `y` numeric vectors.
* Returns a `LowessResult` S3 object containing the smoothed values and optional diagnostics.
* `custom_weights`: Optional numeric vector of per-observation weights. All values must be ≥ 0 and length must match `x`. Batch only.
* `print(model)`: Displays the model configuration.

See [r-streaming.md](r-streaming.md) for the `StreamingLowess` class.

See [r-online.md](r-online.md) for the `OnlineLowess` class.

## Options Structures

### `LowessOptions`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `fraction` | `numeric` | `0.67` | Smoothing fraction (bandwidth) |
| `iterations` | `integer` | `3` | Number of robustifying iterations |
| `delta` | `numeric` | `NULL` | Interpolation distance (NULL for auto) |
| `weight_function` | `character` | `"tricube"` | Weight function name |
| `robustness_method` | `character` | `"bisquare"` | Robustness method name |
| `scaling_method` | `character` | `"mad"` | Residual scaling method |
| `boundary_policy` | `character` | `"extend"` | Boundary handling policy |
| `zero_weight_fallback` | `character` | `"use_local_mean"` | Zero-weight handling strategy |
| `auto_converge` | `numeric` | `NULL` | Auto-convergence tolerance |
| `confidence_intervals` | `numeric` | `NULL` | Confidence level (e.g., 0.95) |
| `prediction_intervals` | `numeric` | `NULL` | Prediction level (e.g., 0.95) |
| `return_diagnostics` | `logical` | `FALSE` | Include diagnostics in result |
| `return_residuals` | `logical` | `FALSE` | Include residuals in result |
| `return_robustness_weights` | `logical` | `FALSE` | Include weights in result |
| `return_se` | `logical` | `FALSE` | Return standard errors |
| `parallel` | `logical` | `TRUE` | Enable parallel execution |
| `cv_method` | `character` | `"kfold"` | CV method (`"kfold"` or `"loocv"`) (Batch only) |
| `cv_k` | `integer` | `5L` | Number of folds for k-fold CV (Batch only) |
| `cv_fractions` | `numeric` | `NULL` | Fractions to test for cross-validation (Batch only) |
| `cv_seed` | `integer` | `NULL` | Random seed for cross-validation shuffling (Batch only) |
| `custom_weights` | `numeric` | `NULL` | Per-observation case weights — passed to `fit()`, not the constructor (Batch only) |

See [r-streaming.md](r-streaming.md) for `StreamingOptions`.

See [r-online.md](r-online.md) for `OnlineOptions`.

## Result Structure

See [r-online.md](r-online.md) for `OnlineOutput`.

### `LowessResult`

An S3 list with class `"LowessResult"` containing:

**Supported S3 Methods:** `print(result)`, `plot(result)`

| Field | Type | Description |
| --- | --- | --- |
| `x` | `numeric` | Sorted x values |
| `y` | `numeric` | Smoothed y values |
| `fraction_used` | `numeric` | Fraction used (set or selected by CV) |
| `iterations_used` | `integer \| NULL` | Robustness iterations actually performed |
| `standard_errors` | `numeric \| NULL` | Per-point standard errors |
| `confidence_lower` | `numeric \| NULL` | Lower confidence bounds |
| `confidence_upper` | `numeric \| NULL` | Upper confidence bounds |
| `prediction_lower` | `numeric \| NULL` | Lower prediction bounds |
| `prediction_upper` | `numeric \| NULL` | Upper prediction bounds |
| `residuals` | `numeric \| NULL` | Residuals (if `return_residuals`) |
| `robustness_weights` | `numeric \| NULL` | Robustness weights (if `return_robustness_weights`) |
| `cv_scores` | `numeric \| NULL` | CV score per tested fraction |
| `diagnostics` | `list \| NULL` | Fit metrics (if `return_diagnostics`) |

### `Diagnostics`

| Field | Type | Description |
| --- | --- | --- |
| `rmse` | `numeric` | Root Mean Squared Error |
| `mae` | `numeric` | Mean Absolute Error |
| `r_squared` | `numeric` | R-squared |
| `residual_sd` | `numeric` | Residual standard deviation |
| `effective_df` | `numeric` | Effective degrees of freedom (NaN if not computed) |
| `aic` | `numeric` | AIC (NaN if not computed) |
| `aicc` | `numeric` | AICc (NaN if not computed) |

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

See [r-streaming.md](r-streaming.md).

### update_mode

See [r-online.md](r-online.md).

## Example

```r
library(rfastlowess)

x <- seq(0, 10, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.2)

# Configure model
model <- Lowess(fraction = 0.5)

# Fit data
result <- fit(model, x, y)

# Print summary
print(result)

# Plot result
plot(result)
```
