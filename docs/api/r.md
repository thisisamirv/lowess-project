# fastLowess R API Reference

The R bindings provide a high-performance interface to the core Rust library, mirroring the Rust API structure.

## Classes

### `Lowess`

The `Lowess` class allows configuring the LOWESS parameters once and fitting multiple datasets using those parameters.

**Constructor:**

```r
library(rfastlowess)
model <- Lowess(fraction = 0.5, iterations = 3)
```

* `...`: Arguments corresponding to `LowessOptions` fields.

**Methods:**

```r
library(rfastlowess)
set.seed(42)
x <- seq(0, 2 * pi, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)

model <- Lowess(fraction = 0.5)
result <- model$fit(x, y, custom_weights = NULL)
```

* Fits the model to the provided `x` and `y` numeric vectors.
* Returns a `LowessResult` S3 object containing the smoothed values and optional diagnostics.
* `custom_weights`: Optional numeric vector of per-observation weights. All values must be ≥ 0 and length must match `x`. Batch only.
* `print(model)`: Displays the model configuration.

### `StreamingLowess`

The `StreamingLowess` class processes data in chunks, suitable for very large datasets or streaming applications.

**Constructor:**

```r
library(rfastlowess)
stream <- StreamingLowess(fraction = 0.3, chunk_size = 50, overlap = 10)
```

* `...`: Arguments corresponding to `LowessOptions` and `StreamingOptions` fields.

**Methods:**

```r
library(rfastlowess)
set.seed(42)
x <- seq(0, 2 * pi, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)

stream <- StreamingLowess(fraction = 0.3, chunk_size = 50, overlap = 10)
partial_result <- stream$process_chunk(x[seq_len(50)], y[seq_len(50)])
```

* Processes a chunk of data. Returns partial results.

```r
library(rfastlowess)
set.seed(42)
x <- seq(0, 2 * pi, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)

stream <- StreamingLowess(fraction = 0.3, chunk_size = 50, overlap = 10)
stream$process_chunk(x, y)
final_result <- stream$finalize()
```

* Finalizes the smoothing process and returns any remaining buffered results.

### `OnlineLowess`

The `OnlineLowess` class updates the model incrementally with new data points.

**Constructor:**

```r
library(rfastlowess)
online <- OnlineLowess(fraction = 0.3, window_capacity = 50)
```

* `...`: Arguments corresponding to `LowessOptions` and `OnlineOptions` fields.

**Methods:**

```r
library(rfastlowess)
set.seed(42)
x <- seq(0, 2 * pi, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)

online <- OnlineLowess(fraction = 0.3, window_capacity = 50)
result <- online$add_point(x[[1L]], y[[1L]])  # returns list or NULL
```

* Adds a single point to the sliding window. Returns a named list (`$smoothed`, `$residual`, …) once the window has enough points, or `NULL` while still filling.

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
| `custom_weights` | `numeric` | `NULL` | Per-observation case weights — passed to `$fit()`, not the constructor (Batch only) |

### `StreamingOptions` (inherits `LowessOptions`)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `chunk_size` | `integer` | `5000L` | Data chunk size |
| `overlap` | `integer` | `500L` | Overlap between chunks |
| `merge_strategy` | `character` | `"weighted_average"` | Strategy for blending overlap regions |

### `OnlineOptions` (inherits `LowessOptions`)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `window_capacity` | `integer` | `1000L` | Max points in sliding window |
| `min_points` | `integer` | `3L` | Min points before smoothing starts |
| `update_mode` | `character` | `"full"` | Update mode (`"full"` or `"incremental"`) |
| `parallel` | `logical` | `FALSE` | Enable parallel execution (off by default; online LOWESS fits one point at a time) |

## Result Structure

### `OnlineOutput` (named list)

Returned by `add_point()` once the window has enough points (`NULL` until then).

| Field | Type | Description |
| --- | --- | --- |
| `smoothed` | `numeric` | Smoothed value for the latest point |
| `std_error` | `numeric` (optional) | Standard error (if requested) |
| `residual` | `numeric` (optional) | Residual y − smoothed (if requested) |
| `robustness_weight` | `numeric` (optional) | Robustness weight (if requested) |
| `iterations_used` | `integer` (optional) | Robustness iterations performed |

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

```r
library(rfastlowess)

x <- seq(0, 10, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.2)

# Configure model
model <- Lowess(fraction = 0.5)

# Fit data
result <- model$fit(x, y)

# Print summary
print(result)

# Plot result
plot(result)
```
