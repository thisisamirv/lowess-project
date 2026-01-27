# fastLowess R API Reference

The R bindings provide a high-performance interface to the core Rust library, mirroring the Rust API structure.

## Classes

### `Lowess`

The `Lowess` class allows configuring the LOWESS parameters once and fitting multiple datasets using those parameters.

**Constructor:**

```r
model <- Lowess(...)
```

* `...`: Arguments corresponding to `LowessOptions` fields.

**Methods:**

```r
result <- model$fit(x, y)
```

* Fits the model to the provided `x` and `y` numeric vectors.
* Returns a `LowessResult` list containing the smoothed values and optional diagnostics.

### `StreamingLowess`

The `StreamingLowess` class processes data in chunks, suitable for very large datasets or streaming applications.

**Constructor:**

```r
stream <- StreamingLowess(...)
```

* `...`: Arguments corresponding to `LowessOptions` and `StreamingOptions` fields.

**Methods:**

```r
partial_result <- stream$process_chunk(x, y)
```

* Processes a chunk of data. Returns partial results.

```r
final_result <- stream$finalize()
```

* Finalizes the smoothing process and returns any remaining buffered results.

### `OnlineLowess`

The `OnlineLowess` class updates the model incrementally with new data points.

**Constructor:**

```r
online <- OnlineLowess(...)
```

* `...`: Arguments corresponding to `LowessOptions` and `OnlineOptions` fields.

**Methods:**

```r
result <- online$add_points(x, y)
```

* Adds new points to the model and returns the smoothed values (retrospective or prospective depending on mode).

## Options Structures

### `LowessOptions`

| Field                       | Type       | Default            | Description                           |
| --------------------------- | ---------- | ------------------ | ------------------------------------- |
| `fraction`                  | `numeric`  | `0.67`             | Smoothing fraction (bandwidth)        |
| `iterations`                | `integer`  | `3`                | Number of robustifying iterations     |
| `delta`                     | `numeric`  | `NULL`             | Interpolation distance (NULL for auto)|
| `weight_function`           | `character`| `"tricube"`        | Weight function name                  |
| `robustness_method`         | `character`| `"bisquare"`       | Robustness method name                |
| `scaling_method`            | `character`| `"mad"`            | Residual scaling method               |
| `boundary_policy`           | `character`| `"extend"`         | Boundary handling policy              |
| `zero_weight_fallback`      | `character`| `"use_local_mean"` | Zero-weight handling strategy         |
| `auto_converge`             | `numeric`  | `NULL`             | Auto-convergence tolerance            |
| `confidence_intervals`      | `numeric`  | `NULL`             | Confidence level (e.g., 0.95)         |
| `prediction_intervals`      | `numeric`  | `NULL`             | Prediction level (e.g., 0.95)         |
| `return_diagnostics`        | `logical`  | `FALSE`            | Include diagnostics in result         |
| `return_residuals`          | `logical`  | `FALSE`            | Include residuals in result           |
| `return_robustness_weights` | `logical`  | `FALSE`            | Include weights in result             |
| `parallel`                  | `logical`  | `TRUE`             | Enable parallel execution             |
| `cv_method`                 | `character`| `"kfold"`          | Cross-validation method ("kfold")     |
| `cv_k`                      | `integer`  | `5`                | Number of CV folds                    |
| `cv_fractions`              | `numeric`  | `NULL`             | Manual fractions for CV grid          |

### `StreamingOptions` (inherits `LowessOptions`)

| Field            | Type       | Default      | Description                |
| ---------------- | ---------- | ------------ | -------------------------- |
| `chunk_size`     | `integer`  | `5000`       | Data chunk size            |
| `overlap`        | `integer`  | `500`        | Overlap size (-1 for auto) |
| `merge_strategy` | `character`| `"weighted"` | Merge strategy for overlap |

### `OnlineOptions` (inherits `LowessOptions`)

| Field             | Type       | Default         | Description                           |
| ----------------- | ---------- | --------------- | ------------------------------------- |
| `window_capacity` | `integer`  | `1000`          | Max window size                       |
| `min_points`      | `integer`  | `2`             | Min points before smoothing           |
| `update_mode`     | `character`| `"incremental"` | Update mode ("full" or "incremental") |

## Result Structure

### `LowessResult`

| Field                | Type       | Description               |
| -------------------- | ---------- | ------------------------- |
| `x`                  | `numeric`  | Smoothed X coordinates    |
| `y`                  | `numeric`  | Smoothed Y coordinates    |
| `valid`              | `logical`  | True if result is valid   |
| `error`              | `character`| Error message if failed   |
| `diagnostics`        | `list`     | Diagnostic metrics list   |
| `residuals`          | `numeric`  | Residuals (if requested)  |
| `confidence_lower`   | `numeric`  | Lower CI bounds           |
| `confidence_upper`   | `numeric`  | Upper CI bounds           |
| `prediction_lower`   | `numeric`  | Lower PI bounds           |
| `prediction_upper`   | `numeric`  | Upper PI bounds           |
| `robustness_weights` | `numeric`  | Robustness weights        |

### `Diagnostics`

| Field          | Type      | Description                 |
| -------------- | --------- | --------------------------- |
| `rmse`         | `numeric` | Root Mean Squared Error     |
| `mae`          | `numeric` | Mean Absolute Error         |
| `r_squared`    | `numeric` | R-squared                   |
| `residual_sd`  | `numeric` | Residual standard deviation |
| `effective_df` | `numeric` | Effective degrees of freedom|
| `aic`          | `numeric` | AIC                         |
| `aicc`         | `numeric` | AICc                        |

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

```r
library(rfastlowess)

x <- seq(0, 10, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.2)

# Configure model
model <- Lowess(fraction = 0.5)

# Fit data
result <- model$fit(x, y)

print(result$y)
```
