# fastLowess Python API Reference

The Python bindings provide a high-performance interface to the core Rust library, mirroring the Rust API structure.

## Classes

### `Lowess`

The `Lowess` class allows configuring the LOWESS parameters once and fitting multiple datasets using those parameters.

**Constructor:**

```python
model = fastlowess.Lowess(**kwargs)
```

* `kwargs`: Keyword arguments corresponding to `LowessOptions` fields.

**Methods:**

```python
result = model.fit(x, y)
```

* Fits the model to the provided `x` and `y` array-like objects.
* Returns a `LowessResult` object containing the smoothed values and optional diagnostics.

### `StreamingLowess`

The `StreamingLowess` class processes data in chunks, suitable for very large datasets or streaming applications.

**Constructor:**

```python
stream = fastlowess.StreamingLowess(**kwargs)
```

* `kwargs`: Keyword arguments corresponding to `LowessOptions` and `StreamingOptions` fields.

**Methods:**

```python
partial_result = stream.process_chunk(x, y)
```

* Processes a chunk of data. Returns partial results.

```python
final_result = stream.finalize()
```

* Finalizes the smoothing process and returns any remaining buffered results.

### `OnlineLowess`

The `OnlineLowess` class updates the model incrementally with new data points.

**Constructor:**

```python
online = fastlowess.OnlineLowess(**kwargs)
```

* `kwargs`: Keyword arguments corresponding to `LowessOptions` and `OnlineOptions` fields.

**Methods:**

```python
result = online.add_points(x, y)
```

* Adds new points to the model and returns the smoothed values (retrospective or prospective depending on mode).

## Options Structures

### `LowessOptions`

| Field                       | Type          | Default            | Description                           |
| --------------------------- | ------------- | ------------------ | ------------------------------------- |
| `fraction`                  | `float`       | `0.67`             | Smoothing fraction (bandwidth)        |
| `iterations`                | `int`         | `3`                | Number of robustifying iterations     |
| `delta`                     | `float`       | `None`             | Interpolation distance (None for auto)|
| `weight_function`           | `str`         | `"tricube"`        | Weight function name                  |
| `robustness_method`         | `str`         | `"bisquare"`       | Robustness method name                |
| `scaling_method`            | `str`         | `"mad"`            | Residual scaling method               |
| `boundary_policy`           | `str`         | `"extend"`         | Boundary handling policy              |
| `zero_weight_fallback`      | `str`         | `"use_local_mean"` | Zero-weight handling strategy         |
| `auto_converge`             | `float`       | `None`             | Auto-convergence tolerance            |
| `confidence_intervals`      | `float`       | `None`             | Confidence level (e.g., 0.95)         |
| `prediction_intervals`      | `float`       | `None`             | Prediction level (e.g., 0.95)         |
| `return_diagnostics`        | `bool`        | `False`            | Include diagnostics in result         |
| `return_residuals`          | `bool`        | `False`            | Include residuals in result           |
| `return_robustness_weights` | `bool`        | `False`            | Include weights in result             |
| `parallel`                  | `bool`        | `True`             | Enable parallel execution             |
| `cv_method`                 | `str`         | `"kfold"`          | Cross-validation method ("kfold")     |
| `cv_k`                      | `int`         | `5`                | Number of CV folds                    |
| `cv_fractions`              | `list[float]` | `None`             | Manual fractions for CV grid          |

### `StreamingOptions` (inherits `LowessOptions`)

| Field            | Type   | Default      | Description                |
| ---------------- | ------ | ------------ | -------------------------- |
| `chunk_size`     | `int`  | `5000`       | Data chunk size            |
| `overlap`        | `int`  | `500`        | Overlap size (-1 for auto) |
| `merge_strategy` | `str`  | `"weighted"` | Merge strategy for overlap |

### `OnlineOptions` (inherits `LowessOptions`)

| Field             | Type   | Default         | Description                           |
| ----------------- | ------ | --------------- | ------------------------------------- |
| `window_capacity` | `int`  | `1000`          | Max window size                       |
| `min_points`      | `int`  | `2`             | Min points before smoothing           |
| `update_mode`     | `str`  | `"incremental"` | Update mode ("full" or "incremental") |

## Result Structure

### `LowessResult`

| Field                | Type      | Description               |
| -------------------- | --------- | ------------------------- |
| `x`                  | `ndarray` | Smoothed X coordinates    |
| `y`                  | `ndarray` | Smoothed Y coordinates    |
| `valid`              | `bool`    | True if result is valid   |
| `error`              | `str`     | Error message if failed   |
| `diagnostics`        | `object`  | Diagnostic metrics object |
| `residuals`          | `ndarray` | Residuals (if requested)  |
| `confidence_lower`   | `ndarray` | Lower CI bounds           |
| `confidence_upper`   | `ndarray` | Upper CI bounds           |
| `prediction_lower`   | `ndarray` | Lower PI bounds           |
| `prediction_upper`   | `ndarray` | Upper PI bounds           |
| `robustness_weights` | `ndarray` | Robustness weights        |

### `Diagnostics`

| Field          | Type    | Description                 |
| -------------- | ------- | --------------------------- |
| `rmse`         | `float` | Root Mean Squared Error     |
| `mae`          | `float` | Mean Absolute Error         |
| `r_squared`    | `float` | R-squared                   |
| `residual_sd`  | `float` | Residual standard deviation |
| `effective_df` | `float` | Effective degrees of freedom|
| `aic`          | `float` | AIC                         |
| `aicc`         | `float` | AICc                        |

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

```python
import fastlowess import Lowess
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.2, 100)

# Configure model
model = Lowess(fraction=0.5)

# Fit data
result = model.fit(x, y)

print("Smoothed Y:", result.y)
```
