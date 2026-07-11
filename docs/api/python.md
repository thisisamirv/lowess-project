# fastLowess Python API Reference

The Python bindings provide a high-performance interface to the core Rust library, mirroring the Rust API structure.

## Classes

### `Lowess`

The `Lowess` class allows configuring the LOWESS parameters once and fitting multiple datasets using those parameters.

**Constructor:**

```python
import fastlowess as fl

model = fl.Lowess(fraction=0.5, iterations=3)
```

* `kwargs`: Keyword arguments corresponding to `LowessOptions` fields.

**Methods:**

```python
import fastlowess as fl
import numpy as np

rng = np.random.default_rng(42)
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x) + rng.normal(0, 0.3, 100)

model = fl.Lowess(fraction=0.5)
result = model.fit(x, y, custom_weights=None)
```

* Fits the model to the provided `x` and `y` array-like objects.
* `custom_weights`: Optional array of per-observation weights. All values must be ≥ 0 and length must match `x`. Batch only.
* Returns a `LowessResult` object containing the smoothed values and optional diagnostics.

### `StreamingLowess`

The `StreamingLowess` class processes data in chunks, suitable for very large datasets or streaming applications.

**Constructor:**

```python
import fastlowess as fl

stream = fl.StreamingLowess(chunk_size=50, overlap=10)
```

* `kwargs`: Keyword arguments corresponding to `LowessOptions` and `StreamingOptions` fields.

**Methods:**

```python
import fastlowess as fl
import numpy as np

rng = np.random.default_rng(42)
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x) + rng.normal(0, 0.3, 100)

stream = fl.StreamingLowess(chunk_size=50, overlap=10)
partial_result = stream.process_chunk(x[:50], y[:50])
```

* Processes a chunk of data. Returns partial results.

```python
import fastlowess as fl
import numpy as np

rng = np.random.default_rng(42)
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x) + rng.normal(0, 0.3, 100)

stream = fl.StreamingLowess(chunk_size=50, overlap=10)
stream.process_chunk(x, y)
final_result = stream.finalize()
```

* Finalizes the smoothing process and returns any remaining buffered results.

### `OnlineLowess`

The `OnlineLowess` class updates the model incrementally with new data points.

**Constructor:**

```python
import fastlowess as fl

online = fl.OnlineLowess(fraction=0.3, window_capacity=50)
```

* `kwargs`: Keyword arguments corresponding to `LowessOptions` and `OnlineOptions` fields.

**Methods:**

```python
import fastlowess as fl

online = fl.OnlineLowess(fraction=0.3, window_capacity=50)
result = online.add_point(1.0, 2.0)  # returns OnlineOutput | None
```

* Adds a single point to the sliding window. Returns an `OnlineOutput` once the window has enough points, or `None` while still filling.

## Options Structures

### `LowessOptions`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `fraction` | `float` | `0.67` | Smoothing fraction (bandwidth) |
| `iterations` | `int` | `3` | Number of robustifying iterations |
| `delta` | `float` | `None` | Interpolation distance (None for auto) |
| `weight_function` | `str` | `"tricube"` | Weight function name |
| `robustness_method` | `str` | `"bisquare"` | Robustness method name |
| `scaling_method` | `str` | `"mad"` | Residual scaling method |
| `boundary_policy` | `str` | `"extend"` | Boundary handling policy |
| `zero_weight_fallback` | `str` | `"use_local_mean"` | Zero-weight handling strategy |
| `auto_converge` | `float` | `None` | Auto-convergence tolerance |
| `confidence_intervals` | `float` | `None` | Confidence level (e.g., 0.95) |
| `prediction_intervals` | `float` | `None` | Prediction level (e.g., 0.95) |
| `return_diagnostics` | `bool` | `False` | Include diagnostics in result |
| `return_residuals` | `bool` | `False` | Include residuals in result |
| `return_robustness_weights` | `bool` | `False` | Include weights in result |
| `return_se` | `bool` | `False` | Return standard errors |
| `parallel` | `bool` | `True` | Enable parallel execution |
| `cv_method` | `str` | `"kfold"` | CV method (`"kfold"` or `"loocv"`) (Batch only) |
| `cv_k` | `int` | `5` | Number of folds for k-fold CV (Batch only) |
| `cv_fractions` | `list[float]` | `None` | Fractions to test for cross-validation (Batch only) |
| `cv_seed` | `int` | `None` | Random seed for cross-validation shuffling (Batch only) |
| `custom_weights` | `list[float]` | `None` | Per-observation case weights — passed to `fit()`, not the constructor (Batch only) |

### `StreamingOptions` (inherits `LowessOptions`)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `chunk_size` | `int` | `5000` | Data chunk size |
| `overlap` | `int` | `500` | Overlap between chunks |
| `merge_strategy` | `str` | `"weighted_average"` | Strategy for blending overlap regions |

### `OnlineOptions` (inherits `LowessOptions`)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `window_capacity` | `int` | `1000` | Max points in sliding window |
| `min_points` | `int` | `3` | Min points before smoothing starts |
| `update_mode` | `str` | `"full"` | Update mode (`"full"` or `"incremental"`) |
| `parallel` | `bool` | `False` | Enable parallel execution (off by default; online LOWESS fits one point at a time) |

## Result Structure

### `OnlineOutput`

Returned by `add_point()` once the window has enough points (`None` until then).

| Field | Type | Description |
| --- | --- | --- |
| `smoothed` | `float` | Smoothed value for the latest point |
| `std_error` | `float \| None` | Standard error (if requested) |
| `residual` | `float \| None` | Residual y − smoothed (if requested) |
| `robustness_weight` | `float \| None` | Robustness weight (if requested) |
| `iterations_used` | `int \| None` | Robustness iterations performed |

### `LowessResult`

| Field | Type | Description |
| --- | --- | --- |
| `x` | `ndarray` | Sorted x values |
| `y` | `ndarray` | Smoothed y values |
| `fraction_used` | `float` | Fraction used (set or selected by CV) |
| `iterations_used` | `int \| None` | Robustness iterations actually performed |
| `standard_errors` | `ndarray \| None` | Per-point standard errors |
| `confidence_lower` | `ndarray \| None` | Lower confidence bounds |
| `confidence_upper` | `ndarray \| None` | Upper confidence bounds |
| `prediction_lower` | `ndarray \| None` | Lower prediction bounds |
| `prediction_upper` | `ndarray \| None` | Upper prediction bounds |
| `residuals` | `ndarray \| None` | Residuals (if `return_residuals`) |
| `robustness_weights` | `ndarray \| None` | Robustness weights (if `return_robustness_weights`) |
| `cv_scores` | `ndarray \| None` | CV score per tested fraction |
| `diagnostics` | `Diagnostics \| None` | Fit metrics (if `return_diagnostics`) |

### `Diagnostics`

| Field | Type | Description |
| --- | --- | --- |
| `rmse` | `float` | Root Mean Squared Error |
| `mae` | `float` | Mean Absolute Error |
| `r_squared` | `float` | R-squared |
| `residual_sd` | `float` | Residual standard deviation |
| `effective_df` | `float \| None` | Effective degrees of freedom (`None` if not computed) |
| `aic` | `float \| None` | AIC (`None` if not computed) |
| `aicc` | `float \| None` | AICc (`None` if not computed) |

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

```python
from fastlowess import Lowess
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.2, 100)

# Configure model
model = Lowess(fraction=0.5)

# Fit data
result = model.fit(x, y)

print("Smoothed Y:", result.y)
```
