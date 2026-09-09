# Out-of-Sample Prediction

Evaluate a fitted Batch model at query points that were not in the training set.

## Overview

:::{note} Adapter support
Out-of-sample prediction is available in **Batch** mode only. Streaming and Online modes do not support it.
:::

`LowessResult.predict(new_x, ...)` evaluates the fit at arbitrary query points, like R's `predict(model, newdata)`.

It reuses `fit()`'s own (possibly `delta`-interpolated) smoothed curve for its `y` output — so predicting at a training `x` always exactly reproduces that point's `fit()` output, regardless of `delta`. A fresh local fit is only run when `return_derivative`, `return_se` (or an interval level), or `max_neighbor_distance` needs the actual regression slope or standard error.

Requires `retain_model=True` on the constructor before `fit()`, otherwise `predict()` raises `LowessError`.

---

## Options

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `return_se` | `bool` | `False` | Include standard errors in the output |
| `confidence_level` | `float \| None` | `None` | Confidence interval coverage level (e.g. `0.95`) |
| `prediction_level` | `float \| None` | `None` | Prediction interval coverage level (e.g. `0.95`) |
| `return_derivative` | `bool` | `False` | Include the local fit's derivative (slope) at each query point |
| `extrapolation` | `str` | `"clamp"` | Behavior for query points outside the training `x`-range |
| `max_extrapolation_distance` | `float \| None` | `None` | Under `"linear"` extrapolation, the max allowed distance beyond the training boundary before erroring |
| `max_neighbor_distance` | `float \| None` | `None` | Max allowed distance to the farthest training point in a query's local window before erroring |

### return_se

Computes standard errors for each query point, using the retained model's residual scale and per-point leverage. Required for `confidence_level`/`prediction_level` to be populated. `False` by default.

### confidence_level

Confidence level for the confidence interval around the mean response at each query point (e.g. `0.95`). Uses the same z-score convention as `fit()`'s own confidence intervals. `None` (default) disables it.

### prediction_level

Confidence level for the prediction interval for a new observation at each query point (e.g. `0.95`). Widens using the same MAD-based residual scale `fit()` uses for its own intervals. `None` (default) disables it.

### return_derivative

Includes the local fit's derivative (slope) at each query point in the output. `False` by default.

### extrapolation

Behavior for query points outside `[min(x_train), max(x_train)]`:

| Policy | Behavior |
| --- | --- |
| `"clamp"` (default) | Clamps the query to the nearest boundary window |
| `"linear"` | Linearly extrapolates from the nearest boundary point's local fit and slope |
| `"error"` | Fails the whole call with `LowessError` |

### max_extrapolation_distance

Under `"linear"` extrapolation, the maximum allowed distance beyond the training boundary before `predict()` raises, instead of returning an unbounded value. `None` (default, uncapped).

### max_neighbor_distance

Maximum allowed distance to the farthest training point in a query's local window before `predict()` raises. Guards against an "empty range" blind spot: a query point can fall within `[min(x_train), max(x_train)]` yet still be far from any real training point (e.g. training `x` in `[0,10]` and `[90,100]`, query at `x=50`). `None` (default, uncapped); applies regardless of `extrapolation`.

## Example

### Basic Usage

:::{jupyter-execute}
import fastlowess as fl
import numpy as np

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([2.1, 4.0, 6.2, 8.0, 10.1])

model = fl.Lowess(fraction=0.7, retain_model=True)
result = model.fit(x, y)

new_x = np.array([1.5, 4.5])
prediction = result.predict(new_x)
print("Predicted y:", prediction.y)
:::

### Standard Errors and Derivative

:::{jupyter-execute}
import fastlowess as fl
import numpy as np

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([2.1, 4.0, 6.2, 8.0, 10.1])

model = fl.Lowess(fraction=0.7, retain_model=True)
result = model.fit(x, y)

prediction = result.predict([2.5], return_se=True, return_derivative=True)
print("y:", prediction.y)
print("SE:", prediction.standard_errors)
print("Derivative:", prediction.derivative)
:::

### Linear Extrapolation

:::{jupyter-execute}
import fastlowess as fl
import numpy as np

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([2.1, 4.0, 6.2, 8.0, 10.1])

model = fl.Lowess(fraction=0.7, retain_model=True)
result = model.fit(x, y)

prediction = result.predict([10.0], extrapolation="linear")
print("Extrapolated y:", prediction.y)
:::
