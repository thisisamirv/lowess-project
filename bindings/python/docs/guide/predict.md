# Out-of-Sample Prediction

Evaluate a fitted Batch model at query points that were not in the training set.

## Overview

:::{note} Adapter support
Out-of-sample prediction is available in **Batch** mode only. Streaming and Online modes do not support it.
:::

`LowessResult.predict(new_x, ...)` evaluates the fit at arbitrary query points, like R's `predict(model, newdata)`.

It reuses `fit()`'s own (possibly `delta`-interpolated) smoothed curve for its `y` output — so predicting at a training `x` always exactly reproduces that point's `fit()` output, regardless of `delta`. A fresh local fit is only run when `"derivative"`, `"se"` (or an interval level), or `max_neighbor_distance` needs the actual regression slope or standard error.

Requires `retain_model=True` on the constructor before `fit()`, otherwise `predict()` raises `LowessError`.

---

## Options

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `outputs` | `Sequence[str]` | `[]` | Select `"se"` and/or `"derivative"` |
| `intervals` | `dict \| None` | `None` | Grouped `confidence`, `prediction`, and optional `bootstrap` options |
| `seed` | `int \| None` | `None` | Reproducible prediction-time bootstrap draws; `0` is valid |
| `extrapolation` | `str` | `"clamp"` | Behavior for query points outside the training `x`-range |
| `max_extrapolation_distance` | `float \| None` | `None` | Under `"linear"` extrapolation, the max allowed distance beyond the training boundary before erroring |
| `max_neighbor_distance` | `float \| None` | `None` | Max allowed distance to the farthest training point in a query's local window before erroring |

### outputs

Use `outputs=["se", "derivative"]` to select optional prediction components. `"se"` computes standard errors for each query point from the retained model's residual scale and per-point leverage; `"derivative"` includes the local fit's slope. Analytic intervals compute their required standard errors even without an explicit `"se"` output.

### intervals

A dict such as `{"confidence": 0.95, "prediction": 0.95, "bootstrap": 200}`. `confidence` bounds the mean response and `prediction` bounds a new observation at each query point. Without bootstrap, these use normal-theory standard errors and the retained residual scale.

Set `bootstrap` to at least `2` to resample the retained Batch residuals, refit the model, and calculate query-point percentile intervals and standard errors. Unknown keys raise `ValueError`.

### seed

Seeds prediction-time bootstrap draws, independently of the fit/CV `seed` on `Lowess`. It does not enable bootstrap by itself.

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

prediction = result.predict(np.array([2.5]), outputs=["se", "derivative"])
print("y:", prediction.y)
print("SE:", prediction.standard_errors)
print("Derivative:", prediction.derivative)
:::

### Bootstrap Intervals

:::{jupyter-execute}
import fastlowess as fl
import numpy as np

x = np.arange(1.0, 11.0)
y = np.array([2.1, 4.0, 6.2, 8.0, 10.1, 11.8, 14.2, 16.1, 17.9, 20.2])

model = fl.Lowess(fraction=0.7, retain_model=True)
result = model.fit(x, y)

prediction = result.predict(
    np.array([2.5, 4.5]),
    intervals={"confidence": 0.95, "prediction": 0.95, "bootstrap": 200},
    seed=7,
)
print("CI present:", prediction.confidence_lower is not None)
:::

### Linear Extrapolation

:::{jupyter-execute}
import fastlowess as fl
import numpy as np

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([2.1, 4.0, 6.2, 8.0, 10.1])

model = fl.Lowess(fraction=0.7, retain_model=True)
result = model.fit(x, y)

prediction = result.predict(np.array([10.0]), extrapolation="linear")
print("Extrapolated y:", prediction.y)
:::
