# OnlineLowess — Python API Reference

See also: [fastLowess Python API Reference](python.md)

## Class

### `OnlineLowess`

The `OnlineLowess` class updates the model incrementally with new data points.

**Constructor:**

```python
import fastlowess as fl

online = fl.OnlineLowess(fraction=0.5, window_capacity=50)
```

**Methods:**

```python
import fastlowess as fl
import numpy as np

x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x) + 0.1

online = fl.OnlineLowess(fraction=0.5, window_capacity=50, min_points=3)

# Returns None until min_points (3) are reached
result = online.add_point(x[0], y[0])  # None
result = online.add_point(x[1], y[1])  # None

# Returns OnlineOutput once enough points are available
result = online.add_point(x[2], y[2])
print(result)
# OnlineOutput(y=0.2266)
```

* Adds a single point to the sliding window. Returns an `OnlineOutput` once the window has enough points, or `None` while still filling.

## Options Structure

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
| `y` | `float` | Smoothed value for the latest point |
| `standard_error` | `float \| None` | Standard error (if requested) |
| `residual` | `float \| None` | Residual y − smoothed (if requested) |
| `robustness_weight` | `float \| None` | Robustness weight (if requested) |
| `iterations_used` | `int \| None` | Robustness iterations performed |

## Options

### update_mode

*See: [Execution Modes](../user-guide/adapters.md)*

* `"full"` (default; alias: `"resmooth"`)
* `"incremental"` (alias: `"single"`)
