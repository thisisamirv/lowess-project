# OnlineLowess API

See also: [fastLowess](api.md)

## When to Use

- Data arrives incrementally (sensors, streams)
- Need real-time smoothed values
- Fixed memory budget

![Online Adapter](../assets/diagrams/online_comparison.svg)

## Class

### `OnlineLowess`

The `OnlineLowess` class updates the model incrementally with new data points.

**Constructor:**

:::{jupyter-execute}
import fastlowess as fl

online = fl.OnlineLowess(fraction=0.5, window_capacity=50)
:::

#### `add_point(x, y)`

Adds a single point to the sliding window and returns the smoothed value for that point, or `None` while the window is still filling up (fewer than `min_points` seen so far). Once the window reaches `window_capacity`, each new point evicts the oldest one, so memory stays bounded regardless of how much history has passed through. `update_mode` controls how much work each call does: `"incremental"` re-fits only the newest point, while `"full"` re-smooths the entire window for a more accurate but slower result.

:::{jupyter-execute}
import fastlowess as fl
import numpy as np

x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x) + 0.1

online = fl.OnlineLowess(fraction=0.5, window_capacity=50, min_points=3)

result = online.add_point(x[0], y[0])  # None
result = online.add_point(x[1], y[1])  # None

result = online.add_point(x[2], y[2])
print(result)
:::

## Options Structure

### `OnlineOptions` (inherits `LowessOptions`)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `fraction` | `float` | `0.67` | Smoothing fraction (bandwidth) |
| `iterations` | `int` | `0` | Number of robustifying iterations (requires `update_mode = "full"`) |
| `weight_function` | `str` | `"tricube"` | Weight function name |
| `robustness_method` | `str` | `"bisquare"` | Robustness method name |
| `delta` | `float` | `None` | Interpolation distance (`None` disables interpolation); positive values require `update_mode="full"` |
| `zero_weight_fallback` | `str` | `"use_local_mean"` | Zero-weight handling strategy |
| `boundary_policy` | `str` | `"extend"` | Boundary handling policy |
| `scaling_method` | `str` | `"mad"` | Residual scaling method |
| `auto_converge` | `float` | `None` | Auto-convergence tolerance; requires `update_mode="full"` and `iterations > 0` |
| `missing` | `str` | `"error"` | Policy for non-finite (NaN/Inf) values in each point |
| `window_capacity` | `int` | `1000` | Max points in sliding window |
| `min_points` | `int` | `2` | Min points before smoothing starts |
| `update_mode` | `str` | `"incremental"` | Update mode (`"full"` or `"incremental"`) |
| `outputs` | `Sequence[str]` | `[]` | Select `weights`, `derivative`, and/or `se`; `se` requires `update_mode="full"` |
| `intervals` | `dict` | `None` | Grouped `confidence`, `prediction`, and per-window `bootstrap` options (requires `update_mode="full"`) |
| `seed` | `int` | `None` | Reproducible bootstrap draws for each full-update window |

Incremental mode fits only the newest point. Positive `delta` is rejected there, and `auto_converge` requires full mode with at least one robustness iteration.

Cross-validation, GPU `backend`, `custom_weights`, `"sorted"`, `"diagnostics"`, `"residuals"`, and `parallel` are Batch-only and not available here; see [fastLowess](api.md) for those.

## Options

### fraction

`fraction` is the most important parameter: it controls the size of the local neighbourhood used at each point.

| Range | Effect | Use case |
| --- | --- | --- |
| 0.1-0.3 | Fine detail | Rapidly changing signals |
| 0.3-0.5 | Balanced | General purpose |
| 0.5-0.7 | Heavy smoothing | Noisy data |
| 0.7-1.0 | Very smooth | Trend extraction |

### iterations

`iterations` controls robustness to outliers, at the cost of speed. Requires `update_mode = "full"`; the default `"incremental"` mode performs a non-robust single-point fit.

| Value | Effect | Performance |
| --- | --- | --- |
| 0 | No robustness | Fastest |
| 1-3 | Moderate | Recommended |
| 4-6 | Strong | Contaminated data |
| 7+ | Very strong | Heavy outliers |

### weight_function

*See: [Weight Functions](../weighting/kernels.md)*

- `"tricube"` (default)
- `"epanechnikov"`
- `"gaussian"`
- `"uniform"` (alias: `"boxcar"`)
- `"biweight"` (alias: `"bisquare"`)
- `"triangle"` (alias: `"triangular"`)
- `"cosine"`

### robustness_method

*See: [Robustness](../weighting/robustness.md)*

- `"bisquare"` (default; alias: `"biweight"`)
- `"huber"`
- `"talwar"`

### delta

Points within `delta` of each other on the x-axis share the same local fit instead of each computing its own regression — an interpolation shortcut that trades a small amount of accuracy for a large speedup on dense, evenly-spaced data. `None` (default) auto-sets it to `0.0` in Online mode, i.e. interpolation is disabled and every point is fit exactly.

### zero_weight_fallback

Behavior when all neighborhood weights are zero:

| Option | Behavior |
| --- | --- |
| `"use_local_mean"` (default; aliases: `"local_mean"`, `"mean"`) | Use the mean of the neighborhood |
| `"return_original"` (alias: `"original"`) | Return the original y value |
| `"return_none"` (alias: `"none"`) | Return `NaN` |

### boundary_policy

*See: [Boundary Handling](../advanced/boundary.md)*

- `"extend"` (default; alias: `"pad"`)
- `"reflect"` (alias: `"mirror"`)
- `"zero"`
- `"noboundary"` (alias: `"none"`)

### scaling_method

*See: [Scaling Methods](../weighting/scaling.md)*

- `"mad"` (default; alias: `"median_absolute_deviation"`)
- `"mar"` (alias: `"median_absolute_residual"`)
- `"mean"` (alias: `"mean_absolute_residual"`)

### auto_converge

*See: [Robustness](../weighting/robustness.md#auto-convergence)*

Convergence tolerance for early stopping of robustness iterations. `None` (default) disables early stopping.

### missing

Policy for handling a non-finite (NaN/Inf) `x` or `y` value passed to `add_point`:

| Option | Behavior |
| --- | --- |
| `"error"` (default) | Raise an error |
| `"drop"` | Silently ignore the point — `add_point` returns `None` instead of adding it to the window |

### window_capacity

Maximum number of most recent points kept in the sliding window; older points are discarded as new ones arrive. Each `add_point()` call costs O(`window_capacity`) rather than growing with total history.

### min_points

Minimum number of points required before smoothing starts. `add_point()` returns `None` until the window reaches this size.

### update_mode

*See: [Execution Modes](../guide/adapter-choice.md)*

| Mode | Alias | Behavior | Speed |
| --- | --- | --- | --- |
| `"incremental"` (default) | `"single"` | Update only affected fits | Faster |
| `"full"` | `"resmooth"` | Recompute entire window | More accurate |

### outputs

Use `outputs=["weights", "derivative", "se"]` to select optional online result components. The `"derivative"` output includes the latest point's local WLS slope. `"se"` requires `update_mode="full"`.

*See: [Intervals](../guide/intervals.md)*

Populates `standard_error` — but only when combined with `update_mode="full"`. The fast `"incremental"` path (the default) never computes standard errors, so combining `"se"` (or `intervals`) with anything other than `"full"` raises at construction time, rather than silently leaving `standard_error` as `None`.

### intervals

*See: [Intervals](../guide/intervals.md)*

A dict such as `{"confidence": 0.95, "prediction": 0.95, "bootstrap": 200}`, populating `confidence_lower`/`confidence_upper` and `prediction_lower`/`prediction_upper` for the latest point. Same `update_mode="full"` requirement as `"se"`. `bootstrap` (at least `2`) refits each sliding window from resampled residuals.

### seed

Seeds bootstrap draws. Each full-update window restarts from the same seed. It does not enable bootstrap by itself; `0` is a valid seed.

## Result Structure

### `OnlineOutput`

Returned by `add_point()` once the window has enough points (`None` until then).

| Field | Type | Description |
| --- | --- | --- |
| `y` | `float` | Smoothed value for the latest point |
| `standard_error` | `float \| None` | Populated when `"se"` is set (requires `update_mode="full"`); otherwise always `None` |
| `confidence_lower` / `confidence_upper` | `float \| None` | Confidence interval bounds around the mean response, if `intervals["confidence"]` was set (requires `update_mode="full"`) |
| `prediction_lower` / `prediction_upper` | `float \| None` | Prediction interval bounds for a new observation, if `intervals["prediction"]` was set (requires `update_mode="full"`) |
| `residual` | `float \| None` | Residual y − smoothed; always present (there is no `"residuals"` output for Online) |
| `robustness_weight` | `float \| None` | Robustness weight, if `"weights"` was set |
| `iterations_used` | `int \| None` | Robustness iterations performed |
| `derivative` | `float \| None` | Local fit derivative/slope for the latest point, if `"derivative"` was requested |

There is no `Diagnostics` object or `"diagnostics"` output for `OnlineLowess`: `OnlineOutput` carries no diagnostics field, since diagnostics like RMSE/R² need more than one point's worth of history to be meaningful.
