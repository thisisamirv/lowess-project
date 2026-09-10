---
title: OnlineLowess API
---
See also: [fastLowess](api.md)

## When to Use

- Data arrives incrementally (sensors, streams)
- Need real-time smoothed values
- Fixed memory budget

![Online Adapter](../../assets/diagrams/online_comparison.svg)

## Class

### `OnlineLowess`

The `OnlineLowess` class updates the model incrementally with new data points.

**Constructor:**

```javascript
const { OnlineLowess } = require('fastlowess');

const online = new OnlineLowess({ fraction: 0.5 }, { window_capacity: 50, min_points: 3 });
// Feed enough points to pass min_points threshold
for (let i = 0; i < 4; i++) {
    const result = online.add_point(i, Math.sin(i * 0.5));
    if (result !== null) console.log("Online smoothed at x=" + i + ":", result.y.toFixed(4));
}
```

```output
Online smoothed at x=2: 0.8415
Online smoothed at x=3: 0.9975
```

- `options`: An object containing `OnlineSmoothOptions` fields (a subset of the Batch `LowessOptions` fields — see below).
- `onlineOptions`: An object containing `OnlineOptions` fields.

#### `add_point(x, y)`

Adds a single point to the sliding window and returns the smoothed value for that point, or `null` while the window is still filling up (fewer than `min_points` seen so far). Once the window reaches `window_capacity`, each new point evicts the oldest one, so memory stays bounded regardless of how much history has passed through. `update_mode` controls how much work each call does: `"incremental"` re-fits only the newest point, while `"full"` re-smooths the entire window for a more accurate but slower result.

```javascript
const { OnlineLowess } = require('fastlowess');

const n = 100;
const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
const y = Float64Array.from(x, xi => Math.sin(xi) + 0.1);

const online = new OnlineLowess({ fraction: 0.5 }, { window_capacity: 50, min_points: 3 });

// Returns null until min_points (3) are reached
online.add_point(x[0], y[0]);  // null
online.add_point(x[1], y[1]);  // null

// Returns OnlineOutput once enough points are available
const result = online.add_point(x[2], y[2]);
console.log("Smoothed y:", result.y);
```

```output
Smoothed y: 0.22659245357374927
```

## Options Structure

### `OnlineSmoothOptions`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `fraction` | `number` | `0.67` | Smoothing fraction (bandwidth) |
| `iterations` | `number` | `3` | Number of robustifying iterations |
| `delta` | `number` | `NaN` | Interpolation distance (`NaN` auto-sets it to 0.0 in Online, i.e. interpolation disabled) |
| `weight_function` | `string` | `"tricube"` | Weight function name |
| `robustness_method` | `string` | `"bisquare"` | Robustness method name |
| `scaling_method` | `string` | `"mad"` | Residual scaling method |
| `boundary_policy` | `string` | `"extend"` | Boundary handling policy |
| `zero_weight_fallback` | `string` | `"use_local_mean"` | Zero-weight handling |
| `missing` | `string` | `"error"` | Policy for non-finite (NaN/Inf) values in each point |
| `auto_converge` | `number` | `null` | Auto-convergence tolerance |
| `return_robustness_weights` | `boolean` | `false` | Include `robustness_weight` in result |
| `window_capacity` | `number` | `1000` | Max points in sliding window |
| `min_points` | `number` | `2` | Min points before smoothing starts |
| `update_mode` | `string` | `"incremental"` | Update mode (`"full"` or `"incremental"`) |
| `return_derivative` | `boolean` | `false` | Include the latest point's local fit derivative (slope) in result |
| `return_se` | `boolean` | `false` | Populate `standard_error` in the result (requires `update_mode: "full"`; throws if combined with `"incremental"`) |
| `confidence_intervals` | `number` | `null` | Confidence level (e.g., 0.95); populates `confidence_lower`/`confidence_upper` (requires `update_mode: "full"`) |
| `prediction_intervals` | `number` | `null` | Prediction level (e.g., 0.95); populates `prediction_lower`/`prediction_upper` (requires `update_mode: "full"`) |

Cross-validation, GPU `backend`, `custom_weights`, `return_sorted`, `return_diagnostics`, `return_residuals`, and `parallel` are Batch-only (or Batch/Streaming-only) and not available here; see [fastLowess](api.md) for those.

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

`iterations` controls robustness to outliers, at the cost of speed.

| Value | Effect | Performance |
| --- | --- | --- |
| 0 | No robustness | Fastest |
| 1-3 | Moderate | Recommended |
| 4-6 | Strong | Contaminated data |
| 7+ | Very strong | Heavy outliers |

### delta

Points within `delta` of each other on the x-axis share the same local fit instead of each computing its own regression — an interpolation shortcut that trades a small amount of accuracy for a large speedup on dense, evenly-spaced data. `NaN` (default) auto-sets it to `0` in Online mode, i.e. interpolation is disabled and every point is fit exactly.

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

### scaling_method

*See: [Scaling Methods](../weighting/scaling.md)*

- `"mad"` (default; alias: `"median_absolute_deviation"`)
- `"mar"` (alias: `"median_absolute_residual"`)
- `"mean"` (alias: `"mean_absolute_residual"`)

### boundary_policy

*See: [Boundary Handling](../advanced/boundary.md)*

- `"extend"` (default; alias: `"pad"`)
- `"reflect"` (alias: `"mirror"`)
- `"zero"`
- `"noboundary"` (alias: `"none"`)

### zero_weight_fallback

Behavior when all neighborhood weights are zero:

| Option | Behavior |
| --- | --- |
| `"use_local_mean"` (default; aliases: `"local_mean"`, `"mean"`) | Use the mean of the neighborhood |
| `"return_original"` (alias: `"original"`) | Return the original y value |
| `"return_none"` (alias: `"none"`) | Return `NaN` |

### missing

Policy for handling a non-finite (NaN/Inf) `x` or `y` value passed to `add_point`:

| Option | Behavior |
| --- | --- |
| `"error"` (default) | Throw an error |
| `"drop"` | Silently ignore the point — `add_point` returns `null` instead of adding it to the window |

### auto_converge

*See: [Robustness](../weighting/robustness.md#auto-convergence)*

Convergence tolerance for early stopping of robustness iterations. `null` (default) disables early stopping.

### return_robustness_weights

Include the robustness weight for the latest point (from the last robustness iteration) in the result.

- `false` (default) — leaves `result.robustness_weight` as `null`
- `true` — populates `result.robustness_weight`

### window_capacity

Maximum number of most recent points kept in the sliding window; older points are discarded as new ones arrive. Each `add_point()` call costs O(`window_capacity`) rather than growing with total history.

### min_points

Minimum number of points required before smoothing starts. `add_point()` returns `null` until the window reaches this size.

### update_mode

*See: [Execution Modes](../guide/adapter-choice.md)*

| Mode | Alias | Behavior | Speed |
| --- | --- | --- | --- |
| `"incremental"` (default) | `"single"` | Update only affected fits | Faster |
| `"full"` | `"resmooth"` | Recompute entire window | More accurate |

### return_derivative

Each point's local WLS fit already computes a slope internally; this exposes the latest point's slope (rate of change of the smoothed curve) in `OnlineOutput.derivative` at effectively no extra computation cost.

- `false` (default) — leaves `result.derivative` as `null`
- `true` — populates it

### return_se

*See: [Intervals](../guide/intervals.md)*

Populates `standard_error` — but only when combined with `update_mode: "full"`. The fast `"incremental"` path (the default) never computes standard errors, so combining `return_se` (or `confidence_intervals`/`prediction_intervals`) with anything other than `"full"` throws at construction time, rather than silently leaving `standard_error` as `null`.

- `false` (default) — leaves `standard_error` as `null`
- `true` — populates `standard_error`, and requires `update_mode: "full"`

### confidence_intervals

*See: [Intervals](../guide/intervals.md)*

Confidence level for the confidence interval around the mean response at the latest point (e.g. `0.95`), populating `confidence_lower`/`confidence_upper`. Same `update_mode: "full"` requirement as `return_se`. `null` (default) disables confidence intervals.

### prediction_intervals

*See: [Intervals](../guide/intervals.md)*

Confidence level for the prediction interval for a new observation at the latest point (e.g. `0.95`), populating `prediction_lower`/`prediction_upper`. Same `update_mode: "full"` requirement as `return_se`. `null` (default) disables prediction intervals.

## Result Structure

### `OnlineOutput`

Returned by `add_point()` once the window has enough points (`null` until then).

| Field | Type | Description |
| --- | --- | --- |
| `y` | `number` | Smoothed value for the latest point |
| `standard_error` | `number \| null` | Populated when `return_se` is set (requires `update_mode: "full"`); otherwise always `null` |
| `confidence_lower` / `confidence_upper` | `number \| null` | Confidence interval bounds around the mean response, if `confidence_intervals` was set (requires `update_mode: "full"`) |
| `prediction_lower` / `prediction_upper` | `number \| null` | Prediction interval bounds for a new observation, if `prediction_intervals` was set (requires `update_mode: "full"`) |
| `residual` | `number \| null` | Residual y − smoothed; always present (there is no `return_residuals` option for Online) |
| `robustness_weight` | `number \| null` | Robustness weight, if `return_robustness_weights` was set |
| `iterations_used` | `number \| null` | Robustness iterations performed |
| `derivative` | `number \| null` | Local fit derivative/slope for the latest point, if `return_derivative` was set |

There is no `Diagnostics` object or `return_diagnostics` option for `OnlineLowess`: `OnlineOutput` carries no diagnostics field, since diagnostics like RMSE/R² need more than one point's worth of history to be meaningful.
