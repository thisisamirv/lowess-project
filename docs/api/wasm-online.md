# OnlineLowess — WebAssembly API Reference

See also: [fastLowess WebAssembly API Reference](wasm.md)

## Class

### `OnlineLowess`

The `OnlineLowess` class updates the model incrementally with new data points.

**Constructor:**

```javascript
const { OnlineLowess } = require('fastlowess-wasm');

const online = new OnlineLowess({ fraction: 0.3 }, { window_capacity: 50, min_points: 5 });
```

* `options`: An object containing `LowessOptions` fields.
* `onlineOptions`: An object containing `OnlineOptions` fields.

**Methods:**

```javascript
const { OnlineLowess } = require('fastlowess-wasm');

const online = new OnlineLowess({ fraction: 0.3 }, { window_capacity: 50, min_points: 5 });
const result = online.add_point(1.0, 2.0);  // returns OnlineOutput | undefined
```

* Adds a single point to the sliding window. Returns an `OnlineOutput` once enough points are available, or `undefined` while the window is still filling.

## Options Structure

### `OnlineOptions` (inherits `LowessOptions`)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `window_capacity` | `number` | `1000` | Max points in sliding window |
| `min_points` | `number` | `3` | Min points before smoothing starts |
| `update_mode` | `string` | `"full"` | Update mode (`"full"` or `"incremental"`) |
| `parallel` | `boolean` | `false` | Enable parallel execution (off by default; online LOWESS fits one point at a time) |

## Result Structure

### `OnlineOutput`

Returned by `add_point()` once the window has enough points (`undefined` until then).

| Field | Type | Description |
| --- | --- | --- |
| `smoothed` | `number` | Smoothed value for the latest point |
| `std_error` | `number \| undefined` | Standard error (if requested) |
| `residual` | `number \| undefined` | Residual y − smoothed (if requested) |
| `robustness_weight` | `number \| undefined` | Robustness weight (if requested) |
| `iterations_used` | `number \| undefined` | Robustness iterations performed |

## Options

### update_mode

*See: [Execution Modes](../user-guide/adapters.md)*

* `"full"` (default; alias: `"resmooth"`)
* `"incremental"` (alias: `"single"`)
