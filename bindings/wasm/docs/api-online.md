# OnlineLowess — WebAssembly API Reference

See also: [fastLowess WebAssembly API Reference](wasm.md)

## Class

### `OnlineLowess`

The `OnlineLowess` class updates the model incrementally with new data points.

**Constructor:**

```javascript
const { OnlineLowess } = require('fastlowess-wasm');

const online = new OnlineLowess({ fraction: 0.5 }, { window_capacity: 50, min_points: 3 });
console.log("typeof add_point:", typeof online.add_point);
```

```output
typeof add_point: function
```

* `options`: An object containing `LowessOptions` fields.
* `onlineOptions`: An object containing `OnlineOptions` fields.

**Methods:**

```javascript
const { OnlineLowess } = require('fastlowess-wasm');

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

* Adds a single point to the sliding window. Returns an `OnlineOutput` once enough points are available, or `null` while the window is still filling.

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

Returned by `add_point()` once the window has enough points (`null` until then).

| Field | Type | Description |
| --- | --- | --- |
| `y` | `number` | Smoothed value for the latest point |
| `standard_error` | `number \| undefined` | Standard error (if requested) |
| `residual` | `number \| undefined` | Residual y − smoothed (if requested) |
| `robustness_weight` | `number \| undefined` | Robustness weight (if requested) |
| `iterations_used` | `number \| undefined` | Robustness iterations performed |

## Options

### update_mode

*See: [Execution Modes](../user-guide/adapters.md)*

* `"full"` (default; alias: `"resmooth"`)
* `"incremental"` (alias: `"single"`)
