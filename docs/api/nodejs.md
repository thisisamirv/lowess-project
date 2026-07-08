# fastLowess Node.js API Reference

The Node.js bindings provide a high-performance interface to the core Rust library, mirroring the Rust API structure.

## Classes

### `Lowess`

The `Lowess` class allows configuring the LOWESS parameters once and fitting multiple datasets using those parameters.

**Constructor:**

```javascript
const model = new Lowess(options);
```

* `options`: An object containing `LowessOptions` fields.

**Methods:**

```javascript
const result = model.fit(x, y);
```

* Fits the model to the provided `x` and `y` typed arrays.
* Returns a `LowessResult` object containing the smoothed values and optional diagnostics.

### `StreamingLowess`

The `StreamingLowess` class processes data in chunks, suitable for very large datasets or streaming applications.

**Constructor:**

```javascript
const stream = new StreamingLowess(options, streamingOptions);
```

* `options`: An object containing `LowessOptions` fields.
* `streamingOptions`: An object containing `StreamingOptions` fields.

**Methods:**

```javascript
const partialResult = stream.process_chunk(x, y);
```

* Processes a chunk of data. Returns partial results.

```javascript
const finalResult = stream.finalize();
```

* Finalizes the smoothing process and returns any remaining buffered results.

### `OnlineLowess`

The `OnlineLowess` class updates the model incrementally with new data points.

**Constructor:**

```javascript
const online = new OnlineLowess(options, onlineOptions);
```

* `options`: An object containing `LowessOptions` fields.
* `onlineOptions`: An object containing `OnlineOptions` fields.

**Methods:**

```javascript
const result = online.add_points(x, y);
```

* Adds new points to the model and returns the smoothed values (retrospective or prospective depending on mode).

## Options Structures

### `LowessOptions`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `fraction` | `number` | `0.67` | Smoothing fraction (bandwidth) |
| `iterations` | `number` | `3` | Number of robustifying iterations |
| `delta` | `number` | `NaN` | Interpolation distance (NaN for auto) |
| `weight_function` | `string` | `"tricube"` | Weight function name |
| `robustness_method` | `string` | `"bisquare"` | Robustness method name |
| `scaling_method` | `string` | `"mad"` | Residual scaling method |
| `boundary_policy` | `string` | `"extend"` | Boundary handling policy |
| `zero_weight_fallback` | `string` | `"useLocalMean"` | Zero-weight handling strategy |
| `auto_converge` | `number` | `null` | Auto-convergence tolerance |
| `confidence_intervals` | `number` | `null` | Confidence level (e.g., 0.95) |
| `prediction_intervals` | `number` | `null` | Prediction level (e.g., 0.95) |
| `return_diagnostics` | `boolean` | `false` | Include diagnostics in result |
| `return_residuals` | `boolean` | `false` | Include residuals in result |
| `return_robustness_weights` | `boolean` | `false` | Include weights in result |
| `parallel` | `boolean` | `true` | Enable parallel execution |
| `cv_method` | `string` | `"kfold"` | Cross-validation method ("kfold") |
| `cv_k` | `number` | `5` | Number of CV folds |
| `cv_fractions` | `number[]` | `null` | Manual fractions for CV grid |

### `StreamingOptions`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `chunk_size` | `number` | `5000` | Data chunk size |
| `overlap` | `number` | `500` | Overlap size (-1 for auto) |
| `merge_strategy` | `string` | `"average"` | Merge strategy for overlap |

### `OnlineOptions`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `window_capacity` | `number` | `100` | Max window size |
| `min_points` | `number` | `2` | Min points before smoothing |
| `update_mode` | `string` | `"incremental"` | Update mode ("full" or "incremental") |

## Result Structure

### `LowessResult`

| Field | Type | Description |
| --- | --- | --- |
| `x` | `Float64Array` | Smoothed X coordinates |
| `y` | `Float64Array` | Smoothed Y coordinates |
| `valid` | `boolean` | True if result is valid |
| `error` | `string` | Error message if failed |
| `diagnostics` | `Diagnostics` | Diagnostic metrics object |
| `residuals` | `Float64Array` | Residuals (if requested) |
| `confidence_lower` | `Float64Array` | Lower CI bounds |
| `confidence_upper` | `Float64Array` | Upper CI bounds |
| `prediction_lower` | `Float64Array` | Lower PI bounds |
| `prediction_upper` | `Float64Array` | Upper PI bounds |
| `robustness_weights` | `Float64Array` | Robustness weights |

### `Diagnostics`

| Field | Type | Description |
| --- | --- | --- |
| `rmse` | `number` | Root Mean Squared Error |
| `mae` | `number` | Mean Absolute Error |
| `r_squared` | `number` | R-squared |
| `residual_sd` | `number` | Residual standard deviation |
| `effective_df` | `number` | Effective degrees of freedom |
| `aic` | `number` | AIC |
| `aicc` | `number` | AICc |

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
* `"noBoundary"`

### Scaling Methods

* `"mad"` (default - Median Absolute Deviation)
* `"mar"` (Median Absolute Residual)
* `"mean"` (Mean Absolute Residual)

### Zero Weight Fallback

* `"useLocalMean"` (default)
* `"returnOriginal"`
* `"returnNone"`

### Merge Strategies (Streaming)

* `"weighted"` (default - weighted average of overlapping chunks)
* `"average"`
* `"left"`
* `"right"`

### Update Modes (Online)

* `"full"` (default - re-smooth entire window)
* `"incremental"` (O(1) update using existing fit)

## Example

```javascript
const { Lowess } = require('fastlowess');

const x = new Float64Array([1, 2, 3, 4, 5]);
const y = new Float64Array([2.1, 4.0, 6.2, 8.0, 10.1]);

// Configure model
const model = new Lowess({ fraction: 0.5 });

// Fit data
const result = model.fit(x, y);

console.log("Smoothed Y:", result.y);
```
