---
title: StreamingLowess API
---
See also: [fastLowess](api.md)

## When to Use

- Dataset >100,000 points
- Memory-constrained environments
- Batch processing pipelines

## Class

### `StreamingLowess`

The `StreamingLowess` class processes data in chunks, suitable for very large datasets or streaming applications.

**Constructor:**

```javascript
const { StreamingLowess } = require('fastlowess');

const stream = new StreamingLowess({ fraction: 0.5 }, { chunk_size: 50, overlap: 10 });
const x = Float64Array.from({ length: 10 }, (_, i) => i);
const y = Float64Array.from({ length: 10 }, (_, i) => i * 0.5);
stream.process_chunk(x, y);
const result = stream.finalize();
console.log("Smoothed", result.y.length, "points via streaming");
```

```output
Smoothed 10 points via streaming
```

- `options`: An object containing `LowessOptions` fields.
- `streamingOptions`: An object containing `StreamingOptions` fields.

**Methods:**

```javascript
const { StreamingLowess } = require('fastlowess');

const n = 100;
const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
const y = Float64Array.from(x, xi => Math.sin(xi) + 0.1);

const stream = new StreamingLowess({ fraction: 0.5 }, { chunk_size: 50, overlap: 10 });
const partialResult = stream.process_chunk(x.slice(0, 50), y.slice(0, 50));
console.log("Fraction used:", partialResult.fraction_used);
```

```output
Fraction used: 0.5
```

- Processes a chunk of data. Returns partial results.

```javascript
const { StreamingLowess } = require('fastlowess');

const n = 100;
const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
const y = Float64Array.from(x, xi => Math.sin(xi) + 0.1);

const stream = new StreamingLowess({ fraction: 0.5 }, { chunk_size: 50, overlap: 10 });
stream.process_chunk(x.slice(0, 50), y.slice(0, 50));
stream.process_chunk(x.slice(50), y.slice(50));
const finalResult = stream.finalize();
console.log("Fraction used:", finalResult.fraction_used);
```

```output
Fraction used: 0.5
```

- Finalizes the smoothing process and returns any remaining buffered results.

## Result Structure

### `LowessResult`

Returned by `process_chunk()` and `finalize()`.

| Field | Type | Description |
| --- | --- | --- |
| `x` | `Float64Array` | Sorted x values |
| `y` | `Float64Array` | Smoothed y values |
| `fraction_used` | `number` | Fraction used |
| `iterations_used` | `number \| null` | Robustness iterations actually performed |
| `residuals` | `Float64Array \| null` | Residuals (if `return_residuals`) |
| `robustness_weights` | `Float64Array \| null` | Robustness weights (if `return_robustness_weights`) |
| `diagnostics` | `Diagnostics \| null` | Fit metrics (if `return_diagnostics`) |
| `dimensions` | `number` | Number of predictor dimensions |

See [nodejs.md](api.md) for the full `LowessResult` field reference.

## Options Structure

### `StreamingOptions` (inherits `LowessOptions`)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `chunk_size` | `number` | `5000` | Data chunk size |
| `overlap` | `number` | `500` | Overlap between chunks |
| `merge_strategy` | `string` | `"weighted_average"` | Strategy for blending overlap regions |

## Options

### merge_strategy

*See: [Merge Strategies](merge.md)*

| Strategy | Alias | Behavior |
| --- | --- | --- |
| `"weighted_average"` (default) | `"weighted"` | Distance-weighted blend |
| `"average"` | `"mean"` | Average overlapping values |
| `"take_first"` | `"first"` | Keep left chunk values |
| `"take_last"` | `"last"` | Keep right chunk values |

![Merge Strategies](../assets/diagrams/merge_comparison.svg)

---

:::caution[Always call finalize()]
The streaming adapter buffers overlap data. Call `finalize()` after the last chunk to retrieve the buffered tail.
:::
