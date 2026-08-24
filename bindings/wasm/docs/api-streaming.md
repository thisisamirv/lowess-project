# StreamingLowess — WebAssembly API Reference

See also: [fastLowess WebAssembly API Reference](wasm.md)

## Class

### `StreamingLowess`

The `StreamingLowess` class processes data in chunks, suitable for very large datasets or streaming applications.

**Constructor:**

```javascript
const { StreamingLowess } = require('fastlowess-wasm');

const stream = new StreamingLowess({ fraction: 0.5 }, { chunk_size: 50, overlap: 10 });
console.log("typeof process_chunk:", typeof stream.process_chunk);
```

```output
typeof process_chunk: function
```

* `options`: An object containing `LowessOptions` fields.
* `streamingOptions`: An object containing `StreamingOptions` fields.

**Methods:**

```javascript
const { StreamingLowess } = require('fastlowess-wasm');

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

* Processes a chunk of data. Returns partial results.

```javascript
const { StreamingLowess } = require('fastlowess-wasm');

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

* Finalizes the smoothing process and returns any remaining buffered results.

## Result Structure

### `LowessResult`

Returned by `process_chunk()` and `finalize()`.

| Field | Type | Description |
| --- | --- | --- |
| `x` | `Float64Array` | Sorted x values |
| `y` | `Float64Array` | Smoothed y values |
| `fraction_used` | `number` | Fraction used |
| `iterations_used` | `number \| undefined` | Robustness iterations actually performed |
| `residuals` | `Float64Array \| undefined` | Residuals (if `return_residuals`) |
| `robustness_weights` | `Float64Array \| undefined` | Robustness weights (if `return_robustness_weights`) |
| `diagnostics` | `Diagnostics \| undefined` | Fit metrics (if `return_diagnostics`) |
| `dimensions` | `number` | Number of predictor dimensions |

See [wasm.md](wasm.md) for the full `LowessResult` field reference.

## Options Structure

### `StreamingOptions` (inherits `LowessOptions`)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `chunk_size` | `number` | `5000` | Data chunk size |
| `overlap` | `number` | `500` | Overlap between chunks |
| `merge_strategy` | `string` | `"weighted_average"` | Strategy for blending overlap regions |

## Options

### merge_strategy

*See: [Merge Strategies](../user-guide/merge.md)*

* `"weighted_average"` (default; alias: `"weighted"`)
* `"average"` (alias: `"mean"`)
* `"take_first"` (alias: `"first"`)
* `"take_last"` (alias: `"last"`)
