# StreamingLowess — Python API Reference

See also: [fastLowess Python API Reference](python.md)

## Class

### `StreamingLowess`

The `StreamingLowess` class processes data in chunks, suitable for very large datasets or streaming applications.

**Constructor:**

```python
import fastlowess as fl

stream = fl.StreamingLowess(chunk_size=50, overlap=10)
```

* `kwargs`: Keyword arguments corresponding to `LowessOptions` and `StreamingOptions` fields.

**Methods:**

```python
import fastlowess as fl
import numpy as np

rng = np.random.default_rng(42)
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x) + rng.normal(0, 0.3, 100)

stream = fl.StreamingLowess(chunk_size=50, overlap=10)
partial_result = stream.process_chunk(x[:50], y[:50])
```

* Processes a chunk of data. Returns partial results.

```python
import fastlowess as fl
import numpy as np

rng = np.random.default_rng(42)
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x) + rng.normal(0, 0.3, 100)

stream = fl.StreamingLowess(chunk_size=50, overlap=10)
stream.process_chunk(x, y)
final_result = stream.finalize()
```

* Finalizes the smoothing process and returns any remaining buffered results.

## Options Structure

### `StreamingOptions` (inherits `LowessOptions`)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `chunk_size` | `int` | `5000` | Data chunk size |
| `overlap` | `int` | `500` | Overlap between chunks |
| `merge_strategy` | `str` | `"weighted_average"` | Strategy for blending overlap regions |

## Options

### merge_strategy

*See: [Merge Strategies](../user-guide/merge.md)*

* `"weighted_average"` (default; alias: `"weighted"`)
* `"average"` (alias: `"mean"`)
* `"take_first"` (alias: `"first"`)
* `"take_last"` (alias: `"last"`)
