---
title: Streaming Adapter
---
Process large datasets in chunks with configurable overlap.

## When to Use

- Dataset >100,000 points
- Memory-constrained environments
- Batch processing pipelines

## Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `chunk_size` | 5000 | Points per chunk |
| `overlap` | 500 | Overlap between chunks |
| `merge_strategy` | `"weighted_average"` | How to merge overlaps |

## Merge Strategies

| Strategy | Behavior |
| --- | --- |
| `"average"` | Average overlapping values |
| `"weighted_average"` | Distance-weighted blend |
| `"take_first"` | Keep left chunk values |
| `"take_last"` | Keep right chunk values |

![Merge Strategies](../assets/diagrams/merge_comparison.svg)

## Example

```javascript
const { StreamingLowess } = require('fastlowess-wasm');

const n = 100;
const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

const dataChunks = [
    { x: x.slice(0, 50), y: y.slice(0, 50) },
    { x: x.slice(50), y: y.slice(50) }
];

const processor = new StreamingLowess(
    { fraction: 0.3, iterations: 2 },
    { chunk_size: 5000, overlap: 500 }
);

// Process chunks
for (const {x, y} of dataChunks) {
    const result = processor.process_chunk(x, y);
    // ...
}

const finalResult = processor.finalize();
console.log("y[0]:", finalResult.y[0].toFixed(4));
```

```output
y[0]: 0.0279
```

---

:::caution[Always call finalize()]
In Rust, always call `processor.finalize()` after processing all chunks to retrieve buffered overlap data.
:::
