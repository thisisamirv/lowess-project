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

### Merge Strategies

| Strategy | Behavior |
| --- | --- |
| `"average"` | Average overlapping values |
| `"weighted_average"` | Distance-weighted blend |
| `"take_first"` | Keep left chunk values |
| `"take_last"` | Keep right chunk values |

![Merge Strategies](../assets/diagrams/merge_comparison.svg)

## Example

```javascript
const { StreamingLowess } = require('fastlowess');

const n = 100;
const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i*7+3)%17)/17-0.5)*0.6);
const dataChunks = Array.from({ length: 5 }, (_, ci) => ({
    x: Float64Array.from({ length: 20 }, (_, i) => ci * 20 + i),
    y: Float64Array.from({ length: 20 }, (_, i) => Math.sin((ci * 20 + i) * 0.1))
}));

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
console.log("Smoothed", finalResult.y.length, "points via streaming");
```

```output
Smoothed 100 points via streaming
```

---

:::caution[Always call finalize()]
In Rust, always call `processor.finalize()` after processing all chunks to retrieve buffered overlap data.
:::
