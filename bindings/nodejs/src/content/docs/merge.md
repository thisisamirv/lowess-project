---
title: Merge Strategies
---
<!-- markdownlint-disable MD024 MD033 MD046 -->
How overlapping chunk boundaries are reconciled in Streaming mode.

## Overview

Streaming LOWESS processes data in fixed-size chunks with a configurable overlap. Points inside the overlap zone are fitted twice — once by the left chunk and once by the right chunk. The `merge_strategy` decides how those two estimates are combined into a single output value.

```text
Chunk A:   [=========|=====]
Chunk B:            [=====|=========]
Overlap:            [=====]
                      ↑
                 merge_strategy
                 applied here
```

| Strategy | Method | Robustness | Speed |
| --- | --- | --- | --- |
| `"average"` | Simple mean of both estimates | Low | Fastest |
| `"take_first"` | Left-chunk estimate only | Low | Fastest |
| `"take_last"` | Right-chunk estimate only | Low | Fastest |
| `"weighted_average"` | Distance-weighted mean | High | Moderate |

![Merge Strategies](../assets/diagrams/merge_comparison.svg)

---

## Average

Takes the arithmetic mean of the left-chunk and right-chunk estimates in the overlap region. Fast and sufficient when both chunks have similar smoothing quality.

**Use when**: Chunks are large and the overlap region has uniform data density.

```javascript
const { StreamingLowess } = require('fastlowess');

const n = 50;
const xChunk = Float64Array.from({ length: n }, (_, i) => i * Math.PI / (n - 1));
const yChunk = Float64Array.from(xChunk, xi => Math.sin(xi));

const processor = new StreamingLowess(
    {},
    { merge_strategy: "average", chunk_size: 5000, overlap: 500 }
);
processor.process_chunk(xChunk, yChunk);
const finalResult = processor.finalize();
console.log("average: smoothed", finalResult.y.length, "points, y[0]:", finalResult.y[0].toFixed(4));
```

```output
average: smoothed 50 points, y[0]: 0.1795
```

---

## Take First

Keeps only the left-chunk estimate in the overlap zone and discards the right-chunk estimate. Produces a definitive, non-revised output as soon as the right boundary of each chunk is reached.

**Use when**: You need final output values immediately after each chunk (no look-ahead revision); left-chunk data quality is higher.

```javascript
const { StreamingLowess } = require('fastlowess');

const processor = new StreamingLowess(
    {},
    { merge_strategy: "take_first" }
);
const x2 = Float64Array.from({ length: 6 }, (_, i) => i);
const y2 = Float64Array.from({ length: 6 }, (_, i) => i * 0.5);
processor.process_chunk(x2, y2);
const r2 = processor.finalize();
console.log("take_first: smoothed", r2.y.length, "points, y[0]:", r2.y[0].toFixed(4));
```

```output
take_first: smoothed 6 points, y[0]: 0.2337
```

---

## Take Last

Keeps only the right-chunk estimate in the overlap zone. The right chunk sees more of the surrounding data, so its fit can be more accurate near the left boundary of the new chunk.

**Use when**: Right-chunk context improves overlap quality; you are post-processing complete data rather than streaming live.

```javascript
const { StreamingLowess } = require('fastlowess');

const processor = new StreamingLowess(
    {},
    { merge_strategy: "take_last" }
);
const x3 = Float64Array.from({ length: 6 }, (_, i) => i);
const y3 = Float64Array.from({ length: 6 }, (_, i) => i * 0.5);
processor.process_chunk(x3, y3);
const r3 = processor.finalize();
console.log("take_last: smoothed", r3.y.length, "points, y[0]:", r3.y[0].toFixed(4));
```

```output
take_last: smoothed 6 points, y[0]: 0.2337
```

---

## Weighted Average

Assigns each overlap point a weight proportional to its proximity to the centre of its respective chunk: points near the left-chunk centre get higher left weight; points near the right-chunk centre get higher right weight. This produces the smoothest transition across chunk boundaries.

$$\hat{y} = \frac{w_L \hat{y}_L + w_R \hat{y}_R}{w_L + w_R}$$

where $w_L$ and $w_R$ are linear distance weights from the chunk centres.

**Use when**: Minimising boundary artefacts is more important than speed; moderate overlap (10–20 % of chunk size).

```javascript
const { StreamingLowess } = require('fastlowess');

const n = 100;
const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

const processor = new StreamingLowess(
    {},
    { merge_strategy: "weighted_average", chunk_size: 5000, overlap: 500 }
);
processor.process_chunk(x, y);
const finalResult = processor.finalize();
console.log("weighted_average: smoothed", finalResult.y.length, "points, y[0]:", finalResult.y[0].toFixed(4));
```

```output
weighted_average: smoothed 100 points, y[0]: 0.1662
```

---

## Choosing a Strategy

| Situation | Recommended Strategy |
| --- | --- |
| General purpose | `"weighted_average"` |
| Maximum throughput | `"average"` |
| Immediate finalised output | `"take_first"` |
| Post-processing, right context better | `"take_last"` |
| Minimising boundary artefacts | `"weighted_average"` |

:::tip[Overlap size matters]
A larger overlap gives the merge strategy more room to blend, reducing boundary artefacts regardless of the strategy chosen. A good starting point is 10 % of `chunk_size`.
:::
