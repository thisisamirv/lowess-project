---
title: Genomic Data Smoothing
---
<!-- markdownlint-disable MD033 -->
LOWESS for methylation profiles, ChIP-seq signals, and other genomic data.

## Overview

Genomic data often contains noise from sequencing depth variation, PCR artifacts, or biological heterogeneity. LOWESS smoothing helps reveal underlying patterns.

---

## Methylation Profile Smoothing

### The Challenge

DNA methylation data (from bisulfite sequencing or arrays) shows position-dependent patterns that can be obscured by measurement noise.

### Solution

A small `fraction = 0.1` lets LOWESS follow fine-scale spatial structure without smearing the transitions between methylated and unmethylated regions. `confidence_intervals = 0.95` produces uncertainty bands that naturally widen at positions with sparser CpG coverage, making low-confidence segments immediately apparent in the plot.

```javascript
const fl = require('fastlowess');

const n = 100;
const positions = Float64Array.from({ length: n }, (_, i) => i * 1000.0);
const observed = Float64Array.from(positions, p => 50 + Math.sin(p / 1000) * 20 + 5);

// positions and observed are your methylation data (Float64Array)
const model = new fl.Lowess({
    fraction: 0.1,
    iterations: 3,
    confidence_intervals: 0.95
});
const result = model.fit(positions, observed);

// Smoothed profile in result.y
// CI bounds in result.confidence_lower/upper
console.log("95% CI: [" + result.confidence_lower[0].toFixed(4) + ", " + result.confidence_upper[0].toFixed(4) + "]");
```

```output
95% CI: [51.6773, 68.7372]
```

---

## ChIP-seq Signal Smoothing

### Application

ChIP-seq experiments produce sparse, noisy coverage data. LOWESS can help identify binding regions.

`fraction = 0.05` provides high spatial resolution — important for resolving narrow binding peaks that would otherwise be smeared into the background. The larger `iterations = 5` is deliberate: Poisson-distributed read counts produce tall, isolated spikes, and extra robustness iterations progressively down-weight them so the estimated background level is not inflated by a handful of extreme counts.

```javascript
const fl = require('fastlowess');

const n = 100;
const positions = Float64Array.from({ length: n }, (_, i) => i * 1000.0);
const observed = Float64Array.from(positions, p => 50 + Math.sin(p / 1000) * 20 + 5);

const model = new fl.Lowess({
    fraction: 0.05,
    iterations: 5
});
const result = model.fit(positions, observed);

// Identify peaks above threshold
let peakCount = 0;
for (const y of result.y) if (y > 65.0) peakCount++;
console.log("y[0]:", result.y[0].toFixed(4));
console.log("Peak count:", peakCount);
```

```output
y[0]: 59.9520
Peak count: 26
```

---

## Large Genome Coverage (Streaming)

For whole-genome data that doesn't fit in memory:

```javascript
const { StreamingLowess } = require('fastlowess');

const n = 1001;
const xChunk = Float64Array.from({ length: n }, (_, i) => i * 10.0);
const yChunk = Float64Array.from(xChunk, p => 50 + Math.sin(p / 100) * 20 + 5.0);

const processor = new StreamingLowess(
    { fraction: 0.05, iterations: 3 },
    { chunk_size: 50, overlap: 10, merge_strategy: "weighted_average" }
);

processor.process_chunk(xChunk, yChunk);
const result = processor.finalize();
console.log("y[0]:", result.y[0].toFixed(4));
```

```output
y[0]: 41.2977
```

---

## Best Practices for Genomic Data

| Consideration | Recommendation |
| --- | --- |
| **Fraction** | 0.05–0.15 (preserve local features) |
| **Iterations** | 3–5 (handle sequencing outliers) |
| **Large data** | Use streaming mode |
| **Sparse regions** | Use `boundary_policy="extend"` |
| **Multiple chromosomes** | Process separately or ensure sorted |

---

## See Also

- [Concepts](../introduction/concepts.md) — How LOWESS works
- [API Reference](../api/api.md) — All options
- [Robustness](../weighting/robustness.md) — Outlier downweighting in depth
- [Merge Strategies](../advanced/merge.md) — Streaming chunk reconciliation
- [Boundary Handling](../advanced/boundary.md) — Edge handling for sparse regions
- [Real-Time Processing](use-case-real-time.md) — For sequencing runs
