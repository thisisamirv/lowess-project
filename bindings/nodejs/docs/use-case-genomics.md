<!-- markdownlint-disable MD033 -->
# Genomic Data Smoothing

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
const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i*7+3)%17)/17-0.5)*0.6);
const positions = Float64Array.from({ length: 1000 }, (_, i) => i * 10.0);
const observed = Float64Array.from(positions, p => 50 + Math.sin(p/100)*20 + Math.random()*5);

// positions and observed are your methylation data (Float64Array)
const model = new fl.Lowess({
    fraction: 0.1,
    iterations: 3,
    confidence_intervals: 0.95
});
const result = model.fit(positions, observed);

// Smoothed profile in result.y
// CI bounds in result.confidence_lower/upper
```

---

## ChIP-seq Signal Smoothing

### Application

ChIP-seq experiments produce sparse, noisy coverage data. LOWESS can help identify binding regions.

`fraction = 0.05` provides high spatial resolution — important for resolving narrow binding peaks that would otherwise be smeared into the background. The larger `iterations = 5` is deliberate: Poisson-distributed read counts produce tall, isolated spikes, and extra robustness iterations progressively down-weight them so the estimated background level is not inflated by a handful of extreme counts.

```javascript
const fl = require('fastlowess');

const n = 100;
const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i*7+3)%17)/17-0.5)*0.6);
const positions = Float64Array.from({ length: 1000 }, (_, i) => i * 10.0);
const observed = Float64Array.from(positions, p => 50 + Math.sin(p/100)*20 + Math.random()*5);

const model = new fl.Lowess({
    fraction: 0.05,
    iterations: 5
});
const result = model.fit(positions, observed);

// Identify peaks above threshold
const smoothed = result.y;
const threshold = 50.0; // Example threshold
const peaks = positions.filter((p, i) => smoothed[i] > threshold);
```

---

## Large Genome Coverage (Streaming)

For whole-genome data that doesn't fit in memory:

```javascript
const { StreamingLowess } = require('fastlowess');

const positions = Float64Array.from({ length: 1000 }, (_, i) => i * 10.0);
const observed = Float64Array.from(positions, p => 50 + Math.sin(p/100)*20 + Math.random()*5);
// Array of genomic chunks to process
const genomicData = [
    { positions: positions.slice(0, 500), coverage: observed.slice(0, 500) },
    { positions: positions.slice(500), coverage: observed.slice(500) }
];

const processor = new StreamingLowess(
    { fraction: 0.05, iterations: 3 },
    { chunk_size: 100000, overlap: 10000 }
);

// Process genomic chunks from stream or file
for (const chunk of genomicData) {
    processor.process_chunk(chunk.positions, chunk.coverage);
}
const result = processor.finalize();
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

- [Concepts](../getting-started/concepts.md) — How LOWESS works
- [Parameters](../user-guide/parameters.md) — All options
- [Robustness](../user-guide/robustness.md) — Outlier downweighting in depth
- [Merge Strategies](../user-guide/merge.md) — Streaming chunk reconciliation
- [Boundary Handling](../user-guide/boundary.md) — Edge handling for sparse regions
- [Real-Time Processing](real-time.md) — For sequencing runs
