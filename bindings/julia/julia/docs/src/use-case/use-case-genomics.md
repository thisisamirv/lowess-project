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

```@example use-case-genomics
using FastLOWESS

positions = collect(0.0:1000.0:99000.0)
observed = 50.0 .+ sin.(positions ./ 1000.0) .* 20.0 .+ 5.0

# positions and observed are your methylation data
model = Lowess(;
    fraction=0.1,
    iterations=3,
    confidence_intervals=0.95
)
result = fit(model, positions, observed)

# Smoothed profile in result.y
# CI bounds in result.confidence_lower/upper
println("95% CI: [", result.confidence_lower[1], ", ", result.confidence_upper[1], "]")
```

---

## ChIP-seq Signal Smoothing

### Application

ChIP-seq experiments produce sparse, noisy coverage data. LOWESS can help identify binding regions.

`fraction = 0.05` provides high spatial resolution — important for resolving narrow binding peaks that would otherwise be smeared into the background. The larger `iterations = 5` is deliberate: Poisson-distributed read counts produce tall, isolated spikes, and extra robustness iterations progressively down-weight them so the estimated background level is not inflated by a handful of extreme counts.

```@example use-case-genomics
using FastLOWESS

positions = collect(0.0:1000.0:99000.0)
observed = 50.0 .+ sin.(positions ./ 1000.0) .* 20.0 .+ 5.0

# positions and observed are your ChIP-seq data
model = Lowess(; fraction=0.05, iterations=5)
result = fit(model, positions, observed)

# Find peaks above threshold
peak_count = count(y -> y > 65.0, result.y)
println("y[0]: ", result.y[1])
println("Peak count: ", peak_count)
```

---

## Large Genome Coverage (Streaming)

For whole-genome data that doesn't fit in memory:

```@example use-case-genomics
using FastLOWESS

positions = collect(0.0:10.0:10000.0)
coverage = 50.0 .+ sin.(positions ./ 100.0) .* 20.0 .+ 5.0

# coverage and positions are chromosome-scale vectors
model = StreamingLowess(;
    fraction=0.05,
    iterations=3,
    chunk_size=50,
    overlap=10,
    merge_strategy="weighted_average"
)
process_chunk(model, positions, coverage)
result = finalize(model)
println("y[0]: ", result.y[1])
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
- [API Reference](../api.md) — All options
- [Robustness](../weighting/robustness.md) — Outlier downweighting in depth
- [Merge Strategies](../advanced/merge.md) — Streaming chunk reconciliation
- [Boundary Handling](../advanced/boundary.md) — Edge handling for sparse regions
- [Real-Time Processing](use-case-real-time.md) — For sequencing runs
