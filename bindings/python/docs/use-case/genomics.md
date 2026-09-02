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

:::{jupyter-execute}
import fastlowess as fl
import numpy as np
import matplotlib.pyplot as plt

## Deterministic methylation profile along a chromosome

positions = np.arange(0, 100000, 1000, dtype=float)
observed = 50.0 + np.sin(positions / 1000.0) * 20.0 + 5.0
observed = observed / 100.0  # Methylation is 0-1

## Smooth with LOWESS

model = fl.Lowess(
    fraction=0.1,           # Small fraction for local detail
    iterations=3,           # Robustness for outliers
    confidence_intervals=0.95
)
result = model.fit(positions, observed)

## Plot

plt.figure(figsize=(12, 5))
plt.scatter(positions, observed, s=2, alpha=0.3, label="Observed")
plt.plot(positions, result.y, "b-", linewidth=2, label="LOWESS smoothed")
plt.fill_between(
    positions,
    result.confidence_lower,
    result.confidence_upper,
    alpha=0.2, label="95% CI"
)
plt.xlabel("Genomic Position (bp)")
plt.ylabel("Methylation Level")
plt.legend()
plt.title("Methylation Profile Smoothing")
plt.show()

print(f"95% CI: [{result.confidence_lower[0]:.4f}, {result.confidence_upper[0]:.4f}]")
:::

---

## ChIP-seq Signal Smoothing

### Application

ChIP-seq experiments produce sparse, noisy coverage data. LOWESS can help identify binding regions.

`fraction = 0.05` provides high spatial resolution — important for resolving narrow binding peaks that would otherwise be smeared into the background. The larger `iterations = 5` is deliberate: Poisson-distributed read counts produce tall, isolated spikes, and extra robustness iterations progressively down-weight them so the estimated background level is not inflated by a handful of extreme counts.

:::{jupyter-execute}
import fastlowess as fl
import numpy as np

positions = np.arange(0, 100000, 1000, dtype=float)
observed = 50.0 + np.sin(positions / 1000.0) * 20.0 + 5.0

## Smooth with robustness for sporadic high counts

model = fl.Lowess(
    fraction=0.05,   # Very local smoothing
    iterations=5,    # Strong robustness
)
result = model.fit(positions, observed)

## Identify peaks (smoothed signal above threshold)

peak_count = int(np.sum(result.y > 65.0))
print(f"y[0]: {result.y[0]:.4f}")
print(f"Peak count: {peak_count}")
:::

---

## Large Genome Coverage (Streaming)

For whole-genome data that doesn't fit in memory:

:::{jupyter-execute}
import fastlowess as fl
import numpy as np

x_chunk = np.arange(0, 10010, 10, dtype=float)
y_chunk = 50.0 + np.sin(x_chunk / 100.0) * 20.0 + 5.0

## Process chromosome-by-chromosome or in chunks

model = fl.StreamingLowess(
    fraction=0.05,
    iterations=3,
    chunk_size=50,
    overlap=10,
    merge_strategy="weighted_average"
)
model.process_chunk(x_chunk, y_chunk)
result = model.finalize()
print(f"y[0]: {result.y[0]:.4f}")
:::

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
- [Real-Time Processing](real-time.md) — For sequencing runs
