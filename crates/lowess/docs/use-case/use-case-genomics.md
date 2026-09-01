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

```rust
use lowess::prelude::*;

fn main() -> Result<(), LowessError> {
    let n = 100usize;
    let positions: Vec<f64> = (0..n).map(|i| i as f64 * 1000.0).collect();
    let observed: Vec<f64> = positions.iter().map(|&p| 50.0 + (p / 1000.0).sin() * 20.0 + 5.0).collect();

    let model = Lowess::new()
        .fraction(0.1)
        .iterations(3)
        .confidence_intervals(0.95)
        .build()?;

    let result = model.fit(&positions, &observed)?;
    // result.y contains smoothed methylation profile
    // result.confidence_lower/upper contain 95% CI bounds

    if let (Some(lo), Some(hi)) = (&result.confidence_lower, &result.confidence_upper) {
        println!("95% CI: [{}, {}]", lo[0], hi[0]);
    }
    Ok(())
}
```

```output
95% CI: [51.67728847225472, 68.73718630709101]
```

---

## ChIP-seq Signal Smoothing

### Application

ChIP-seq experiments produce sparse, noisy coverage data. LOWESS can help identify binding regions.

`fraction = 0.05` provides high spatial resolution — important for resolving narrow binding peaks that would otherwise be smeared into the background. The larger `iterations = 5` is deliberate: Poisson-distributed read counts produce tall, isolated spikes, and extra robustness iterations progressively down-weight them so the estimated background level is not inflated by a handful of extreme counts.

```rust
use lowess::prelude::*;

fn main() -> Result<(), LowessError> {
    let n = 100usize;
    let positions: Vec<f64> = (0..n).map(|i| i as f64 * 1000.0).collect();
    let observed: Vec<f64> = positions.iter().map(|&p| 50.0 + (p / 1000.0).sin() * 20.0 + 5.0).collect();

    let model = Lowess::new()
        .fraction(0.05)
        .iterations(5)
        .build()?;

    let result = model.fit(&positions, &observed)?;

    // Find peaks above threshold
    let peak_count = result.y.iter().filter(|&&y| y > 65.0).count();

    println!("y[0]: {}", result.y[0]);
    println!("Peak count: {}", peak_count);
    Ok(())
}
```

```output
y[0]: 59.951990929979125
Peak count: 26
```

---

## Large Genome Coverage (Streaming)

For whole-genome data that doesn't fit in memory:

```rust
use lowess::prelude::*;

fn main() -> Result<(), LowessError> {
    let x_chunk: Vec<f64> = (0..1001).map(|i| i as f64 * 10.0).collect();
    let y_chunk: Vec<f64> = x_chunk.iter().map(|&p| 50.0 + (p / 100.0).sin() * 20.0 + 5.0).collect();

    let mut processor = StreamingLowess::new()
        .fraction(0.05)
        .iterations(3)
        .chunk_size(50)
        .overlap(10)
        .merge_strategy("weighted_average")
        .build()?;

    processor.process_chunk(&x_chunk, &y_chunk)?;
    let result = processor.finalize()?;
    println!("y[0]: {}", result.y[0]);

    Ok(())
}
```

```output
y[0]: 41.29765849569398
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

- [Concepts](crate::doc::introduction::concepts) — How LOWESS works
- [API Reference](crate::doc::api) — All options
- [Robustness](crate::doc::weighting::robustness) — Outlier downweighting in depth
- [Merge Strategies](crate::doc::advanced::merge) — Streaming chunk reconciliation
- [Boundary Handling](crate::doc::advanced::boundary) — Edge handling for sparse regions
- [Real-Time Processing](crate::doc::use_case::real_time) — For sequencing runs
