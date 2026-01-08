# Genomic Data Smoothing

LOWESS for methylation profiles, ChIP-seq signals, and other genomic data.

## Overview

Genomic data often contains noise from sequencing depth variation, PCR artifacts, or biological heterogeneity. LOWESS smoothing helps reveal underlying patterns.

---

## Methylation Profile Smoothing

### The Challenge

DNA methylation data (from bisulfite sequencing or arrays) shows position-dependent patterns that can be obscured by measurement noise.

### Solution

=== "R"
    ```r
    library(rfastlowess)

    # Simulate methylation data
    set.seed(42)
    n <- 1000
    positions <- sort(runif(n, 0, 1e6))
    
    # True pattern
    true_meth <- 0.5 + 0.3 * sin(positions / 1e5)
    
    # Observed with noise
    observed <- true_meth + rnorm(n, sd = 0.15)
    observed <- pmax(0, pmin(1, observed))

    # Smooth
    result <- fastlowess(
        positions, observed,
        fraction = 0.1,
        iterations = 3,
        confidence_intervals = 0.95
    )

    # Plot
    plot(positions, observed, pch = ".", col = "gray",
         xlab = "Genomic Position (bp)", ylab = "Methylation Level",
         main = "Methylation Profile Smoothing")
    lines(result$x, result$y, col = "blue", lwd = 2)
    lines(result$x, result$confidence_lower, col = "blue", lty = 2)
    lines(result$x, result$confidence_upper, col = "blue", lty = 2)
    ```

=== "Python"
    ```python
    import fastlowess as fl
    import numpy as np
    import matplotlib.pyplot as plt

    # Simulate methylation data along a chromosome
    np.random.seed(42)
    n_positions = 1000
    positions = np.sort(np.random.uniform(0, 1e6, n_positions))
    
    # True methylation pattern (varies along chromosome)
    true_methylation = 0.5 + 0.3 * np.sin(positions / 1e5)
    
    # Observed with noise
    observed = true_methylation + np.random.normal(0, 0.15, n_positions)
    observed = np.clip(observed, 0, 1)  # Methylation is 0-1

    # Smooth with LOWESS
    result = fl.smooth(
        positions, observed,
        fraction=0.1,           # Small fraction for local detail
        iterations=3,           # Robustness for outliers
        confidence_intervals=0.95
    )

    # Plot
    plt.figure(figsize=(12, 5))
    plt.scatter(positions, observed, s=2, alpha=0.3, label="Observed")
    plt.plot(positions, result["y"], "b-", lwd=2, label="LOWESS smoothed")
    plt.fill_between(
        positions,
        result["confidence_lower"],
        result["confidence_upper"],
        alpha=0.2, label="95% CI"
    )
    plt.xlabel("Genomic Position (bp)")
    plt.ylabel("Methylation Level")
    plt.legend()
    plt.title("Methylation Profile Smoothing")
    plt.show()
    ```

=== "Rust"
    ```rust
    use fastLowess::prelude::*;
    use ndarray::Array1;

    let positions: Array1<f64> = /* sorted genomic positions */;
    let observed: Array1<f64> = /* methylation levels 0-1 */;

    let model = Lowess::new()
        .fraction(0.1)
        .iterations(3)
        .confidence_intervals(0.95)
        .adapter(Batch)
        .build()?;

    let result = model.fit(&positions, &observed)?;
    // result.y contains smoothed methylation profile
    // result.confidence_lower/upper contain 95% CI bounds
    ```

---

## ChIP-seq Signal Smoothing

### Application

ChIP-seq experiments produce sparse, noisy coverage data. LOWESS can help identify binding regions.

=== "R"
    ```r
    set.seed(123)
    positions <- seq(0, 10000, by = 10)
    n <- length(positions)

    # Simulate peaks
    background <- 10
    peak1 <- 50 * exp(-((positions - 2000)^2) / (2 * 200^2))
    peak2 <- 80 * exp(-((positions - 5000)^2) / (2 * 300^2))
    peak3 <- 40 * exp(-((positions - 8000)^2) / (2 * 150^2))
    
    true_signal <- background + peak1 + peak2 + peak3
    observed <- rpois(n, true_signal)

    result <- fastlowess(
        positions, observed,
        fraction = 0.05,
        iterations = 5
    )

    # Find peaks
    threshold <- quantile(result$y, 0.75)
    peak_positions <- positions[result$y > threshold]
    ```

=== "Python"
    ```python
    # Simulate ChIP-seq coverage with peaks
    np.random.seed(123)
    positions = np.arange(0, 10000, 10)
    n = len(positions)

    # Background + peaks
    background = 10
    peak1 = 50 * np.exp(-((positions - 2000) ** 2) / (2 * 200 ** 2))
    peak2 = 80 * np.exp(-((positions - 5000) ** 2) / (2 * 300 ** 2))
    peak3 = 40 * np.exp(-((positions - 8000) ** 2) / (2 * 150 ** 2))
    
    true_signal = background + peak1 + peak2 + peak3
    observed = np.random.poisson(true_signal)  # Poisson noise

    # Smooth with robustness for sporadic high counts
    result = fl.smooth(
        positions, observed.astype(float),
        fraction=0.05,   # Very local smoothing
        iterations=5,    # Strong robustness
        return_residuals=True
    )

    # Identify peaks (smoothed signal significantly above background)
    threshold = np.percentile(result["y"], 75)
    peaks = positions[result["y"] > threshold]
    print(f"Peak regions: {peaks}")
    ```

=== "Rust"
    ```rust
    let positions: Array1<f64> = Array1::range(0.0, 10000.0, 10.0);
    let observed: Array1<f64> = /*ChIP-seq counts*/;

    let model = Lowess::new()
        .fraction(0.05)
        .iterations(5)
        .return_residuals()
        .adapter(Batch)
        .build()?;

    let result = model.fit(&positions, &observed)?;

    // Find peaks above threshold
    let threshold = /* compute 75th percentile */;
    let peak_positions: Vec<f64> = positions
        .iter()
        .zip(result.y.iter())
        .filter(|(_, &y)| y > threshold)
        .map(|(&p, _)| p)
        .collect();
    ```

---

## Large Genome Coverage (Streaming)

For whole-genome data that doesn't fit in memory:

=== "R"
    ```r
    result <- fastlowess_streaming(
        positions, coverage,
        fraction = 0.05,
        chunk_size = 100000,
        overlap = 10000,
        merge_strategy = "weighted"
    )
    ```

=== "Python"
    ```python
    # Process chromosome-by-chromosome or in chunks
    result = fl.smooth_streaming(
        positions, coverage,
        fraction=0.05,
        chunk_size=100000,    # 100kb chunks
        overlap=10000,        # 10kb overlap
        merge_strategy="weighted"
    )
    ```

=== "Rust"
    ```rust
    use fastLowess::prelude::*;

    let model = Lowess::new()
        .fraction(0.05)
        .iterations(3)
        .adapter(Streaming {
            chunk_size: 100_000,
            overlap: 10_000,
            merge_strategy: Weighted,
        })
        .build()?;

    // Process chunks from file or stream
    let mut processor = model.processor();
    for chunk in chromosome_chunks {
        processor.process_chunk(&chunk.positions, &chunk.coverage)?;
    }
    let result = processor.finalize()?;
    ```

---

## Best Practices for Genomic Data

| Consideration            | Recommendation                      |
|--------------------------|-------------------------------------|
| **Fraction**             | 0.05–0.15 (preserve local features) |
| **Iterations**           | 3–5 (handle sequencing outliers)    |
| **Large data**           | Use streaming mode                  |
| **Sparse regions**       | Use `boundary_policy="extend"`      |
| **Multiple chromosomes** | Process separately or ensure sorted |

---

## See Also

- [Concepts](../getting-started/concepts.md) — How LOWESS works
- [Parameters](../user-guide/parameters.md) — All options
- [Real-Time Processing](real-time.md) — For sequencing runs
