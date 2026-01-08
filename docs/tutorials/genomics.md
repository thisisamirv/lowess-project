# Genomic Data Smoothing

LOWESS for methylation profiles, ChIP-seq signals, and other genomic data.

## Overview

Genomic data often contains noise from sequencing depth variation, PCR artifacts, or biological heterogeneity. LOWESS smoothing helps reveal underlying patterns.

---

## Methylation Profile Smoothing

### The Challenge

DNA methylation data (from bisulfite sequencing or arrays) shows position-dependent patterns that can be obscured by measurement noise.

### Solution

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

---

## ChIP-seq Signal Smoothing

### Application

ChIP-seq experiments produce sparse, noisy coverage data. LOWESS can help identify binding regions.

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

---

## Large Genome Coverage (Streaming)

For whole-genome data that doesn't fit in memory:

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
