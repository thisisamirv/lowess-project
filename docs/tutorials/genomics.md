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
    model <- Lowess(
        fraction = 0.1,
        iterations = 3,
        confidence_intervals = 0.95
    )
    result <- model$fit(positions, observed)

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
    model = fl.Lowess(
        fraction=0.1,           # Small fraction for local detail
        iterations=3,           # Robustness for outliers
        confidence_intervals=0.95
    )
    result = model.fit(positions, observed)

    # Plot
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
    ```

=== "Rust"
    ```rust
    let positions = x.clone();
    let observed = y.clone();

    let model = Lowess::new()
        .fraction(0.1)
        .iterations(3)
        .confidence_intervals(0.95)
        .build()?;

    let result = model.fit(&positions, &observed)?;
    // result.y contains smoothed methylation profile
    // result.confidence_lower/upper contain 95% CI bounds
    ```

=== "Julia"
    ```julia
    using FastLOWESS

    # positions and observed are your methylation data
    model = Lowess(;
        fraction=0.1,
        iterations=3,
        confidence_intervals=0.95
    )
    result = fit(model, positions, observed)

    # Smoothed profile in result.y
    # CI bounds in result.confidence_lower/upper
    ```

=== "Node.js"
    ```javascript
    const fl = require('fastlowess');

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

=== "WebAssembly"
    ```javascript
    import { smooth } from 'fastlowess-wasm';

    // positions and observed are your methylation data (Float64Array)
    const result = smooth(positions, observed, {
        fraction: 0.1,
        iterations: 3,
        confidence_intervals: 0.95
    });

    // Smoothed profile in result.y
    // CI bounds in result.confidence_lower/upper
    ```

=== "C++"
    ```cpp
    #include "fastlowess.hpp"

    // positions and observed are std::vector<double>
    fastlowess::Lowess model({ .fraction = 0.1, .iterations = 3, .confidence_intervals = 0.95 });
    auto result = model.fit(positions, observed).value();

    // Smoothed profile in result.y_vector()
    // CI bounds in result.confidence_lower()/result.confidence_upper()
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

    model <- Lowess(
        fraction = 0.05,
        iterations = 5
    )
    result <- model$fit(positions, observed)

    # Find peaks
    threshold <- quantile(result$y, 0.75)
    peak_positions <- positions[result$y > threshold]
    ```

=== "Python"
    ```python
    # Simulate ChIP-seq coverage with peaks
    np.random.seed(123)
    positions = np.arange(0, 10000, 10, dtype=float)
    n = len(positions)

    # Background + peaks
    background = 10
    peak1 = 50 * np.exp(-((positions - 2000) ** 2) / (2 * 200 ** 2))
    peak2 = 80 * np.exp(-((positions - 5000) ** 2) / (2 * 300 ** 2))
    peak3 = 40 * np.exp(-((positions - 8000) ** 2) / (2 * 150 ** 2))
    
    true_signal = background + peak1 + peak2 + peak3
    observed = np.random.poisson(true_signal)  # Poisson noise

    # Smooth with robustness for sporadic high counts
    model = fl.Lowess(
        fraction=0.05,   # Very local smoothing
        iterations=5,    # Strong robustness
        return_residuals=True
    )
    result = model.fit(positions, observed.astype(float))

    # Identify peaks (smoothed signal significantly above background)
    threshold = np.percentile(result.y, 75)
    peaks = positions[result.y > threshold]
    print(f"Peak regions: {peaks}")
    ```

=== "Rust"
    ```rust
    let positions: Vec<f64> = (0..1000).map(|i| i as f64 *10.0).collect(); // 0 to 9990 step 10
    let observed: Vec<f64> = positions.iter().map(|&p| (p / 1000.0).sin().abs()* 100.0 + 10.0).collect();

    let model = Lowess::new()
        .fraction(0.05)
        .iterations(5)
        .return_residuals()
        .build()?;

    let result = model.fit(&positions, &observed)?;

    // Find peaks above threshold
    let threshold = result.y.iter().copied()
        .fold(f64::NEG_INFINITY, f64::max) * 0.75;
    let peak_positions: Vec<f64> = positions.iter().zip(result.y.iter())
        .filter(|(_, &y)| y > threshold)
        .map(|(&p, _)| p)
        .collect();
    ```

=== "Julia"
    ```julia
    using FastLOWESS

    # positions and observed are your ChIP-seq data
    model = Lowess(; fraction=0.05, iterations=5)
    result = fit(model, positions, observed)

    # Find peaks above 75th percentile
    threshold = quantile(result.y, 0.75)
    peak_indices = findall(y -> y > threshold, result.y)
    peak_positions = positions[peak_indices]
    ```

=== "Node.js"
    ```javascript
    const fl = require('fastlowess');

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

=== "WebAssembly"
    ```javascript
    import { smooth } from 'fastlowess-wasm';

    const result = smooth(positions, observed, {
        fraction: 0.05,
        iterations: 5
    });

    // Find peaks
    const smoothed = result.y;
    const peaks = positions.filter((p, i) => smoothed[i] > 25.0);
    ```

=== "C++"
    ```cpp
    #include "fastlowess.hpp"

    fastlowess::Lowess model({ .fraction = 0.05, .iterations = 5 });
    auto result = model.fit(positions, observed).value();

    // Find peaks above threshold
    std::vector<double> peaks;
    const auto& y_vals = result.y_vector();
    const auto& x_vals = result.x_vector();
    for (size_t i = 0; i < y_vals.size(); ++i) {
        if (y_vals[i] > 25.0) {
            peaks.push_back(x_vals[i]);
        }
    }
    ```

---

## Large Genome Coverage (Streaming)

For whole-genome data that doesn't fit in memory:

=== "R"
    ```r
    model <- StreamingLowess(
        fraction = 0.05,
        chunk_size = 100000,
        overlap = 10000,
        merge_strategy = "weighted_average"
    )
    result <- model$process_chunk(positions, coverage)
    final <- model$finalize()
    ```

=== "Python"
    ```python
    # Process chromosome-by-chromosome or in chunks
    model = fl.StreamingLowess(
        fraction=0.05,
        chunk_size=100000,    # 100kb chunks
        overlap=10000,        # 10kb overlap
        merge_strategy="weighted_average"
    )
    model.process_chunk(positions, coverage)
    result = model.finalize()
    ```

=== "Rust"
    ```rust
    let mut processor = StreamingLowess::new()
        .fraction(0.05)
        .iterations(3)
        .chunk_size(50)
        .overlap(10)
        .merge_strategy("weighted_average")
        .build()?;

    processor.process_chunk(&x_chunk, &y_chunk)?;
    let result = processor.finalize()?;
    ```

=== "Julia"
    ```julia
    using FastLOWESS

    # coverage and positions are chromosome-scale vectors
    model = StreamingLowess(;
        fraction=0.05,
        chunk_size=100000,
        overlap=10000,
        merge_strategy="weighted_average"
    )
    process_chunk(model, positions, coverage)
    result = finalize(model)
    ```

=== "Node.js"
    ```javascript
    const { StreamingLowess } = require('fastlowess');

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

=== "WebAssembly"
    ```javascript
    const processor = new StreamingLowess(
        { fraction: 0.05, iterations: 3 },
        { chunk_size: 100, overlap: 10 }
    );

    processor.process_chunk(xChunk, yChunk);
    const result = processor.finalize();
    ```

=== "C++"
    ```cpp
    #include "fastlowess.hpp"

    // coverage and positions are chromosome-scale vectors
    fastlowess::StreamingOptions s_opts;
    s_opts.fraction = 0.05;
    s_opts.iterations = 3;
    s_opts.chunk_size = 100000;
    s_opts.overlap = 10000;
    fastlowess::StreamingLowess stream(s_opts);
    (void)stream.process_chunk(positions, coverage);
    auto result = stream.finalize().value();
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
