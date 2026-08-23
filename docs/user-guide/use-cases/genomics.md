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
=== "C++"
    ```cpp
    #include <fastlowess.hpp>
    #include <cmath>
    #include <iostream>
    #include <vector>

    int main() {
        const int n = 100;
        std::vector<double> positions(n), observed(n);
        for (int i = 0; i < n; ++i) {
            positions[i] = i * 1000.0;
            observed[i] = 50.0 + std::sin(positions[i] / 1000.0) * 20.0 + 5.0;
        }

        // positions and observed are std::vector<double>
        fastlowess::Lowess model({ .fraction = 0.1, .iterations = 3, .confidence_intervals = 0.95 });
        auto result = model.fit(positions, observed).value();

        // Smoothed profile in result.y_vector()
        // CI bounds in result.confidence_lower()/result.confidence_upper()

        return 0;
    }
    ```

---

## ChIP-seq Signal Smoothing

### Application

ChIP-seq experiments produce sparse, noisy coverage data. LOWESS can help identify binding regions.

`fraction = 0.05` provides high spatial resolution — important for resolving narrow binding peaks that would otherwise be smeared into the background. The larger `iterations = 5` is deliberate: Poisson-distributed read counts produce tall, isolated spikes, and extra robustness iterations progressively down-weight them so the estimated background level is not inflated by a handful of extreme counts.

=== "Python"
    ```python
    import fastlowess as fl
    import numpy as np

    np.random.seed(42)
    positions = np.arange(0, 10000, 10, dtype=float)
    coverage = np.random.poisson(50, len(positions)).astype(float)

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
=== "C++"
    ```cpp
    #include <fastlowess.hpp>
    #include <cmath>
    #include <iostream>
    #include <vector>

    int main() {
        const int n = 100;
        std::vector<double> positions(n), observed(n);
        for (int i = 0; i < n; ++i) {
            positions[i] = i * 1000.0;
            observed[i] = 50.0 + std::sin(positions[i] / 1000.0) * 20.0 + 5.0;
        }

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

        return 0;
    }
    ```

---

## Large Genome Coverage (Streaming)

For whole-genome data that doesn't fit in memory:

=== "Python"
    ```python
    import fastlowess as fl
    import numpy as np

    np.random.seed(42)
    positions = np.arange(0, 10000, 10, dtype=float)
    coverage = np.random.poisson(50, len(positions)).astype(float)

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
=== "C++"
    ```cpp
    #include <fastlowess.hpp>
    #include <cmath>
    #include <iostream>
    #include <vector>

    int main() {
        const int n = 100;
        std::vector<double> positions(n), coverage(n);
        for (int i = 0; i < n; ++i) {
            positions[i] = i * 1000.0;
            coverage[i] = 50.0 + std::sin(positions[i] / 1000.0) * 20.0 + 5.0;
        }

        // coverage and positions are chromosome-scale vectors
        fastlowess::StreamingOptions s_opts;
        s_opts.fraction = 0.05;
        s_opts.iterations = 3;
        s_opts.chunk_size = 100000;
        s_opts.overlap = 10000;
        fastlowess::StreamingLowess stream(s_opts);
        (void)stream.process_chunk(positions, coverage);
        auto result = stream.finalize().value();

        return 0;
    }
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
