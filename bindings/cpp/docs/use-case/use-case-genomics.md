\page use_case_genomics Genomic Data Smoothing

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

    std::cout << "95% CI: [" << result.confidence_lower()[0] << ", " << result.confidence_upper()[0] << "]\n";
    return 0;
}
```

```output
95% CI: [51.6773, 68.7372]
```

---

## ChIP-seq Signal Smoothing

### Application

ChIP-seq experiments produce sparse, noisy coverage data. LOWESS can help identify binding regions.

`fraction = 0.05` provides high spatial resolution — important for resolving narrow binding peaks that would otherwise be smeared into the background. The larger `iterations = 5` is deliberate: Poisson-distributed read counts produce tall, isolated spikes, and extra robustness iterations progressively down-weight them so the estimated background level is not inflated by a handful of extreme counts.

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
    int peak_count = 0;
    for (double y : result.y_vector()) {
        if (y > 65.0) ++peak_count;
    }

    std::cout << "y[0]: " << result.y_vector()[0] << "\n";
    std::cout << "Peak count: " << peak_count << "\n";
    return 0;
}
```

```output
y[0]: 59.952
Peak count: 26
```

---

## Large Genome Coverage (Streaming)

For whole-genome data that doesn't fit in memory:

```cpp
#include <fastlowess.hpp>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    const int n = 1001;
    std::vector<double> x_chunk(n), y_chunk(n);
    for (int i = 0; i < n; ++i) {
        x_chunk[i] = i * 10.0;
        y_chunk[i] = 50.0 + std::sin(x_chunk[i] / 100.0) * 20.0 + 5.0;
    }

    fastlowess::StreamingOptions s_opts;
    s_opts.fraction = 0.05;
    s_opts.iterations = 3;
    s_opts.chunk_size = 50;
    s_opts.overlap = 10;
    s_opts.merge_strategy = "weighted_average";
    fastlowess::StreamingLowess stream(s_opts);
    (void)stream.process_chunk(x_chunk, y_chunk);
    auto result = stream.finalize().value();

    std::cout << "y[0]: " << result.y_vector()[0] << "\n";
    return 0;
}
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

- [Concepts](concepts.md) — How LOWESS works
- [API Reference](api.md) — All options
- [Robustness](robustness.md) — Outlier downweighting in depth
- [Merge Strategies](merge.md) — Streaming chunk reconciliation
- [Boundary Handling](boundary.md) — Edge handling for sparse regions
- [Real-Time Processing](use-case-real-time.md) — For sequencing runs
