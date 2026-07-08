# Benchmarks

Speedup relative to Python's `statsmodels.lowess` (higher is better):

| Category                        | statsmodels | R (stats) | Serial | Parallel | GPU     |
|---------------------------------|-------------|-----------|--------|----------|---------|
| **Clustered**                   | 163ms       | 83×       | 203×   | **433×** | 32×     |
| **Constant Y**                  | 134ms       | 92×       | 212×   | **410×** | 18×     |
| **Delta** (large–none)          | 105ms       | 2×        | 4×     | 6×       | **16×** |
| **Extreme Outliers**            | 489ms       | 106×      | 201×   | **388×** | 29×     |
| **Financial** (500–10K)         | 106ms       | 105×      | 252×   | **293×** | 12×     |
| **Fraction** (0.05–0.67)        | 221ms       | 104×      | 228×   | **391×** | 22×     |
| **Genomic** (1K–50K)            | 1833ms      | 7×        | 9×     | 20×      | **95×** |
| **High Noise**                  | 435ms       | 133×      | 134×   | **375×** | 32×     |
| **Iterations** (0–10)           | 204ms       | 115×      | 224×   | **386×** | 18×     |
| **Scale** (1K–50K)              | 1841ms      | 264×      | 487×   | **581×** | 98×     |
| **Scientific** (500–10K)        | 167ms       | 109×      | 205×   | **314×** | 15×     |
| **Scale Large**\* (100K–2M)     | —           | —         | 1×     | **1.4×** | 0.3×    |

\*Scale Large benchmarks are relative to Serial (statsmodels cannot handle these sizes)

*The numbers are the average across a range of scenarios for each category (e.g., Delta from none, to small, medium, and large).*
