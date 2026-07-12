# Benchmarks

Speedup relative to R's `stats::lowess` (higher is better):

| Category | R (baseline) | rfastlowess (Serial) | rfastlowess (Parallel) |
| --- | --- | --- | --- |
| **Clustered** | 2.34ms | 2.0× | **2.5×** |
| **Constant Y** | 1.81ms | 1.7× | **3.2×** |
| **Extreme Outliers** | 5.81ms | 1.5× | **2.6×** |
| **Financial** (500–5K) | 0.65ms | **2.0×** | 1.4× |
| **Fraction** (0.05–0.67) | 3.8ms | 1.6× | **3.2×** |
| **Genomic** (1K–100K) | 11.2ms | 2.2× | **2.4×** |
| **High Noise** | 7.08ms | 1.5× | **3.6×** |
| **Iterations** (0–10) | 3.0ms | 1.9× | **2.7×** |
| **Scale** (1K–10K) | 1.6ms | 1.5× | **1.6×** |
| **Scientific** (500–5K) | 0.9ms | 1.4× | 1.4× |

*The R column shows the average time across scenarios in multi-scenario categories. Speedups are averages across the same range.*
