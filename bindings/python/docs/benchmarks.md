# Benchmarks

Speedup relative to R's `stats::lowess` (higher is better):

| Category | R (baseline) | fastlowess (Serial) | fastlowess (Parallel) |
| --- | --- | --- | --- |
| **Clustered** | 1.88ms | 0.8× | **2.1×** |
| **Constant Y** | 1.50ms | 0.7× | **2.0×** |
| **Extreme Outliers** | 5.32ms | 0.9× | **2.0×** |
| **Financial** (500–5K) | 0.60ms | 1.0× | **1.3×** |
| **Fraction** (0.05–0.67) | 3.24ms | 0.9× | **2.4×** |
| **Genomic** (1K–100K) | 8.50ms | 1.0× | **1.9×** |
| **High Noise** | 6.98ms | 0.9× | **3.0×** |
| **Iterations** (0–10) | 2.28ms | 0.8× | **1.7×** |
| **Large** (50K, delta=0) | 10157.00ms | 1.6× | **7.3×** |
| **Large** (50K, delta=auto) | 11.48ms | 0.8× | **2.0×** |
| **Large** (50K, 10 iter) | 37167.03ms | 2.7× | **9.8×** |
| **Large** (20K, fraction=0.67) | 10900.56ms | 2.3× | **12.2×** |
| **Scale** (1K–10K) | 1.36ms | 1.0× | **1.4×** |
| **Scientific** (500–5K) | 0.87ms | 1.3× | **1.4×** |

*The R column shows the average time across scenarios in multi-scenario categories. Speedups are averages across the same range.*

## GPU Backend

For large batch datasets, the GPU backend can outperform CPU-parallel execution. The crossover point is driven by window size (`fraction × n`): at `fraction = 0.5`, GPU overtakes CPU around n ≥ 50K; at smaller fractions, around n ≥ 100K–250K. At n = 1M (`fraction = 0.5`, 3 iterations), GPU is **6.6×** faster than CPU-parallel (1.24s → 187ms). See [benchmarks/README.md](https://github.com/thisisamirv/lowess-project/blob/main/benchmarks/README.md#gpu-benchmarks) for the full sweep and transfer-overhead breakdown.
