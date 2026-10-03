\page benchmarks Benchmarks

# Benchmarks

## CPU Benchmarks

Speedup relative to R's `stats::lowess` (higher is better):

| Category | R baseline | Serial | Parallel |
| --- | --- | --- | --- |
| **Clustered** | 1.88 ms | 0.8× | **2.1×** |
| **Constant Y** | 1.50 ms | 0.7× | **2.0×** |
| **Extreme Outliers** | 5.32 ms | 0.9× | **2.0×** |
| **Financial** (500–5K) | 0.60 ms | 1.0× | **1.3×** |
| **Fraction** (0.05–0.67) | 3.24 ms | 0.9× | **2.4×** |
| **Genomic** (1K–100K) | 8.50 ms | 1.0× | **1.9×** |
| **High Noise** | 6.98 ms | 0.9× | **3.0×** |
| **Iterations** (0–10) | 2.28 ms | 0.8× | **1.7×** |
| **Large** (50K, delta=0) | 10157.00 ms | 1.6× | **7.3×** |
| **Large** (50K, delta=auto) | 11.48 ms | 0.8× | **2.0×** |
| **Large** (50K, 10 iter) | 37167.03 ms | 2.7× | **9.8×** |
| **Large** (20K, fraction=0.67) | 10900.56 ms | 2.3× | **12.2×** |
| **Scale** (1K–10K) | 1.36 ms | 1.0× | **1.4×** |
| **Scientific** (500–5K) | 0.87 ms | 1.3× | **1.4×** |

*The R column shows average time across scenarios in multi-scenario categories. Speedups are averages across the same range.*

---

## GPU Backend

For large batch datasets the GPU backend outperforms CPU-parallel execution. The crossover depends on `fraction × n`:

| Scenario | CPU-Parallel | GPU | Speedup |
| --- | --- | --- | --- |
| n = 1M, fraction = 0.5 | 1.24 s | 187 ms | **6.6×** |

At `fraction = 0.5`, GPU overtakes CPU around n ≥ 50K; at smaller fractions, around n ≥ 100K–250K. See the benchmarks README in the source repository for the full sweep and transfer-overhead breakdown.
