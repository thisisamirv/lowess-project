<!-- markdownlint-disable MD024 MD046 -->
# Benchmarks

## CPU Benchmarks

Speedup relative to R's `stats::lowess` (higher is better):

| Category | R baseline | Serial | Parallel |
| --- | --- | --- | --- |
| **Clustered** | 2.34 ms | 2.0× | **2.5×** |
| **Constant Y** | 1.81 ms | 1.7× | **3.2×** |
| **Extreme Outliers** | 5.81 ms | 1.5× | **2.6×** |
| **Financial** (500–5K) | 0.65 ms | **2.0×** | 1.4× |
| **Fraction** (0.05–0.67) | 3.8 ms | 1.6× | **3.2×** |
| **Genomic** (1K–100K) | 11.2 ms | 2.2× | **2.4×** |
| **High Noise** | 7.08 ms | 1.5× | **3.6×** |
| **Iterations** (0–10) | 3.0 ms | 1.9× | **2.7×** |
| **Scale** (1K–10K) | 1.6 ms | 1.5× | **1.6×** |
| **Scientific** (500–5K) | 0.9 ms | 1.4× | 1.4× |

*The R column shows average time across scenarios in multi-scenario categories.
Speedups are averages across the same range.*

---

## GPU Backend

For large batch datasets the GPU backend outperforms CPU-parallel execution.
The crossover depends on `fraction × n`:

| Scenario | CPU-Parallel | GPU | Speedup |
| --- | --- | --- | --- |
| n = 1M, fraction = 0.5 | 1.24 s | 187 ms | **6.6×** |

At `fraction = 0.5`, GPU overtakes CPU around n ≥ 50K; at smaller fractions,
around n ≥ 100K–250K. See the benchmarks README in the source repository for
the full sweep and transfer-overhead breakdown.

---

## Reproducing Benchmarks

Use `std::chrono` to time serial vs parallel runs:

```cpp
#include <fastlowess.hpp>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

static double bench_ms(auto fn, int reps = 10) {
    // Warm-up
    fn();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < reps; ++i) fn();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
}

int main() {
    const int n = 5000;
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = static_cast<double>(i) / (n - 1) * 10.0;
        y[i] = std::sin(x[i]) + ((i * 7 + 3) % 17 / 17.0 - 0.5) * 0.6;
    }

    fastlowess::LowessOptions opts;
    opts.fraction = 0.67;

    auto serial_ms = bench_ms([&] {
        opts.parallel = false;
        fastlowess::Lowess(opts).fit(x, y);
    });

    auto parallel_ms = bench_ms([&] {
        opts.parallel = true;
        fastlowess::Lowess(opts).fit(x, y);
    });

    std::cout << "Serial:   " << serial_ms   << " ms\n";
    std::cout << "Parallel: " << parallel_ms << " ms\n";
    std::cout << "Speedup:  " << serial_ms / parallel_ms << "×\n";
    return 0;
}
```
