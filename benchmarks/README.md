# Benchmarks

Compares `stats::lowess` (base R) against `rfastlowess` (this package) across a set of representative scenarios, and profiles GPU backend performance.

## Scenarios

| Category | Variants | Description |
| --- | --- | --- |
| **Scalability** | n = 1 000 / 5 000 / 10 000 | Sine wave, fraction 0.1, 3 robustness iterations |
| **Fraction** | 0.05 – 0.67 (6 levels) | Effect of smoothing span, n = 5 000 |
| **Iterations** | 0 – 10 (6 levels) | Effect of robustness iterations on outlier data, n = 5 000 |
| **Financial** | n = 500 / 1 000 / 5 000 | Cumulative-return time series, fraction 0.1 |
| **Scientific** | n = 500 / 1 000 / 5 000 | Damped-oscillator signal, fraction 0.15 |
| **Genomic** | n = 1 000 / 5 000 / 100 000 | Step-function expression data, fraction 0.1 |
| **Pathological** | clustered, high-noise, extreme outliers, constant y | Edge cases: clustered x-values, high-noise signal, extreme outliers, and a constant y signal |
| **Large Scale** | n = 20 000 / 50 000 | Stress tests at scale: exact fit (`delta = 0`), interpolation shortcut, high iteration count, high fraction |

## Results

Median times per scenario, and rfastlowess's speedup over `stats::lowess` (in parentheses):

| Scenario | `stats::lowess` | rfastlowess (serial) | rfastlowess (parallel) |
| --- | ---: | ---: | ---: |
| scale_1000 | 0.31 ms | 0.28 ms (1.1×) | 0.28 ms (1.1×) |
| scale_5000 | 1.34 ms | 1.36 ms (1.0×) | 1.00 ms (1.3×) |
| scale_10000 | 2.42 ms | 2.75 ms (0.9×) | 1.39 ms (1.7×) |
| fraction_0.05 | 0.82 ms | 0.97 ms (0.8×) | 0.97 ms (0.8×) |
| fraction_0.1 | 1.63 ms | 1.35 ms (1.2×) | 0.87 ms (1.9×) |
| fraction_0.2 | 2.40 ms | 2.50 ms (1.0×) | 1.06 ms (2.3×) |
| fraction_0.3 | 3.15 ms | 3.68 ms (0.9×) | 1.22 ms (2.6×) |
| fraction_0.5 | 4.89 ms | 5.90 ms (0.8×) | 1.61 ms (3.0×) |
| fraction_0.67 | 6.52 ms | 7.68 ms (0.8×) | 1.83 ms (3.6×) |
| iterations_0 | 0.47 ms | 0.74 ms (0.6×) | 0.33 ms (1.4×) |
| iterations_1 | 0.99 ms | 1.25 ms (0.8×) | 0.77 ms (1.3×) |
| iterations_2 | 1.52 ms | 1.75 ms (0.9×) | 0.88 ms (1.7×) |
| iterations_3 | 2.02 ms | 2.24 ms (0.9×) | 1.16 ms (1.7×) |
| iterations_5 | 2.93 ms | 3.27 ms (0.9×) | 1.54 ms (1.9×) |
| iterations_10 | 5.75 ms | 6.33 ms (0.9×) | 2.59 ms (2.2×) |
| financial_500 | 0.11 ms | 0.13 ms (0.9×) | 0.14 ms (0.8×) |
| financial_1000 | 0.18 ms | 0.19 ms (0.9×) | 0.21 ms (0.9×) |
| financial_5000 | 1.51 ms | 1.19 ms (1.3×) | 0.66 ms (2.3×) |
| scientific_500 | 0.32 ms | 0.22 ms (1.4×) | 0.22 ms (1.4×) |
| scientific_1000 | 0.56 ms | 0.38 ms (1.5×) | 0.39 ms (1.4×) |
| scientific_5000 | 1.73 ms | 1.89 ms (0.9×) | 1.16 ms (1.5×) |
| genomic_1000 | 0.32 ms | 0.28 ms (1.2×) | 0.29 ms (1.1×) |
| genomic_5000 | 1.17 ms | 1.25 ms (0.9×) | 0.91 ms (1.3×) |
| genomic_100000 | 24.02 ms | 26.26 ms (0.9×) | 6.99 ms (3.4×) |
| clustered | 1.88 ms | 2.38 ms (0.8×) | 0.89 ms (2.1×) |
| high_noise | 6.98 ms | 8.08 ms (0.9×) | 2.34 ms (3.0×) |
| extreme_outliers | 5.32 ms | 6.15 ms (0.9×) | 2.67 ms (2.0×) |
| constant_y | 1.50 ms | 2.16 ms (0.7×) | 0.73 ms (2.0×) |

At small scenario sizes, fixed overhead (FFI call, allocation) can outweigh `rfastlowess`'s per-point algorithmic work, so serial speedup dips below 1× on several scenarios; the parallel backend recovers a speedup in almost every case once there's enough work to spread across threads. See [Large Scale Benchmarks](#large-scale-benchmarks) below for how this trend continues at higher n.

## Large Scale Benchmarks

Every other scenario above completes in well under 100ms, which doesn't stress-test performance differences at scale. The `large` category forces `delta = 0` (disabling `stats::lowess`'s interpolation shortcut, which skips recomputation between nearby points) for an exact, apples-to-apples comparison, across four variants:

| Variant | Size | Description |
| --- | --- | --- |
| `large_delta_0` | 50 000 | Exact fit baseline: fraction 0.1, 3 iterations, `delta = 0` |
| `large_delta_0.1` | 50 000 | Same workload with `delta` left at its default (auto ≈ 0.1), showing the interpolation shortcut's speedup |
| `large_high_iter` | 50 000 | 10 robustness iterations instead of 3 (still `delta = 0`) |
| `large_high_fraction` | 20 000 | Fraction 0.67 (wider local window) at a smaller size, since 0.67 × 50 000 is too slow to run repeatedly |

Median times, and fastLowess's speedup over `stats::lowess`:

| Variant | `stats::lowess` | fastLowess (serial) | fastLowess (parallel) | Speedup (serial) | Speedup (parallel) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `large_delta_0` | 10.16 s | 6.32 s | 1.38 s | 1.6× | 7.3× |
| `large_delta_0.1` | 11.48 ms | 13.75 ms | 5.77 ms | 0.8× | 2.0× |
| `large_high_iter` | 37.17 s | 13.62 s | 3.81 s | 2.7× | 9.8× |
| `large_high_fraction` | 10.90 s | 4.72 s | 0.90 s | 2.3× | 12.2× |

The speedup grows with the amount of per-point work (more iterations, wider fraction): parallel execution pays off most when there's more to parallelize, reaching **12.2×** at `large_high_fraction`. The `large_delta_0.1` variant shows how much of `stats::lowess`'s exact-fit cost simply disappears once its interpolation shortcut is allowed to kick in (10.16s → 11.48ms) — fastLowess's own shortcut yields a similar drop (6.32s → 13.75ms serial), though at that point fixed per-call overhead dominates enough that the serial backend no longer outpaces `stats::lowess`.

## GPU Benchmarks

### CPU-parallel vs GPU crossover (`bench-cpu-vs-gpu`)

Sweeps n × fraction × robustness iterations to find where the GPU backend beats CPU-parallel. Sine-wave data, 10 timed runs after 3 warm-up runs per cell.

**Crossover:** first n where GPU median < CPU median

| fraction | iterations=0 | iterations=3 |
| ---: | ---: | ---: |
| 0.1 | n ≥ 100K | n ≥ 250K |
| 0.3 | n ≥ 100K | n ≥ 100K |
| 0.5 | n ≥ 50K | n ≥ 50K |

The crossover is driven by window size (fraction × n), not iteration count — a larger window makes each local fit heavier, favouring the GPU sooner. At n = 1M with fraction = 0.5 and 3 iterations, GPU is **6.6×** faster (1.24 s → 187 ms).

Selected results:

| n | fraction | iter | CPU (med) | GPU (med) | speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 000 | 0.5 | 3 | 4.3 ms | 6.6 ms | 0.65× CPU |
| 50 000 | 0.5 | 0 | 4.3 ms | 3.7 ms | 1.15× GPU |
| 100 000 | 0.5 | 3 | 74.0 ms | 21.3 ms | 3.5× GPU |
| 250 000 | 0.3 | 3 | 135 ms | 30.3 ms | 4.5× GPU |
| 1 000 000 | 0.5 | 3 | 1 237 ms | 187 ms | 6.6× GPU |

### End-to-end GPU fit (`bench-gpu-rust`)

Measures total wall-clock time for a complete GPU LOWESS fit (upload + kernel + download) using the Rust `fastLowess` GPU backend directly. Sine-wave data, fraction 0.3, 3 robustness iterations, 10 timed runs after 2 warm-up runs.

| n | mean | median | min | max |
| ---: | ---: | ---: | ---: | ---: |
| 1 000 | 7.3 ms | 7.1 ms | 5.3 ms | 9.0 ms |
| 5 000 | 6.8 ms | 6.5 ms | 5.3 ms | 9.2 ms |
| 10 000 | 7.1 ms | 6.9 ms | 5.5 ms | 9.1 ms |
| 50 000 | 9.1 ms | 8.9 ms | 8.3 ms | 10.3 ms |
| 100 000 | 15.7 ms | 15.7 ms | 13.6 ms | 17.9 ms |
| 500 000 | 67.0 ms | 66.1 ms | 65.3 ms | 73.2 ms |
| 1 000 000 | 155.3 ms | 154.5 ms | 148.3 ms | 170.6 ms |

The flat ~7 ms cost at small n is dominated by GPU executor initialisation and PCIe round-trip latency, not compute. Compute begins to dominate above n ≈ 50 000.

### Host ↔ device transfer overhead (`bench-gpu-transfer`)

Isolates buffer transfer cost from kernel execution. Upload sends 2 × n f32 (x, y); download retrieves 3 × n f32 (y\_smooth, weights, residuals). 20 timed runs after 5 warm-up runs.

| n | upload | download | round-trip | bandwidth |
| ---: | ---: | ---: | ---: | ---: |
| 1 000 | 0.57 ms | 0.59 ms | 1.16 ms | 0.017 GB/s |
| 5 000 | 0.67 ms | 0.64 ms | 1.30 ms | 0.077 GB/s |
| 10 000 | 0.47 ms | 0.49 ms | 0.96 ms | 0.209 GB/s |
| 50 000 | 0.99 ms | 0.84 ms | 1.83 ms | 0.546 GB/s |
| 100 000 | 0.76 ms | 1.35 ms | 2.11 ms | 0.946 GB/s |
| 500 000 | 3.14 ms | 4.79 ms | 7.93 ms | 1.261 GB/s |
| 1 000 000 | 5.39 ms | 10.31 ms | 15.70 ms | 1.274 GB/s |

Round-trip latency floors at ~0.7 ms regardless of size. Bandwidth saturates near 1.3 GB/s at n ≥ 500 000. Transfer is ≈10–16% of total GPU fit time at n ≥ 500 000, so kernel execution dominates at scale as expected.

## Running

```sh
# Build and install rfastlowess to system R (required before benchmarking)
make install

# Run benchmarks
make bench-r                    # stats::lowess only
make bench-rfastlowess-serial
make bench-rfastlowess-parallel

# GPU benchmarks
make bench-cpu-vs-gpu           # CPU vs GPU sweep → output/rust_benchmark_cpu_vs_gpu.json
make bench-gpu-rust             # end-to-end GPU fit → output/rust_benchmark_gpu.json
make bench-gpu-transfer         # transfer overhead only → output/gpu_transfer.json

# Generate comparison plot (output/benchmark_comparison.svg)
make compare
```

Output JSON files are written to `output/`.
