---
title: Benchmarks
---
<!-- markdownlint-disable MD024 MD046 -->

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

:::note
The WebAssembly build runs single-threaded (no `parallel` option). The figures
above reflect the native Node.js binding with worker-thread parallelism. WASM
serial performance is comparable to the Serial column.
:::

---

## Reproducing Benchmarks

Use `performance.now()` to time serial WASM runs:

```javascript
const { Lowess } = require('fastlowess-wasm');

function benchMs(fn, reps = 10) {
    fn(); // warm-up
    const { performance } = require('perf_hooks');
    const t0 = performance.now();
    for (let i = 0; i < reps; i++) fn();
    return (performance.now() - t0) / reps;
}

const n = 5000;
const x = Float64Array.from({ length: n }, (_, i) => (i / (n - 1)) * 10);
const y = Float64Array.from(x, (xi, i) =>
    Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6
);

const ms = benchMs(() => new Lowess({ fraction: 0.67 }).fit(x, y));
console.log(`WASM: ${ms.toFixed(2)} ms`);
```

```output
WASM: 16.73 ms
```
