// CPU-parallel vs GPU LOWESS benchmark.
//
// Sweeps n × fraction × robustness_iterations and records median wall-clock
// time for each backend. Outputs rust_benchmark_cpu_vs_gpu.json so that
// compare_cpu_gpu.py can find the crossover points.

use fastLowess::internals::api::Backend;
use fastLowess::prelude::*;
use serde::Serialize;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

// ─── output schema ───────────────────────────────────────────────────────────

#[derive(Serialize)]
struct Row {
    n: usize,
    fraction: f64,
    robustness_iterations: usize,
    cpu_median_ms: f64,
    cpu_mean_ms: f64,
    gpu_median_ms: f64,
    gpu_mean_ms: f64,
    /// >1 means GPU is faster; <1 means CPU is faster
    gpu_speedup: f64,
    winner: &'static str,
}

#[derive(Serialize)]
struct Output {
    n_timed_runs: usize,
    n_warmup_runs: usize,
    results: Vec<Row>,
}

// ─── helpers ─────────────────────────────────────────────────────────────────

fn generate_sine(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut state: u64 = 42;
    let lcg = |s: &mut u64| -> f64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*s >> 33) as f64) / (u32::MAX as f64) * 0.4 - 0.2
    };
    let x: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64 * 10.0).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + lcg(&mut state)).collect();
    (x, y)
}

fn median(sorted: &[f64]) -> f64 {
    let m = sorted.len();
    if m % 2 == 0 {
        (sorted[m / 2 - 1] + sorted[m / 2]) / 2.0
    } else {
        sorted[m / 2]
    }
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn r3(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}

fn time_fit(x: &[f64], y: &[f64], fraction: f64, iters: usize, backend: Backend) -> f64 {
    let m = Lowess::new()
        .fraction(fraction)
        .iterations(iters)
        .backend(backend)
        .build()
        .unwrap_or_else(|e| panic!("build failed ({:?}): {}", backend, e));
    let t0 = Instant::now();
    m.fit(x, y)
        .unwrap_or_else(|e| panic!("fit failed ({:?}): {}", backend, e));
    t0.elapsed().as_secs_f64() * 1000.0
}

// ─── main ────────────────────────────────────────────────────────────────────

fn main() {
    let output_path: PathBuf = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("../output/rust_benchmark_cpu_vs_gpu.json"));

    let sizes: &[usize] = &[
        1_000, 2_000, 5_000, 10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000,
    ];
    let fractions: &[f64] = &[0.1, 0.3, 0.5];
    let iter_counts: &[usize] = &[0, 3];

    let n_timed = 10usize;
    let n_warmup = 3usize;

    // Pre-warm the GPU executor once so singleton init doesn't skew first cell.
    eprintln!("Warming up GPU executor...");
    {
        let (x, y) = generate_sine(1_000);
        for _ in 0..n_warmup {
            let _ = time_fit(&x, &y, 0.3, 3, Backend::GPU);
        }
    }
    eprintln!("GPU warm-up done.\n");

    let total_cells = sizes.len() * fractions.len() * iter_counts.len();
    let mut results: Vec<Row> = Vec::with_capacity(total_cells);
    let mut cell = 0usize;

    eprintln!(
        "{:>10}  {:>8}  {:>5}  {:>10}  {:>10}  {:>8}  {}",
        "n", "fraction", "iter", "cpu_med", "gpu_med", "speedup", "winner"
    );

    for &n in sizes {
        let (x, y) = generate_sine(n);

        for &fraction in fractions {
            for &iters in iter_counts {
                cell += 1;
                eprint!(
                    "[{}/{}] n={} f={} i={}  ",
                    total_cells, cell, n, fraction, iters
                );

                // ── CPU parallel ──────────────────────────────────────────
                for _ in 0..n_warmup {
                    let _ = time_fit(&x, &y, fraction, iters, Backend::CPU);
                }
                let mut cpu_times: Vec<f64> = (0..n_timed)
                    .map(|_| time_fit(&x, &y, fraction, iters, Backend::CPU))
                    .collect();
                cpu_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

                // ── GPU ───────────────────────────────────────────────────
                // Per-cell warm-up matches CPU warm-up: lets wgpu settle buffer
                // allocations and dispatch parameters for this (n, fraction, iters).
                for _ in 0..n_warmup {
                    let _ = time_fit(&x, &y, fraction, iters, Backend::GPU);
                }
                let mut gpu_times: Vec<f64> = (0..n_timed)
                    .map(|_| time_fit(&x, &y, fraction, iters, Backend::GPU))
                    .collect();
                gpu_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

                let cpu_med = r3(median(&cpu_times));
                let cpu_avg = r3(mean(&cpu_times));
                let gpu_med = r3(median(&gpu_times));
                let gpu_avg = r3(mean(&gpu_times));
                let speedup = r3(cpu_med / gpu_med);
                let winner = if speedup >= 1.0 { "GPU" } else { "CPU" };

                eprintln!(
                    "cpu={:.2}ms  gpu={:.2}ms  {:.2}×  {}",
                    cpu_med, gpu_med, speedup, winner
                );

                results.push(Row {
                    n,
                    fraction,
                    robustness_iterations: iters,
                    cpu_median_ms: cpu_med,
                    cpu_mean_ms: cpu_avg,
                    gpu_median_ms: gpu_med,
                    gpu_mean_ms: gpu_avg,
                    gpu_speedup: speedup,
                    winner,
                });
            }
        }
    }

    let out = Output {
        n_timed_runs: n_timed,
        n_warmup_runs: n_warmup,
        results,
    };

    let json = serde_json::to_string_pretty(&out).expect("JSON serialisation failed");
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).ok();
    }
    fs::write(&output_path, &json).expect("Failed to write output JSON");
    eprintln!("\nResults written to {}", output_path.display());
}
