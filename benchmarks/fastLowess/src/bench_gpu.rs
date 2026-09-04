// GPU end-to-end LOWESS benchmark — outputs rust_benchmark_gpu.json.
//
// Mirrors the shape of rfastlowess_parallel.json so gpu_transfer.py
// can compute meaningful transfer-fraction comparisons.

use fastLowess::prelude::*;
use serde::Serialize;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Serialize)]
struct BenchEntry {
    name: String,
    size: usize,
    iterations: usize,
    mean_time_ms: f64,
    std_time_ms: f64,
    median_time_ms: f64,
    min_time_ms: f64,
    max_time_ms: f64,
    fitted: Vec<f64>,
}

#[derive(Serialize)]
struct BenchOutput {
    scalability: Vec<BenchEntry>,
}

fn generate_sine(n: usize) -> (Vec<f64>, Vec<f64>) {
    // Deterministic pseudo-random noise via LCG so we don't pull in rand.
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

fn bench_size(n: usize, n_iter: usize, warmup: usize) -> BenchEntry {
    let (x, y) = generate_sine(n);

    let model = Lowess::new()
        .fraction(0.3)
        .iterations(3)
        .backend("gpu")
        .build()
        .expect("GPU model build failed");

    // Share model across calls by rebuilding each time (fit consumes the model).
    let run = || -> (f64, Vec<f64>) {
        let m = Lowess::new()
            .fraction(0.3)
            .iterations(3)
            .backend("gpu")
            .build()
            .expect("GPU model build failed");
        let t0 = Instant::now();
        let result = m.fit(&x, &y).expect("GPU fit failed");
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        (elapsed_ms, result.y)
    };

    // Warm-up (initialises GPU executor singleton)
    drop(model); // drop the unused build above
    for _ in 0..warmup {
        let _ = run();
    }

    let mut times = Vec::with_capacity(n_iter);
    let mut last_fitted = Vec::new();
    for _ in 0..n_iter {
        let (ms, fitted) = run();
        times.push(ms);
        last_fitted = fitted;
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let variance = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / times.len() as f64;
    let std = variance.sqrt();
    let median = if times.len() % 2 == 0 {
        (times[times.len() / 2 - 1] + times[times.len() / 2]) / 2.0
    } else {
        times[times.len() / 2]
    };

    eprintln!(
        "  {:>9}  mean={:.3}ms  median={:.3}ms  min={:.3}ms  max={:.3}ms",
        format_n(n),
        mean,
        median,
        times[0],
        times[times.len() - 1],
    );

    BenchEntry {
        name: format!("scale_{}", n),
        size: n,
        iterations: n_iter,
        mean_time_ms: round3(mean),
        std_time_ms: round3(std),
        median_time_ms: round3(median),
        min_time_ms: round3(times[0]),
        max_time_ms: round3(times[times.len() - 1]),
        fitted: last_fitted.iter().map(|v| round4(*v)).collect(),
    }
}

fn round3(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}
fn round4(v: f64) -> f64 {
    (v * 10000.0).round() / 10000.0
}
fn format_n(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{}M", n / 1_000_000)
    } else if n >= 1_000 {
        format!("{}K", n / 1_000)
    } else {
        n.to_string()
    }
}

fn main() {
    let output_path: PathBuf = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("../output/rust_benchmark_gpu.json"));

    let sizes: Vec<usize> = vec![1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000];
    let n_iter = 10usize;
    let warmup = 2usize;

    eprintln!(
        "GPU LOWESS benchmark  (n_iter={}, warmup={})",
        n_iter, warmup
    );
    eprintln!("  {:>9}  {}", "n", "timing");

    let scalability: Vec<BenchEntry> = sizes
        .iter()
        .map(|&n| bench_size(n, n_iter, warmup))
        .collect();

    let out = BenchOutput { scalability };
    let json = serde_json::to_string_pretty(&out).expect("JSON serialisation failed");

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).ok();
    }
    fs::write(&output_path, json).expect("Failed to write output JSON");
    eprintln!("\nResults written to {}", output_path.display());
}
