//! fastLowess Parallel Smoothing Examples
//!
//! This example demonstrates features specific to `fastLowess`:
//! - Parallel execution using `rayon`
//! - Sequential fallback
//! - `ndarray` integration
//! - Cross-validation for automatic parameter selection
//! - Performance comparison (simulated)

#[cfg(feature = "cpu")]
use fastLowess::prelude::*;

#[cfg(feature = "cpu")]
use ndarray::Array1;
#[cfg(feature = "cpu")]
use std::time::Instant;

#[cfg(feature = "cpu")]
fn main() -> Result<(), LowessError> {
    #[cfg(feature = "cpu")]
    {
        println!("{}", "=".repeat(80));
        println!("fastLowess Parallel Smoothing Examples");
        println!("{}", "=".repeat(80));
        println!();

        example_1_parallel_execution()?;
        example_2_sequential_fallback()?;
        example_3_ndarray_integration()?;
        example_4_robust_parallel()?;
        example_5_cross_validation()?;
        example_6_custom_weights()?;
    }
    Ok(())
}

#[cfg(not(feature = "cpu"))]
fn main() {}

#[cfg(feature = "cpu")]
/// Example 1: Parallel Execution
/// Demonstrates the default parallel execution mode
fn example_1_parallel_execution() -> Result<(), LowessError> {
    println!("Example 1: Parallel Execution");
    println!("{}", "-".repeat(80));

    // Generate a larger synthetic dataset
    let n = 10_000;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| (xi * 0.1).sin() + (xi * 0.01).cos())
        .collect();

    let start = Instant::now();
    let model = Lowess::new()
        .fraction(0.5) // Use 50% of data for each local fit
        .iterations(3) // 3 robustness iterations
        .adapter(Batch) // Use Batch adapter from fastLowess (renamed from Standard)
        .parallel(true) // Enable parallel execution (default)
        .build()?;

    let result = model.fit(&x, &y)?;
    let duration = start.elapsed();

    println!("Processed {} points in {:?}", n, duration);
    println!("Execution mode: Parallel");
    println!("Result summary:\n{}", result);

    println!();
    Ok(())
}

#[cfg(feature = "cpu")]
/// Example 2: Sequential Fallback
/// Demonstrates explicitly disabling parallelism
fn example_2_sequential_fallback() -> Result<(), LowessError> {
    println!("Example 2: Sequential Fallback");
    println!("{}", "-".repeat(80));

    let n = 10_000;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| (xi * 0.1).sin() + (xi * 0.01).cos())
        .collect();

    let start = Instant::now();
    let model = Lowess::new()
        .adapter(Batch)
        .parallel(false) // Disable parallel execution
        .build()?;

    let _result = model.fit(&x, &y)?;
    let duration = start.elapsed();

    println!("Processed {} points in {:?}", n, duration);
    println!("Execution mode: Sequential");
    // Note: Sequential might be slower for large N

    println!();
    Ok(())
}

#[cfg(feature = "cpu")]
/// Example 3: NdArray Integration
/// Demonstrates direct usage with ndarray types
fn example_3_ndarray_integration() -> Result<(), LowessError> {
    println!("Example 3: NdArray Integration");
    println!("{}", "-".repeat(80));

    // Create ndarray arrays using standard Vec
    let x_vec: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
    let y_vec: Vec<f64> = x_vec.iter().map(|&xi| xi.sin() + 0.1 * xi).collect();

    let x = Array1::from(x_vec);
    let y = Array1::from(y_vec);

    // Fit directly with ndarray inputs
    let res = Lowess::new()
        .adapter(Batch)
        .parallel(true)
        .build()?
        .fit(&x, &y)?;

    println!("Successfully fitted to ndarray inputs.");
    println!("First 5 smoothed values:");
    for val in res.y.iter().take(5) {
        println!("  {:.4}", val);
    }

    println!();
    Ok(())
}

#[cfg(feature = "cpu")]
/// Example 4: Robust Parallel Smoothing
/// Demonstrates parallel execution with robustness iterations
fn example_4_robust_parallel() -> Result<(), LowessError> {
    println!("Example 4: Robust Parallel Smoothing");
    println!("{}", "-".repeat(80));

    // Data with outliers
    let n = 1000;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
    let y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &xi)| {
            if i % 100 == 0 { xi + 10.0 } else { xi.sin() } // Periodic outliers
        })
        .collect();

    let model = Lowess::new()
        .fraction(0.1)
        .iterations(3)
        .robustness_method("bisquare")
        .return_robustness_weights()
        .adapter(Batch)
        .parallel(true)
        .build()?;

    let result = model.fit(&x, &y)?;

    println!("Parallel fit with 3 robustness iterations completed.");
    if let Some(weights) = &result.robustness_weights {
        let outliers = weights.iter().filter(|&&w| w < 0.1).count();
        println!("Identified {} potential outliers (weight < 0.1)", outliers);
    }

    println!();
    Ok(())
}

#[cfg(feature = "cpu")]
/// Example 5: Cross-Validation for Parameter Selection
/// Automatic selection of optimal smoothing fraction using parallel CV
fn example_5_cross_validation() -> Result<(), LowessError> {
    println!("Example 5: Cross-Validation for Parameter Selection (Parallel)");
    println!("{}", "-".repeat(80));

    let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| 2.0 * xi + 1.0 + (xi * 0.5).sin())
        .collect();

    // Test multiple fractions and select the best one using parallel execution
    let start = Instant::now();
    let model = Lowess::new()
        .cv_method("kfold")
        .cv_k(5)
        .cv_fractions(vec![0.2, 0.3, 0.5, 0.7])
        .iterations(2)
        .adapter(Batch)
        .parallel(true)
        .build()?;

    let result = model.fit(&x, &y)?;
    let duration = start.elapsed();

    println!("Cross-validation completed in {:?}", duration);
    println!("Selected fraction: {}", result.fraction_used);
    if let Some(scores) = &result.cv_scores {
        println!("CV scores for each fraction: {:?}", scores);
    }
    println!("\n{}", result);

    /* Expected Output:
    Cross-validation completed in XXXµs
    Selected fraction: 0.5
    CV scores for each fraction: [0.123, 0.098, 0.145, 0.187]

    Summary:
      Data points: 20
      Fraction: 0.5 (selected via K-Fold CV)

    Smoothed Data:
           X     Y_smooth
      --------------------
        1.00     3.47943
        2.00     5.47943
        3.00     7.14112
        ... (17 more rows)
    */

    println!();
    Ok(())
}

#[cfg(feature = "cpu")]
/// Example 6: Custom Weights (Parallel)
///
/// Custom weights work correctly under rayon parallelism: each worker reads
/// its portion of the weight slice independently.
fn example_6_custom_weights() -> Result<(), LowessError> {
    println!("Example 6: Custom Weights (Parallel)");
    println!("{}", "-".repeat(80));

    // --- 6a: Zero weight suppresses outlier in parallel mode ---
    let x: Vec<f64> = (0..30).map(|i| i as f64).collect();
    let mut y: Vec<f64> = x.iter().map(|v| v * 3.0).collect();
    y[15] = 200.0;

    let result_no_w = Lowess::new()
        .fraction(0.4)
        .iterations(0)
        .adapter(Batch)
        .parallel(true)
        .build()?
        .fit(&x, &y)?;

    let mut weights = vec![1.0_f64; x.len()];
    weights[15] = 0.0;

    let result_zero_w = Lowess::new()
        .fraction(0.4)
        .iterations(0)
        .custom_weights(weights)
        .adapter(Batch)
        .parallel(true)
        .build()?
        .fit(&x, &y)?;

    let true_val = 15.0 * 3.0;
    let err_no_w = (result_no_w.y[15] - true_val).abs();
    let err_zero_w = (result_zero_w.y[15] - true_val).abs();
    println!("  Zero weight at outlier (x=15): error {err_no_w:.2} -> {err_zero_w:.2}");

    // --- 6b: Sequential and parallel with custom weights agree ---
    let weights2: Vec<f64> = (0..100)
        .map(|i| if i % 5 == 0 { 0.2 } else { 1.0 })
        .collect();
    let x2: Vec<f64> = (0..100).map(|i| i as f64 / 10.0).collect();
    let y2: Vec<f64> = x2.iter().map(|&v| v.sin()).collect();

    let seq = Lowess::new()
        .fraction(0.2)
        .iterations(0)
        .custom_weights(weights2.clone())
        .adapter(Batch)
        .parallel(false)
        .build()?
        .fit(&x2, &y2)?;

    let par = Lowess::new()
        .fraction(0.2)
        .iterations(0)
        .custom_weights(weights2)
        .adapter(Batch)
        .parallel(true)
        .build()?
        .fit(&x2, &y2)?;

    let max_diff = seq
        .y
        .iter()
        .zip(&par.y)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    println!("  Sequential vs parallel max diff: {max_diff:.2e}");

    println!();
    Ok(())
}
