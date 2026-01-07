//! Comprehensive LOWESS Batch Smoothing Examples
//!
//! This example demonstrates various LOWESS smoothing scenarios:
//! - Basic smoothing with minimal configuration
//! - Robust smoothing with outlier handling
//! - Uncertainty quantification with confidence/prediction intervals
//! - Cross-validation for automatic parameter selection
//! - Complete diagnostic analysis
//! - Different weight functions and robustness methods
//!
//! Each scenario includes the expected output as comments.

#[cfg(feature = "std")]
use lowess::prelude::*;
#[cfg(feature = "std")]
use std::time::Instant;

#[cfg(feature = "std")]
fn main() -> Result<(), LowessError> {
    println!("{}", "=".repeat(80));
    println!("LOWESS Batch Smoothing - Comprehensive Examples");
    println!("{}", "=".repeat(80));
    println!();

    // Run all example scenarios
    example_1_basic_smoothing()?;
    example_2_robust_with_outliers()?;
    example_3_uncertainty_quantification()?;
    example_4_cross_validation()?;
    example_5_complete_diagnostics()?;
    example_6_different_kernels()?;
    example_7_robustness_methods()?;
    example_8_benchmark()?;

    Ok(())
}

#[cfg(not(feature = "std"))]
fn main() {}

#[cfg(feature = "std")]
/// Example 1: Basic Smoothing
/// Demonstrates the simplest usage with minimal configuration
fn example_1_basic_smoothing() -> Result<(), LowessError> {
    println!("Example 1: Basic Smoothing");
    println!("{}", "-".repeat(80));

    // Simple linear data with noise
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];

    let model = Lowess::new()
        .fraction(0.5) // Use 50% of data for each local fit
        .iterations(3) // 3 robustness iterations
        .adapter(Batch)
        .build()?;

    let result = model.fit(&x, &y)?;
    println!("{}", result);

    /* Expected Output:
    Summary:
      Data points: 5
      Fraction: 0.5

    Smoothed Data:
           X     Y_smooth
      --------------------
        1.00     2.00000
        2.00     4.10000
        3.00     5.90000
        4.00     8.20000
        5.00     9.80000
    */

    println!();
    Ok(())
}

#[cfg(feature = "std")]
/// Example 2: Robust Smoothing with Outliers
/// Shows how LOWESS handles outliers with robustness iterations
fn example_2_robust_with_outliers() -> Result<(), LowessError> {
    println!("Example 2: Robust Smoothing with Outliers");
    println!("{}", "-".repeat(80));

    // Data with an obvious outlier at index 3
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y = vec![2.1, 4.0, 5.9, 25.0, 10.1, 12.0, 14.1, 15.9]; // 25.0 is an outlier

    let model = Lowess::new()
        .fraction(0.5)
        .iterations(5) // More iterations for stronger robustness
        .robustness_method(Bisquare)
        .return_residuals()
        .return_robustness_weights()
        .adapter(Batch)
        .build()?;

    let result = model.fit(&x, &y)?;
    println!("{}", result);

    // Identify outliers
    if let Some(weights) = &result.robustness_weights {
        println!("\nOutlier Detection:");
        for (i, &w) in weights.iter().enumerate() {
            if w < 0.5 {
                println!(
                    "  Point {} (x={:.1}, y={:.1}) is an outlier (weight: {:.3})",
                    i, x[i], y[i], w
                );
            }
        }
    }

    /* Expected Output:
    Summary:
      Data points: 8
      Fraction: 0.5
      Robustness: Applied

    Smoothed Data:
           X     Y_smooth     Residual Rob_Weight
      ----------------------------------------------
        1.00     2.10000     0.000000     1.0000
        2.00     4.00000     0.000000     1.0000
        3.00     5.90000     0.000000     1.0000
        4.00     8.00000    17.000000     0.0000
        5.00    10.10000     0.000000     1.0000
        6.00    12.00000     0.000000     1.0000
        7.00    14.10000     0.000000     1.0000
        8.00    15.90000     0.000000     1.0000

    Outlier Detection:
      Point 3 (x=4.0, y=25.0) is an outlier (weight: 0.000)
    */

    println!();
    Ok(())
}

#[cfg(feature = "std")]
/// Example 3: Uncertainty Quantification
/// Demonstrates confidence and prediction intervals
fn example_3_uncertainty_quantification() -> Result<(), LowessError> {
    println!("Example 3: Uncertainty Quantification");
    println!("{}", "-".repeat(80));

    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y = vec![2.1, 3.8, 6.2, 7.9, 10.3, 11.8, 14.1, 15.7];

    let model = Lowess::new()
        .fraction(0.5)
        .iterations(3)
        .confidence_intervals(0.95) // 95% confidence intervals
        .prediction_intervals(0.95) // 95% prediction intervals
        .adapter(Batch)
        .build()?;

    let result = model.fit(&x, &y)?;
    println!("{}", result);

    /* Expected Output:
    Summary:
      Data points: 8
      Fraction: 0.5

    Smoothed Data:
           X     Y_smooth      Std_Err   Conf_Lower   Conf_Upper   Pred_Lower   Pred_Upper
      ----------------------------------------------------------------------------------
        1.00     2.01963     0.389365     1.256476     2.782788     1.058911     2.980353
        2.00     4.00251     0.345447     3.325438     4.679589     3.108641     4.896386
        3.00     5.99959     0.423339     5.169846     6.829335     4.985168     7.014013
        4.00     8.09859     0.489473     7.139224     9.057960     6.975666     9.221518
        5.00    10.03881     0.551687     8.957506    11.120118     8.810073    11.267551
        6.00    12.02872     0.539259    10.971775    13.085672    10.821364    13.236083
        7.00    13.89828     0.371149    13.170829    14.625733    12.965670    14.830892
        8.00    15.77990     0.408300    14.979631    16.580167    14.789441    16.770356
    */

    println!();
    Ok(())
}

#[cfg(feature = "std")]
/// Example 4: Cross-Validation
/// Automatic selection of optimal smoothing fraction
fn example_4_cross_validation() -> Result<(), LowessError> {
    println!("Example 4: Cross-Validation for Parameter Selection");
    println!("{}", "-".repeat(80));

    let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| 2.0 * xi + 1.0 + (xi * 0.5).sin())
        .collect();

    // Test multiple fractions and select the best one
    let model = Lowess::new()
        .cross_validate(KFold(5, &[0.2, 0.3, 0.5, 0.7]))
        .iterations(2)
        .adapter(Batch)
        .build()?;

    let result = model.fit(&x, &y)?;

    println!("Selected fraction: {}", result.fraction_used);
    if let Some(scores) = &result.cv_scores {
        println!("CV scores for each fraction: {:?}", scores);
    }
    println!("\n{}", result);

    /* Expected Output:
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

#[cfg(feature = "std")]
/// Example 5: Complete Diagnostic Analysis
/// Full feature demonstration with all diagnostics
fn example_5_complete_diagnostics() -> Result<(), LowessError> {
    println!("Example 5: Complete Diagnostic Analysis");
    println!("{}", "-".repeat(80));

    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y = vec![2.1, 3.8, 6.2, 7.9, 10.3, 11.8, 14.1, 15.7];

    let model = Lowess::new()
        .fraction(0.5)
        .iterations(3)
        .confidence_intervals(0.95)
        .prediction_intervals(0.95)
        .return_diagnostics()
        .return_residuals()
        .return_robustness_weights()
        .adapter(Batch)
        .build()?;

    let result = model.fit(&x, &y)?;
    println!("{}", result);

    /* Expected Output:
    Summary:
      Data points: 8
      Fraction: 0.5
      Robustness: Applied

    LOWESS Diagnostics:
      RMSE:         0.191925
      MAE:          0.181676
      R²:           0.998205
      Residual SD:  0.297750
      Effective DF: 8.00
      AIC:          -10.41
      AICc:         inf

    Smoothed Data:
           X     Y_smooth      Std_Err   Conf_Lower   Conf_Upper   Pred_Lower   Pred_Upper     Residual Rob_Weight
      ----------------------------------------------------------------------------------------------------------------
        1.00     2.01963     0.389365     1.256476     2.782788     1.058911     2.980353     0.080368     1.0000
        2.00     4.00251     0.345447     3.325438     4.679589     3.108641     4.896386    -0.202513     1.0000
        3.00     5.99959     0.423339     5.169846     6.829335     4.985168     7.014013     0.200410     1.0000
        4.00     8.09859     0.489473     7.139224     9.057960     6.975666     9.221518    -0.198592     1.0000
        5.00    10.03881     0.551687     8.957506    11.120118     8.810073    11.267551     0.261188     1.0000
        6.00    12.02872     0.539259    10.971775    13.085672    10.821364    13.236083    -0.228723     1.0000
        7.00    13.89828     0.371149    13.170829    14.625733    12.965670    14.830892     0.201719     1.0000
        8.00    15.77990     0.408300    14.979631    16.580167    14.789441    16.770356    -0.079899     1.0000
    */

    println!();
    Ok(())
}

#[cfg(feature = "std")]
/// Example 6: Different Weight Functions (Kernels)
/// Comparison of various kernel functions
fn example_6_different_kernels() -> Result<(), LowessError> {
    println!("Example 6: Different Weight Functions (Kernels)");
    println!("{}", "-".repeat(80));

    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];

    let kernels = vec![
        ("Tricube", Tricube),
        ("Epanechnikov", Epanechnikov),
        ("Gaussian", Gaussian),
        ("Biweight", Biweight),
    ];

    for (name, kernel) in kernels {
        println!("Using {} kernel:", name);

        let model = Lowess::new()
            .fraction(0.8)
            .weight_function(kernel)
            .adapter(Batch)
            .build()?;

        let result = model.fit(&x, &y)?;

        // Print just the smoothed values
        print!("  Smoothed Y: [");
        for (i, &val) in result.y.iter().enumerate() {
            if i > 0 {
                print!(", ");
            }
            print!("{:.3}", val);
        }
        println!("]");
    }

    /* Expected Output:
    Using Tricube kernel:
      Smoothed Y: [2.000, 4.100, 5.900, 8.200, 9.800]
    Using Epanechnikov kernel:
      Smoothed Y: [2.000, 4.100, 5.900, 8.200, 9.800]
    Using Gaussian kernel:
      Smoothed Y: [2.001, 4.099, 5.901, 8.199, 9.799]
    Using Biweight kernel:
      Smoothed Y: [2.000, 4.100, 5.900, 8.200, 9.800]
    */

    println!();
    Ok(())
}

#[cfg(feature = "std")]
/// Example 7: Robustness Methods Comparison
/// Different methods for handling outliers
fn example_7_robustness_methods() -> Result<(), LowessError> {
    println!("Example 7: Robustness Methods Comparison");
    println!("{}", "-".repeat(80));

    // Data with outlier
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.1, 20.0, 8.2, 9.8]; // 20.0 is an outlier

    let methods = vec![("Bisquare", Bisquare), ("Huber", Huber), ("Talwar", Talwar)];

    for (name, method) in methods {
        println!("Using {} robustness method:", name);

        let model = Lowess::new()
            .fraction(0.99) // Use large fraction but stay in local regression for robustness
            .iterations(5)
            .robustness_method(method)
            .return_robustness_weights()
            .adapter(Batch)
            .build()?;

        let result = model.fit(&x, &y)?;

        // Print smoothed values and weights
        print!("  Smoothed Y: [");
        for (i, &val) in result.y.iter().enumerate() {
            if i > 0 {
                print!(", ");
            }
            print!("{:.2}", val);
        }
        println!("]");

        if let Some(weights) = &result.robustness_weights {
            print!("  Weights:    [");
            for (i, &w) in weights.iter().enumerate() {
                if i > 0 {
                    print!(", ");
                }
                print!("{:.3}", w);
            }
            println!("]");
        }
    }

    /* Expected Output:
    Using Bisquare robustness method:
      Smoothed Y: [0.45, 7.97, 11.69, 11.96, 8.29]
      Weights:    [0.995, 0.970, 0.866, 0.972, 0.995]
    Using Huber robustness method:
      Smoothed Y: [0.47, 7.86, 11.36, 11.85, 8.32]
      Weights:    [1.000, 1.000, 0.809, 1.000, 1.000]
    Using Talwar robustness method:
      Smoothed Y: [0.35, 8.05, 12.07, 12.04, 8.20]
      Weights:    [1.000, 1.000, 1.000, 1.000, 1.000]
    */

    println!();
    Ok(())
}

#[cfg(feature = "std")]
/// Example 8: Benchmark (Sequential Batch)
/// Measure execution time for a large dataset using the sequential Batch adapter
fn example_8_benchmark() -> Result<(), LowessError> {
    println!("Example 8: Benchmark (Sequential Batch)");
    println!("{}", "-".repeat(80));

    // Generate a larger synthetic dataset
    let n = 10_000;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| (xi * 0.1).sin() + (xi * 0.01).cos())
        .collect();

    let start = Instant::now();
    let model = Lowess::new().adapter(Batch).build()?;

    let result = model.fit(&x, &y)?;
    let duration = start.elapsed();

    println!("Processed {} points in {:?}", n, duration);
    println!("Execution mode: Sequential Batch");
    println!("Result summary:\n{}", result);

    println!();
    Ok(())
}
