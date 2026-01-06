#![cfg(feature = "dev")]
//! Tests for the Batch adapter.
//!
//! The Batch adapter is the standard execution mode for LOWESS smoothing,
//! supporting all features including:
//! - Robust iterations with configurable robustness methods
//! - Confidence and prediction intervals
//! - Cross-validation for parameter selection
//! - Diagnostic metrics (RMSE, MAE, R², etc.)
//! - Auto-convergence detection
//!
//! ## Test Organization
//!
//! 1. **Basic Functionality** - Core smoothing behavior
//! 2. **Intervals and Diagnostics** - Standard errors, confidence/prediction intervals
//! 3. **Cross-Validation** - Automatic parameter selection (K-fold, LOOCV)
//! 4. **Auto-Convergence** - Automatic iteration stopping
//! 5. **Edge Cases** - Boundary conditions and error handling

use approx::assert_relative_eq;
use lowess::prelude::*;
use num_traits::float::Float;

use lowess::internals::adapters::batch::BatchLowessBuilder;
use lowess::internals::math::boundary::BoundaryPolicy;

// ============================================================================
// Basic Functionality Tests
// ============================================================================

/// Test basic LOWESS smoothing with the Batch adapter.
///
/// Verifies:
/// - Correct output length
/// - All values are finite
/// - Fraction is correctly applied
/// - Mean preservation (approximately)
#[test]
fn test_batch_basic_smoothing() {
    let n = 50;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|v| 2.0 * v + (v * 0.1).sin()).collect();

    let result = Lowess::new()
        .fraction(0.25)
        .iterations(2)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .expect("Batch smoothing should succeed");

    // Verify output properties
    assert_eq!(result.y.len(), y.len(), "Output length should match input");
    assert!(
        result.y.iter().all(|v| v.is_finite()),
        "All smoothed values should be finite"
    );
    assert_eq!(
        result.fraction_used, 0.25,
        "Fraction should be as specified"
    );

    // Check mean preservation (LOWESS approximately preserves mean)
    let mean_input = y.iter().sum::<f64>() / n as f64;
    let mean_output = result.y.iter().sum::<f64>() / n as f64;
    assert_relative_eq!(
        mean_output,
        mean_input,
        max_relative = 1e-3,
        epsilon = 1e-10
    );
}

/// Test LOWESS with robustness iterations.
///
/// Verifies that robustness weights are computed and returned when requested.
#[test]
fn test_batch_with_robustness_weights() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.1, 20.0, 8.2, 9.8]; // Point 2 is an outlier

    let result = Lowess::new()
        .fraction(0.5)
        .iterations(5)
        .robustness_method(Bisquare)
        .return_robustness_weights()
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .expect("Robust smoothing should succeed");

    // Verify robustness weights are returned
    assert!(
        result.robustness_weights.is_some(),
        "Robustness weights should be returned when requested"
    );

    // Verify weights are in [0, 1]
    let weights = result.robustness_weights.as_ref().expect("weights");
    assert_eq!(weights.len(), x.len());
    assert!(
        weights.iter().all(|&w| (0.0..=1.0).contains(&w)),
        "Robustness weights should be in [0, 1]"
    );
}

// ============================================================================
// Intervals and Diagnostics Tests
// ============================================================================

/// Test standard error computation.
///
/// Verifies that standard errors are computed when confidence intervals are requested.
#[test]
fn test_batch_standard_errors() {
    let n = 30;
    let x: Vec<f64> = (0..n).map(|i| i as f64 / 2.0).collect();
    let y: Vec<f64> = x.iter().map(|v| v * v).collect();

    let result = Lowess::new()
        .fraction(0.3)
        .confidence_intervals(0.95)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .expect("Smoothing with confidence intervals should succeed");

    // Verify standard errors are computed
    assert!(
        result.standard_errors.is_some(),
        "Standard errors should be computed for confidence intervals"
    );

    let se = result.standard_errors.unwrap();
    assert_eq!(se.len(), n, "SE count should match input");
    assert!(
        se.iter().all(|v| *v >= 0.0),
        "All standard errors should be non-negative"
    );
}

/// Test confidence and prediction intervals.
///
/// Verifies that both types of intervals are computed correctly.
#[test]
fn test_batch_confidence_and_prediction_intervals() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y = vec![2.1, 3.8, 6.2, 7.9, 10.3, 11.8, 14.1, 15.7];

    let result = Lowess::new()
        .fraction(0.5)
        .confidence_intervals(0.95)
        .prediction_intervals(0.95)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .expect("Smoothing with intervals should succeed");

    // Verify confidence intervals
    assert!(result.confidence_lower.is_some(), "CI lower should exist");
    assert!(result.confidence_upper.is_some(), "CI upper should exist");

    let ci_lower = result.confidence_lower.as_ref().unwrap();
    let ci_upper = result.confidence_upper.as_ref().unwrap();

    // Verify prediction intervals
    assert!(result.prediction_lower.is_some(), "PI lower should exist");
    assert!(result.prediction_upper.is_some(), "PI upper should exist");

    let pi_lower = result.prediction_lower.as_ref().unwrap();
    let pi_upper = result.prediction_upper.as_ref().unwrap();

    // Verify interval properties
    for i in 0..x.len() {
        // Confidence intervals should contain the smoothed value
        assert!(
            ci_lower[i] <= result.y[i] && result.y[i] <= ci_upper[i],
            "CI should contain smoothed value at index {}",
            i
        );

        // Prediction intervals should be wider than confidence intervals
        assert!(
            pi_lower[i] <= ci_lower[i],
            "PI lower should be <= CI lower at index {}",
            i
        );
        assert!(
            pi_upper[i] >= ci_upper[i],
            "PI upper should be >= CI upper at index {}",
            i
        );
    }
}

/// Test diagnostic metrics computation.
///
/// Verifies that RMSE, MAE, R², and other diagnostics are computed correctly.
#[test]
fn test_batch_diagnostics() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];

    let result = Lowess::new()
        .fraction(0.5)
        .return_diagnostics()
        .return_residuals()
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .expect("Smoothing with diagnostics should succeed");

    // Verify diagnostics are returned
    assert!(
        result.diagnostics.is_some(),
        "Diagnostics should be returned when requested"
    );

    let diag = result.diagnostics.unwrap();

    // Verify diagnostic values are reasonable
    assert!(diag.rmse >= 0.0, "RMSE should be non-negative");
    assert!(diag.mae >= 0.0, "MAE should be non-negative");
    assert!(
        diag.r_squared >= 0.0 && diag.r_squared <= 1.0,
        "R² should be in [0, 1]"
    );
    assert!(
        diag.residual_sd >= 0.0,
        "Residual SD should be non-negative"
    );

    // Verify residuals are returned
    assert!(
        result.residuals.is_some(),
        "Residuals should be returned when requested"
    );
    assert_eq!(
        result.residuals.unwrap().len(),
        x.len(),
        "Residual count should match input"
    );
}

// ============================================================================
// Cross-Validation Tests
// ============================================================================

/// Test K-fold cross-validation for fraction selection.
///
/// Verifies that CV selects one of the candidate fractions and returns scores.
#[test]
fn test_batch_cv_kfold() {
    let x = (0..10).map(|v| v as f64).collect::<Vec<_>>();
    let y = x.iter().map(|v| v * v + 0.1).collect::<Vec<_>>();
    let fractions = vec![0.2, 0.5, 0.8];

    let result = Lowess::new()
        .cross_validate(KFold(3, &fractions))
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .expect("CV should succeed");

    // Verify CV scores are returned
    assert!(
        result.cv_scores.is_some(),
        "CV scores should be returned for each fraction"
    );
    assert_eq!(
        result.cv_scores.as_ref().unwrap().len(),
        fractions.len(),
        "Should have one score per fraction"
    );

    // Verify selected fraction is one of the candidates
    assert!(
        fractions.contains(&result.fraction_used),
        "Selected fraction should be from candidates"
    );
}

/// Test cross-validation reproducibility with seeds.
#[test]
fn test_batch_cv_reproducibility() {
    let x = (0..20).map(|v| v as f64).collect::<Vec<_>>();
    let y = x.iter().map(|v| v.sin() + 0.1).collect::<Vec<_>>();
    let fractions = vec![0.3, 0.5, 0.7];
    let seed = 12345u64;

    // Run 1
    let result1 = Lowess::new()
        .cross_validate(KFold(5, &fractions).seed(seed))
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    // Run 2 (same seed)
    let result2 = Lowess::new()
        .cross_validate(KFold(5, &fractions).seed(seed))
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    // Run 3 (different seed)
    let result3 = Lowess::new()
        .cross_validate(KFold(5, &fractions).seed(seed + 1))
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    // Run 1 and 2 MUST be identical
    assert_eq!(
        result1.cv_scores, result2.cv_scores,
        "Folds should be identical with same seed"
    );
    assert_eq!(result1.fraction_used, result2.fraction_used);

    // Run 1 and 3 SHOULD be different (high probability with shuffling)
    assert_ne!(
        result1.cv_scores, result3.cv_scores,
        "Folds should be different with different seeds"
    );
}

/// Test leave-one-out cross-validation (LOOCV).
///
/// Verifies that LOOCV works correctly for small datasets.
#[test]
fn test_batch_cv_loocv() {
    let x = vec![0.0, 1.0, 2.0, 3.0];
    let y = vec![0.0, 1.0, 4.0, 9.0];
    let fractions = vec![0.5, 0.8];

    let result = Lowess::new()
        .cross_validate(LOOCV(&fractions))
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .expect("LOOCV should succeed");

    // Verify selected fraction is one of the candidates
    assert!(
        fractions.contains(&result.fraction_used),
        "Selected fraction should be from candidates"
    );

    // Verify CV scores are returned
    assert!(
        result.cv_scores.is_some(),
        "CV scores should be returned for LOOCV"
    );
}

// ============================================================================
// Auto-Convergence Tests
// ============================================================================

/// Test automatic convergence detection.
///
/// Verifies that iterations stop early when convergence is detected.
#[test]
fn test_batch_auto_convergence() {
    let x = vec![0.0, 1.0, 2.0];
    let y = vec![1.0, 2.0, 3.0]; // Perfect line - should converge quickly

    let result = Lowess::new()
        .fraction(0.5)
        .iterations(5)
        .auto_converge(1e-12)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .expect("Auto-convergence should succeed");

    // Verify iterations_used is returned
    assert!(
        result.iterations_used.is_some(),
        "Iterations used should be tracked with auto-convergence"
    );

    let iters = result.iterations_used.unwrap();

    // Should converge before max iterations for perfect line
    assert!(
        iters <= 5,
        "Should converge within max iterations (used: {})",
        iters
    );
}

/// Test that auto-convergence respects max_iterations.
///
/// Verifies that iterations don't exceed the specified maximum.
#[test]
fn test_batch_auto_convergence_max_iterations() {
    let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&v| v + (v * 0.5).sin() * 5.0).collect();

    let max_iters = 3;
    let result = Lowess::new()
        .fraction(0.3)
        .iterations(max_iters)
        .auto_converge(1e-10)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .expect("Auto-convergence with max should succeed");

    assert!(result.iterations_used.is_some());
    let iters = result.iterations_used.unwrap();

    assert!(
        iters <= max_iters,
        "Should not exceed max iterations (max: {}, used: {})",
        max_iters,
        iters
    );
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

/// Test BatchLowessBuilder default values.
#[test]
fn test_batch_builder_defaults() {
    let b = BatchLowessBuilder::<f64>::default();
    assert_eq!(b.iterations, 3);
}

/// Test BatchLowessBuilder setters.
#[test]
fn test_batch_builder_setters() {
    let b = BatchLowessBuilder::<f64>::default().boundary_policy(BoundaryPolicy::Extend);
    assert_eq!(b.boundary_policy, BoundaryPolicy::Extend);
}

/// Test with minimum viable dataset (3 points).
///
/// Verifies that LOWESS works with very small datasets.
#[test]
fn test_batch_minimum_dataset() {
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![1.0, 2.0, 3.0];

    let result = Lowess::new()
        .fraction(0.67)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .expect("Should work with 3 points");

    assert_eq!(result.y.len(), 3);
    assert!(result.y.iter().all(|v| v.is_finite()));
}

/// Test with all features enabled simultaneously.
///
/// Verifies that all features can be used together without conflicts.
#[test]
fn test_batch_all_features_combined() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y = vec![2.1, 3.8, 6.2, 7.9, 10.3, 11.8, 14.1, 15.7];

    let result = Lowess::new()
        .fraction(0.5)
        .iterations(3)
        .weight_function(Tricube)
        .robustness_method(Bisquare)
        .confidence_intervals(0.95)
        .prediction_intervals(0.95)
        .return_diagnostics()
        .return_residuals()
        .return_robustness_weights()
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .expect("All features should work together");

    // Verify all requested outputs are present
    assert!(result.standard_errors.is_some(), "SE should be present");
    assert!(
        result.confidence_lower.is_some(),
        "CI lower should be present"
    );
    assert!(
        result.confidence_upper.is_some(),
        "CI upper should be present"
    );
    assert!(
        result.prediction_lower.is_some(),
        "PI lower should be present"
    );
    assert!(
        result.prediction_upper.is_some(),
        "PI upper should be present"
    );
    assert!(
        result.diagnostics.is_some(),
        "Diagnostics should be present"
    );
    assert!(result.residuals.is_some(), "Residuals should be present");
    assert!(
        result.robustness_weights.is_some(),
        "Robustness weights should be present"
    );
}

// ============================================================================
// Additional Edge Cases
// ============================================================================

/// Test with large unsorted dataset.
#[test]
fn test_batch_unsorted_data_large() {
    // Create large dataset in reverse order
    let n = 100;
    let x: Vec<f64> = (0..n).map(|i| (n - i - 1) as f64).collect();
    let y: Vec<f64> = (0..n).map(|i| ((n - i - 1) * 2) as f64).collect();

    let result = Lowess::new()
        .fraction(0.3)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    // Result should be in original (unsorted) order
    assert_eq!(result.x.len(), n);
    assert_eq!(result.y.len(), n);

    // Verify x values match original order
    for (i, &x_val) in x.iter().enumerate() {
        assert_eq!(result.x[i], x_val);
    }
}

/// Test with all identical x-values (degenerate case).
#[test]
fn test_batch_all_identical_x() {
    let x = vec![5.0; 10];
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

    let result = Lowess::new()
        .fraction(0.5)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    // With identical x-values, LOWESS handles this as a degenerate case
    // All smoothed values should be finite and within the range of y values
    for &smoothed in &result.y {
        assert!(smoothed.is_finite(), "Smoothed value should be finite");
        assert!(
            (0.0..=11.0).contains(&smoothed),
            "Smoothed value should be in reasonable range: {}",
            smoothed
        );
    }
}

/// Test with all identical y-values (zero variance).
#[test]
fn test_batch_all_identical_y() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![10.0; 5];

    let result = Lowess::new()
        .fraction(0.5)
        .iterations(3)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    // All smoothed values should be 10.0
    for &smoothed in &result.y {
        assert_relative_eq!(smoothed, 10.0, epsilon = 1e-6);
    }
}

/// Test with fraction exactly 1.0 (global regression).
#[test]
fn test_batch_fraction_exactly_one() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

    let result = Lowess::new()
        .fraction(1.0)
        .iterations(0)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    // With fraction=1.0, should perform global linear regression
    // For perfect linear data, should reproduce exactly
    for (result_val, y_val) in result.y.iter().zip(y.iter()) {
        assert_relative_eq!(result_val, y_val, epsilon = 1e-6);
    }
}

/// Test delta=0.0 vs auto-computed delta.
#[test]
fn test_batch_delta_zero_vs_auto() {
    let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi * 2.0 + 1.0).collect();

    // Test with delta=0.0 (no optimization)
    let result_no_delta = Lowess::new()
        .fraction(0.3)
        .delta(0.0)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    // Test with auto delta (should use optimization)
    let result_auto_delta = Lowess::new()
        .fraction(0.3)
        .adapter(Batch)
        // delta not set, will auto-compute
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    // Results should be similar but not necessarily identical
    assert_eq!(result_no_delta.y.len(), result_auto_delta.y.len());

    // Check that results are reasonably close
    for (i, _) in x.iter().enumerate() {
        assert_relative_eq!(result_no_delta.y[i], result_auto_delta.y[i], epsilon = 0.5);
    }
}

/// Test with extreme outliers.
#[test]
fn test_batch_extreme_outliers() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let mut y = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];

    // Add extreme outliers
    y[2] = 100.0; // Extreme high
    y[7] = -100.0; // Extreme low

    let result = Lowess::new()
        .fraction(0.5)
        .iterations(3) // Use robustness to handle outliers
        .return_robustness_weights()
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    // Robustness should downweight outliers
    // Check that smoothed values are reasonable (not pulled to extremes)
    for &smoothed in &result.y {
        assert!(
            smoothed > -50.0 && smoothed < 50.0,
            "Smoothed value should not be extreme: {}",
            smoothed
        );
    }

    // Robustness weights for outliers should be lower
    if let Some(weights) = result.robustness_weights {
        assert!(weights[2] < 0.5, "Outlier should have low weight");
        assert!(weights[7] < 0.5, "Outlier should have low weight");
    }
}
