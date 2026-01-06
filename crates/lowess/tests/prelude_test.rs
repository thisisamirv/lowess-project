#![cfg(feature = "dev")]
//! Tests for the prelude module.
//!
//! These tests verify that the prelude exports all necessary types and traits
//! for convenient usage of the LOWESS API. The prelude should provide a
//! one-stop import for common LOWESS functionality.
//!
//! ## Test Organization
//!
//! 1. **Import Verification** - All prelude exports are accessible
//! 2. **Type Usage** - Types can be used without qualification
//! 3. **Builder Pattern** - Complete workflows work with prelude imports

use lowess::prelude::*;

// ============================================================================
// Import Verification Tests
// ============================================================================

/// Test that all prelude imports work correctly.
///
/// Verifies that the prelude exports all necessary types for LOWESS usage.
#[test]
fn test_prelude_imports() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

    // Verify Lowess (LowessBuilder), Adapter variants, and Result are useable
    let result = Lowess::new().adapter(Batch).build().unwrap().fit(&x, &y);

    assert!(result.is_ok(), "Basic fit should work with prelude imports");
}

/// Test RobustnessMethod is available.
///
/// Verifies that RobustnessMethod enum is exported.
#[test]
fn test_prelude_robustness_method() {
    let _ = Lowess::<f64>::new().robustness_method(Bisquare);
    let _ = Lowess::<f64>::new().robustness_method(Huber);
    let _ = Lowess::<f64>::new().robustness_method(Talwar);
}

/// Test WeightFunction is available.
///
/// Verifies that WeightFunction enum is exported.
#[test]
fn test_prelude_weight_function() {
    let _ = Lowess::<f64>::new().weight_function(Tricube);
    let _ = Lowess::<f64>::new().weight_function(Epanechnikov);
    let _ = Lowess::<f64>::new().weight_function(Gaussian);
    let _ = Lowess::<f64>::new().weight_function(Biweight);
}

/// Test CrossValidationStrategy is available.
///
/// Verifies that CrossValidationStrategy enum is exported.
#[test]
fn test_prelude_cross_validation() {
    let _ = Lowess::<f64>::new().cross_validate(KFold(5, &[0.5]));
    let _ = Lowess::<f64>::new().cross_validate(LOOCV(&[0.5]));
}

/// Test ZeroWeightFallback is available.
///
/// Verifies that ZeroWeightFallback enum is exported.
#[test]
fn test_prelude_zero_weight_fallback() {
    let _ = Lowess::<f64>::new().zero_weight_fallback(UseLocalMean);
    let _ = Lowess::<f64>::new().zero_weight_fallback(ReturnOriginal);
    let _ = Lowess::<f64>::new().zero_weight_fallback(ReturnNone);
}

/// Test adapter types are available.
///
/// Verifies that all adapter types are exported.
#[test]
fn test_prelude_adapters() {
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![2.0, 4.0, 6.0];

    // Batch adapter
    let _ = Lowess::<f64>::new()
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y);

    // Streaming adapter
    let _ = Lowess::<f64>::new().adapter(Streaming).build();

    // Online adapter
    let _ = Lowess::<f64>::new().adapter(Online).build();
}

/// Test complete workflow with prelude.
///
/// Verifies that a complete LOWESS workflow works with only prelude imports.
#[test]
fn test_prelude_complete_workflow() {
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0];

    let result = Lowess::<f64>::new()
        .fraction(0.5)
        .iterations(3)
        .robustness_method(Bisquare)
        .weight_function(Tricube)
        .confidence_intervals(0.95)
        .return_diagnostics()
        .return_residuals()
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .expect("Complete workflow should succeed");

    // Verify all requested outputs are present
    assert_eq!(result.y.len(), x.len());
    assert!(result.has_confidence_intervals());
    assert!(result.diagnostics.is_some());
    assert!(result.residuals.is_some());
}

/// Test error types are available.
///
/// Verifies that error handling works with prelude imports.
#[test]
fn test_prelude_error_handling() {
    let x: Vec<f64> = vec![];
    let y: Vec<f64> = vec![];

    let result = Lowess::<f64>::new()
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y);

    // Should be able to match on error types from prelude
    assert!(result.is_err());
}
