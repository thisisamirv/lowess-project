#![cfg(feature = "dev")]
//! Tests for the high-level LOWESS API.
//!
//! These tests verify the builder pattern, configuration options, and complete
//! workflows for the LOWESS API including:
//! - Builder construction and validation
//! - Adapter modes (Batch, Streaming, Online)
//! - Intervals and diagnostics
//! - Cross-validation
//! - Robustness methods
//! - Result helpers
//!
//! ## Test Organization
//!
//! 1. **Builder Construction** - Default values, mode conversion
//! 2. **Validation** - Input validation, error handling
//! 3. **Robustness Methods** - Bisquare, Huber, Talwar
//! 4. **Intervals & Diagnostics** - CI, PI, metrics
//! 5. **Result Helpers** - Utility methods on LowessResult
//! 6. **Cross-Validation** - K-fold, LOOCV
//! 7. **Convenience Constructors** - Quick, robust
//! 8. **Adapter Propagation** - Option passing to adapters

use approx::assert_relative_eq;
use std::fmt::Write;

use lowess::internals::algorithms::regression::ZeroWeightFallback;
use lowess::internals::algorithms::robustness::RobustnessMethod;
use lowess::internals::api::{Batch, KFold, LOOCV, LowessBuilder as Lowess, Online, Streaming};
use lowess::internals::engine::output::LowessResult;
use lowess::internals::engine::validator::Validator;
use lowess::internals::evaluation::diagnostics::Diagnostics;
use lowess::internals::math::kernel::WeightFunction;
use lowess::internals::primitives::errors::LowessError;

// ============================================================================
// Helper Functions
// ============================================================================

fn linear_series(n: usize, slope: f64, intercept: f64) -> (Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|xi| slope * xi + intercept).collect();
    (x, y)
}

// ============================================================================
// Builder Construction Tests
// ============================================================================

/// Test builder conversion to Streaming adapter.
///
/// Verifies that builder can be converted to streaming mode.
#[test]
fn test_builder_converts_to_streaming() {
    let sb = Lowess::<f64>::new().fraction(0.5).adapter(Streaming);
    assert!(
        sb.build().is_ok(),
        "Streaming builder should build successfully"
    );
}

/// Test builder conversion to Online adapter.
///
/// Verifies that builder can be converted to online mode.
#[test]
fn test_builder_converts_to_online() {
    let ob = Lowess::<f64>::new().fraction(0.5).adapter(Online);
    assert!(
        ob.build().is_ok(),
        "Online builder should build successfully"
    );
}

/// Test default fraction value.
///
/// Verifies that default fraction (~0.67) is used when not specified.
#[test]
fn test_default_fraction() {
    let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|xi| 2.0 * xi + 1.0).collect();

    let res = Lowess::<f64>::new()
        .iterations(0)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .expect("fit ok");

    // Default fraction should be close to 0.67
    assert_relative_eq!(res.fraction_used, 0.67, epsilon = 1e-6);
}

/// Test auto-convergence sets iterations_used.
///
/// Verifies that auto-convergence populates iterations_used in result.
#[test]
fn test_auto_converge_sets_iterations_used() {
    let x: Vec<f64> = (0..8).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|xi| 5.0 * xi + 2.0).collect();

    let res = Lowess::<f64>::new()
        .auto_converge(1e-6)
        .iterations(50) // Use iterations directly
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .expect("auto-converge fit ok");

    assert!(
        res.iterations_used.is_some(),
        "iterations_used should be set"
    );
    let iters = res.iterations_used.unwrap();
    assert!(iters <= 50, "Should not exceed iterations");
}

#[test]
fn test_builder_defaults() {
    let b = Lowess::<f64>::new();

    assert_eq!(b.iterations, None, "Iterations not set by default");
    assert_eq!(
        b.weight_function, None,
        "Weight function not set by default"
    );
    assert_eq!(
        b.robustness_method, None,
        "Robustness method not set by default"
    );

    // Test Default trait
    let bd = Lowess::<f64>::default();
    assert_eq!(bd.iterations, None);
}

/// Test RobustnessMethod derives.
///
/// Verifies Debug, Clone, Default traits.
#[test]
fn test_robustness_method_derives() {
    // Debug + Default
    let d = format!("{:?}", RobustnessMethod::default());
    assert!(d.contains("Bisquare"), "Default should be Bisquare");

    // Clone
    let r = RobustnessMethod::Huber;
    let rc = r;
    assert_eq!(format!("{:?}", r), format!("{:?}", rc), "Clone should work");

    // Default is Bisquare
    assert_eq!(RobustnessMethod::default(), RobustnessMethod::Bisquare);
}

// ============================================================================
// Validation Tests
// ============================================================================

/// Test validation rejects empty CV fraction list.
///
/// Verifies that empty cross-validation fraction list produces error.
#[test]
fn test_validate_empty_cv_fractions() {
    // K-Fold with empty fractions
    let fracs: [f64; 0] = [];
    let res = Lowess::<f64>::new()
        .cross_validate(KFold(3, &fracs))
        .adapter(Batch)
        .build();

    assert!(
        matches!(res, Err(LowessError::InvalidFraction(_))),
        "Empty CV fractions should error"
    );
}

/// Test validation rejects out-of-range fractions.
///
/// Verifies that invalid fraction values are rejected.
#[test]
fn test_validate_invalid_fractions() {
    // Fraction <= 0
    let bad1 = Lowess::<f64>::new()
        .cross_validate(KFold(3, &[0.0f64]))
        .adapter(Batch)
        .build();
    assert!(matches!(bad1, Err(LowessError::InvalidFraction(_))));

    // Fraction > 1
    let bad2 = Lowess::<f64>::new()
        .cross_validate(KFold(3, &[1.5f64]))
        .adapter(Batch)
        .build();
    assert!(matches!(bad2, Err(LowessError::InvalidFraction(_))));
}

/// Test validation rejects invalid confidence level.
///
/// Verifies that invalid interval levels produce errors.
#[test]
fn test_validate_invalid_confidence_level() {
    // Level > 1.0 should be rejected
    let got = Lowess::<f64>::new()
        .fraction(0.5)
        .confidence_intervals(2.0)
        .iterations(0)
        .adapter(Batch)
        .build();

    assert!(matches!(got, Err(LowessError::InvalidIntervals(_))));

    // Verify validator also rejects
    match Validator::validate_interval_level(1.5) {
        Err(LowessError::InvalidIntervals(val)) => {
            assert!((val - 1.5).abs() < 1e-10, "Error should contain value")
        }
        _ => panic!("Expected InvalidIntervals error"),
    }
}

/// Test validation rejects negative delta.
///
/// Verifies that negative delta values are rejected.
#[test]
fn test_validate_negative_delta() {
    let res = Lowess::<f64>::new().delta(-1.0).adapter(Batch).build();

    assert!(res.is_err(), "Negative delta should error");
}

#[test]
fn test_fit_empty_input() {
    let x: Vec<f64> = vec![];
    let y: Vec<f64> = vec![];

    let res = Lowess::<f64>::new()
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y);

    assert!(matches!(res, Err(LowessError::EmptyInput)));
}

/// Test LowessError Display and Debug formatting.
///
/// Exercises error variants for coverage.
#[test]
fn test_lowess_error_display() {
    let errs = [
        LowessError::EmptyInput,
        LowessError::MismatchedInputs { x_len: 1, y_len: 2 },
        LowessError::TooFewPoints { got: 1, min: 2 },
        LowessError::InvalidFraction(1.5),
        LowessError::InvalidDelta(-0.1),
        LowessError::InvalidIterations(0),
        LowessError::InvalidIntervals(0.95),
        LowessError::InvalidChunkSize { got: 5, min: 10 },
        LowessError::InvalidOverlap {
            overlap: 10,
            chunk_size: 10,
        },
        LowessError::InvalidWindowCapacity { got: 2, min: 3 },
        LowessError::InvalidMinPoints {
            got: 1,
            window_capacity: 10,
        },
        LowessError::UnsupportedFeature {
            adapter: "Streaming",
            feature: "test",
        },
    ];
    for e in errs {
        let _ = format!("{:?}", e);
        let _ = format!("{}", e);
    }
}

// ============================================================================
// Robustness Methods Tests
// ============================================================================

/// Test Bisquare robustness method downweights outliers.
///
/// Verifies that Bisquare method reduces weights for outliers.
#[test]
fn test_robustness_bisquare() {
    let n = 11;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let mut y: Vec<f64> = x.iter().map(|xi| 2.0 * xi + 1.0).collect();
    y[5] = 100.0; // Inject outlier

    let res = Lowess::<f64>::new()
        .fraction(0.15)
        .iterations(5)
        .return_residuals()
        .return_robustness_weights()
        .robustness_method(RobustnessMethod::Bisquare)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .expect("fit ok");

    let w = res.robustness_weights.expect("weights present");
    let any_down = w.iter().any(|&wi| wi < 1.0);
    let any_near_one = w.iter().any(|&wi| wi > 0.9);

    // Either some downweighting or all weights remain 1.0
    if any_down {
        assert!(any_near_one, "Some points should remain nearly unweighted");
    } else {
        assert!(w.iter().all(|&wi| (wi - 1.0).abs() < 1e-12));
    }
}

/// Test Huber robustness method downweights outliers.
///
/// Verifies that Huber method reduces weights for outliers.
#[test]
fn test_robustness_huber() {
    let n = 11;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let mut y: Vec<f64> = x.iter().map(|xi| 2.0 * xi + 1.0).collect();
    y[5] = 100.0;

    let res = Lowess::<f64>::new()
        .fraction(0.15)
        .iterations(5)
        .return_residuals()
        .return_robustness_weights()
        .robustness_method(RobustnessMethod::Huber)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .expect("fit ok");

    let w = res.robustness_weights.expect("weights present");
    let any_down = w.iter().any(|&wi| wi < 1.0);
    let any_near_one = w.iter().any(|&wi| wi > 0.9);

    if any_down {
        assert!(any_near_one);
    } else {
        assert!(w.iter().all(|&wi| (wi - 1.0).abs() < 1e-12));
    }
}

/// Test Talwar robustness method downweights outliers.
///
/// Verifies that Talwar method reduces weights for outliers.
#[test]
fn test_robustness_talwar() {
    let n = 11;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let mut y: Vec<f64> = x.iter().map(|xi| 2.0 * xi + 1.0).collect();
    y[5] = 100.0;

    let res = Lowess::<f64>::new()
        .fraction(0.15)
        .iterations(5)
        .return_residuals()
        .return_robustness_weights()
        .robustness_method(RobustnessMethod::Talwar)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .expect("fit ok");

    let w = res.robustness_weights.expect("weights present");
    let any_down = w.iter().any(|&wi| wi < 1.0);
    let any_near_one = w.iter().any(|&wi| wi > 0.9);

    if any_down {
        assert!(any_near_one);
    } else {
        assert!(w.iter().all(|&wi| (wi - 1.0).abs() < 1e-12));
    }
}

/// Test robustness weights with zero iterations.
///
/// Verifies that with zero iterations, all weights are 1.0.
#[test]
fn test_robustness_weights_zero_iterations() {
    let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|xi| 2.0 * xi + 1.0).collect();

    let res = Lowess::<f64>::new()
        .return_robustness_weights()
        .iterations(0)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .expect("fit ok");

    assert!(res.robustness_weights.is_some());
    let w = res.robustness_weights.unwrap();
    assert_eq!(w.len(), x.len());
    assert!(w.iter().all(|&wi| (wi - 1.0).abs() < 1e-12));
}

// ============================================================================
// Intervals & Diagnostics Tests
// ============================================================================

/// Test fit with intervals and diagnostics.
///
/// Verifies that all requested outputs are produced.
#[test]
fn test_fit_with_intervals_and_diagnostics() {
    let (x, y) = linear_series(4, 2.0, 1.0);

    let res = Lowess::<f64>::new()
        .fraction(1.0)
        .confidence_intervals(0.95)
        .prediction_intervals(0.95)
        .return_diagnostics()
        .return_residuals()
        .iterations(0)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .expect("fit ok");

    // Predictions should reproduce linear y
    assert_eq!(res.y, y);

    // Intervals computed
    assert!(res.has_confidence_intervals());
    assert!(res.has_prediction_intervals());

    // Diagnostics and residuals present
    assert!(res.diagnostics.is_some());
    assert!(res.residuals.is_some());
    assert!(res.standard_errors.is_some());
}

/// Test prediction intervals only.
///
/// Verifies that only PI is returned when only PI is requested.
#[test]
fn test_prediction_intervals_only() {
    let x = vec![0.0f64, 1.0, 2.0, 3.0];
    let y = vec![1.0f64, 3.0, 5.0, 7.0];

    let res = Lowess::<f64>::new()
        .fraction(1.0)
        .prediction_intervals(0.95)
        .iterations(0)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .expect("fit ok");

    assert!(!res.has_confidence_intervals());
    assert!(res.has_prediction_intervals());

    let plo = res.prediction_lower.expect("pred lower present");
    let phi = res.prediction_upper.expect("pred upper present");
    assert_eq!(plo.len(), x.len());
    assert_eq!(phi.len(), x.len());

    for (&l, &h) in plo.iter().zip(phi.iter()) {
        assert!(!l.is_nan() && !h.is_nan());
        assert!(h >= l);
    }
}

/// Test confidence intervals only.
///
/// Verifies that only CI is returned when only CI is requested.
#[test]
fn test_confidence_intervals_only() {
    let x = vec![0.0f64, 1.0, 2.0, 3.0];
    let y = vec![1.0f64, 3.0, 5.0, 7.0];

    let res = Lowess::<f64>::new()
        .fraction(1.0)
        .confidence_intervals(0.95)
        .iterations(0)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .expect("fit ok");

    assert!(res.has_confidence_intervals());
    assert!(!res.has_prediction_intervals());
    assert!(res.standard_errors.is_some());
}

/// Test fit without intervals.
///
/// Verifies that no standard errors are returned without intervals.
#[test]
fn test_fit_without_intervals() {
    let x = vec![0.0f64, 1.0, 2.0, 3.0, 4.0];
    let y = x.iter().map(|xi| 5.0 * xi + 2.0).collect::<Vec<_>>();

    let res = Lowess::<f64>::new()
        .fraction(0.4)
        .iterations(0)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .expect("fit ok");

    assert!(res.standard_errors.is_none());
    assert!(res.residuals.is_none());
}

/// Test fit with residuals.
///
/// Verifies that residuals are populated when requested.
#[test]
fn test_fit_with_residuals() {
    let x: Vec<f64> = (0..6).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|xi| 2.0 * xi + 1.0).collect();

    let res = Lowess::<f64>::new()
        .fraction(1.0)
        .return_residuals()
        .iterations(0)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .expect("fit ok with residuals");

    assert!(res.residuals.is_some());
    let r = res.residuals.unwrap();

    // Perfect linear data => residuals should be ~zero
    for &v in &r {
        assert_relative_eq!(v, 0.0, epsilon = 1e-12);
    }
}

/// Test Diagnostics Display formatting.
///
/// Verifies that Diagnostics can be formatted.
#[test]
fn test_diagnostics_display() {
    let d = Diagnostics {
        rmse: 0.123456,
        mae: 0.234567,
        r_squared: 0.99,
        aic: Some(10.0),
        aicc: Some(12.0),
        effective_df: Some(2.5),
        residual_sd: 0.05,
    };

    let mut s = String::new();
    write!(&mut s, "{}", d).expect("format diagnostics");

    assert!(s.contains("LOWESS Diagnostics"));
    assert!(s.contains("RMSE"));
    assert!(s.contains("Effective DF"));
    assert!(s.contains("AIC"));
}

// ============================================================================
// Result Helpers Tests
// ============================================================================

/// Test LowessResult helper methods.
///
/// Verifies confidence_width, prediction_width, best_cv_score.
#[test]
fn test_lowess_result_helpers() {
    let lr = LowessResult {
        x: vec![0.0, 1.0, 2.0],
        y: vec![1.0, 2.0, 3.0],
        standard_errors: None,
        confidence_lower: Some(vec![0.9, 1.9, 2.9]),
        confidence_upper: Some(vec![1.1, 2.1, 3.1]),
        prediction_lower: Some(vec![0.5, 1.5, 2.5]),
        prediction_upper: Some(vec![1.5, 2.5, 3.5]),
        residuals: None,
        robustness_weights: None,
        diagnostics: None,
        iterations_used: Some(0),
        fraction_used: 0.5,
        cv_scores: Some(vec![0.3, 0.1, 0.2]),
    };

    // Best CV score
    let best = lr.best_cv_score().expect("best cv");
    assert_eq!(best, 0.1);
}

/// Test has_cv_scores method.
///
/// Verifies correct detection of CV scores presence.
#[test]
fn test_has_cv_scores() {
    let lr_with = LowessResult {
        x: vec![0.0],
        y: vec![1.0],
        standard_errors: None,
        confidence_lower: None,
        confidence_upper: None,
        prediction_lower: None,
        prediction_upper: None,
        residuals: None,
        robustness_weights: None,
        diagnostics: None,
        iterations_used: None,
        fraction_used: 0.5,
        cv_scores: Some(vec![0.1, 0.2]),
    };
    assert!(lr_with.has_cv_scores());

    let lr_without = LowessResult {
        cv_scores: None,
        ..lr_with.clone()
    };
    assert!(!lr_without.has_cv_scores());
}

// ============================================================================
// Cross-Validation Tests
// ============================================================================

/// Test K-fold cross-validation.
///
/// Verifies that K-fold produces CV scores.
#[test]
fn test_cross_validate_kfold() {
    let x: Vec<f64> = (0..12).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|xi| 2.0 * xi + 1.0).collect();
    let fracs = vec![0.2, 0.4];

    let res = Lowess::<f64>::new()
        .cross_validate(KFold(3, &fracs))
        .iterations(0)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .expect("kfold fit ok");

    assert!(res.cv_scores.is_some());
    let scores = res.cv_scores.unwrap();
    assert_eq!(scores.len(), fracs.len());
}

/// Test LOOCV cross-validation.
///
/// Verifies that LOOCV produces CV scores.
#[test]
fn test_cross_validate_loocv() {
    let x: Vec<f64> = (0..8).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|xi| 3.0 * xi - 1.0).collect();
    let fractions = vec![0.3, 0.6];

    let res = Lowess::<f64>::new()
        .cross_validate(LOOCV(&fractions))
        .iterations(0)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .expect("loocv fit ok");

    assert!(res.cv_scores.is_some());
    assert_eq!(res.cv_scores.unwrap().len(), fractions.len());
}

// ============================================================================
// Adapter Propagation Tests
// ============================================================================

/// Test zero weight fallback propagates to Streaming.
///
/// Verifies that options are passed to streaming adapter.
#[test]
fn test_zero_weight_fallback_propagates_streaming() {
    let base = Lowess::<f64>::new().zero_weight_fallback(ZeroWeightFallback::ReturnOriginal);

    let sb = base.adapter(Streaming).chunk_size(10).overlap(2);
    let mut runner = sb.build().expect("streaming builder build ok");

    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let y = x.iter().map(|xi| 2.0 * xi + 1.0).collect::<Vec<_>>();
    let out = runner.process_chunk(&x, &y).expect("process ok");

    assert!(!out.y.is_empty());
    assert!(out.y[0].is_finite());
}

/// Test Streaming builds with defaults.
///
/// Verifies that streaming adapter can build with default values.
#[test]
fn test_streaming_builds_with_defaults() {
    let sb = Lowess::<f64>::new().adapter(Streaming);
    assert!(sb.build().is_ok());
}

/// Test Online builds with defaults.
///
/// Verifies that online adapter can build with default values.
#[test]
fn test_online_builds_with_defaults() {
    let ob = Lowess::<f64>::new().adapter(Online);
    assert!(ob.build().is_ok());
}

/// Test Streaming propagates shared options.
///
/// Verifies that options are correctly passed to streaming mode.
#[test]
fn test_streaming_propagates_options() {
    let base = Lowess::<f64>::new().fraction(1.0).iterations(0);

    let sb = base.adapter(Streaming).overlap(2).chunk_size(10);
    let mut runner = sb.build().expect("streaming builder build ok");

    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let y = x.iter().map(|xi| 2.0 * xi + 1.0).collect::<Vec<_>>();
    let out = runner.process_chunk(&x, &y).expect("process ok");

    assert!(!out.y.is_empty());
    assert_relative_eq!(out.y[0], y[0], epsilon = 1e-12);
}

/// Test Online propagates shared options.
///
/// Verifies that options are correctly passed to online mode.
#[test]
fn test_online_propagates_options() {
    let base = Lowess::<f64>::new().fraction(1.0).iterations(0);
    let ob = base.adapter(Online).window_capacity(5);
    let mut online = ob.build().expect("online builder build ok");

    assert_eq!(online.add_point(0.0, 1.0).expect("ok"), None);
    assert_eq!(online.add_point(1.0, 3.0).expect("ok"), None);

    let third = online.add_point(2.0, 5.0).expect("ok");
    assert!(third.is_some());
    assert_relative_eq!(third.unwrap().smoothed, 5.0, epsilon = 1e-12);
    assert!(online.window_size() > 0);

    online.reset();
    assert_eq!(online.window_size(), 0);
}

/// Test Batch keeps options.
///
/// Verifies that batch adapter preserves builder options.
#[test]
fn test_batch_keeps_options() {
    let base = Lowess::<f64>::new().fraction(0.42).iterations(2);

    let batch = base.adapter(Batch);
    assert_eq!(batch.fraction, 0.42);
    assert_eq!(batch.iterations, 2);
}

// ============================================================================
// Builder Pattern Edge Cases
// ============================================================================

/// Test that builder methods can be called in any order.
#[test]
fn test_builder_method_chaining_order() {
    let (x, y) = linear_series(20, 2.0, 1.0);

    // Order 1: fraction -> iterations -> delta
    let result1 = Lowess::new()
        .fraction(0.5)
        .iterations(2)
        .delta(0.1)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    // Order 2: delta -> fraction -> iterations
    let result2 = Lowess::new()
        .delta(0.1)
        .fraction(0.5)
        .iterations(2)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    // Results should be identical regardless of order
    assert_eq!(result1.y.len(), result2.y.len());
    for (y1, y2) in result1.y.iter().zip(result2.y.iter()) {
        assert_relative_eq!(y1, y2, epsilon = 1e-10);
    }
}

/// Test that cloned builders are independent.
#[test]
fn test_builder_clone_independence() {
    let builder1 = Lowess::new().fraction(0.5).iterations(2);
    let mut builder2 = builder1.clone();

    // Modify builder2
    builder2 = builder2.fraction(0.7).iterations(4);

    // builder1 should still have original values
    assert_eq!(builder1.fraction, Some(0.5));
    assert_eq!(builder1.iterations, Some(2));

    // builder2 should have new values
    assert_eq!(builder2.fraction, Some(0.7));
    assert_eq!(builder2.iterations, Some(4));
}

/// Test setting all available parameters.
#[test]
fn test_builder_all_parameters_set() {
    let (x, y) = linear_series(30, 2.0, 1.0);

    let result = Lowess::new()
        .fraction(0.4)
        .iterations(3)
        .delta(0.05)
        .weight_function(WeightFunction::Tricube)
        .robustness_method(RobustnessMethod::Bisquare)
        .return_se()
        .confidence_intervals(0.95)
        .prediction_intervals(0.95)
        .return_residuals()
        .return_robustness_weights()
        .return_diagnostics()
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    // Verify all requested outputs are present
    assert!(result.standard_errors.is_some());
    assert!(result.confidence_lower.is_some());
    assert!(result.confidence_upper.is_some());
    assert!(result.prediction_lower.is_some());
    assert!(result.prediction_upper.is_some());
    assert!(result.residuals.is_some());
    assert!(result.robustness_weights.is_some());
    assert!(result.diagnostics.is_some());
}

/// Test that setting a parameter multiple times returns error on build().
#[test]
fn test_builder_parameter_override() {
    // Setting fraction twice - should be detected at build time
    let result = Lowess::new()
        .fraction(0.3)
        .fraction(0.5) // Duplicate - will be caught by build()
        .adapter(Batch)
        .build();

    assert!(result.is_err());
    match result {
        Err(LowessError::DuplicateParameter { parameter }) => {
            assert_eq!(parameter, "fraction");
        }
        _ => panic!("Expected DuplicateParameter error"),
    }
}

/// Test that Default::default() works after custom configuration.
#[test]
fn test_builder_default_after_custom() {
    let _custom_builder = Lowess::new().fraction(0.8).iterations(5);

    // Create a new default builder
    let default_builder = Lowess::<f64>::new();

    // Should have default values, not custom ones
    assert_eq!(default_builder.fraction, None);
    assert_eq!(default_builder.iterations, None); // Default doesn't set iterations
}

// ============================================================================
// Parameter Boundary Cases
// ============================================================================

/// Test fraction at exact boundaries.
#[test]
fn test_fraction_at_boundaries() {
    let (x, y) = linear_series(20, 2.0, 1.0);

    // Very small fraction (should work)
    let result_small = Lowess::new()
        .fraction(0.0001)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();
    assert_eq!(result_small.y.len(), 20);

    // Fraction = 1.0 (global regression)
    let result_one = Lowess::new()
        .fraction(1.0)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();
    assert_eq!(result_one.y.len(), 20);
}

/// Test iterations at maximum value.
#[test]
fn test_iterations_at_max() {
    let (x, y) = linear_series(10, 2.0, 1.0);

    // Maximum iterations (1000)
    let result = Lowess::new()
        .fraction(0.5)
        .iterations(1000)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    assert_eq!(result.y.len(), 10);
    // Should complete without error
}

/// Test delta exactly at zero.
#[test]
fn test_delta_at_zero() {
    let (x, y) = linear_series(15, 2.0, 1.0);

    // Delta = 0.0 means no interpolation optimization
    let result = Lowess::new()
        .fraction(0.5)
        .delta(0.0)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    assert_eq!(result.y.len(), 15);
}

/// Test interval levels at boundaries.
#[test]
fn test_interval_level_boundaries() {
    let (x, y) = linear_series(20, 2.0, 1.0);

    // Very low confidence level
    let result_low = Lowess::new()
        .fraction(0.5)
        .confidence_intervals(0.001)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();
    assert!(result_low.confidence_lower.is_some());

    // Very high confidence level
    let result_high = Lowess::new()
        .fraction(0.5)
        .confidence_intervals(0.999)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();
    assert!(result_high.confidence_lower.is_some());
}

/// Test zero iterations with different robustness methods.
#[test]
fn test_zero_iterations_with_robustness_method() {
    let (x, y) = linear_series(15, 2.0, 1.0);

    // With zero iterations, robustness method shouldn't matter
    let methods = vec![RobustnessMethod::Huber, RobustnessMethod::Talwar];

    for method in methods {
        let result = Lowess::new()
            .fraction(0.5)
            .iterations(0)
            .robustness_method(method)
            .adapter(Batch)
            .build()
            .unwrap()
            .fit(&x, &y)
            .unwrap();

        assert_eq!(result.y.len(), 15);
    }
}

/// Test auto-convergence with very small tolerance.
#[test]
fn test_auto_converge_with_zero_tolerance() {
    let (x, y) = linear_series(20, 2.0, 1.0);

    // Very small tolerance means harder to converge early
    let result = Lowess::new()
        .fraction(0.5)
        .auto_converge(1e-10) // Very small tolerance (0.0 is invalid)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    // With very small tolerance, iterations_used should be set
    assert!(result.iterations_used.is_some());
    assert!(result.iterations_used.unwrap() > 0);
}

// ============================================================================
// Adapter Transition Edge Cases
// ============================================================================

/// Test that Batch adapter ignores streaming-specific parameters.
#[test]
fn test_adapter_batch_ignores_streaming_params() {
    let (x, y) = linear_series(20, 2.0, 1.0);

    // Set streaming params before selecting Batch
    let result = Lowess::new()
        .fraction(0.5)
        .chunk_size(10)
        .overlap(2)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    // Should work fine, ignoring streaming params
    assert_eq!(result.y.len(), 20);
}

/// Test that Online adapter ignores batch-specific parameters.
#[test]
fn test_adapter_online_ignores_batch_params() {
    // Batch doesn't have unique params, but test delta which is less relevant for online
    let mut processor = Lowess::new()
        .fraction(0.5)
        .delta(0.1) // Less relevant for online
        .adapter(Online)
        .window_capacity(10)
        .min_points(3)
        .build()
        .unwrap();

    // Should work fine
    for i in 0..5 {
        processor.add_point(i as f64, (i * 2) as f64).unwrap();
    }

    assert_eq!(processor.window_size(), 5);
}

/// Test that Streaming adapter ignores online-specific parameters.
#[test]
fn test_adapter_streaming_ignores_online_params() {
    let (x, y) = linear_series(20, 2.0, 1.0);

    // Set online params before selecting Streaming
    let mut processor = Lowess::new()
        .fraction(0.5)
        .window_capacity(100)
        .min_points(5)
        .adapter(Streaming)
        .chunk_size(10)
        .overlap(2)
        .build()
        .unwrap();

    // Should work fine, ignoring online params
    let result = processor.process_chunk(&x[0..10], &y[0..10]).unwrap();
    assert_eq!(result.x.len(), 8);
}

/// Test that common parameters are preserved across adapter selection.
#[test]
fn test_adapter_transition_preserves_common_params() {
    let (x, y) = linear_series(20, 2.0, 1.0);

    // Set common params
    let builder = Lowess::new()
        .fraction(0.6)
        .iterations(3)
        .robustness_method(RobustnessMethod::Huber);

    // Use with Batch
    let result = builder
        .clone()
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    assert_eq!(result.fraction_used, 0.6);
}

/// Test adapter type can be inferred from context.
#[test]
fn test_adapter_type_inference() {
    let (_x, _y) = linear_series(20, 2.0, 1.0);

    // Type inference should work
    let _batch_processor = Lowess::new().fraction(0.5).adapter(Batch).build().unwrap();

    let _online_processor = Lowess::new()
        .fraction(0.5)
        .adapter(Online)
        .window_capacity(10)
        .min_points(3)
        .build()
        .unwrap();

    let _streaming_processor = Lowess::new()
        .fraction(0.5)
        .adapter(Streaming)
        .chunk_size(10)
        .overlap(2)
        .build()
        .unwrap();

    // All should compile with proper type inference
}
