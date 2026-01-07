#![cfg(feature = "dev")]
//! Tests for LOWESS output structures.
//!
//! These tests verify the LowessResult structure and its methods:
//! - Query methods (has_confidence_intervals, has_prediction_intervals, etc.)
//! - Accessor methods (smoothed, best_cv_score)
//! - Interval width calculations
//! - Display formatting
//!
//! ## Test Organization
//!
//! 1. **Query Methods** - Checking presence of optional outputs
//! 2. **Accessor Methods** - Getting data from result
//! 3. **Interval Widths** - Confidence and prediction interval widths
//! 4. **Edge Cases** - Empty, None values, partial data
//! 5. **Display** - Formatting output

use approx::assert_relative_eq;
use num_traits::Float;

use lowess::internals::engine::output::LowessResult;
use lowess::internals::evaluation::diagnostics::Diagnostics;

// ============================================================================
// Test Helper Trait
// ============================================================================

/// Helper trait for testing LowessResult methods.
///
/// These methods are only used in tests and are not part of the public API.
trait LowessResultTestExt<T: Float> {
    /// Get the smoothed values as a slice.
    fn smoothed(&self) -> &[T];

    /// Get confidence interval width at each point.
    fn confidence_width(&self) -> Option<Vec<T>>;

    /// Get prediction interval width at each point.
    fn prediction_width(&self) -> Option<Vec<T>>;
}

impl<T: Float> LowessResultTestExt<T> for LowessResult<T> {
    fn smoothed(&self) -> &[T] {
        &self.y
    }

    fn confidence_width(&self) -> Option<Vec<T>> {
        match (&self.confidence_lower, &self.confidence_upper) {
            (Some(lower), Some(upper)) => Some(
                lower
                    .iter()
                    .zip(upper.iter())
                    .map(|(l, u)| *u - *l)
                    .collect(),
            ),
            _ => None,
        }
    }

    fn prediction_width(&self) -> Option<Vec<T>> {
        match (&self.prediction_lower, &self.prediction_upper) {
            (Some(lower), Some(upper)) => Some(
                lower
                    .iter()
                    .zip(upper.iter())
                    .map(|(l, u)| *u - *l)
                    .collect(),
            ),
            _ => None,
        }
    }
}

/// Test has_confidence_intervals with both bounds present.
///
/// Verifies that method returns true when both bounds exist.
#[test]
fn test_has_confidence_intervals_true() {
    let lr = LowessResult {
        x: vec![1.0],
        y: vec![2.0],
        standard_errors: None,
        confidence_lower: Some(vec![1.5]),
        confidence_upper: Some(vec![2.5]),
        prediction_lower: None,
        prediction_upper: None,
        residuals: None,
        robustness_weights: None,
        diagnostics: None,
        iterations_used: None,
        fraction_used: 0.5,
        cv_scores: None,
    };

    assert!(
        lr.has_confidence_intervals(),
        "Should have confidence intervals"
    );
}

/// Test has_confidence_intervals with missing bounds.
///
/// Verifies that method returns false when bounds are missing.
#[test]
fn test_has_confidence_intervals_false() {
    let lr = LowessResult {
        x: vec![1.0],
        y: vec![2.0],
        standard_errors: None,
        confidence_lower: Some(vec![1.5]),
        confidence_upper: None, // Missing upper
        prediction_lower: None,
        prediction_upper: None,
        residuals: None,
        robustness_weights: None,
        diagnostics: None,
        iterations_used: None,
        fraction_used: 0.5,
        cv_scores: None,
    };

    assert!(
        !lr.has_confidence_intervals(),
        "Should not have complete confidence intervals"
    );
}

/// Test has_prediction_intervals with both bounds present.
///
/// Verifies that method returns true when both bounds exist.
#[test]
fn test_has_prediction_intervals_true() {
    let lr = LowessResult {
        x: vec![1.0],
        y: vec![2.0],
        standard_errors: None,
        confidence_lower: None,
        confidence_upper: None,
        prediction_lower: Some(vec![0.5]),
        prediction_upper: Some(vec![3.5]),
        residuals: None,
        robustness_weights: None,
        diagnostics: None,
        iterations_used: None,
        fraction_used: 0.5,
        cv_scores: None,
    };

    assert!(
        lr.has_prediction_intervals(),
        "Should have prediction intervals"
    );
}

/// Test has_prediction_intervals with missing bounds.
///
/// Verifies that method returns false when bounds are missing.
#[test]
fn test_has_prediction_intervals_false() {
    let lr = LowessResult {
        x: vec![1.0],
        y: vec![2.0],
        standard_errors: None,
        confidence_lower: None,
        confidence_upper: None,
        prediction_lower: None,
        prediction_upper: Some(vec![3.5]), // Only upper
        residuals: None,
        robustness_weights: None,
        diagnostics: None,
        iterations_used: None,
        fraction_used: 0.5,
        cv_scores: None,
    };

    assert!(
        !lr.has_prediction_intervals(),
        "Should not have complete prediction intervals"
    );
}

/// Test has_cv_scores with scores present.
///
/// Verifies that method returns true when CV scores exist.
#[test]
fn test_has_cv_scores_true() {
    let lr = LowessResult {
        x: vec![1.0],
        y: vec![2.0],
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
        cv_scores: Some(vec![0.1, 0.2, 0.3]),
    };

    assert!(lr.has_cv_scores(), "Should have CV scores");
}

/// Test has_cv_scores with no scores.
///
/// Verifies that method returns false when CV scores are absent.
#[test]
fn test_has_cv_scores_false() {
    let lr = LowessResult {
        x: vec![1.0],
        y: vec![2.0],
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
        cv_scores: None,
    };

    assert!(!lr.has_cv_scores(), "Should not have CV scores");
}

// ============================================================================
// Accessor Methods Tests
// ============================================================================

/// Test smoothed accessor method.
///
/// Verifies that smoothed() returns reference to y values.
#[test]
fn test_smoothed_accessor() {
    let y_vals = vec![1.0, 2.0, 3.0];
    let lr = LowessResult {
        x: vec![0.0, 1.0, 2.0],
        y: y_vals.clone(),
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
        cv_scores: None,
    };

    assert_eq!(lr.smoothed(), &y_vals[..]);
}

/// Test best_cv_score with scores present.
///
/// Verifies that minimum score is returned.
#[test]
fn test_best_cv_score_present() {
    let lr = LowessResult {
        x: vec![1.0],
        y: vec![2.0],
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
        cv_scores: Some(vec![0.3, 0.1, 0.2, 0.5]),
    };

    let best = lr.best_cv_score();
    assert!(best.is_some());
    assert_relative_eq!(best.unwrap(), 0.1, epsilon = 1e-12);
}

/// Test best_cv_score with no scores.
///
/// Verifies that None is returned when CV scores are absent.
#[test]
fn test_best_cv_score_none() {
    let lr = LowessResult {
        x: vec![1.0],
        y: vec![2.0],
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
        cv_scores: None,
    };

    assert!(lr.best_cv_score().is_none());
}

/// Test best_cv_score with single score.
///
/// Verifies correct handling of single CV score.
#[test]
fn test_best_cv_score_single() {
    let lr = LowessResult {
        x: vec![1.0],
        y: vec![2.0],
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
        cv_scores: Some(vec![0.42]),
    };

    let best = lr.best_cv_score();
    assert!(best.is_some());
    assert_relative_eq!(best.unwrap(), 0.42, epsilon = 1e-12);
}

// ============================================================================
// Interval Width Tests
// ============================================================================

/// Test confidence_width calculation.
///
/// Verifies that width is computed as upper - lower.
#[test]
fn test_confidence_width() {
    let lr = LowessResult {
        x: vec![0.0, 1.0, 2.0],
        y: vec![1.0, 2.0, 3.0],
        standard_errors: None,
        confidence_lower: Some(vec![0.9, 1.9, 2.9]),
        confidence_upper: Some(vec![1.1, 2.1, 3.1]),
        prediction_lower: None,
        prediction_upper: None,
        residuals: None,
        robustness_weights: None,
        diagnostics: None,
        iterations_used: None,
        fraction_used: 0.5,
        cv_scores: None,
    };

    let widths = lr.confidence_width().expect("Should have widths");
    assert_eq!(widths.len(), 3);
    for &w in &widths {
        assert_relative_eq!(w, 0.2, epsilon = 1e-12);
    }
}

/// Test confidence_width with no intervals.
///
/// Verifies that None is returned when intervals are absent.
#[test]
fn test_confidence_width_none() {
    let lr = LowessResult {
        x: vec![1.0],
        y: vec![2.0],
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
        cv_scores: None,
    };

    assert!(lr.confidence_width().is_none());
}

/// Test prediction_width calculation.
///
/// Verifies that width is computed as upper - lower.
#[test]
fn test_prediction_width() {
    let lr = LowessResult {
        x: vec![0.0, 1.0, 2.0],
        y: vec![1.0, 2.0, 3.0],
        standard_errors: None,
        confidence_lower: None,
        confidence_upper: None,
        prediction_lower: Some(vec![0.5, 1.5, 2.5]),
        prediction_upper: Some(vec![1.5, 2.5, 3.5]),
        residuals: None,
        robustness_weights: None,
        diagnostics: None,
        iterations_used: None,
        fraction_used: 0.5,
        cv_scores: None,
    };

    let widths = lr.prediction_width().expect("Should have widths");
    assert_eq!(widths.len(), 3);
    for &w in &widths {
        assert_relative_eq!(w, 1.0, epsilon = 1e-12);
    }
}

/// Test prediction_width with no intervals.
///
/// Verifies that None is returned when intervals are absent.
#[test]
fn test_prediction_width_none() {
    let lr = LowessResult {
        x: vec![1.0],
        y: vec![2.0],
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
        cv_scores: None,
    };

    assert!(lr.prediction_width().is_none());
}

/// Test that prediction intervals are wider than confidence intervals.
///
/// Verifies expected relationship between interval types.
#[test]
fn test_prediction_wider_than_confidence() {
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
        iterations_used: None,
        fraction_used: 0.5,
        cv_scores: None,
    };

    let conf_w = lr.confidence_width().unwrap();
    let pred_w = lr.prediction_width().unwrap();

    for i in 0..conf_w.len() {
        assert!(pred_w[i] > conf_w[i], "Prediction interval should be wider");
    }
}

// ============================================================================
// Edge Cases Tests
// ============================================================================

/// Test result with all optional fields None.
///
/// Verifies correct handling of minimal result.
#[test]
fn test_minimal_result() {
    let lr = LowessResult {
        x: vec![1.0, 2.0],
        y: vec![10.0, 20.0],
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
        cv_scores: None,
    };

    assert!(!lr.has_confidence_intervals());
    assert!(!lr.has_prediction_intervals());
    assert!(!lr.has_cv_scores());
    assert!(lr.best_cv_score().is_none());
    assert!(lr.confidence_width().is_none());
    assert!(lr.prediction_width().is_none());
}

/// Test result with all optional fields Some.
///
/// Verifies correct handling of maximal result.
#[test]
fn test_maximal_result() {
    let lr = LowessResult {
        x: vec![1.0, 2.0],
        y: vec![10.0, 20.0],
        standard_errors: Some(vec![0.1, 0.2]),
        confidence_lower: Some(vec![9.8, 19.6]),
        confidence_upper: Some(vec![10.2, 20.4]),
        prediction_lower: Some(vec![9.5, 19.0]),
        prediction_upper: Some(vec![10.5, 21.0]),
        residuals: Some(vec![0.0, 0.0]),
        robustness_weights: Some(vec![1.0, 1.0]),
        diagnostics: Some(Diagnostics {
            rmse: 0.0,
            mae: 0.0,
            r_squared: 1.0,
            aic: Some(10.0),
            aicc: Some(12.0),
            effective_df: Some(2.0),
            residual_sd: 0.0,
        }),
        iterations_used: Some(3),
        fraction_used: 0.5,
        cv_scores: Some(vec![0.1, 0.2]),
    };

    assert!(lr.has_confidence_intervals());
    assert!(lr.has_prediction_intervals());
    assert!(lr.has_cv_scores());
    assert!(lr.best_cv_score().is_some());
    assert!(lr.confidence_width().is_some());
    assert!(lr.prediction_width().is_some());
}

/// Test result with empty CV scores.
///
/// Verifies handling of empty scores vector.
#[test]
fn test_empty_cv_scores() {
    let lr = LowessResult {
        x: vec![1.0],
        y: vec![2.0],
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
        cv_scores: Some(vec![]),
    };

    assert!(lr.has_cv_scores());
    assert!(lr.best_cv_score().is_none());
}

// ============================================================================
// Display Tests
// ============================================================================

/// Test Display implementation basic output.
///
/// Verifies that Display produces non-empty output.
#[test]
fn test_display_basic() {
    let lr = LowessResult {
        x: vec![1.0, 2.0, 3.0],
        y: vec![10.0, 20.0, 30.0],
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
        cv_scores: None,
    };

    let output = format!("{}", lr);
    assert!(!output.is_empty());
    assert!(output.contains("Summary"));
    assert!(output.contains("Data points: 3"));
    assert!(output.contains("Fraction:"));
    assert!(output.contains("0.5"));
}

// ============================================================================
// Additional Output Edge Cases
// ============================================================================

/// Test best_cv_score with NaN values in scores.
#[test]
fn test_best_cv_score_with_nan() {
    let result = LowessResult {
        x: vec![1.0, 2.0, 3.0],
        y: vec![2.0, 4.0, 6.0],
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
        cv_scores: Some(vec![0.5, f64::NAN, 0.3, 0.7]),
    };

    let best = result.best_cv_score();
    // Should return 0.3 (minimum non-NaN value)
    assert_eq!(best, Some(0.3));
}

/// Test best_cv_score with all equal scores.
#[test]
fn test_best_cv_score_all_equal() {
    let result = LowessResult {
        x: vec![1.0, 2.0, 3.0],
        y: vec![2.0, 4.0, 6.0],
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
        cv_scores: Some(vec![0.5, 0.5, 0.5]),
    };

    let best = result.best_cv_score();
    assert_eq!(best, Some(0.5));
}

/// Test display formatting with empty vectors.
#[test]
fn test_display_with_empty_vectors() {
    let result = LowessResult {
        x: vec![],
        y: vec![],
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
        cv_scores: None,
    };

    let display_str = format!("{}", result);
    assert!(display_str.contains("Data points: 0"));
}

/// Test Display with all optional fields.
///
/// Verifies that all fields are included in output.
#[test]
fn test_display_with_all_fields() {
    let lr = LowessResult {
        x: vec![1.0, 2.0],
        y: vec![10.0, 20.0],
        standard_errors: Some(vec![0.1, 0.2]),
        confidence_lower: Some(vec![9.8, 19.6]),
        confidence_upper: Some(vec![10.2, 20.4]),
        prediction_lower: Some(vec![9.5, 19.0]),
        prediction_upper: Some(vec![10.5, 21.0]),
        residuals: Some(vec![0.1, -0.1]),
        robustness_weights: Some(vec![1.0, 0.9]),
        diagnostics: Some(Diagnostics {
            rmse: 0.05,
            mae: 0.04,
            r_squared: 0.99,
            aic: Some(10.0),
            aicc: Some(12.0),
            effective_df: Some(2.0),
            residual_sd: 0.05,
        }),
        iterations_used: Some(3),
        fraction_used: 0.5,
        cv_scores: Some(vec![0.1, 0.2]),
    };

    let output = format!("{}", lr);
    assert!(output.contains("Iterations: 3"));
    assert!(output.contains("Robustness: Applied"));
    assert!(output.contains("Best CV score"));
    assert!(output.contains("LOWESS Diagnostics"));
    assert!(output.contains("Std_Err"));
    assert!(output.contains("Conf_Lower"));
    assert!(output.contains("Pred_Lower"));
    assert!(output.contains("Residual"));
    assert!(output.contains("Rob_Weight"));
}

/// Test Display with large dataset.
///
/// Verifies that large datasets are truncated with ellipsis.
#[test]
fn test_display_large_dataset() {
    let n = 100;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = (0..n).map(|i| (i * 2) as f64).collect();

    let lr = LowessResult {
        x,
        y,
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
        cv_scores: None,
    };

    let output = format!("{}", lr);
    assert!(output.contains("Data points: 100"));
    assert!(
        output.contains("..."),
        "Should have ellipsis for truncated data"
    );
}
