#![cfg(feature = "dev")]
//! Tests for confidence and prediction interval computation.
//!
//! These tests verify the interval estimation functionality used in LOWESS for:
//! - Standard error computation
//! - Confidence intervals
//! - Prediction intervals
//! - Z-score approximation
//! - Interval validation
//! - Residual-bootstrap standard errors and intervals
//!
//! ## Test Organization
//!
//! 1. **Standard Error** - Point and window SE computation
//! 2. **Z-Score** - Normal distribution quantiles
//! 3. **Confidence Intervals** - CI computation and validation
//! 4. **Prediction Intervals** - PI computation and validation
//! 5. **Edge Cases** - Zero bandwidth, zero weights, invalid levels

use approx::assert_relative_eq;
use lowess::prelude::*;

use lowess::internals::api::Batch;
use lowess::internals::engine::validator::Validator;
use lowess::internals::evaluation::intervals::IntervalMethod;
use lowess::internals::primitives::errors::LowessError;

// ============================================================================
// Helper Functions
// ============================================================================

fn uniform_weight_fn<T: num_traits::Float>(_u: T) -> T {
    T::one()
}

// ============================================================================
// Standard Error Tests
// ============================================================================

/// Test point SE computation at center point.
///
/// Verifies correct SE calculation for middle point.
#[test]
fn test_compute_point_se_center() {
    let x = vec![0.0f64, 1.0, 2.0];
    let y = vec![0.0f64, 1.0, 0.0];
    let y_smooth = vec![0.0f64, 0.0, 0.0];
    let robustness = vec![1.0f64; 3];

    let est = IntervalMethod::se();
    let mut std_errors = vec![0.0; x.len()];

    est.compute_window_se(
        &x,
        &y,
        &y_smooth,
        3,
        &robustness,
        &mut std_errors,
        &uniform_weight_fn,
    );

    let se = std_errors[1];
    // Uniform weights on a symmetric 3-point window: the exact local-linear
    // variance multiplier is sum_k l_k^2 = 1/3 (each l_k = 1/3) and the
    // kernel-corrected residual df is sum(w) - 2 + sum(w^2)/sum(w) = 3 - 2 + 1 = 2,
    // so SE = sqrt((1/2) * (1/3)) = sqrt(1/6).
    let expected = (1.0f64 / 6.0f64).sqrt();

    assert_relative_eq!(se, expected, epsilon = 1e-12);
}

/// Test SE with zero bandwidth (identical x values).
///
/// Verifies that zero bandwidth produces zero SE.
#[test]
fn test_compute_se_zero_bandwidth() {
    let x = vec![1.0f64, 1.0];
    let y = vec![1.0f64, 2.0];
    let y_smooth = vec![1.0f64, 2.0];
    let robustness = vec![1.0f64; 2];

    let est = IntervalMethod::se();
    let mut std_errors = vec![0.0; x.len()];

    est.compute_window_se(
        &x,
        &y,
        &y_smooth,
        2,
        &robustness,
        &mut std_errors,
        &|_: f64| 1.0,
    );

    assert_eq!(std_errors[0], 0.0, "SE should be zero for zero bandwidth");
}

/// Test SE with zero weights.
///
/// Verifies that zero robustness weights produce zero SE.
#[test]
fn test_compute_se_zero_weights() {
    let x = vec![0.0f64, 1.0];
    let y = vec![0.0f64, 1.0];
    let ys = vec![0.0f64, 1.0];
    let robustness_zero = vec![0.0f64; 2];

    let est = IntervalMethod::se();
    let mut std_errors = vec![0.0; x.len()];

    est.compute_window_se(
        &x,
        &y,
        &ys,
        2,
        &robustness_zero,
        &mut std_errors,
        &|_: f64| 1.0,
    );

    assert_eq!(std_errors[0], 0.0, "SE should be zero for zero weights");
}

/// A fully down-weighted point (robustness weight 0) must still have a positive
/// standard error. The leverage term comes from the local design (kernel weight),
/// not from the point's own robustness weight, so a down-weighted observation's
/// interval must not collapse to zero width.
#[test]
fn test_compute_se_downweighted_point_has_positive_se() {
    let x = vec![0.0f64, 1.0, 2.0, 3.0, 4.0];
    let y = vec![1.0f64, 0.0, 100.0, 0.0, 1.0];
    let y_smooth = vec![0.0f64, 0.0, 0.0, 0.0, 0.0];
    // Point 2 is a gross outlier, fully down-weighted by robustness iterations.
    let robustness = vec![1.0f64, 1.0, 0.0, 1.0, 1.0];

    let est = IntervalMethod::se();
    let mut std_errors = vec![0.0; x.len()];

    est.compute_window_se(
        &x,
        &y,
        &y_smooth,
        5,
        &robustness,
        &mut std_errors,
        &uniform_weight_fn,
    );

    // SE must not collapse to zero. The exact local-linear variance multiplier is
    // sum_k l_k^2 = 1/4 here (symmetric window, the down-weighted centre point
    // contributes nothing to the design), and df = sum(w) - 2 + sum(w^2)/sum(w)
    // = 4 - 2 + 4/4 = 3, so SE = sqrt((2/3) * (1/4)) = sqrt(1/6).
    assert_relative_eq!(std_errors[2], (1.0f64 / 6.0f64).sqrt(), epsilon = 1e-12);
}

/// Test SE with insufficient degrees of freedom.
///
/// Verifies that df <= 0 produces zero SE.
#[test]
fn test_compute_se_insufficient_df() {
    let x = vec![0.0f64, 1.0];
    let y = vec![0.0f64, 1.0];
    let ys = vec![0.0f64, 0.0];
    let robustness_ones = vec![1.0f64; 2];

    let est = IntervalMethod::se();
    let mut std_errors = vec![0.0; x.len()];

    // With two points and constant weight 0.4: sum(w) = 0.8, sum(w^2) = 0.32, so
    // the kernel-corrected df = 0.8 - 2 + 0.32/0.8 = -0.8 <= 0 => SE = 0.
    est.compute_window_se(
        &x,
        &y,
        &ys,
        2,
        &robustness_ones,
        &mut std_errors,
        &|_: f64| 0.4,
    );

    assert_eq!(std_errors[0], 0.0, "SE should be zero for df <= 0");
}

/// Test window SE computation.
///
/// Verifies that SE vector is correctly populated.
#[test]
fn test_compute_window_se_vector() {
    let x = vec![0.0f64, 1.0, 2.0];
    let y = vec![0.0f64, 1.0, 0.0];
    let y_smooth = vec![0.0f64, 0.0, 0.0];
    let robustness = vec![1.0f64; 3];
    let mut std_err = vec![0.0f64; 3];

    let estimator = IntervalMethod::se();
    estimator.compute_window_se(
        &x,
        &y,
        &y_smooth,
        3,
        &robustness,
        &mut std_err,
        &uniform_weight_fn,
    );

    // Middle element should match expected value. Uniform weights on a symmetric
    // 3-point window give variance multiplier 1/3 and df = 3 - 2 + 1 = 2, so
    // SE = sqrt((1/2) * (1/3)) = sqrt(1/6).
    let expected_mid = (1.0f64 / 6.0f64).sqrt();
    assert_relative_eq!(std_err[1], expected_mid, epsilon = 1e-12);
    assert_eq!(std_err.len(), 3, "SE vector should have correct length");
}

// ============================================================================
// Z-Score Tests
// ============================================================================

/// Test z-score for common confidence levels.
///
/// Verifies correct z-scores for 90%, 95%, 99%.
#[test]
fn test_z_score_common_levels() {
    let z95 = IntervalMethod::approximate_z_score(0.95f64).expect("z95");
    assert_relative_eq!(z95, 1.96f64, epsilon = 1e-6);

    let z99 = IntervalMethod::approximate_z_score(0.99f64).expect("z99");
    assert_relative_eq!(z99, 2.576f64, epsilon = 1e-6);

    let z90 = IntervalMethod::approximate_z_score(0.90f64).expect("z90");
    assert_relative_eq!(z90, 1.645f64, epsilon = 1e-6);
}

/// Test z-score for arbitrary level.
///
/// Verifies that arbitrary confidence levels produce finite z-scores.
#[test]
fn test_z_score_arbitrary() {
    let z = IntervalMethod::approximate_z_score(0.87f64).expect("z");

    assert!(z.is_finite(), "Z-score should be finite");
    assert!(z > 0.0, "Z-score should be positive");
}

/// Test very high confidence level to hit Acklam's tail regions.
#[test]
fn test_acklam_tails() {
    let z_999 = IntervalMethod::<f64>::approximate_z_score(0.999).unwrap();
    assert!(z_999 > 3.0, "z_999 was {}", z_999);

    let z_001 = IntervalMethod::<f64>::approximate_z_score(0.001).unwrap();
    assert!(z_001 > 0.0);
}

// ============================================================================
// Confidence Interval Tests
// ============================================================================

/// Test confidence interval computation.
///
/// Verifies that CI is computed correctly.
#[test]
fn test_confidence_intervals() {
    let y_smooth = vec![10.0f64, 20.0];
    let std_err = vec![1.0f64, 2.0];
    let level = 0.95f64;
    let residuals = vec![0.0f64; 2];

    let estimator = IntervalMethod::confidence(level);
    let (cl, cu, _, _) = estimator
        .compute_intervals(&y_smooth, &std_err, &residuals)
        .expect("intervals");

    assert!(cl.is_some(), "CI lower should be computed");
    assert!(cu.is_some(), "CI upper should be computed");

    let lower = cl.unwrap();
    let upper = cu.unwrap();

    // Verify intervals contain smoothed values
    for i in 0..y_smooth.len() {
        assert!(
            lower[i] <= y_smooth[i] && y_smooth[i] <= upper[i],
            "CI should contain smoothed value"
        );
    }
}

/// Test prediction interval computation.
///
/// Verifies that PI is computed correctly.
#[test]
fn test_prediction_intervals() {
    let y_smooth = vec![10.0f64, 20.0];
    let std_err = vec![1.0f64, 2.0];
    let level = 0.95f64;
    let residuals = vec![0.0f64; 2];

    let estimator = IntervalMethod::prediction(level);
    let (_, _, pl, pu) = estimator
        .compute_intervals(&y_smooth, &std_err, &residuals)
        .expect("intervals");

    assert!(pl.is_some(), "PI lower should be computed");
    assert!(pu.is_some(), "PI upper should be computed");
}

/// Test that PI is wider than CI.
///
/// Verifies that prediction intervals include residual variance.
#[test]
fn test_pi_wider_than_ci() {
    let y_smooth = vec![10.0f64, 20.0];
    let std_err = vec![1.0f64, 2.0];
    let level = 0.95f64;
    let residuals_noisy = vec![3.0, -3.0]; // Non-zero residuals

    let estimator_ci = IntervalMethod::confidence(level);
    let (cl, cu, _, _) = estimator_ci
        .compute_intervals(&y_smooth, &std_err, &residuals_noisy)
        .expect("CI");

    let estimator_pi = IntervalMethod::prediction(level);
    let (_, _, pl, pu) = estimator_pi
        .compute_intervals(&y_smooth, &std_err, &residuals_noisy)
        .expect("PI");

    let w_ci = cu.unwrap()[0] - cl.unwrap()[0];
    let w_pi = pu.unwrap()[0] - pl.unwrap()[0];

    assert!(w_pi > w_ci, "PI should be wider than CI");
}

// ============================================================================
// Integration Tests
// ============================================================================

/// Test complete interval method workflow.
///
/// Verifies SE computation and interval calculation together.
#[test]
fn test_interval_method_workflow() {
    let x = vec![0.0f64, 1.0, 2.0];
    let y = vec![0.0f64, 1.0, 0.0];
    let y_smooth = vec![0.0f64, 0.0, 0.0];
    let robustness = vec![1.0f64; 3];
    let level = 0.95f64;

    let estimator = IntervalMethod::confidence(level);

    // Compute SE
    let mut std_errors = vec![0.0f64; 3];
    estimator.compute_window_se(
        &x,
        &y,
        &y_smooth,
        3,
        &robustness,
        &mut std_errors,
        &uniform_weight_fn,
    );

    // Uniform weights on a symmetric 3-point window: variance multiplier 1/3 and
    // df = 3 - 2 + 1 = 2, so SE = sqrt((1/2) * (1/3)) = sqrt(1/6).
    let expected_se_mid = (1.0f64 / 6.0f64).sqrt();
    assert_relative_eq!(std_errors[1], expected_se_mid, epsilon = 1e-12);

    // Compute intervals
    let residuals = vec![0.0f64; 3];
    let (ci_lower, ci_upper, _, _) = estimator
        .compute_intervals(&y_smooth, &std_errors, &residuals)
        .expect("Intervals");

    // Verify lengths
    assert_eq!(ci_lower.unwrap().len(), 3, "CI lower should have 3 values");
    assert_eq!(ci_upper.unwrap().len(), 3, "CI upper should have 3 values");

    // Verify level
    assert_relative_eq!(estimator.level, 0.95, epsilon = 1e-12);
}

/// Test residual SD with minimum required points.
#[test]
fn test_interval_edge_cases() {
    let lowess = Lowess::<f64>::new()
        .fraction(1.0)
        .intervals(IntervalsBuilder::new().confidence(0.95))
        .build()
        .unwrap();

    let x = [1.0, 2.0];
    let y = [2.0, 4.0];
    let result = lowess.fit(&x, &y).unwrap();
    assert!(result.standard_errors.is_some());
}

// ============================================================================
// Validation Tests
// ============================================================================

/// Test interval level validation.
///
/// Verifies that validator correctly checks interval levels.
#[test]
fn test_validate_level() {
    assert!(
        Validator::validate_interval_level(0.95).is_ok(),
        "Valid level should pass"
    );

    match Validator::validate_interval_level(1.5) {
        Err(LowessError::InvalidIntervals(v)) => {
            assert!(
                (v - 1.5).abs() < 1e-10,
                "Error should contain invalid value"
            )
        }
        _ => panic!("Expected InvalidIntervals error"),
    }

    match Validator::validate_interval_level(-0.1) {
        Err(LowessError::InvalidIntervals(v)) => {
            assert!(
                (v + 0.1).abs() < 1e-10,
                "Error should contain invalid value"
            )
        }
        _ => panic!("Expected InvalidIntervals error"),
    }

    match Validator::validate_interval_level(f64::NAN) {
        Err(LowessError::InvalidIntervals(v)) => {
            assert!(v.is_nan(), "Error should contain NaN")
        }
        _ => panic!("Expected InvalidIntervals error"),
    }
}

/// Test interval method constructors.
///
/// Verifies that different constructors produce correct configurations.
#[test]
fn test_interval_method_constructors() {
    let ci = IntervalMethod::confidence(0.95);
    assert_relative_eq!(ci.level, 0.95, epsilon = 1e-12);

    let pi = IntervalMethod::prediction(0.99);
    assert_relative_eq!(pi.level, 0.99, epsilon = 1e-12);

    let se: IntervalMethod<f64> = IntervalMethod::se();
    // SE method should have default level
    assert!(
        se.level > 0.0 && se.level < 1.0,
        "SE should have valid level"
    );

    // Test Default trait
    let d = IntervalMethod::<f64>::default();
    assert!(!d.confidence);
}

// ============================================================================
// Internal Interval Edge Cases
// ============================================================================

/// Test residual SD with edge point counts (n=0, n=1).
#[test]
fn test_residual_sd_edge_points() {
    // Internal method access via IntervalMethod
    // We can't call private methods directly from integration tests unless we use internals re-export
    // Let's check if compute_intervals handles them correctly or if we can test calculate_residual_sd

    // Actually compute_intervals uses calculate_residual_sd internally.
    let ys = vec![10.0f64];
    let ses = vec![0.1f64];
    let residuals = vec![0.0f64];

    let estimator = IntervalMethod::prediction(0.95);
    let result = estimator.compute_intervals(&ys, &ses, &residuals);
    assert!(result.is_ok());
}

/// Test Z-score with extremely high precision.
#[test]
fn test_z_score_high_precision() {
    // Test very close to 1.0
    let z_extreme = IntervalMethod::<f64>::approximate_z_score(0.9999).unwrap();
    assert!(z_extreme > 3.8); // z for 0.9999 is ~3.89

    // Fast path for 0.95
    let z_95 = IntervalMethod::<f64>::approximate_z_score(0.95).unwrap();
    assert_relative_eq!(z_95, 1.96, epsilon = 1e-6);
}

/// Test intervals when standard error is zero.
#[test]
fn test_intervals_degenerate_se() {
    let ys = vec![10.0];
    let ses = vec![0.0]; // Zero SE
    let residuals = vec![1.0]; // Non-zero residual

    let estimator = IntervalMethod::confidence(0.95);
    let (cl, cu, _, _) = estimator.compute_intervals(&ys, &ses, &residuals).unwrap();

    let clv = cl.unwrap();
    let cuv = cu.unwrap();

    // Width should be clamped to EPS
    assert!(cuv[0] > clv[0]);
    assert_relative_eq!(cuv[0] - clv[0], 1e-12, epsilon = 1e-15);
}

// ============================================================================
// Residual Bootstrap Tests

#[test]
fn test_grouped_intervals_reproduce_seeded_output() {
    let x: Vec<f64> = (0..30).map(|i| i as f64 * 0.2).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| xi.sin() + 0.2 * (xi * 7.0).sin())
        .collect();
    let fit = || {
        Lowess::new()
            .fraction(0.5)
            .iterations(0)
            .seed(7)
            .intervals(
                IntervalsBuilder::new()
                    .confidence(0.95)
                    .prediction(0.95)
                    .bootstrap(40),
            )
            .build()
            .unwrap()
            .fit(&x, &y)
            .unwrap()
    };
    let (first, second) = (fit(), fit());
    assert_eq!(first.y, second.y);
    assert_eq!(first.standard_errors, second.standard_errors);
    assert_eq!(first.confidence_lower, second.confidence_lower);
    assert_eq!(first.confidence_upper, second.confidence_upper);
    assert_eq!(first.prediction_lower, second.prediction_lower);
    assert_eq!(first.prediction_upper, second.prediction_upper);
}

#[test]
fn test_grouped_intervals_work_in_streaming_and_online() {
    let x: Vec<f64> = (0..15).map(|i| i as f64 * 0.2).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();
    let mut streaming = StreamingLowess::new()
        .chunk_size(15)
        .overlap(2)
        .seed(7)
        .intervals(IntervalsBuilder::new().bootstrap(20))
        .build()
        .unwrap();
    let streamed = streaming.process_chunk(&x, &y).unwrap();
    assert_eq!(
        streamed.standard_errors.as_ref().unwrap().len(),
        streamed.y.len()
    );
    assert!(streamed.confidence_lower.is_none());

    let mut online = OnlineLowess::new()
        .update_mode("full")
        .min_points(3)
        .seed(7)
        .intervals(IntervalsBuilder::new().prediction(0.95).bootstrap(20))
        .build()
        .unwrap();
    for (&xi, &yi) in x.iter().zip(&y) {
        if let Some(output) = online.add_point(xi, yi).unwrap() {
            assert!(output.standard_error.is_some());
            assert!(output.prediction_lower.is_some());
            assert!(output.confidence_lower.is_none());
        }
    }
}

#[test]
fn test_grouped_intervals_preserve_validation() {
    let invalid = Lowess::<f64>::new()
        .intervals(IntervalsBuilder::new().confidence(0.95).bootstrap(1))
        .build()
        .err();
    assert_eq!(invalid, Some(LowessError::InvalidBootstrapSamples(1)));

    let incremental = OnlineLowess::<f64>::new()
        .intervals(IntervalsBuilder::new().bootstrap(20))
        .build()
        .err();
    assert_eq!(
        incremental,
        Some(LowessError::StandardErrorRequiresFullUpdateMode)
    );

    let repeated = Lowess::<f64>::new()
        .intervals(IntervalsBuilder::new().confidence(0.90))
        .intervals(IntervalsBuilder::new().confidence(0.95))
        .build()
        .err();
    assert_eq!(
        repeated,
        Some(LowessError::DuplicateParameter {
            parameter: "intervals"
        })
    );
}
// ============================================================================

fn noisy_sine(n: usize, skewed: bool) -> (Vec<f64>, Vec<f64>) {
    let mut state: u64 = 12345;
    let mut uniform = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64 * 6.0).collect();
    let y = x
        .iter()
        .map(|&xi| {
            let u: f64 = uniform().max(1e-12);
            // Exponential(1) - 1 is mean-zero and right-skewed; uniform is symmetric.
            let e = if skewed { -u.ln() - 1.0 } else { u - 0.5 };
            xi.sin() + 0.3 * e
        })
        .collect();
    (x, y)
}

/// Bootstrap SEs and intervals have the right length, are finite, and nest (PI contains CI).
#[test]
fn test_bootstrap_intervals_shape_and_ordering() {
    let (x, y) = noisy_sine(120, false);
    let res = Lowess::new()
        .fraction(0.3)
        .iterations(0)
        .seed(7)
        .intervals(
            IntervalsBuilder::new()
                .confidence(0.95)
                .prediction(0.95)
                .bootstrap(200),
        )
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let se = res.standard_errors.as_ref().unwrap();
    let (cl, cu) = (
        res.confidence_lower.as_ref().unwrap(),
        res.confidence_upper.as_ref().unwrap(),
    );
    let (pl, pu) = (
        res.prediction_lower.as_ref().unwrap(),
        res.prediction_upper.as_ref().unwrap(),
    );
    assert_eq!(se.len(), x.len());
    for i in 0..x.len() {
        assert!(se[i] > 0.0 && se[i].is_finite());
        assert!(cl[i] < cu[i]);
        assert!(
            pl[i] <= cl[i] && cu[i] <= pu[i],
            "PI should contain CI at {i}"
        );
    }
}

/// The same seed reproduces the intervals; a different seed does not.
#[test]
fn test_bootstrap_is_reproducible_with_seed() {
    let (x, y) = noisy_sine(80, false);
    let run = |seed| {
        Lowess::new()
            .fraction(0.4)
            .seed(seed)
            .intervals(IntervalsBuilder::new().confidence(0.9).bootstrap(50))
            .adapter(Batch)
            .build()
            .unwrap()
            .fit(&x, &y)
            .unwrap()
            .confidence_lower
            .unwrap()
    };
    assert_eq!(run(1), run(1));
    assert_ne!(run(1), run(2));
}

/// Bootstrap SEs agree in magnitude with the analytic ones on symmetric noise.
#[test]
fn test_bootstrap_se_matches_analytic_magnitude() {
    let (x, y) = noisy_sine(200, false);
    let fit = |boot: bool| {
        let b = Lowess::new().fraction(0.3).iterations(0).return_se();
        let b = if boot {
            b.intervals(IntervalsBuilder::new().bootstrap(400))
        } else {
            b
        };
        b.adapter(Batch)
            .build()
            .unwrap()
            .fit(&x, &y)
            .unwrap()
            .standard_errors
            .unwrap()
    };
    let (analytic, boot) = (fit(false), fit(true));
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let ratio = mean(&boot) / mean(&analytic);
    assert!(
        (0.5..2.0).contains(&ratio),
        "bootstrap/analytic SE ratio {ratio} out of range"
    );
}

/// Right-skewed noise produces a longer upper prediction-interval tail.
#[test]
fn test_bootstrap_prediction_interval_reflects_skew() {
    let (x, y) = noisy_sine(300, true);
    let res = Lowess::new()
        .fraction(0.3)
        .iterations(0)
        .intervals(IntervalsBuilder::new().prediction(0.9).bootstrap(300))
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();
    let (pl, pu) = (res.prediction_lower.unwrap(), res.prediction_upper.unwrap());
    let upper: f64 = pu.iter().zip(&res.y).map(|(u, f)| u - f).sum();
    let lower: f64 = res.y.iter().zip(&pl).map(|(f, l)| f - l).sum();
    assert!(
        upper > lower,
        "right-skewed noise should give a longer upper PI tail ({upper} vs {lower})"
    );
}

/// Fewer than 2 bootstrap replicates is rejected at build time.
#[test]
fn test_bootstrap_invalid_sample_count() {
    let err = Lowess::<f64>::new()
        .intervals(IntervalsBuilder::new().bootstrap(1))
        .adapter(Batch)
        .build()
        .err();
    assert_eq!(err, Some(LowessError::InvalidBootstrapSamples(1)));
}

/// Incremental Online updates cannot bootstrap the current window.
#[test]
fn test_bootstrap_online_requires_full_update() {
    let err = OnlineLowess::<f64>::new()
        .intervals(IntervalsBuilder::new().bootstrap(10))
        .build()
        .err();
    assert_eq!(err, Some(LowessError::StandardErrorRequiresFullUpdateMode));
}

#[test]
fn test_bootstrap_online_full_matches_batch_window() {
    let x: Vec<f64> = (0..15).map(|i| i as f64 * 0.2).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| xi.sin() + 0.2 * (xi * 7.0).sin())
        .collect();
    let mut online = OnlineLowess::new()
        .fraction(0.6)
        .iterations(0)
        .delta(0.0)
        .seed(7)
        .window_capacity(10)
        .min_points(5)
        .update_mode("full")
        .intervals(
            IntervalsBuilder::new()
                .confidence(0.95)
                .prediction(0.95)
                .bootstrap(40),
        )
        .build()
        .unwrap();

    for idx in 0..x.len() {
        let output = online.add_point(x[idx], y[idx]).unwrap();
        if idx + 1 < 5 {
            assert!(output.is_none());
            continue;
        }
        let output = output.unwrap();
        let start = (idx + 1).saturating_sub(10);
        let batch = Lowess::<f64>::new()
            .fraction(0.6)
            .iterations(0)
            .delta(0.0)
            .seed(7)
            .intervals(
                IntervalsBuilder::new()
                    .confidence(0.95)
                    .prediction(0.95)
                    .bootstrap(40),
            )
            .build()
            .unwrap()
            .fit(&x[start..=idx], &y[start..=idx])
            .unwrap();
        assert_eq!(output.y, *batch.y.last().unwrap());
        assert_eq!(
            output.standard_error,
            batch.standard_errors.unwrap().last().copied()
        );
        assert_eq!(
            output.confidence_lower,
            batch.confidence_lower.unwrap().last().copied()
        );
        assert_eq!(
            output.confidence_upper,
            batch.confidence_upper.unwrap().last().copied()
        );
        assert_eq!(
            output.prediction_lower,
            batch.prediction_lower.unwrap().last().copied()
        );
        assert_eq!(
            output.prediction_upper,
            batch.prediction_upper.unwrap().last().copied()
        );
    }
}

#[test]
fn test_bootstrap_online_invalid_samples() {
    let err = OnlineLowess::<f64>::new()
        .update_mode("full")
        .intervals(IntervalsBuilder::new().bootstrap(1))
        .build()
        .err();
    assert_eq!(err, Some(LowessError::InvalidBootstrapSamples(1)));
}

#[test]
fn test_bootstrap_online_se_without_interval_levels() {
    let mut online = OnlineLowess::new()
        .update_mode("full")
        .min_points(3)
        .seed(7)
        .intervals(IntervalsBuilder::new().bootstrap(20))
        .build()
        .unwrap();
    for idx in 0..5 {
        let x = idx as f64;
        let output = online.add_point(x, x.sin()).unwrap();
        if idx >= 2 {
            let output = output.unwrap();
            assert!(output.standard_error.unwrap().is_finite());
            assert!(output.confidence_lower.is_none());
            assert!(output.prediction_lower.is_none());
        }
    }
}

#[test]
fn test_bootstrap_streaming_matches_combined_batch_windows() {
    let x: Vec<f64> = (0..30).map(|i| i as f64 * 0.2).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| xi.sin() + 0.2 * (xi * 7.0).sin())
        .collect();
    let mut streaming = StreamingLowess::<f64>::new()
        .fraction(0.6)
        .iterations(0)
        .delta(0.0)
        .seed(7)
        .chunk_size(15)
        .overlap(3)
        .merge_strategy("take_last")
        .intervals(
            IntervalsBuilder::new()
                .confidence(0.95)
                .prediction(0.95)
                .bootstrap(40),
        )
        .build()
        .unwrap();

    for (start, end) in [(0, 15), (12, 30)] {
        let chunk_start = if start == 0 { 0 } else { 15 };
        let output = streaming
            .process_chunk(&x[chunk_start..end], &y[chunk_start..end])
            .unwrap();
        let batch = Lowess::<f64>::new()
            .fraction(0.6)
            .iterations(0)
            .delta(0.0)
            .seed(7)
            .intervals(
                IntervalsBuilder::new()
                    .confidence(0.95)
                    .prediction(0.95)
                    .bootstrap(40),
            )
            .build()
            .unwrap()
            .fit(&x[start..end], &y[start..end])
            .unwrap();
        let n = output.y.len();
        assert_eq!(output.x, batch.x[..n]);
        assert_eq!(
            output.standard_errors.as_ref().unwrap(),
            &batch.standard_errors.unwrap()[..n]
        );
        assert_eq!(
            output.confidence_lower.as_ref().unwrap(),
            &batch.confidence_lower.unwrap()[..n]
        );
        assert_eq!(
            output.confidence_upper.as_ref().unwrap(),
            &batch.confidence_upper.unwrap()[..n]
        );
        assert_eq!(
            output.prediction_lower.as_ref().unwrap(),
            &batch.prediction_lower.unwrap()[..n]
        );
        assert_eq!(
            output.prediction_upper.as_ref().unwrap(),
            &batch.prediction_upper.unwrap()[..n]
        );
    }
    let tail = streaming.finalize().unwrap();
    assert_eq!(tail.x, x[27..]);
    assert_eq!(tail.standard_errors.as_ref().unwrap().len(), 3);
    assert_eq!(tail.confidence_lower.as_ref().unwrap().len(), 3);
    assert_eq!(tail.prediction_upper.as_ref().unwrap().len(), 3);
}

#[test]
fn test_bootstrap_streaming_invalid_samples() {
    let err = StreamingLowess::<f64>::new()
        .intervals(IntervalsBuilder::new().bootstrap(1))
        .build()
        .err();
    assert_eq!(err, Some(LowessError::InvalidBootstrapSamples(1)));
}

#[test]
fn test_bootstrap_streaming_se_without_interval_levels() {
    let x: Vec<f64> = (0..12).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();
    let mut model = StreamingLowess::new()
        .chunk_size(12)
        .overlap(2)
        .seed(7)
        .intervals(IntervalsBuilder::new().bootstrap(20))
        .build()
        .unwrap();
    let output = model.process_chunk(&x, &y).unwrap();
    assert_eq!(
        output.standard_errors.as_ref().unwrap().len(),
        output.y.len()
    );
    assert!(output.confidence_lower.is_none());
    assert!(output.prediction_lower.is_none());
    assert_eq!(model.finalize().unwrap().standard_errors.unwrap().len(), 2);
}
