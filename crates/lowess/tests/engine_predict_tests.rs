#![cfg(feature = "dev")]
//! Tests for Batch out-of-sample prediction (`.retain_model()` + `LowessResult::predict()`).

use lowess::internals::engine::predict::{ExtrapolationPolicy, PredictOptions};
use lowess::prelude::*;

/// predict() must error when `.retain_model(true)` was not set before `fit()`.
#[test]
fn test_predict_unavailable_without_retain_model() {
    let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&v| v * 2.0).collect();

    let result = Lowess::new()
        .fraction(0.5)
        .build()
        .unwrap()
        .fit(&x, &y)
        .expect("fit should succeed");

    let err = result
        .predict(&[5.0], PredictOptions::default())
        .unwrap_err();
    assert_eq!(err, LowessError::PredictionUnavailable);
}

/// predict() at training x-values should closely match fit()'s y (same local fit,
/// same window), as a correctness sanity check.
#[test]
fn test_predict_matches_fit_at_training_points() {
    let n = 60;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&v| (v * 0.1).sin() + v * 0.02).collect();

    let result = Lowess::new()
        .fraction(0.3)
        .retain_model(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .expect("fit should succeed");

    let predicted = result
        .predict(&x, PredictOptions::default())
        .expect("predict should succeed");
    assert_eq!(predicted.y.len(), x.len());

    for (fitted, pred) in result.y.iter().zip(predicted.y.iter()) {
        assert!(
            (fitted - pred).abs() < 1e-6,
            "predict() at a training x should match fit()'s y: {fitted} vs {pred}"
        );
    }
}

/// Same as `test_predict_matches_fit_at_training_points`, but with a non-zero `delta`
/// (the default `delta = 0.0` never skips points, so it never exercises delta-interpolated
/// points). `predict()` reuses `fit()`'s own `y_smooth` curve for in-range points, so it
/// should still exactly reproduce `fit()`'s output at training points regardless of delta.
#[test]
fn test_predict_matches_fit_at_training_points_with_delta() {
    let n = 60;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&v| (v * 0.1).sin() + v * 0.02).collect();

    let result = Lowess::new()
        .fraction(0.3)
        .delta(2.0)
        .retain_model(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .expect("fit should succeed");

    let predicted = result
        .predict(&x, PredictOptions::default())
        .expect("predict should succeed");

    for (fitted, pred) in result.y.iter().zip(predicted.y.iter()) {
        assert!(
            (fitted - pred).abs() < 1e-10,
            "predict() at a training x should match fit()'s y even with delta-skipped \
             points: {fitted} vs {pred}"
        );
    }
}

/// predict() at a midpoint between two training x-values should land close to the
/// average of their fitted y-values, since LOWESS produces a locally smooth curve.
#[test]
fn test_predict_interpolates_smooth_function() {
    let n = 100;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * 0.2).collect();
    let y: Vec<f64> = x.iter().map(|&v| v.sin()).collect();

    let result = Lowess::new()
        .fraction(0.3)
        .retain_model(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .expect("fit should succeed");

    // Midpoints strictly between consecutive training x-values, away from the boundary.
    let mid_x: Vec<f64> = (20..n - 20).map(|i| (x[i] + x[i + 1]) / 2.0).collect();
    let predicted = result
        .predict(&mid_x, PredictOptions::default())
        .expect("predict should succeed");

    let max_err = (20..n - 20)
        .zip(predicted.y.iter())
        .map(|(i, &p)| {
            let neighbor_avg = (result.y[i] + result.y[i + 1]) / 2.0;
            (p - neighbor_avg).abs()
        })
        .fold(0.0_f64, f64::max);

    assert!(
        max_err < 0.05,
        "predict() at a midpoint should be close to its neighbors' average, got max err {max_err}"
    );
}

/// predict() at points outside the training range must not panic under the default
/// (Clamp) extrapolation policy.
#[test]
fn test_predict_out_of_range_clamp_does_not_panic() {
    let x: Vec<f64> = (0..30).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&v| v * 0.5).collect();

    let result = Lowess::new()
        .fraction(0.4)
        .retain_model(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .expect("fit should succeed");

    let predicted = result
        .predict(&[-100.0, -1.0, 1000.0], PredictOptions::default())
        .expect("predict should not error on out-of-range x under Clamp");
    assert!(predicted.y.iter().all(|v| v.is_finite()));
}

/// `ExtrapolationPolicy::Error` must fail the whole call when any query point is
/// outside the training x-range.
#[test]
fn test_predict_out_of_range_error_policy() {
    let x: Vec<f64> = (0..30).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&v| v * 0.5).collect();

    let result = Lowess::new()
        .fraction(0.4)
        .retain_model(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .expect("fit should succeed");

    let options = PredictOptions {
        extrapolation: ExtrapolationPolicy::Error,
        ..PredictOptions::default()
    };

    let err = result.predict(&[1000.0], options).unwrap_err();
    assert!(matches!(err, LowessError::PredictOutOfRange { .. }));

    // In-range queries must still succeed under the same policy.
    result
        .predict(&[5.0], options)
        .expect("in-range query should succeed under Error policy");
}

/// predict() must reject NaN/Inf query points instead of silently producing an
/// unspecified result (NaN comparisons against the training range are always false).
#[test]
fn test_predict_rejects_non_finite_new_x() {
    let x: Vec<f64> = (0..30).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&v| v * 0.5).collect();

    let result = Lowess::new()
        .fraction(0.4)
        .retain_model(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .expect("fit should succeed");

    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let err = result
            .predict(&[5.0, bad], PredictOptions::default())
            .unwrap_err();
        assert!(matches!(err, LowessError::InvalidNumericValue(_)));
    }
}

/// `ExtrapolationPolicy::Linear` should extend roughly linearly beyond the training
/// range, tracking a linear input function closely. Uses `boundary_policy("noboundary")`
/// so the boundary-region slope isn't biased by `Extend`'s flat-y padding (which exists
/// to stabilize *smoothing* at the edges, not to preserve the true local slope there).
#[test]
fn test_predict_out_of_range_linear_policy() {
    let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&v| 2.0 * v + 3.0).collect();

    let result = Lowess::new()
        .fraction(0.3)
        .boundary_policy("noboundary")
        .retain_model(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .expect("fit should succeed");

    let options = PredictOptions {
        extrapolation: ExtrapolationPolicy::Linear,
        ..PredictOptions::default()
    };

    let predicted = result
        .predict(&[60.0, 100.0], options)
        .expect("predict should succeed under Linear policy");

    // The underlying function is exactly linear, so linear extrapolation should track
    // it much more closely than plain clamping would (which would flatten at the edge).
    assert!(
        (predicted.y[0] - (2.0 * 60.0 + 3.0)).abs() < 5.0,
        "got {}",
        predicted.y[0]
    );
    assert!(
        (predicted.y[1] - (2.0 * 100.0 + 3.0)).abs() < 5.0,
        "got {}",
        predicted.y[1]
    );
}

/// `max_extrapolation_distance` should cap `ExtrapolationPolicy::Linear` instead of
/// returning an unbounded first-order Taylor extension.
#[test]
fn test_predict_extrapolation_linear_respects_max_distance() {
    let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&v| 2.0 * v + 3.0).collect();

    let result = Lowess::new()
        .fraction(0.3)
        .boundary_policy("noboundary")
        .retain_model(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .expect("fit should succeed");

    let options = PredictOptions {
        extrapolation: ExtrapolationPolicy::Linear,
        max_extrapolation_distance: Some(10.0),
        ..PredictOptions::default()
    };

    // Within the cap: still succeeds.
    result
        .predict(&[55.0], options)
        .expect("within max_extrapolation_distance should succeed");

    // Beyond the cap: errors instead of extrapolating unbounded.
    let err = result.predict(&[100.0], options).unwrap_err();
    assert!(matches!(err, LowessError::ExtrapolationTooFar { .. }));

    // The cap is ignored under Clamp.
    let clamp_options = PredictOptions {
        extrapolation: ExtrapolationPolicy::Clamp,
        max_extrapolation_distance: Some(10.0),
        ..PredictOptions::default()
    };
    result
        .predict(&[100.0], clamp_options)
        .expect("max_extrapolation_distance should not apply under Clamp");
}

/// A query point can fall between two clusters of training data (a gap) and still pass
/// the `[min(x_train), max(x_train)]` range check, yet be far from any real training
/// point. `max_neighbor_distance` should catch this.
#[test]
fn test_predict_max_neighbor_distance_catches_1d_gap() {
    // Training data clustered at [0, 10] and [90, 100]: x=50 is in-range but sits in an
    // empty gap far from any actual training point.
    let mut x: Vec<f64> = (0..11).map(|i| i as f64).collect();
    x.extend((90..=100).map(|i| i as f64));
    let y: Vec<f64> = x.iter().map(|&v| v * 2.0).collect();

    let result = Lowess::new()
        .fraction(0.3)
        .retain_model(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .expect("fit should succeed");

    // No cap: the gap is silently treated as in-range (original behavior).
    result
        .predict(&[50.0], PredictOptions::default())
        .expect("uncapped predict should not error, even in the gap");

    // With a cap: the gap's local window is much farther than a point actually near
    // training data.
    let options = PredictOptions {
        max_neighbor_distance: Some(5.0),
        ..PredictOptions::default()
    };
    let err = result.predict(&[50.0], options).unwrap_err();
    assert!(matches!(err, LowessError::SparseNeighborhood { .. }));

    // A point actually near real training data has a tight local window and should
    // still succeed under the same cap.
    result
        .predict(&[5.0], options)
        .expect("a point near real training data should have a tight local window");
}

/// `return_derivative` should expose the local slope, which for a linear function should
/// closely match the true slope everywhere away from the boundary.
#[test]
fn test_predict_return_derivative() {
    let x: Vec<f64> = (0..60).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&v| 3.0 * v - 1.0).collect();

    let result = Lowess::new()
        .fraction(0.3)
        .retain_model(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .expect("fit should succeed");

    let options = PredictOptions {
        return_derivative: true,
        ..PredictOptions::default()
    };

    let predicted = result
        .predict(&[20.0, 30.0, 40.0], options)
        .expect("predict should succeed");

    let derivative = predicted.derivative.expect("derivative should be present");
    for &slope in &derivative {
        assert!(
            (slope - 3.0).abs() < 1e-6,
            "expected slope ~3.0, got {slope}"
        );
    }
}

/// `return_se`/`confidence_level`/`prediction_level` should populate the corresponding
/// output fields, with confidence intervals narrower than prediction intervals.
#[test]
fn test_predict_se_and_intervals() {
    let n = 80;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &v)| v * 0.1 + if i % 7 == 0 { 0.5 } else { 0.0 })
        .collect();

    let result = Lowess::new()
        .fraction(0.3)
        .retain_model(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .expect("fit should succeed");

    let options = PredictOptions {
        return_se: true,
        confidence_level: Some(0.95),
        prediction_level: Some(0.95),
        ..PredictOptions::default()
    };

    let new_x = vec![10.0, 25.0, 40.0, 55.0, 70.0];
    let predicted = result
        .predict(&new_x, options)
        .expect("predict should succeed");

    let se = predicted.standard_errors.expect("standard_errors present");
    assert_eq!(se.len(), new_x.len());
    assert!(se.iter().all(|&s| s.is_finite() && s >= 0.0));

    let cl = predicted.confidence_lower.expect("confidence_lower");
    let cu = predicted.confidence_upper.expect("confidence_upper");
    let pl = predicted.prediction_lower.expect("prediction_lower");
    let pu = predicted.prediction_upper.expect("prediction_upper");

    for i in 0..new_x.len() {
        assert!(cl[i] <= predicted.y[i] && predicted.y[i] <= cu[i]);
        assert!(pl[i] <= predicted.y[i] && predicted.y[i] <= pu[i]);
        // Prediction intervals must be at least as wide as confidence intervals.
        assert!(
            (pu[i] - pl[i]) >= (cu[i] - cl[i]) - 1e-9,
            "prediction interval should not be narrower than confidence interval at i={i}"
        );
    }
}
