#![cfg(feature = "dev")]
//! Tests for Batch out-of-sample prediction (`.retain_model()` + `Predict::call()`).

use approx::assert_relative_eq;
use lowess::internals::adapters::predict::Predict;
use lowess::internals::engine::executor::ExtrapolationPolicy;
use lowess::internals::evaluation::intervals::IntervalMethod;
use lowess::internals::math::boundary::BoundaryPolicy;
use lowess::internals::primitives::window::Window;
use lowess::prelude::*;

fn gaussian_se_over_all_observations(
    x: &[f64],
    y: &[f64],
    y_smooth: &[f64],
    window: &Window,
    x_query: f64,
    robustness_weights: &[f64],
) -> f64 {
    let bandwidth = window.max_distance(x, x_query);
    let (mut sum_w_r2, mut sum_w, mut s1, mut s2) = (0.0, 0.0, 0.0, 0.0);
    let (mut t0, mut t1, mut t2) = (0.0, 0.0, 0.0);

    for j in 0..x.len() {
        let dx = x[j] - x_query;
        let u = dx.abs() / bandwidth;
        let weight = (-0.5 * u * u).exp() * robustness_weights[j];
        let residual = y[j] - y_smooth[j];
        sum_w_r2 += weight * residual * residual;
        sum_w += weight;
        s1 += weight * dx;
        s2 += weight * dx * dx;
        t0 += weight * weight;
        t1 += weight * weight * dx;
        t2 += weight * weight * dx * dx;
    }

    IntervalMethod::<f64>::compute_se(sum_w, sum_w_r2, s1, s2, t0, t1, t2)
}

#[test]
fn test_gaussian_fit_standard_errors_use_all_observations() {
    let x: Vec<f64> = (0..9).map(f64::from).collect();
    let y: Vec<f64> = x.iter().map(|&value| value * value).collect();
    let fraction = 0.34;
    let result = Lowess::new()
        .fraction(fraction)
        .iterations(0)
        .weight_function("gaussian")
        .boundary_policy(BoundaryPolicy::NoBoundary)
        .intervals(IntervalsBuilder::new().confidence(0.95))
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let index = 4;
    let window_size = Window::calculate_span(x.len(), fraction);
    let mut window = Window::initialize(index, window_size, x.len());
    window.recenter(&x, index, x.len());
    let expected = gaussian_se_over_all_observations(
        &x,
        &y,
        &result.y,
        &window,
        x[index],
        &vec![1.0; x.len()],
    );

    assert_relative_eq!(
        result.standard_errors.as_ref().unwrap()[index],
        expected,
        epsilon = 1e-12
    );
}

#[test]
fn test_gaussian_prediction_standard_errors_use_all_observations() {
    let x: Vec<f64> = (0..9).map(f64::from).collect();
    let y: Vec<f64> = x.iter().map(|&value| value * value).collect();
    let fraction = 0.34;
    let result = Lowess::new()
        .fraction(fraction)
        .iterations(0)
        .weight_function("gaussian")
        .boundary_policy(BoundaryPolicy::NoBoundary)
        .retain_model(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let query = 4.5;
    let prediction = Predict::new()
        .return_se()
        .build()
        .unwrap()
        .call(&result, &[query])
        .unwrap();
    let state = result.fit_state.as_deref().unwrap();
    let seed = Window::locate(&state.x, query);
    let mut window = Window::initialize(seed, state.window_size, state.x.len());
    window.recenter_at(&state.x, query, state.x.len());
    let expected = gaussian_se_over_all_observations(
        &state.x,
        &state.y,
        &state.y_smooth,
        &window,
        query,
        &state.robustness_weights,
    );

    assert_relative_eq!(
        prediction.standard_errors.unwrap()[0],
        expected,
        epsilon = 1e-12
    );
}

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

    let err = Predict::new()
        .build()
        .unwrap()
        .call(&result, &[5.0])
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

    let predicted = Predict::new()
        .build()
        .unwrap()
        .call(&result, &x)
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

    let predicted = Predict::new()
        .build()
        .unwrap()
        .call(&result, &x)
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
    let predicted = Predict::new()
        .build()
        .unwrap()
        .call(&result, &mid_x)
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

    let predicted = Predict::new()
        .build()
        .unwrap()
        .call(&result, &[-100.0, -1.0, 1000.0])
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

    let options = Predict::new()
        .extrapolation(ExtrapolationPolicy::Error)
        .build()
        .unwrap();

    let err = options.call(&result, &[1000.0]).unwrap_err();
    assert!(matches!(err, LowessError::PredictOutOfRange { .. }));

    // In-range queries must still succeed under the same policy.
    options
        .call(&result, &[5.0])
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
        let err = Predict::new()
            .build()
            .unwrap()
            .call(&result, &[5.0, bad])
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

    let options = Predict::new()
        .extrapolation(ExtrapolationPolicy::Linear)
        .build()
        .unwrap();

    let predicted = options
        .call(&result, &[60.0, 100.0])
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

    let options = Predict::new()
        .extrapolation(ExtrapolationPolicy::Linear)
        .max_extrapolation_distance(10.0)
        .build()
        .unwrap();

    // Within the cap: still succeeds.
    options
        .call(&result, &[55.0])
        .expect("within max_extrapolation_distance should succeed");

    // Beyond the cap: errors instead of extrapolating unbounded.
    let err = options.call(&result, &[100.0]).unwrap_err();
    assert!(matches!(err, LowessError::ExtrapolationTooFar { .. }));

    // The cap is ignored under Clamp.
    let clamp_options = Predict::new()
        .extrapolation(ExtrapolationPolicy::Clamp)
        .max_extrapolation_distance(10.0)
        .build()
        .unwrap();
    clamp_options
        .call(&result, &[100.0])
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
    Predict::new()
        .build()
        .unwrap()
        .call(&result, &[50.0])
        .expect("uncapped predict should not error, even in the gap");

    // With a cap: the gap's local window is much farther than a point actually near
    // training data.
    let options = Predict::new().max_neighbor_distance(5.0).build().unwrap();
    let err = options.call(&result, &[50.0]).unwrap_err();
    assert!(matches!(err, LowessError::SparseNeighborhood { .. }));

    // A point actually near real training data has a tight local window and should
    // still succeed under the same cap.
    options
        .call(&result, &[5.0])
        .expect("a point near real training data should have a tight local window");
}

/// `outputs(["derivative"])` should expose the local slope, which for a linear function should
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

    let options = Predict::new().outputs(["derivative"]).build().unwrap();

    let predicted = options
        .call(&result, &[20.0, 30.0, 40.0])
        .expect("predict should succeed");

    let derivative = predicted.derivative.expect("derivative should be present");
    for &slope in &derivative {
        assert!(
            (slope - 3.0).abs() < 1e-6,
            "expected slope ~3.0, got {slope}"
        );
    }
}

/// `"se"`/`confidence_intervals`/`prediction_intervals` should populate the corresponding
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

    let options = Predict::new()
        .outputs(["se"])
        .confidence_intervals(0.95)
        .prediction_intervals(0.95)
        .build()
        .unwrap();

    let new_x = vec![10.0, 25.0, 40.0, 55.0, 70.0];
    let predicted = options
        .call(&result, &new_x)
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

#[test]
fn test_predict_standard_errors_include_custom_weights() {
    let x: Vec<f64> = (0..30).map(f64::from).collect();
    let mut y: Vec<f64> = x
        .iter()
        .map(|&value| 0.3 * value + (0.7 * value).sin())
        .collect();
    y[14] += 10.0;
    let mut custom_weights = vec![1.0; x.len()];
    custom_weights[14] = 0.05;
    let query = 12.5;

    let result = Lowess::new()
        .fraction(0.5)
        .boundary_policy(BoundaryPolicy::NoBoundary)
        .custom_weights(custom_weights)
        .retain_model(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();
    let prediction = Predict::new()
        .return_se()
        .build()
        .unwrap()
        .call(&result, &[query])
        .unwrap();

    let state = result.fit_state.as_deref().unwrap();
    let seed = Window::locate(&state.x, query);
    let mut window = Window::initialize(seed, state.window_size, state.x.len());
    window.recenter_at(&state.x, query, state.x.len());
    let expected = IntervalMethod::<f64>::compute_se_at_query(
        &state.x,
        &state.y,
        &state.y_smooth,
        &window,
        query,
        &state.robustness_weights,
        state.custom_weights.as_deref(),
        &|u| state.weight_function.compute_weight(u),
    );
    assert_relative_eq!(
        prediction.standard_errors.unwrap()[0],
        expected,
        epsilon = 1e-12
    );
}

#[test]
fn test_predict_bootstrap_intervals_are_seeded_at_query_points() {
    let x: Vec<f64> = (0..60).map(|i| i as f64 * 0.1).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| xi.sin() + 0.15 * (xi * 9.0).cos())
        .collect();
    let fitted = Lowess::new()
        .fraction(0.35)
        .iterations(1)
        .retain_model(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();
    let predict = |seed| {
        Predict::new()
            .intervals(
                IntervalsBuilder::new()
                    .confidence(0.9)
                    .prediction(0.9)
                    .bootstrap(40),
            )
            .seed(seed)
            .outputs(["derivative"])
            .extrapolation("linear")
            .build()
            .unwrap()
            .call(&fitted, &[0.55, 2.25, 6.1])
            .unwrap()
    };
    let first = predict(41);
    let again = predict(41);
    let other = predict(42);
    assert_eq!(first.standard_errors, again.standard_errors);
    assert_eq!(first.confidence_lower, again.confidence_lower);
    assert_eq!(first.prediction_upper, again.prediction_upper);
    assert_ne!(first.standard_errors, other.standard_errors);
    assert!(first.derivative.is_some());
    for values in [
        first.standard_errors.unwrap(),
        first.confidence_lower.unwrap(),
        first.confidence_upper.unwrap(),
        first.prediction_lower.unwrap(),
        first.prediction_upper.unwrap(),
    ] {
        assert_eq!(values.len(), 3);
        assert!(values.iter().all(|value| value.is_finite()));
    }
}

#[test]
fn test_predict_accepts_different_confidence_and_prediction_levels() {
    let x: Vec<f64> = (0..40).map(f64::from).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&value| {
            (0.25 * value).sin()
                + if (value as usize).is_multiple_of(9) {
                    0.4
                } else {
                    0.0
                }
        })
        .collect();
    let fitted = Lowess::new()
        .fraction(0.4)
        .retain_model(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();
    let options = Predict::new()
        .confidence_intervals(0.9)
        .prediction_intervals(0.99)
        .build()
        .unwrap();
    let result = options.call(&fitted, &[20.5]).unwrap();
    let confidence_width =
        result.confidence_upper.unwrap()[0] - result.confidence_lower.unwrap()[0];
    let prediction_width =
        result.prediction_upper.unwrap()[0] - result.prediction_lower.unwrap()[0];
    assert!(prediction_width > confidence_width);
}

#[test]
fn test_predict_bootstrap_se_only_and_validation() {
    assert!(matches!(
        Predict::<f64>::new()
            .intervals(IntervalsBuilder::new().bootstrap(1))
            .build(),
        Err(LowessError::InvalidBootstrapSamples(1))
    ));
    let x: Vec<f64> = (0..30).map(|i| i as f64 * 0.1).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| xi.sin() + 0.1 * (xi * 7.0).cos())
        .collect();
    let retained = Lowess::new()
        .retain_model(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();
    let plain = Lowess::new().build().unwrap().fit(&x, &y).unwrap();
    let options = Predict::new()
        .seed(5)
        .intervals(IntervalsBuilder::new().bootstrap(12))
        .build()
        .unwrap();
    let predicted = options.call(&retained, &[0.25, 1.25]).unwrap();
    assert_eq!(predicted.standard_errors.unwrap().len(), 2);
    assert!(predicted.confidence_lower.is_none());
    assert!(predicted.prediction_lower.is_none());
    assert!(matches!(
        options.call(&plain, &[0.25]),
        Err(LowessError::PredictionUnavailable)
    ));
}
