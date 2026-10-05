use core::ops::Range;

use lowess::internals::algorithms::robustness::RobustnessMethod;
use lowess::internals::evaluation::diagnostics::{Diagnostics, DiagnosticsState};
use lowess::internals::math::scaling::ScalingMethod;
use lowess::prelude::*;

fn assert_close(actual: f64, expected: f64, abs_tol: f64, rel_tol: f64) {
    assert!(actual.is_finite(), "actual value is not finite: {actual}");
    assert!(
        expected.is_finite(),
        "expected value is not finite: {expected}"
    );
    let error = (actual - expected).abs();
    let tolerance = abs_tol + rel_tol * expected.abs();
    assert!(
        error <= tolerance,
        "actual {actual:.17e}, expected {expected:.17e}, error {error:.3e} > {tolerance:.3e}"
    );
}

fn assert_slice_close(actual: &[f64], expected: &[f64], abs_tol: f64, rel_tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual_value, &expected_value)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual_value.is_finite(),
            "actual[{index}] is not finite: {actual_value}"
        );
        let error = (actual_value - expected_value).abs();
        let tolerance = abs_tol + rel_tol * expected_value.abs();
        assert!(
            error <= tolerance,
            "value[{index}] actual {actual_value:.17e}, expected {expected_value:.17e}, \
             error {error:.3e} > {tolerance:.3e}"
        );
    }
}

fn gaussian_se_reference(
    x: &[f64],
    y: &[f64],
    y_smooth: &[f64],
    x_query: f64,
    bandwidth: f64,
    observations: Range<usize>,
) -> f64 {
    let (mut sum_w_r2, mut sum_w, mut s1, mut s2) = (0.0, 0.0, 0.0, 0.0);
    let (mut t0, mut t1, mut t2) = (0.0, 0.0, 0.0);

    for index in observations {
        let dx = x[index] - x_query;
        let u = dx.abs() / bandwidth;
        let weight = (-0.5 * u * u).exp();
        let residual = y[index] - y_smooth[index];
        sum_w_r2 += weight * residual * residual;
        sum_w += weight;
        s1 += weight * dx;
        s2 += weight * dx * dx;
        t0 += weight * weight;
        t1 += weight * weight * dx;
        t2 += weight * weight * dx * dx;
    }

    let determinant = sum_w * s2 - s1 * s1;
    let leverage = (s2 * s2 * t0 - 2.0 * s1 * s2 * t1 + s1 * s1 * t2) / (determinant * determinant);
    let degrees_of_freedom = sum_w - 2.0 + t0 / sum_w;
    (sum_w_r2 / degrees_of_freedom * leverage).sqrt()
}

fn fit_curve(x: &[f64], y: &[f64], iterations: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let result = Lowess::new()
        .fraction(0.35)
        .iterations(iterations)
        .delta(0.0)
        .boundary_policy("noboundary")
        .outputs(["derivative", "weights"])
        .build()
        .unwrap()
        .fit(x, y)
        .unwrap();

    (
        result.y,
        result.derivative.unwrap(),
        result.robustness_weights.unwrap(),
    )
}

#[test]
fn gaussian_fit_and_query_standard_errors_use_full_kernel_support() {
    let x: Vec<f64> = (0..9).map(f64::from).collect();
    let y: Vec<f64> = x.iter().map(|&value| value * value).collect();
    let fraction = 0.34;
    let fit = Lowess::new()
        .fraction(fraction)
        .iterations(0)
        .delta(0.0)
        .weight_function("gaussian")
        .boundary_policy("noboundary")
        .intervals(IntervalsBuilder::new().confidence(0.95))
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let training_index = 4;
    let training_bandwidth = 1.0;
    let training_reference = gaussian_se_reference(
        &x,
        &y,
        &fit.y,
        x[training_index],
        training_bandwidth,
        0..x.len(),
    );
    let training_window_only =
        gaussian_se_reference(&x, &y, &fit.y, x[training_index], training_bandwidth, 3..6);
    assert!((training_reference - training_window_only).abs() > 1e-3);
    assert_close(
        fit.standard_errors.as_ref().unwrap()[training_index],
        training_reference,
        1e-12,
        1e-12,
    );

    let retained_fit = Lowess::new()
        .fraction(fraction)
        .iterations(0)
        .delta(0.0)
        .weight_function("gaussian")
        .boundary_policy("noboundary")
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
        .call(&retained_fit, &[query])
        .unwrap();
    let query_reference = gaussian_se_reference(&x, &y, &retained_fit.y, query, 1.5, 0..x.len());
    let query_window_only = gaussian_se_reference(&x, &y, &retained_fit.y, query, 1.5, 4..7);
    assert!((query_reference - query_window_only).abs() > 1e-3);
    assert_close(
        prediction.standard_errors.as_ref().unwrap()[0],
        query_reference,
        1e-12,
        1e-12,
    );
}

#[test]
fn streaming_r_squared_handles_large_offsets_and_chunked_accumulation() {
    let offset = 1.0e12;
    let y = [offset, offset + 1.0, offset - 1.0];
    let y_smooth = [offset; 3];

    let mut one_update = DiagnosticsState::<f64>::new();
    one_update.update(&y, &y_smooth);
    let one_update = one_update.finalize();
    assert_close(one_update.r_squared, 0.0, 1e-12, 1e-12);
    assert_close(one_update.residual_sd, 1.0, 1e-12, 1e-12);

    let mut partitioned = DiagnosticsState::<f64>::new();
    partitioned.update(&y[..1], &y_smooth[..1]);
    partitioned.update(&y[1..], &y_smooth[1..]);
    let partitioned = partitioned.finalize();
    assert_close(partitioned.r_squared, one_update.r_squared, 1e-12, 1e-12);
    assert_close(partitioned.rmse, one_update.rmse, 1e-12, 1e-12);
    assert_close(partitioned.mae, one_update.mae, 1e-12, 1e-12);
    assert_close(
        partitioned.residual_sd,
        one_update.residual_sd,
        1e-12,
        1e-12,
    );

    let mut streaming = StreamingLowess::new()
        .fraction(1.0)
        .iterations(0)
        .chunk_size(10)
        .overlap(0)
        .boundary_policy("noboundary")
        .outputs(["diagnostics"])
        .build()
        .unwrap();
    let emitted = streaming.process_chunk(&[0.0; 3], &y).unwrap();
    let diagnostics = emitted.diagnostics.unwrap();
    assert_close(diagnostics.r_squared, 0.0, 1e-12, 1e-12);
}

#[test]
fn streaming_overlap_emits_each_point_once_and_diagnostics_match_emitted_data() {
    let x: Vec<f64> = (0..31).map(f64::from).collect();
    let mut y: Vec<f64> = x.iter().map(|&value| (0.3 * value).sin()).collect();
    y[16] += 3.0;

    let mut streaming = StreamingLowess::new()
        .fraction(0.3)
        .iterations(0)
        .chunk_size(15)
        .overlap(3)
        .boundary_policy("noboundary")
        .outputs(["diagnostics"])
        .build()
        .unwrap();

    let first = streaming.process_chunk(&x[..15], &y[..15]).unwrap();
    let second = streaming.process_chunk(&x[15..], &y[15..]).unwrap();
    let final_chunk = streaming.finalize().unwrap();
    let mut emitted_x = first.x;
    emitted_x.extend(second.x);
    emitted_x.extend(final_chunk.x);
    let mut emitted_y = first.y;
    emitted_y.extend(second.y);
    emitted_y.extend(final_chunk.y);

    assert_eq!(emitted_x, x);
    assert_eq!(emitted_y.len(), y.len());

    let residuals: Vec<f64> = emitted_x
        .iter()
        .zip(emitted_y.iter())
        .map(|(&xi, &yhat)| y[xi as usize] - yhat)
        .collect();
    let count = residuals.len() as f64;
    let mean_y = y.iter().sum::<f64>() / count;
    let ss_tot = y.iter().map(|value| (value - mean_y).powi(2)).sum::<f64>();
    let ss_res = residuals.iter().map(|value| value * value).sum::<f64>();
    let mean_residual = residuals.iter().sum::<f64>() / count;
    let residual_variance = residuals
        .iter()
        .map(|value| (value - mean_residual).powi(2))
        .sum::<f64>()
        / (count - 1.0);
    let diagnostics = final_chunk.diagnostics.unwrap();

    assert_close(diagnostics.rmse, (ss_res / count).sqrt(), 1e-12, 1e-12);
    assert_close(
        diagnostics.mae,
        residuals.iter().map(|value| value.abs()).sum::<f64>() / count,
        1e-12,
        1e-12,
    );
    assert_close(diagnostics.r_squared, 1.0 - ss_res / ss_tot, 1e-12, 1e-12);
    assert_close(
        diagnostics.residual_sd,
        residual_variance.sqrt(),
        1e-12,
        1e-12,
    );
}

#[test]
fn empty_mar_scale_range_stops_without_mutating_weights() {
    let residuals = [1.0_f64, 2.0];
    let mut weights = [0.25_f64, 0.75];
    let original_weights = weights;
    let mut scratch = [0.0_f64; 2];

    let degenerate = RobustnessMethod::Bisquare.apply_robustness_weights(
        &residuals,
        &mut weights,
        ScalingMethod::MAR,
        &mut scratch,
        0..0,
    );

    assert!(degenerate);
    assert_eq!(weights, original_weights);
}

#[test]
fn fits_obey_x_scale_translation_and_response_affine_invariants() {
    let x: Vec<f64> = (0..81).map(f64::from).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&value| 2.0 * value + 0.2 * (0.1 * value).sin())
        .collect();
    let (base_y, base_derivative, _) = fit_curve(&x, &y, 0);

    let translated_x: Vec<f64> = x.iter().map(|&value| value + 1.0e8).collect();
    let (translated_y, translated_derivative, _) = fit_curve(&translated_x, &y, 0);
    assert_slice_close(&translated_y, &base_y, 1e-7, 1e-8);
    assert_slice_close(&translated_derivative, &base_derivative, 1e-6, 1e-7);

    let x_scale = 1e-9;
    let scaled_x: Vec<f64> = x.iter().map(|&value| value * x_scale).collect();
    let (scaled_y, scaled_derivative, _) = fit_curve(&scaled_x, &y, 0);
    assert_slice_close(&scaled_y, &base_y, 1e-9, 1e-9);
    let normalized_derivative: Vec<f64> = scaled_derivative
        .iter()
        .map(|&value| value * x_scale)
        .collect();
    assert_slice_close(&normalized_derivative, &base_derivative, 1e-7, 1e-7);

    let mut contaminated_y = y.clone();
    contaminated_y[40] += 8.0;
    let (base_robust_y, base_robust_derivative, base_weights) = fit_curve(&x, &contaminated_y, 2);
    let response_scale = -3.25;
    let response_shift = 7.0;
    let transformed_y: Vec<f64> = contaminated_y
        .iter()
        .map(|&value| response_scale * value + response_shift)
        .collect();
    let (transformed_fit, transformed_derivative, transformed_weights) =
        fit_curve(&x, &transformed_y, 2);
    let expected_fit: Vec<f64> = base_robust_y
        .iter()
        .map(|&value| response_scale * value + response_shift)
        .collect();
    let expected_derivative: Vec<f64> = base_robust_derivative
        .iter()
        .map(|&value| response_scale * value)
        .collect();
    assert_slice_close(&transformed_fit, &expected_fit, 1e-7, 1e-7);
    assert_slice_close(&transformed_derivative, &expected_derivative, 1e-6, 1e-7);
    assert_slice_close(&transformed_weights, &base_weights, 1e-8, 1e-8);
}

#[test]
fn tied_x_sorting_and_permutation_preserve_the_fit() {
    let x = [4.0, 1.0, 2.0, 0.0, 2.0, 3.0, 1.0];
    let y = [7.0, -1.0, 3.0, 2.0, 4.0, 5.0, 0.0];
    let fit = |x: &[f64], y: &[f64], sorted: bool| {
        let mut builder = Lowess::new()
            .fraction(0.5)
            .iterations(0)
            .delta(0.0)
            .boundary_policy("noboundary");
        if sorted {
            builder = builder.outputs(["sorted"]);
        }
        builder.build().unwrap().fit(x, y).unwrap()
    };

    let original_order = fit(&x, &y, false);
    let sorted_order = fit(&x, &y, true);
    let mut permutation: Vec<usize> = (0..x.len()).collect();
    permutation.sort_by(|&left, &right| x[left].total_cmp(&x[right]));
    let expected_x: Vec<f64> = permutation.iter().map(|&index| x[index]).collect();
    assert_eq!(sorted_order.x, expected_x);
    for (sorted_index, &original_index) in permutation.iter().enumerate() {
        assert_close(
            original_order.y[original_index],
            sorted_order.y[sorted_index],
            1e-12,
            1e-12,
        );
    }

    let shuffled = [3, 6, 1, 5, 0, 4, 2];
    let shuffled_x: Vec<f64> = shuffled.iter().map(|&index| x[index]).collect();
    let shuffled_y: Vec<f64> = shuffled.iter().map(|&index| y[index]).collect();
    let shuffled_sorted = fit(&shuffled_x, &shuffled_y, true);
    assert_eq!(shuffled_sorted.x, sorted_order.x);
    assert_slice_close(&shuffled_sorted.y, &sorted_order.y, 1e-12, 1e-12);
}

#[test]
fn delta_interpolation_preserves_an_exact_line() {
    let x: Vec<f64> = (0..41).map(|index| index as f64 / 40.0).collect();
    let y: Vec<f64> = x.iter().map(|&value| 3.0 * value - 2.0).collect();

    for delta in [0.0, 0.2] {
        let result = Lowess::new()
            .fraction(0.35)
            .iterations(0)
            .delta(delta)
            .boundary_policy("noboundary")
            .outputs(["derivative"])
            .build()
            .unwrap()
            .fit(&x, &y)
            .unwrap();
        assert_slice_close(&result.y, &y, 1e-12, 1e-12);
        assert_slice_close(
            result.derivative.as_ref().unwrap(),
            &vec![3.0; x.len()],
            1e-12,
            1e-12,
        );
    }
}

#[test]
fn zero_custom_weights_follow_the_configured_fallback() {
    let x: Vec<f64> = (0..7).map(f64::from).collect();
    let y = [2.0, -1.0, 4.0, 8.0, 3.0, -2.0, 6.0];
    let zero_weights = vec![0.0; x.len()];

    let original = Lowess::new()
        .fraction(0.5)
        .iterations(0)
        .delta(0.0)
        .boundary_policy("noboundary")
        .zero_weight_fallback("return_original")
        .custom_weights(zero_weights.clone())
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();
    assert_eq!(original.y, y);

    let local_mean = Lowess::new()
        .fraction(0.5)
        .iterations(0)
        .delta(0.0)
        .boundary_policy("noboundary")
        .zero_weight_fallback("use_local_mean")
        .custom_weights(zero_weights)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();
    assert!(local_mean.y.iter().all(|value| value.is_finite()));
    assert_ne!(local_mean.y, y);
}

#[test]
fn f32_local_linear_fit_remains_finite_at_small_x_scale() {
    let scale = 1e-6_f32;
    let x: Vec<f32> = (0..40).map(|index| scale * index as f32 / 39.0).collect();
    let y: Vec<f32> = x.iter().map(|&value| 2.0 * value).collect();
    let result = Lowess::<f32>::new()
        .fraction(0.35)
        .iterations(0)
        .delta(0.0)
        .boundary_policy("noboundary")
        .outputs(["derivative"])
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    assert!(result.y.iter().all(|value| value.is_finite()));
    for &derivative in result.derivative.as_ref().unwrap() {
        assert!((derivative - 2.0).abs() < 1e-3, "slope={derivative}");
    }
}

#[test]
fn even_scales_remain_finite_when_the_inputs_are_near_f64_max() {
    let value = f64::MAX;
    for (method, expected) in [
        (ScalingMethod::MAR, value),
        (ScalingMethod::MAD, 0.0),
        (ScalingMethod::Mean, value),
    ] {
        let mut values = [value; 4];
        let actual = method.compute(&mut values);
        assert_close(actual, expected, 0.0, 1e-15);
    }
}

#[test]
fn batch_diagnostics_resist_overflow_in_squares_and_absolute_sums() {
    let scale = 1.0e154_f64;
    let y = [-scale, 0.0, scale];
    let y_smooth = [0.0; 3];

    assert_close(
        Diagnostics::calculate_rmse(&y, &y_smooth),
        scale * (2.0_f64 / 3.0).sqrt(),
        1e138,
        1e-12,
    );
    assert_close(
        Diagnostics::calculate_mae(&y, &y_smooth),
        2.0 * scale / 3.0,
        1e138,
        1e-12,
    );
    assert_close(
        Diagnostics::calculate_r_squared(&y, &y_smooth),
        0.0,
        1e-12,
        1e-12,
    );
}

#[test]
fn bisquare_weights_are_invariant_to_large_residual_scales() {
    let base_residuals = [0.0_f64, 0.0, 4.0, 8.0, 16.0];
    let scaled_residuals: Vec<f64> = base_residuals
        .iter()
        .map(|&residual| residual * 1.0e307)
        .collect();
    let weights_for = |residuals: &[f64]| {
        let mut weights = vec![1.0; residuals.len()];
        let mut scratch = vec![0.0; residuals.len()];
        RobustnessMethod::Bisquare.apply_robustness_weights(
            residuals,
            &mut weights,
            ScalingMethod::MAR,
            &mut scratch,
            0..residuals.len(),
        );
        weights
    };

    let expected = weights_for(&base_residuals);
    let actual = weights_for(&scaled_residuals);
    assert_slice_close(&actual, &expected, 1e-12, 1e-12);
}

#[test]
fn aic_stays_finite_when_raw_residual_squares_overflow() {
    let residuals = [1.0e154_f64, -1.0e154];
    let aic = Diagnostics::calculate_aic(&residuals, 2.0);
    let expected = 4.0 * 1.0e154_f64.ln() + 4.0;

    assert_close(aic, expected, 1e-12, 1e-12);
}

#[test]
fn common_large_custom_weight_scaling_preserves_the_fit() {
    let x: Vec<f64> = (0..9).map(f64::from).collect();
    let y: Vec<f64> = x.iter().map(|&value| (0.4 * value).sin() + value).collect();
    let fit_with_weights = |weights| {
        Lowess::new()
            .fraction(0.5)
            .iterations(0)
            .delta(0.0)
            .boundary_policy("noboundary")
            .custom_weights(weights)
            .build()
            .unwrap()
            .fit(&x, &y)
            .unwrap()
            .y
    };

    let unit_weights = fit_with_weights(vec![1.0; x.len()]);
    let large_weights = fit_with_weights(vec![1.0e308; x.len()]);
    assert_slice_close(&large_weights, &unit_weights, 1e-12, 1e-12);

    let tied_x = [3.0_f64; 3];
    let tied_y = [1.0_f64, 2.0, 6.0];
    let fit_tied_with_weights = |weight| {
        Lowess::new()
            .fraction(0.5)
            .iterations(0)
            .boundary_policy("noboundary")
            .custom_weights(vec![weight; tied_x.len()])
            .build()
            .unwrap()
            .fit(&tied_x, &tied_y)
            .unwrap()
            .y
    };
    let tied_unit = fit_tied_with_weights(1.0);
    let tied_large = fit_tied_with_weights(1.0e308);
    assert_slice_close(&tied_large, &tied_unit, 1e-12, 1e-12);
}

#[test]
fn batch_and_streaming_r_squared_center_values_before_accumulating() {
    let base = 9_007_199_254_740_992.0_f64;
    let y = [base, base + 2.0, base + 2.0];
    let y_smooth = [base; 3];

    // Offsets from base are [0, 2, 2], so SS_tot = 8/3 and SS_res = 8.
    let batch_r_squared = Diagnostics::calculate_r_squared(&y, &y_smooth);
    assert_close(batch_r_squared, -2.0, 1e-12, 1e-12);

    let mut state = DiagnosticsState::<f64>::new();
    state.update(&y, &y_smooth);
    assert_close(state.finalize().r_squared, -2.0, 1e-12, 1e-12);
}
