#![cfg(feature = "dev")]
use approx::assert_abs_diff_eq;
use fastLowess::prelude::*;
use ndarray::Array1;

#[test]
fn test_standard_batch_sequential() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

    // Sequential fit
    let res = Lowess::new()
        .parallel(false)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    assert_eq!(res.y.len(), 5);
    // Linear data should be perfectly fitted
    assert_abs_diff_eq!(res.y[0], 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(res.y[4], 10.0, epsilon = 1e-6);
}

#[test]
fn test_standard_batch_parallel() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

    // Parallel fit
    let res = Lowess::new()
        .parallel(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    assert_eq!(res.y.len(), 5);
    assert_abs_diff_eq!(res.y[0], 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(res.y[4], 10.0, epsilon = 1e-6);
}

#[test]
fn test_ndarray_integration() {
    let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let y = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);

    // Fit with ndarray
    let res = Lowess::new()
        .parallel(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    assert_eq!(res.y.len(), 5);
    assert_abs_diff_eq!(res.y[0], 2.0, epsilon = 1e-6);
}

#[test]
fn test_robustness() {
    // Larger dataset to ensure robust statistics work (N=20)
    let n = 20;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let mut y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi).collect();

    // Add heavy outlier at index 10 (x=10)
    // Expected y=20, set to 100
    y[10] = 100.0;

    // Fit with robustness (Bisquare, 5 iterations)
    let res = Lowess::new()
        .fraction(0.5)
        .iterations(5)
        .robustness_method("bisquare")
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    // The smoothed value at x=10 should be close to 20.0, not 100.0
    // Without robustness, it would be pulled significantly higher.
    let smoothed_val = res.y[10];
    assert!(
        smoothed_val < 35.0,
        "Smoothed value {} is too high (outlier not suppressed, expected ~20)",
        smoothed_val
    );
    assert!(smoothed_val > 10.0);
}

#[test]
fn test_streaming_adapter() {
    let n = 100;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi).collect();

    let mut processor = StreamingLowess::new()
        .fraction(0.2)
        .chunk_size(20)
        .overlap(5)
        .build()
        .unwrap();

    let mut total_points = 0;

    // Process in two big chunks manually to simulate stream
    let split = 50;

    // First half
    let res1 = processor.process_chunk(&x[0..split], &y[0..split]).unwrap();
    total_points += res1.x.len();

    // Second half
    let res2 = processor.process_chunk(&x[split..n], &y[split..n]).unwrap();
    total_points += res2.x.len();

    // Finalize
    let res3 = processor.finalize().unwrap();
    total_points += res3.x.len();

    // Streaming adapter might output slightly fewer points due to windowing/edge effects depending on config,
    // but for simple linear data and these settings it should be close to N.
    // Ideally it's exactly N if boundary handling extends properly.
    // Let's just check we got *some* output and values are reasonable.
    assert!(total_points > 80);

    if !res1.y.is_empty() {
        // Note: With boundary padding in lowess v0.5.0, edge values may shift slightly
        // The smoothed value should still be close to the expected linear trend
        let expected_y = 2.0 * res1.x[0]; // y = 2x
        assert_abs_diff_eq!(res1.y[0], expected_y, epsilon = 5.0);
    }
}

#[test]
fn test_streaming_adapter_return_derivative() {
    let n = 30;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();

    let mut processor = StreamingLowess::new()
        .fraction(1.0)
        .iterations(0)
        .return_derivative()
        .chunk_size(20)
        .overlap(5)
        .build()
        .unwrap();

    let res1 = processor.process_chunk(&x[0..20], &y[0..20]).unwrap();
    let deriv1 = res1.derivative.expect("derivative should be present");
    for &d in &deriv1 {
        assert_abs_diff_eq!(d, 2.0, epsilon = 1e-9);
    }

    let res2 = processor.process_chunk(&x[20..n], &y[20..n]).unwrap();
    let deriv2 = res2.derivative.expect("derivative should be present");
    for &d in &deriv2 {
        assert_abs_diff_eq!(d, 2.0, epsilon = 1e-9);
    }

    let res3 = processor.finalize().unwrap();
    let deriv3 = res3.derivative.expect("derivative should be present");
    for &d in &deriv3 {
        assert_abs_diff_eq!(d, 2.0, epsilon = 1e-9);
    }
}

#[test]
fn test_streaming_adapter_return_se_and_intervals() {
    let n = 30;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();

    let mut processor = StreamingLowess::new()
        .fraction(0.9)
        .return_se()
        .confidence_intervals(0.95)
        .prediction_intervals(0.95)
        .chunk_size(20)
        .overlap(5)
        .build()
        .unwrap();

    let res1 = processor.process_chunk(&x[0..20], &y[0..20]).unwrap();
    let se = res1.standard_errors.expect("standard_errors should be present");
    let cl = res1.confidence_lower.expect("confidence_lower should be present");
    let cu = res1.confidence_upper.expect("confidence_upper should be present");
    let pl = res1.prediction_lower.expect("prediction_lower should be present");
    let pu = res1.prediction_upper.expect("prediction_upper should be present");
    assert_eq!(se.len(), res1.y.len());
    for i in 0..res1.y.len() {
        assert!(se[i].is_finite() && se[i] >= 0.0);
        assert!(cl[i] <= res1.y[i] && res1.y[i] <= cu[i]);
        assert!(pl[i] <= res1.y[i] && res1.y[i] <= pu[i]);
    }

    let res2 = processor.process_chunk(&x[20..n], &y[20..n]).unwrap();
    assert!(res2.standard_errors.is_some());
    assert!(res2.confidence_lower.is_some());
    assert!(res2.prediction_lower.is_some());
}

#[test]
fn test_streaming_return_derivative_parallel_matches_sequential() {
    let n = 40;
    let x: Vec<f64> = (0..n).map(|i| i as f64 + (i as f64 * 0.37).sin()).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + xi / 5.0).collect();

    let mut seq = StreamingLowess::new()
        .fraction(0.4)
        .return_derivative()
        .parallel(false)
        .chunk_size(20)
        .overlap(5)
        .build()
        .unwrap();
    let mut par = StreamingLowess::new()
        .fraction(0.4)
        .return_derivative()
        .parallel(true)
        .chunk_size(20)
        .overlap(5)
        .build()
        .unwrap();

    let seq1 = seq.process_chunk(&x[0..20], &y[0..20]).unwrap();
    let par1 = par.process_chunk(&x[0..20], &y[0..20]).unwrap();
    for (&s, &p) in seq1
        .derivative
        .as_ref()
        .unwrap()
        .iter()
        .zip(par1.derivative.as_ref().unwrap().iter())
    {
        assert_abs_diff_eq!(s, p, epsilon = 1e-9);
    }

    let seq2 = seq.process_chunk(&x[20..n], &y[20..n]).unwrap();
    let par2 = par.process_chunk(&x[20..n], &y[20..n]).unwrap();
    for (&s, &p) in seq2
        .derivative
        .as_ref()
        .unwrap()
        .iter()
        .zip(par2.derivative.as_ref().unwrap().iter())
    {
        assert_abs_diff_eq!(s, p, epsilon = 1e-9);
    }
}

#[test]
fn test_online_adapter() {
    let mut processor = OnlineLowess::new()
        .min_points(3)
        .window_capacity(10)
        .build()
        .unwrap();

    // 1st point (not enough)
    let out1 = processor.add_point(1.0, 2.0).unwrap();
    assert!(out1.is_none());

    // 2nd point (not enough)
    let out2 = processor.add_point(2.0, 4.0).unwrap();
    assert!(out2.is_none());

    // 3rd point (enough!)
    let out3 = processor.add_point(3.0, 6.0).unwrap();
    assert!(out3.is_some());
    let val = out3.unwrap();
    assert_abs_diff_eq!(val.y, 6.0, epsilon = 0.1);

    // Bulk add
    let x_bulk = vec![4.0, 5.0];
    let y_bulk = vec![8.0, 10.0];
    let results = processor.add_points(&x_bulk, &y_bulk).unwrap();
    assert_eq!(results.len(), 2);
    assert!(results[0].is_some());
    assert!(results[1].is_some());
}

#[test]
fn test_online_adapter_return_derivative() {
    let mut processor = OnlineLowess::new()
        .fraction(1.0)
        .return_derivative()
        .min_points(2)
        .window_capacity(10)
        .build()
        .unwrap();

    let mut last = None;
    for i in 0..6 {
        last = processor.add_point(i as f64, 2.0 * i as f64 + 1.0).unwrap();
    }
    assert_abs_diff_eq!(last.unwrap().derivative.unwrap(), 2.0, epsilon = 1e-9);
}

#[test]
fn test_online_adapter_return_se_and_intervals_full_mode() {
    let mut processor = OnlineLowess::new()
        .fraction(0.9)
        .return_se()
        .confidence_intervals(0.95)
        .prediction_intervals(0.95)
        .update_mode("full")
        .min_points(10)
        .window_capacity(30)
        .build()
        .unwrap();

    let mut saw_all = false;
    for i in 0..20 {
        let x = i as f64;
        let y = 2.0 * x + 1.0;
        if let Some(output) = processor.add_point(x, y).unwrap()
            && output.standard_error.is_some()
            && output.confidence_lower.is_some()
            && output.prediction_lower.is_some()
        {
            saw_all = true;
        }
    }
    assert!(
        saw_all,
        "standard_error/confidence/prediction bounds should all be populated in Full mode"
    );
}

#[test]
fn test_online_adapter_se_requires_full_mode() {
    let result = OnlineLowess::new()
        .fraction(0.9)
        .return_se()
        .min_points(10)
        .window_capacity(30)
        .build();

    assert!(
        result.is_err(),
        "return_se() without update_mode(\"full\") should fail to build"
    );
}

#[test]
fn test_consistency() {
    // Verify that parallel and sequential computation yield identical results
    let n = 20;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + (xi / 10.0).exp()).collect();

    let seq_res = Lowess::new()
        .parallel(false)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let par_res = Lowess::new()
        .parallel(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    for i in 0..n {
        assert_abs_diff_eq!(seq_res.y[i], par_res.y[i], epsilon = 1e-10);
    }
}

#[test]
fn test_error_handling() {
    let x = vec![1.0, 2.0, 3.0];
    let y_short = vec![1.0, 2.0];

    let model = Lowess::new().build().unwrap();

    let err = model.fit(&x, &y_short);
    assert!(err.is_err());

    match err {
        Err(LowessError::MismatchedInputs { .. }) => (), // Expected
        _ => panic!("Expected MismatchedInputs error"),
    }
}

// ============================================================================
// Custom Weights Tests
// ============================================================================

/// Sequential and parallel runs with identical custom_weights produce the same result.
#[test]
fn test_custom_weights_parallel_matches_sequential() {
    let x: Vec<f64> = (0..30).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|v| v * 0.5 + (v * 0.3).sin()).collect();
    let weights: Vec<f64> = (0..30).map(|i| 1.0 + (i % 3) as f64).collect();

    let result_seq = Lowess::new()
        .fraction(0.4)
        .iterations(2)
        .custom_weights(weights.clone())
        .parallel(false)
        .build()
        .unwrap()
        .fit(&x, &y)
        .expect("sequential fit with custom_weights should succeed");

    let result_par = Lowess::new()
        .fraction(0.4)
        .iterations(2)
        .custom_weights(weights)
        .parallel(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .expect("parallel fit with custom_weights should succeed");

    assert_eq!(result_seq.y.len(), result_par.y.len());
    for (s, p) in result_seq.y.iter().zip(result_par.y.iter()) {
        assert_abs_diff_eq!(s, p, epsilon = 1e-10);
    }
}

/// Zeroing an outlier's weight reduces its influence under parallel execution.
#[test]
fn test_custom_weights_zero_weight_parallel() {
    let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let mut y: Vec<f64> = x.iter().map(|v| v * 2.0).collect();
    y[10] = 200.0; // outlier

    let mut weights = vec![1.0_f64; 20];
    weights[10] = 0.0;

    let result_no_w = Lowess::new()
        .fraction(0.5)
        .iterations(0)
        .parallel(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let result_zero_w = Lowess::new()
        .fraction(0.5)
        .iterations(0)
        .custom_weights(weights)
        .parallel(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let true_val = 10.0 * 2.0;
    let err_no_w = (result_no_w.y[10] - true_val).abs();
    let err_zero_w = (result_zero_w.y[10] - true_val).abs();

    assert!(
        err_zero_w < err_no_w,
        "zeroing outlier weight (parallel) should reduce error at that point \
         (err_no_weights={err_no_w:.2}, err_zero_weight={err_zero_w:.2})"
    );
}
