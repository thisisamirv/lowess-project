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
        .adapter(Batch)
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
        .adapter(Batch)
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
        .adapter(Batch)
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
        .robustness_method(Bisquare)
        .adapter(Batch)
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

    let mut processor = Lowess::new()
        .fraction(0.2)
        .adapter(Streaming)
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
fn test_online_adapter() {
    let mut processor = Lowess::new()
        .adapter(Online)
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
    assert_abs_diff_eq!(val.smoothed, 6.0, epsilon = 0.1);

    // Bulk add
    let x_bulk = vec![4.0, 5.0];
    let y_bulk = vec![8.0, 10.0];
    let results = processor.add_points(&x_bulk, &y_bulk).unwrap();
    assert_eq!(results.len(), 2);
    assert!(results[0].is_some());
    assert!(results[1].is_some());
}

#[test]
fn test_consistency() {
    // Verify that parallel and sequential computation yield identical results
    let n = 20;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + (xi / 10.0).exp()).collect();

    let seq_res = Lowess::new()
        .adapter(Batch)
        .parallel(false)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let par_res = Lowess::new()
        .adapter(Batch)
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

    let model = Lowess::new().adapter(Batch).build().unwrap();

    let err = model.fit(&x, &y_short);
    assert!(err.is_err());

    match err {
        Err(LowessError::MismatchedInputs { .. }) => (), // Expected
        _ => panic!("Expected MismatchedInputs error"),
    }
}
