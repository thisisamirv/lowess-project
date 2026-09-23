#![cfg(feature = "dev")]
use approx::assert_abs_diff_eq;
use fastLowess::prelude::*;

#[test]
fn test_smooth_pass_consistency_robust() {
    let n = 50;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();

    // Sequential fit with 3 iterations
    let seq_res = Lowess::new()
        .fraction(0.3)
        .iterations(3)
        .parallel(false)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    // Parallel fit with 3 iterations
    let par_res = Lowess::new()
        .fraction(0.3)
        .iterations(3)
        .parallel(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    for i in 0..n {
        assert_abs_diff_eq!(seq_res.y[i], par_res.y[i], epsilon = 1e-12);
    }
    println!("Robust smooth pass consistency (3 iters): OK");
}

/// Pins both fastLowess paths to the sparse initial fit produced by
/// `stats::lowess(x, y, f = 0.525, iter = 0)`.
#[test]
fn test_parallel_matches_r_sparse_initial_fit() {
    let x = vec![3.572503, -6.211315, 0.0, 55.277736, 55.177418, 30.381348];
    let y = vec![0.0, 0.0, 1.282067, 0.0, 1.011657, 0.0];

    let sequential = Lowess::new()
        .fraction(0.525)
        .iterations(0)
        .boundary_policy("noboundary")
        .scaling_method("mar")
        .zero_weight_fallback("return_original")
        .outputs(["sorted"])
        .parallel(false)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let result = Lowess::new()
        .fraction(0.525)
        .iterations(0)
        .boundary_policy("noboundary")
        .scaling_method("mar")
        .zero_weight_fallback("return_original")
        .outputs(["sorted"])
        .parallel(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let expected_x = [-6.211315, 0.0, 3.572503, 30.381348, 55.177418, 55.277736];
    let expected_y = [
        0.0,
        1.282067,
        0.0,
        1.765_511_178_753_244e-6,
        0.5058285502438145,
        0.5058284503611018,
    ];
    for i in 0..expected_x.len() {
        assert_abs_diff_eq!(sequential.x[i], expected_x[i], epsilon = 1e-12);
        assert_abs_diff_eq!(sequential.y[i], expected_y[i], epsilon = 1e-6);
        assert_abs_diff_eq!(result.x[i], expected_x[i], epsilon = 1e-12);
        assert_abs_diff_eq!(result.y[i], expected_y[i], epsilon = 1e-6);
    }
    assert_eq!(result.y, sequential.y);
}

/// Verifies that the parallel `return_derivative` pass produces the same per-point
/// local fit slope as the sequential implementation, including delta-skipped
/// (interpolated) points.
#[test]
fn test_derivative_pass_consistency() {
    let n = 80;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| (xi * 0.1).sin() + xi * 0.02).collect();

    let seq_res = Lowess::new()
        .fraction(0.3)
        .delta(3.0) // force some points to be delta-skipped/interpolated
        .outputs(["derivative"])
        .parallel(false)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let par_res = Lowess::new()
        .fraction(0.3)
        .delta(3.0)
        .outputs(["derivative"])
        .parallel(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let seq_derivative = seq_res.derivative.expect("sequential derivative");
    let par_derivative = par_res.derivative.expect("parallel derivative");
    assert_eq!(seq_derivative.len(), par_derivative.len());

    for i in 0..n {
        assert_abs_diff_eq!(seq_derivative[i], par_derivative[i], epsilon = 1e-9);
    }
    println!("Derivative pass consistency: OK");
}
