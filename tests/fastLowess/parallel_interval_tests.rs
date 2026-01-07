#![cfg(feature = "dev")]
use approx::assert_abs_diff_eq;
use fastLowess::prelude::*;
use ndarray::Array1;

#[test]
fn test_parallel_interval_estimation() {
    // Generate sample data
    let n = 100;
    let x_vec: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
    let y_vec: Vec<f64> = x_vec
        .iter()
        .map(|&xi| xi.sin() + 0.1 * (xi * 10.0).sin())
        .collect();

    let x = Array1::from_vec(x_vec);
    let y = Array1::from_vec(y_vec);

    // Run Sequential Intervals
    let seq_model = Lowess::new()
        .fraction(0.3)
        .iterations(2)
        .confidence_intervals(0.95)
        .prediction_intervals(0.95)
        .adapter(Batch)
        .parallel(false)
        .build()
        .unwrap();

    let seq_result = seq_model.fit(&x, &y).unwrap();

    // Run Parallel Intervals
    let par_model = Lowess::new()
        .fraction(0.3)
        .iterations(2)
        .confidence_intervals(0.95)
        .prediction_intervals(0.95)
        .adapter(Batch)
        .parallel(true)
        .build()
        .unwrap();

    let par_result = par_model.fit(&x, &y).unwrap();

    // Compare results
    assert_eq!(par_result.y, seq_result.y);

    let par_std_err = par_result.standard_errors.as_ref().unwrap();
    let seq_std_err = seq_result.standard_errors.as_ref().unwrap();

    for (p, s) in par_std_err.iter().zip(seq_std_err.iter()) {
        assert_abs_diff_eq!(p, s, epsilon = 1e-10);
    }

    let par_conf_lower = par_result.confidence_lower.as_ref().unwrap();
    let seq_conf_lower = seq_result.confidence_lower.as_ref().unwrap();
    for (p, s) in par_conf_lower.iter().zip(seq_conf_lower.iter()) {
        assert_abs_diff_eq!(p, s, epsilon = 1e-10);
    }

    let par_pred_lower = par_result.prediction_lower.as_ref().unwrap();
    let seq_pred_lower = seq_result.prediction_lower.as_ref().unwrap();
    for (p, s) in par_pred_lower.iter().zip(seq_pred_lower.iter()) {
        assert_abs_diff_eq!(p, s, epsilon = 1e-10);
    }

    println!("Parallel and Sequential Intervals match exactly!");
}
