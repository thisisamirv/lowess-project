#![cfg(feature = "dev")]
use approx::assert_abs_diff_eq;
use fastLowess::prelude::*;
use lowess::internals::engine::predict::Predict;

/// Parallel and sequential predict() must produce identical results.
#[test]
fn test_predict_pass_consistency() {
    let n = 80;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| (xi * 0.1).sin()).collect();
    let new_x: Vec<f64> = (0..20).map(|i| 0.5 + i as f64 * 3.7).collect();

    let seq_res = Lowess::new()
        .fraction(0.3)
        .parallel(false)
        .retain_model(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let par_res = Lowess::new()
        .fraction(0.3)
        .parallel(true)
        .retain_model(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let options = Predict::new()
        .return_se()
        .return_derivative()
        .confidence_intervals(0.95)
        .prediction_intervals(0.95)
        .build()
        .unwrap();

    let seq_pred = options.call(&seq_res, &new_x).unwrap();
    let par_pred = options.call(&par_res, &new_x).unwrap();

    for i in 0..new_x.len() {
        assert_abs_diff_eq!(seq_pred.y[i], par_pred.y[i], epsilon = 1e-12);
        assert_abs_diff_eq!(
            seq_pred.standard_errors.as_ref().unwrap()[i],
            par_pred.standard_errors.as_ref().unwrap()[i],
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            seq_pred.derivative.as_ref().unwrap()[i],
            par_pred.derivative.as_ref().unwrap()[i],
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            seq_pred.confidence_lower.as_ref().unwrap()[i],
            par_pred.confidence_lower.as_ref().unwrap()[i],
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            seq_pred.prediction_upper.as_ref().unwrap()[i],
            par_pred.prediction_upper.as_ref().unwrap()[i],
            epsilon = 1e-12
        );
    }
}
