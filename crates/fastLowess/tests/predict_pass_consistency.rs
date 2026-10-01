#![cfg(feature = "dev")]
use approx::assert_abs_diff_eq;
use fastLowess::prelude::*;
use lowess::internals::adapters::predict::Predict;

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
        .outputs(["se", "derivative"])
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

#[test]
fn test_parallel_bootstrap_predict_matches_sequential() {
    let x: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| xi.sin() + 0.12 * (xi * 8.0).cos())
        .collect();
    let fit = |parallel| {
        Lowess::new()
            .fraction(0.35)
            .parallel(parallel)
            .retain_model(true)
            .build()
            .unwrap()
            .fit(&x, &y)
            .unwrap()
    };
    let options = Predict::new()
        .intervals(
            IntervalsBuilder::new()
                .confidence(0.9)
                .prediction(0.9)
                .bootstrap(20),
        )
        .seed(41)
        .build()
        .unwrap();
    let queries = [0.25, 1.75, 3.75];
    let sequential = options.call(&fit(false), &queries).unwrap();
    let parallel = options.call(&fit(true), &queries).unwrap();
    assert_eq!(sequential.standard_errors, parallel.standard_errors);
    assert_eq!(sequential.confidence_lower, parallel.confidence_lower);
    assert_eq!(sequential.prediction_upper, parallel.prediction_upper);
}

#[cfg(feature = "gpu")]
#[test]
fn test_gpu_backed_bootstrap_predict() {
    let x: Vec<f64> = (0..30).map(|i| i as f64 * 0.1).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| xi.sin() + 0.1 * (xi * 8.0).cos())
        .collect();
    let fitted = Lowess::new()
        .fraction(0.35)
        .backend("gpu")
        .retain_model(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();
    let predicted = Predict::new()
        .intervals(
            IntervalsBuilder::new()
                .confidence(0.9)
                .prediction(0.9)
                .bootstrap(12),
        )
        .seed(51)
        .build()
        .unwrap()
        .call(&fitted, &[0.25, 1.25])
        .unwrap();
    assert_eq!(predicted.standard_errors.unwrap().len(), 2);
    assert_eq!(predicted.confidence_lower.unwrap().len(), 2);
    assert_eq!(predicted.prediction_upper.unwrap().len(), 2);
}
