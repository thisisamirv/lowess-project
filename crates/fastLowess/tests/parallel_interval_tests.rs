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

/// Parallel bootstrap refits must reproduce the sequential bootstrap exactly, since
/// resampling happens before the refits are scheduled. 300 replicates spans two batches.
#[test]
fn test_parallel_bootstrap_matches_sequential() {
    let n = 120;
    let x_vec: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
    let y_vec: Vec<f64> = x_vec
        .iter()
        .map(|&xi| xi.sin() + 0.2 * (xi * 7.0).sin())
        .collect();
    let x = Array1::from_vec(x_vec);
    let y = Array1::from_vec(y_vec);

    let fit = |parallel: bool| {
        Lowess::new()
            .fraction(0.3)
            .iterations(1)
            .confidence_intervals(0.95)
            .prediction_intervals(0.9)
            .bootstrap_intervals(300)
            .bootstrap_seed(11)
            .parallel(parallel)
            .build()
            .unwrap()
            .fit(&x, &y)
            .unwrap()
    };
    let (seq, par) = (fit(false), fit(true));

    assert_eq!(par.y, seq.y);
    assert_eq!(par.standard_errors, seq.standard_errors);
    assert_eq!(par.confidence_lower, seq.confidence_lower);
    assert_eq!(par.confidence_upper, seq.confidence_upper);
    assert_eq!(par.prediction_lower, seq.prediction_lower);
    assert_eq!(par.prediction_upper, seq.prediction_upper);
    assert!(par.standard_errors.unwrap().iter().all(|&s| s > 0.0));
}

#[test]
fn test_streaming_bootstrap_parallel_matches_sequential() {
    let x: Vec<f64> = (0..40).map(|i| i as f64 * 0.2).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| xi.sin() + 0.1 * (xi * 5.0).sin())
        .collect();
    let fit = |parallel| {
        let mut model = fastLowess::prelude::StreamingLowess::new()
            .fraction(0.6)
            .iterations(0)
            .chunk_size(20)
            .overlap(4)
            .confidence_intervals(0.95)
            .prediction_intervals(0.9)
            .bootstrap_intervals(40)
            .bootstrap_seed(11)
            .parallel(parallel)
            .build()
            .unwrap();
        let first = model.process_chunk(&x[..20], &y[..20]).unwrap();
        let second = model.process_chunk(&x[20..], &y[20..]).unwrap();
        let tail = model.finalize().unwrap();
        [first, second, tail]
    };

    let sequential = fit(false);
    let parallel = fit(true);
    for (seq, par) in sequential.iter().zip(parallel.iter()) {
        assert_eq!(par.y, seq.y);
        assert_eq!(par.standard_errors, seq.standard_errors);
        assert_eq!(par.confidence_lower, seq.confidence_lower);
        assert_eq!(par.confidence_upper, seq.confidence_upper);
        assert_eq!(par.prediction_lower, seq.prediction_lower);
        assert_eq!(par.prediction_upper, seq.prediction_upper);
    }
}
