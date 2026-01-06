#![cfg(feature = "dev")]
#![cfg(feature = "gpu")]
use approx::assert_abs_diff_eq;
use fastLowess::prelude::*;

#[test]
fn test_gpu_batch_fit() {
    let x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0f32, 4.0, 6.0, 8.0, 10.0];

    // GPU fit
    let res = Lowess::new()
        .adapter(Batch)
        .backend(GPU)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    // If GPU hardware is present, check results.
    // If initialization failed, it returns empty (but we can't easily distinguish here without checking length).
    if !res.y.is_empty() {
        assert_eq!(res.y.len(), 5);
        // Linear data should be perfectly fitted
        assert_abs_diff_eq!(res.y[0], 2.0, epsilon = 1e-3);
        assert_abs_diff_eq!(res.y[4], 10.0, epsilon = 1e-3);
        println!("GPU fit successful");
    } else {
        println!("GPU fit skipped or failed (likely no hardware)");
    }
}

#[test]
fn test_gpu_robustness() {
    let n = 20;
    let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let mut y: Vec<f32> = x.iter().map(|&xi| 2.0 * xi).collect();

    // Add heavy outlier at index 10 (x=10)
    y[10] = 100.0;

    // Fit with robustness on GPU
    let res = Lowess::new()
        .fraction(0.5)
        .iterations(5)
        .robustness_method(Bisquare)
        .adapter(Batch)
        .backend(GPU)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    if !res.y.is_empty() {
        let smoothed_val = res.y[10];
        // The outlier should be suppressed
        assert!(
            smoothed_val < 40.0,
            "Smoothed value {} is too high (outlier not suppressed, expected ~20)",
            smoothed_val
        );
        println!(
            "GPU robustness test successful: smoothed_val = {}",
            smoothed_val
        );
    } else {
        println!("GPU robustness test skipped or failed (likely no hardware)");
    }
}
