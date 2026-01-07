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
        .adapter(Batch)
        .parallel(false)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    // Parallel fit with 3 iterations
    let par_res = Lowess::new()
        .fraction(0.3)
        .iterations(3)
        .adapter(Batch)
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
