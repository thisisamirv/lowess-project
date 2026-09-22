#![cfg(feature = "dev")]
//! Statistical regression tests: numeric robustness across data scales and
//! cross-validation aggregation consistency.
//!
//! These guard concrete defects found while auditing the numerics:
//!  * the WLS/OLS degeneracy check used an *absolute* tolerance, which silently
//!    zeroed the local-linear (and global OLS) slope whenever the x-values were
//!    small in magnitude, degrading the fit to a local/global mean;
//!  * k-fold CV averaged per-fold RMSEs instead of pooling, so it disagreed with
//!    LOOCV even when the folds were identical (k == n).

use lowess::prelude::*;

fn lcg_noise(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u1 = ((s >> 32) as f64) / (u32::MAX as f64) + 1e-12;
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u2 = ((s >> 32) as f64) / (u32::MAX as f64);
        out.push((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos());
    }
    out
}

/// A local-linear fit reproduces a straight line exactly in the interior for any x
/// magnitude. An absolute degeneracy tolerance in `solve_wls` used to zero the
/// slope for x-scales at or below ~1e-4.
#[test]
fn wls_slope_is_scale_invariant() {
    let n = 200;
    let slope = 2.0;
    for scale in [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-12] {
        let x: Vec<f64> = (0..n).map(|i| scale * i as f64 / (n - 1) as f64).collect();
        // Offset-free y so the test isolates the degeneracy tolerance rather than
        // the (separate) conditioning effect of a large y offset.
        let y: Vec<f64> = x.iter().map(|&xi| slope * xi).collect();
        let res = Lowess::new()
            .fraction(0.3)
            .iterations(0)
            .outputs(["derivative"])
            .build()
            .unwrap()
            .fit(&x, &y)
            .unwrap();
        let d = res.derivative.unwrap();
        for (i, &di) in d.iter().enumerate().take(3 * n / 4).skip(n / 4) {
            assert!(
                (di - slope).abs() < 1e-6,
                "scale={scale:e}: derivative[{i}]={di}, expected {slope}"
            );
        }
    }
}

/// The `fraction >= 1.0` global OLS branch similarly preserves the slope at any x
/// magnitude (same absolute-tolerance defect in `fit_ols`).
#[test]
fn ols_slope_is_scale_invariant() {
    let n = 100;
    let slope = 3.0;
    for scale in [1.0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12] {
        let x: Vec<f64> = (0..n).map(|i| scale * i as f64 / (n - 1) as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| slope * xi).collect();
        let res = Lowess::new()
            .fraction(1.0)
            .outputs(["derivative"])
            .build()
            .unwrap()
            .fit(&x, &y)
            .unwrap();
        let d = res.derivative.unwrap()[0];
        assert!(
            (d - slope).abs() < 1e-6,
            "scale={scale:e}: OLS slope={d}, expected {slope}"
        );
    }
}

/// Local-linear standard errors must scale linearly with the data magnitude rather
/// than collapse for small-magnitude x.
#[test]
fn wls_std_errors_scale_linearly() {
    let n = 100;
    let noise = lcg_noise(n, 11);
    let mut normalized_ref: Vec<f64> = Vec::new();
    for scale in [1.0, 1e-3, 1e-6, 1e-9] {
        let x: Vec<f64> = (0..n).map(|i| scale * i as f64 / (n - 1) as f64).collect();
        let y: Vec<f64> = x
            .iter()
            .zip(noise.iter())
            .map(|(&xi, &e)| 2.0 * xi + scale * 0.3 * e)
            .collect();
        let res = Lowess::new()
            .fraction(0.3)
            .iterations(0)
            .confidence_intervals(0.95)
            .build()
            .unwrap()
            .fit(&x, &y)
            .unwrap();
        let se = res.standard_errors.unwrap();
        let normalized: Vec<f64> = se.iter().map(|&v| v / scale).collect();
        assert!(
            normalized.iter().all(|&v| v.is_finite() && v > 0.0),
            "scale={scale:e}: SEs must be finite and positive"
        );
        if normalized_ref.is_empty() {
            normalized_ref = normalized;
        } else {
            for i in 0..n {
                let r = normalized_ref[i];
                assert!(
                    (normalized[i] - r).abs() <= 1e-6 * r,
                    "scale={scale:e}: SE[{i}] not scale-proportional ({} vs {r})",
                    normalized[i]
                );
            }
        }
    }
}

/// With `k == n` every test fold holds a single point, so k-fold CV *is*
/// leave-one-out and must reproduce the LOOCV score.
#[test]
fn kfold_with_k_equal_n_matches_loocv() {
    let n = 60;
    let x: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
    let noise = lcg_noise(n, 7);
    let y: Vec<f64> = x
        .iter()
        .zip(noise.iter())
        .map(|(&xi, &e)| xi.sin() + 0.2 * e)
        .collect();

    let loo = Lowess::new()
        .cv(CVBuilder::method("loocv").fractions(vec![0.3]))
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();
    let kfold = Lowess::new()
        .cv(CVBuilder::method("kfold").k(n).fractions(vec![0.3]))
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let loo_score = loo.cv_scores.unwrap()[0];
    let kfold_score = kfold.cv_scores.unwrap()[0];
    assert!(
        (loo_score - kfold_score).abs() <= 1e-9 * loo_score.abs().max(1.0),
        "k-fold (k=n) score {kfold_score} should equal LOOCV score {loo_score}"
    );
}
