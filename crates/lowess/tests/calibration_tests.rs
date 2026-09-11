#![cfg(feature = "dev")]
//! Empirical calibration check for confidence-interval standard errors.
//!
//! On *linear* truth the local-linear estimator is unbiased, so the Monte-Carlo
//! standard deviation of the fitted values is the true standard error and can be
//! compared directly against the reported `standard_errors`. This guards the exact
//! local-linear variance multiplier (`sum_k l_k^2`) and the kernel-corrected
//! residual degrees of freedom used by `IntervalMethod::compute_se`; the previous
//! `w_i / sum(w)` leverage with a `sum(w) - 2` denominator inflated the reported
//! SE by ~10-30% on these settings.
use lowess::prelude::*;

struct Rng(u64);
impl Rng {
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 32) as u32
    }
    fn norm(&mut self) -> f64 {
        let u1 = (self.next_u32() as f64) / (u32::MAX as f64) + 1e-12;
        let u2 = (self.next_u32() as f64) / (u32::MAX as f64);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

#[test]
fn reported_se_is_calibrated_on_linear_truth() {
    let n = 200;
    let x: Vec<f64> = (0..n).map(|i| 10.0 * i as f64 / (n - 1) as f64).collect();
    let truth: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
    let sigma = 0.3;
    let reps = 100;

    for frac in [0.1, 0.2, 0.3, 0.5] {
        let mut fits = vec![vec![0.0f64; n]; reps];
        let mut ses = vec![vec![0.0f64; n]; reps];
        let mut rng = Rng(20260910);
        for r in 0..reps {
            let mut y = truth.clone();
            for yi in y.iter_mut() {
                *yi += sigma * rng.norm();
            }
            let res = Lowess::new()
                .fraction(frac)
                .iterations(0)
                .confidence_intervals(0.95)
                .build()
                .unwrap()
                .fit(&x, &y)
                .unwrap();
            fits[r] = res.y;
            ses[r] = res.standard_errors.unwrap();
        }

        // Average the reported/true SE ratio over the interior, away from the
        // boundary-padded edges.
        let lo = n / 10;
        let hi = n - n / 10;
        let mut ratio_sum = 0.0;
        let mut cnt = 0;
        for i in lo..hi {
            let mean = fits.iter().map(|f| f[i]).sum::<f64>() / reps as f64;
            let var = fits.iter().map(|f| (f[i] - mean).powi(2)).sum::<f64>() / (reps - 1) as f64;
            let true_se = var.sqrt();
            let reported = ses.iter().map(|s| s[i]).sum::<f64>() / reps as f64;
            ratio_sum += reported / true_se;
            cnt += 1;
        }
        let ratio = ratio_sum / cnt as f64;
        assert!(
            (ratio - 1.0).abs() < 0.06,
            "reported/true SE ratio at fraction {frac} is {ratio:.4}, expected ~1.0"
        );
    }
}
