//! Confidence and prediction intervals for LOWESS smoothing.
//!
//! This module provides tools for quantifying uncertainty in LOWESS smoothing
//! through standard errors, confidence intervals, and prediction intervals,
//! either analytically (normal theory) or via a residual bootstrap.
// ## srrstats Compliance
//
// @srrstats {RE5.0} Confidence intervals for the mean smoothed function.
// Prediction intervals for new observations.
// Acklam's rational approximation for inverse normal CDF (z-scores).
// Residual-bootstrap percentile intervals and bootstrap standard errors.

// External dependencies
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::cmp::Ordering;
use num_traits::Float;
#[cfg(feature = "std")]
use std::vec::Vec;

// Internal dependencies
use crate::math::scaling::ScalingMethod;
use crate::primitives::errors::LowessError;
use crate::primitives::window::Window;

// Configuration for computing confidence/prediction intervals and standard errors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IntervalMethod<T> {
    // Desired probability coverage (e.g., 0.95 for 95% intervals).
    pub level: T,

    // Whether to compute confidence intervals for the mean function.
    pub confidence: bool,

    // Whether to compute prediction intervals for new observations.
    pub prediction: bool,

    // Whether to return estimated standard errors for fitted values.
    pub se: bool,
}

impl<T: Float> Default for IntervalMethod<T> {
    fn default() -> Self {
        Self::none()
    }
}

impl<T: Float> IntervalMethod<T> {
    // No intervals or standard errors.
    fn none() -> Self {
        Self {
            level: T::from(0.95).unwrap(),
            confidence: false,
            prediction: false,
            se: false,
        }
    }

    // Confidence intervals only at the specified level.
    pub fn confidence(level: T) -> Self {
        Self {
            level,
            confidence: true,
            prediction: false,
            se: true,
        }
    }

    // Prediction intervals only at the specified level.
    pub fn prediction(level: T) -> Self {
        Self {
            level,
            confidence: false,
            prediction: true,
            se: true,
        }
    }

    // Standard errors only (no intervals).
    pub fn se() -> Self {
        Self {
            level: T::from(0.95).unwrap(),
            confidence: false,
            prediction: false,
            se: true,
        }
    }
}

impl<T: Float> IntervalMethod<T> {
    // Constant to convert MAD to an unbiased estimate of sigma for normal data.
    const MAD_TO_STD_FACTOR: f64 = 1.4826;

    // Minimum tuned-scale absolute epsilon to avoid division by zero.
    const MIN_TUNED_SCALE: f64 = 1e-12;

    // Number of parameters in local linear regression (intercept + slope).
    const LINEAR_PARAMS: f64 = 2.0;

    // Estimate the residual standard deviation using a robust method.
    // Default sigma_hat = 1.4826 * MAD(residuals).
    pub fn calculate_residual_sd(residuals: &[T]) -> T {
        let n = residuals.len();
        let scale_const = T::from(Self::MAD_TO_STD_FACTOR).unwrap();

        if n == 1 {
            return residuals[0].abs() * scale_const;
        }

        let mut vals = residuals.to_vec();
        let mad = ScalingMethod::MAD.compute(&mut vals);
        if mad > T::zero() {
            mad * scale_const
        } else {
            // Apply minimum scale to avoid division by zero
            let min_eps = T::from(Self::MIN_TUNED_SCALE).unwrap();
            min_eps * scale_const
        }
    }

    // Core mathematical function for computing the standard error of the local
    // *linear* fit at a point, from the local design moments.
    //
    // The pointwise variance multiplier is the exact local-linear (sandwich /
    // squared equivalent-kernel) form
    //     e1' (X'WX)^-1 (X'W^2 X) (X'WX)^-1 e1
    // where the design is centered at the point of interest, `dx = x_j - x0`,
    // `w_j` are the combined kernel/robustness weights, and `X = [1, dx]`. This
    // equals `sum_k l_k^2`, the squared norm of the equivalent-kernel row, and is
    // the variance of the local linear estimator under independent errors. The
    // simpler `w_i / sum(w)` "local constant" weight is only correct for a
    // uniform kernel on a symmetric window: it understates the leverage at
    // boundary/extrapolating points (where it should be much larger) and
    // overstates it on a symmetric interior window (where `sum_k l_k^2 =
    // sum(w^2)/sum(w)^2 < 1/sum(w)`).
    //
    // `s1`, `s2` are the first/second design moments with weights `w` (the
    // `X'WX` block) and `t0`, `t1`, `t2` the same moments with weights `w^2` (the
    // `X'W^2 X` block).
    //
    // The residual variance uses `sum(w r^2) / df` with the effective residual
    // degrees of freedom `df = sum(w) - 2 + sum(w^2)/sum(w)`. For a local linear
    // fit the weighted residual sum of squares has expectation
    //     sigma^2 * (sum(w) - 2 + sum(w^2)/sum(w)),
    // so the usual ordinary-least-squares denominator `sum(w) - 2` (which mixes a
    // sum of kernel weights with a parameter count) inflates the variance.
    pub fn compute_se(sum_w: T, sum_w_r2: T, s1: T, s2: T, t0: T, t1: T, t2: T) -> T {
        if sum_w <= T::zero() {
            return T::zero();
        }

        let two = T::from(Self::LINEAR_PARAMS).unwrap();

        let det = sum_w * s2 - s1 * s1;
        if det <= T::zero() {
            return T::zero();
        }

        // Squared norm of the equivalent-kernel row (exact variance multiplier).
        let leverage = (s2 * s2 * t0 - two * s1 * s2 * t1 + s1 * s1 * t2) / (det * det);
        if leverage <= T::zero() {
            return T::zero();
        }

        // Effective residual degrees of freedom (kernel-corrected).
        let df = sum_w - two + t0 / sum_w;
        if df <= T::zero() {
            return T::zero();
        }

        let variance = sum_w_r2 / df;

        (variance * leverage).sqrt()
    }

    // Compute standard errors for all points in a smoothed series.
    #[allow(clippy::too_many_arguments)]
    pub fn compute_window_se<F>(
        &self,
        x: &[T],
        y: &[T],
        y_smooth: &[T],
        window_size: usize,
        robustness_weights: &[T],
        std_errors: &mut [T],
        weight_fn: &F,
    ) where
        F: Fn(T) -> T,
    {
        // Early exit if no intervals or SE requested
        if !self.se && !self.confidence && !self.prediction {
            return;
        }

        let n = x.len();

        for (i, se) in std_errors.iter_mut().enumerate().take(n) {
            // Initialize and center window
            let mut window = Window::initialize(i, window_size, n);
            window.recenter(x, i, n);

            let idx = i;
            let left = window.left;
            let right = window.right;

            // Compute bandwidth
            let x_current = x[idx];
            let bandwidth_left = x_current - x[left];
            let bandwidth_right = x[right] - x_current;
            let bandwidth = T::max(bandwidth_left, bandwidth_right);

            if bandwidth <= T::zero() {
                *se = T::zero();
                continue;
            }

            // Compute weight for current point (distance = 0)
            let u_idx = T::zero();
            let kernel_val = weight_fn(u_idx);
            let w_idx = kernel_val * robustness_weights[idx];

            // Accumulate weighted residual variance and local design moments.
            let mut sum_w_r2 = T::zero();
            let mut sum_w = T::zero();
            let mut s1 = T::zero();
            let mut s2 = T::zero();
            let mut t0 = T::zero();
            let mut t1 = T::zero();
            let mut t2 = T::zero();

            for j in left..=right {
                let dist = (x[j] - x_current).abs();
                let u = dist / bandwidth;
                let w = if j == idx {
                    w_idx
                } else {
                    weight_fn(u) * robustness_weights[j]
                };

                let r = y[j] - y_smooth[j];
                let dx = x[j] - x_current;
                sum_w_r2 = sum_w_r2 + w * r * r;
                sum_w = sum_w + w;
                s1 = s1 + w * dx;
                s2 = s2 + w * dx * dx;
                t0 = t0 + w * w;
                t1 = t1 + w * w * dx;
                t2 = t2 + w * w * dx * dx;
            }

            *se = Self::compute_se(sum_w, sum_w_r2, s1, s2, t0, t1, t2);
        }
    }

    // Estimate the standard error of the fitted curve at an arbitrary out-of-sample query
    // point, generalizing `compute_window_se`'s per-training-point formula (same local-linear
    // design-moment leverage and local-residual-variance approach). There is no training
    // observation exactly at `x_query`, so the design is centered on the query point and the
    // moments are accumulated over the surrounding window.
    #[allow(clippy::too_many_arguments)]
    pub fn compute_se_at_query<F>(
        x: &[T],
        y: &[T],
        y_smooth: &[T],
        window: &Window,
        x_query: T,
        robustness_weights: &[T],
        weight_fn: &F,
    ) -> T
    where
        F: Fn(T) -> T,
    {
        let bandwidth = window.max_distance(x, x_query);
        if bandwidth <= T::zero() {
            return T::zero();
        }

        let mut sum_w_r2 = T::zero();
        let mut sum_w = T::zero();
        let mut s1 = T::zero();
        let mut s2 = T::zero();
        let mut t0 = T::zero();
        let mut t1 = T::zero();
        let mut t2 = T::zero();

        for j in window.left..=window.right {
            let dist = (x[j] - x_query).abs();
            let u = dist / bandwidth;
            let w = weight_fn(u) * robustness_weights[j];
            let r = y[j] - y_smooth[j];
            let dx = x[j] - x_query;
            sum_w_r2 = sum_w_r2 + w * r * r;
            sum_w = sum_w + w;
            s1 = s1 + w * dx;
            s2 = s2 + w * dx * dx;
            t0 = t0 + w * w;
            t1 = t1 + w * w * dx;
            t2 = t2 + w * w * dx * dx;
        }

        Self::compute_se(sum_w, sum_w_r2, s1, s2, t0, t1, t2)
    }

    // Compute requested intervals (confidence and/or prediction).
    #[allow(clippy::type_complexity)]
    pub fn compute_intervals(
        &self,
        y_smooth: &[T],
        std_errors: &[T],
        residuals: &[T],
    ) -> Result<
        (
            Option<Vec<T>>, // confidence lower
            Option<Vec<T>>, // confidence upper
            Option<Vec<T>>, // prediction lower
            Option<Vec<T>>, // prediction upper
        ),
        LowessError,
    > {
        // Compute confidence intervals if requested
        let (mut conf_lower, mut conf_upper) = if self.confidence {
            let (lower, upper) = self
                .compute_confidence_intervals_impl(y_smooth, std_errors)
                .map_err(|_| LowessError::InvalidIntervals(self.level.to_f64().unwrap_or(0.0)))?;
            (Some(lower), Some(upper))
        } else {
            (None, None)
        };

        // Compute prediction intervals if requested
        let (mut pred_lower, mut pred_upper) = if self.prediction {
            let rsd = Self::calculate_residual_sd(residuals);
            let (lower, upper) = self
                .compute_prediction_intervals_impl(y_smooth, std_errors, rsd)
                .map_err(|_| LowessError::InvalidIntervals(self.level.to_f64().unwrap_or(0.0)))?;
            (Some(lower), Some(upper))
        } else {
            (None, None)
        };

        // Guard against degenerate intervals
        let residual_sd = Self::calculate_residual_sd(residuals);
        let any_std_nonzero = std_errors.iter().any(|&s| s > T::zero());

        if residual_sd > T::zero() || any_std_nonzero {
            let eps = T::from(1e-12).unwrap_or_else(|| T::from(1e-6).unwrap());

            // Fix degenerate confidence intervals
            if let (Some(lo), Some(hi)) = (&mut conf_lower, &mut conf_upper) {
                for (l, h) in lo.iter_mut().zip(hi.iter_mut()) {
                    let width = *h - *l;
                    if !width.is_finite() || width <= T::zero() {
                        *h = *l + eps;
                    }
                }
            }

            // Fix degenerate prediction intervals
            if let (Some(lo), Some(hi)) = (&mut pred_lower, &mut pred_upper) {
                for (l, h) in lo.iter_mut().zip(hi.iter_mut()) {
                    let width = *h - *l;
                    if !width.is_finite() || width <= T::zero() {
                        *h = *l + eps;
                    }
                }
            }
        }

        Ok((conf_lower, conf_upper, pred_lower, pred_upper))
    }

    fn compute_confidence_intervals_impl(
        &self,
        y_smooth: &[T],
        std_errors: &[T],
    ) -> Result<(Vec<T>, Vec<T>), &'static str> {
        let z = Self::approximate_z_score(self.level)?;

        let lower: Vec<T> = y_smooth
            .iter()
            .zip(std_errors.iter())
            .map(|(&ys, &se)| ys - z * se)
            .collect();

        let upper: Vec<T> = y_smooth
            .iter()
            .zip(std_errors.iter())
            .map(|(&ys, &se)| ys + z * se)
            .collect();

        Ok((lower, upper))
    }

    fn compute_prediction_intervals_impl(
        &self,
        y_smooth: &[T],
        std_errors: &[T],
        residual_sd: T,
    ) -> Result<(Vec<T>, Vec<T>), &'static str> {
        let z = Self::approximate_z_score(self.level)?;
        let rsd_sq = residual_sd * residual_sd;

        let lower: Vec<T> = y_smooth
            .iter()
            .zip(std_errors.iter())
            .map(|(&ys, &se)| {
                let pred_se = (se * se + rsd_sq).sqrt();
                ys - z * pred_se
            })
            .collect();

        let upper: Vec<T> = y_smooth
            .iter()
            .zip(std_errors.iter())
            .map(|(&ys, &se)| {
                let pred_se = (se * se + rsd_sq).sqrt();
                ys + z * pred_se
            })
            .collect();

        Ok((lower, upper))
    }

    // Approximate the critical value (Z-score) for a given confidence level.
    // z = Phi^-1((1 + p) / 2) where Phi^-1 is the inverse standard normal CDF.
    pub fn approximate_z_score(confidence_level: T) -> Result<T, &'static str> {
        let cl_f = confidence_level.to_f64().unwrap_or(0.95);

        // Convert confidence level to cumulative probability
        let p = (1.0 + cl_f) / 2.0;

        // Fast paths for common confidence levels
        let z = if (cl_f - 0.99).abs() < 1e-6 {
            2.576
        } else if (cl_f - 0.95).abs() < 1e-6 {
            1.960
        } else if (cl_f - 0.90).abs() < 1e-6 {
            1.645
        } else {
            // Use Acklam's algorithm for other values
            Self::acklam_inverse_cdf(p)
        };

        Ok(T::from(z).unwrap_or_else(|| T::one()))
    }

    // Rational approximation of the inverse standard normal CDF.
    fn acklam_inverse_cdf(p: f64) -> f64 {
        if p <= 0.0 || p >= 1.0 {
            return 0.0;
        }

        // Coefficients for central region
        const A: [f64; 6] = [
            -3.969_683_028_665_376e1,
            2.209_460_984_245_205e2,
            -2.759_285_104_469_687e2,
            1.383_577_518_672_69e2,
            -3.066_479_806_614_716e1,
            2.506_628_277_459_239e0,
        ];
        const B: [f64; 5] = [
            -5.447_609_879_822_406e1,
            1.615_858_368_580_409e2,
            -1.556_989_798_598_866e2,
            6.680_131_188_771_972e1,
            -1.328_068_155_288_572e1,
        ];

        // Coefficients for tail regions
        const C: [f64; 6] = [
            -7.784_894_002_430_293e-3,
            -3.223_964_580_411_365e-1,
            -2.400_758_277_161_838e0,
            -2.549_732_539_343_734e0,
            4.374_664_141_464_968e0,
            2.938_163_982_698_783e0,
        ];
        const D: [f64; 4] = [
            7.784_695_709_041_462e-3,
            3.224_671_290_700_398e-1,
            2.445_134_137_142_996e0,
            3.754_408_661_907_416e0,
        ];

        const P_LOW: f64 = 0.02425;
        const P_HIGH: f64 = 0.97575;

        if p < P_LOW {
            // Lower tail
            let q = (-2.0 * p.ln()).sqrt();
            let num = ((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5];
            let den = (((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0;
            num / den
        } else if p > P_HIGH {
            // Upper tail
            let q = (-2.0 * (1.0 - p).ln()).sqrt();
            let num = ((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5];
            let den = (((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0;
            -(num / den)
        } else {
            // Central region
            let q = p - 0.5;
            let r = q * q;
            let num = (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q;
            let den = ((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0;
            num / den
        }
    }
}

// Residual bootstrap

// Seed used when none is supplied, so bootstrap output is reproducible by default.
pub const DEFAULT_BOOTSTRAP_SEED: u64 = 0x5EED_B007;

// Minimum number of bootstrap replicates (needed for a sample standard deviation).
pub const MIN_BOOTSTRAP_SAMPLES: usize = 2;

// Replicates handed to the refit callback at once (bounds memory, gives parallel passes work).
pub const BOOTSTRAP_BATCH_SIZE: usize = 256;

// Minimal no-std PRNG (64-bit LCG) for bootstrap resampling.
#[derive(Debug, Clone)]
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.state >> 32) as u32
    }

    // Uniform index in `0..n` (Lemire multiply-shift; bias is negligible for n << 2^32).
    fn next_index(&mut self, n: usize) -> usize {
        ((self.next_u32() as u64 * n as u64) >> 32) as usize
    }
}

// Residual-bootstrap configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BootstrapConfig {
    // Number of bootstrap replicates (refits).
    pub n_boot: usize,

    // PRNG seed; `None` uses `DEFAULT_BOOTSTRAP_SEED`.
    pub seed: Option<u64>,
}

// Bootstrap standard errors and percentile intervals (all in the fit's point order).
#[derive(Debug, Clone)]
pub struct BootstrapOutput<T> {
    pub std_errors: Vec<T>,
    pub confidence_lower: Option<Vec<T>>,
    pub confidence_upper: Option<Vec<T>>,
    pub prediction_lower: Option<Vec<T>>,
    pub prediction_upper: Option<Vec<T>>,
}

impl BootstrapConfig {
    // Run the residual bootstrap.
    //
    // For each replicate `b`, builds `y*_i = y_hat_i + e*_i` with `e*` drawn with
    // replacement from the centered residuals, and refits via `refit` (same x, same
    // smoothing configuration), which receives up to `BOOTSTRAP_BATCH_SIZE` replicates
    // at a time and returns one fit per replicate. Then, per point:
    // - SE = sample SD of the replicate fits `y_hat*_b`;
    // - CI = percentile interval of `y_hat*_b`;
    // - PI = percentile interval of `y_hat*_b + e**_b` (a fresh residual draw), which
    //   combines estimation uncertainty with the empirical (non-normal) noise.
    pub fn compute<T, F>(
        &self,
        method: &IntervalMethod<T>,
        y_smooth: &[T],
        residuals: &[T],
        refit: F,
    ) -> Result<BootstrapOutput<T>, LowessError>
    where
        T: Float,
        F: FnMut(&[Vec<T>]) -> Result<Vec<Vec<T>>, LowessError>,
    {
        self.compute_at(method, y_smooth, residuals, y_smooth.len(), refit)
    }

    pub fn compute_at<T, F>(
        &self,
        method: &IntervalMethod<T>,
        y_smooth: &[T],
        residuals: &[T],
        n_output: usize,
        mut refit: F,
    ) -> Result<BootstrapOutput<T>, LowessError>
    where
        T: Float,
        F: FnMut(&[Vec<T>]) -> Result<Vec<Vec<T>>, LowessError>,
    {
        if self.n_boot < MIN_BOOTSTRAP_SAMPLES {
            return Err(LowessError::InvalidBootstrapSamples(self.n_boot));
        }

        let n = y_smooth.len();
        let b = self.n_boot;
        let mut rng = SimpleRng::new(self.seed.unwrap_or(DEFAULT_BOOTSTRAP_SEED));

        let n_t = T::from(n).unwrap();
        let mean_r = residuals.iter().fold(T::zero(), |acc, &r| acc + r) / n_t;
        let centered: Vec<T> = residuals.iter().map(|&r| r - mean_r).collect();

        // Point-major layout (`fits[i * b + k]`) so each point's replicates are contiguous.
        let mut fits: Vec<T> = Vec::new();
        fits.resize(n_output * b, T::zero());

        // Draws happen here, in replicate order, so results don't depend on how `refit` schedules work.
        let mut done = 0;
        while done < b {
            let len = BOOTSTRAP_BATCH_SIZE.min(b - done);
            let batch: Vec<Vec<T>> = (0..len)
                .map(|_| {
                    y_smooth
                        .iter()
                        .map(|&yh| yh + centered[rng.next_index(n)])
                        .collect()
                })
                .collect();
            let batch_fits = refit(&batch)?;
            for (j, fit) in batch_fits.iter().enumerate().take(len) {
                for (i, &f) in fit.iter().enumerate().take(n_output) {
                    fits[i * b + done + j] = f;
                }
            }
            done += len;
        }

        let half_alpha = (T::one() - method.level) / T::from(2.0).unwrap();
        let q_lo = half_alpha;
        let q_hi = T::one() - half_alpha;
        let b_t = T::from(b).unwrap();

        let mut std_errors = Vec::with_capacity(n_output);
        let (mut cl, mut cu) = (Vec::new(), Vec::new());
        let (mut pl, mut pu) = (Vec::new(), Vec::new());
        let mut pred: Vec<T> = Vec::new();
        if method.prediction {
            pred.resize(b, T::zero());
        }

        for i in 0..n_output {
            let col = &mut fits[i * b..(i + 1) * b];

            let mean = col.iter().fold(T::zero(), |acc, &v| acc + v) / b_t;
            let ss = col
                .iter()
                .fold(T::zero(), |acc, &v| acc + (v - mean) * (v - mean));
            std_errors.push((ss / (b_t - T::one())).sqrt());

            if method.prediction {
                for (p, &f) in pred.iter_mut().zip(col.iter()) {
                    *p = f + centered[rng.next_index(n)];
                }
                sort_floats(&mut pred);
                pl.push(quantile_sorted(&pred, q_lo));
                pu.push(quantile_sorted(&pred, q_hi));
            }

            if method.confidence {
                sort_floats(col);
                cl.push(quantile_sorted(col, q_lo));
                cu.push(quantile_sorted(col, q_hi));
            }
        }

        let (confidence_lower, confidence_upper) = if method.confidence {
            (Some(cl), Some(cu))
        } else {
            (None, None)
        };
        let (prediction_lower, prediction_upper) = if method.prediction {
            (Some(pl), Some(pu))
        } else {
            (None, None)
        };

        Ok(BootstrapOutput {
            std_errors,
            confidence_lower,
            confidence_upper,
            prediction_lower,
            prediction_upper,
        })
    }
}

fn sort_floats<T: Float>(v: &mut [T]) {
    v.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
}

// Linearly interpolated quantile of sorted data (Hyndman & Fan type 7, R's default).
fn quantile_sorted<T: Float>(sorted: &[T], q: T) -> T {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let h = q * T::from(n - 1).unwrap();
    let lo = h.floor().to_usize().unwrap_or(0).min(n - 1);
    let hi = (lo + 1).min(n - 1);
    let frac = h - T::from(lo).unwrap();
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}
