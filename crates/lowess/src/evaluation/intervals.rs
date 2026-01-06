//! Confidence and prediction intervals for LOWESS smoothing.
//!
//! ## Purpose
//!
//! This module provides tools for quantifying uncertainty in LOWESS smoothing
//! through standard errors, confidence intervals, and prediction intervals.
//!
//! ## Design notes
//!
//! * **Methodology**: Uses local leverage and robust weighted residuals.
//! * **Approximation**: Z-scores estimated via Acklam's inverse normal CDF.
//! * **Flexibility**: Configurable coverage levels and interval types.
//!
//! ## Key concepts
//!
//! * **Standard Errors (SE)**: Uncertainty in fitted values due to sampling.
//! * **Confidence Intervals (CI)**: Uncertainty in the estimated mean curve.
//! * **Prediction Intervals (PI)**: Uncertainty for new observations (wider than CI).
//! * **Leverage**: Influence of an observation on its own fitted value.
//!
//! ## Invariants
//!
//! * Confidence levels must satisfy 0 < level < 1.
//! * Prediction intervals are always wider than confidence intervals.
//! * Standard errors are non-negative.
//!
//! ## Non-goals
//!
//! * This module does not perform the smoothing or iterative refinement.
//! * This module does not provide bootstrap or simulation-based intervals.
//! * This module does not handle simultaneous confidence bands.

// Feature-gated imports
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

// External dependencies
use num_traits::Float;

// Internal dependencies
use crate::math::scaling::ScalingMethod;
use crate::primitives::errors::LowessError;
use crate::primitives::window::Window;

// ============================================================================
// Interval Configuration
// ============================================================================

/// Configuration for computing confidence/prediction intervals and standard errors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IntervalMethod<T> {
    /// Desired probability coverage (e.g., 0.95 for 95% intervals).
    pub level: T,

    /// Whether to compute confidence intervals for the mean function.
    pub confidence: bool,

    /// Whether to compute prediction intervals for new observations.
    pub prediction: bool,

    /// Whether to return estimated standard errors for fitted values.
    pub se: bool,
}

impl<T: Float> Default for IntervalMethod<T> {
    fn default() -> Self {
        Self::none()
    }
}

impl<T: Float> IntervalMethod<T> {
    // ========================================================================
    // Constructors
    // ========================================================================

    /// No intervals or standard errors.
    fn none() -> Self {
        Self {
            level: T::from(0.95).unwrap(),
            confidence: false,
            prediction: false,
            se: false,
        }
    }

    /// Confidence intervals only at the specified level.
    pub fn confidence(level: T) -> Self {
        Self {
            level,
            confidence: true,
            prediction: false,
            se: true,
        }
    }

    /// Prediction intervals only at the specified level.
    pub fn prediction(level: T) -> Self {
        Self {
            level,
            confidence: false,
            prediction: true,
            se: true,
        }
    }

    /// Standard errors only (no intervals).
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
    // ========================================================================
    // Constants
    // ========================================================================

    /// Constant to convert MAD to an unbiased estimate of sigma for normal data.
    ///
    /// For normally distributed data, MAD × 1.4826 ≈ standard deviation.
    const MAD_TO_STD_FACTOR: f64 = 1.4826;

    /// Minimum tuned-scale absolute epsilon to avoid division by zero.
    const MIN_TUNED_SCALE: f64 = 1e-12;

    /// Number of parameters in local linear regression (intercept + slope).
    const LINEAR_PARAMS: f64 = 2.0;

    // ========================================================================
    // Robust Scale Estimation
    // ========================================================================

    /// Estimate the residual standard deviation using a robust method.
    /// Default sigma_hat = 1.4826 * MAD(residuals).
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

    // ========================================================================
    // Standard Error Computation
    // ========================================================================

    /// Core mathematical function for computing standard error at a point.
    /// SE = sqrt(sigma_local^2 * l_ii), where
    /// sigma_local^2 = (sum w_k r_k^2) / ((sum w_k) - 2) and
    /// l_ii = w_i / sum w_k.
    pub fn compute_se(sum_w: T, sum_w_r2: T, w_idx: T) -> T {
        // Effective degrees of freedom for weighted regression
        if sum_w <= T::zero() {
            return T::zero();
        }

        let effective_n = sum_w;
        let df = effective_n - T::from(Self::LINEAR_PARAMS).unwrap();

        if df <= T::zero() {
            return T::zero();
        }

        let variance = sum_w_r2 / df;
        let leverage = w_idx / sum_w; // Normalized leverage

        (variance * leverage).sqrt()
    }

    /// Compute standard errors for all points in a smoothed series.
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

            // Accumulate weighted residual variance
            let mut sum_w_r2 = T::zero();
            let mut sum_w = T::zero();

            for j in left..=right {
                let dist = (x[j] - x_current).abs();
                let u = dist / bandwidth;
                let w = if j == idx {
                    w_idx
                } else {
                    weight_fn(u) * robustness_weights[j]
                };

                let r = y[j] - y_smooth[j];
                sum_w_r2 = sum_w_r2 + w * r * r;
                sum_w = sum_w + w;
            }

            *se = Self::compute_se(sum_w, sum_w_r2, w_idx);
        }
    }

    // ========================================================================
    // Interval Computation
    // ========================================================================

    /// Compute requested intervals (confidence and/or prediction).
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

    // ========================================================================
    // Z-Score Approximation
    // ========================================================================

    /// Approximate the critical value (Z-score) for a given confidence level.
    /// z = Phi^-1((1 + p) / 2) where Phi^-1 is the inverse standard normal CDF.
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

    /// Rational approximation of the inverse standard normal CDF.
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
