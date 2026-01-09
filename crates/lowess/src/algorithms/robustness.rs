//! Robustness weight computation for outlier downweighting.
//!
//! This module implements iterative reweighted least squares (IRLS) for robust
//! LOWESS smoothing. After an initial fit, residuals are computed and used to
//! downweight outliers in subsequent iterations.

// External dependencies
use num_traits::Float;

// Internal dependencies
use crate::math::scaling::ScalingMethod;

// Robustness weighting method for outlier downweighting.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum RobustnessMethod {
    // Bisquare (Tukey's biweight) - default and most common.
    #[default]
    Bisquare,

    // Huber weights - less aggressive downweighting.
    Huber,

    // Talwar (hard threshold) - most aggressive.
    Talwar,
}

impl RobustnessMethod {
    // Default tuning constant for bisquare robustness weights.
    const DEFAULT_BISQUARE_C: f64 = 6.0;

    // Default tuning constant for Huber weights.
    const DEFAULT_HUBER_C: f64 = 1.345;

    // Default tuning constant for Talwar weights.
    const DEFAULT_TALWAR_C: f64 = 2.5;

    // Minimum scale threshold relative to mean absolute residual.
    const SCALE_THRESHOLD: f64 = 1e-7;

    // Minimum tuned-scale absolute epsilon to avoid division by zero.
    const MIN_TUNED_SCALE: f64 = 1e-12;

    // Apply robustness weights using the configured method.
    pub fn apply_robustness_weights<T: Float>(
        &self,
        residuals: &[T],
        weights: &mut [T],
        scaling_method: ScalingMethod,
        scratch: &mut [T],
    ) {
        if residuals.is_empty() {
            return;
        }

        let base_scale = self.compute_scale(residuals, scaling_method, scratch);

        let (method_type, tuning_constant) = match self {
            Self::Bisquare => (0, Self::DEFAULT_BISQUARE_C),
            Self::Huber => (1, Self::DEFAULT_HUBER_C),
            Self::Talwar => (2, Self::DEFAULT_TALWAR_C),
        };

        let c_t = T::from(tuning_constant).unwrap();

        for (i, &r) in residuals.iter().enumerate() {
            weights[i] = match method_type {
                0 => Self::bisquare_weight(r, base_scale, c_t),
                1 => Self::huber_weight(r, base_scale, c_t),
                _ => Self::talwar_weight(r, base_scale, c_t),
            };
        }
    }

    // Compute robust scale estimate with MAD fallback.
    fn compute_scale<T: Float>(
        &self,
        residuals: &[T],
        scaling_method: ScalingMethod,
        scratch: &mut [T],
    ) -> T {
        // Use scratch buffer for scaling to avoid allocation
        scratch.copy_from_slice(residuals);
        let scale = scaling_method.compute(scratch);

        // Compute MAR (Mean Absolute Residual) inline as fallback
        let n = residuals.len();
        let mut sum_abs = T::zero();
        for &r in residuals {
            sum_abs = sum_abs + r.abs();
        }
        let mean_abs = sum_abs / T::from(n).unwrap();

        let relative_threshold = T::from(Self::SCALE_THRESHOLD).unwrap() * mean_abs;
        let absolute_threshold = T::from(Self::MIN_TUNED_SCALE).unwrap();
        let scale_threshold = relative_threshold.max(absolute_threshold);

        if scale <= scale_threshold {
            // Scale is too small, use MAR as fallback
            mean_abs.max(scale)
        } else {
            scale
        }
    }

    // Compute bisquare weight.
    //
    // # Formula
    //
    // cmad = 6 * median(|residuals|)
    // c1 = 0.001 * cmad
    // c9 = 0.999 * cmad
    //
    // w = 1.0                    if |r| <= c1
    // w = (1 - (r/cmad)^2)^2     if c1 < |r| <= c9
    // w = 0.0                    if |r| > c9
    #[inline]
    fn bisquare_weight<T: Float>(residual: T, scale: T, c: T) -> T {
        if scale <= T::zero() {
            return T::one();
        }

        let min_eps = T::from(Self::MIN_TUNED_SCALE).unwrap();
        // Ensure c is at least min_eps so tuned_scale isn't zero
        let c_clamped = c.max(min_eps);
        // cmad = c * scale (e.g., 6.0 * MAR)
        let cmad = (scale * c_clamped).max(min_eps);

        // Boundary thresholds
        let c1 = T::from(0.001).unwrap() * cmad;
        let c9 = T::from(0.999).unwrap() * cmad;

        let r = residual.abs();

        if r <= c1 {
            // Very small residual: weight = 1.0
            T::one()
        } else if r <= c9 {
            // Apply bisquare formula: (1 - (r/cmad)^2)^2
            let u = r / cmad;
            let tmp = T::one() - u * u;
            tmp * tmp
        } else {
            // Large residual: weight = 0.0
            T::zero()
        }
    }

    // Compute Huber weight.
    //
    // # Formula
    //
    // u = |r| / s
    //
    // w(u) = 1      if u <= c
    //
    // w(u) = c / u  if u > c
    #[inline]
    fn huber_weight<T: Float>(residual: T, scale: T, c: T) -> T {
        if scale <= T::zero() {
            return T::one();
        }

        let u = (residual / scale).abs();
        if u <= c { T::one() } else { c / u }
    }

    // Compute Talwar weight.
    //
    // # Formula
    //
    // u = |r| / s
    //
    // w(u) = 1  if u <= c
    //
    // w(u) = 0  if u > c
    #[inline]
    fn talwar_weight<T: Float>(residual: T, scale: T, c: T) -> T {
        if scale <= T::zero() {
            return T::one();
        }

        let u = (residual / scale).abs();
        if u <= c { T::one() } else { T::zero() }
    }
}
