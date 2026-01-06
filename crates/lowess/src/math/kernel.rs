//! Kernel (weight) functions for LOWESS smoothing.
//!
//! ## Purpose
//!
//! This module provides kernel functions that define distances-based weights for
//! local regression. It controls the influence of neighboring points on the fit.
//!
//! ## Design notes
//!
//! * **Normalization**: Maps distances u = |x - x_i| / bandwidth to weights.
//! * **Efficiency**: Uses precomputed properties (variance, roughness) for performance.
//! * **Support**: Most kernels are bounded on [-1, 1] for efficiency.
//!
//! ## Key concepts
//!
//! * **Tricube**: The default kernel (Cleveland's original), smooth and efficient.
//! * **Bias-Variance properties**: Each kernel has associated moments (mu_2) and roughness (R).
//!
//! ## Invariants
//!
//! * Kernels are non-negative (K(u) >= 0) and symmetric (K(u) = K(-u)).
//! * Bounded kernels return exactly zero outside their support.
//!
//! ## Non-goals
//!
//! * This module does not perform weight normalization.
//! * This module does not handle bandwidth selection logic.

// External dependencies
use core::f64::consts::{PI, SQRT_2};
use num_traits::Float;

// ============================================================================
// Mathematical Constants
// ============================================================================

/// Square root of 2*pi, used in Gaussian kernel calculations.
const SQRT_2PI: f64 = 2.5066282746310005024157652848110452530069867406099_f64;

/// Square root of pi, used in kernel property calculations.
const SQRT_PI: f64 = 1.772453850905516027298167483341145182797_f64;

/// pi/2, used in cosine kernel calculations.
const PI_OVER_2: f64 = PI / 2.0;

/// Cutoff for Gaussian kernel evaluation.
///
/// Beyond this normalized distance, the Gaussian kernel value is effectively
/// zero (exp(-6^2/2) approx 6.9e-9). This prevents numerical underflow and improves
/// performance.
const GAUSSIAN_CUTOFF: f64 = 6.0;

// ============================================================================
// Kernel Properties
// ============================================================================

/// # Mathematical Properties
///
/// | Kernel       | Formula                     | Efficiency† | R(K)                    | mu_2(K)               |
/// |--------------|-----------------------------|-------------|-------------------------|-----------------------|
/// | Cosine       | cos(pi*u/2)                 | 0.9995      | pi^2 / 16               | 1 - 8/pi^2            |
/// | Epanechnikov | 1 - u^2                     | 1.0000      | 3/5                     | 1/5                   |
/// | Gaussian     | exp(-u^2 / 2)               | 0.9607      | 1 / (2 * sqrt(pi))      | 1                     |
/// | Biweight     | (1 - u^2)^2                 | 0.9951      | 5/7                     | 1/7                   |
/// | Triangular   | 1 - |u|                     | 0.9887      | 2/3                     | 1/6                   |
/// | Tricube      | (1 - |u|^3)^3               | 0.9983      | 1225/1729               | 35/243                |
/// | Uniform      | 1                           | 0.9432      | 1/2                     | 1/3                   |
///
/// † Relative AMISE efficiency (Epanechnikov = 1.0) from Wand & Jones (1995)
///
/// **Note**: The mu_2(K) column lists the unnormalized second moment.
/// Use the `integrator()` method to obtain the normalized moment when needed.
///
/// Mathematical properties of a kernel function.
///
/// These properties are used for bias-variance analysis and efficiency
/// calculations.
struct KernelProperties {
    /// Kernel integral: c_K = integral K(u) du.
    integrator: f64,

    /// Unnormalized second moment: mu_2(K) = integral u^2 K(u) du.
    variance: f64,

    /// Kernel roughness: R(K) = integral K(u)^2 du.
    roughness: f64,
}

/// Precomputed properties for the Cosine kernel.
const COSINE_PROPERTIES: KernelProperties = KernelProperties {
    integrator: 4.0 / PI,
    variance: 4.0 / PI - 32.0 / (PI * PI * PI),
    roughness: 1.0,
};

/// Precomputed properties for the Epanechnikov kernel.
const EPANECHNIKOV_PROPERTIES: KernelProperties = KernelProperties {
    integrator: 4.0 / 3.0,
    variance: 4.0 / 15.0,
    roughness: 16.0 / 15.0,
};

/// Precomputed properties for the Gaussian kernel.
const GAUSSIAN_PROPERTIES: KernelProperties = KernelProperties {
    integrator: SQRT_2PI,
    variance: SQRT_2 * SQRT_PI,
    roughness: SQRT_PI,
};

/// Precomputed properties for the Biweight kernel.
const BIWEIGHT_PROPERTIES: KernelProperties = KernelProperties {
    integrator: 16.0 / 15.0,
    variance: 16.0 / 105.0,
    roughness: 256.0 / 315.0,
};

/// Precomputed properties for the Triangular kernel.
const TRIANGULAR_PROPERTIES: KernelProperties = KernelProperties {
    integrator: 1.0,
    variance: 1.0 / 6.0,
    roughness: 2.0 / 3.0,
};

/// Precomputed properties for the Tricube kernel.
const TRICUBE_PROPERTIES: KernelProperties = KernelProperties {
    integrator: 81.0 / 70.0,
    variance: 1.0 / 6.0,
    roughness: 6_561.0 / 6_916.0,
};

/// Precomputed properties for the Uniform kernel.
const UNIFORM_PROPERTIES: KernelProperties = KernelProperties {
    integrator: 2.0,
    variance: 2.0 / 3.0,
    roughness: 2.0,
};

// ============================================================================
// Weight Function Enum
// ============================================================================

/// Weight function (kernel) for LOWESS smoothing.
///
/// Each kernel defines a function K: ℝ → [0, ∞) that maps normalized
/// distances to weights. Bounded kernels have support on [-1, 1], while
/// the Gaussian kernel has unbounded support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WeightFunction {
    /// Cosine kernel: K(u) = cos(pi * u / 2) for |u| < 1.
    Cosine,

    /// Epanechnikov kernel: K(u) = (1 - u^2) for |u| < 1.
    Epanechnikov,

    /// Gaussian kernel: K(u) = exp(-u^2 / 2).
    Gaussian,

    /// Biweight (quartic) kernel: K(u) = (1 - u^2)^2 for |u| < 1.
    Biweight,

    /// Triangular (linear) kernel: K(u) = (1 - |u|) for |u| < 1.
    Triangle,

    /// Tricube kernel: K(u) = (1 - |u|^3)^3 for |u| < 1.
    ///
    /// This is the default and recommended kernel choice.
    #[default]
    Tricube,

    /// Uniform (rectangular) kernel: K(u) = 1 for |u| < 1.
    Uniform,
}

impl WeightFunction {
    // ========================================================================
    // Metadata Methods
    // ========================================================================

    /// Get the name of the weight function.
    #[inline]
    pub const fn name(&self) -> &'static str {
        match self {
            WeightFunction::Cosine => "Cosine",
            WeightFunction::Epanechnikov => "Epanechnikov",
            WeightFunction::Gaussian => "Gaussian",
            WeightFunction::Biweight => "Biweight",
            WeightFunction::Triangle => "Triangle",
            WeightFunction::Tricube => "Tricube",
            WeightFunction::Uniform => "Uniform",
        }
    }

    /// Get the kernel properties.
    const fn properties(&self) -> &'static KernelProperties {
        match self {
            WeightFunction::Cosine => &COSINE_PROPERTIES,
            WeightFunction::Epanechnikov => &EPANECHNIKOV_PROPERTIES,
            WeightFunction::Gaussian => &GAUSSIAN_PROPERTIES,
            WeightFunction::Biweight => &BIWEIGHT_PROPERTIES,
            WeightFunction::Triangle => &TRIANGULAR_PROPERTIES,
            WeightFunction::Tricube => &TRICUBE_PROPERTIES,
            WeightFunction::Uniform => &UNIFORM_PROPERTIES,
        }
    }

    // ========================================================================
    // Kernel Property Accessors
    // ========================================================================

    /// Get the unnormalized variance (second moment) of the kernel.
    #[inline]
    pub fn variance(&self) -> f64 {
        self.properties().variance
    }

    /// Get the roughness of the kernel.
    #[inline]
    pub fn roughness(&self) -> f64 {
        self.properties().roughness
    }

    /// Get the kernel integrator.
    #[inline]
    pub fn integrator(&self) -> f64 {
        self.properties().integrator
    }

    /// AMISE relative efficiency (Epanechnikov = 1.0).
    ///
    /// This measures how efficient the kernel is for density estimation
    /// in terms of asymptotic mean integrated squared error (AMISE).
    /// Higher values are better.
    ///
    /// # Formula
    ///
    /// Calculated as:
    /// ```text
    /// (R(K_E)/R(K))^(4/5) * (mu_2(K_E)/mu_2(K))^(2/5) * (c(K)/c(K_E))^2
    /// ```
    ///
    /// where K_E is the Epanechnikov kernel (the AMISE-optimal kernel).
    #[inline]
    pub fn efficiency(&self) -> f64 {
        let stats = self.properties();
        let ep_stats = &EPANECHNIKOV_PROPERTIES;

        let r_ratio = ep_stats.roughness / stats.roughness;
        let v_ratio = ep_stats.variance / stats.variance;
        let c_ratio = stats.integrator / ep_stats.integrator;

        r_ratio.powf(4.0 / 5.0) * v_ratio.powf(2.0 / 5.0) * c_ratio.powf(2.0)
    }

    // ========================================================================
    // Support Methods
    // ========================================================================

    /// Returns the support interval for bounded kernels.
    #[inline]
    pub fn support(&self) -> Option<(f64, f64)> {
        match self {
            WeightFunction::Gaussian => None, // Unbounded
            _ => Some((-1.0, 1.0)),           // All others are bounded on [-1, 1]
        }
    }

    /// Returns `true` if the kernel has bounded support.
    #[inline]
    fn is_bounded(&self) -> bool {
        self.support().is_some()
    }

    // ========================================================================
    // Weight Computation
    // ========================================================================

    /// Compute the unnormalized weight K(u) for a given normalized distance.
    #[inline]
    pub fn compute_weight<T: Float>(&self, u: T) -> T {
        let abs_u = u.abs();

        // Fast path for bounded kernels: return 0 if outside support
        if self.is_bounded() && abs_u >= T::one() {
            return T::zero();
        }

        match self {
            WeightFunction::Cosine => {
                let pi_over_2 = T::from(PI_OVER_2).unwrap();
                (pi_over_2 * abs_u).cos()
            }

            WeightFunction::Epanechnikov => T::one() - abs_u * abs_u,

            WeightFunction::Gaussian => {
                // Convert to f64 for exponential calculation
                let u_f64 = abs_u.to_f64().unwrap_or(f64::INFINITY);

                // Use cutoff to avoid underflow to zero
                if u_f64 > GAUSSIAN_CUTOFF {
                    T::from(f64::MIN_POSITIVE).unwrap_or_else(T::zero)
                } else {
                    let val = (-0.5 * u_f64 * u_f64).exp().max(f64::MIN_POSITIVE);
                    T::from(val).unwrap_or_else(T::zero)
                }
            }

            WeightFunction::Biweight => {
                let tmp = T::one() - abs_u * abs_u;
                tmp * tmp
            }

            WeightFunction::Triangle => T::one() - abs_u,

            WeightFunction::Tricube => {
                let tmp = T::one() - abs_u * abs_u * abs_u;
                tmp * tmp * tmp
            }

            WeightFunction::Uniform => T::one(),
        }
    }

    /// Apply the kernel weighting to a window of points.
    #[allow(clippy::too_many_arguments)]
    pub fn compute_window_weights<T: Float>(
        &self,
        x: &[T],
        left: usize,
        right: usize,
        x_current: T,
        bandwidth: T,
        h1: T,
        h9: T,
        weights: &mut [T],
    ) -> (T, usize) {
        let n = x.len();

        // Safety guard for empty input or invalid window
        if left >= n || right >= n || left > right {
            return (T::zero(), left);
        }

        // Degenerate bandwidth: zero all weights in window
        if bandwidth <= T::zero() {
            let mut i = left;
            while i < n {
                weights[i] = T::zero();
                i += 1;
            }
            return (T::zero(), left);
        }

        let mut sum = T::zero();
        let mut rightmost = left;

        // Skip points to the left of (x_current - h9) for efficiency
        let lower_bound = x_current - h9;
        let mut start = left;
        while start < n && x[start] < lower_bound {
            start += 1;
        }

        // Zero the skipped region [left..start)
        if start > left {
            let mut i = left;
            while i < start {
                weights[i] = T::zero();
                i += 1;
            }
        }

        // Compute weights from start onwards using while loop for performance
        let mut j = start;
        while j <= right {
            let xj = x[j];
            let distance = (xj - x_current).abs();

            if distance > h9 {
                if xj > x_current {
                    // Beyond h9 on right side (x is sorted): zero remaining in window and break
                    let mut k = j;
                    while k <= right {
                        weights[k] = T::zero();
                        k += 1;
                    }
                    break;
                }
                // Beyond h9 on left side (defensive)
                weights[j] = T::zero();
                j += 1;
                continue;
            }

            // Compute weight: use 1.0 for very close points, otherwise evaluate kernel
            let w_k = if distance <= h1 {
                T::one()
            } else {
                self.compute_weight(distance / bandwidth)
            };

            weights[j] = w_k;
            sum = sum + w_k;
            rightmost = j;
            j += 1;
        }

        (sum, rightmost)
    }
}
