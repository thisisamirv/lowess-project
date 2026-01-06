//! Robust scale estimation using MAR or MAD.
//!
//! ## Purpose
//!
//! This module provides robust scale estimation methods, which are resistant
//! to outliers.
//!
//! ## Design notes
//!
//! * **Algorithm**: Uses Quickselect for O(n) median finding.
//! * **Memory**: Reuses allocated buffers to minimize memory allocations.
//! * **Methods**:
//!     - MAR: Median Absolute Residual: `median(|r|)`
//!     - MAD: Median Absolute Deviation: `median(|r - median(r)|)`
//!
//! ## Invariants
//!
//! * Scale >= 0 for any input.
//! * Handles even and odd population sizes correctly.

// External dependencies
use core::cmp::Ordering::Equal;
use num_traits::Float;

/// Method for measuring the scale of residuals.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ScalingMethod {
    /// Median Absolute Residual: `median(|r|)`.
    MAR,

    /// Median Absolute Deviation: `median(|r - median(r)|)`.
    #[default]
    MAD,
}

impl ScalingMethod {
    /// Compute the scale of the given values using the selected method.
    ///
    /// The input slice is modified in-place to avoid allocations.
    pub fn compute<T: Float>(&self, vals: &mut [T]) -> T {
        match self {
            Self::MAR => self.compute_mar(vals),
            Self::MAD => self.compute_mad(vals),
        }
    }

    /// Compute the Median Absolute Deviation (MAD).
    #[inline]
    fn compute_mad<T: Float>(&self, vals: &mut [T]) -> T {
        if vals.is_empty() {
            return T::zero();
        }

        // Step 1: Compute median of residuals
        let median: T = Self::median_inplace(vals);

        // Step 2: Compute absolute deviations from median
        for val in vals.iter_mut() {
            *val = (*val - median).abs();
        }

        // Step 3: Return median of absolute deviations
        Self::median_inplace(vals)
    }

    /// Compute the Median Absolute Residual (uncentered).
    #[inline]
    fn compute_mar<T: Float>(&self, vals: &mut [T]) -> T {
        if vals.is_empty() {
            return T::zero();
        }

        // Step 1: Compute absolute values
        for val in vals.iter_mut() {
            *val = val.abs();
        }

        // Step 2: Return median of absolute values
        Self::median_inplace(vals)
    }

    /// Internal helper function to compute median in-place using Quickselect.
    #[inline]
    fn median_inplace<T: Float>(vals: &mut [T]) -> T {
        let n = vals.len();
        if n == 0 {
            return T::zero();
        }

        let mid = n / 2;

        if n % 2 == 0 {
            // Even length: average of two middle values
            vals.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap_or(Equal));
            let upper = vals[mid];

            // Find the largest value in the lower half using simple loop
            let mut lower = vals[0];
            let mut i = 1;
            while i < mid {
                if vals[i] > lower {
                    lower = vals[i];
                }
                i += 1;
            }

            (lower + upper) / T::from(2.0).unwrap()
        } else {
            // Odd length: middle value
            vals.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap_or(Equal));
            vals[mid]
        }
    }
}
