//! Interpolation and delta optimization for LOWESS smoothing.
//!
//! This module provides utilities for optimized LOWESS performance through
//! delta-based point skipping and linear interpolation. When data points are
//! densely sampled, fitting every point is computationally expensive and
//! often unnecessary.
// ## srrstats Compliance
//
// @srrstats {G1.6} Delta-based interpolation reduces O(n^2) to O(n×k) for dense data.
// @srrstats {RE2.2} Linear interpolation between anchor points for non-fitted values.

// External dependencies
use core::result::Result;
use num_traits::Float;

// Internal dependencies
use crate::primitives::errors::LowessError;

// Calculate delta parameter for interpolation optimization.
pub fn calculate_delta<T: Float>(delta: Option<T>, x_sorted: &[T]) -> Result<T, LowessError> {
    match delta {
        Some(d) => Ok(d),
        None => {
            // Compute default delta as 1% of x-range
            if x_sorted.is_empty() {
                Ok(T::zero())
            } else {
                let range = x_sorted[x_sorted.len() - 1] - x_sorted[0];
                Ok(T::from(0.01).unwrap_or(T::zero()) * range)
            }
        }
    }
}

// Interpolate gap between two fitted anchor points.
pub fn interpolate_gap<T: Float>(x: &[T], y_smooth: &mut [T], last_fitted: usize, current: usize) {
    // No gap to interpolate
    if current <= last_fitted + 1 {
        return;
    }

    let x0 = x[last_fitted];
    let x1 = x[current];
    let y0 = y_smooth[last_fitted];
    let y1 = y_smooth[current];

    let denom = x1 - x0;

    if denom <= T::zero() {
        // Duplicate or decreasing x-values: use simple average
        let avg = (y0 + y1) / T::from(2.0).unwrap_or(T::one() + T::one());
        y_smooth[(last_fitted + 1)..current].fill(avg);
        return;
    }

    // Cleveland/R interpolation order. The algebraically equivalent slope
    // form can erase roundoff residuals used by later robustness passes.
    for k in (last_fitted + 1)..current {
        let alpha = (x[k] - x0) / denom;
        y_smooth[k] = alpha * y1 + (T::one() - alpha) * y0;
    }
}

// Fill in `derivative` for the gap between two anchor points with the constant slope of
// the linear segment connecting them - the same slope `interpolate_gap` used to fill
// `y_smooth`, so the returned derivative stays consistent with the smoothed curve there.
pub fn interpolate_gap_derivative<T: Float>(
    x: &[T],
    y_smooth: &[T],
    derivative: &mut [T],
    last_fitted: usize,
    current: usize,
) {
    if current <= last_fitted + 1 {
        return;
    }

    let denom = x[current] - x[last_fitted];
    let slope = if denom > T::zero() {
        (y_smooth[current] - y_smooth[last_fitted]) / denom
    } else {
        T::zero()
    };
    derivative[(last_fitted + 1)..current].fill(slope);
}
