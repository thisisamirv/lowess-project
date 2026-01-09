//! Interpolation and delta optimization for LOWESS smoothing.
//!
//! This module provides utilities for optimized LOWESS performance through
//! delta-based point skipping and linear interpolation. When data points are
//! densely sampled, fitting every point is computationally expensive and
//! often unnecessary.

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
                Ok(T::from(0.01).unwrap() * range)
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
        let avg = (y0 + y1) / T::from(2.0).unwrap();
        y_smooth[(last_fitted + 1)..current].fill(avg);
        return;
    }

    // Linear interpolation: y = y0 + (xi - x0) * slope
    let slope = (y1 - y0) / denom;
    for k in (last_fitted + 1)..current {
        y_smooth[k] = y0 + (x[k] - x0) * slope;
    }
}
