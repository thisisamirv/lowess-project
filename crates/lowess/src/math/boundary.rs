//! Boundary padding strategies for local regression.
//!
//! ## Purpose
//!
//! This module implements boundary padding strategies to reduce smoothing bias at
//! data edges. By providing context beyond the original boundaries, local
//! regression can perform better near the start and end of the dataset.
//!
//! ## Design notes
//!
//! * **Strategy Pattern**: Uses `BoundaryPolicy` enum to select the padding method.
//! * **Allocation**: Creates new vectors for padded data (necessary for extension).
//!
//! ## Key concepts
//!
//! * **Boundary Effect**: The tendency for local regression to have higher bias at edges.
//! * **Padding strategies**: `Extend` (extrapolate x, repeat y), `Reflect` (mirror), `Zero` (pad 0).
//!
//! ## Invariants
//!
//! * Padding length is limited to half the window size or `n - 1`.
//! * Original data is preserved in the middle of the padded range.
//!
//! ## Non-goals
//!
//! * This module does not perform in-place modification of input data.

// Feature-gated imports
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

// External dependencies
use num_traits::Float;

// ============================================================================
// Boundary Policy
// ============================================================================

/// Policy for handling boundaries at the start and end of a data stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BoundaryPolicy {
    /// Linearly extrapolate x-values and replicate y-values to provide context.
    #[default]
    Extend,

    /// Mirror values across the boundary.
    Reflect,

    /// Use zero padding beyond data boundaries.
    Zero,

    /// No boundary padding (standard LOWESS behavior).
    NoBoundary,
}

// ============================================================================
// Boundary Padding Function
// ============================================================================

/// Apply a boundary policy to pad the input data.
///
/// Returns augmented x and y vectors with boundary padding applied.
/// For `NoBoundary`, returns clones of the original data.
pub fn apply_boundary_policy<T: Float>(
    x: &[T],
    y: &[T],
    window_size: usize,
    policy: BoundaryPolicy,
) -> (Vec<T>, Vec<T>) {
    let n = x.len();

    // Handle NoBoundary case first
    if policy == BoundaryPolicy::NoBoundary {
        return (x.to_vec(), y.to_vec());
    }

    // Number of points to pad on each side (half-window)
    let pad_len = (window_size / 2).min(n - 1);
    if pad_len == 0 {
        return (x.to_vec(), y.to_vec());
    }

    let total_len = n + 2 * pad_len;
    let mut px = Vec::with_capacity(total_len);
    let mut py = Vec::with_capacity(total_len);

    // 1. Prepend padding
    match policy {
        BoundaryPolicy::Extend => {
            let x0 = x[0];
            let y0 = y[0];
            let dx = x[1] - x[0];
            for i in (1..=pad_len).rev() {
                px.push(x0 - T::from(i).unwrap() * dx);
                py.push(y0);
            }
        }
        BoundaryPolicy::Reflect => {
            let x0 = x[0];
            for i in (1..=pad_len).rev() {
                px.push(x0 - (x[i] - x0));
                py.push(y[i]);
            }
        }
        BoundaryPolicy::Zero => {
            let x0 = x[0];
            let dx = x[1] - x[0];
            for i in (1..=pad_len).rev() {
                px.push(x0 - T::from(i).unwrap() * dx);
                py.push(T::zero());
            }
        }
        BoundaryPolicy::NoBoundary => unreachable!(),
    }

    // 2. Add original data
    px.extend_from_slice(x);
    py.extend_from_slice(y);

    // 3. Append padding
    match policy {
        BoundaryPolicy::Extend => {
            let xn = x[n - 1];
            let yn = y[n - 1];
            let dx = x[n - 1] - x[n - 2];
            for i in 1..=pad_len {
                px.push(xn + T::from(i).unwrap() * dx);
                py.push(yn);
            }
        }
        BoundaryPolicy::Reflect => {
            let xn = x[n - 1];
            for i in 1..=pad_len {
                px.push(xn + (xn - x[n - 1 - i]));
                py.push(y[n - 1 - i]);
            }
        }
        BoundaryPolicy::Zero => {
            let xn = x[n - 1];
            let dx = x[n - 1] - x[n - 2];
            for i in 1..=pad_len {
                px.push(xn + T::from(i).unwrap() * dx);
                py.push(T::zero());
            }
        }
        BoundaryPolicy::NoBoundary => unreachable!(),
    }

    (px, py)
}
