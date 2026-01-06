//! Sorting utilities for LOWESS input data.
//!
//! ## Purpose
//!
//! This module provides utilities for sorting input data by x-coordinates and
//! mapping results back to the original order.
//!
//! ## Design notes
//!
//! * **Stability**: Uses stable sorting to preserve the relative order of equal x-values.
//! * **Robustness**: Non-finite values (NaN, Inf) are moved to the end of the sequence.
//! * **Efficiency**: Maintains an O(n) index mapping for restoring original order.
//!
//! ## Key concepts
//!
//! ### Sort-Process-Unsort Pattern
//! 1. **Sort**: Input data is sorted by x-coordinates, creating an index mapping.
//! 2. **Process**: LOWESS smoothing operates on the sorted sequence.
//! 3. **Unsort**: Results are mapped back to original indices in O(n) time.
//!
//! ## Invariants
//!
//! * Sorted x-values are strictly non-decreasing (for finite values).
//! * The index mapping is a valid permutation of `0..n`.
//! * Non-finite values maintain their relative insertion order at the end.
//!
//! ## Non-goals
//!
//! * This module does not perform data validation or LOWESS calculation.
//!

// Feature-gated imports
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// External dependencies
use core::cmp::Ordering;
use num_traits::Float;

// ============================================================================
// Data Structures
// ============================================================================

/// Result of sorting input data by x-coordinates.
pub struct SortedData<T> {
    /// Sorted x-coordinates (finite values first).
    pub x: Vec<T>,

    /// Y-coordinates reordered to match sorted x-coordinates.
    pub y: Vec<T>,

    /// Index mapping where `indices[sorted_pos] = original_pos`.
    pub indices: Vec<usize>,
}

// ============================================================================
// Sorting Functions
// ============================================================================

/// Sort input data by x-coordinates in ascending order.
///
/// 1. Checks if data is already sorted (fast path).
/// 2. Pairs x with original indices.
///    - We only sort x and index to keep the tuple size small (16 bytes for f64 vs 24 bytes)
///    - This reduces data movement during sorting.
/// 3. Performs a stable sort.
/// 4. Extracts sorted arrays and permutation mapping.
#[inline]
pub fn sort_by_x<T: Float>(x: &[T], y: &[T]) -> SortedData<T> {
    let n = x.len();

    // Fast path: check if data is already sorted by x
    let is_sorted = x.windows(2).all(|w| w[0] <= w[1]);
    if is_sorted {
        return SortedData {
            x: x.to_vec(),
            y: y.to_vec(),
            indices: (0..n).collect(),
        };
    }

    // Create tuples of (x_value, original_index)
    // We only sort x and index to keep the tuple size small (16 bytes for f64 vs 24 bytes)
    let mut pairs: Vec<(T, usize)> = x.iter().enumerate().map(|(i, &xi)| (xi, i)).collect();

    // Stable sort to preserve order of equal x values for determinism
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

    // Extract sorted components
    SortedData {
        x: pairs.iter().map(|p| p.0).collect(),
        y: pairs.iter().map(|p| y[p.1]).collect(),
        indices: pairs.iter().map(|p| p.1).collect(),
    }
}

/// Map sorted results back to the original input order in O(n) time.
#[inline]
pub fn unsort<T: Float>(sorted_values: &[T], indices: &[usize]) -> Vec<T> {
    let n = indices.len();
    let mut result = vec![T::zero(); n];

    // Map each sorted position back to its original position
    for (sorted_idx, &orig_idx) in indices.iter().enumerate() {
        result[orig_idx] = sorted_values[sorted_idx];
    }

    result
}
