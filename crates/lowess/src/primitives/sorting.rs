//! Sorting utilities for LOWESS input data.
//!
//! This module provides utilities for sorting input data by x-coordinates and
//! mapping results back to the original order.

// External dependencies
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::cmp::Ordering;
use num_traits::Float;

// Result of sorting input data by x-coordinates.
pub struct SortedData<T> {
    // Sorted x-coordinates (finite values first).
    pub x: Vec<T>,

    // Y-coordinates reordered to match sorted x-coordinates.
    pub y: Vec<T>,

    // Index mapping where `indices[sorted_pos] = original_pos`.
    pub indices: Vec<usize>,
}

// Sort input data by x-coordinates in ascending order.
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

// Map sorted results back to the original input order in O(n) time.
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
