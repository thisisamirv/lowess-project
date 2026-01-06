//! Memory management and buffer recycling for LOWESS operations.
//!
//! ## Purpose
//!
//! This module provides centralized, reusable workspaces to minimize dynamic memory
//! allocations during LOWESS fitting. By allocating buffers once and recycling them
//! across multiple query points or robustness iterations, we significantly reduce
//! allocator pressure and improve cache locality.
//!
//! ## Design notes
//!
//! * **Centralized Ownership**: Buffer structs hold all necessary scratch space for
//!   their respective execution contexts (batch, online, streaming).
//! * **Lazy Expansion**: Buffers are grown on demand via `ensure_capacity` but never shrunk,
//!   stabilizing at the maximum required size for the dataset.
//! * **1D-focused**: This implementation is optimized for univariate LOWESS.
//!
//! ## Key concepts
//!
//! * **Slot**: A reusable vector wrapper with automatic capacity management.
//! * **LowessBuffer**: Working memory for the LOWESS executor (smoothed values, residuals, weights).
//! * **OnlineBuffer**: Scratch space for the online (incremental) adapter.
//! * **StreamingBuffer**: Overlap buffers for the streaming adapter.
//! * **CVBuffer**: Specialized buffers for cross-validation subsets.
//!
//! ## Invariants
//!
//! * Buffers are only logically cleared (e.g., `vec.clear()`), not deallocated, between iterations.
//! * Capacity is monotonically increasing; `ensure_capacity` only reallocates if current capacity is insufficient.
//!
//! ## Non-goals
//!
//! * Thread-local automatic caching (buffers are explicitly passed to allow parallel execution with one buffer per thread).
//! * Dynamic shrinking or aggressive memory reclamation (performance is prioritized over minimal footprint).

// Feature-gated dependencies
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

// External dependencies
use core::ops::{Deref, DerefMut};
use num_traits::{One, Zero};

// ============================================================================
// Slot - Unified Vector Abstraction
// ============================================================================

/// A reusable vector slot with automatic capacity management.
#[derive(Debug, Clone)]
pub struct Slot<T>(Vec<T>);

impl<T> Slot<T> {
    /// Create a new slot with the given initial capacity.
    #[inline]
    pub fn new(capacity: usize) -> Self {
        Self(Vec::with_capacity(capacity))
    }

    /// Clear the slot (sets length to 0, preserves capacity).
    #[inline]
    pub fn clear(&mut self) {
        self.0.clear();
    }

    /// Get a reference to the underlying vector.
    #[inline]
    pub fn as_vec(&self) -> &Vec<T> {
        &self.0
    }

    /// Get a mutable reference to the underlying vector.
    #[inline]
    pub fn as_vec_mut(&mut self) -> &mut Vec<T> {
        &mut self.0
    }
}

impl<T> Default for Slot<T> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<T> Deref for Slot<T> {
    type Target = Vec<T>;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for Slot<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T> From<Vec<T>> for Slot<T> {
    fn from(v: Vec<T>) -> Self {
        Self(v)
    }
}

// ============================================================================
// CV Buffers
// ============================================================================

/// Buffers used during cross-validation to hold training and test subsets.
#[derive(Debug, Clone)]
pub struct CVBuffer<T> {
    /// Training subset x-values.
    pub train_x: Vec<T>,
    /// Training subset y-values.
    pub train_y: Vec<T>,
    /// Test subset x-values (for K-Fold).
    pub test_x: Vec<T>,
    /// Test subset y-values (for K-Fold).
    pub test_y: Vec<T>,
    /// Sorted training subset x-values.
    pub sorted_train_x: Vec<T>,
    /// Sorted training subset y-values.
    pub sorted_train_y: Vec<T>,
}

impl<T> Default for CVBuffer<T> {
    fn default() -> Self {
        Self {
            train_x: Vec::new(),
            train_y: Vec::new(),
            test_x: Vec::new(),
            test_y: Vec::new(),
            sorted_train_x: Vec::new(),
            sorted_train_y: Vec::new(),
        }
    }
}

impl<T: Clone> CVBuffer<T> {
    /// Create a new CV buffer with estimated capacities.
    pub fn new() -> Self {
        Self {
            train_x: Vec::new(),
            train_y: Vec::new(),
            test_x: Vec::new(),
            test_y: Vec::new(),
            sorted_train_x: Vec::new(),
            sorted_train_y: Vec::new(),
        }
    }

    /// Ensure sufficient capacity for subsets.
    pub fn ensure_capacity(&mut self, n_total: usize, dims: usize) {
        if self.train_x.capacity() < n_total * dims {
            self.train_x.reserve(n_total * dims);
        }
        if self.train_y.capacity() < n_total {
            self.train_y.reserve(n_total);
        }
        // Test sets are usually smaller, but ensure safe defaults
        if self.test_x.capacity() < (n_total / 2 + 1) * dims {
            self.test_x.reserve((n_total / 2 + 1) * dims);
        }
        if self.test_y.capacity() < (n_total / 2 + 1) {
            self.test_y.reserve(n_total / 2 + 1);
        }
        if self.sorted_train_x.capacity() < n_total * dims {
            self.sorted_train_x.reserve(n_total * dims);
        }
        if self.sorted_train_y.capacity() < n_total {
            self.sorted_train_y.reserve(n_total);
        }
    }
}

// ============================================================================
// LowessBuffer - Working Memory for LOWESS Executor
// ============================================================================

/// Working memory for the LOWESS executor.
///
/// This buffer holds all scratch space needed during LOWESS smoothing iterations,
/// including smoothed values, convergence tracking, robustness weights, and kernel weights.
#[derive(Debug, Clone)]
pub struct LowessBuffer<T> {
    /// Current smoothed y-values.
    pub y_smooth: Slot<T>,

    /// Previous iteration values (for convergence check).
    pub y_prev: Slot<T>,

    /// Robustness weights (updated each iteration).
    pub robustness_weights: Slot<T>,

    /// Residuals buffer (y - y_smooth).
    pub residuals: Slot<T>,

    /// Kernel weights scratch buffer.
    pub weights: Slot<T>,
}

impl<T> Default for LowessBuffer<T> {
    fn default() -> Self {
        Self {
            y_smooth: Slot::default(),
            y_prev: Slot::default(),
            robustness_weights: Slot::default(),
            residuals: Slot::default(),
            weights: Slot::default(),
        }
    }
}

impl<T: Clone> LowessBuffer<T> {
    /// Create a buffer pre-allocated for `n` data points.
    pub fn with_capacity(n: usize) -> Self {
        Self {
            y_smooth: Slot::new(n),
            y_prev: Slot::new(n),
            robustness_weights: Slot::new(n),
            residuals: Slot::new(n),
            weights: Slot::new(n),
        }
    }

    /// Prepare buffers for a dataset of size `n`.
    /// Resizes all slots to `n` if they are smaller; clears them if they are larger.
    pub fn prepare(&mut self, n: usize, use_convergence: bool)
    where
        T: Zero + One + Clone,
    {
        // Resize or clear based on needs
        self.y_smooth.as_vec_mut().assign(n, T::zero());

        if use_convergence {
            self.y_prev.as_vec_mut().assign(n, T::zero());
        } else {
            self.y_prev.clear();
        }

        self.robustness_weights.as_vec_mut().assign(n, T::one());
        self.residuals.as_vec_mut().assign(n, T::zero());
        self.weights.as_vec_mut().assign(n, T::zero());
    }
}

/// Helper trait to simplify resizing and filling vectors.
pub trait VecExt<T> {
    /// Resize the vector to `n` and fill with `val`.
    fn assign(&mut self, n: usize, val: T);
    /// Replaces the vector contents with `slice`, reusing capacity.
    fn assign_slice(&mut self, slice: &[T]);
}

impl<T: Clone> VecExt<T> for Vec<T> {
    fn assign(&mut self, n: usize, val: T) {
        if self.len() != n {
            self.clear();
            self.resize(n, val);
        } else {
            self.fill(val);
        }
    }

    fn assign_slice(&mut self, slice: &[T]) {
        self.clear();
        self.extend_from_slice(slice);
    }
}

// ============================================================================
// OnlineBuffer - Scratch Space for Online Adapter
// ============================================================================

/// Scratch buffers for the online (incremental) LOWESS adapter.
///
/// These buffers hold temporary copies of the sliding window data
/// for use during smoothing operations.
#[derive(Debug, Clone)]
pub struct OnlineBuffer<T> {
    /// Scratch buffer for x-values from the sliding window.
    pub scratch_x: Slot<T>,

    /// Scratch buffer for y-values from the sliding window.
    pub scratch_y: Slot<T>,

    /// Scratch buffer for kernel weights.
    pub weights: Slot<T>,

    /// Scratch buffer for robustness weights.
    pub robustness_weights: Slot<T>,
}

impl<T> Default for OnlineBuffer<T> {
    fn default() -> Self {
        Self {
            scratch_x: Slot::default(),
            scratch_y: Slot::default(),
            weights: Slot::default(),
            robustness_weights: Slot::default(),
        }
    }
}

impl<T: Clone> OnlineBuffer<T> {
    /// Create a buffer pre-allocated for `capacity` points.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            scratch_x: Slot::new(capacity),
            scratch_y: Slot::new(capacity),
            weights: Slot::new(capacity),
            robustness_weights: Slot::new(capacity),
        }
    }

    /// Clear all slots (preserves capacity).
    pub fn clear(&mut self) {
        self.scratch_x.clear();
        self.scratch_y.clear();
        self.weights.clear();
        self.robustness_weights.clear();
    }
}

// ============================================================================
// StreamingBuffer - Overlap Buffers for Streaming Adapter
// ============================================================================

/// Overlap buffers for the streaming LOWESS adapter.
///
/// These buffers hold data from the overlap region of the previous chunk,
/// enabling smooth transitions between chunk boundaries.
#[derive(Debug, Clone)]
pub struct StreamingBuffer<T> {
    /// Overlap region x-values from the previous chunk.
    pub overlap_x: Slot<T>,

    /// Overlap region y-values from the previous chunk.
    pub overlap_y: Slot<T>,

    /// Smoothed values for the overlap region.
    pub overlap_smoothed: Slot<T>,

    /// Robustness weights for the overlap region.
    pub overlap_robustness_weights: Slot<T>,

    /// Reusable work buffer for LOWESS operations on chunks.
    pub work_buffer: LowessBuffer<T>,
}

impl<T> Default for StreamingBuffer<T> {
    fn default() -> Self {
        Self {
            overlap_x: Slot::default(),
            overlap_y: Slot::default(),
            overlap_smoothed: Slot::default(),
            overlap_robustness_weights: Slot::default(),
            work_buffer: LowessBuffer::default(),
        }
    }
}

impl<T: Clone> StreamingBuffer<T> {
    /// Create a buffer pre-allocated for `overlap` points and `chunk_size`.
    pub fn with_capacity(overlap: usize, chunk_size: usize) -> Self {
        Self {
            overlap_x: Slot::new(overlap),
            overlap_y: Slot::new(overlap),
            overlap_smoothed: Slot::new(overlap),
            overlap_robustness_weights: Slot::new(overlap),
            work_buffer: LowessBuffer::with_capacity(chunk_size),
        }
    }

    /// Clear all slots (preserves capacity).
    pub fn clear(&mut self) {
        self.overlap_x.clear();
        self.overlap_y.clear();
        self.overlap_smoothed.clear();
        self.overlap_robustness_weights.clear();
    }
}
