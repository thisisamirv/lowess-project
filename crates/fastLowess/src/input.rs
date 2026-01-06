//! Input abstractions for LOWESS smoothing.
//!
//! ## Purpose
//!
//! This module provides a unified abstraction for LOWESS inputs, allowing the
//! `fit` method to process multiple data formats (slices, vectors, ndarray)
//! through a single interface.
//!
//! ## Design notes
//!
//! * **Zero-copy where possible**: Provides direct slice access to underlying data buffers.
//! * **Interoperability**: Bridges standard Rust collections with specialized numerical libraries.
//! * **Fail-fast validation**: Ensures memory continuity for multi-dimensional types before processing.
//!
//! ## Key concepts
//!
//! * **LowessInput Trait**: The core abstraction that requires types to provide a contiguous slice view.
//! * **Memory Continuity**: Essential for efficient LOWESS kernel processing.
//!
//! ## Invariants
//!
//! * Returned slices must represent all elements in the input container.
//! * Inputs must be contiguous in memory; non-contiguous inputs return an error.
//!
//! ## Non-goals
//!
//! * This module does not perform data cleaning or imputation.
//! * This module does not handle data reshaping or dimensionality reduction.

// External dependencies
use ndarray::{ArrayBase, Data, Ix1};
use num_traits::Float;

// Export dependencies from lowess crate
use lowess::internals::primitives::errors::LowessError;

/// Trait for types that can be used as input for LOWESS smoothing.
pub trait LowessInput<T: Float> {
    /// Convert the input to a contiguous slice.
    fn as_lowess_slice(&self) -> Result<&[T], LowessError>;
}

impl<T: Float> LowessInput<T> for [T] {
    fn as_lowess_slice(&self) -> Result<&[T], LowessError> {
        Ok(self)
    }
}

impl<T: Float> LowessInput<T> for Vec<T> {
    fn as_lowess_slice(&self) -> Result<&[T], LowessError> {
        Ok(self.as_slice())
    }
}

impl<T: Float, S> LowessInput<T> for ArrayBase<S, Ix1>
where
    S: Data<Elem = T>,
{
    fn as_lowess_slice(&self) -> Result<&[T], LowessError> {
        self.as_slice().ok_or_else(|| {
            LowessError::InvalidInput("ndarray input must be contiguous in memory".to_string())
        })
    }
}
