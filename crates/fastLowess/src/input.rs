//! Input abstractions for LOWESS smoothing.
//!
//! This module provides a unified abstraction for LOWESS inputs, allowing the
//! `fit` method to process multiple data formats (slices, vectors, ndarray)
//! through a single interface.

// External dependencies
#[cfg(feature = "cpu")]
use ndarray::{ArrayBase, Data, Ix1};
use num_traits::Float;

// Export dependencies from lowess crate
use lowess::internals::primitives::errors::LowessError;

// Trait for types that can be used as input for LOWESS smoothing.
pub trait LowessInput<T: Float> {
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

#[cfg(feature = "cpu")]
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
