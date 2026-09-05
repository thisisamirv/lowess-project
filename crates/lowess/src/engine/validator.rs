//! Input validation for LOWESS configuration and data.
//!
//! This module provides comprehensive validation functions for LOWESS
//! configuration parameters and input data. It checks requirements
//! such as input lengths, finite values, and parameter bounds.
// ## srrstats Compliance
//
// @srrstats {G2.0} Input validation: non-empty arrays, matching lengths, finite values.
// @srrstats {G2.1} Edge case handling: minimum points, parameter bounds, duplicates.
// @srrstats {G2.3} Informative error messages for invalid configuration.

// External dependencies
#[cfg(not(feature = "std"))]
use alloc::format;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use num_traits::Float;
#[cfg(feature = "std")]
use std::vec::Vec;

// Internal dependencies
use crate::primitives::errors::LowessError;

// Policy for handling non-finite (NaN/Inf) values in input data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MissingPolicy {
    // Return an error if any input value is non-finite (default).
    #[default]
    Error,

    // Silently remove observations where x or y is non-finite before fitting.
    Drop,
}

// Validation utility for LOWESS configuration and input data.
pub struct Validator;

impl Validator {
    // Validate non-empty arrays and matching lengths. Runs ahead of any
    // missing-value filtering, since a length mismatch is a caller error,
    // not a data-quality issue that `MissingPolicy::Drop` should mask.
    pub fn validate_lengths<T: Float>(x: &[T], y: &[T]) -> Result<(), LowessError> {
        if x.is_empty() || y.is_empty() {
            return Err(LowessError::EmptyInput);
        }
        if x.len() != y.len() {
            return Err(LowessError::MismatchedInputs {
                x_len: x.len(),
                y_len: y.len(),
            });
        }
        Ok(())
    }

    // Remove observations where x or y is non-finite, keeping any
    // `custom_weights` in lockstep. Used when `MissingPolicy::Drop` is set.
    pub fn drop_non_finite<T: Float>(
        x: &[T],
        y: &[T],
        custom_weights: Option<&[T]>,
    ) -> (Vec<T>, Vec<T>, Option<Vec<T>>) {
        let n = x.len();
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        let mut ws = custom_weights.map(|w| Vec::with_capacity(w.len().min(n)));
        for i in 0..n {
            if x[i].is_finite() && y[i].is_finite() {
                xs.push(x[i]);
                ys.push(y[i]);
                if let (Some(w), Some(wv)) = (custom_weights, ws.as_mut())
                    && i < w.len()
                {
                    wv.push(w[i]);
                }
            }
        }
        (xs, ys, ws)
    }

    // Validate input arrays for LOWESS smoothing.
    pub fn validate_inputs<T: Float>(x: &[T], y: &[T]) -> Result<(), LowessError> {
        Self::validate_lengths(x, y)?;
        let n = x.len();

        // Check: Sufficient points for regression
        if n < 2 {
            return Err(LowessError::TooFewPoints { got: n, min: 2 });
        }

        // Check: All values finite (combined loop for cache locality)
        for i in 0..n {
            if !x[i].is_finite() {
                return Err(LowessError::InvalidNumericValue(format!(
                    "x[{}]={}",
                    i,
                    x[i].to_f64().unwrap_or(f64::NAN)
                )));
            }
            if !y[i].is_finite() {
                return Err(LowessError::InvalidNumericValue(format!(
                    "y[{}]={}",
                    i,
                    y[i].to_f64().unwrap_or(f64::NAN)
                )));
            }
        }

        Ok(())
    }

    // Validate a single numeric value for finiteness.
    pub fn validate_scalar<T: Float>(val: T, name: &str) -> Result<(), LowessError> {
        if !val.is_finite() {
            return Err(LowessError::InvalidNumericValue(format!(
                "{}={}",
                name,
                val.to_f64().unwrap_or(f64::NAN)
            )));
        }
        Ok(())
    }

    // Validate the smoothing fraction (bandwidth) parameter.
    pub fn validate_fraction<T: Float>(fraction: T) -> Result<(), LowessError> {
        if !fraction.is_finite() || fraction <= T::zero() || fraction > T::one() {
            return Err(LowessError::InvalidFraction(
                fraction.to_f64().unwrap_or(f64::NAN),
            ));
        }
        Ok(())
    }

    // Validate the number of robustness iterations.
    pub fn validate_iterations(iterations: usize) -> Result<(), LowessError> {
        const MAX_ITERATIONS: usize = 1000;
        if iterations > MAX_ITERATIONS {
            return Err(LowessError::InvalidIterations(iterations));
        }
        Ok(())
    }

    // Validate the confidence/prediction interval level.
    pub fn validate_interval_level<T: Float>(level: T) -> Result<(), LowessError> {
        if !level.is_finite() || level <= T::zero() || level >= T::one() {
            return Err(LowessError::InvalidIntervals(
                level.to_f64().unwrap_or(f64::NAN),
            ));
        }
        Ok(())
    }

    // Validate a collection of candidate fractions for cross-validation.
    pub fn validate_cv_fractions<T: Float>(fracs: &[T]) -> Result<(), LowessError> {
        if fracs.is_empty() {
            return Err(LowessError::InvalidFraction(0.0));
        }

        for &f in fracs {
            Self::validate_fraction(f)?;
        }

        Ok(())
    }

    // Validate the number of folds for k-fold cross-validation.
    pub fn validate_kfold(k: usize) -> Result<(), LowessError> {
        if k < 2 {
            return Err(LowessError::InvalidNumericValue(format!(
                "k-fold must be at least 2, got {}",
                k
            )));
        }
        Ok(())
    }

    // Validate the auto-convergence tolerance.
    pub fn validate_tolerance<T: Float>(tol: T) -> Result<(), LowessError> {
        if !tol.is_finite() || tol <= T::zero() {
            return Err(LowessError::InvalidTolerance(
                tol.to_f64().unwrap_or(f64::NAN),
            ));
        }
        Ok(())
    }

    // Validate delta optimization parameter (equivalent to cell size in some contexts).
    pub fn validate_delta<T: Float>(delta: T) -> Result<(), LowessError> {
        if !delta.is_finite() || delta < T::zero() {
            return Err(LowessError::InvalidDelta(
                delta.to_f64().unwrap_or(f64::NAN),
            ));
        }
        Ok(())
    }

    // Validate the chunk size for shared processing in streaming mode.
    pub fn validate_chunk_size(chunk_size: usize, min: usize) -> Result<(), LowessError> {
        if chunk_size < min {
            return Err(LowessError::InvalidChunkSize {
                got: chunk_size,
                min,
            });
        }
        Ok(())
    }

    // Validate the overlap between consecutive chunks in streaming mode.
    pub fn validate_overlap(overlap: usize, chunk_size: usize) -> Result<(), LowessError> {
        if overlap >= chunk_size {
            return Err(LowessError::InvalidOverlap {
                overlap,
                chunk_size,
            });
        }
        Ok(())
    }

    // Validate the maximum capacity of the sliding window in online mode.
    pub fn validate_window_capacity(window_capacity: usize, min: usize) -> Result<(), LowessError> {
        if window_capacity < min {
            return Err(LowessError::InvalidWindowCapacity {
                got: window_capacity,
                min,
            });
        }
        Ok(())
    }

    // Validate the activation threshold for online smoothing.
    pub fn validate_min_points(
        min_points: usize,
        window_capacity: usize,
    ) -> Result<(), LowessError> {
        if min_points < 2 || min_points > window_capacity {
            return Err(LowessError::InvalidMinPoints {
                got: min_points,
                window_capacity,
            });
        }
        Ok(())
    }

    // Validate that no parameters were set multiple times in the builder.
    pub fn validate_no_duplicates(
        duplicate_param: Option<&'static str>,
    ) -> Result<(), LowessError> {
        if let Some(param) = duplicate_param {
            return Err(LowessError::DuplicateParameter { parameter: param });
        }
        Ok(())
    }

    // Validate per-observation custom weights:
    // - `weights` has the same length as the number of observations `n`
    // - All weight values are finite and non-negative
    pub fn validate_custom_weights<T: Float>(weights: &[T], n: usize) -> Result<(), LowessError> {
        if weights.len() != n {
            return Err(LowessError::InvalidInput(format!(
                "custom_weights length ({}) must match the number of observations ({})",
                weights.len(),
                n
            )));
        }
        for (i, &w) in weights.iter().enumerate() {
            if !w.is_finite() || w < T::zero() {
                return Err(LowessError::InvalidInput(format!(
                    "custom_weights[{}] = {} is invalid: weights must be finite and non-negative",
                    i,
                    w.to_f64().unwrap_or(f64::NAN)
                )));
            }
        }
        Ok(())
    }
}
