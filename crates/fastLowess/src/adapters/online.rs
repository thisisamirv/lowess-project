//! Online adapter for incremental LOWESS smoothing.
//!
//! This module provides the online (incremental) execution adapter for LOWESS
//! smoothing. It maintains a sliding window of recent observations and produces
//! smoothed values for new points as they arrive.
// ## srrstats Compliance
//
// @srrstats {G1.6} Sliding window for real-time incremental updates (always sequential; no parallel/GPU backend).
// @srrstats {G2.1} Configurable min_points threshold before smoothing starts.

// Feature-gated imports
use crate::input::LowessInput;

// External dependencies
use num_traits::Float;
use std::fmt::Debug;
use std::result::Result;

// Export dependencies from lowess crate
use lowess::internals::adapters::online::{OnlineLowess, OnlineLowessBuilder, OnlineOutput};
use lowess::internals::algorithms::regression::WLSSolver;
use lowess::internals::primitives::errors::LowessError;

// Builder for online LOWESS processor. Online processing is always sequential
// (one point at a time), so there is no parallel or GPU backend option here.
#[derive(Debug, Clone)]
pub struct ParallelOnlineLowessBuilder<T: Float> {
    // Base builder from the lowess crate
    pub base: OnlineLowessBuilder<T>,
    // Parse errors from string-accepting builder methods; reported together by `build()`.
    pub(crate) parse_errors: Vec<LowessError>,
}

impl<T: Float> Default for ParallelOnlineLowessBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(private_bounds)]
impl<T: Float> ParallelOnlineLowessBuilder<T> {
    fn new() -> Self {
        Self {
            base: OnlineLowessBuilder::default(),
            parse_errors: Vec::new(),
        }
    }
}

// Online LOWESS processor with parallel support.
pub struct ParallelOnlineLowess<T: Float> {
    processor: OnlineLowess<T>,
}

impl<T: Float + WLSSolver + Debug + Send + Sync + 'static> ParallelOnlineLowess<T> {
    // Add a new point and return the smoothed value.
    pub fn add_point(&mut self, x: T, y: T) -> Result<Option<OnlineOutput<T>>, LowessError> {
        self.processor.add_point(x, y)
    }

    // Add multiple points and return their smoothed values.
    pub fn add_points<I1, I2>(
        &mut self,
        x: &I1,
        y: &I2,
    ) -> Result<Vec<Option<OnlineOutput<T>>>, LowessError>
    where
        I1: LowessInput<T> + ?Sized,
        I2: LowessInput<T> + ?Sized,
    {
        let x_slice = x.as_lowess_slice()?;
        let y_slice = y.as_lowess_slice()?;

        if x_slice.len() != y_slice.len() {
            return Err(LowessError::InvalidInput("x and y lengths differ".into()));
        }

        let mut results = Vec::with_capacity(x_slice.len());
        for (xi, yi) in x_slice.iter().zip(y_slice.iter()) {
            results.push(self.add_point(*xi, *yi)?);
        }
        Ok(results)
    }

    // Reset the processor, clearing all window data.
    pub fn reset(&mut self) {
        self.processor.reset();
    }
}

#[allow(private_bounds)]
impl<T: Float + WLSSolver + Debug + Send + Sync + 'static> ParallelOnlineLowessBuilder<T> {
    // Build the online processor.
    pub fn build(self) -> Result<ParallelOnlineLowess<T>, LowessError> {
        // Check for deferred parse errors
        if !self.parse_errors.is_empty() {
            return Err(LowessError::ParseErrors(self.parse_errors));
        }

        // Online processing is always sequential (one point at a time), so
        // there is no parallel/GPU dispatch here — delegate directly.
        let processor = self.base.build()?;

        Ok(ParallelOnlineLowess { processor })
    }
}
