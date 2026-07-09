//! Online adapter for incremental LOWESS smoothing.
//!
//! This module provides the online (incremental) execution adapter for LOWESS
//! smoothing. It maintains a sliding window of recent observations and produces
//! smoothed values for new points as they arrive.
//!
//! ## srrstats Compliance
//!
//! @srrstats {G1.6} Sliding window with optional parallel re-smoothing.
//! @srrstats {G2.1} Configurable min_points threshold before smoothing starts.

// Feature-gated imports
#[cfg(feature = "cpu")]
use crate::engine::executor::smooth_pass_parallel;
#[cfg(feature = "gpu")]
use crate::engine::gpu::fit_pass_gpu;
use crate::input::LowessInput;

// External dependencies
use num_traits::Float;
use std::fmt::Debug;
use std::result::Result;

// Export dependencies from lowess crate
use lowess::internals::adapters::online::{OnlineLowess, OnlineLowessBuilder, OnlineOutput};
use lowess::internals::algorithms::regression::WLSSolver;
use lowess::internals::primitives::backend::Backend;
use lowess::internals::primitives::errors::LowessError;

// Builder for online LOWESS processor with parallel support.
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
        let mut base = OnlineLowessBuilder::default();
        base.parallel = Some(false); // Default to non-parallel in fastLowess for Online
        Self {
            base,
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

        // Configure the base builder with parallel callback if enabled
        let mut builder = self.base.clone();

        match builder.backend.unwrap_or(Backend::CPU) {
            Backend::CPU => {
                #[cfg(feature = "cpu")]
                {
                    if builder.parallel.unwrap_or(false) {
                        builder.custom_smooth_pass = Some(smooth_pass_parallel);
                    } else {
                        builder.custom_smooth_pass = None;
                    }
                }
                #[cfg(not(feature = "cpu"))]
                {
                    builder.custom_smooth_pass = None;
                }
            }
            Backend::GPU => {
                #[cfg(feature = "gpu")]
                {
                    // For Online, we currently use fit_pass_gpu as a hook
                    builder.custom_fit_pass = Some(fit_pass_gpu);
                }
                #[cfg(not(feature = "gpu"))]
                {
                    return Err(LowessError::UnsupportedFeature {
                        adapter: "Online",
                        feature: "GPU backend (requires 'gpu' feature)",
                    });
                }
            }
        }

        // Delegate execution to the base implementation
        let processor = builder.build()?;

        Ok(ParallelOnlineLowess { processor })
    }
}
