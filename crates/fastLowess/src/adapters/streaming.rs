//! Streaming adapter for large-scale LOWESS smoothing.
//!
//! This module provides the streaming execution adapter for LOWESS smoothing
//! on datasets too large to fit in memory. It divides the data into overlapping
//! chunks, processes each chunk independently, and merges the results while
//! handling boundary effects.
//!
//! ## srrstats Compliance
//!
//! @srrstats {G1.6} Chunk-based streaming with parallel execution per chunk.
//! @srrstats {G3.0} Rayon parallelization injected for chunk processing.

// Internal dependencies
#[cfg(feature = "cpu")]
use crate::engine::executor::smooth_pass_parallel;
#[cfg(feature = "gpu")]
use crate::engine::gpu::fit_pass_gpu;

// External dependencies
use num_traits::Float;
use std::fmt::Debug;
use std::result::Result;

// Export dependencies from lowess crate
use lowess::internals::adapters::streaming::{StreamingLowess, StreamingLowessBuilder};
use lowess::internals::algorithms::regression::WLSSolver;
use lowess::internals::engine::output::LowessResult;
use lowess::internals::primitives::backend::Backend;
use lowess::internals::primitives::errors::LowessError;

// Builder for streaming LOWESS processor with parallel support.
#[derive(Debug, Clone)]
pub struct ParallelStreamingLowessBuilder<T: Float> {
    // Base builder from the lowess crate
    pub base: StreamingLowessBuilder<T>,
    // Parse errors from string-accepting builder methods; reported together by `build()`.
    pub(crate) parse_errors: Vec<LowessError>,
}

impl<T: Float> Default for ParallelStreamingLowessBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(private_bounds)]
impl<T: Float> ParallelStreamingLowessBuilder<T> {
    fn new() -> Self {
        let mut base = StreamingLowessBuilder::default();
        base.parallel = Some(true); // Default to parallel in fastLowess for Streaming
        Self {
            base,
            parse_errors: Vec::new(),
        }
    }
}

// Streaming LOWESS processor with parallel support.
pub struct ParallelStreamingLowess<T: Float> {
    processor: StreamingLowess<T>,
}

impl<T: Float + WLSSolver + Debug + Send + Sync + 'static> ParallelStreamingLowess<T> {
    // Process a chunk of data.
    pub fn process_chunk(&mut self, x: &[T], y: &[T]) -> Result<LowessResult<T>, LowessError> {
        self.processor.process_chunk(x, y)
    }

    // Finalize processing and get remaining buffered data.
    pub fn finalize(&mut self) -> Result<LowessResult<T>, LowessError> {
        self.processor.finalize()
    }
}

#[allow(private_bounds)]
impl<T: Float + WLSSolver + Debug + Send + Sync + 'static> ParallelStreamingLowessBuilder<T> {
    // Build the streaming processor.
    pub fn build(self) -> Result<ParallelStreamingLowess<T>, LowessError> {
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
                    if builder.parallel.unwrap_or(true) {
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
                    builder.custom_fit_pass = Some(fit_pass_gpu);
                }
                #[cfg(not(feature = "gpu"))]
                {
                    return Err(LowessError::UnsupportedFeature {
                        adapter: "Streaming",
                        feature: "GPU backend (requires 'gpu' feature)",
                    });
                }
            }
        }

        // Delegate execution to the base implementation
        let processor = builder.build()?;

        Ok(ParallelStreamingLowess { processor })
    }
}
