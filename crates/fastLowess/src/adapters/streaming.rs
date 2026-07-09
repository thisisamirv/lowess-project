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
use crate::parse::IntoEnum;
use lowess::internals::adapters::streaming::{
    MergeStrategy, StreamingLowess, StreamingLowessBuilder,
};
use lowess::internals::algorithms::regression::{WLSSolver, ZeroWeightFallback};
use lowess::internals::algorithms::robustness::RobustnessMethod;
use lowess::internals::engine::output::LowessResult;
use lowess::internals::math::boundary::BoundaryPolicy;
use lowess::internals::math::kernel::WeightFunction;
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
    // Create a new streaming LOWESS builder with default parameters.
    fn new() -> Self {
        let mut base = StreamingLowessBuilder::default();
        base.parallel = Some(true); // Default to parallel in fastLowess for Streaming
        Self {
            base,
            parse_errors: Vec::new(),
        }
    }

    // Set parallel execution mode.
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.base.parallel = Some(parallel);
        self
    }

    // Set the execution backend.
    pub fn backend(mut self, backend: Backend) -> Self {
        self.base.backend = Some(backend);
        self
    }

    // Set the smoothing fraction (span).
    pub fn fraction(mut self, fraction: T) -> Self {
        self.base.fraction = fraction;
        self
    }

    // Set the number of robustness iterations.
    pub fn iterations(mut self, iterations: usize) -> Self {
        self.base.iterations = iterations;
        self
    }

    // Set the delta parameter for interpolation optimization.
    pub fn delta(mut self, delta: T) -> Self {
        self.base.delta = delta;
        self
    }

    // Set the kernel weight function.
    pub fn weight_function(mut self, wf: impl IntoEnum<WeightFunction>) -> Self {
        match wf.into_enum() {
            Ok(w) => self.base.weight_function = w,
            Err(e) => {
                self.parse_errors.push(e);
            }
        }
        self
    }

    // Set the robustness method for outlier handling.
    pub fn robustness_method(mut self, method: impl IntoEnum<RobustnessMethod>) -> Self {
        match method.into_enum() {
            Ok(m) => self.base.robustness_method = m,
            Err(e) => {
                self.parse_errors.push(e);
            }
        }
        self
    }

    // Set the zero-weight fallback policy.
    pub fn zero_weight_fallback(mut self, fallback: impl IntoEnum<ZeroWeightFallback>) -> Self {
        match fallback.into_enum() {
            Ok(f) => self.base.zero_weight_fallback = f,
            Err(e) => {
                self.parse_errors.push(e);
            }
        }
        self
    }

    // Set the boundary handling policy.
    pub fn boundary_policy(mut self, policy: impl IntoEnum<BoundaryPolicy>) -> Self {
        match policy.into_enum() {
            Ok(p) => self.base.boundary_policy = p,
            Err(e) => {
                self.parse_errors.push(e);
            }
        }
        self
    }

    // Enable auto-convergence for robustness iterations.
    pub fn auto_converge(mut self, tolerance: T) -> Self {
        self.base.auto_converge = Some(tolerance);
        self
    }

    // Enable returning residuals in the output.
    pub fn compute_residuals(mut self, enabled: bool) -> Self {
        self.base.compute_residuals = enabled;
        self
    }

    // Enable returning robustness weights in the result.
    pub fn return_robustness_weights(mut self, enabled: bool) -> Self {
        self.base.return_robustness_weights = enabled;
        self
    }

    // Set the chunk size for processing.
    pub fn chunk_size(mut self, size: usize) -> Self {
        self.base.chunk_size = size;
        self
    }

    // Set the overlap between consecutive chunks.
    pub fn overlap(mut self, size: usize) -> Self {
        self.base.overlap = size;
        self
    }

    // Set the merge strategy for overlapping values.
    pub fn merge_strategy(mut self, strategy: impl IntoEnum<MergeStrategy>) -> Self {
        match strategy.into_enum() {
            Ok(s) => self.base.merge_strategy = s,
            Err(e) => {
                self.parse_errors.push(e);
            }
        }
        self
    }

    // Enable returning diagnostics in the result.
    pub fn return_diagnostics(mut self, enabled: bool) -> Self {
        self.base.return_diagnostics = enabled;
        self
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
