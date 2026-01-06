//! Streaming adapter for large-scale LOWESS smoothing.
//!
//! ## Purpose
//!
//! This module provides the streaming execution adapter for LOWESS smoothing
//! on datasets too large to fit in memory. It divides the data into overlapping
//! chunks, processes each chunk independently, and merges the results while
//! handling boundary effects.
//!
//! ## Design notes
//!
//! * **Strategy**: Processes data in fixed-size chunks with configurable overlap.
//! * **Merging**: Merges overlapping regions using configurable strategies (Average, Weighted).
//! * **Sorting**: Automatically sorts data within each chunk.
//! * **Parallelism**: Adds parallel execution via `rayon` (fastLowess extension).
//! * **Generics**: Generic over `Float` types.
//!
//! ## Key concepts
//!
//! * **Chunked Processing**: Divides stream into `chunk_size` pieces.
//! * **Overlap**: Ensures smooth transitions, typically 2x window size.
//! * **Merging**: Handles value conflicts in overlapping regions.
//! * **Boundary Policies**: Handles edge effects at stream start/end.
//!
//! ## Invariants
//!
//! * Chunk size must be larger than overlap.
//! * Overlap must be sufficient for local smoothing window.
//! * Values must be finite.
//! * At least 2 points per chunk.
//!
//! ## Non-goals
//!
//! * This adapter does not support confidence/prediction intervals.
//! * This adapter does not support cross-validation.
//! * This adapter does not handle batch processing.
//! * This adapter does not handle incremental updates.
//! * This adapter requires chunks to be provided in stream order.

// Feature-gated imports
#[cfg(feature = "cpu")]
use crate::engine::executor::smooth_pass_parallel;
#[cfg(feature = "gpu")]
use crate::engine::gpu::fit_pass_gpu;

// External dependencies
use num_traits::Float;
use std::fmt::Debug;
use std::result::Result;

// Export dependencies from lowess crate
use lowess::internals::adapters::streaming::MergeStrategy;
use lowess::internals::adapters::streaming::{StreamingLowess, StreamingLowessBuilder};
use lowess::internals::algorithms::regression::WLSSolver;
use lowess::internals::algorithms::regression::ZeroWeightFallback;
use lowess::internals::algorithms::robustness::RobustnessMethod;
use lowess::internals::engine::output::LowessResult;
use lowess::internals::math::boundary::BoundaryPolicy;
use lowess::internals::math::kernel::WeightFunction;
use lowess::internals::primitives::backend::Backend;
use lowess::internals::primitives::errors::LowessError;

// ============================================================================
// Extended Streaming LOWESS Builder
// ============================================================================

/// Builder for streaming LOWESS processor with parallel support.
#[derive(Debug, Clone)]
pub struct ParallelStreamingLowessBuilder<T: Float> {
    /// Base builder from the lowess crate
    pub base: StreamingLowessBuilder<T>,
}

impl<T: Float> Default for ParallelStreamingLowessBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> ParallelStreamingLowessBuilder<T> {
    /// Create a new streaming LOWESS builder with default parameters.
    fn new() -> Self {
        let base = StreamingLowessBuilder::default().parallel(true); // Default to parallel in fastLowess for Streaming
        Self { base }
    }

    /// Set parallel execution mode.
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.base = self.base.parallel(parallel);
        self
    }

    /// Set the execution backend.
    pub fn backend(mut self, backend: Backend) -> Self {
        self.base = self.base.backend(backend);
        self
    }

    // ========================================================================
    // Shared Setters
    // ========================================================================

    /// Set the smoothing fraction (span).
    pub fn fraction(mut self, fraction: T) -> Self {
        self.base = self.base.fraction(fraction);
        self
    }

    /// Set the number of robustness iterations.
    pub fn iterations(mut self, iterations: usize) -> Self {
        self.base = self.base.iterations(iterations);
        self
    }

    /// Set the delta parameter for interpolation optimization.
    pub fn delta(mut self, delta: T) -> Self {
        self.base = self.base.delta(delta);
        self
    }

    /// Set the kernel weight function.
    pub fn weight_function(mut self, wf: WeightFunction) -> Self {
        self.base = self.base.weight_function(wf);
        self
    }

    /// Set the robustness method for outlier handling.
    pub fn robustness_method(mut self, method: RobustnessMethod) -> Self {
        self.base = self.base.robustness_method(method);
        self
    }

    /// Set the zero-weight fallback policy.
    pub fn zero_weight_fallback(mut self, fallback: ZeroWeightFallback) -> Self {
        self.base = self.base.zero_weight_fallback(fallback);
        self
    }

    /// Set the boundary handling policy.
    pub fn boundary_policy(mut self, policy: BoundaryPolicy) -> Self {
        self.base = self.base.boundary_policy(policy);
        self
    }

    /// Enable auto-convergence for robustness iterations.
    pub fn auto_converge(mut self, tolerance: T) -> Self {
        self.base = self.base.auto_converge(tolerance);
        self
    }

    /// Enable returning residuals in the output.
    pub fn compute_residuals(mut self, enabled: bool) -> Self {
        self.base = self.base.compute_residuals(enabled);
        self
    }

    /// Enable returning robustness weights in the result.
    pub fn return_robustness_weights(mut self, enabled: bool) -> Self {
        self.base = self.base.return_robustness_weights(enabled);
        self
    }

    // ========================================================================
    // Streaming-Specific Setters
    // ========================================================================

    /// Set the chunk size for processing.
    pub fn chunk_size(mut self, size: usize) -> Self {
        self.base = self.base.chunk_size(size);
        self
    }

    /// Set the overlap between consecutive chunks.
    pub fn overlap(mut self, size: usize) -> Self {
        self.base = self.base.overlap(size);
        self
    }

    /// Set the merge strategy for overlapping values.
    pub fn merge_strategy(mut self, strategy: MergeStrategy) -> Self {
        self.base = self.base.merge_strategy(strategy);
        self
    }

    /// Enable returning diagnostics in the result.
    pub fn return_diagnostics(mut self, enabled: bool) -> Self {
        self.base = self.base.return_diagnostics(enabled);
        self
    }
}

// ============================================================================
// Extended Streaming LOWESS Processor
// ============================================================================

/// Streaming LOWESS processor with parallel support.
pub struct ParallelStreamingLowess<T: Float> {
    processor: StreamingLowess<T>,
}

impl<T: Float + WLSSolver + Debug + Send + Sync + 'static> ParallelStreamingLowess<T> {
    /// Process a chunk of data.
    pub fn process_chunk(&mut self, x: &[T], y: &[T]) -> Result<LowessResult<T>, LowessError> {
        self.processor.process_chunk(x, y)
    }

    /// Finalize processing and get remaining buffered data.
    pub fn finalize(&mut self) -> Result<LowessResult<T>, LowessError> {
        self.processor.finalize()
    }
}

impl<T: Float + WLSSolver + Debug + Send + Sync + 'static> ParallelStreamingLowessBuilder<T> {
    /// Build the streaming processor.
    pub fn build(self) -> Result<ParallelStreamingLowess<T>, LowessError> {
        // Check for deferred errors from adapter conversion
        if let Some(ref err) = self.base.deferred_error {
            return Err(err.clone());
        }

        // Configure the base builder with parallel callback if enabled
        let mut builder = self.base.clone();

        match builder.backend.unwrap_or(Backend::CPU) {
            Backend::CPU => {
                #[cfg(feature = "cpu")]
                {
                    if builder.parallel.unwrap_or(true) {
                        builder = builder.custom_smooth_pass(smooth_pass_parallel);
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
