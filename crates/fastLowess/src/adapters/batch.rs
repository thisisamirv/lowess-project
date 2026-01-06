//! Batch adapter for standard LOWESS smoothing.
//!
//! ## Purpose
//!
//! This module provides the batch execution adapter for LOWESS smoothing.
//! It handles complete datasets in memory with optional parallel processing,
//! making it suitable for small to medium-sized datasets.
//!
//! ## Design notes
//!
//! * **Processing**: Processes entire dataset in a single pass.
//! * **Sorting**: Automatically sorts data by x-values and unsorts results.
//! * **Delegation**: Delegates computation to the execution engine.
//! * **Parallelism**: Adds parallel execution via `rayon` (fastLowess extension).
//! * **Generics**: Generic over `Float` types.
//!
//! ## Key concepts
//!
//! * **Batch Processing**: Validates, sorts, filters, executes, and unsorts.
//! * **Builder Pattern**: Fluent API for configuration with sensible defaults.
//! * **Automatic Sorting**: Ensures x-values are sorted for the algorithm.
//! * **Parallel Execution**: Uses Rayon for multi-threaded processing.
//!
//! ## Invariants
//!
//! * Input arrays x and y must have the same length.
//! * All values must be finite.
//! * At least 2 data points are required.
//! * Output order matches input order.
//!
//! ## Non-goals
//!
//! * This adapter does not handle streaming data (use streaming adapter).
//! * This adapter does not handle incremental updates (use online adapter).
//! * This adapter does not handle missing values.

// Feature-gated imports
#[cfg(feature = "cpu")]
use crate::engine::executor::smooth_pass_parallel;
#[cfg(feature = "gpu")]
use crate::engine::gpu::fit_pass_gpu;
#[cfg(feature = "cpu")]
use crate::evaluation::cv::cv_pass_parallel;
#[cfg(feature = "cpu")]
use crate::evaluation::intervals::interval_pass_parallel;

// External dependencies
use num_traits::Float;
use std::fmt::Debug;
use std::result::Result;

// Export dependencies from lowess crate
use lowess::internals::adapters::batch::BatchLowessBuilder;
use lowess::internals::algorithms::regression::WLSSolver;
use lowess::internals::algorithms::regression::ZeroWeightFallback;
use lowess::internals::algorithms::robustness::RobustnessMethod;
use lowess::internals::engine::output::LowessResult;
use lowess::internals::evaluation::cv::CVKind;
use lowess::internals::math::boundary::BoundaryPolicy;
use lowess::internals::math::kernel::WeightFunction;
use lowess::internals::primitives::backend::Backend;
use lowess::internals::primitives::errors::LowessError;

// Internal dependencies
use crate::input::LowessInput;

// ============================================================================
// Extended Batch LOWESS Builder
// ============================================================================

/// Builder for batch LOWESS processor with parallel support.
#[derive(Debug, Clone)]
pub struct ParallelBatchLowessBuilder<T: Float> {
    /// Base builder from the lowess crate
    pub base: BatchLowessBuilder<T>,
}

impl<T: Float> Default for ParallelBatchLowessBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> ParallelBatchLowessBuilder<T> {
    /// Create a new batch LOWESS builder with default parameters.
    ///
    /// # Defaults
    ///
    /// * All base parameters from lowess BatchLowessBuilder
    /// * parallel: true (fastLowess extension)
    fn new() -> Self {
        let base = BatchLowessBuilder::default().parallel(true); // Default to parallel in fastLowess
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
    // Batch-Specific Setters
    // ========================================================================

    /// Enable returning diagnostics in the result.
    pub fn return_diagnostics(mut self, enabled: bool) -> Self {
        self.base = self.base.return_diagnostics(enabled);
        self
    }

    /// Enable confidence intervals at the specified level.
    pub fn confidence_intervals(mut self, level: T) -> Self {
        self.base = self.base.confidence_intervals(level);
        self
    }

    /// Enable prediction intervals at the specified level.
    pub fn prediction_intervals(mut self, level: T) -> Self {
        self.base = self.base.prediction_intervals(level);
        self
    }

    /// Enable cross-validation with the specified fractions.
    pub fn cross_validate(mut self, fractions: Vec<T>) -> Self {
        self.base = self.base.cross_validate(fractions);
        self
    }

    /// Set the cross-validation method.
    pub fn cv_kind(mut self, method: CVKind) -> Self {
        self.base = self.base.cv_kind(method);
        self
    }

    // ========================================================================
    // Build Method
    // ========================================================================

    /// Build the batch processor.
    pub fn build(self) -> Result<ParallelBatchLowess<T>, LowessError> {
        // Check for deferred errors from adapter conversion
        if let Some(ref err) = self.base.deferred_error {
            return Err(err.clone());
        }

        // Validate by attempting to build the base processor
        // This reuses the validation logic centralized in the lowess crate
        let _ = self.base.clone().build()?;

        Ok(ParallelBatchLowess { config: self })
    }
}

// ============================================================================
// Extended Batch LOWESS Processor
// ============================================================================

/// Batch LOWESS processor with parallel support.
pub struct ParallelBatchLowess<T: Float> {
    config: ParallelBatchLowessBuilder<T>,
}

impl<T: Float + WLSSolver + Debug + Send + Sync + 'static> ParallelBatchLowess<T> {
    /// Perform LOWESS smoothing on the provided data.
    pub fn fit<I1, I2>(self, x: &I1, y: &I2) -> Result<LowessResult<T>, LowessError>
    where
        I1: LowessInput<T> + ?Sized,
        I2: LowessInput<T> + ?Sized,
    {
        let x_slice = x.as_lowess_slice()?;
        let y_slice = y.as_lowess_slice()?;

        // Configure the base builder with parallel callback if enabled
        let mut builder = self.config.base;

        match builder.backend.unwrap_or(Backend::CPU) {
            Backend::CPU => {
                #[cfg(feature = "cpu")]
                {
                    if builder.parallel.unwrap_or(true) {
                        builder = builder
                            .custom_smooth_pass(smooth_pass_parallel)
                            .custom_cv_pass(cv_pass_parallel)
                            .custom_interval_pass(interval_pass_parallel);
                    } else {
                        // Resets - though they are None by default
                        // but explicitly clearing just in case
                        builder.custom_smooth_pass = None;
                        builder.custom_cv_pass = None;
                        builder.custom_interval_pass = None;
                    }
                }
                #[cfg(not(feature = "cpu"))]
                {
                    // Fallback to sequential if cpu feature is disabled
                    builder.custom_smooth_pass = None;
                    builder.custom_cv_pass = None;
                    builder.custom_interval_pass = None;
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
                        adapter: "Batch",
                        feature: "GPU backend (requires 'gpu' feature)",
                    });
                }
            }
        }

        // Delegate execution to the base implementation
        let processor = builder.build()?;
        processor.fit(x_slice, y_slice)
    }
}
