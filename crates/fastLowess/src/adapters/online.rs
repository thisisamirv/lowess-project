//! Online adapter for incremental LOWESS smoothing.
//!
//! ## Purpose
//!
//! This module provides the online (incremental) execution adapter for LOWESS
//! smoothing. It maintains a sliding window of recent observations and produces
//! smoothed values for new points as they arrive.
//!
//! ## Design notes
//!
//! * **Storage**: Uses a fixed-size circular buffer (VecDeque) for the sliding window.
//! * **Eviction**: Automatically evicts oldest points when capacity is reached.
//! * **Processing**: Performs smoothing on the current window for each new point.
//! * **Parallelism**: Adds parallel execution via `rayon` (fastLowess extension).
//! * **Generics**: Generic over `Float` types.
//!
//! ## Key concepts
//!
//! * **Sliding Window**: Maintains recent history up to `capacity`.
//! * **Incremental Processing**: Validates, adds, evicts, and smooths.
//! * **Initialization Phase**: Returns `None` until `min_points` are accumulated.
//! * **Update Modes**: Supports `Incremental` (fast) and `Full` (accurate) modes.
//!
//! ## Invariants
//!
//! * Window size never exceeds capacity.
//! * All values in window are finite.
//! * At least `min_points` are required before smoothing.
//! * Window maintains insertion order (oldest to newest).
//!
//! ## Non-goals
//!
//! * This adapter does not support confidence/prediction intervals.
//! * This adapter does not compute diagnostic statistics.
//! * This adapter does not support cross-validation.
//! * This adapter does not handle out-of-order points.

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
use lowess::internals::adapters::online::OnlineOutput;
use lowess::internals::adapters::online::UpdateMode;
use lowess::internals::adapters::online::{OnlineLowess, OnlineLowessBuilder};
use lowess::internals::algorithms::regression::WLSSolver;
use lowess::internals::algorithms::regression::ZeroWeightFallback;
use lowess::internals::algorithms::robustness::RobustnessMethod;
use lowess::internals::math::boundary::BoundaryPolicy;
use lowess::internals::math::kernel::WeightFunction;
use lowess::internals::primitives::backend::Backend;
use lowess::internals::primitives::errors::LowessError;

// Internal dependencies
use crate::input::LowessInput;

// ============================================================================
// Extended Online LOWESS Builder
// ============================================================================

/// Builder for online LOWESS processor with parallel support.
#[derive(Debug, Clone)]
pub struct ParallelOnlineLowessBuilder<T: Float> {
    /// Base builder from the lowess crate
    pub base: OnlineLowessBuilder<T>,
}

impl<T: Float> Default for ParallelOnlineLowessBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> ParallelOnlineLowessBuilder<T> {
    /// Create a new online LOWESS builder with default parameters.
    fn new() -> Self {
        let base = OnlineLowessBuilder::default().parallel(false); // Default to non-parallel in fastLowess for Online
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
    // Online-Specific Setters
    // ========================================================================

    /// Set the maximum window capacity.
    pub fn window_capacity(mut self, capacity: usize) -> Self {
        self.base = self.base.window_capacity(capacity);
        self
    }

    /// Set the minimum points required before smoothing starts.
    pub fn min_points(mut self, min_points: usize) -> Self {
        self.base = self.base.min_points(min_points);
        self
    }

    /// Set the update mode for incremental processing.
    pub fn update_mode(mut self, mode: UpdateMode) -> Self {
        self.base = self.base.update_mode(mode);
        self
    }
}

// ============================================================================
// Extended Online LOWESS Processor
// ============================================================================

/// Online LOWESS processor with parallel support.
pub struct ParallelOnlineLowess<T: Float> {
    processor: OnlineLowess<T>,
}

impl<T: Float + WLSSolver + Debug + Send + Sync + 'static> ParallelOnlineLowess<T> {
    /// Add a new point and return the smoothed value.
    pub fn add_point(&mut self, x: T, y: T) -> Result<Option<OnlineOutput<T>>, LowessError> {
        self.processor.add_point(x, y)
    }

    /// Add multiple points and return their smoothed values.
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

    /// Reset the processor, clearing all window data.
    pub fn reset(&mut self) {
        self.processor.reset();
    }
}

impl<T: Float + WLSSolver + Debug + Send + Sync + 'static> ParallelOnlineLowessBuilder<T> {
    /// Build the online processor.
    pub fn build(self) -> Result<ParallelOnlineLowess<T>, LowessError> {
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
                    if builder.parallel.unwrap_or(false) {
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
                    // For Online, we currently use fit_pass_gpu as a hook
                    // However, OnlineLowess in 'lowess' might not support full custom fit pass
                    // in the same way Batch does, but since fit_pass_gpu implements the
                    // FitPassFn signature, we can hook it if the base adapter supports it.
                    // OnlineLowessBuilder DOES have custom_fit_pass.
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
