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
//! * values must be finite.
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
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

// External dependencies
use core::fmt::Debug;
use core::mem::take;
use num_traits::Float;

// Internal dependencies
use crate::algorithms::regression::{WLSSolver, ZeroWeightFallback};
use crate::algorithms::robustness::RobustnessMethod;
use crate::engine::executor::{CVPassFn, FitPassFn, IntervalPassFn, SmoothPassFn};
use crate::engine::executor::{LowessConfig, LowessExecutor};
use crate::engine::output::LowessResult;
use crate::engine::validator::Validator;
use crate::evaluation::diagnostics::DiagnosticsState;
use crate::math::boundary::BoundaryPolicy;
use crate::math::kernel::WeightFunction;
use crate::math::scaling::ScalingMethod;
use crate::primitives::backend::Backend;
use crate::primitives::buffer::{StreamingBuffer, VecExt};
use crate::primitives::errors::LowessError;
use crate::primitives::sorting::sort_by_x;

// ============================================================================
// Merge Strategy
// ============================================================================

/// Strategy for merging overlapping regions between streaming chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MergeStrategy {
    /// Arithmetic mean of overlapping smoothed values: `(v1 + v2) / 2`.
    Average,

    /// Distance-based weights that favor values from the center of each chunk:
    /// v1 * (1 - alpha) + v2 * alpha where `alpha` is the relative position within the overlap.
    #[default]
    WeightedAverage,

    /// Use the value from the first chunk in processing order.
    TakeFirst,

    /// Use the value from the last chunk in processing order.
    TakeLast,
}

// ============================================================================
// Streaming LOWESS Builder
// ============================================================================

/// Builder for streaming LOWESS processor.
#[derive(Debug, Clone)]
pub struct StreamingLowessBuilder<T: Float> {
    /// Chunk size for processing
    pub chunk_size: usize,

    /// Overlap between chunks
    pub overlap: usize,

    /// Smoothing fraction (span)
    pub fraction: T,

    /// Number of robustness iterations
    pub iterations: usize,

    /// Convergence tolerance for early stopping (None = disabled)
    pub auto_convergence: Option<T>,

    /// Delta parameter for interpolation
    pub delta: T,

    /// Kernel weight function
    pub weight_function: WeightFunction,

    /// Boundary handling policy
    pub boundary_policy: BoundaryPolicy,

    /// Robustness method
    pub robustness_method: RobustnessMethod,

    /// Policy for handling zero-weight neighborhoods
    pub zero_weight_fallback: ZeroWeightFallback,

    /// Merging strategy for overlapping chunks
    pub merge_strategy: MergeStrategy,

    /// Whether to return residuals
    pub compute_residuals: bool,

    /// Scaling method for robust scale estimation (MAR/MAD)
    pub scaling_method: ScalingMethod,

    /// Whether to return diagnostics
    pub return_diagnostics: bool,

    /// Whether to return robustness weights
    pub return_robustness_weights: bool,

    /// Deferred error from adapter conversion
    pub deferred_error: Option<LowessError>,

    // ++++++++++++++++++++++++++++++++++++++
    // +               DEV                  +
    // ++++++++++++++++++++++++++++++++++++++
    /// Custom smooth pass function.
    #[doc(hidden)]
    pub custom_smooth_pass: Option<SmoothPassFn<T>>,

    /// Custom cross-validation pass function.
    #[doc(hidden)]
    pub custom_cv_pass: Option<CVPassFn<T>>,

    /// Custom interval estimation pass function.
    #[doc(hidden)]
    pub custom_interval_pass: Option<IntervalPassFn<T>>,

    /// Custom fit pass function.
    #[doc(hidden)]
    pub custom_fit_pass: Option<FitPassFn<T>>,

    /// Execution backend hint.
    #[doc(hidden)]
    pub backend: Option<Backend>,

    /// Parallel execution hint.
    #[doc(hidden)]
    pub parallel: Option<bool>,

    /// Tracks if any parameter was set multiple times (for validation)
    #[doc(hidden)]
    pub(crate) duplicate_param: Option<&'static str>,
}

impl<T: Float> Default for StreamingLowessBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> StreamingLowessBuilder<T> {
    /// Create a new streaming LOWESS builder with default parameters.
    fn new() -> Self {
        Self {
            chunk_size: 5000,
            overlap: 500,
            fraction: T::from(0.1).unwrap(),
            iterations: 2,
            delta: T::zero(),
            weight_function: WeightFunction::default(),
            boundary_policy: BoundaryPolicy::default(),
            robustness_method: RobustnessMethod::default(),
            zero_weight_fallback: ZeroWeightFallback::default(),
            merge_strategy: MergeStrategy::default(),
            compute_residuals: false,
            scaling_method: ScalingMethod::default(),
            return_diagnostics: false,
            return_robustness_weights: false,
            auto_convergence: None,
            deferred_error: None,
            custom_smooth_pass: None,
            custom_cv_pass: None,
            custom_interval_pass: None,
            custom_fit_pass: None,
            backend: None,
            parallel: None,
            duplicate_param: None,
        }
    }

    // ========================================================================
    // Shared Setters
    // ========================================================================

    /// Set the smoothing fraction (span).
    pub fn fraction(mut self, fraction: T) -> Self {
        self.fraction = fraction;
        self
    }

    /// Set the number of robustness iterations.
    pub fn iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Set the delta parameter for interpolation optimization.
    pub fn delta(mut self, delta: T) -> Self {
        self.delta = delta;
        self
    }

    /// Set kernel weight function.
    pub fn weight_function(mut self, weight_function: WeightFunction) -> Self {
        self.weight_function = weight_function;
        self
    }

    /// Set the robustness method for outlier handling.
    pub fn robustness_method(mut self, method: RobustnessMethod) -> Self {
        self.robustness_method = method;
        self
    }

    /// Set the zero-weight fallback policy.
    pub fn zero_weight_fallback(mut self, fallback: ZeroWeightFallback) -> Self {
        self.zero_weight_fallback = fallback;
        self
    }

    /// Set the boundary handling policy.
    pub fn boundary_policy(mut self, policy: BoundaryPolicy) -> Self {
        self.boundary_policy = policy;
        self
    }

    /// Enable auto-convergence for robustness iterations.
    pub fn auto_converge(mut self, tolerance: T) -> Self {
        self.auto_convergence = Some(tolerance);
        self
    }

    /// Enable returning residuals in the output.
    pub fn compute_residuals(mut self, enabled: bool) -> Self {
        self.compute_residuals = enabled;
        self
    }

    /// Enable returning robustness weights in the result.
    pub fn return_robustness_weights(mut self, enabled: bool) -> Self {
        self.return_robustness_weights = enabled;
        self
    }

    // ========================================================================
    // Streaming-Specific Setters
    // ========================================================================

    /// Set chunk size for processing.
    pub fn chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    /// Set overlap between chunks.
    pub fn overlap(mut self, overlap: usize) -> Self {
        self.overlap = overlap;
        self
    }

    /// Set the merge strategy for overlapping chunks.
    pub fn merge_strategy(mut self, strategy: MergeStrategy) -> Self {
        self.merge_strategy = strategy;
        self
    }

    /// Set whether to return diagnostics.
    pub fn return_diagnostics(mut self, return_diagnostics: bool) -> Self {
        self.return_diagnostics = return_diagnostics;
        self
    }

    // ++++++++++++++++++++++++++++++++++++++
    // +               DEV                  +
    // ++++++++++++++++++++++++++++++++++++++

    /// Set a custom smooth pass function.
    #[doc(hidden)]
    pub fn custom_smooth_pass(mut self, pass: SmoothPassFn<T>) -> Self {
        self.custom_smooth_pass = Some(pass);
        self
    }

    /// Set a custom cross-validation pass function.
    #[doc(hidden)]
    pub fn custom_cv_pass(mut self, pass: CVPassFn<T>) -> Self {
        self.custom_cv_pass = Some(pass);
        self
    }

    /// Set a custom interval estimation pass function.
    #[doc(hidden)]
    pub fn custom_interval_pass(mut self, pass: IntervalPassFn<T>) -> Self {
        self.custom_interval_pass = Some(pass);
        self
    }

    /// Set the execution backend hint.
    #[doc(hidden)]
    pub fn backend(mut self, backend: Backend) -> Self {
        self.backend = Some(backend);
        self
    }

    /// Set parallel execution hint.
    #[doc(hidden)]
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.parallel = Some(parallel);
        self
    }

    // ========================================================================
    // Build Method
    // ========================================================================

    /// Build the streaming processor.
    pub fn build(self) -> Result<StreamingLowess<T>, LowessError> {
        if let Some(err) = self.deferred_error {
            return Err(err);
        }

        // Check for duplicate parameter configuration
        Validator::validate_no_duplicates(self.duplicate_param)?;

        // Validate fraction
        Validator::validate_fraction(self.fraction)?;

        // Validate iterations
        Validator::validate_iterations(self.iterations)?;

        // Validate delta
        Validator::validate_delta(self.delta)?;

        // Validate chunk size
        Validator::validate_chunk_size(self.chunk_size, 10)?;

        // Validate overlap
        Validator::validate_overlap(self.overlap, self.chunk_size)?;

        let has_diag = self.return_diagnostics;
        let overlap = self.overlap;
        let chunk_size = self.chunk_size;
        Ok(StreamingLowess {
            config: self,
            buffer: StreamingBuffer::with_capacity(overlap, chunk_size),
            diagnostics_state: if has_diag {
                Some(DiagnosticsState::new())
            } else {
                None
            },
        })
    }
}

// ============================================================================
// Streaming LOWESS Processor
// ============================================================================

/// Streaming LOWESS processor for large datasets.
pub struct StreamingLowess<T: Float> {
    config: StreamingLowessBuilder<T>,
    /// Pre-allocated overlap buffers
    buffer: StreamingBuffer<T>,
    diagnostics_state: Option<DiagnosticsState<T>>,
}

impl<T: Float + WLSSolver + Debug + Send + Sync + 'static> StreamingLowess<T> {
    /// Process a chunk of data.
    pub fn process_chunk(&mut self, x: &[T], y: &[T]) -> Result<LowessResult<T>, LowessError> {
        // Validate inputs using standard validator
        Validator::validate_inputs(x, y)?;

        // Sort chunk by x
        let sorted = sort_by_x(x, y);

        // Configure LOWESS for this chunk
        // Combine with overlap from previous chunk
        let prev_overlap_len: usize = self.buffer.overlap_smoothed.len();
        let (combined_x, combined_y) = if self.buffer.overlap_x.is_empty() {
            // No overlap: move sorted data directly (no clone needed)
            (sorted.x, sorted.y)
        } else {
            let mut cx: Vec<T> = take(&mut *self.buffer.overlap_x);
            cx.extend_from_slice(&sorted.x);
            let mut cy: Vec<T> = take(&mut *self.buffer.overlap_y);
            cy.extend_from_slice(&sorted.y);
            (cx, cy)
        };

        let zero_flag = self.config.zero_weight_fallback.to_u8();

        let config = LowessConfig {
            fraction: Some(self.config.fraction),
            iterations: self.config.iterations,
            delta: self.config.delta,
            weight_function: self.config.weight_function,
            zero_weight_fallback: zero_flag,
            robustness_method: self.config.robustness_method,
            boundary_policy: self.config.boundary_policy,
            scaling_method: self.config.scaling_method,
            cv_fractions: None,
            cv_kind: None,
            auto_convergence: self.config.auto_convergence,
            return_variance: None,
            cv_seed: None,
            // ++++++++++++++++++++++++++++++++++++++
            // +               DEV                  +
            // ++++++++++++++++++++++++++++++++++++++
            custom_smooth_pass: self.config.custom_smooth_pass,
            custom_cv_pass: self.config.custom_cv_pass,
            custom_interval_pass: self.config.custom_interval_pass,
            custom_fit_pass: self.config.custom_fit_pass,
            parallel: self.config.parallel.unwrap_or(false),
            backend: self.config.backend,
        };
        // Execute LOWESS on combined data
        // Use pre-allocated work_buffer to minimize allocations
        let result = LowessExecutor::from_config(&config).run(
            &combined_x,
            &combined_y,
            Some(&mut self.buffer.work_buffer),
        );
        let smoothed = result.smoothed;
        let robustness_weights = result.robustness_weights;
        let iterations = result.iterations.unwrap_or(0);
        // Determine how much to return vs buffer
        let combined_len = combined_x.len();
        let overlap_start = combined_len.saturating_sub(self.config.overlap);
        let return_start = prev_overlap_len;

        // Build output: merged overlap (if any) + new data
        let mut y_smooth_out: Vec<T> = Vec::new();
        if prev_overlap_len > 0 {
            // Merge the overlap region
            // Use as_vec() and index instead of take() to preserve buffered data if possible,
            // but we need to merge into a results vector anyway.
            let prev_smooth = self.buffer.overlap_smoothed.as_vec();
            for (i, (&prev_val, &curr_val)) in prev_smooth
                .iter()
                .zip(smoothed.iter())
                .take(prev_overlap_len)
                .enumerate()
            {
                let merged = match self.config.merge_strategy {
                    MergeStrategy::Average => (prev_val + curr_val) / T::from(2.0).unwrap(),
                    MergeStrategy::WeightedAverage => {
                        let weight = T::from(i as f64 / prev_overlap_len as f64).unwrap();
                        prev_val * (T::one() - weight) + curr_val * weight
                    }
                    MergeStrategy::TakeFirst => prev_val,
                    MergeStrategy::TakeLast => curr_val,
                };
                y_smooth_out.push(merged);
            }
        }

        // Merge robustness weights if requested
        let mut rob_weights_out: Option<Vec<T>> = if self.config.return_robustness_weights {
            Some(Vec::new())
        } else {
            None
        };

        if let Some(ref mut rw_out) = rob_weights_out {
            if prev_overlap_len > 0 {
                let prev_rw = self.buffer.overlap_robustness_weights.as_vec();
                for (i, (&prev_val, &curr_val)) in prev_rw
                    .iter()
                    .zip(robustness_weights.iter())
                    .take(prev_overlap_len)
                    .enumerate()
                {
                    let merged = match self.config.merge_strategy {
                        MergeStrategy::Average => (prev_val + curr_val) / T::from(2.0).unwrap(),
                        MergeStrategy::WeightedAverage => {
                            let weight = T::from(i as f64 / prev_overlap_len as f64).unwrap();
                            prev_val * (T::one() - weight) + curr_val * weight
                        }
                        MergeStrategy::TakeFirst => prev_val,
                        MergeStrategy::TakeLast => curr_val,
                    };
                    rw_out.push(merged);
                }
            }
        }

        // Add non-overlap portion
        if return_start < overlap_start {
            y_smooth_out.extend_from_slice(&smoothed[return_start..overlap_start]);
            if let Some(ref mut rw_out) = rob_weights_out {
                rw_out.extend_from_slice(&robustness_weights[return_start..overlap_start]);
            }
        }
        // Calculate residuals for output
        let residuals_out = if self.config.compute_residuals {
            let y_slice = &combined_y[return_start..return_start + y_smooth_out.len()];
            Some(
                y_slice
                    .iter()
                    .zip(y_smooth_out.iter())
                    .map(|(y, s)| *y - *s)
                    .collect(),
            )
        } else {
            None
        };

        // Buffer overlap for next chunk
        if overlap_start < combined_len {
            VecExt::assign_slice(
                self.buffer.overlap_x.as_vec_mut(),
                &combined_x[overlap_start..],
            );
            VecExt::assign_slice(
                self.buffer.overlap_y.as_vec_mut(),
                &combined_y[overlap_start..],
            );
            VecExt::assign_slice(
                self.buffer.overlap_smoothed.as_vec_mut(),
                &smoothed[overlap_start..],
            );
            if self.config.return_robustness_weights {
                VecExt::assign_slice(
                    self.buffer.overlap_robustness_weights.as_vec_mut(),
                    &robustness_weights[overlap_start..],
                );
            }
        } else {
            self.buffer.overlap_x.clear();
            self.buffer.overlap_y.clear();
            self.buffer.overlap_smoothed.clear();
            self.buffer.overlap_robustness_weights.clear();
        }

        // Note: We return results in sorted order (by x) for streaming chunks.
        // Unsorting partial results is ambiguous since we only return a subset of the chunk.
        // The full batch adapter handles global unsorting when processing complete datasets.
        let x_out = combined_x[return_start..return_start + y_smooth_out.len()].to_vec();

        // Update diagnostics cumulatively
        let diagnostics = if let Some(ref mut state) = self.diagnostics_state {
            let y_emitted = &combined_y[return_start..return_start + y_smooth_out.len()];
            state.update(y_emitted, &y_smooth_out);
            Some(state.finalize())
        } else {
            None
        };

        Ok(LowessResult {
            x: x_out,
            y: y_smooth_out,
            standard_errors: None,
            confidence_lower: None,
            confidence_upper: None,
            prediction_lower: None,
            prediction_upper: None,
            residuals: residuals_out,
            robustness_weights: rob_weights_out,
            diagnostics,
            iterations_used: Some(iterations),
            fraction_used: self.config.fraction,
            cv_scores: None,
        })
    }

    /// Finalize processing and get any remaining buffered data.
    pub fn finalize(&mut self) -> Result<LowessResult<T>, LowessError> {
        if self.buffer.overlap_x.is_empty() {
            return Ok(LowessResult {
                x: Vec::new(),
                y: Vec::new(),
                standard_errors: None,
                confidence_lower: None,
                confidence_upper: None,
                prediction_lower: None,
                prediction_upper: None,
                residuals: None,
                robustness_weights: None,
                diagnostics: None,
                iterations_used: None,
                fraction_used: self.config.fraction,
                cv_scores: None,
            });
        }

        // Return buffered overlap data
        let residuals = if self.config.compute_residuals {
            let mut res = Vec::with_capacity(self.buffer.overlap_x.len());
            for (i, &smoothed) in self.buffer.overlap_smoothed.iter().enumerate() {
                res.push(self.buffer.overlap_y[i] - smoothed);
            }
            Some(res)
        } else {
            None
        };

        let robustness_weights = if self.config.return_robustness_weights {
            Some(take(&mut *self.buffer.overlap_robustness_weights))
        } else {
            None
        };

        // Update diagnostics for the final overlap
        let diagnostics = if let Some(ref mut state) = self.diagnostics_state {
            state.update(&self.buffer.overlap_y, &self.buffer.overlap_smoothed);
            Some(state.finalize())
        } else {
            None
        };

        let result = LowessResult {
            x: self.buffer.overlap_x.as_vec().clone(),
            y: self.buffer.overlap_smoothed.as_vec().clone(),
            standard_errors: None,
            confidence_lower: None,
            confidence_upper: None,
            prediction_lower: None,
            prediction_upper: None,
            residuals,
            robustness_weights,
            diagnostics,
            iterations_used: None,
            fraction_used: self.config.fraction,
            cv_scores: None,
        };

        // Clear buffers
        self.buffer.clear();

        Ok(result)
    }

    /// Reset the processor state.
    pub fn reset(&mut self) {
        self.buffer.clear();
    }
}
