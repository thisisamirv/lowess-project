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
#[cfg(not(feature = "std"))]
use alloc::collections::VecDeque;
#[cfg(feature = "std")]
use std::collections::VecDeque;

// External dependencies
use core::fmt::Debug;
use num_traits::Float;

// Internal dependencies
use crate::algorithms::regression::{WLSSolver, ZeroWeightFallback};
use crate::algorithms::robustness::RobustnessMethod;
use crate::engine::executor::{CVPassFn, FitPassFn, IntervalPassFn, SmoothPassFn};
use crate::engine::executor::{LowessConfig, LowessExecutor};
use crate::engine::validator::Validator;
use crate::math::boundary::BoundaryPolicy;
use crate::math::kernel::WeightFunction;
use crate::math::scaling::ScalingMethod;
use crate::primitives::backend::Backend;
use crate::primitives::buffer::{OnlineBuffer, VecExt};
use crate::primitives::errors::LowessError;

// ============================================================================
// Update Mode
// ============================================================================

/// Update mode for online LOWESS processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UpdateMode {
    /// Recompute all points in the window from scratch.
    Full,

    /// Optimized incremental update.
    #[default]
    Incremental,
}

// ============================================================================
// Online LOWESS Builder
// ============================================================================

/// Builder for online LOWESS processor.
#[derive(Debug, Clone)]
pub struct OnlineLowessBuilder<T: Float> {
    /// Window capacity (maximum number of points to retain)
    pub window_capacity: usize,

    /// Minimum points before smoothing starts
    pub min_points: usize,

    /// Smoothing fraction (span)
    pub fraction: T,

    /// Delta parameter for interpolation optimization
    pub delta: T,

    /// Number of robustness iterations
    pub iterations: usize,

    /// Convergence tolerance for early stopping (None = disabled)
    pub auto_convergence: Option<T>,

    /// Kernel weight function
    pub weight_function: WeightFunction,

    /// Update mode for incremental processing
    pub update_mode: UpdateMode,

    /// Robustness method
    pub robustness_method: RobustnessMethod,

    /// Policy for handling zero-weight neighborhoods
    pub zero_weight_fallback: ZeroWeightFallback,

    /// Policy for handling data boundaries
    pub boundary_policy: BoundaryPolicy,

    /// Scaling method for robust scale estimation (MAR/MAD)
    pub scaling_method: ScalingMethod,

    /// Whether to return residuals
    pub compute_residuals: bool,

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

impl<T: Float> Default for OnlineLowessBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> OnlineLowessBuilder<T> {
    /// Create a new online LOWESS builder with default parameters.
    fn new() -> Self {
        Self {
            window_capacity: 1000,
            min_points: 3,
            fraction: T::from(0.2).unwrap(),
            delta: T::zero(),
            iterations: 1,
            weight_function: WeightFunction::default(),
            update_mode: UpdateMode::default(),
            robustness_method: RobustnessMethod::default(),
            zero_weight_fallback: ZeroWeightFallback::default(),
            boundary_policy: BoundaryPolicy::default(),
            scaling_method: ScalingMethod::default(),
            compute_residuals: false,
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

    /// Set the delta parameter for interpolation-based optimization.
    pub fn delta(mut self, delta: T) -> Self {
        self.delta = delta;
        self
    }

    /// Set the kernel weight function.
    pub fn weight_function(mut self, wf: WeightFunction) -> Self {
        self.weight_function = wf;
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
    // Online-Specific Setters
    // ========================================================================

    /// Set window capacity (maximum number of points to retain).
    pub fn window_capacity(mut self, capacity: usize) -> Self {
        self.window_capacity = capacity;
        self
    }

    /// Set minimum points before smoothing starts.
    pub fn min_points(mut self, min: usize) -> Self {
        self.min_points = min;
        self
    }

    /// Set the update mode for incremental processing.
    pub fn update_mode(mut self, mode: UpdateMode) -> Self {
        self.update_mode = mode;
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

    /// Build the online processor.
    pub fn build(self) -> Result<OnlineLowess<T>, LowessError> {
        if let Some(err) = self.deferred_error {
            return Err(err);
        }

        // Check for duplicate parameter configuration
        Validator::validate_no_duplicates(self.duplicate_param)?;

        // Validate fraction
        Validator::validate_fraction(self.fraction)?;

        // Validate iterations
        Validator::validate_iterations(self.iterations)?;

        // Validate configuration early
        Validator::validate_window_capacity(self.window_capacity, 3)?;
        Validator::validate_min_points(self.min_points, self.window_capacity)?;

        let capacity = self.window_capacity;
        Ok(OnlineLowess {
            config: self,
            window_x: VecDeque::with_capacity(capacity),
            window_y: VecDeque::with_capacity(capacity),
            buffer: OnlineBuffer::with_capacity(capacity),
        })
    }
}

// ============================================================================
// Online LOWESS Output
// ============================================================================

/// Result of a single online update.
#[derive(Debug, Clone, PartialEq)]
pub struct OnlineOutput<T> {
    /// Smoothed value for the latest point
    pub smoothed: T,

    /// Standard error (if computed)
    pub std_error: Option<T>,

    /// Residual (y - smoothed)
    pub residual: Option<T>,

    /// Robustness weight for the latest point (if computed)
    pub robustness_weight: Option<T>,
}

// ============================================================================
// Online LOWESS Processor
// ============================================================================

/// Online LOWESS processor for streaming data.
pub struct OnlineLowess<T: Float> {
    config: OnlineLowessBuilder<T>,
    window_x: VecDeque<T>,
    window_y: VecDeque<T>,
    /// Pre-allocated scratch buffers for smoothing
    buffer: OnlineBuffer<T>,
}

impl<T: Float + WLSSolver + Debug + Send + Sync + 'static> OnlineLowess<T> {
    /// Add a new point and get its smoothed value.
    pub fn add_point(&mut self, x: T, y: T) -> Result<Option<OnlineOutput<T>>, LowessError> {
        // Validate new point
        Validator::validate_scalar(x, "x")?;
        Validator::validate_scalar(y, "y")?;

        // Add to window
        self.window_x.push_back(x);
        self.window_y.push_back(y);

        // Evict oldest if over capacity
        if self.window_x.len() > self.config.window_capacity {
            self.window_x.pop_front();
            self.window_y.pop_front();
        }

        // Check if we have enough points
        if self.window_x.len() < self.config.min_points {
            return Ok(None);
        }

        // Convert window to vectors for smoothing using scratch buffers
        self.buffer.clear(); // Clear buffers
        self.buffer.scratch_x.extend(self.window_x.iter().copied());
        self.buffer.scratch_y.extend(self.window_y.iter().copied());

        let x_vec = &*self.buffer.scratch_x;
        let y_vec = &*self.buffer.scratch_y;

        // Special case: exactly two points, use exact linear fit
        if x_vec.len() == 2 {
            let x0 = x_vec[0];
            let x1 = x_vec[1];
            let y0 = y_vec[0];
            let y1 = y_vec[1];

            let smoothed = if x1 != x0 {
                let slope = (y1 - y0) / (x1 - x0);
                y0 + slope * (x - x0)
            } else {
                // Identical x: use mean for stability
                (y0 + y1) / T::from(2.0).unwrap()
            };

            let residual = y - smoothed;

            return Ok(Some(OnlineOutput {
                smoothed,
                std_error: None,
                residual: Some(residual),
                robustness_weight: Some(T::one()),
            }));
        }

        // Smooth using LOWESS for windows of size >= 3
        let zero_flag = self.config.zero_weight_fallback.to_u8();

        // Choose update strategy based on configuration
        let (smoothed, std_err, rob_weight) = match self.config.update_mode {
            UpdateMode::Incremental => {
                // Incremental mode: fit only the latest point
                let n = x_vec.len();
                let window_size = (self.config.fraction * T::from(n).unwrap())
                    .ceil()
                    .to_usize()
                    .unwrap_or(n)
                    .max(2)
                    .min(n);

                // Use pre-allocated scratch buffers
                VecExt::assign(self.buffer.weights.as_vec_mut(), n, T::zero());
                VecExt::assign(self.buffer.robustness_weights.as_vec_mut(), n, T::one());

                let (smoothed_val, _) = LowessExecutor::fit_single_point(
                    x_vec,
                    y_vec,
                    n - 1, // Latest point
                    window_size,
                    false, // No robustness for single point
                    &self.buffer.robustness_weights,
                    &mut self.buffer.weights,
                    self.config.weight_function,
                    self.config.zero_weight_fallback,
                );

                (smoothed_val, None, Some(T::one()))
            }
            UpdateMode::Full => {
                // Full mode: re-smooth entire window
                let config = LowessConfig {
                    fraction: Some(self.config.fraction),
                    iterations: self.config.iterations,
                    delta: self.config.delta,
                    weight_function: self.config.weight_function,
                    robustness_method: self.config.robustness_method,
                    zero_weight_fallback: zero_flag,
                    boundary_policy: self.config.boundary_policy,
                    scaling_method: self.config.scaling_method,
                    auto_convergence: self.config.auto_convergence,
                    cv_fractions: None,
                    cv_kind: None,
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

                let result = LowessExecutor::run_with_config(x_vec, y_vec, config.clone());
                let smoothed_vec = result.smoothed;
                let se_vec = result.std_errors;

                let smoothed_val = smoothed_vec.last().copied().ok_or_else(|| {
                    LowessError::InvalidNumericValue("No smoothed output produced".into())
                })?;
                let std_err = se_vec.as_ref().and_then(|v| v.last().copied());
                let rob_weight = if self.config.return_robustness_weights {
                    result.robustness_weights.last().copied()
                } else {
                    None
                };

                (smoothed_val, std_err, rob_weight)
            }
        };

        let residual = y - smoothed;

        Ok(Some(OnlineOutput {
            smoothed,
            std_error: std_err,
            residual: Some(residual),
            robustness_weight: rob_weight,
        }))
    }

    /// Get the current window size.
    pub fn window_size(&self) -> usize {
        self.window_x.len()
    }

    /// Clear the window.
    pub fn reset(&mut self) {
        self.window_x.clear();
        self.window_y.clear();
        self.buffer.clear();
    }
}
