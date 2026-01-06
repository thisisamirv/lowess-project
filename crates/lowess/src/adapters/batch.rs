//! Batch adapter for standard LOWESS smoothing.
//!
//! ## Purpose
//!
//! This module provides the batch execution adapter for LOWESS smoothing.
//! It handles complete datasets in memory with sequential processing, making
//! it suitable for small to medium-sized datasets.
//!
//! ## Design notes
//!
//! * **Processing**: Processes entire dataset in a single pass.
//! * **Sorting**: Automatically sorts data by x-values and unsorts results.
//! * **Delegation**: Delegates computation to the execution engine.
//! * **Generics**: Generic over `Float` types.
//!
//! ## Key concepts
//!
//! * **Batch Processing**: Validates, sorts, filters, executes, and unsorts.
//! * **Builder Pattern**: Fluent API for configuration with sensible defaults.
//! * **Automatic Sorting**: Ensures x-values are sorted for the algorithm.
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
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

// External dependencies
use core::fmt::Debug;
use num_traits::Float;

// Internal dependencies
use crate::algorithms::interpolation::calculate_delta;
use crate::algorithms::regression::{WLSSolver, ZeroWeightFallback};
use crate::algorithms::robustness::RobustnessMethod;
use crate::engine::executor::{CVPassFn, FitPassFn, IntervalPassFn, SmoothPassFn};
use crate::engine::executor::{LowessConfig, LowessExecutor};
use crate::engine::output::LowessResult;
use crate::engine::validator::Validator;
use crate::evaluation::cv::CVKind;
use crate::evaluation::diagnostics::Diagnostics;
use crate::evaluation::intervals::IntervalMethod;
use crate::math::boundary::BoundaryPolicy;
use crate::math::kernel::WeightFunction;
use crate::math::scaling::ScalingMethod;
use crate::primitives::backend::Backend;
use crate::primitives::errors::LowessError;
use crate::primitives::sorting::{sort_by_x, unsort};

// ============================================================================
// Batch LOWESS Builder
// ============================================================================

/// Builder for batch LOWESS processor.
#[derive(Debug, Clone)]
pub struct BatchLowessBuilder<T: Float> {
    /// Smoothing fraction (span)
    pub fraction: T,

    /// Number of robustness iterations
    pub iterations: usize,

    /// Optimization delta
    pub delta: Option<T>,

    /// Kernel weight function
    pub weight_function: WeightFunction,

    /// Robustness method
    pub robustness_method: RobustnessMethod,

    /// Confidence/Prediction interval configuration
    pub interval_type: Option<IntervalMethod<T>>,

    /// Fractions for cross-validation
    pub cv_fractions: Option<Vec<T>>,

    /// Cross-validation method kind
    pub cv_kind: Option<CVKind>,

    /// Cross-validation seed
    pub cv_seed: Option<u64>,

    /// Deferred error from adapter conversion
    pub deferred_error: Option<LowessError>,

    /// Tolerance for auto-convergence
    pub auto_convergence: Option<T>,

    /// Whether to compute diagnostic statistics
    pub return_diagnostics: bool,

    /// Whether to return residuals
    pub compute_residuals: bool,

    /// Whether to return robustness weights
    pub return_robustness_weights: bool,

    /// Policy for handling zero-weight neighborhoods
    pub zero_weight_fallback: ZeroWeightFallback,

    /// Policy for handling data boundaries
    pub boundary_policy: BoundaryPolicy,

    /// Scaling method for robust scale estimation (MAR/MAD)
    pub scaling_method: ScalingMethod,

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

impl<T: Float> Default for BatchLowessBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> BatchLowessBuilder<T> {
    /// Create a new batch LOWESS builder with default parameters.
    fn new() -> Self {
        Self {
            fraction: T::from(0.67).unwrap(),
            iterations: 3,
            delta: None,
            weight_function: WeightFunction::default(),
            robustness_method: RobustnessMethod::default(),
            interval_type: None,
            cv_fractions: None,
            cv_kind: None,
            cv_seed: None,
            deferred_error: None,
            auto_convergence: None,
            return_diagnostics: false,
            compute_residuals: false,
            return_robustness_weights: false,
            zero_weight_fallback: ZeroWeightFallback::default(),
            boundary_policy: BoundaryPolicy::default(),
            scaling_method: ScalingMethod::default(),
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
        self.delta = Some(delta);
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
    // Batch-Specific Setters
    // ========================================================================

    /// Enable returning diagnostics in the result.
    pub fn return_diagnostics(mut self, enabled: bool) -> Self {
        self.return_diagnostics = enabled;
        self
    }

    /// Enable confidence intervals at the specified level.
    pub fn confidence_intervals(mut self, level: T) -> Self {
        self.interval_type = Some(IntervalMethod::confidence(level));
        self
    }

    /// Enable prediction intervals at the specified level.
    pub fn prediction_intervals(mut self, level: T) -> Self {
        self.interval_type = Some(IntervalMethod::prediction(level));
        self
    }

    /// Enable cross-validation with the specified fractions.
    pub fn cross_validate(mut self, fractions: Vec<T>) -> Self {
        self.cv_fractions = Some(fractions);
        self
    }

    /// Set the cross-validation method.
    pub fn cv_kind(mut self, kind: CVKind) -> Self {
        self.cv_kind = Some(kind);
        self
    }

    // ++++++++++++++++++++++++++++++++++++++
    // +               DEV                  +
    // ++++++++++++++++++++++++++++++++++++++

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

    // ========================================================================
    // Build Method
    // ========================================================================

    /// Build the batch processor.
    pub fn build(self) -> Result<BatchLowess<T>, LowessError> {
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
        if let Some(delta) = self.delta {
            Validator::validate_delta(delta)?;
        }

        // Validate interval type
        if let Some(ref method) = self.interval_type {
            Validator::validate_interval_level(method.level)?;
        }

        // Validate CV fractions and method
        if let Some(ref fracs) = self.cv_fractions {
            Validator::validate_cv_fractions(fracs)?;
        }
        if let Some(CVKind::KFold(k)) = self.cv_kind {
            Validator::validate_kfold(k)?;
        }

        // Validate auto convergence tolerance
        if let Some(tol) = self.auto_convergence {
            Validator::validate_tolerance(tol)?;
        }

        Ok(BatchLowess { config: self })
    }
}

// ============================================================================
// Batch LOWESS Processor
// ============================================================================

/// Batch LOWESS processor.
pub struct BatchLowess<T: Float> {
    config: BatchLowessBuilder<T>,
}

impl<T: Float + WLSSolver + Debug + Send + Sync + 'static> BatchLowess<T> {
    /// Perform LOWESS smoothing on the provided data.
    pub fn fit(self, x: &[T], y: &[T]) -> Result<LowessResult<T>, LowessError> {
        Validator::validate_inputs(x, y)?;

        // Sort data by x using sorting module
        let sorted = sort_by_x(x, y);
        let delta = calculate_delta(self.config.delta, &sorted.x)?;

        let zw_flag: u8 = self.config.zero_weight_fallback.to_u8();

        // Configure batch execution
        let config = LowessConfig {
            fraction: Some(self.config.fraction),
            iterations: self.config.iterations,
            delta,
            weight_function: self.config.weight_function,
            zero_weight_fallback: zw_flag,
            robustness_method: self.config.robustness_method,
            cv_fractions: self.config.cv_fractions,
            cv_kind: self.config.cv_kind,
            auto_convergence: self.config.auto_convergence,
            return_variance: self.config.interval_type,
            boundary_policy: self.config.boundary_policy,
            scaling_method: self.config.scaling_method,
            cv_seed: self.config.cv_seed,
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

        // Execute unified LOWESS
        let result = LowessExecutor::run_with_config(&sorted.x, &sorted.y, config);

        let y_smooth = result.smoothed;
        let std_errors = result.std_errors;
        let iterations_used = result.iterations;
        let fraction_used = result.used_fraction;
        let cv_scores = result.cv_scores;

        // Calculate residuals
        let residuals: Vec<T> = sorted
            .y
            .iter()
            .zip(y_smooth.iter())
            .map(|(&orig, &smoothed_val)| orig - smoothed_val)
            .collect();

        // Get robustness weights from executor result (final iteration weights)
        let rob_weights = if self.config.return_robustness_weights {
            result.robustness_weights
        } else {
            Vec::new()
        };

        // Compute diagnostic statistics if requested
        let diagnostics = if self.config.return_diagnostics {
            Some(Diagnostics::compute(
                &sorted.y,
                &y_smooth,
                &residuals,
                std_errors.as_deref(),
            ))
        } else {
            None
        };

        // Compute intervals
        let (conf_lower, conf_upper, pred_lower, pred_upper) =
            match (&self.config.interval_type, &std_errors) {
                (Some(method), Some(se)) => {
                    let (cl, cu, pl, pu) = method.compute_intervals(&y_smooth, se, &residuals)?;
                    (cl, cu, pl, pu)
                }
                _ => (None, None, None, None),
            };

        // Unsort results using sorting module
        let indices = &sorted.indices;
        let y_smooth_out = unsort(&y_smooth, indices);
        let std_errors_out = std_errors.as_ref().map(|se| unsort(se, indices));
        let residuals_out = if self.config.compute_residuals {
            Some(unsort(&residuals, indices))
        } else {
            None
        };
        let rob_weights_out = if self.config.return_robustness_weights {
            Some(unsort(&rob_weights, indices))
        } else {
            None
        };
        let cl_out = conf_lower.as_ref().map(|v| unsort(v, indices));
        let cu_out = conf_upper.as_ref().map(|v| unsort(v, indices));
        let pl_out = pred_lower.as_ref().map(|v| unsort(v, indices));
        let pu_out = pred_upper.as_ref().map(|v| unsort(v, indices));

        Ok(LowessResult {
            x: x.to_vec(),
            y: y_smooth_out,
            standard_errors: std_errors_out,
            confidence_lower: cl_out,
            confidence_upper: cu_out,
            prediction_lower: pl_out,
            prediction_upper: pu_out,
            residuals: residuals_out,
            robustness_weights: rob_weights_out,
            fraction_used,
            iterations_used,
            cv_scores,
            diagnostics,
        })
    }
}
