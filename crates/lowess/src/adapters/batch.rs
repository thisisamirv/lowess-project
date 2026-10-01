//! Batch adapter for standard LOWESS smoothing.
//!
//! This module provides the batch execution adapter for LOWESS smoothing.
//! It handles complete datasets in memory with sequential processing, making
//! it suitable for small to medium-sized datasets.
// ## srrstats Compliance
//
// @srrstats {G1.3} Builder pattern for fluent, validated configuration.
// @srrstats {G2.0} Comprehensive input validation via Validator before processing.

// External dependencies
#[cfg(not(feature = "std"))]
use alloc::sync::Arc;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::fmt::Debug;
use num_traits::Float;
#[cfg(feature = "std")]
use std::sync::Arc;
#[cfg(feature = "std")]
use std::vec::Vec;

// Internal dependencies
use crate::adapters::defaults::*;
use crate::algorithms::defaults::*;
use crate::algorithms::interpolation::calculate_delta;
use crate::algorithms::regression::{WLSSolver, ZeroWeightFallback};
use crate::algorithms::robustness::RobustnessMethod;
use crate::engine::executor::LowessResult;
use crate::engine::executor::{
    BootstrapPassFn, CVPassFn, DerivativePassFn, FitPassFn, IntervalPassFn, SmoothPassFn,
};
use crate::engine::executor::{BootstrapPredictState, PredictPassFn};
use crate::engine::executor::{LowessConfig, LowessExecutor};
use crate::engine::validator::Validator;
use crate::evaluation::cv::CVKind;
use crate::evaluation::defaults::{DEFAULT_CV_SEED, DEFAULT_FRACTION};
use crate::evaluation::diagnostics::Diagnostics;
use crate::evaluation::intervals::{
    BootstrapConfig, BootstrapOutput, IntervalMethod, MIN_BOOTSTRAP_SAMPLES,
};
use crate::math::boundary::BoundaryPolicy;
use crate::math::defaults::*;
use crate::math::kernel::WeightFunction;
use crate::math::scaling::ScalingMethod;
use crate::primitives::backend::Backend;
use crate::primitives::errors::LowessError;
use crate::primitives::policies::MissingPolicy;
use crate::primitives::sorting::{SortedData, sort_by_x, unsort};

pub type BootstrapComputeFn<T> = fn(
    &[T],
    &[T],
    &[T],
    &IntervalMethod<T>,
    BootstrapConfig,
    &LowessConfig<T>,
) -> Result<BootstrapOutput<T>, LowessError>;

// Builder for batch LOWESS processor.
#[derive(Debug, Clone)]
pub struct BatchLowessBuilder<T: Float> {
    // Smoothing fraction (span)
    pub fraction: T,

    // Number of robustness iterations
    pub iterations: usize,

    // Optimization delta
    pub delta: Option<T>,

    // Kernel weight function
    pub weight_function: WeightFunction,

    // Robustness method
    pub robustness_method: RobustnessMethod,

    // Confidence/Prediction interval configuration
    pub interval_type: Option<IntervalMethod<T>>,

    // Use residual-bootstrap SEs/intervals instead of the analytic ones.
    pub bootstrap: Option<BootstrapConfig>,

    // Fractions for cross-validation
    pub cv_fractions: Option<Vec<T>>,

    // Cross-validation method kind
    pub cv_kind: Option<CVKind>,

    // Cross-validation seed
    pub cv_seed: Option<u64>,

    // Deferred error from adapter conversion
    pub deferred_error: Option<LowessError>,

    // Tolerance for auto-convergence
    pub auto_converge: Option<T>,

    // Whether to compute diagnostic statistics
    pub return_diagnostics: bool,

    // Whether to return residuals
    pub compute_residuals: bool,

    // Whether to return robustness weights
    pub return_robustness_weights: bool,

    // Whether to return per-point local fit derivative (slope)
    pub return_derivative: bool,

    // Whether to return results sorted ascending by x instead of in original input order
    pub return_sorted: bool,

    // Policy for handling non-finite (NaN/Inf) values in input data
    pub missing: MissingPolicy,

    // Policy for handling zero-weight neighborhoods
    pub zero_weight_fallback: ZeroWeightFallback,

    // Policy for handling data boundaries
    pub boundary_policy: BoundaryPolicy,

    // Scaling method for robust scale estimation (MAR/MAD)
    pub scaling_method: ScalingMethod,

    // ++++++++++++++++++++++++++++++++++++++
    // +               DEV                  +
    // ++++++++++++++++++++++++++++++++++++++
    // Custom smooth pass function.
    pub custom_smooth_pass: Option<SmoothPassFn<T>>,

    // Custom cross-validation pass function.
    pub custom_cv_pass: Option<CVPassFn<T>>,

    // Custom interval estimation pass function.
    pub custom_interval_pass: Option<IntervalPassFn<T>>,

    // Custom derivative (local fit slope) estimation pass function.
    pub custom_derivative_pass: Option<DerivativePassFn<T>>,

    // Custom fit pass function.
    pub custom_fit_pass: Option<FitPassFn<T>>,

    // Execution backend hint.
    pub backend: Option<Backend>,

    // Parallel execution hint.
    pub parallel: Option<bool>,

    // Whether to delegate boundary handling (padding)
    pub delegate_boundary_handling: bool,

    // Tracks if any parameter was set multiple times (for validation)
    pub(crate) duplicate_param: Option<&'static str>,

    // Per-observation case weights. When provided, multiplies each local kernel weight:
    // `w_ij = custom_weights[j] * K(d_ij / h) * robustness_j`.
    pub custom_weights: Option<Vec<T>>,

    // Whether to retain fitted-model state for later `Predict::call()` calls.
    pub retain_model: bool,

    // Custom (e.g. parallel) predict pass function.
    pub custom_predict_pass: Option<PredictPassFn<T>>,

    // Custom (e.g. parallel) bootstrap refit pass function.
    pub custom_bootstrap_pass: Option<BootstrapPassFn<T>>,
    pub custom_bootstrap_compute: Option<BootstrapComputeFn<T>>,
}

impl<T: Float> Default for BatchLowessBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> BatchLowessBuilder<T> {
    // Create a new batch LOWESS builder with default parameters.
    fn new() -> Self {
        Self {
            fraction: T::from(DEFAULT_FRACTION).unwrap(),
            iterations: DEFAULT_ITERATIONS,
            delta: default_batch_delta(),
            weight_function: DEFAULT_WEIGHT_FUNCTION_ENUM,
            robustness_method: DEFAULT_ROBUSTNESS_METHOD_ENUM,
            interval_type: None,
            bootstrap: None,
            cv_fractions: None,
            cv_kind: None,
            cv_seed: DEFAULT_CV_SEED,
            deferred_error: None,
            auto_converge: default_auto_converge(),
            return_diagnostics: DEFAULT_RETURN_DIAGNOSTICS,
            compute_residuals: DEFAULT_RETURN_RESIDUALS,
            return_robustness_weights: DEFAULT_RETURN_ROBUSTNESS_WEIGHTS,
            return_derivative: DEFAULT_RETURN_DERIVATIVE,
            return_sorted: DEFAULT_RETURN_SORTED,
            missing: DEFAULT_MISSING_POLICY_ENUM,
            zero_weight_fallback: DEFAULT_ZERO_WEIGHT_FALLBACK_ENUM,
            boundary_policy: DEFAULT_BOUNDARY_POLICY_ENUM,
            scaling_method: DEFAULT_SCALING_METHOD_ENUM,
            custom_smooth_pass: None,
            custom_cv_pass: None,
            custom_interval_pass: None,
            custom_derivative_pass: None,
            custom_fit_pass: None,
            backend: None,
            delegate_boundary_handling: false,
            parallel: None,
            duplicate_param: None,
            custom_weights: None,
            retain_model: DEFAULT_RETAIN_MODEL,
            custom_predict_pass: None,
            custom_bootstrap_pass: None,
            custom_bootstrap_compute: None,
        }
    }

    // Build the batch processor.
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
        if let Some(bc) = self.bootstrap
            && bc.n_boot < MIN_BOOTSTRAP_SAMPLES
        {
            return Err(LowessError::InvalidBootstrapSamples(bc.n_boot));
        }

        // Validate CV fractions and method
        if let Some(ref fracs) = self.cv_fractions {
            Validator::validate_cv_fractions(fracs)?;
        }
        if let Some(CVKind::KFold(k)) = self.cv_kind {
            Validator::validate_kfold(k)?;
        }

        // Validate auto convergence tolerance
        if let Some(tol) = self.auto_converge {
            Validator::validate_tolerance(tol)?;
        }

        // Validate custom weights if provided (length validated against data in fit())
        // (structural validation only; length is unknown until data is provided)

        Ok(BatchLowess { config: self })
    }
}

// Batch LOWESS processor.
pub struct BatchLowess<T: Float> {
    config: BatchLowessBuilder<T>,
}

impl<T: Float + WLSSolver + Debug + Send + Sync + 'static> BatchLowess<T> {
    // Perform LOWESS smoothing on the provided data.
    pub fn fit(self, x: &[T], y: &[T]) -> Result<LowessResult<T>, LowessError> {
        Validator::validate_lengths(x, y)?;

        // Apply the missing-value policy before any other validation, so a
        // `Drop` policy sees the filtered data and `Error` sees the raw data.
        let (x_owned, y_owned, custom_weights) = match self.config.missing {
            MissingPolicy::Drop => {
                Validator::drop_non_finite(x, y, self.config.custom_weights.as_deref())
            }
            MissingPolicy::Error => (x.to_vec(), y.to_vec(), self.config.custom_weights.clone()),
        };
        let x: &[T] = &x_owned;
        let y: &[T] = &y_owned;

        Validator::validate_inputs(x, y)?;

        // Validate custom weights length against data
        if let Some(ref cw) = custom_weights {
            Validator::validate_custom_weights(cw, y.len())?;
        }

        // Sort data by x using sorting module, unless GPU is used
        let sorted = if self.config.backend == Some(Backend::GPU) {
            SortedData {
                x: x.to_vec(),
                y: y.to_vec(),
                indices: (0..x.len()).collect(),
            }
        } else {
            sort_by_x(x, y)
        };

        let delta = calculate_delta(self.config.delta, &sorted.x)?;

        let zw_flag: u8 = self.config.zero_weight_fallback.to_u8();

        let bootstrap = self
            .config
            .bootstrap
            .filter(|_| self.config.interval_type.is_some());

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
            auto_converge: self.config.auto_converge,
            // Bootstrap replaces the analytic SE pass entirely.
            return_variance: if bootstrap.is_some() {
                None
            } else {
                self.config.interval_type
            },
            boundary_policy: self.config.boundary_policy,
            scaling_method: self.config.scaling_method,
            return_derivative: self.config.return_derivative,
            cv_seed: self.config.cv_seed,
            // ++++++++++++++++++++++++++++++++++++++
            // +               DEV                  +
            // ++++++++++++++++++++++++++++++++++++++
            custom_smooth_pass: self.config.custom_smooth_pass,
            custom_cv_pass: self.config.custom_cv_pass,
            custom_interval_pass: self.config.custom_interval_pass,
            custom_derivative_pass: self.config.custom_derivative_pass,
            custom_fit_pass: self.config.custom_fit_pass,
            parallel: self.config.parallel.unwrap_or(false),
            backend: self.config.backend,
            delegate_boundary_handling: self.config.delegate_boundary_handling,
            custom_weights,
            retain_model: self.config.retain_model,
            custom_predict_pass: self.config.custom_predict_pass,
        };

        // Refit config for bootstrap replicates: same smoother, CV-selected fraction fixed later.
        let boot_base = bootstrap.map(|_| {
            let mut c = config.clone();
            c.cv_fractions = None;
            c.cv_kind = None;
            c.return_derivative = false;
            c.retain_model = false;
            c
        });

        let retained_refit = self.config.retain_model.then(|| {
            let mut refit = config.clone();
            refit.cv_fractions = None;
            refit.cv_kind = None;
            refit.return_variance = None;
            refit.return_derivative = false;
            refit
        });

        // Execute unified LOWESS
        let result = LowessExecutor::run_with_config(&sorted.x, &sorted.y, config)?;

        let y_smooth = result.smoothed;
        let mut predict_state = result.predict_state;
        let mut std_errors = result.std_errors;
        let iterations_used = result.iterations;
        let fraction_used = result.used_fraction;
        let cv_scores = result.cv_scores;
        let derivative = result.derivative;

        // Get residuals from backend or calculate them
        let residuals: Vec<T> = if let Some(r) = result.residuals {
            r
        } else {
            sorted
                .y
                .iter()
                .zip(y_smooth.iter())
                .map(|(&orig, &smoothed_val)| orig - smoothed_val)
                .collect()
        };

        if let (Some(state), Some(mut refit_config)) = (
            predict_state.as_mut().and_then(Arc::get_mut),
            retained_refit,
        ) {
            refit_config.fraction = Some(fraction_used);
            state.bootstrap_state = Some(BootstrapPredictState {
                x: sorted.x.clone(),
                smoothed: y_smooth.clone(),
                residuals: residuals.clone(),
                refit_config,
            });
        }

        let boot_out = match (bootstrap, boot_base, &self.config.interval_type) {
            (Some(bc), Some(mut base), Some(method)) => {
                base.fraction = Some(fraction_used);
                let pass = self.config.custom_bootstrap_pass;
                let out = if let Some(compute) = self.config.custom_bootstrap_compute {
                    compute(&sorted.x, &y_smooth, &residuals, method, bc, &base)?
                } else {
                    bc.compute(method, &y_smooth, &residuals, |replicates| match pass {
                        Some(p) => p(&sorted.x, replicates, &base),
                        None => replicates
                            .iter()
                            .map(|y_star| {
                                LowessExecutor::run_with_config(&sorted.x, y_star, base.clone())
                                    .map(|r| r.smoothed)
                            })
                            .collect(),
                    })?
                };
                std_errors = Some(out.std_errors.clone());
                Some(out)
            }
            _ => None,
        };

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
            match (&self.config.interval_type, &std_errors, boot_out) {
                (_, _, Some(b)) => (
                    b.confidence_lower,
                    b.confidence_upper,
                    b.prediction_lower,
                    b.prediction_upper,
                ),
                (Some(method), Some(se), None) => {
                    // Check if result already has computed intervals (e.g. from GPU)
                    if result.confidence_lower.is_some() || result.prediction_lower.is_some() {
                        (
                            result.confidence_lower,
                            result.confidence_upper,
                            result.prediction_lower,
                            result.prediction_upper,
                        )
                    } else {
                        let (cl, cu, pl, pu) =
                            method.compute_intervals(&y_smooth, se, &residuals)?;
                        (cl, cu, pl, pu)
                    }
                }
                _ => (None, None, None, None),
            };

        // Unsort results back to original input order, unless `return_sorted` was
        // requested, in which case the already-sorted arrays are returned as-is.
        let indices = &sorted.indices;
        let (
            x_out,
            y_smooth_out,
            std_errors_out,
            residuals_out,
            rob_weights_out,
            derivative_out,
            cl_out,
            cu_out,
            pl_out,
            pu_out,
        ) = if self.config.return_sorted {
            (
                sorted.x.clone(),
                y_smooth,
                std_errors,
                if self.config.compute_residuals {
                    Some(residuals)
                } else {
                    None
                },
                if self.config.return_robustness_weights {
                    Some(rob_weights)
                } else {
                    None
                },
                derivative,
                conf_lower,
                conf_upper,
                pred_lower,
                pred_upper,
            )
        } else {
            (
                x.to_vec(),
                unsort(&y_smooth, indices),
                std_errors.as_ref().map(|se| unsort(se, indices)),
                if self.config.compute_residuals {
                    Some(unsort(&residuals, indices))
                } else {
                    None
                },
                if self.config.return_robustness_weights {
                    Some(unsort(&rob_weights, indices))
                } else {
                    None
                },
                derivative.as_ref().map(|d| unsort(d, indices)),
                conf_lower.as_ref().map(|v| unsort(v, indices)),
                conf_upper.as_ref().map(|v| unsort(v, indices)),
                pred_lower.as_ref().map(|v| unsort(v, indices)),
                pred_upper.as_ref().map(|v| unsort(v, indices)),
            )
        };

        Ok(LowessResult {
            x: x_out,
            y: y_smooth_out,
            standard_errors: std_errors_out,
            confidence_lower: cl_out,
            confidence_upper: cu_out,
            prediction_lower: pl_out,
            prediction_upper: pu_out,
            residuals: residuals_out,
            robustness_weights: rob_weights_out,
            derivative: derivative_out,
            fraction_used,
            iterations_used,
            cv_scores,
            diagnostics,
            fit_state: predict_state,
        })
    }
}
