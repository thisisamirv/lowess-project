//! Execution engine for LOWESS smoothing operations.
//!
//! This module provides the core execution engine that orchestrates LOWESS
//! smoothing operations. It handles the iteration loop, robustness weight
//! updates, convergence checking, cross-validation, and variance estimation.
//! The executor is the central component that coordinates all lower-level
//! algorithms to produce smoothed results.
// ## srrstats Compliance
//
// @srrstats {RE2.0} Core LOWESS execution: boundary handling, iteration loop, convergence.
// @srrstats {G2.2} Auto-convergence tolerance for early stopping of iterations.
// Configurable robustness iterations with convergence monitoring.

// External dependencies
#[cfg(not(feature = "std"))]
use alloc::sync::Arc;
#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::fmt::Debug;
use core::iter::repeat_n;
use core::mem::swap;
use num_traits::Float;
#[cfg(feature = "std")]
use std::sync::Arc;
#[cfg(feature = "std")]
use std::vec;
#[cfg(feature = "std")]
use std::vec::Vec;

// Internal dependencies
use crate::adapters::defaults::*;
use crate::algorithms::defaults::*;
use crate::algorithms::interpolation::{interpolate_gap, interpolate_gap_derivative};
use crate::algorithms::regression::{LinearFit, RegressionContext, WLSSolver, ZeroWeightFallback};
use crate::algorithms::robustness::RobustnessMethod;
use crate::engine::predict::{PredictPassFn, PredictState};
use crate::evaluation::cv::CVKind;
use crate::evaluation::intervals::IntervalMethod;
use crate::math::boundary::{BoundaryPolicy, apply_boundary_policy};
use crate::math::defaults::*;
use crate::math::kernel::WeightFunction;
use crate::math::scaling::ScalingMethod;
use crate::primitives::backend::Backend;
use crate::primitives::buffer::LowessBuffer;
use crate::primitives::errors::LowessError;
use crate::primitives::window::Window;

// Signature for custom smooth pass function
pub type SmoothPassFn<T> = fn(
    &[T],           // x
    &[T],           // y
    usize,          // window_size
    T,              // delta (interpolation optimization threshold)
    bool,           // use_robustness
    &[T],           // robustness_weights
    &mut [T],       // output (y_smooth)
    WeightFunction, // weight_function
    u8,             // zero_weight_flag
    Option<&[T]>,   // custom_weights (per-observation user weights)
);

// Signature for custom cross-validation pass function
pub type CVPassFn<T> = fn(
    &[T],             // x
    &[T],             // y
    &[T],             // candidate fractions
    CVKind,           // CV strategy
    &LowessConfig<T>, // Config for internal fits
) -> Result<(T, Vec<T>), LowessError>; // (best_fraction, scores)

// Signature for custom interval estimation pass function
pub type IntervalPassFn<T> = fn(
    &[T],               // x
    &[T],               // y
    &[T],               // y_smooth
    usize,              // window_size
    &[T],               // robustness_weights
    WeightFunction,     // weight_function
    &IntervalMethod<T>, // interval configuration
) -> Vec<T>; // standard errors

// Signature for custom derivative (local fit slope) estimation pass function
pub type DerivativePassFn<T> = fn(
    &[T],           // x
    &[T],           // y
    usize,          // window_size
    T,              // delta
    &[T],           // robustness_weights
    WeightFunction, // weight_function
    u8,             // zero_weight_flag
    Option<&[T]>,   // custom_weights
) -> Vec<T>; // derivative

// Result tuple from an iteration loop or fit pass.
#[allow(clippy::type_complexity)]
pub type IterationResult<T> = (
    Vec<T>,         // smoothed
    Option<Vec<T>>, // std_errors
    usize,          // iterations
    Vec<T>,         // robustness_weights
    Option<Vec<T>>, // residuals
    Option<Vec<T>>, // confidence_lower
    Option<Vec<T>>, // confidence_upper
    Option<Vec<T>>, // prediction_lower
    Option<Vec<T>>, // prediction_upper
);

// Signature for custom iteration batch pass function (GPU acceleration).
pub type FitPassFn<T> = fn(
    &[T],             // x
    &[T],             // y
    &LowessConfig<T>, // full configuration
) -> Result<IterationResult<T>, LowessError>;

// Output from LOWESS execution.
#[derive(Debug, Clone)]
pub struct ExecutorOutput<T> {
    // Smoothed y-values.
    pub smoothed: Vec<T>,

    // Standard errors (if SE estimation or intervals were requested).
    pub std_errors: Option<Vec<T>>,

    // Number of iterations performed (if auto-convergence was active).
    pub iterations: Option<usize>,

    // Smoothing fraction used (selected by CV or configured).
    pub used_fraction: T,

    // RMSE scores for each tested fraction (if CV was performed).
    pub cv_scores: Option<Vec<T>>,

    // Final robustness weights from iterative refinement.
    pub robustness_weights: Vec<T>,

    // Residuals (y - y_smooth), if available from backend.
    pub residuals: Option<Vec<T>>,

    // Confidence interval lower bounds (if intervals were computed).
    pub confidence_lower: Option<Vec<T>>,

    // Confidence interval upper bounds (if intervals were computed).
    pub confidence_upper: Option<Vec<T>>,

    // Prediction interval lower bounds (if intervals were computed).
    pub prediction_lower: Option<Vec<T>>,

    // Prediction interval upper bounds (if intervals were computed).
    pub prediction_upper: Option<Vec<T>>,

    // Per-point local fit derivative (slope), if `return_derivative` was set.
    pub derivative: Option<Vec<T>>,

    // Retained fitted-model state for `Predict::call()`, if `retain_model` was set.
    pub predict_state: Option<Arc<PredictState<T>>>,
}

// Configuration for LOWESS execution.
#[derive(Debug, Clone)]
pub struct LowessConfig<T> {
    // Smoothing fraction (0, 1].
    // If `None` and `cv_fractions` are provided, bandwidth selection is performed.
    pub fraction: Option<T>,

    // Number of robustness iterations (0 means initial fit only).
    pub iterations: usize,

    // Delta parameter for linear interpolation optimization.
    pub delta: T,

    // Kernel weight function used for local regression.
    pub weight_function: WeightFunction,

    // Zero-weight fallback policy (via [`ZeroWeightFallback`]).
    pub zero_weight_fallback: u8,

    // Robustness weighting method for outlier downweighting.
    pub robustness_method: RobustnessMethod,

    // Candidate fractions to evaluate during cross-validation.
    pub cv_fractions: Option<Vec<T>>,

    // Cross-validation strategy (e.g., K-Fold or LOOCV).
    pub cv_kind: Option<CVKind>,

    // Seed for random number generation in cross-validation.
    pub cv_seed: Option<u64>,

    // Convergence tolerance for early stopping of robustness iterations.
    pub auto_converge: Option<T>,

    // Configuration for standard errors and intervals.
    pub return_variance: Option<IntervalMethod<T>>,

    // Boundary handling policy.
    pub boundary_policy: BoundaryPolicy,

    // Scaling method for robust scale estimation.
    pub scaling_method: ScalingMethod,

    // Whether to compute per-point local fit derivative (slope) (Batch only).
    pub return_derivative: bool,

    // ++++++++++++++++++++++++++++++++++++++
    // +               DEV                  +
    // ++++++++++++++++++++++++++++++++++++++
    // Custom smooth pass function (enables parallel execution).
    pub custom_smooth_pass: Option<SmoothPassFn<T>>,

    // Custom cross-validation pass function.
    pub custom_cv_pass: Option<CVPassFn<T>>,

    // Custom interval estimation pass function.
    pub custom_interval_pass: Option<IntervalPassFn<T>>,

    // Custom derivative (local fit slope) estimation pass function.
    pub custom_derivative_pass: Option<DerivativePassFn<T>>,

    // Custom iteration batch pass function for GPU acceleration.
    pub custom_fit_pass: Option<FitPassFn<T>>,

    // Execution backend hint for extension crates.
    pub backend: Option<Backend>,

    // Whether to use parallel execution
    pub parallel: bool,

    // Whether to delegate boundary handling (padding) to the custom_fit_pass
    pub delegate_boundary_handling: bool,

    // Per-observation case weights. When provided, multiplies each local kernel weight:
    // `w_ij = custom_weights[j] * K(d_ij / h) * robustness_j`.
    pub custom_weights: Option<Vec<T>>,

    // Whether to retain fitted-model state for later `Predict::call()` calls (Batch only).
    pub retain_model: bool,

    // Custom (e.g. parallel) predict pass function.
    pub custom_predict_pass: Option<PredictPassFn<T>>,
}

impl<T: Float> Default for LowessConfig<T> {
    fn default() -> Self {
        Self {
            fraction: None,
            iterations: DEFAULT_ITERATIONS,
            delta: T::from(DEFAULT_DELTA).unwrap(),
            weight_function: DEFAULT_WEIGHT_FUNCTION_ENUM,
            zero_weight_fallback: 0,
            robustness_method: DEFAULT_ROBUSTNESS_METHOD_ENUM,
            cv_fractions: None,
            cv_kind: None,
            cv_seed: DEFAULT_CV_SEED,
            auto_converge: default_auto_converge(),
            return_variance: None,
            boundary_policy: DEFAULT_BOUNDARY_POLICY_ENUM,
            scaling_method: DEFAULT_SCALING_METHOD_ENUM,
            return_derivative: false,
            custom_smooth_pass: None,
            custom_cv_pass: None,
            custom_interval_pass: None,
            custom_derivative_pass: None,
            custom_weights: None,
            custom_fit_pass: None,
            parallel: false,
            backend: None,
            delegate_boundary_handling: false,
            retain_model: false,
            custom_predict_pass: None,
        }
    }
}

// Unified executor for LOWESS smoothing operations.
#[derive(Debug, Clone)]
pub struct LowessExecutor<T: Float> {
    // Smoothing fraction (0, 1].
    pub fraction: T,

    // Number of robustness iterations.
    pub iterations: usize,

    // Delta for interpolation optimization.
    pub delta: T,

    // Kernel weight function.
    pub weight_function: WeightFunction,

    // Zero weight fallback flag (0=UseLocalMean, 1=ReturnOriginal, 2=ReturnNone).
    pub zero_weight_fallback: u8,

    // Robustness method for iterative refinement.
    pub robustness_method: RobustnessMethod,

    // Boundary handling policy.
    pub boundary_policy: BoundaryPolicy,

    // Scaling method for robust scale estimation.
    pub scaling_method: ScalingMethod,

    // Auto-convergence tolerance.
    pub auto_converge: Option<T>,

    // Interval estimation method.
    pub interval_method: Option<IntervalMethod<T>>,

    // Whether to compute per-point local fit derivative (slope) (Batch only).
    pub return_derivative: bool,

    // ++++++++++++++++++++++++++++++++++++++
    // +               DEV                  +
    // ++++++++++++++++++++++++++++++++++++++
    // Custom smooth pass function (e.g., for parallel execution).
    pub custom_smooth_pass: Option<SmoothPassFn<T>>,

    // Custom cross-validation pass function.
    pub custom_cv_pass: Option<CVPassFn<T>>,

    // Custom interval estimation pass function.
    pub custom_interval_pass: Option<IntervalPassFn<T>>,

    // Custom derivative (local fit slope) estimation pass function.
    pub custom_derivative_pass: Option<DerivativePassFn<T>>,

    // Custom iteration batch pass function for GPU acceleration.
    pub custom_fit_pass: Option<FitPassFn<T>>,

    // Execution backend hint for extension crates.
    pub backend: Option<Backend>,

    // Whether to use parallel execution
    pub parallel: bool,

    pub delegate_boundary_handling: bool,

    // Per-observation case weights applied as `w_ij = custom_weights[j] * K(d_ij / h) * robustness_j`.
    pub custom_weights: Option<Vec<T>>,

    // Whether to retain fitted-model state for later `Predict::call()` calls (Batch only).
    pub retain_model: bool,

    // Custom (e.g. parallel) predict pass function.
    pub custom_predict_pass: Option<PredictPassFn<T>>,
}

struct IterationLoopOptions<'a, T: Float> {
    eff_fraction: T,
    window_size: usize,
    niter: usize,
    delta: T,
    weight_function: WeightFunction,
    zero_weight_flag: u8,
    robustness_updater: &'a RobustnessMethod,
    interval_method: Option<&'a IntervalMethod<T>>,
    convergence_tolerance: Option<T>,
    smooth_pass_fn: Option<SmoothPassFn<T>>,
    interval_pass_fn: Option<IntervalPassFn<T>>,
    custom_weights: Option<&'a [T]>,
}

struct RobustnessUpdateBuffers<'a, T> {
    residuals: &'a mut [T],
    robustness_weights: &'a mut [T],
    scratch: &'a mut [T],
}

impl<T: Float> Default for LowessExecutor<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> LowessExecutor<T> {
    // Create a new executor with default parameters.
    pub fn new() -> Self {
        Self {
            fraction: T::from(DEFAULT_FRACTION).unwrap_or_else(|| T::from(0.5).unwrap()),
            iterations: DEFAULT_ITERATIONS,
            delta: T::from(DEFAULT_DELTA).unwrap_or_else(|| T::from(0.5).unwrap()),
            weight_function: DEFAULT_WEIGHT_FUNCTION_ENUM,
            zero_weight_fallback: 0,
            robustness_method: DEFAULT_ROBUSTNESS_METHOD_ENUM,
            boundary_policy: DEFAULT_BOUNDARY_POLICY_ENUM,
            scaling_method: DEFAULT_SCALING_METHOD_ENUM,
            auto_converge: default_auto_converge(),
            interval_method: None,
            return_derivative: false,
            custom_smooth_pass: None,
            custom_cv_pass: None,
            custom_interval_pass: None,
            custom_derivative_pass: None,
            custom_fit_pass: None,
            parallel: false,
            backend: None,
            delegate_boundary_handling: false,
            custom_weights: None,
            retain_model: false,
            custom_predict_pass: None,
        }
    }

    // Create a new executor from a `LowessConfig`.
    pub fn from_config(config: &LowessConfig<T>) -> Self {
        let default_frac = T::from(DEFAULT_FRACTION).unwrap_or_else(|| T::from(0.5).unwrap());
        Self::new()
            .fraction(config.fraction.unwrap_or(default_frac))
            .iterations(config.iterations)
            .delta(config.delta)
            .weight_function(config.weight_function)
            .zero_weight_fallback(config.zero_weight_fallback)
            .robustness_method(config.robustness_method)
            .boundary_policy(config.boundary_policy)
            .scaling_method(config.scaling_method)
            .auto_converge(config.auto_converge)
            .interval_method(config.return_variance)
            .return_derivative(config.return_derivative)
            // ++++++++++++++++++++++++++++++++++++++
            // +               DEV                  +
            // ++++++++++++++++++++++++++++++++++++++
            .custom_smooth_pass(config.custom_smooth_pass)
            .custom_cv_pass(config.custom_cv_pass)
            .custom_interval_pass(config.custom_interval_pass)
            .custom_derivative_pass(config.custom_derivative_pass)
            .custom_fit_pass(config.custom_fit_pass)
            .parallel(config.parallel)
            .backend(config.backend)
            .delegate_boundary_handling(config.delegate_boundary_handling)
            .retain_model(config.retain_model)
            .custom_predict_pass(config.custom_predict_pass)
    }

    // Convert executor settings back to a `LowessConfig`.
    pub fn to_config(
        &self,
        fraction: Option<T>,
        tolerance: Option<T>,
        interval_method: Option<&IntervalMethod<T>>,
    ) -> LowessConfig<T> {
        LowessConfig {
            fraction: fraction.or(Some(self.fraction)),
            iterations: self.iterations,
            delta: self.delta,
            weight_function: self.weight_function,
            zero_weight_fallback: self.zero_weight_fallback,
            robustness_method: self.robustness_method,
            cv_fractions: None,
            cv_kind: None,
            cv_seed: None,
            auto_converge: tolerance,
            return_variance: interval_method.cloned(),
            boundary_policy: self.boundary_policy,
            scaling_method: self.scaling_method,
            return_derivative: self.return_derivative,
            // ++++++++++++++++++++++++++++++++++++++
            // +               DEV                  +
            // ++++++++++++++++++++++++++++++++++++++
            custom_smooth_pass: self.custom_smooth_pass,
            custom_cv_pass: self.custom_cv_pass,
            custom_interval_pass: self.custom_interval_pass,
            custom_derivative_pass: self.custom_derivative_pass,
            custom_fit_pass: self.custom_fit_pass,
            parallel: self.parallel,
            backend: self.backend,
            delegate_boundary_handling: self.delegate_boundary_handling,
            custom_weights: self.custom_weights.clone(),
            retain_model: self.retain_model,
            custom_predict_pass: self.custom_predict_pass,
        }
    }

    // Set the smoothing fraction (bandwidth).
    pub fn fraction(mut self, frac: T) -> Self {
        self.fraction = frac;
        self
    }

    // Set the number of robustness iterations.
    pub fn iterations(mut self, niter: usize) -> Self {
        self.iterations = niter;
        self
    }

    // Set the delta parameter for interpolation optimization.
    pub fn delta(mut self, delta: T) -> Self {
        self.delta = delta;
        self
    }

    // Set the kernel weight function.
    pub fn weight_function(mut self, wf: WeightFunction) -> Self {
        self.weight_function = wf;
        self
    }

    // Set the zero weight fallback policy flag.
    pub fn zero_weight_fallback(mut self, flag: u8) -> Self {
        self.zero_weight_fallback = flag;
        self
    }

    // Set the robustness method for iterative refinement.
    pub fn robustness_method(mut self, method: RobustnessMethod) -> Self {
        self.robustness_method = method;
        self
    }

    // Set the boundary handling policy.
    pub fn boundary_policy(mut self, policy: BoundaryPolicy) -> Self {
        self.boundary_policy = policy;
        self
    }

    // Set the scaling method for robust scale estimation.
    pub fn scaling_method(mut self, method: ScalingMethod) -> Self {
        self.scaling_method = method;
        self
    }

    // Set the auto-convergence tolerance.
    pub fn auto_converge(mut self, tolerance: Option<T>) -> Self {
        self.auto_converge = tolerance;
        self
    }

    // Set the interval estimation method.
    pub fn interval_method(mut self, method: Option<IntervalMethod<T>>) -> Self {
        self.interval_method = method;
        self
    }

    // Set whether to compute per-point local fit derivative (slope).
    pub fn return_derivative(mut self, return_derivative: bool) -> Self {
        self.return_derivative = return_derivative;
        self
    }
    // ++++++++++++++++++++++++++++++++++++++
    // +               DEV                  +
    // ++++++++++++++++++++++++++++++++++++++

    // Set a custom smooth pass function (e.g., for parallelization).
    pub fn custom_smooth_pass(mut self, smooth_pass_fn: Option<SmoothPassFn<T>>) -> Self {
        self.custom_smooth_pass = smooth_pass_fn;
        self
    }

    // Set a custom cross-validation pass function.
    pub fn custom_cv_pass(mut self, cv_pass_fn: Option<CVPassFn<T>>) -> Self {
        self.custom_cv_pass = cv_pass_fn;
        self
    }

    // Set a custom interval estimation pass function.
    pub fn custom_interval_pass(mut self, interval_pass_fn: Option<IntervalPassFn<T>>) -> Self {
        self.custom_interval_pass = interval_pass_fn;
        self
    }

    // Set a custom derivative (local fit slope) estimation pass function.
    pub fn custom_derivative_pass(
        mut self,
        derivative_pass_fn: Option<DerivativePassFn<T>>,
    ) -> Self {
        self.custom_derivative_pass = derivative_pass_fn;
        self
    }

    // Set whether to use parallel execution.
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    // Set the execution backend hint.
    pub fn backend(mut self, backend: Option<Backend>) -> Self {
        self.backend = backend;
        self
    }

    pub fn delegate_boundary_handling(mut self, delegate: bool) -> Self {
        self.delegate_boundary_handling = delegate;
        self
    }

    // Set whether to retain fitted-model state for later `Predict::call()` calls.
    pub fn retain_model(mut self, retain: bool) -> Self {
        self.retain_model = retain;
        self
    }

    // Set a custom (e.g. parallel) predict pass function.
    pub fn custom_predict_pass(mut self, predict_pass_fn: Option<PredictPassFn<T>>) -> Self {
        self.custom_predict_pass = predict_pass_fn;
        self
    }

    // Set a custom iteration batch pass function (e.g., for GPU acceleration).
    pub fn custom_fit_pass(mut self, fit_pass_fn: Option<FitPassFn<T>>) -> Self {
        self.custom_fit_pass = fit_pass_fn;
        self
    }

    // Set per-observation case weights.
    pub fn custom_weights(mut self, weights: Vec<T>) -> Self {
        self.custom_weights = Some(weights);
        self
    }

    // Set per-observation case weights from an Option (None clears them).
    pub fn custom_weights_opt(self, weights: Option<Vec<T>>) -> Self {
        match weights {
            Some(w) => self.custom_weights(w),
            None => self,
        }
    }

    // Smooth data using a `LowessConfig` payload.
    pub fn run_with_config(
        x: &[T],
        y: &[T],
        config: LowessConfig<T>,
    ) -> Result<ExecutorOutput<T>, LowessError>
    where
        T: Float + WLSSolver + Debug + Send + Sync + 'static,
    {
        // Extract custom_weights before building executor (CV inner runs should not use them,
        // since CV sub-indexes data and weights would have mismatched length).
        let custom_weights = config.custom_weights.clone();
        let executor = LowessExecutor::from_config(&config);

        // Handle cross-validation if configured
        if let Some(ref cv_fracs) = config.cv_fractions {
            let cv_kind = config.cv_kind.unwrap_or(CVKind::KFold(5));

            // Run CV to find best fraction
            let (best_frac, scores) = if let Some(callback) = config.custom_cv_pass {
                callback(x, y, cv_fracs, cv_kind, &config)?
            } else {
                use crate::primitives::buffer::CVBuffer;
                let mut cv_buffer = CVBuffer::new();
                cv_kind.run(
                    x,
                    y,
                    1, // dimensions for LOWESS
                    cv_fracs,
                    config.cv_seed,
                    |tx, ty, f| {
                        executor
                            .clone() // Clone executor to set fraction for CV
                            .fraction(f)
                            .iterations(config.iterations)
                            .delta(config.delta)
                            .weight_function(config.weight_function)
                            .zero_weight_fallback(config.zero_weight_fallback)
                            .robustness_method(config.robustness_method)
                            .boundary_policy(config.boundary_policy)
                            .scaling_method(config.scaling_method)
                            .custom_smooth_pass(config.custom_smooth_pass)
                            .custom_cv_pass(config.custom_cv_pass)
                            .custom_interval_pass(config.custom_interval_pass)
                            .custom_derivative_pass(config.custom_derivative_pass)
                            .custom_fit_pass(config.custom_fit_pass)
                            .parallel(config.parallel)
                            .backend(config.backend)
                            .delegate_boundary_handling(config.delegate_boundary_handling)
                            // CV candidate fits never need retained model state or a
                            // per-point derivative - only the final fit (below) does,
                            // avoiding wasted clones/compute per candidate.
                            .retain_model(false)
                            .return_derivative(false)
                            .run(tx, ty, None)
                            .map(|output| output.smoothed)
                    },
                    None::<fn(&[T], &[T], &[T], T) -> Vec<T>>,
                    &mut cv_buffer,
                )?
            };

            // Run final pass with best fraction
            let mut output = executor
                .fraction(best_frac)
                .iterations(config.iterations)
                .delta(config.delta)
                .weight_function(config.weight_function)
                .zero_weight_fallback(config.zero_weight_fallback)
                .robustness_method(config.robustness_method)
                .boundary_policy(config.boundary_policy)
                .scaling_method(config.scaling_method)
                .auto_converge(config.auto_converge)
                .interval_method(config.return_variance)
                .custom_smooth_pass(config.custom_smooth_pass)
                .custom_cv_pass(config.custom_cv_pass)
                .custom_interval_pass(config.custom_interval_pass)
                .custom_derivative_pass(config.custom_derivative_pass)
                .custom_fit_pass(config.custom_fit_pass)
                .parallel(config.parallel)
                .backend(config.backend)
                .delegate_boundary_handling(config.delegate_boundary_handling)
                .custom_weights_opt(custom_weights)
                .run(x, y, None)?;
            output.cv_scores = Some(scores);
            output.used_fraction = best_frac;
            Ok(output)
        } else {
            // Direct run (no CV)
            executor.custom_weights_opt(custom_weights).run(x, y, None)
        }
    }

    // Execute smoothing with explicit overrides for specific parameters.
    pub fn run(
        &self,
        x: &[T],
        y: &[T],
        buffer: Option<&mut LowessBuffer<T>>,
    ) -> Result<ExecutorOutput<T>, LowessError>
    where
        T: Float + WLSSolver + Debug + Send + Sync + 'static,
    {
        let n = x.len();
        let eff_fraction = self.fraction;
        let target_iterations = self.iterations;
        let tolerance = self.auto_converge;
        let confidence_method = self.interval_method.as_ref();

        // Handle global regression (fraction >= 1.0)
        if eff_fraction >= T::one() {
            let model = LinearFit::fit_ols(x, y);
            let smoothed = x.iter().map(|&xi| model.predict(xi)).collect();
            return Ok(ExecutorOutput {
                smoothed,
                std_errors: confidence_method.map(|_| model.ols_std_errors(x, y)),
                iterations: None,
                used_fraction: eff_fraction,
                cv_scores: None,
                robustness_weights: vec![T::one(); n],
                residuals: None,
                confidence_lower: None,
                confidence_upper: None,
                prediction_lower: None,
                prediction_upper: None,
                // A global regression is a single line: its slope is the same
                // constant everywhere, so this is exact (not an approximation).
                derivative: self.return_derivative.then(|| vec![model.slope; n]),
                // Out-of-sample prediction is not supported for the fraction >= 1.0
                // (plain global OLS) special case; predict() will error if requested.
                predict_state: None,
            });
        }

        // Calculate window size
        let window_size = Window::calculate_span(n, eff_fraction);

        // Handle boundary padding (unless delegated)
        let (x_in, y_in, pad_len) = if !self.delegate_boundary_handling
            && self.boundary_policy != BoundaryPolicy::NoBoundary
        {
            let (px, py) = apply_boundary_policy(x, y, window_size, self.boundary_policy);
            let pad = (px.len() - x.len()) / 2;
            (px, py, pad)
        } else {
            (x.to_vec(), y.to_vec(), 0)
        };

        let x_ref = &x_in;
        let y_ref = &y_in;

        // When boundary padding is applied the padded arrays are larger than the original data.
        // custom_weights must be padded to the same length (using 1.0 for synthetic boundary
        // points so they participate in the local fit without extra up- or down-weighting).
        let padded_cw: Option<Vec<T>> = (pad_len > 0)
            .then(|| {
                self.custom_weights.as_ref().map(|cw| {
                    let padded_len = x_in.len();
                    let mut pv: Vec<T> = Vec::with_capacity(padded_len);
                    pv.extend(repeat_n(T::one(), pad_len));
                    pv.extend_from_slice(cw);
                    pv.extend(repeat_n(T::one(), padded_len - pad_len - cw.len()));
                    pv
                })
            })
            .flatten();
        let effective_custom_weights: Option<&[T]> = if pad_len > 0 {
            padded_cw.as_deref()
        } else {
            self.custom_weights.as_deref()
        };

        // Run the iteration loop
        let (
            (
                mut smoothed,
                mut std_errors,
                iterations,
                mut robustness_weights,
                mut residuals,
                confidence_lower,
                confidence_upper,
                prediction_lower,
                prediction_upper,
            ),
            inline_derivative,
        ) = self.iteration_loop_with_callback(
            x_ref,
            y_ref,
            IterationLoopOptions {
                eff_fraction,
                window_size,
                niter: target_iterations,
                delta: self.delta,
                weight_function: self.weight_function,
                zero_weight_flag: self.zero_weight_fallback,
                robustness_updater: &self.robustness_method,
                interval_method: confidence_method,
                convergence_tolerance: tolerance,
                smooth_pass_fn: self.custom_smooth_pass,
                interval_pass_fn: self.custom_interval_pass,
                custom_weights: effective_custom_weights,
            },
            buffer,
        )?;

        // Capture retained model state (if requested) from the padded arrays/weights
        // actually used for fitting, before they get sliced back to the original range.
        // `residual_sd` is filled in below, once the unpadded residuals are available.
        let mut predict_state = self.retain_model.then(|| PredictState {
            x: x_in.clone(),
            y: y_in.clone(),
            y_smooth: smoothed.clone(),
            robustness_weights: robustness_weights.clone(),
            window_size,
            weight_function: self.weight_function,
            zero_weight_fallback: self.zero_weight_fallback,
            custom_weights: effective_custom_weights.map(|w| w.to_vec()),
            residual_sd: T::zero(),
            train_min_x: x[0],
            train_max_x: x[n - 1],
            custom_predict_pass: self.custom_predict_pass,
        });

        // Per-point local fit derivative (slope), if requested. `iteration_loop_with_callback`
        // already captures this for free during its final smoothing iteration when possible;
        // only fall back to a separate re-fit pass when that wasn't possible (a custom smooth
        // or derivative pass is in play).
        let mut derivative = inline_derivative.or_else(|| {
            self.return_derivative.then(|| {
                Self::compute_derivative(
                    x_ref,
                    y_ref,
                    window_size,
                    self.delta,
                    &robustness_weights,
                    self.weight_function,
                    self.zero_weight_fallback,
                    effective_custom_weights,
                    self.custom_derivative_pass,
                )
            })
        });

        // Slice back to original range if padded
        if pad_len > 0 {
            Self::slice_results(
                n,
                pad_len,
                &mut smoothed,
                &mut std_errors,
                &mut robustness_weights,
            );
            // Slice residuals if present
            if let Some(ref mut resid) = residuals {
                let mut sliced = Vec::with_capacity(n);
                sliced.extend_from_slice(&resid[pad_len..n + pad_len]);
                *resid = sliced;
            }
            if let Some(ref mut deriv) = derivative {
                let mut sliced = Vec::with_capacity(n);
                sliced.extend_from_slice(&deriv[pad_len..n + pad_len]);
                *deriv = sliced;
            }
        }

        // Compute the global residual SD from the unpadded residuals (matching how
        // `Diagnostics::compute` derives it in `adapters::batch`), for prediction intervals.
        if let Some(ref mut state) = predict_state {
            let unpadded_y = &y_in[pad_len..pad_len + n];
            let unpadded_residuals: Vec<T> = unpadded_y
                .iter()
                .zip(smoothed.iter())
                .map(|(&yi, &si)| yi - si)
                .collect();
            state.residual_sd = IntervalMethod::calculate_residual_sd(&unpadded_residuals);
        }

        Ok(ExecutorOutput {
            smoothed,
            std_errors,
            iterations: Some(iterations),
            used_fraction: eff_fraction,
            cv_scores: None,
            robustness_weights,
            residuals,
            confidence_lower,
            confidence_upper,
            prediction_lower,
            prediction_upper,
            derivative,
            // Wrapped here (once all mutations above are done) so cloning a `LowessResult`
            // (e.g. to hand to multiple worker threads) is a cheap refcount bump instead of
            // deep-copying the whole padded training set.
            predict_state: predict_state.map(Arc::new),
        })
    }

    // Perform the full LOWESS iteration loop.
    fn iteration_loop_with_callback(
        &self,
        x: &[T],
        y: &[T],
        options: IterationLoopOptions<'_, T>,
        buffer: Option<&mut LowessBuffer<T>>,
    ) -> Result<(IterationResult<T>, Option<Vec<T>>), LowessError>
    where
        T: Float + WLSSolver + Debug + Send + Sync + 'static,
    {
        let IterationLoopOptions {
            eff_fraction,
            window_size,
            niter,
            delta,
            weight_function,
            zero_weight_flag,
            robustness_updater,
            interval_method,
            convergence_tolerance,
            smooth_pass_fn,
            interval_pass_fn,
            custom_weights,
        } = options;

        if let Some(fit_pass) = self.custom_fit_pass {
            let config = self.to_config(Some(eff_fraction), convergence_tolerance, interval_method);
            return fit_pass(x, y, &config).map(|r| (r, None));
        }

        let n = x.len();
        let mut internal_buffers;
        let buffers = if let Some(b) = buffer {
            b.prepare(n, convergence_tolerance.is_some());
            b
        } else {
            internal_buffers = LowessBuffer::with_capacity(n);
            internal_buffers.prepare(n, convergence_tolerance.is_some());
            &mut internal_buffers
        };
        let mut iterations_performed = 0;

        // Whether the final smoothing pass below can capture the local fit's derivative
        // for free (it's already computed alongside `y` in the same regression solve),
        // avoiding a separate re-fit pass in `run()` afterward. Not possible when a custom
        // smooth or derivative pass is in play, since those don't go through this capture.
        let capture_derivative_inline = self.return_derivative
            && smooth_pass_fn.is_none()
            && self.custom_derivative_pass.is_none();

        // Copy initial y values to y_smooth
        buffers.y_smooth.copy_from_slice(y);

        // Smoothing iterations with robustness updates
        for iter in 0..=niter {
            iterations_performed = iter;

            // Swap buffers if checking convergence (save previous state)
            if convergence_tolerance.is_some() && iter > 0 {
                swap(&mut buffers.y_smooth, &mut buffers.y_prev);
            }

            // Perform smoothing pass
            if let Some(callback) = smooth_pass_fn {
                callback(
                    x,
                    y,
                    window_size,
                    delta,
                    iter > 0, // use_robustness
                    &buffers.robustness_weights,
                    &mut buffers.y_smooth,
                    weight_function,
                    zero_weight_flag,
                    custom_weights,
                );
            } else {
                Self::smooth_pass(
                    x,
                    y,
                    window_size,
                    delta,
                    iter > 0, // use_robustness
                    &buffers.robustness_weights,
                    &mut buffers.y_smooth,
                    weight_function,
                    &mut buffers.weights,
                    zero_weight_flag,
                    custom_weights,
                    // Every iteration keeps overwriting the same scratch slot (cheap - the
                    // slope is already produced by the same solve as `y`); whichever
                    // iteration turns out to be the last (whether via convergence break
                    // or exhausting `niter`) leaves the correct final-iteration slope.
                    capture_derivative_inline
                        .then(|| buffers.derivative.as_vec_mut().as_mut_slice()),
                );
            }

            // Check convergence if tolerance is provided (skip on first iteration)
            if let Some(tol) = convergence_tolerance
                && iter > 0
                && Self::check_convergence(&buffers.y_smooth, &buffers.y_prev, tol)
            {
                break;
            }

            // Update robustness weights for next iteration (skip last). If the residual
            // scale has collapsed to ~zero (mirrors `stats::lowess`'s early exit), stop
            // robustifying here and keep this iteration's fit rather than reweight from
            // an unreliable near-zero scale.
            if iter < niter
                && Self::update_robustness_weights(
                    y,
                    &buffers.y_smooth,
                    RobustnessUpdateBuffers {
                        residuals: &mut buffers.residuals,
                        robustness_weights: &mut buffers.robustness_weights,
                        scratch: &mut buffers.scale_scratch,
                    },
                    robustness_updater,
                    self.scaling_method,
                )
            {
                break;
            }
        }

        // Compute standard errors if requested
        let std_errors = interval_method.map(|im| {
            Self::compute_std_errors(
                x,
                y,
                &buffers.y_smooth,
                window_size,
                &buffers.robustness_weights,
                weight_function,
                im,
                interval_pass_fn,
            )
        });

        let inline_derivative =
            capture_derivative_inline.then(|| buffers.derivative.as_vec().clone());

        Ok((
            (
                buffers.y_smooth.as_vec().clone(),
                std_errors,
                iterations_performed,
                buffers.robustness_weights.as_vec().clone(),
                None, // Residuals are not returned by the CPU executor
                None, // Confidence intervals are not returned by the CPU executor
                None, // Prediction intervals are not returned by the CPU executor
                None, // Confidence intervals are not returned by the CPU executor
                None, // Prediction intervals are not returned by the CPU executor
            ),
            inline_derivative,
        ))
    }

    // Perform a single smoothing pass over all points. `derivative`, if `Some`, is filled
    // in with the local fit's slope at no extra cost (`fit_with_derivative()` already
    // computes it alongside the smoothed value in the same regression solve) - this lets
    // the final iteration double as the derivative-collection pass, avoiding a separate
    // re-fit afterward.
    #[allow(clippy::too_many_arguments)]
    pub fn smooth_pass(
        x: &[T],
        y: &[T],
        window_size: usize,
        delta: T,
        use_robustness: bool,
        robustness_weights: &[T],
        y_smooth: &mut [T],
        weight_function: WeightFunction,
        weights: &mut [T],
        zero_weight_flag: u8,
        custom_weights: Option<&[T]>,
        mut derivative: Option<&mut [T]>,
    ) where
        T: WLSSolver,
    {
        let zero_weight_fallback = ZeroWeightFallback::from_u8(zero_weight_flag);

        // Fit first point
        let window = Self::fit_first_point(
            x,
            y,
            window_size,
            use_robustness,
            robustness_weights,
            weights,
            weight_function,
            zero_weight_fallback,
            y_smooth,
            custom_weights,
            derivative.as_deref_mut(),
        );

        // Fit remaining points with interpolation
        Self::fit_and_interpolate_remaining(
            x,
            y,
            delta,
            use_robustness,
            robustness_weights,
            weights,
            weight_function,
            zero_weight_fallback,
            y_smooth,
            window,
            custom_weights,
            derivative,
        );
    }

    // Compute standard errors for smoothed values.
    #[allow(clippy::too_many_arguments)]
    pub fn compute_std_errors(
        x: &[T],
        y: &[T],
        y_smooth: &[T],
        window_size: usize,
        robustness_weights: &[T],
        weight_function: WeightFunction,
        interval_method: &IntervalMethod<T>,
        interval_pass_fn: Option<IntervalPassFn<T>>,
    ) -> Vec<T> {
        if let Some(callback) = interval_pass_fn {
            return callback(
                x,
                y,
                y_smooth,
                window_size,
                robustness_weights,
                weight_function,
                interval_method,
            );
        }

        let n = x.len();
        let mut std_errors = vec![T::zero(); n];

        // Use the interval method's logic to compute SE
        interval_method.compute_window_se(
            x,
            y,
            y_smooth,
            window_size,
            robustness_weights,
            &mut std_errors,
            &|u| weight_function.compute_weight(u),
        );

        std_errors
    }

    // Check convergence between current and previous smoothed values.
    pub fn check_convergence(y_smooth: &[T], y_prev: &[T], tolerance: T) -> bool {
        let max_change = y_smooth
            .iter()
            .zip(y_prev.iter())
            .fold(T::zero(), |maxv, (&current, &previous)| {
                T::max(maxv, (current - previous).abs())
            });

        max_change <= tolerance
    }

    // Update robustness weights based on residuals. Returns `true` if the residual scale
    // was found to be degenerate (see `RobustnessMethod::apply_robustness_weights`), in
    // which case `robustness_weights` was left unchanged and the caller should stop
    // robustifying.
    fn update_robustness_weights(
        y: &[T],
        y_smooth: &[T],
        buffers: RobustnessUpdateBuffers<'_, T>,
        robustness_updater: &RobustnessMethod,
        scaling_method: ScalingMethod,
    ) -> bool {
        // Inline compute_residuals: residuals[i] = y[i] - y_smooth[i]
        for i in 0..y.len() {
            buffers.residuals[i] = y[i] - y_smooth[i];
        }
        let stopped = robustness_updater.apply_robustness_weights(
            buffers.residuals,
            buffers.robustness_weights,
            scaling_method,
            buffers.scratch,
        );
        stopped
    }

    // Helper to slice result buffers back to original data length when padding was used.
    fn slice_results(
        n: usize,
        pad_len: usize,
        smoothed: &mut Vec<T>,
        std_errors: &mut Option<Vec<T>>,
        robustness_weights: &mut Vec<T>,
    ) {
        smoothed.drain(0..pad_len);
        smoothed.truncate(n);

        if let Some(se) = std_errors.as_mut() {
            se.drain(0..pad_len);
            se.truncate(n);
        }

        robustness_weights.drain(0..pad_len);
        robustness_weights.truncate(n);
    }

    // Fit the first point and initialize the smoothing window.
    #[allow(clippy::too_many_arguments)]
    pub fn fit_single_point(
        x: &[T],
        y: &[T],
        idx: usize,
        window_size: usize,
        use_robustness: bool,
        robustness_weights: &[T],
        weights: &mut [T],
        weight_function: WeightFunction,
        zero_weight_fallback: ZeroWeightFallback,
        custom_weights: Option<&[T]>,
    ) -> (T, T, Window)
    where
        T: WLSSolver,
    {
        let n = x.len();
        let mut window = Window::initialize(idx, window_size, n);
        window.recenter(x, idx, n);

        let mut ctx = RegressionContext {
            x,
            y,
            idx,
            window,
            use_robustness,
            robustness_weights,
            weights,
            weight_function,
            zero_weight_fallback,
            custom_weights,
        };

        let (val, slope) = ctx.fit_with_derivative().unwrap_or((y[idx], T::zero()));
        (val, slope, window)
    }

    // Fit the first point and initialize the smoothing window. `derivative`, if `Some`,
    // receives the local fit's slope at the first point (used by `compute_derivative`).
    #[allow(clippy::too_many_arguments)]
    pub fn fit_first_point(
        x: &[T],
        y: &[T],
        window_size: usize,
        use_robustness: bool,
        robustness_weights: &[T],
        weights: &mut [T],
        weight_function: WeightFunction,
        zero_weight_fallback: ZeroWeightFallback,
        y_smooth: &mut [T],
        custom_weights: Option<&[T]>,
        derivative: Option<&mut [T]>,
    ) -> Window
    where
        T: WLSSolver,
    {
        let (val, slope, window) = Self::fit_single_point(
            x,
            y,
            0,
            window_size,
            use_robustness,
            robustness_weights,
            weights,
            weight_function,
            zero_weight_fallback,
            custom_weights,
        );
        y_smooth[0] = val;
        if let Some(d) = derivative {
            d[0] = slope;
        }
        window
    }

    // Main fitting loop: iterate through remaining points with delta-skipping
    // and linear interpolation. `derivative`, if `Some`, receives the local fit's slope
    // at each anchor point and the constant interpolated slope for delta-skipped points
    // (matching `y_smooth`'s own piecewise-linear interpolation there).
    #[allow(clippy::too_many_arguments)]
    // `as_deref_mut()` reborrows `Option<&mut [T]>` on each use below (needed since the
    // loop calls it repeatedly); clippy's `needless_option_as_deref` doesn't account for
    // that and would otherwise suggest moving `derivative` out on first use.
    #[allow(clippy::needless_option_as_deref)]
    fn fit_and_interpolate_remaining(
        x: &[T],
        y: &[T],
        delta: T,
        use_robustness: bool,
        robustness_weights: &[T],
        weights: &mut [T],
        weight_function: WeightFunction,
        zero_weight_fallback: ZeroWeightFallback,
        y_smooth: &mut [T],
        mut window: Window,
        custom_weights: Option<&[T]>,
        mut derivative: Option<&mut [T]>,
    ) where
        T: WLSSolver,
    {
        let n = x.len();
        let mut last_fitted = 0usize;

        // Main loop: fit anchor points and interpolate between them
        while last_fitted < n - 1 {
            let cutpoint = x[last_fitted] + delta;

            // Binary search to find the first index where x > cutpoint
            let next_idx =
                x[last_fitted + 1..].partition_point(|&xi| xi <= cutpoint) + last_fitted + 1;

            // Handle tied x-values: copy fitted value to all points with same x
            // Check the range [last_fitted+1, next_idx) for ties with last_fitted
            let mut tie_end = last_fitted;
            let x_last = x[last_fitted];
            for i in (last_fitted + 1)..next_idx.min(n) {
                if x[i] == x_last {
                    y_smooth[i] = y_smooth[last_fitted];
                    if let Some(d) = derivative.as_deref_mut() {
                        d[i] = d[last_fitted];
                    }
                    tie_end = i;
                } else {
                    break; // x is sorted, so no more ties
                }
            }
            if tie_end > last_fitted {
                last_fitted = tie_end;
            }

            // Determine current anchor point to fit
            // Either last point within delta range, or at minimum last_fitted+1
            let current = usize::max(next_idx.saturating_sub(1), last_fitted + 1).min(n - 1);

            // Check if we've made progress
            if current <= last_fitted {
                break;
            }

            // Update window to be centered around current point
            window.recenter(x, current, n);

            // Fit current point
            let mut ctx = RegressionContext {
                x,
                y,
                idx: current,
                window,
                use_robustness,
                robustness_weights,
                weights,
                weight_function,
                zero_weight_fallback,
                custom_weights,
            };

            let (val, slope) = ctx.fit_with_derivative().unwrap_or((y[current], T::zero()));
            y_smooth[current] = val;
            if let Some(d) = derivative.as_deref_mut() {
                d[current] = slope;
            }

            // Linearly interpolate between last fitted and current
            interpolate_gap(x, y_smooth, last_fitted, current);
            if let Some(d) = derivative.as_deref_mut() {
                interpolate_gap_derivative(x, y_smooth, d, last_fitted, current);
            }
            last_fitted = current;
        }

        // Final interpolation to the end if necessary
        if last_fitted < n.saturating_sub(1) {
            // Fit the last point explicitly
            let final_idx = n - 1;
            window.recenter(x, final_idx, n);

            let mut ctx = RegressionContext {
                x,
                y,
                idx: final_idx,
                window,
                use_robustness,
                robustness_weights,
                weights,
                weight_function,
                zero_weight_fallback,
                custom_weights,
            };

            let (val, slope) = ctx
                .fit_with_derivative()
                .unwrap_or((y[final_idx], T::zero()));
            y_smooth[final_idx] = val;
            if let Some(d) = derivative.as_deref_mut() {
                d[final_idx] = slope;
            }
            interpolate_gap(x, y_smooth, last_fitted, final_idx);
            if let Some(d) = derivative.as_deref_mut() {
                interpolate_gap_derivative(x, y_smooth, d, last_fitted, final_idx);
            }
        }
    }

    // Compute per-point local fit derivative (slope) using the FINAL (converged)
    // robustness weights, reusing the same delta-skip anchor selection as the smoothing
    // pass: anchors get an exact slope from their local WLS fit; skipped
    // (delta-interpolated) points get the constant slope of the linear segment
    // connecting their neighboring anchors, matching `y_smooth`'s own interpolation there.
    #[allow(clippy::too_many_arguments)]
    pub fn compute_derivative(
        x: &[T],
        y: &[T],
        window_size: usize,
        delta: T,
        robustness_weights: &[T],
        weight_function: WeightFunction,
        zero_weight_flag: u8,
        custom_weights: Option<&[T]>,
        derivative_pass_fn: Option<DerivativePassFn<T>>,
    ) -> Vec<T>
    where
        T: WLSSolver,
    {
        if let Some(callback) = derivative_pass_fn {
            return callback(
                x,
                y,
                window_size,
                delta,
                robustness_weights,
                weight_function,
                zero_weight_flag,
                custom_weights,
            );
        }

        let n = x.len();
        let mut derivative = vec![T::zero(); n];
        if n == 0 {
            return derivative;
        }

        let zero_weight_fallback = ZeroWeightFallback::from_u8(zero_weight_flag);
        let mut weights = vec![T::zero(); n];
        let mut y_scratch = vec![T::zero(); n];

        let window = Self::fit_first_point(
            x,
            y,
            window_size,
            true, // final robustness_weights already reflect the converged iteration
            robustness_weights,
            &mut weights,
            weight_function,
            zero_weight_fallback,
            &mut y_scratch,
            custom_weights,
            Some(&mut derivative),
        );

        Self::fit_and_interpolate_remaining(
            x,
            y,
            delta,
            true,
            robustness_weights,
            &mut weights,
            weight_function,
            zero_weight_fallback,
            &mut y_scratch,
            window,
            custom_weights,
            Some(&mut derivative),
        );

        derivative
    }
}
