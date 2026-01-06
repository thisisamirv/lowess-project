//! Execution engine for LOWESS smoothing operations.
//!
//! ## Purpose
//!
//! This module provides the core execution engine that orchestrates LOWESS
//! smoothing operations. It handles the iteration loop, robustness weight
//! updates, convergence checking, cross-validation, and variance estimation.
//! The executor is the central component that coordinates all lower-level
//! algorithms to produce smoothed results.
//!
//! ## Design notes
//!
//! * Provides both configuration-based and parameter-based entry points.
//! * Handles cross-validation for automatic fraction selection.
//! * Supports auto-convergence for adaptive iteration counts.
//! * Manages working buffers efficiently to minimize allocations.
//! * Uses delta optimization for performance on dense data.
//! * Separates concerns: fitting, interpolation, robustness, convergence.
//! * Generic over `Float` types to support f32 and f64.
//!
//! ## Invariants
//!
//! * Input x-values are assumed to be monotonically increasing (sorted).
//! * All working buffers have the same length as input data.
//! * Robustness weights are always in [0, 1].
//! * Window size is at least 2 and at most n.
//! * Iteration count is non-negative.
//!
//! ## Non-goals
//!
//! * This module does not validate input data (handled by `validator`).
//! * This module does not sort input data (caller's responsibility).
//! * This module does not provide public-facing result formatting.
//! * This module does not handle parallel execution directly (handled by adapters).

// Feature-gated imports
#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec;
#[cfg(feature = "std")]
use std::vec::Vec;

// External dependencies
use core::fmt::Debug;
use core::mem::swap;
use num_traits::Float;

// Internal dependencies
use crate::algorithms::interpolation::interpolate_gap;
use crate::algorithms::regression::{LinearFit, RegressionContext, WLSSolver, ZeroWeightFallback};
use crate::algorithms::robustness::RobustnessMethod;
use crate::evaluation::cv::CVKind;
use crate::evaluation::intervals::IntervalMethod;
use crate::math::boundary::{BoundaryPolicy, apply_boundary_policy};
use crate::math::kernel::WeightFunction;
use crate::math::scaling::ScalingMethod;
use crate::primitives::backend::Backend;
pub use crate::primitives::buffer::LowessBuffer;
use crate::primitives::window::Window;

// ============================================================================
// Type Definitions
// ============================================================================

/// Signature for custom smooth pass function
#[doc(hidden)]
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
);

/// Signature for custom cross-validation pass function
#[doc(hidden)]
pub type CVPassFn<T> = fn(
    &[T],             // x
    &[T],             // y
    &[T],             // candidate fractions
    CVKind,           // CV strategy
    &LowessConfig<T>, // Config for internal fits
) -> (T, Vec<T>); // (best_fraction, scores)

/// Signature for custom interval estimation pass function
#[doc(hidden)]
pub type IntervalPassFn<T> = fn(
    &[T],               // x
    &[T],               // y
    &[T],               // y_smooth
    usize,              // window_size
    &[T],               // robustness_weights
    WeightFunction,     // weight_function
    &IntervalMethod<T>, // interval configuration
) -> Vec<T>; // standard errors

/// Signature for custom iteration batch pass function (GPU acceleration).
#[doc(hidden)]
pub type FitPassFn<T> = fn(
    &[T],             // x
    &[T],             // y
    &LowessConfig<T>, // full configuration
) -> (
    Vec<T>,         // smoothed
    Option<Vec<T>>, // std_errors
    usize,          // iterations
    Vec<T>,         // robustness_weights
);

/// Output from LOWESS execution.
#[derive(Debug, Clone)]
pub struct ExecutorOutput<T> {
    /// Smoothed y-values.
    pub smoothed: Vec<T>,

    /// Standard errors (if SE estimation or intervals were requested).
    pub std_errors: Option<Vec<T>>,

    /// Number of iterations performed (if auto-convergence was active).
    pub iterations: Option<usize>,

    /// Smoothing fraction used (selected by CV or configured).
    pub used_fraction: T,

    /// RMSE scores for each tested fraction (if CV was performed).
    pub cv_scores: Option<Vec<T>>,

    /// Final robustness weights from iterative refinement.
    pub robustness_weights: Vec<T>,
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for LOWESS execution.
#[derive(Debug, Clone)]
pub struct LowessConfig<T> {
    /// Smoothing fraction (0, 1].
    /// If `None` and `cv_fractions` are provided, bandwidth selection is performed.
    pub fraction: Option<T>,

    /// Number of robustness iterations (0 means initial fit only).
    pub iterations: usize,

    /// Delta parameter for linear interpolation optimization.
    pub delta: T,

    /// Kernel weight function used for local regression.
    pub weight_function: WeightFunction,

    /// Zero-weight fallback policy (via [`ZeroWeightFallback`]).
    pub zero_weight_fallback: u8,

    /// Robustness weighting method for outlier downweighting.
    pub robustness_method: RobustnessMethod,

    /// Candidate fractions to evaluate during cross-validation.
    pub cv_fractions: Option<Vec<T>>,

    /// Cross-validation strategy (e.g., K-Fold or LOOCV).
    pub cv_kind: Option<CVKind>,

    /// Seed for random number generation in cross-validation.
    pub cv_seed: Option<u64>,

    /// Convergence tolerance for early stopping of robustness iterations.
    pub auto_convergence: Option<T>,

    /// Configuration for standard errors and intervals.
    pub return_variance: Option<IntervalMethod<T>>,

    /// Boundary handling policy.
    pub boundary_policy: BoundaryPolicy,

    /// Scaling method for robust scale estimation.
    pub scaling_method: ScalingMethod,

    // ++++++++++++++++++++++++++++++++++++++
    // +               DEV                  +
    // ++++++++++++++++++++++++++++++++++++++
    /// Custom smooth pass function (enables parallel execution).
    #[doc(hidden)]
    pub custom_smooth_pass: Option<SmoothPassFn<T>>,

    /// Custom cross-validation pass function.
    #[doc(hidden)]
    pub custom_cv_pass: Option<CVPassFn<T>>,

    /// Custom interval estimation pass function.
    #[doc(hidden)]
    pub custom_interval_pass: Option<IntervalPassFn<T>>,

    /// Custom iteration batch pass function for GPU acceleration.
    #[doc(hidden)]
    pub custom_fit_pass: Option<FitPassFn<T>>,

    /// Execution backend hint for extension crates.
    #[doc(hidden)]
    pub backend: Option<Backend>,

    /// Whether to use parallel execution
    #[doc(hidden)]
    pub parallel: bool,
}

impl<T: Float> Default for LowessConfig<T> {
    fn default() -> Self {
        Self {
            fraction: None,
            iterations: 3,
            delta: T::zero(),
            weight_function: WeightFunction::default(),
            zero_weight_fallback: 0,
            robustness_method: RobustnessMethod::default(),
            cv_fractions: None,
            cv_kind: None,
            cv_seed: None,
            auto_convergence: None,
            return_variance: None,
            boundary_policy: BoundaryPolicy::default(),
            scaling_method: ScalingMethod::default(),
            custom_smooth_pass: None,
            custom_cv_pass: None,
            custom_interval_pass: None,
            custom_fit_pass: None,
            parallel: false,
            backend: None,
        }
    }
}

/// Unified executor for LOWESS smoothing operations.
#[derive(Debug, Clone)]
pub struct LowessExecutor<T: Float> {
    /// Smoothing fraction (0, 1].
    pub fraction: T,

    /// Number of robustness iterations.
    pub iterations: usize,

    /// Delta for interpolation optimization.
    pub delta: T,

    /// Kernel weight function.
    pub weight_function: WeightFunction,

    /// Zero weight fallback flag (0=UseLocalMean, 1=ReturnOriginal, 2=ReturnNone).
    pub zero_weight_fallback: u8,

    /// Robustness method for iterative refinement.
    pub robustness_method: RobustnessMethod,

    /// Boundary handling policy.
    pub boundary_policy: BoundaryPolicy,

    /// Scaling method for robust scale estimation.
    pub scaling_method: ScalingMethod,

    /// Auto-convergence tolerance.
    pub auto_convergence: Option<T>,

    /// Interval estimation method.
    pub interval_method: Option<IntervalMethod<T>>,

    // ++++++++++++++++++++++++++++++++++++++
    // +               DEV                  +
    // ++++++++++++++++++++++++++++++++++++++
    /// Custom smooth pass function (e.g., for parallel execution).
    #[doc(hidden)]
    pub custom_smooth_pass: Option<SmoothPassFn<T>>,

    /// Custom cross-validation pass function.
    #[doc(hidden)]
    pub custom_cv_pass: Option<CVPassFn<T>>,

    /// Custom interval estimation pass function.
    #[doc(hidden)]
    pub custom_interval_pass: Option<IntervalPassFn<T>>,

    /// Custom iteration batch pass function for GPU acceleration.
    #[doc(hidden)]
    pub custom_fit_pass: Option<FitPassFn<T>>,

    /// Execution backend hint for extension crates.
    #[doc(hidden)]
    pub backend: Option<Backend>,

    /// Whether to use parallel execution
    #[doc(hidden)]
    pub parallel: bool,
}

impl<T: Float> Default for LowessExecutor<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> LowessExecutor<T> {
    // ========================================================================
    // Constructor and Builder Methods
    // ========================================================================

    /// Create a new executor with default parameters.
    pub fn new() -> Self {
        Self {
            fraction: T::from(0.67).unwrap_or_else(|| T::from(0.5).unwrap()),
            iterations: 3,
            delta: T::zero(),
            weight_function: WeightFunction::Tricube,
            zero_weight_fallback: 0,
            robustness_method: RobustnessMethod::Bisquare,
            boundary_policy: BoundaryPolicy::default(),
            scaling_method: ScalingMethod::default(),
            auto_convergence: None,
            interval_method: None,
            custom_smooth_pass: None,
            custom_cv_pass: None,
            custom_interval_pass: None,
            custom_fit_pass: None,
            parallel: false,
            backend: None,
        }
    }

    /// Create a new executor from a `LowessConfig`.
    pub fn from_config(config: &LowessConfig<T>) -> Self {
        let default_frac = T::from(0.67).unwrap_or_else(|| T::from(0.5).unwrap());
        Self::new()
            .fraction(config.fraction.unwrap_or(default_frac))
            .iterations(config.iterations)
            .delta(config.delta)
            .weight_function(config.weight_function)
            .zero_weight_fallback(config.zero_weight_fallback)
            .robustness_method(config.robustness_method)
            .boundary_policy(config.boundary_policy)
            .scaling_method(config.scaling_method)
            .auto_convergence(config.auto_convergence)
            .interval_method(config.return_variance)
            // ++++++++++++++++++++++++++++++++++++++
            // +               DEV                  +
            // ++++++++++++++++++++++++++++++++++++++
            .custom_smooth_pass(config.custom_smooth_pass)
            .custom_cv_pass(config.custom_cv_pass)
            .custom_interval_pass(config.custom_interval_pass)
            .custom_fit_pass(config.custom_fit_pass)
            .parallel(config.parallel)
            .backend(config.backend)
    }

    /// Convert executor settings back to a `LowessConfig`.
    #[doc(hidden)]
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
            auto_convergence: tolerance,
            return_variance: interval_method.cloned(),
            boundary_policy: self.boundary_policy,
            scaling_method: self.scaling_method,
            // ++++++++++++++++++++++++++++++++++++++
            // +               DEV                  +
            // ++++++++++++++++++++++++++++++++++++++
            custom_smooth_pass: self.custom_smooth_pass,
            custom_cv_pass: self.custom_cv_pass,
            custom_interval_pass: self.custom_interval_pass,
            custom_fit_pass: self.custom_fit_pass,
            parallel: self.parallel,
            backend: self.backend,
        }
    }

    /// Set the smoothing fraction (bandwidth).
    pub fn fraction(mut self, frac: T) -> Self {
        self.fraction = frac;
        self
    }

    /// Set the number of robustness iterations.
    pub fn iterations(mut self, niter: usize) -> Self {
        self.iterations = niter;
        self
    }

    /// Set the delta parameter for interpolation optimization.
    pub fn delta(mut self, delta: T) -> Self {
        self.delta = delta;
        self
    }

    /// Set the kernel weight function.
    pub fn weight_function(mut self, wf: WeightFunction) -> Self {
        self.weight_function = wf;
        self
    }

    /// Set the zero weight fallback policy flag.
    pub fn zero_weight_fallback(mut self, flag: u8) -> Self {
        self.zero_weight_fallback = flag;
        self
    }

    /// Set the robustness method for iterative refinement.
    pub fn robustness_method(mut self, method: RobustnessMethod) -> Self {
        self.robustness_method = method;
        self
    }

    /// Set the boundary handling policy.
    pub fn boundary_policy(mut self, policy: BoundaryPolicy) -> Self {
        self.boundary_policy = policy;
        self
    }

    /// Set the scaling method for robust scale estimation.
    pub fn scaling_method(mut self, method: ScalingMethod) -> Self {
        self.scaling_method = method;
        self
    }

    /// Set the auto-convergence tolerance.
    pub fn auto_convergence(mut self, tolerance: Option<T>) -> Self {
        self.auto_convergence = tolerance;
        self
    }

    /// Set the interval estimation method.
    pub fn interval_method(mut self, method: Option<IntervalMethod<T>>) -> Self {
        self.interval_method = method;
        self
    }
    // ++++++++++++++++++++++++++++++++++++++
    // +               DEV                  +
    // ++++++++++++++++++++++++++++++++++++++

    /// Set a custom smooth pass function (e.g., for parallelization).
    #[doc(hidden)]
    pub fn custom_smooth_pass(mut self, smooth_pass_fn: Option<SmoothPassFn<T>>) -> Self {
        self.custom_smooth_pass = smooth_pass_fn;
        self
    }

    /// Set a custom cross-validation pass function.
    #[doc(hidden)]
    pub fn custom_cv_pass(mut self, cv_pass_fn: Option<CVPassFn<T>>) -> Self {
        self.custom_cv_pass = cv_pass_fn;
        self
    }

    /// Set a custom interval estimation pass function.
    #[doc(hidden)]
    pub fn custom_interval_pass(mut self, interval_pass_fn: Option<IntervalPassFn<T>>) -> Self {
        self.custom_interval_pass = interval_pass_fn;
        self
    }

    /// Set whether to use parallel execution.
    #[doc(hidden)]
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// Set the execution backend hint.
    #[doc(hidden)]
    pub fn backend(mut self, backend: Option<Backend>) -> Self {
        self.backend = backend;
        self
    }

    /// Set a custom iteration batch pass function (e.g., for GPU acceleration).
    #[doc(hidden)]
    pub fn custom_fit_pass(mut self, fit_pass_fn: Option<FitPassFn<T>>) -> Self {
        self.custom_fit_pass = fit_pass_fn;
        self
    }

    // ========================================================================
    // Main Entry Points
    // ========================================================================

    /// Smooth data using a `LowessConfig` payload.
    pub fn run_with_config(x: &[T], y: &[T], config: LowessConfig<T>) -> ExecutorOutput<T>
    where
        T: Float + WLSSolver + Debug + Send + Sync + 'static,
    {
        let executor = LowessExecutor::from_config(&config);

        // Handle cross-validation if configured
        if let Some(ref cv_fracs) = config.cv_fractions {
            let cv_kind = config.cv_kind.unwrap_or(CVKind::KFold(5));

            // Run CV to find best fraction
            let (best_frac, scores) = if let Some(callback) = config.custom_cv_pass {
                callback(x, y, cv_fracs, cv_kind, &config)
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
                            .custom_fit_pass(config.custom_fit_pass)
                            .parallel(config.parallel)
                            .backend(config.backend)
                            .run(tx, ty, None)
                            .smoothed
                    },
                    None::<fn(&[T], &[T], &[T], T) -> Vec<T>>,
                    &mut cv_buffer,
                )
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
                .auto_convergence(config.auto_convergence)
                .interval_method(config.return_variance)
                .custom_smooth_pass(config.custom_smooth_pass)
                .custom_cv_pass(config.custom_cv_pass)
                .custom_interval_pass(config.custom_interval_pass)
                .custom_fit_pass(config.custom_fit_pass)
                .parallel(config.parallel)
                .backend(config.backend)
                .run(x, y, None);
            output.cv_scores = Some(scores);
            output.used_fraction = best_frac;
            output
        } else {
            // Direct run (no CV)
            executor.run(x, y, None)
        }
    }

    /// Execute smoothing with explicit overrides for specific parameters.
    ///
    /// # Special Cases
    ///
    /// * **Insufficient data** (n < 2): Returns original y-values.
    /// * **Global regression** (fraction >= 1.0): Performs OLS on the entire dataset.
    pub fn run(&self, x: &[T], y: &[T], buffer: Option<&mut LowessBuffer<T>>) -> ExecutorOutput<T>
    where
        T: Float + WLSSolver + Debug + Send + Sync + 'static,
    {
        let n = x.len();
        let eff_fraction = self.fraction;
        let target_iterations = self.iterations;
        let tolerance = self.auto_convergence;
        let confidence_method = self.interval_method.as_ref();

        // Handle global regression (fraction >= 1.0)
        if eff_fraction >= T::one() {
            let model = LinearFit::fit_ols(x, y);
            let smoothed = x.iter().map(|&xi| model.predict(xi)).collect();
            return ExecutorOutput {
                smoothed,
                std_errors: if confidence_method.is_some() {
                    Some(vec![T::zero(); n])
                } else {
                    None
                },
                iterations: None,
                used_fraction: eff_fraction,
                cv_scores: None,
                robustness_weights: vec![T::one(); n],
            };
        }

        // Calculate window size
        let window_size = Window::calculate_span(n, eff_fraction);

        // Handle boundary padding
        let (x_in, y_in, pad_len) = if self.boundary_policy != BoundaryPolicy::Extend {
            let (px, py) = apply_boundary_policy(x, y, window_size, self.boundary_policy);
            let pad = (px.len() - x.len()) / 2;
            (px, py, pad)
        } else {
            (x.to_vec(), y.to_vec(), 0)
        };

        let x_ref = &x_in;
        let y_ref = &y_in;

        // Run the iteration loop
        let (mut smoothed, mut std_errors, iterations, mut robustness_weights) = self
            .iteration_loop_with_callback(
                x_ref,
                y_ref,
                eff_fraction,
                window_size,
                target_iterations,
                self.delta,
                self.weight_function,
                self.zero_weight_fallback,
                &self.robustness_method,
                confidence_method,
                tolerance,
                self.custom_smooth_pass,
                self.custom_interval_pass,
                buffer,
            );

        // Slice back to original range if padded
        if pad_len > 0 {
            Self::slice_results(
                n,
                pad_len,
                &mut smoothed,
                &mut std_errors,
                &mut robustness_weights,
            );
        }

        ExecutorOutput {
            smoothed,
            std_errors,
            iterations: if tolerance.is_some() {
                Some(iterations)
            } else {
                None
            },
            used_fraction: eff_fraction,
            cv_scores: None,
            robustness_weights,
        }
    }

    /// Perform the full LOWESS iteration loop.
    #[allow(clippy::too_many_arguments)]
    pub fn iteration_loop_with_callback(
        &self,
        x: &[T],
        y: &[T],
        eff_fraction: T,
        window_size: usize,
        niter: usize,
        delta: T,
        weight_function: WeightFunction,
        zero_weight_flag: u8,
        robustness_updater: &RobustnessMethod,
        interval_method: Option<&IntervalMethod<T>>,
        convergence_tolerance: Option<T>,
        smooth_pass_fn: Option<SmoothPassFn<T>>,
        interval_pass_fn: Option<IntervalPassFn<T>>,
        buffer: Option<&mut LowessBuffer<T>>,
    ) -> (Vec<T>, Option<Vec<T>>, usize, Vec<T>)
    where
        T: Float + WLSSolver + Debug + Send + Sync + 'static,
    {
        if self.custom_fit_pass.is_some() {
            let config = self.to_config(Some(eff_fraction), convergence_tolerance, interval_method);
            return (self.custom_fit_pass.unwrap())(x, y, &config);
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
                );
            }

            // Check convergence if tolerance is provided (skip on first iteration)
            if let Some(tol) = convergence_tolerance {
                if iter > 0 && Self::check_convergence(&buffers.y_smooth, &buffers.y_prev, tol) {
                    break;
                }
            }

            // Update robustness weights for next iteration (skip last)
            if iter < niter {
                Self::update_robustness_weights(
                    y,
                    &buffers.y_smooth,
                    &mut buffers.residuals,
                    &mut buffers.robustness_weights,
                    robustness_updater,
                    self.scaling_method,
                    &mut buffers.weights,
                );
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

        (
            buffers.y_smooth.as_vec().clone(),
            std_errors,
            iterations_performed,
            buffers.robustness_weights.as_vec().clone(),
        )
    }

    // ========================================================================
    // Main Algorithmic Logic
    // ========================================================================

    /// Perform a single smoothing pass over all points.
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
        );
    }

    /// Compute standard errors for smoothed values.
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

    // ========================================================================
    // Helpers
    // ========================================================================

    /// Check convergence between current and previous smoothed values.
    pub fn check_convergence(y_smooth: &[T], y_prev: &[T], tolerance: T) -> bool {
        let max_change = y_smooth
            .iter()
            .zip(y_prev.iter())
            .fold(T::zero(), |maxv, (&current, &previous)| {
                T::max(maxv, (current - previous).abs())
            });

        max_change <= tolerance
    }

    /// Update robustness weights based on residuals.
    pub fn update_robustness_weights(
        y: &[T],
        y_smooth: &[T],
        residuals: &mut [T],
        robustness_weights: &mut [T],
        robustness_updater: &RobustnessMethod,
        scaling_method: ScalingMethod,
        scratch: &mut [T],
    ) {
        // Inline compute_residuals: residuals[i] = y[i] - y_smooth[i]
        for i in 0..y.len() {
            residuals[i] = y[i] - y_smooth[i];
        }
        robustness_updater.apply_robustness_weights(
            residuals,
            robustness_weights,
            scaling_method,
            scratch,
        );
    }

    /// Helper to slice result buffers back to original data length when padding was used.
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

    // ========================================================================
    // Specialized Fitting Functions
    // ========================================================================

    /// Fit the first point and initialize the smoothing window.
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
    ) -> (T, Window)
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
        };

        (ctx.fit().unwrap_or_else(|| y[idx]), window)
    }

    /// Fit the first point and initialize the smoothing window.
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
    ) -> Window
    where
        T: WLSSolver,
    {
        let (val, window) = Self::fit_single_point(
            x,
            y,
            0,
            window_size,
            use_robustness,
            robustness_weights,
            weights,
            weight_function,
            zero_weight_fallback,
        );
        y_smooth[0] = val;
        window
    }

    /// Main fitting loop: iterate through remaining points with delta-skipping
    /// and linear interpolation.
    /// Uses binary search (partition_point) instead of linear scan to find
    /// the next anchor point. This reduces the overhead from O(n) to O(log n)
    /// per anchor, providing significant speedup when delta is large.
    #[allow(clippy::too_many_arguments)]
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
    ) where
        T: WLSSolver,
    {
        let n = x.len();
        let mut last_fitted = 0usize;

        // Main loop: fit anchor points and interpolate between them
        while last_fitted < n - 1 {
            let cutpoint = x[last_fitted] + delta;

            // Binary search to find the first index where x > cutpoint
            // This is O(log n) instead of O(n) linear scan
            let next_idx =
                x[last_fitted + 1..].partition_point(|&xi| xi <= cutpoint) + last_fitted + 1;

            // Handle tied x-values: copy fitted value to all points with same x
            // Check the range [last_fitted+1, next_idx) for ties with last_fitted
            let mut tie_end = last_fitted;
            let x_last = x[last_fitted];
            for i in (last_fitted + 1)..next_idx.min(n) {
                if x[i] == x_last {
                    y_smooth[i] = y_smooth[last_fitted];
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
            };

            y_smooth[current] = ctx.fit().unwrap_or_else(|| y[current]);

            // Linearly interpolate between last fitted and current
            interpolate_gap(x, y_smooth, last_fitted, current);
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
            };

            y_smooth[final_idx] = ctx.fit().unwrap_or_else(|| y[final_idx]);
            interpolate_gap(x, y_smooth, last_fitted, final_idx);
        }
    }
}
