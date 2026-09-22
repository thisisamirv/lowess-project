//! Online adapter for incremental LOWESS smoothing.
//!
//! This module provides the online (incremental) execution adapter for LOWESS
//! smoothing. It maintains a sliding window of recent observations and produces
//! smoothed values for new points as they arrive.
// ## srrstats Compliance
//
// @srrstats {G1.6} Sliding window for real-time incremental updates.
// @srrstats {G2.1} Configurable minimum points before smoothing activates.

// External dependencies
#[cfg(not(feature = "std"))]
use alloc::collections::VecDeque;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::fmt::Debug;
use num_traits::Float;
#[cfg(feature = "std")]
use std::collections::VecDeque;

// Internal dependencies
use crate::adapters::defaults::*;
use crate::algorithms::defaults::*;
use crate::algorithms::regression::{WLSSolver, ZeroWeightFallback};
use crate::algorithms::robustness::RobustnessMethod;
use crate::engine::executor::{
    CVPassFn, DerivativePassFn, FitPassFn, IntervalPassFn, SmoothPassFn,
};
use crate::engine::executor::{LowessConfig, LowessExecutor};
use crate::engine::validator::{MissingPolicy, Validator};
use crate::evaluation::intervals::IntervalMethod;
use crate::math::boundary::BoundaryPolicy;
use crate::math::defaults::*;
use crate::math::kernel::WeightFunction;
use crate::math::scaling::ScalingMethod;
use crate::primitives::buffer::{OnlineBuffer, VecExt};
use crate::primitives::errors::LowessError;

// Update mode for online LOWESS processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UpdateMode {
    // Recompute all points in the window from scratch.
    Full,

    // Optimized incremental update.
    #[default]
    Incremental,
}

// Builder for online LOWESS processor.
#[derive(Debug, Clone)]
pub struct OnlineLowessBuilder<T: Float> {
    // Window capacity (maximum number of points to retain)
    pub window_capacity: usize,

    // Minimum points before smoothing starts
    pub min_points: usize,

    // Smoothing fraction (span)
    pub fraction: T,

    // Delta parameter for interpolation optimization
    pub delta: T,

    // Number of robustness iterations
    pub iterations: usize,

    // Convergence tolerance for early stopping (None = disabled)
    pub auto_converge: Option<T>,

    // Kernel weight function
    pub weight_function: WeightFunction,

    // Update mode for incremental processing
    pub update_mode: UpdateMode,

    // Robustness method
    pub robustness_method: RobustnessMethod,

    // Policy for handling zero-weight neighborhoods
    pub zero_weight_fallback: ZeroWeightFallback,

    // Policy for handling data boundaries
    pub boundary_policy: BoundaryPolicy,

    // Scaling method for robust scale estimation (MAR/MAD)
    pub scaling_method: ScalingMethod,

    // Whether to return robustness weights
    pub return_robustness_weights: bool,

    // Include the per-point local fit derivative (slope) in the output.
    pub return_derivative: bool,

    // Interval estimation method (standard error only - `Full` update mode only;
    // `OnlineOutput` has no confidence/prediction bounds, just `standard_error`).
    pub interval_type: Option<IntervalMethod<T>>,

    // Policy for handling non-finite (NaN/Inf) values in input data
    pub missing: MissingPolicy,

    // Deferred error from adapter conversion
    pub deferred_error: Option<LowessError>,

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

    // Tracks if any parameter was set multiple times (for validation)
    pub(crate) duplicate_param: Option<&'static str>,
}

impl<T: Float> Default for OnlineLowessBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> OnlineLowessBuilder<T> {
    // Create a new online LOWESS builder with default parameters.
    fn new() -> Self {
        Self {
            window_capacity: DEFAULT_ONLINE_WINDOW_CAPACITY,
            min_points: DEFAULT_ONLINE_MIN_POINTS,
            fraction: T::from(DEFAULT_FRACTION).unwrap(),
            delta: T::from(DEFAULT_DELTA).unwrap(),
            iterations: DEFAULT_ONLINE_ITERATIONS,
            weight_function: DEFAULT_WEIGHT_FUNCTION_ENUM,
            update_mode: DEFAULT_ONLINE_UPDATE_MODE_ENUM,
            robustness_method: DEFAULT_ROBUSTNESS_METHOD_ENUM,
            zero_weight_fallback: DEFAULT_ZERO_WEIGHT_FALLBACK_ENUM,
            boundary_policy: DEFAULT_BOUNDARY_POLICY_ENUM,
            scaling_method: DEFAULT_SCALING_METHOD_ENUM,
            return_robustness_weights: DEFAULT_RETURN_ROBUSTNESS_WEIGHTS,
            return_derivative: DEFAULT_RETURN_DERIVATIVE,
            interval_type: None,
            auto_converge: default_auto_converge(),
            missing: DEFAULT_MISSING_POLICY_ENUM,
            deferred_error: None,
            custom_smooth_pass: None,
            custom_cv_pass: None,
            custom_interval_pass: None,
            custom_derivative_pass: None,
            custom_fit_pass: None,
            duplicate_param: None,
        }
    }

    // Build the online processor.
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

        // Validate that return_se()/confidence_intervals()/prediction_intervals() is
        // only combined with update_mode("full")
        Validator::validate_online_se_update_mode(self.interval_type, self.update_mode)?;

        // Validate that robustness iterations are only combined with update_mode("full")
        Validator::validate_online_iterations_update_mode(self.iterations, self.update_mode)?;

        let capacity = self.window_capacity;
        Ok(OnlineLowess {
            config: self,
            window_x: VecDeque::with_capacity(capacity),
            window_y: VecDeque::with_capacity(capacity),
            buffer: OnlineBuffer::with_capacity(capacity),
        })
    }
}

// Result of a single online update.
#[derive(Debug, Clone, PartialEq)]
pub struct OnlineOutput<T> {
    // Smoothed value for the latest point
    pub y: T,

    // Standard error (if computed)
    pub standard_error: Option<T>,

    // Residual (raw input y minus this output's y)
    pub residual: Option<T>,

    // Confidence interval bounds around the mean response for the latest point (`Full`
    // update mode only, via `.confidence_intervals(level)`).
    pub confidence_lower: Option<T>,
    pub confidence_upper: Option<T>,

    // Prediction interval bounds for a new observation at the latest point (`Full`
    // update mode only, via `.prediction_intervals(level)`).
    pub prediction_lower: Option<T>,
    pub prediction_upper: Option<T>,

    // Robustness weight for the latest point (if computed)
    pub robustness_weight: Option<T>,

    // Number of robustness iterations actually performed (if tracked).
    pub iterations_used: Option<usize>,

    // Local fit derivative (slope) for the latest point (if requested).
    pub derivative: Option<T>,
}

// Online LOWESS processor for streaming data.
pub struct OnlineLowess<T: Float> {
    config: OnlineLowessBuilder<T>,
    window_x: VecDeque<T>,
    window_y: VecDeque<T>,
    // Pre-allocated scratch buffers for smoothing
    buffer: OnlineBuffer<T>,
}

impl<T: Float + WLSSolver + Debug + Send + Sync + 'static> OnlineLowess<T> {
    // Add a new point and get its smoothed value.
    pub fn add_point(&mut self, x: T, y: T) -> Result<Option<OnlineOutput<T>>, LowessError> {
        // Validate new point according to the configured missing-value policy
        match self.config.missing {
            MissingPolicy::Error => {
                Validator::validate_scalar(x, "x")?;
                Validator::validate_scalar(y, "y")?;
            }
            MissingPolicy::Drop => {
                if !x.is_finite() || !y.is_finite() {
                    return Ok(None);
                }
            }
        }

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

            let (smoothed, slope) = if x1 != x0 {
                let slope = (y1 - y0) / (x1 - x0);
                (y0 + slope * (x - x0), slope)
            } else {
                // Identical x: use mean for stability
                ((y0 + y1) / T::from(2.0).unwrap(), T::zero())
            };

            let residual = y - smoothed;

            return Ok(Some(OnlineOutput {
                y: smoothed,
                standard_error: None,
                residual: Some(residual),
                confidence_lower: None,
                confidence_upper: None,
                prediction_lower: None,
                prediction_upper: None,
                robustness_weight: Some(T::one()),
                iterations_used: Some(0),
                derivative: self.config.return_derivative.then_some(slope),
            }));
        }

        // Smooth using LOWESS for windows of size >= 3
        let zero_flag = self.config.zero_weight_fallback.to_u8();

        // Choose update strategy based on configuration
        let (smoothed, std_err, ci_bounds, rob_weight, iterations_used, derivative) =
            match self.config.update_mode {
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

                    let (smoothed_val, slope, _) = LowessExecutor::fit_single_point(
                        x_vec,
                        y_vec,
                        n - 1, // Latest point
                        window_size,
                        false, // No robustness for single point
                        &self.buffer.robustness_weights,
                        &mut self.buffer.weights,
                        self.config.weight_function,
                        self.config.zero_weight_fallback,
                        None, // custom_weights not used in online incremental mode
                    );

                    let deriv = self.config.return_derivative.then_some(slope);
                    (
                        smoothed_val,
                        None,
                        (None, None, None, None),
                        Some(T::one()),
                        None,
                        deriv,
                    )
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
                        auto_converge: self.config.auto_converge,
                        cv_fractions: None,
                        cv_kind: None,
                        return_variance: self.config.interval_type,
                        cv_seed: None,
                        return_derivative: self.config.return_derivative,
                        // ++++++++++++++++++++++++++++++++++++++
                        // +               DEV                  +
                        // ++++++++++++++++++++++++++++++++++++++
                        custom_smooth_pass: self.config.custom_smooth_pass,
                        custom_cv_pass: self.config.custom_cv_pass,
                        custom_interval_pass: self.config.custom_interval_pass,
                        custom_derivative_pass: self.config.custom_derivative_pass,
                        custom_fit_pass: self.config.custom_fit_pass,
                        parallel: false,
                        backend: None,
                        delegate_boundary_handling: false,
                        custom_weights: None,
                        retain_model: false,
                        custom_predict_pass: None,
                    };

                    let result = LowessExecutor::run_with_config(x_vec, y_vec, config.clone())?;
                    let iterations_used_val = result.iterations;
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
                    let deriv = result.derivative.as_ref().and_then(|v| v.last().copied());

                    // Confidence/prediction interval bounds for the latest point, computed
                    // from the whole window's smoothed values/SE/residuals the same way
                    // Batch does (`IntervalMethod::compute_intervals`), then taking the
                    // last element - the executor itself only produces plain std_errors.
                    let ci_bounds = if let (Some(method), Some(se)) =
                        (&self.config.interval_type, se_vec.as_ref())
                    {
                        let residuals: Vec<T> = y_vec
                            .iter()
                            .zip(smoothed_vec.iter())
                            .map(|(&yi, &si)| yi - si)
                            .collect();
                        let (cl, cu, pl, pu) =
                            method.compute_intervals(&smoothed_vec, se, &residuals)?;
                        (
                            cl.as_ref().and_then(|v| v.last().copied()),
                            cu.as_ref().and_then(|v| v.last().copied()),
                            pl.as_ref().and_then(|v| v.last().copied()),
                            pu.as_ref().and_then(|v| v.last().copied()),
                        )
                    } else {
                        (None, None, None, None)
                    };

                    (
                        smoothed_val,
                        std_err,
                        ci_bounds,
                        rob_weight,
                        iterations_used_val,
                        deriv,
                    )
                }
            };

        let residual = y - smoothed;

        let (confidence_lower, confidence_upper, prediction_lower, prediction_upper) = ci_bounds;

        Ok(Some(OnlineOutput {
            y: smoothed,
            standard_error: std_err,
            residual: Some(residual),
            confidence_lower,
            confidence_upper,
            prediction_lower,
            prediction_upper,
            robustness_weight: rob_weight,
            iterations_used,
            derivative,
        }))
    }

    // Get the current window size.
    pub fn window_size(&self) -> usize {
        self.window_x.len()
    }

    // Clear the window.
    pub fn reset(&mut self) {
        self.window_x.clear();
        self.window_y.clear();
        self.buffer.clear();
    }
}
