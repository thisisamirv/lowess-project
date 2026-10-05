//! Out-of-sample prediction for fitted Batch LOWESS models.
//!
//! This module holds the fitted-model state retained by `Predict::call()`
//! (Batch adapter only, opt-in via `.retain_model(true)`) and the logic that
//! evaluates the local WLS fit at arbitrary query points not in the training set.

// External dependencies
#[cfg(not(feature = "std"))]
use alloc::format;
#[cfg(not(feature = "std"))]
use alloc::string::ToString;
#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::fmt::Debug;
use num_traits::Float;
#[cfg(feature = "std")]
use std::string::ToString;
#[cfg(feature = "std")]
use std::vec;
#[cfg(feature = "std")]
use std::vec::Vec;

// Internal dependencies
use crate::algorithms::regression::{RegressionContext, WLSSolver, ZeroWeightFallback};
use crate::engine::executor::LowessExecutor;
use crate::engine::executor::LowessResult;
use crate::engine::executor::{ExtrapolationPolicy, PredictQuery, PredictState, RawPredictValues};
use crate::engine::validator::Validator;
use crate::evaluation::intervals::{
    BootstrapConfig, IntervalMethod, IntervalsBuilder, MIN_BOOTSTRAP_SAMPLES,
};
use crate::primitives::errors::LowessError;
use crate::primitives::window::Window;

// Options controlling a `Predict::call()` invocation.
#[derive(Debug, Clone)]
pub struct PredictBuilder<T> {
    // Include standard errors in the output.
    pub return_se: bool,

    // Confidence interval coverage level (e.g. `Some(0.95)`), or `None` to skip.
    pub confidence_intervals: Option<T>,

    // Prediction interval coverage level (e.g. `Some(0.95)`), or `None` to skip.
    pub prediction_intervals: Option<T>,
    pub bootstrap: Option<usize>,
    pub seed: Option<u64>,

    // Include the local WLS fit's derivative (slope) at each query point.
    pub return_derivative: bool,

    // Behavior for query points outside the training x-range.
    pub extrapolation: ExtrapolationPolicy,

    // Under `ExtrapolationPolicy::Linear`, the maximum allowed distance beyond the
    // training boundary before `predict()` errors with `LowessError::ExtrapolationTooFar`,
    // instead of returning the first-order Taylor extension's unbounded value. `None`
    // (default) preserves the original, uncapped behavior. Ignored under `Clamp`/`Error`.
    pub max_extrapolation_distance: Option<T>,

    // Maximum allowed distance to the farthest training point in a query's local window
    // before `predict()` errors with `LowessError::SparseNeighborhood`. Guards against the
    // `[min(x_train), max(x_train)]` range check's blind spot: a point can fall between
    // two clusters of training data (e.g. training x in [0,10] and [90,100], query at
    // x=50) and still pass that check, yet be far from any real training point. `None`
    // (default) preserves the original behavior of silently predicting there. Applies
    // regardless of `extrapolation`/whether the range check flagged the point as
    // out-of-range.
    pub max_neighbor_distance: Option<T>,

    // Set by `extrapolation(...)` when given an invalid string; surfaced by `build()`.
    pub pending_error: Option<LowessError>,
}

impl<T: Float> Default for PredictBuilder<T> {
    fn default() -> Self {
        Self {
            return_se: false,
            confidence_intervals: None,
            prediction_intervals: None,
            bootstrap: None,
            seed: None,
            return_derivative: false,
            extrapolation: ExtrapolationPolicy::default(),
            max_extrapolation_distance: None,
            max_neighbor_distance: None,
            pending_error: None,
        }
    }
}

impl<T: Float> PredictBuilder<T> {
    // Create a new `PredictBuilder` with default values, matching `Lowess::new()`'s
    // constructor-style entry point.
    pub fn new() -> Self {
        Self::default()
    }

    // Include standard errors in the output.
    pub fn return_se(mut self) -> Self {
        self.return_se = true;
        self
    }

    // Request a confidence interval at the given coverage level (e.g. `0.95`).
    pub fn confidence_intervals(mut self, level: T) -> Self {
        self.confidence_intervals = Some(level);
        self
    }

    // Request a prediction interval at the given coverage level (e.g. `0.95`).
    pub fn prediction_intervals(mut self, level: T) -> Self {
        self.prediction_intervals = Some(level);
        self
    }
    pub fn intervals(mut self, options: IntervalsBuilder<T>) -> Self {
        self.confidence_intervals = options.confidence;
        self.prediction_intervals = options.prediction;
        self.bootstrap = options.bootstrap;
        self
    }
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    // Include the local WLS fit's derivative (slope) at each query point.
    pub fn return_derivative(mut self) -> Self {
        self.return_derivative = true;
        self
    }

    // Select optional prediction output components: "se" and/or "derivative".
    // Unknown names are deferred to `build()` through the existing pending error path.
    pub fn outputs<I, S>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        for name in names {
            match name.as_ref() {
                "se" => self.return_se = true,
                "derivative" => self.return_derivative = true,
                other => {
                    self.pending_error = Some(LowessError::InvalidOption {
                        option: "predict_outputs",
                        value: other.to_string(),
                        valid: "se, derivative",
                    });
                }
            }
        }
        self
    }

    // Under `"linear"` extrapolation, the maximum allowed distance beyond the training
    // boundary before `call()` errors instead of returning an unbounded value.
    pub fn max_extrapolation_distance(mut self, distance: T) -> Self {
        self.max_extrapolation_distance = Some(distance);
        self
    }

    // Maximum allowed distance to the farthest training point in a query's local window
    // before `call()` errors, catching in-range-but-sparse query points.
    pub fn max_neighbor_distance(mut self, distance: T) -> Self {
        self.max_neighbor_distance = Some(distance);
        self
    }

    // Validates this configuration and produces a ready-to-call `PredictQuery`. Mandatory:
    // `PredictQuery` has no public constructor of its own, so `.build()` is the only way to
    // obtain one.
    pub fn build(self) -> Result<PredictQuery<T>, LowessError> {
        if let Some(e) = self.pending_error {
            return Err(e);
        }
        if let Some(level) = self.confidence_intervals {
            Validator::validate_interval_level(level)?;
        }
        if let Some(level) = self.prediction_intervals {
            Validator::validate_interval_level(level)?;
        }
        if let Some(n_boot) = self.bootstrap
            && n_boot < MIN_BOOTSTRAP_SAMPLES
        {
            return Err(LowessError::InvalidBootstrapSamples(n_boot));
        }
        Ok(PredictQuery {
            return_se: self.return_se,
            confidence_intervals: self.confidence_intervals,
            prediction_intervals: self.prediction_intervals,
            bootstrap: self.bootstrap.map(|n_boot| BootstrapConfig {
                n_boot,
                seed: self.seed,
            }),
            return_derivative: self.return_derivative,
            extrapolation: self.extrapolation,
            max_extrapolation_distance: self.max_extrapolation_distance,
            max_neighbor_distance: self.max_neighbor_distance,
        })
    }
}

// `Predict::new()` is the friendly, common-case entry point for `PredictBuilder`, matching
// `Lowess` being an alias for `LowessBuilder<T, BatchMode>`.
pub type Predict<T = f64> = PredictBuilder<T>;

impl<T: Float + WLSSolver + Debug + Send + Sync + 'static> PredictQuery<T> {
    // Evaluate `result` (a fitted Batch model) at out-of-sample x-values not in the
    // training set, per these options, similar to R's `predict(model, newdata)`.
    //
    // Requires `.retain_model(true)` on the builder that produced `result` (Batch adapter
    // only); returns `LowessError::PredictionUnavailable` otherwise.
    //
    // Reuses `fit()`'s own (possibly delta-interpolated) smoothed curve for its `y`
    // output, so it exactly reproduces `fit()`'s value at any x already in the training
    // set, regardless of `delta()`. A fresh local WLS fit is only computed when
    // `return_derivative`/`return_se`/an interval method/`max_neighbor_distance` needs
    // the actual regression slope or standard error.
    pub fn call(
        &self,
        result: &LowessResult<T>,
        new_x: &[T],
    ) -> Result<PredictOutput<T>, LowessError> {
        match &result.fit_state {
            Some(state) => predict_batch(state, new_x, self),
            None => Err(LowessError::PredictionUnavailable),
        }
    }
}

// Result of a `Predict::call()` invocation.
#[derive(Debug, Clone)]
pub struct PredictOutput<T> {
    // Predicted y-values, one per query point in `new_x`.
    pub y: Vec<T>,

    // Standard errors, if `return_se`/`confidence_intervals`/`prediction_intervals` was requested.
    pub standard_errors: Option<Vec<T>>,

    // Confidence interval bounds for the mean response, if `confidence_intervals` was set.
    pub confidence_lower: Option<Vec<T>>,
    pub confidence_upper: Option<Vec<T>>,

    // Prediction interval bounds for a new observation, if `prediction_intervals` was set.
    pub prediction_lower: Option<Vec<T>>,
    pub prediction_upper: Option<Vec<T>>,

    // Local WLS fit's derivative (slope) at each query point, if `return_derivative` was set.
    pub derivative: Option<Vec<T>>,
}

// Evaluate the fitted model at a single out-of-sample query point, returning
// `(y, slope, standard_error)`. `weights_scratch` (sized to `state.x.len()`) is reusable
// working space for kernel weights. `standard_error` is only computed when `need_se`.
//
// `pub` (hidden) so extension crates like fastLowess can reuse it for a parallel
// `PredictPassFn` implementation.
pub fn predict_one_full<T: Float + WLSSolver>(
    state: &PredictState<T>,
    x_query: T,
    weights_scratch: &mut [T],
    options: &PredictQuery<T>,
    need_se: bool,
) -> Result<(T, T, Option<T>), LowessError> {
    let n = state.x.len();
    if n == 0 {
        return Ok((T::zero(), T::zero(), None));
    }

    let min_x = state.train_min_x;
    let max_x = state.train_max_x;
    let out_of_range = x_query < min_x || x_query > max_x;

    if out_of_range && options.extrapolation == ExtrapolationPolicy::Error {
        return Err(LowessError::PredictOutOfRange {
            query: x_query.to_f64().unwrap_or(0.0),
            min: min_x.to_f64().unwrap_or(0.0),
            max: max_x.to_f64().unwrap_or(0.0),
        });
    }

    // For `Linear` extrapolation, evaluate the local fit AT the boundary (clamped point)
    // and extend it using that boundary fit's own slope; otherwise evaluate directly at
    // `x_query` (the `Clamp` policy's window-clamping happens naturally via `Window`).
    let extrapolate_linear = out_of_range && options.extrapolation == ExtrapolationPolicy::Linear;
    let eval_point = if extrapolate_linear {
        if x_query < min_x { min_x } else { max_x }
    } else {
        x_query
    };

    if extrapolate_linear && let Some(max_dist) = options.max_extrapolation_distance {
        let dist = (x_query - eval_point).abs();
        if dist > max_dist {
            return Err(LowessError::ExtrapolationTooFar {
                distance: dist.to_f64().unwrap_or(0.0),
                max_distance: max_dist.to_f64().unwrap_or(0.0),
            });
        }
    }

    let need_gradient = options.return_derivative || extrapolate_linear;

    // A requested neighbor-sparsity check needs a real local window to measure against,
    // so it must bypass the y_smooth-interpolation fast path below even when nothing
    // else would.
    let need_neighborhood_check = options.max_neighbor_distance.is_some();

    // Fast path: no exact regression slope/leverage needed - linearly interpolate the
    // already-fitted `y_smooth` curve at `eval_point` instead of running a fresh local WLS
    // fit. This is exactly how `fit()` itself fills in delta-skipped points, so it matches
    // `fit()`'s own output at training points regardless of `delta()`.
    if !need_gradient && !need_se && !need_neighborhood_check {
        return Ok((
            interpolate_y_smooth(&state.x, &state.y_smooth, eval_point),
            T::zero(),
            None,
        ));
    }

    let seed = Window::locate(&state.x, eval_point);
    let mut window = Window::initialize(seed, state.window_size, n);
    window.recenter_at(&state.x, eval_point, n);

    if let Some(max_dist) = options.max_neighbor_distance {
        let bandwidth = T::max(
            eval_point - state.x[window.left],
            state.x[window.right] - eval_point,
        );
        if bandwidth > max_dist {
            return Err(LowessError::SparseNeighborhood {
                distance: bandwidth.to_f64().unwrap_or(0.0),
                max_distance: max_dist.to_f64().unwrap_or(0.0),
            });
        }
    }

    let mut ctx = RegressionContext {
        x: &state.x,
        y: &state.y,
        idx: seed,
        window,
        use_robustness: true,
        robustness_weights: &state.robustness_weights,
        weights: weights_scratch,
        weight_function: state.weight_function,
        zero_weight_fallback: ZeroWeightFallback::from_u8(state.zero_weight_fallback),
        custom_weights: state.custom_weights.as_deref(),
    };

    // Only `slope` is used below; the value comes from the already-fitted curve instead
    // (see `interpolate_y_smooth` above), so it matches `fit()`'s own output.
    let (_, slope) = ctx.predict_at(eval_point).unwrap_or((T::zero(), T::zero()));

    let boundary_y = interpolate_y_smooth(&state.x, &state.y_smooth, eval_point);
    let y = if extrapolate_linear {
        boundary_y + slope * (x_query - eval_point)
    } else {
        boundary_y
    };

    let se = if need_se {
        Some(IntervalMethod::compute_se_at_query_with_support(
            &state.x,
            &state.y,
            &state.y_smooth,
            &window,
            eval_point,
            &state.robustness_weights,
            state.custom_weights.as_deref(),
            state.weight_function.support().is_none(),
            &|u| state.weight_function.compute_weight(u),
        ))
    } else {
        None
    };

    Ok((y, slope, se))
}

// Linearly interpolate the already-fitted `y_smooth` curve at an arbitrary point within
// (or exactly at) `x`'s range. `x` is sorted; `query` should already be clamped into
// `[x[0], x[n - 1]]` by the caller. Since `fit()` fills every delta-skipped point by
// linearly interpolating between its neighboring exact fits, any two adjacent entries of
// `y_smooth` already lie on the correct line for the segment between them - so this always
// reproduces `fit()`'s own value at (or between) training points, independent of `delta()`.
fn interpolate_y_smooth<T: Float>(x: &[T], y_smooth: &[T], query: T) -> T {
    let n = x.len();
    if n == 0 {
        return T::zero();
    }
    let right = Window::locate(x, query);
    if right == 0 || x[right] <= query {
        return y_smooth[right];
    }
    let left = right - 1;
    let span = x[right] - x[left];
    if span <= T::zero() {
        return y_smooth[left];
    }
    let t = (query - x[left]) / span;
    y_smooth[left] + t * (y_smooth[right] - y_smooth[left])
}

// Serial fallback for `predict_batch` when no `custom_predict_pass` is set.
fn predict_batch_serial<T: Float + WLSSolver>(
    state: &PredictState<T>,
    new_x: &[T],
    options: &PredictQuery<T>,
    need_se: bool,
) -> RawPredictValues<T> {
    let mut scratch = vec![T::zero(); state.x.len().max(1)];
    let mut y = Vec::with_capacity(new_x.len());
    let mut derivative = options
        .return_derivative
        .then(|| Vec::with_capacity(new_x.len()));
    let mut se = need_se.then(|| Vec::with_capacity(new_x.len()));

    for &q in new_x {
        let (yi, slope, sei) = predict_one_full(state, q, &mut scratch, options, need_se)?;
        y.push(yi);
        if let Some(d) = derivative.as_mut() {
            d.push(slope);
        }
        if let Some(s) = se.as_mut() {
            s.push(sei.unwrap_or(T::zero()));
        }
    }

    Ok((y, derivative, se))
}

// Evaluate the fitted model at a batch of out-of-sample query points, per `options`.
// Delegates the per-point work to `state.custom_predict_pass` if set (e.g. fastLowess's
// Rayon-parallel implementation), otherwise evaluates serially; either way, the shared
// confidence/prediction interval math is applied here. `new_x` need not be sorted - each
// query point is independently located via binary search.
pub fn predict_batch<T: Float + WLSSolver + Debug + Send + Sync + 'static>(
    state: &PredictState<T>,
    new_x: &[T],
    options: &PredictQuery<T>,
) -> Result<PredictOutput<T>, LowessError> {
    for (i, &val) in new_x.iter().enumerate() {
        if !val.is_finite() {
            return Err(LowessError::InvalidNumericValue(format!(
                "new_x[{}]={}",
                i,
                val.to_f64().unwrap_or(f64::NAN)
            )));
        }
    }

    let need_se = options.bootstrap.is_none()
        && (options.return_se
            || options.confidence_intervals.is_some()
            || options.prediction_intervals.is_some());

    let (y, derivative, se) = if let Some(pass) = state.custom_predict_pass {
        pass(state, new_x, options, need_se)?
    } else {
        predict_batch_serial(state, new_x, options, need_se)?
    };

    if let Some(bootstrap) = options.bootstrap {
        let retained = state
            .bootstrap_state
            .as_ref()
            .ok_or(LowessError::PredictionUnavailable)?;
        for level in [options.confidence_intervals, options.prediction_intervals]
            .into_iter()
            .flatten()
        {
            IntervalMethod::<T>::approximate_z_score(level)
                .map_err(|_| LowessError::InvalidIntervals(level.to_f64().unwrap_or(0.0)))?;
        }
        let method = IntervalMethod {
            level: options
                .confidence_intervals
                .or(options.prediction_intervals)
                .unwrap_or_else(|| T::from(0.95).unwrap()),
            prediction_level: options.prediction_intervals,
            confidence: options.confidence_intervals.is_some(),
            prediction: options.prediction_intervals.is_some(),
            se: true,
        };
        let intervals = bootstrap.compute_at(
            &method,
            &retained.smoothed,
            &retained.residuals,
            new_x.len(),
            |replicates| {
                replicates
                    .iter()
                    .map(|sample| {
                        let refit = LowessExecutor::run_with_config(
                            &retained.x,
                            sample,
                            retained.refit_config.clone(),
                        )?;
                        let refit_state = refit
                            .predict_state
                            .ok_or(LowessError::PredictionUnavailable)?;
                        let (predictions, _, _) =
                            predict_batch_serial(&refit_state, new_x, options, false)?;
                        Ok(predictions)
                    })
                    .collect::<Result<Vec<_>, _>>()
            },
        )?;
        return Ok(PredictOutput {
            y,
            standard_errors: Some(intervals.std_errors),
            confidence_lower: intervals.confidence_lower,
            confidence_upper: intervals.confidence_upper,
            prediction_lower: intervals.prediction_lower,
            prediction_upper: intervals.prediction_upper,
            derivative,
        });
    }

    let (confidence_lower, confidence_upper) = if let Some(level) = options.confidence_intervals {
        let se_vals = se.as_deref().unwrap_or(&[]);
        let z = IntervalMethod::<T>::approximate_z_score(level)
            .map_err(|_| LowessError::InvalidIntervals(level.to_f64().unwrap_or(0.0)))?;
        let lower: Vec<T> = y.iter().zip(se_vals).map(|(&yi, &s)| yi - z * s).collect();
        let upper: Vec<T> = y.iter().zip(se_vals).map(|(&yi, &s)| yi + z * s).collect();
        (Some(lower), Some(upper))
    } else {
        (None, None)
    };

    let (prediction_lower, prediction_upper) = if let Some(level) = options.prediction_intervals {
        let se_vals = se.as_deref().unwrap_or(&[]);
        let z = IntervalMethod::<T>::approximate_z_score(level)
            .map_err(|_| LowessError::InvalidIntervals(level.to_f64().unwrap_or(0.0)))?;
        let rsd_sq = state.residual_sd * state.residual_sd;
        let lower: Vec<T> = y
            .iter()
            .zip(se_vals)
            .map(|(&yi, &s)| yi - z * (s * s + rsd_sq).sqrt())
            .collect();
        let upper: Vec<T> = y
            .iter()
            .zip(se_vals)
            .map(|(&yi, &s)| yi + z * (s * s + rsd_sq).sqrt())
            .collect();
        (Some(lower), Some(upper))
    } else {
        (None, None)
    };

    Ok(PredictOutput {
        y,
        standard_errors: se,
        confidence_lower,
        confidence_upper,
        prediction_lower,
        prediction_upper,
        derivative,
    })
}
