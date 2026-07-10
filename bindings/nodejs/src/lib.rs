//! Node.js bindings for fastLowess using N-API.

use napi::bindgen_prelude::*;
use napi_derive::napi;

use ::fastLowess::internals::adapters::online::ParallelOnlineLowess;
use ::fastLowess::internals::adapters::streaming::ParallelStreamingLowess;
use ::fastLowess::internals::api::LowessBuilder;
use ::fastLowess::internals::binding_support;
use ::fastLowess::prelude::LowessResult as InnerLowessResult;

fn to_napi_error(err: binding_support::BindingError) -> Error {
    let status = match err.category {
        binding_support::BindingErrorCategory::InvalidArg => Status::InvalidArg,
        binding_support::BindingErrorCategory::Runtime => Status::GenericFailure,
    };
    Error::new(status, err.message)
}

fn map_invalid_arg<T, E: ToString>(result: std::result::Result<T, E>) -> Result<T> {
    binding_support::map_invalid_arg(result).map_err(to_napi_error)
}

fn map_runtime<T, E: ToString>(result: std::result::Result<T, E>) -> Result<T> {
    binding_support::map_runtime(result).map_err(to_napi_error)
}

/// Diagnostic statistics for the LOWESS fit.
#[napi(object)]
pub struct Diagnostics {
    /// Root Mean Squared Error.
    pub rmse: f64,
    /// Mean Absolute Error.
    pub mae: f64,
    /// R-squared (coefficient of determination).
    #[napi(js_name = "r_squared")]
    pub r_squared: f64,
    /// Akaike Information Criterion (if computed).
    pub aic: Option<f64>,
    /// Corrected AIC (if computed).
    pub aicc: Option<f64>,
    /// Effective degrees of freedom (if computed).
    #[napi(js_name = "effective_df")]
    pub effective_df: Option<f64>,
    /// Residual standard deviation.
    #[napi(js_name = "residual_sd")]
    pub residual_sd: f64,
}

/// Result of a LOWESS fit.
#[napi]
pub struct LowessResultObj {
    inner: InnerLowessResult<f64>,
}

#[napi]
impl LowessResultObj {
    /// Get the sorted x values.
    #[napi(getter)]
    pub fn get_x(&self) -> Float64Array {
        Float64Array::from(self.inner.x.as_slice())
    }

    /// Get the smoothed y values.
    #[napi(getter)]
    pub fn get_y(&self) -> Float64Array {
        Float64Array::from(self.inner.y.as_slice())
    }

    /// Get residuals (if requested).
    #[napi(getter)]
    pub fn get_residuals(&self) -> Option<Float64Array> {
        self.inner
            .residuals
            .as_ref()
            .map(|v| Float64Array::from(v.as_slice()))
    }

    /// Get standard errors (if requested/computed).
    #[napi(getter, js_name = "standard_errors")]
    pub fn get_standard_errors(&self) -> Option<Float64Array> {
        self.inner
            .standard_errors
            .as_ref()
            .map(|v| Float64Array::from(v.as_slice()))
    }

    /// Get lower confidence bounds (if requested).
    #[napi(getter, js_name = "confidence_lower")]
    pub fn get_confidence_lower(&self) -> Option<Float64Array> {
        self.inner
            .confidence_lower
            .as_ref()
            .map(|v| Float64Array::from(v.as_slice()))
    }

    /// Get upper confidence bounds (if requested).
    #[napi(getter, js_name = "confidence_upper")]
    pub fn get_confidence_upper(&self) -> Option<Float64Array> {
        self.inner
            .confidence_upper
            .as_ref()
            .map(|v| Float64Array::from(v.as_slice()))
    }

    /// Get lower prediction bounds (if requested).
    #[napi(getter, js_name = "prediction_lower")]
    pub fn get_prediction_lower(&self) -> Option<Float64Array> {
        self.inner
            .prediction_lower
            .as_ref()
            .map(|v| Float64Array::from(v.as_slice()))
    }

    /// Get upper prediction bounds (if requested).
    #[napi(getter, js_name = "prediction_upper")]
    pub fn get_prediction_upper(&self) -> Option<Float64Array> {
        self.inner
            .prediction_upper
            .as_ref()
            .map(|v| Float64Array::from(v.as_slice()))
    }

    /// Get robustness weights (if requested).
    #[napi(getter, js_name = "robustness_weights")]
    pub fn get_robustness_weights(&self) -> Option<Float64Array> {
        self.inner
            .robustness_weights
            .as_ref()
            .map(|v| Float64Array::from(v.as_slice()))
    }

    /// Get diagnostics (if requested).
    #[napi(getter)]
    pub fn get_diagnostics(&self) -> Option<Diagnostics> {
        self.inner.diagnostics.as_ref().map(|d| Diagnostics {
            rmse: d.rmse,
            mae: d.mae,
            r_squared: d.r_squared,
            aic: d.aic,
            aicc: d.aicc,
            effective_df: d.effective_df,
            residual_sd: d.residual_sd,
        })
    }

    /// Get cross-validation scores (if CV was performed).
    #[napi(getter, js_name = "cv_scores")]
    pub fn get_cv_scores(&self) -> Option<Float64Array> {
        self.inner
            .cv_scores
            .as_ref()
            .map(|v| Float64Array::from(v.as_slice()))
    }

    /// Get the fraction used for smoothing.
    #[napi(getter, js_name = "fraction_used")]
    pub fn get_fraction_used(&self) -> f64 {
        self.inner.fraction_used
    }

    /// Get the number of iterations performed.
    #[napi(getter, js_name = "iterations_used")]
    pub fn get_iterations_used(&self) -> Option<u32> {
        self.inner.iterations_used.map(|i| i as u32)
    }
}

/// Configuration options for LOWESS smoothing.
#[napi(object)]
pub struct SmoothOptions {
    /// Smoothing fraction (0 < fraction <= 1). Default: 0.67.
    pub fraction: Option<f64>,
    /// Number of robustness iterations. Default: 3.
    pub iterations: Option<u32>,
    /// Delta for interpolation speedup. Default: NaN (auto).
    /// Set to 0.0 to disable interpolation.
    pub delta: Option<f64>,
    /// Weight function ("tricube", "epanechnikov", "gaussian", "uniform", "biweight", "triangle", "cosine"). Default: "tricube".
    #[napi(js_name = "weight_function")]
    pub weight_function: Option<String>,
    /// Robustness method ("bisquare", "huber", "talwar"). Default: "bisquare".
    #[napi(js_name = "robustness_method")]
    pub robustness_method: Option<String>,
    /// Fallback strategy when weights are zero ("use_local_mean", "return_original", "return_none"). Default: "use_local_mean".
    #[napi(js_name = "zero_weight_fallback")]
    pub zero_weight_fallback: Option<String>,
    /// Boundary handling ("extend", "reflect", "zero", "noboundary"). Default: "extend".
    #[napi(js_name = "boundary_policy")]
    pub boundary_policy: Option<String>,
    /// Scaling method ("mad", "mar", "mean"). Default: "mad".
    #[napi(js_name = "scaling_method")]
    pub scaling_method: Option<String>,
    /// Auto-convergence tolerance. Default: None.
    #[napi(js_name = "auto_converge")]
    pub auto_converge: Option<f64>,
    /// Return residuals in result. Default: false.
    #[napi(js_name = "return_residuals")]
    pub return_residuals: Option<bool>,
    /// Return robustness weights in result. Default: false.
    #[napi(js_name = "return_robustness_weights")]
    pub return_robustness_weights: Option<bool>,
    /// Return diagnostics (RMSE, etc.). Default: false.
    #[napi(js_name = "return_diagnostics")]
    pub return_diagnostics: Option<bool>,
    /// Calculate confidence intervals (e.g., 0.95). Default: None.
    #[napi(js_name = "confidence_intervals")]
    pub confidence_intervals: Option<f64>,
    /// Calculate prediction intervals. Default: None.
    #[napi(js_name = "prediction_intervals")]
    pub prediction_intervals: Option<f64>,
    /// Fractions to use for cross-validation.
    #[napi(js_name = "cv_fractions")]
    pub cv_fractions: Option<Vec<f64>>,
    /// CV method ("loocv", "kfold"). Default: "kfold".
    #[napi(js_name = "cv_method")]
    pub cv_method: Option<String>,
    /// Number of folds for K-Fold CV. Default: 5.
    #[napi(js_name = "cv_k")]
    pub cv_k: Option<u32>,
    /// Random seed for reproducible K-Fold cross-validation. Default: None.
    #[napi(js_name = "cv_seed")]
    pub cv_seed: Option<i64>,
    /// Compute standard errors. Default: false.
    #[napi(js_name = "return_se")]
    pub return_se: Option<bool>,
    /// Enable parallel execution. Default: true.
    pub parallel: Option<bool>,
}

/// Build a LowessBuilder from an optional SmoothOptions, applying all fields.
fn options_to_builder(opts: Option<&SmoothOptions>) -> Result<LowessBuilder<f64>> {
    let mut builder = LowessBuilder::<f64>::new();
    if let Some(opts) = opts {
        let configured_builder = map_invalid_arg(binding_support::apply_builder_options(
            builder,
            binding_support::BuilderOptionSet {
                fraction: opts.fraction,
                iterations: opts.iterations.map(|v| v as usize),
                delta: opts.delta,
                weight_function: opts.weight_function.as_deref(),
                robustness_method: opts.robustness_method.as_deref(),
                zero_weight_fallback: opts.zero_weight_fallback.as_deref(),
                boundary_policy: opts.boundary_policy.as_deref(),
                scaling_method: opts.scaling_method.as_deref(),
                auto_converge: opts.auto_converge,
                return_residuals: opts.return_residuals.unwrap_or(false),
                return_robustness_weights: opts.return_robustness_weights.unwrap_or(false),
                return_diagnostics: opts.return_diagnostics.unwrap_or(false),
                return_se: opts.return_se.unwrap_or(false),
                confidence_intervals: opts.confidence_intervals,
                prediction_intervals: opts.prediction_intervals,
                parallel: opts.parallel,
                chunk_size: None,
                overlap: None,
                merge_strategy: None,
                window_capacity: None,
                min_points: None,
                update_mode: None,
                cv_fractions: opts.cv_fractions.as_deref(),
                cv_method: opts.cv_method.as_deref(),
                cv_k: opts.cv_k.map(|v| v as usize),
                cv_seed: opts.cv_seed.map(|s| s as u64),
            },
        ))?;
        builder = configured_builder;
    }
    Ok(builder)
}

/// Batch LOWESS smoothing.
#[napi]
pub struct Lowess {
    options: Option<SmoothOptions>,
}

#[napi]
impl Lowess {
    /// Create a new batch LOWESS smoother.
    #[napi(constructor)]
    pub fn new(options: Option<SmoothOptions>) -> Self {
        Self { options }
    }

    /// Fit the model.
    #[napi]
    pub fn fit(
        &self,
        x: Float64Array,
        y: Float64Array,
        custom_weights: Option<Vec<f64>>,
    ) -> Result<LowessResultObj> {
        let builder = self.create_builder()?;
        let model = map_runtime(binding_support::build_batch(builder, custom_weights))?;
        let result = map_runtime(model.fit(x.as_ref(), y.as_ref()))?;
        Ok(LowessResultObj { inner: result })
    }

    /// Fit the model asynchronously.
    #[napi(js_name = "fit_async")]
    pub fn fit_async(
        &self,
        x: Float64Array,
        y: Float64Array,
        custom_weights: Option<Vec<f64>>,
    ) -> Result<AsyncTask<LowessTask>> {
        let mut builder = self.create_builder()?;
        if let Some(cw) = custom_weights {
            builder = builder.custom_weights(cw);
        }
        let x_vec = x.as_ref().to_vec();
        let y_vec = y.as_ref().to_vec();

        Ok(AsyncTask::new(LowessTask {
            builder,
            x: x_vec,
            y: y_vec,
        }))
    }

    fn create_builder(&self) -> Result<LowessBuilder<f64>> {
        options_to_builder(self.options.as_ref())
    }
}

pub struct LowessTask {
    builder: LowessBuilder<f64>,
    x: Vec<f64>,
    y: Vec<f64>,
}

impl Task for LowessTask {
    type Output = InnerLowessResult<f64>;
    type JsValue = LowessResultObj;

    fn compute(&mut self) -> Result<Self::Output> {
        let model = map_runtime(binding_support::build_batch(self.builder.clone(), None))?;
        map_runtime(model.fit(&self.x, &self.y))
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(LowessResultObj { inner: output })
    }
}

/// Configuration options for streaming processing.
#[napi(object)]
pub struct StreamingOptions {
    /// Size of each data chunk. Default: 5000.
    #[napi(js_name = "chunk_size")]
    pub chunk_size: Option<u32>,
    /// Header/footer overlap size. Default: chunk_size / 10, min. 1.
    pub overlap: Option<u32>,
    /// Strategy for merging chunk overlaps. Default: "weighted_average".
    #[napi(js_name = "merge_strategy")]
    pub merge_strategy: Option<String>,
}

/// Streaming LOWESS smoother for large datasets.
#[napi]
pub struct StreamingLowess {
    inner: ParallelStreamingLowess<f64>,
}

#[napi]
impl StreamingLowess {
    /// Create a new streaming LOWESS smoother.
    #[napi(constructor)]
    pub fn new(
        options: Option<SmoothOptions>,
        streaming_opts: Option<StreamingOptions>,
    ) -> Result<Self> {
        let builder = options_to_builder(options.as_ref())?;

        let (chunk_size, overlap, merge_strategy) = match streaming_opts {
            Some(s) => (
                s.chunk_size.map(|v| v as usize),
                s.overlap.map(|v| v as usize),
                s.merge_strategy,
            ),
            None => (None, None, None),
        };

        let model = map_runtime(binding_support::build_streaming(
            builder,
            chunk_size,
            overlap,
            merge_strategy.as_deref(),
        ))?;

        Ok(StreamingLowess { inner: model })
    }

    /// Process a chunk of data.
    #[napi(js_name = "process_chunk")]
    pub fn process_chunk(&mut self, x: Float64Array, y: Float64Array) -> Result<LowessResultObj> {
        let result: InnerLowessResult<f64> =
            map_runtime(self.inner.process_chunk(x.as_ref(), y.as_ref()))?;
        Ok(LowessResultObj { inner: result })
    }

    /// Finalize the stream and return remaining data.
    #[napi]
    pub fn finalize(&mut self) -> Result<LowessResultObj> {
        let result: InnerLowessResult<f64> = map_runtime(self.inner.finalize())?;
        Ok(LowessResultObj { inner: result })
    }
}

/// Configuration options for online processing.
#[napi(object)]
pub struct OnlineOptions {
    /// Maximum number of points to keep in the window. Default: 1000.
    #[napi(js_name = "window_capacity")]
    pub window_capacity: Option<u32>,
    /// Minimum points required before smoothing starts. Default: 3.
    #[napi(js_name = "min_points")]
    pub min_points: Option<u32>,
    /// Update mode ("full", "incremental"). Default: "full".
    #[napi(js_name = "update_mode")]
    pub update_mode: Option<String>,
}

/// Result of a single online update step.
#[napi(object)]
pub struct OnlineOutput {
    /// Smoothed value for the latest point.
    pub smoothed: f64,
    /// Standard error (if computed).
    #[napi(js_name = "std_error")]
    pub std_error: Option<f64>,
    /// Residual y − smoothed (if computed).
    pub residual: Option<f64>,
    /// Robustness weight for the latest point (if computed).
    #[napi(js_name = "robustness_weight")]
    pub robustness_weight: Option<f64>,
    /// Number of robustness iterations performed (if applicable).
    #[napi(js_name = "iterations_used")]
    pub iterations_used: Option<u32>,
}

/// Online LOWESS smoother for real-time data.
#[napi]
pub struct OnlineLowess {
    inner: ParallelOnlineLowess<f64>,
}

#[napi]
impl OnlineLowess {
    /// Create a new online LOWESS smoother.
    #[napi(constructor)]
    pub fn new(options: Option<SmoothOptions>, online_opts: Option<OnlineOptions>) -> Result<Self> {
        let builder = options_to_builder(options.as_ref())?;

        let (window_capacity, min_points, update_mode) = match online_opts {
            Some(o) => (
                o.window_capacity.map(|v| v as usize),
                o.min_points.map(|v| v as usize),
                o.update_mode,
            ),
            None => (None, None, None),
        };

        let model = map_runtime(binding_support::build_online(
            builder,
            window_capacity,
            min_points,
            update_mode.as_deref(),
        ))?;

        Ok(OnlineLowess { inner: model })
    }

    /// Add a single point and get the smoothed value if enough points are available.
    #[napi(js_name = "add_point")]
    pub fn add_point(&mut self, x: f64, y: f64) -> Result<Option<OnlineOutput>> {
        let output = self.inner.add_point(x, y).map_err(|e| {
            to_napi_error(binding_support::BindingError::invalid_arg(e.to_string()))
        })?;
        Ok(output.map(|o| OnlineOutput {
            smoothed: o.smoothed,
            std_error: o.std_error,
            residual: o.residual,
            robustness_weight: o.robustness_weight,
            iterations_used: o.iterations_used.map(|i| i as u32),
        }))
    }
}
