//! Node.js bindings for fastLowess using N-API.

use napi::bindgen_prelude::*;
use napi_derive::napi;

use ::fastLowess::internals::binding_support;
use ::fastLowess::internals::adapters::online::ParallelOnlineLowess;
use ::fastLowess::internals::adapters::streaming::ParallelStreamingLowess;
use ::fastLowess::internals::api::{Batch, LowessBuilder, Online, Streaming};
use ::fastLowess::prelude::{LowessError, LowessResult};

/// Diagnostic statistics for the LOWESS fit.
#[napi(object)]
pub struct Diagnostics {
    /// Root Mean Squared Error.
    pub rmse: f64,
    /// Mean Absolute Error.
    pub mae: f64,
    /// R-squared (coefficient of determination).
    pub r_squared: f64,
    /// Akaike Information Criterion (if computed).
    pub aic: Option<f64>,
    /// Corrected AIC (if computed).
    pub aicc: Option<f64>,
    /// Effective degrees of freedom (if computed).
    pub effective_df: Option<f64>,
    /// Residual standard deviation.
    pub residual_sd: f64,
}

/// Result of a LOWESS fit.
#[napi]
pub struct LowessResultObj {
    inner: LowessResult<f64>,
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
    pub weight_function: Option<String>,
    /// Robustness method ("bisquare", "huber", "talwar"). Default: "bisquare".
    pub robustness_method: Option<String>,
    /// Fallback strategy when weights are zero ("use_local_mean", "return_original", "return_none"). Default: "use_local_mean".
    pub zero_weight_fallback: Option<String>,
    /// Boundary handling ("extend", "reflect", "zero", "noboundary"). Default: "extend".
    pub boundary_policy: Option<String>,
    /// Scaling method ("mad", "mar", "mean"). Default: "mad".
    pub scaling_method: Option<String>,
    /// Auto-convergence tolerance. Default: None.
    pub auto_converge: Option<f64>,
    /// Return residuals in result. Default: false.
    pub return_residuals: Option<bool>,
    /// Return robustness weights in result. Default: false.
    pub return_robustness_weights: Option<bool>,
    /// Return diagnostics (RMSE, etc.). Default: false.
    pub return_diagnostics: Option<bool>,
    /// Calculate confidence intervals (e.g., 0.95). Default: None.
    pub confidence_intervals: Option<f64>,
    /// Calculate prediction intervals. Default: None.
    pub prediction_intervals: Option<f64>,
    /// Fractions to use for cross-validation.
    pub cv_fractions: Option<Vec<f64>>,
    /// CV method ("loocv", "kfold"). Default: "kfold".
    pub cv_method: Option<String>,
    /// Number of folds for K-Fold CV. Default: 5.
    pub cv_k: Option<u32>,
    /// Enable parallel execution. Default: true.
    pub parallel: Option<bool>,
    /// Per-observation case weights. Must have the same length as input data.
    pub custom_weights: Option<Vec<f64>>,
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
    pub fn fit(&self, x: Float64Array, y: Float64Array) -> Result<LowessResultObj> {
        let builder = self.create_builder()?;
        let model = builder
            .adapter(Batch)
            .build()
            .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))?;

        let result = model
            .fit(x.as_ref(), y.as_ref())
            .map_err(|e: LowessError| Error::new(Status::GenericFailure, e.to_string()))?;

        Ok(LowessResultObj { inner: result })
    }

    /// Fit the model asynchronously.
    #[napi(js_name = "fit_async")]
    pub fn fit_async(&self, x: Float64Array, y: Float64Array) -> Result<AsyncTask<LowessTask>> {
        let builder = self.create_builder()?;
        let x_vec = x.as_ref().to_vec();
        let y_vec = y.as_ref().to_vec();

        Ok(AsyncTask::new(LowessTask {
            builder,
            x: x_vec,
            y: y_vec,
        }))
    }

    fn create_builder(&self) -> Result<LowessBuilder<f64>> {
        let o = self.options.as_ref();
        let mut builder = binding_support::apply_builder_options(
            LowessBuilder::<f64>::new(),
            binding_support::BuilderOptionSet {
                fraction: o.and_then(|x| x.fraction),
                iterations: o.and_then(|x| x.iterations).map(|n| n as usize),
                delta: o.and_then(|x| x.delta),
                weight_function: o.and_then(|x| x.weight_function.as_deref()),
                robustness_method: o.and_then(|x| x.robustness_method.as_deref()),
                zero_weight_fallback: o.and_then(|x| x.zero_weight_fallback.as_deref()),
                boundary_policy: o.and_then(|x| x.boundary_policy.as_deref()),
                scaling_method: o.and_then(|x| x.scaling_method.as_deref()),
                auto_converge: o.and_then(|x| x.auto_converge),
                return_residuals: o.map_or(false, |x| x.return_residuals.unwrap_or(false)),
                return_robustness_weights: o
                    .map_or(false, |x| x.return_robustness_weights.unwrap_or(false)),
                return_diagnostics: o.map_or(false, |x| x.return_diagnostics.unwrap_or(false)),
                return_se: false,
                confidence_intervals: o.and_then(|x| x.confidence_intervals),
                prediction_intervals: o.and_then(|x| x.prediction_intervals),
                parallel: o.and_then(|x| x.parallel),
                chunk_size: None,
                overlap: None,
                merge_strategy: None,
                window_capacity: None,
                min_points: None,
                update_mode: None,
                cv_fractions: o.and_then(|x| x.cv_fractions.as_deref()),
                cv_method: o.and_then(|x| x.cv_method.as_deref()),
                cv_k: o.and_then(|x| x.cv_k).map(|k| k as usize),
                cv_seed: None,
            },
        )
        .map_err(|e| Error::new(Status::InvalidArg, e))?;

        if let Some(cw) = o.and_then(|x| x.custom_weights.clone()) {
            builder = builder.custom_weights(cw);
        }

        Ok(builder)
    }
}

pub struct LowessTask {
    builder: LowessBuilder<f64>,
    x: Vec<f64>,
    y: Vec<f64>,
}

impl Task for LowessTask {
    type Output = LowessResult<f64>;
    type JsValue = LowessResultObj;

    fn compute(&mut self) -> Result<Self::Output> {
        let model = self
            .builder
            .clone()
            .adapter(Batch)
            .build()
            .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))?;

        model
            .fit(&self.x, &self.y)
            .map_err(|e: LowessError| Error::new(Status::GenericFailure, e.to_string()))
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(LowessResultObj { inner: output })
    }
}

/// Configuration options for streaming processing.
#[napi(object)]
pub struct StreamingOptions {
    /// Size of each data chunk. Default: 5000.
    pub chunk_size: Option<u32>,
    /// Header/footer overlap size. Default: 500.
    pub overlap: Option<u32>,
    /// Strategy for merging chunks (not exposed yet).
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
        let o = options.as_ref();
        let so = streaming_opts.as_ref();
        let model = binding_support::apply_builder_options(
            LowessBuilder::<f64>::new(),
            binding_support::BuilderOptionSet {
                fraction: o.and_then(|x| x.fraction),
                iterations: o.and_then(|x| x.iterations).map(|n| n as usize),
                delta: o.and_then(|x| x.delta),
                weight_function: o.and_then(|x| x.weight_function.as_deref()),
                robustness_method: o.and_then(|x| x.robustness_method.as_deref()),
                zero_weight_fallback: o.and_then(|x| x.zero_weight_fallback.as_deref()),
                boundary_policy: o.and_then(|x| x.boundary_policy.as_deref()),
                scaling_method: o.and_then(|x| x.scaling_method.as_deref()),
                auto_converge: o.and_then(|x| x.auto_converge),
                return_residuals: o.map_or(false, |x| x.return_residuals.unwrap_or(false)),
                return_robustness_weights: o
                    .map_or(false, |x| x.return_robustness_weights.unwrap_or(false)),
                return_diagnostics: o.map_or(false, |x| x.return_diagnostics.unwrap_or(false)),
                return_se: false,
                confidence_intervals: None,
                prediction_intervals: None,
                parallel: o.and_then(|x| x.parallel),
                chunk_size: Some(
                    so.and_then(|x| x.chunk_size)
                        .map(|n| n as usize)
                        .unwrap_or(5000),
                ),
                overlap: Some(
                    so.and_then(|x| x.overlap)
                        .map(|n| n as usize)
                        .unwrap_or(500),
                ),
                merge_strategy: so.and_then(|x| x.merge_strategy.as_deref()),
                window_capacity: None,
                min_points: None,
                update_mode: None,
                cv_fractions: None,
                cv_method: None,
                cv_k: None,
                cv_seed: None,
            },
        )
        .map_err(|e| Error::new(Status::InvalidArg, e))?
        .adapter(Streaming)
        .build()
        .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))?;

        Ok(StreamingLowess { inner: model })
    }

    /// Process a chunk of data.
    #[napi(js_name = "process_chunk")]
    pub fn process_chunk(&mut self, x: Float64Array, y: Float64Array) -> Result<LowessResultObj> {
        let result: LowessResult<f64> = self
            .inner
            .process_chunk(x.as_ref(), y.as_ref())
            .map_err(|e: LowessError| Error::new(Status::GenericFailure, e.to_string()))?;
        Ok(LowessResultObj { inner: result })
    }

    /// Finalize the stream and return remaining data.
    #[napi]
    pub fn finalize(&mut self) -> Result<LowessResultObj> {
        let result: LowessResult<f64> = self
            .inner
            .finalize()
            .map_err(|e: LowessError| Error::new(Status::GenericFailure, e.to_string()))?;
        Ok(LowessResultObj { inner: result })
    }
}

/// Configuration options for online processing.
#[napi(object)]
pub struct OnlineOptions {
    /// Maximum number of points to keep in the window. Default: 100.
    pub window_capacity: Option<u32>,
    /// Minimum points required before smoothing starts. Default: 2.
    pub min_points: Option<u32>,
    /// Update mode ("full", "incremental"). Default: "full".
    pub update_mode: Option<String>,
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
        let o = options.as_ref();
        let oo = online_opts.as_ref();
        let model = binding_support::apply_builder_options(
            LowessBuilder::<f64>::new(),
            binding_support::BuilderOptionSet {
                fraction: o.and_then(|x| x.fraction),
                iterations: o.and_then(|x| x.iterations).map(|n| n as usize),
                delta: None,
                weight_function: None,
                robustness_method: None,
                zero_weight_fallback: None,
                boundary_policy: None,
                scaling_method: None,
                auto_converge: None,
                return_residuals: false,
                return_robustness_weights: false,
                return_diagnostics: false,
                return_se: false,
                confidence_intervals: None,
                prediction_intervals: None,
                parallel: o.and_then(|x| x.parallel),
                chunk_size: None,
                overlap: None,
                merge_strategy: None,
                window_capacity: Some(
                    oo.and_then(|x| x.window_capacity)
                        .map(|n| n as usize)
                        .unwrap_or(100),
                ),
                min_points: Some(
                    oo.and_then(|x| x.min_points)
                        .map(|n| n as usize)
                        .unwrap_or(2),
                ),
                update_mode: oo.and_then(|x| x.update_mode.as_deref()),
                cv_fractions: None,
                cv_method: None,
                cv_k: None,
                cv_seed: None,
            },
        )
        .map_err(|e| Error::new(Status::InvalidArg, e))?
        .adapter(Online)
        .build()
        .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))?;

        Ok(OnlineLowess { inner: model })
    }

    /// Add a single point and get the smoothed value if enough points are available.
    #[napi(js_name = "add_point")]
    pub fn add_point(&mut self, x: f64, y: f64) -> Result<Option<f64>> {
        let result = self
            .inner
            .add_point(x, y)
            .map_err(|e: LowessError| Error::new(Status::GenericFailure, e.to_string()))?;
        Ok(result.map(|o| o.smoothed))
    }
}
