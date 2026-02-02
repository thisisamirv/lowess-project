//! Node.js bindings for fastLowess using N-API.

#![allow(non_snake_case)]

use napi::bindgen_prelude::*;
use napi_derive::napi;

use ::fastLowess::internals::adapters::online::ParallelOnlineLowess;
use ::fastLowess::internals::adapters::streaming::ParallelStreamingLowess;
use ::fastLowess::internals::api::{
    BoundaryPolicy, RobustnessMethod, ScalingMethod, UpdateMode, WeightFunction, ZeroWeightFallback,
};
use ::fastLowess::prelude::{
    Batch, KFold, LOOCV, Lowess as LowessBuilder, LowessError, LowessResult, MAD, MAR, Mean,
    Online, Streaming,
};

/// Parse weight function from string
fn parse_weight_function(name: &str) -> Result<WeightFunction> {
    match name.to_lowercase().as_str() {
        "tricube" => Ok(WeightFunction::Tricube),
        "epanechnikov" => Ok(WeightFunction::Epanechnikov),
        "gaussian" => Ok(WeightFunction::Gaussian),
        "uniform" | "boxcar" => Ok(WeightFunction::Uniform),
        "biweight" | "bisquare" => Ok(WeightFunction::Biweight),
        "triangle" | "triangular" => Ok(WeightFunction::Triangle),
        "cosine" => Ok(WeightFunction::Cosine),
        _ => Err(Error::new(
            Status::InvalidArg,
            format!("Unknown weight function: {}", name),
        )),
    }
}

/// Parse robustness method from string
fn parse_robustness_method(name: &str) -> Result<RobustnessMethod> {
    match name.to_lowercase().as_str() {
        "bisquare" | "biweight" => Ok(RobustnessMethod::Bisquare),
        "huber" => Ok(RobustnessMethod::Huber),
        "talwar" => Ok(RobustnessMethod::Talwar),
        _ => Err(Error::new(
            Status::InvalidArg,
            format!("Unknown robustness method: {}", name),
        )),
    }
}

/// Parse zero weight fallback from string
fn parse_zero_weight_fallback(name: &str) -> Result<ZeroWeightFallback> {
    match name.to_lowercase().as_str() {
        "use_local_mean" | "local_mean" | "mean" => Ok(ZeroWeightFallback::UseLocalMean),
        "return_original" | "original" => Ok(ZeroWeightFallback::ReturnOriginal),
        "return_none" | "none" | "nan" => Ok(ZeroWeightFallback::ReturnNone),
        _ => Err(Error::new(
            Status::InvalidArg,
            format!("Unknown zero weight fallback: {}", name),
        )),
    }
}

/// Parse boundary policy from string
fn parse_boundary_policy(name: &str) -> Result<BoundaryPolicy> {
    match name.to_lowercase().as_str() {
        "extend" | "pad" => Ok(BoundaryPolicy::Extend),
        "reflect" | "mirror" => Ok(BoundaryPolicy::Reflect),
        "zero" | "none" => Ok(BoundaryPolicy::Zero),
        "noboundary" => Ok(BoundaryPolicy::NoBoundary),
        _ => Err(Error::new(
            Status::InvalidArg,
            format!("Unknown boundary policy: {}", name),
        )),
    }
}

/// Parse scaling method from string
fn parse_scaling_method(name: &str) -> Result<ScalingMethod> {
    match name.to_lowercase().as_str() {
        "mad" => Ok(MAD),
        "mar" => Ok(MAR),
        "mean" => Ok(Mean),
        _ => Err(Error::new(
            Status::InvalidArg,
            format!(
                "Unknown scaling method: {}. Valid options: mad, mar, mean",
                name
            ),
        )),
    }
}

/// Parse update mode from string
fn parse_update_mode(name: &str) -> Result<UpdateMode> {
    match name.to_lowercase().as_str() {
        "full" | "resmooth" => Ok(UpdateMode::Full),
        "incremental" | "single" => Ok(UpdateMode::Incremental),
        _ => Err(Error::new(
            Status::InvalidArg,
            format!("Unknown update mode: {}", name),
        )),
    }
}

/// Diagnostic statistics for the LOWESS fit.
#[napi(object)]
pub struct Diagnostics {
    /// Root Mean Squared Error.
    pub rmse: f64,
    /// Mean Absolute Error.
    pub mae: f64,
    /// R-squared (coefficient of determination).
    pub rSquared: f64,
    /// Akaike Information Criterion (if computed).
    pub aic: Option<f64>,
    /// Corrected AIC (if computed).
    pub aicc: Option<f64>,
    /// Effective degrees of freedom (if computed).
    pub effectiveDf: Option<f64>,
    /// Residual standard deviation.
    pub residualSd: f64,
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
    #[napi(getter)]
    pub fn get_standard_errors(&self) -> Option<Float64Array> {
        self.inner
            .standard_errors
            .as_ref()
            .map(|v| Float64Array::from(v.as_slice()))
    }

    /// Get lower confidence bounds (if requested).
    #[napi(getter)]
    pub fn get_confidence_lower(&self) -> Option<Float64Array> {
        self.inner
            .confidence_lower
            .as_ref()
            .map(|v| Float64Array::from(v.as_slice()))
    }

    /// Get upper confidence bounds (if requested).
    #[napi(getter)]
    pub fn get_confidence_upper(&self) -> Option<Float64Array> {
        self.inner
            .confidence_upper
            .as_ref()
            .map(|v| Float64Array::from(v.as_slice()))
    }

    /// Get lower prediction bounds (if requested).
    #[napi(getter)]
    pub fn get_prediction_lower(&self) -> Option<Float64Array> {
        self.inner
            .prediction_lower
            .as_ref()
            .map(|v| Float64Array::from(v.as_slice()))
    }

    /// Get upper prediction bounds (if requested).
    #[napi(getter)]
    pub fn get_prediction_upper(&self) -> Option<Float64Array> {
        self.inner
            .prediction_upper
            .as_ref()
            .map(|v| Float64Array::from(v.as_slice()))
    }

    /// Get robustness weights (if requested).
    #[napi(getter)]
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
            rSquared: d.r_squared,
            aic: d.aic,
            aicc: d.aicc,
            effectiveDf: d.effective_df,
            residualSd: d.residual_sd,
        })
    }

    /// Get cross-validation scores (if CV was performed).
    #[napi(getter)]
    pub fn get_cv_scores(&self) -> Option<Float64Array> {
        self.inner
            .cv_scores
            .as_ref()
            .map(|v| Float64Array::from(v.as_slice()))
    }

    /// Get the fraction used for smoothing.
    #[napi(getter)]
    pub fn get_fraction_used(&self) -> f64 {
        self.inner.fraction_used
    }

    /// Get the number of iterations performed.
    #[napi(getter)]
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
    /// Weight function ("tricube", "gaussian", etc.). Default: "tricube".
    pub weightFunction: Option<String>,
    /// Robustness method ("bisquare", "huber"). Default: "bisquare".
    pub robustnessMethod: Option<String>,
    /// Fallback strategy when weights are zero ("use_local_mean").
    pub zeroWeightFallback: Option<String>,
    /// Boundary handling ("extend", "reflect"). Default: "extend".
    pub boundaryPolicy: Option<String>,
    /// Scaling method ("mad", "mar"). Default: "mad".
    pub scalingMethod: Option<String>,
    /// Auto-convergence tolerance. Default: None.
    pub autoConverge: Option<f64>,
    /// Return residuals in result. Default: false.
    pub returnResiduals: Option<bool>,
    /// Return robustness weights in result. Default: false.
    pub returnRobustnessWeights: Option<bool>,
    /// Return diagnostics (RMSE, etc.). Default: false.
    pub returnDiagnostics: Option<bool>,
    /// Calculate confidence intervals (e.g., 0.95). Default: None.
    pub confidenceIntervals: Option<f64>,
    /// Calculate prediction intervals. Default: None.
    pub predictionIntervals: Option<f64>,
    /// Fractions to use for cross-validation.
    pub cvFractions: Option<Vec<f64>>,
    /// CV method ("loocv", "kfold"). Default: "kfold".
    pub cvMethod: Option<String>,
    /// Number of folds for K-Fold CV. Default: 5.
    pub cvK: Option<u32>,
    /// Enable parallel execution. Default: true.
    pub parallel: Option<bool>,
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
    #[napi]
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
        let mut builder = LowessBuilder::new();
        let options = &self.options;

        if let Some(opts) = options {
            if let Some(f) = opts.fraction {
                builder = builder.fraction(f);
            }
            if let Some(iter) = opts.iterations {
                builder = builder.iterations(iter as usize);
            }
            if let Some(d) = opts.delta {
                builder = builder.delta(d);
            }
            if let Some(wf) = &opts.weightFunction {
                builder = builder.weight_function(parse_weight_function(wf)?);
            }
            if let Some(rm) = &opts.robustnessMethod {
                builder = builder.robustness_method(parse_robustness_method(rm)?);
            }
            if let Some(zw) = &opts.zeroWeightFallback {
                builder = builder.zero_weight_fallback(parse_zero_weight_fallback(zw)?);
            }
            if let Some(bp) = &opts.boundaryPolicy {
                builder = builder.boundary_policy(parse_boundary_policy(bp)?);
            }
            if let Some(sm) = &opts.scalingMethod {
                builder = builder.scaling_method(parse_scaling_method(sm)?);
            }
            if let Some(ac) = opts.autoConverge {
                builder = builder.auto_converge(ac);
            }
            if opts.returnResiduals.unwrap_or(false) {
                builder = builder.return_residuals();
            }
            if opts.returnRobustnessWeights.unwrap_or(false) {
                builder = builder.return_robustness_weights();
            }
            if opts.returnDiagnostics.unwrap_or(false) {
                builder = builder.return_diagnostics();
            }
            if let Some(ci) = opts.confidenceIntervals {
                builder = builder.confidence_intervals(ci);
            }
            if let Some(pi) = opts.predictionIntervals {
                builder = builder.prediction_intervals(pi);
            }
            if let Some(par) = opts.parallel {
                builder = builder.parallel(par);
            }

            // Cross-validation
            if let Some(fractions) = &opts.cvFractions {
                let method = opts.cvMethod.as_deref().unwrap_or("kfold");
                let k = opts.cvK.unwrap_or(5) as usize;

                match method.to_lowercase().as_str() {
                    "simple" | "loo" | "loocv" | "leave_one_out" => {
                        builder = builder.cross_validate(LOOCV(fractions));
                    }
                    "kfold" | "k_fold" | "k-fold" => {
                        builder = builder.cross_validate(KFold(k, fractions));
                    }
                    _ => {
                        return Err(Error::new(
                            Status::InvalidArg,
                            format!("Unknown CV method: {}. Valid options: loocv, kfold", method),
                        ));
                    }
                };
            }
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
    pub chunkSize: Option<u32>,
    /// Header/footer overlap size. Default: 500.
    pub overlap: Option<u32>,
    /// Strategy for merging chunks (not exposed yet).
    pub mergeStrategy: Option<String>,
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
        let mut builder = LowessBuilder::new();

        if let Some(opts) = options {
            if let Some(f) = opts.fraction {
                builder = builder.fraction(f);
            }
            if let Some(iter) = opts.iterations {
                builder = builder.iterations(iter as usize);
            }
            if let Some(d) = opts.delta {
                builder = builder.delta(d);
            }
            if let Some(wf) = opts.weightFunction {
                builder = builder.weight_function(parse_weight_function(&wf)?);
            }
            if let Some(rm) = opts.robustnessMethod {
                builder = builder.robustness_method(parse_robustness_method(&rm)?);
            }
            if let Some(zw) = opts.zeroWeightFallback {
                builder = builder.zero_weight_fallback(parse_zero_weight_fallback(&zw)?);
            }
            if let Some(bp) = opts.boundaryPolicy {
                builder = builder.boundary_policy(parse_boundary_policy(&bp)?);
            }
            if let Some(sm) = opts.scalingMethod {
                builder = builder.scaling_method(parse_scaling_method(&sm)?);
            }
            if let Some(ac) = opts.autoConverge {
                builder = builder.auto_converge(ac);
            }
            if opts.returnResiduals.unwrap_or(false) {
                builder = builder.return_residuals();
            }
            if opts.returnRobustnessWeights.unwrap_or(false) {
                builder = builder.return_robustness_weights();
            }
            if opts.returnDiagnostics.unwrap_or(false) {
                builder = builder.return_diagnostics();
            }
            if let Some(par) = opts.parallel {
                builder = builder.parallel(par);
            }
        }

        let mut chunk_size = 5000;
        let mut overlap = 500;

        if let Some(sopts) = streaming_opts {
            if let Some(cs) = sopts.chunkSize {
                chunk_size = cs as usize;
            }
            if let Some(ov) = sopts.overlap {
                overlap = ov as usize;
            }
        }

        let model = builder
            .adapter(Streaming)
            .chunk_size(chunk_size)
            .overlap(overlap)
            .build()
            .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))?;

        Ok(StreamingLowess { inner: model })
    }

    /// Process a chunk of data.
    #[napi]
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
    pub windowCapacity: Option<u32>,
    /// Minimum points required before smoothing starts. Default: 2.
    pub minPoints: Option<u32>,
    /// Update mode ("full", "incremental"). Default: "full".
    pub updateMode: Option<String>,
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
        let mut builder = LowessBuilder::new();

        if let Some(opts) = options {
            if let Some(f) = opts.fraction {
                builder = builder.fraction(f);
            }
            if let Some(iter) = opts.iterations {
                builder = builder.iterations(iter as usize);
            }
            if let Some(par) = opts.parallel {
                builder = builder.parallel(par);
            }
        }

        let mut window_capacity = 100;
        let mut min_points = 2;
        let mut update_mode = UpdateMode::Full;

        if let Some(oopts) = online_opts {
            if let Some(wc) = oopts.windowCapacity {
                window_capacity = wc as usize;
            }
            if let Some(mp) = oopts.minPoints {
                min_points = mp as usize;
            }
            if let Some(um) = oopts.updateMode {
                update_mode = parse_update_mode(&um)?;
            }
        }

        let model = builder
            .adapter(Online)
            .window_capacity(window_capacity)
            .min_points(min_points)
            .update_mode(update_mode)
            .build()
            .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))?;

        Ok(OnlineLowess { inner: model })
    }

    /// Add new points to the window and get smoothed values.
    #[napi]
    pub fn add_points(&mut self, x: Float64Array, y: Float64Array) -> Result<LowessResultObj> {
        let result = self
            .inner
            .add_points(x.as_ref(), y.as_ref())
            .map_err(|e: LowessError| Error::new(Status::GenericFailure, e.to_string()))?;

        // Extract smoothed values from results
        let x_vec = x.as_ref().to_vec();

        let smoothed: Vec<f64> = result
            .iter()
            .zip(y.as_ref().iter())
            .map(|(opt, &original_y)| opt.as_ref().map_or(original_y, |o| o.smoothed))
            .collect();

        let inner_result = LowessResult {
            x: x_vec,
            y: smoothed,
            standard_errors: None,
            confidence_lower: None,
            confidence_upper: None,
            prediction_lower: None,
            prediction_upper: None,
            residuals: None,
            robustness_weights: None,
            diagnostics: None,
            iterations_used: None,
            fraction_used: 0.0,
            cv_scores: None,
        };

        Ok(LowessResultObj {
            inner: inner_result,
        })
    }
}
