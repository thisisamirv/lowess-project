//! WebAssembly bindings for fastLowess.

use js_sys::Float64Array;
use serde::Deserialize;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

use ::fastLowess::internals::adapters::online::ParallelOnlineLowess;
use ::fastLowess::internals::adapters::streaming::ParallelStreamingLowess;
use ::fastLowess::internals::api::{
    BoundaryPolicy, RobustnessMethod, ScalingMethod, UpdateMode, WeightFunction, ZeroWeightFallback,
};
use ::fastLowess::prelude::{
    Batch, KFold, LOOCV, Lowess as LowessBuilder, LowessResult, MAD, MAR, Mean, Online, Streaming,
};

#[derive(Deserialize)]
pub struct SmoothOptions {
    pub fraction: Option<f64>,
    pub iterations: Option<usize>,
    pub delta: Option<f64>,
    #[serde(rename = "weightFunction")]
    pub weight_function: Option<String>,
    #[serde(rename = "robustnessMethod")]
    pub robustness_method: Option<String>,
    #[serde(rename = "zeroWeightFallback")]
    pub zero_weight_fallback: Option<String>,
    #[serde(rename = "boundaryPolicy")]
    pub boundary_policy: Option<String>,
    #[serde(rename = "scalingMethod")]
    pub scaling_method: Option<String>,
    #[serde(rename = "autoConverge")]
    pub auto_converge: Option<f64>,
    #[serde(rename = "returnResiduals")]
    pub return_residuals: Option<bool>,
    #[serde(rename = "returnRobustnessWeights")]
    pub return_robustness_weights: Option<bool>,
    #[serde(rename = "returnDiagnostics")]
    pub return_diagnostics: Option<bool>,
    #[serde(rename = "confidenceIntervals")]
    pub confidence_intervals: Option<f64>,
    #[serde(rename = "predictionIntervals")]
    pub prediction_intervals: Option<f64>,
    #[serde(rename = "parallel")]
    pub parallel: Option<bool>,
    #[serde(rename = "cvFractions")]
    pub cv_fractions: Option<Vec<f64>>,
    #[serde(rename = "cvMethod")]
    pub cv_method: Option<String>,
    #[serde(rename = "cvK")]
    pub cv_k: Option<u32>,
}

#[derive(Deserialize)]
pub struct StreamingOptions {
    #[serde(rename = "chunkSize")]
    pub chunk_size: Option<usize>,
    #[serde(rename = "overlap")]
    pub overlap: Option<usize>,
}

#[derive(Deserialize)]
pub struct OnlineOptions {
    #[serde(rename = "windowCapacity")]
    pub window_capacity: Option<usize>,
    #[serde(rename = "minPoints")]
    pub min_points: Option<usize>,
    #[serde(rename = "updateMode")]
    pub update_mode: Option<String>,
}

fn parse_weight_function(name: &str) -> Result<WeightFunction, JsValue> {
    match name.to_lowercase().as_str() {
        "tricube" => Ok(WeightFunction::Tricube),
        "epanechnikov" => Ok(WeightFunction::Epanechnikov),
        "gaussian" => Ok(WeightFunction::Gaussian),
        "uniform" | "boxcar" => Ok(WeightFunction::Uniform),
        "biweight" | "bisquare" => Ok(WeightFunction::Biweight),
        "triangle" | "triangular" => Ok(WeightFunction::Triangle),
        "cosine" => Ok(WeightFunction::Cosine),
        _ => Err(JsValue::from_str(&format!(
            "Unknown weight function: {}",
            name
        ))),
    }
}

fn parse_robustness_method(name: &str) -> Result<RobustnessMethod, JsValue> {
    match name.to_lowercase().as_str() {
        "bisquare" | "biweight" => Ok(RobustnessMethod::Bisquare),
        "huber" => Ok(RobustnessMethod::Huber),
        "talwar" => Ok(RobustnessMethod::Talwar),
        _ => Err(JsValue::from_str(&format!(
            "Unknown robustness method: {}",
            name
        ))),
    }
}

fn parse_zero_weight_fallback(name: &str) -> Result<ZeroWeightFallback, JsValue> {
    match name.to_lowercase().as_str() {
        "use_local_mean" | "local_mean" | "mean" => Ok(ZeroWeightFallback::UseLocalMean),
        "return_original" | "original" => Ok(ZeroWeightFallback::ReturnOriginal),
        "return_none" | "none" | "nan" => Ok(ZeroWeightFallback::ReturnNone),
        _ => Err(JsValue::from_str(&format!(
            "Unknown zero weight fallback: {}",
            name
        ))),
    }
}

fn parse_boundary_policy(name: &str) -> Result<BoundaryPolicy, JsValue> {
    match name.to_lowercase().as_str() {
        "extend" | "pad" => Ok(BoundaryPolicy::Extend),
        "reflect" | "mirror" => Ok(BoundaryPolicy::Reflect),
        "zero" | "none" => Ok(BoundaryPolicy::Zero),
        "noboundary" => Ok(BoundaryPolicy::NoBoundary),
        _ => Err(JsValue::from_str(&format!(
            "Unknown boundary policy: {}",
            name
        ))),
    }
}

fn parse_scaling_method(name: &str) -> Result<ScalingMethod, JsValue> {
    match name.to_lowercase().as_str() {
        "mad" => Ok(MAD),
        "mar" => Ok(MAR),
        "mean" => Ok(Mean),
        _ => Err(JsValue::from_str(&format!(
            "Unknown scaling method: {}. Valid options: mad, mar, mean",
            name
        ))),
    }
}

fn parse_update_mode(name: &str) -> Result<UpdateMode, JsValue> {
    match name.to_lowercase().as_str() {
        "full" | "resmooth" => Ok(UpdateMode::Full),
        "incremental" | "single" => Ok(UpdateMode::Incremental),
        _ => Err(JsValue::from_str(&format!("Unknown update mode: {}", name))),
    }
}

#[wasm_bindgen]
pub struct Diagnostics {
    pub rmse: f64,
    pub mae: f64,
    #[wasm_bindgen(js_name = rSquared)]
    pub r_squared: f64,
    pub aic: Option<f64>,
    pub aicc: Option<f64>,
    #[wasm_bindgen(js_name = effectiveDf)]
    pub effective_df: Option<f64>,
    #[wasm_bindgen(js_name = residualSd)]
    pub residual_sd: f64,
}

#[wasm_bindgen]
pub struct LowessResultWasm {
    inner: LowessResult<f64>,
}

#[wasm_bindgen]
impl LowessResultWasm {
    #[wasm_bindgen(getter)]
    pub fn x(&self) -> Float64Array {
        unsafe { Float64Array::view(&self.inner.x) }
    }

    #[wasm_bindgen(getter)]
    pub fn y(&self) -> Float64Array {
        unsafe { Float64Array::view(&self.inner.y) }
    }

    #[wasm_bindgen(getter)]
    pub fn residuals(&self) -> Option<Float64Array> {
        self.inner
            .residuals
            .as_ref()
            .map(|v| unsafe { Float64Array::view(v) })
    }

    #[wasm_bindgen(getter, js_name = standardErrors)]
    pub fn standard_errors(&self) -> Option<Float64Array> {
        self.inner
            .standard_errors
            .as_ref()
            .map(|v| unsafe { Float64Array::view(v) })
    }

    #[wasm_bindgen(getter, js_name = confidenceLower)]
    pub fn confidence_lower(&self) -> Option<Float64Array> {
        self.inner
            .confidence_lower
            .as_ref()
            .map(|v| unsafe { Float64Array::view(v) })
    }

    #[wasm_bindgen(getter, js_name = confidenceUpper)]
    pub fn confidence_upper(&self) -> Option<Float64Array> {
        self.inner
            .confidence_upper
            .as_ref()
            .map(|v| unsafe { Float64Array::view(v) })
    }

    #[wasm_bindgen(getter, js_name = predictionLower)]
    pub fn prediction_lower(&self) -> Option<Float64Array> {
        self.inner
            .prediction_lower
            .as_ref()
            .map(|v| unsafe { Float64Array::view(v) })
    }

    #[wasm_bindgen(getter, js_name = predictionUpper)]
    pub fn prediction_upper(&self) -> Option<Float64Array> {
        self.inner
            .prediction_upper
            .as_ref()
            .map(|v| unsafe { Float64Array::view(v) })
    }

    #[wasm_bindgen(getter, js_name = robustnessWeights)]
    pub fn robustness_weights(&self) -> Option<Float64Array> {
        self.inner
            .robustness_weights
            .as_ref()
            .map(|v| unsafe { Float64Array::view(v) })
    }

    #[wasm_bindgen(getter)]
    pub fn diagnostics(&self) -> Option<Diagnostics> {
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

    #[wasm_bindgen(getter, js_name = cvScores)]
    pub fn cv_scores(&self) -> Option<Float64Array> {
        self.inner
            .cv_scores
            .as_ref()
            .map(|v| unsafe { Float64Array::view(v) })
    }

    #[wasm_bindgen(getter, js_name = fractionUsed)]
    pub fn fraction_used(&self) -> f64 {
        self.inner.fraction_used
    }

    #[wasm_bindgen(getter, js_name = iterationsUsed)]
    pub fn iterations_used(&self) -> Option<u32> {
        self.inner.iterations_used.map(|i| i as u32)
    }
}

/// Fit the LOWESS model to data.
///
/// @param {Float64Array} x - X coordinates.
/// @param {Float64Array} y - Y coordinates.
/// @param {any} [options] - Configuration object.
/// @returns {LowessResultWasm} The result of the smoothing.
#[wasm_bindgen]
pub fn smooth(
    x: &Float64Array,
    y: &Float64Array,
    options: JsValue,
) -> Result<LowessResultWasm, JsValue> {
    let mut builder = LowessBuilder::new();

    if !options.is_undefined() && !options.is_null() {
        let opts: SmoothOptions = serde_wasm_bindgen::from_value(options)?;

        if let Some(f) = opts.fraction {
            builder = builder.fraction(f);
        }
        if let Some(iter) = opts.iterations {
            builder = builder.iterations(iter);
        }
        if let Some(d) = opts.delta {
            builder = builder.delta(d);
        }
        if let Some(wf) = opts.weight_function {
            builder = builder.weight_function(parse_weight_function(&wf)?);
        }
        if let Some(rm) = opts.robustness_method {
            builder = builder.robustness_method(parse_robustness_method(&rm)?);
        }
        if let Some(zw) = opts.zero_weight_fallback {
            builder = builder.zero_weight_fallback(parse_zero_weight_fallback(&zw)?);
        }
        if let Some(bp) = opts.boundary_policy {
            builder = builder.boundary_policy(parse_boundary_policy(&bp)?);
        }
        if let Some(sm) = opts.scaling_method {
            builder = builder.scaling_method(parse_scaling_method(&sm)?);
        }
        if let Some(ac) = opts.auto_converge {
            builder = builder.auto_converge(ac);
        }
        if opts.return_residuals.unwrap_or(false) {
            builder = builder.return_residuals();
        }
        if opts.return_robustness_weights.unwrap_or(false) {
            builder = builder.return_robustness_weights();
        }
        if opts.return_diagnostics.unwrap_or(false) {
            builder = builder.return_diagnostics();
        }
        if let Some(ci) = opts.confidence_intervals {
            builder = builder.confidence_intervals(ci);
        }
        if let Some(pi) = opts.prediction_intervals {
            builder = builder.prediction_intervals(pi);
        }
        if let Some(par) = opts.parallel {
            builder = builder.parallel(par);
        }

        // Cross-validation
        if let Some(fractions) = opts.cv_fractions {
            let method = opts.cv_method.as_deref().unwrap_or("kfold");
            let k = opts.cv_k.unwrap_or(5) as usize;

            match method.to_lowercase().as_str() {
                "simple" | "loo" | "loocv" | "leave_one_out" => {
                    builder = builder.cross_validate(LOOCV(&fractions));
                }
                "kfold" | "k_fold" | "k-fold" => {
                    builder = builder.cross_validate(KFold(k, &fractions));
                }
                _ => {
                    return Err(JsValue::from_str(&format!(
                        "Unknown CV method: {}. Valid options: loocv, kfold",
                        method
                    )));
                }
            };
        }
    }

    let x_vec = x.to_vec();
    let y_vec = y.to_vec();

    let model = builder
        .adapter(Batch)
        .build()
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let result = model
        .fit(&x_vec, &y_vec)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(LowessResultWasm { inner: result })
}

/// Streaming LOWESS smoother.
#[wasm_bindgen]
pub struct StreamingLowessWasm {
    inner: ParallelStreamingLowess<f64>,
}

#[wasm_bindgen]
impl StreamingLowessWasm {
    /// Create a new streaming smoother.
    #[wasm_bindgen(constructor)]
    pub fn new(options: JsValue, streaming_opts: JsValue) -> Result<StreamingLowessWasm, JsValue> {
        let mut builder = LowessBuilder::new();

        if !options.is_undefined() && !options.is_null() {
            let opts: SmoothOptions = serde_wasm_bindgen::from_value(options)?;

            if let Some(f) = opts.fraction {
                builder = builder.fraction(f);
            }
            if let Some(iter) = opts.iterations {
                builder = builder.iterations(iter);
            }
            if let Some(d) = opts.delta {
                builder = builder.delta(d);
            }
            if let Some(wf) = opts.weight_function {
                builder = builder.weight_function(parse_weight_function(&wf)?);
            }
            if let Some(rm) = opts.robustness_method {
                builder = builder.robustness_method(parse_robustness_method(&rm)?);
            }
            if let Some(zw) = opts.zero_weight_fallback {
                builder = builder.zero_weight_fallback(parse_zero_weight_fallback(&zw)?);
            }
            if let Some(bp) = opts.boundary_policy {
                builder = builder.boundary_policy(parse_boundary_policy(&bp)?);
            }
            if let Some(sm) = opts.scaling_method {
                builder = builder.scaling_method(parse_scaling_method(&sm)?);
            }
            if let Some(ac) = opts.auto_converge {
                builder = builder.auto_converge(ac);
            }
            if let Some(par) = opts.parallel {
                builder = builder.parallel(par);
            }
        }

        let mut chunk_size = 5000;
        let mut overlap = 500;

        if !streaming_opts.is_undefined() && !streaming_opts.is_null() {
            let sopts: StreamingOptions = serde_wasm_bindgen::from_value(streaming_opts)?;
            if let Some(cs) = sopts.chunk_size {
                chunk_size = cs;
            }
            if let Some(ov) = sopts.overlap {
                overlap = ov;
            }
        }

        let model = builder
            .adapter(Streaming)
            .chunk_size(chunk_size)
            .overlap(overlap)
            .build()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(StreamingLowessWasm { inner: model })
    }

    #[wasm_bindgen(js_name = processChunk)]
    pub fn process_chunk(
        &mut self,
        x: &Float64Array,
        y: &Float64Array,
    ) -> Result<LowessResultWasm, JsValue> {
        let x_vec = x.to_vec();
        let y_vec = y.to_vec();
        let result: ::fastLowess::prelude::LowessResult<f64> = self
            .inner
            .process_chunk(&x_vec, &y_vec)
            .map_err(|e: ::fastLowess::prelude::LowessError| JsValue::from_str(&e.to_string()))?;
        Ok(LowessResultWasm { inner: result })
    }

    pub fn finalize(&mut self) -> Result<LowessResultWasm, JsValue> {
        let result: ::fastLowess::prelude::LowessResult<f64> = self
            .inner
            .finalize()
            .map_err(|e: ::fastLowess::prelude::LowessError| JsValue::from_str(&e.to_string()))?;
        Ok(LowessResultWasm { inner: result })
    }
}

/// Online LOWESS smoother.
#[wasm_bindgen]
pub struct OnlineLowessWasm {
    inner: ParallelOnlineLowess<f64>,
}

#[wasm_bindgen]
impl OnlineLowessWasm {
    /// Create a new online smoother.
    #[wasm_bindgen(constructor)]
    pub fn new(options: JsValue, online_opts: JsValue) -> Result<OnlineLowessWasm, JsValue> {
        let mut builder = LowessBuilder::new();

        if !options.is_undefined() && !options.is_null() {
            let opts: SmoothOptions = serde_wasm_bindgen::from_value(options)?;

            if let Some(f) = opts.fraction {
                builder = builder.fraction(f);
            }
            if let Some(iter) = opts.iterations {
                builder = builder.iterations(iter);
            }
            if let Some(par) = opts.parallel {
                builder = builder.parallel(par);
            }
        }

        let mut window_capacity = 100;
        let mut min_points = 2;
        let mut update_mode = UpdateMode::Full;

        if !online_opts.is_undefined() && !online_opts.is_null() {
            let oopts: OnlineOptions = serde_wasm_bindgen::from_value(online_opts)?;
            if let Some(wc) = oopts.window_capacity {
                window_capacity = wc;
            }
            if let Some(mp) = oopts.min_points {
                min_points = mp;
            }
            if let Some(um) = oopts.update_mode {
                update_mode = parse_update_mode(&um)?;
            }
        }

        let model = builder
            .adapter(Online)
            .window_capacity(window_capacity)
            .min_points(min_points)
            .update_mode(update_mode)
            .build()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(OnlineLowessWasm { inner: model })
    }

    pub fn update(&mut self, x: f64, y: f64) -> Result<Option<f64>, JsValue> {
        let result = self
            .inner
            .add_point(x, y)
            .map_err(|e: ::fastLowess::prelude::LowessError| JsValue::from_str(&e.to_string()))?;
        Ok(result.map(|o| o.smoothed))
    }
}
