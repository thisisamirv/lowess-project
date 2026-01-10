//! WebAssembly bindings for fastLowess.

use js_sys::{Float64Array, Object, Reflect};
use wasm_bindgen::prelude::*;

use ::fastLowess::internals::adapters::online::ParallelOnlineLowess;
use ::fastLowess::internals::adapters::streaming::ParallelStreamingLowess;
use ::fastLowess::internals::api::{
    BoundaryPolicy, RobustnessMethod, ScalingMethod, UpdateMode, WeightFunction, ZeroWeightFallback,
};
use ::fastLowess::prelude::{
    Batch, KFold, LOOCV, Lowess as LowessBuilder, LowessResult, MAD, MAR, Online, Streaming,
};

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
        _ => Err(JsValue::from_str(&format!(
            "Unknown scaling method: {}",
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

#[wasm_bindgen]
pub fn smooth(
    x: &Float64Array,
    y: &Float64Array,
    options: &JsValue,
) -> Result<LowessResultWasm, JsValue> {
    let mut builder = LowessBuilder::new();

    if !options.is_undefined() && !options.is_null() {
        let options = Object::from(options.clone());

        if let Ok(f) = Reflect::get(&options, &JsValue::from_str("fraction"))
            && let Some(val) = f.as_f64()
        {
            builder = builder.fraction(val);
        }
        if let Ok(iter) = Reflect::get(&options, &JsValue::from_str("iterations"))
            && let Some(val) = iter.as_f64()
        {
            builder = builder.iterations(val as usize);
        }
        if let Ok(delta) = Reflect::get(&options, &JsValue::from_str("delta"))
            && let Some(val) = delta.as_f64()
        {
            builder = builder.delta(val);
        }
        if let Ok(wf) = Reflect::get(&options, &JsValue::from_str("weightFunction"))
            && let Some(val) = wf.as_string()
        {
            builder = builder.weight_function(parse_weight_function(&val)?);
        }
        if let Ok(rm) = Reflect::get(&options, &JsValue::from_str("robustnessMethod"))
            && let Some(val) = rm.as_string()
        {
            builder = builder.robustness_method(parse_robustness_method(&val)?);
        }
        if let Ok(zw) = Reflect::get(&options, &JsValue::from_str("zeroWeightFallback"))
            && let Some(val) = zw.as_string()
        {
            builder = builder.zero_weight_fallback(parse_zero_weight_fallback(&val)?);
        }
        if let Ok(bp) = Reflect::get(&options, &JsValue::from_str("boundaryPolicy"))
            && let Some(val) = bp.as_string()
        {
            builder = builder.boundary_policy(parse_boundary_policy(&val)?);
        }
        if let Ok(sm) = Reflect::get(&options, &JsValue::from_str("scalingMethod"))
            && let Some(val) = sm.as_string()
        {
            builder = builder.scaling_method(parse_scaling_method(&val)?);
        }
        if let Ok(ac) = Reflect::get(&options, &JsValue::from_str("autoConverge"))
            && let Some(val) = ac.as_f64()
        {
            builder = builder.auto_converge(val);
        }
        if let Ok(rr) = Reflect::get(&options, &JsValue::from_str("returnResiduals"))
            && rr.as_bool().unwrap_or(false)
        {
            builder = builder.return_residuals();
        }
        if let Ok(rw) = Reflect::get(&options, &JsValue::from_str("returnRobustnessWeights"))
            && rw.as_bool().unwrap_or(false)
        {
            builder = builder.return_robustness_weights();
        }
        if let Ok(rd) = Reflect::get(&options, &JsValue::from_str("returnDiagnostics"))
            && rd.as_bool().unwrap_or(false)
        {
            builder = builder.return_diagnostics();
        }
        if let Ok(ci) = Reflect::get(&options, &JsValue::from_str("confidenceIntervals"))
            && let Some(val) = ci.as_f64()
        {
            builder = builder.confidence_intervals(val);
        }
        if let Ok(pi) = Reflect::get(&options, &JsValue::from_str("predictionIntervals"))
            && let Some(val) = pi.as_f64()
        {
            builder = builder.prediction_intervals(val);
        }
        if let Ok(par) = Reflect::get(&options, &JsValue::from_str("parallel"))
            && let Some(val) = par.as_bool()
        {
            builder = builder.parallel(val);
        }

        // Cross-validation
        if let Ok(cv_fractions) = Reflect::get(&options, &JsValue::from_str("cvFractions"))
            && !cv_fractions.is_undefined()
            && !cv_fractions.is_null()
        {
            // Convert JS number array to Vec<f64>
            let fractions: Vec<f64> = js_sys::Array::from(&cv_fractions)
                .iter()
                .map(|val| val.as_f64().unwrap_or(0.0))
                .collect();

            let cv_method = if let Ok(m) = Reflect::get(&options, &JsValue::from_str("cvMethod"))
                && let Some(val) = m.as_string()
            {
                val
            } else {
                "kfold".to_string()
            };

            let cv_k = if let Ok(k) = Reflect::get(&options, &JsValue::from_str("cvK"))
                && let Some(val) = k.as_f64()
            {
                val as usize
            } else {
                5
            };

            match cv_method.to_lowercase().as_str() {
                "simple" | "loo" | "loocv" | "leave_one_out" => {
                    builder = builder.cross_validate(LOOCV(&fractions));
                }
                "kfold" | "k_fold" | "k-fold" => {
                    builder = builder.cross_validate(KFold(cv_k, &fractions));
                }
                _ => {
                    return Err(JsValue::from_str(&format!(
                        "Unknown CV method: {}. Valid options: loocv, kfold",
                        cv_method
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

#[wasm_bindgen]
pub struct StreamingLowessWasm {
    inner: ParallelStreamingLowess<f64>,
}

#[wasm_bindgen]
impl StreamingLowessWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(
        options: &JsValue,
        streaming_opts: &JsValue,
    ) -> Result<StreamingLowessWasm, JsValue> {
        let mut builder = LowessBuilder::new();

        if !options.is_undefined() && !options.is_null() {
            let options = Object::from(options.clone());
            if let Ok(f) = Reflect::get(&options, &JsValue::from_str("fraction"))
                && let Some(val) = f.as_f64()
            {
                builder = builder.fraction(val);
            }
            if let Ok(iter) = Reflect::get(&options, &JsValue::from_str("iterations"))
                && let Some(val) = iter.as_f64()
            {
                builder = builder.iterations(val as usize);
            }
            if let Ok(delta) = Reflect::get(&options, &JsValue::from_str("delta"))
                && let Some(val) = delta.as_f64()
            {
                builder = builder.delta(val);
            }
            if let Ok(wf) = Reflect::get(&options, &JsValue::from_str("weightFunction"))
                && let Some(val) = wf.as_string()
            {
                builder = builder.weight_function(parse_weight_function(&val)?);
            }
            if let Ok(rm) = Reflect::get(&options, &JsValue::from_str("robustnessMethod"))
                && let Some(val) = rm.as_string()
            {
                builder = builder.robustness_method(parse_robustness_method(&val)?);
            }
            if let Ok(zw) = Reflect::get(&options, &JsValue::from_str("zeroWeightFallback"))
                && let Some(val) = zw.as_string()
            {
                builder = builder.zero_weight_fallback(parse_zero_weight_fallback(&val)?);
            }
            if let Ok(bp) = Reflect::get(&options, &JsValue::from_str("boundaryPolicy"))
                && let Some(val) = bp.as_string()
            {
                builder = builder.boundary_policy(parse_boundary_policy(&val)?);
            }
            if let Ok(sm) = Reflect::get(&options, &JsValue::from_str("scalingMethod"))
                && let Some(val) = sm.as_string()
            {
                builder = builder.scaling_method(parse_scaling_method(&val)?);
            }
            if let Ok(ac) = Reflect::get(&options, &JsValue::from_str("autoConverge"))
                && let Some(val) = ac.as_f64()
            {
                builder = builder.auto_converge(val);
            }
            if let Ok(par) = Reflect::get(&options, &JsValue::from_str("parallel"))
                && let Some(val) = par.as_bool()
            {
                builder = builder.parallel(val);
            }
        }

        let mut chunk_size = 5000;
        let mut overlap = 500;

        if !streaming_opts.is_undefined() && !streaming_opts.is_null() {
            let sopts = Object::from(streaming_opts.clone());
            if let Ok(cs) = Reflect::get(&sopts, &JsValue::from_str("chunkSize"))
                && let Some(val) = cs.as_f64()
            {
                chunk_size = val as usize;
            }
            if let Ok(ov) = Reflect::get(&sopts, &JsValue::from_str("overlap"))
                && let Some(val) = ov.as_f64()
            {
                overlap = val as usize;
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

#[wasm_bindgen]
pub struct OnlineLowessWasm {
    inner: ParallelOnlineLowess<f64>,
}

#[wasm_bindgen]
impl OnlineLowessWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(options: &JsValue, online_opts: &JsValue) -> Result<OnlineLowessWasm, JsValue> {
        let mut builder = LowessBuilder::new();

        if !options.is_undefined() && !options.is_null() {
            let options = Object::from(options.clone());
            if let Ok(f) = Reflect::get(&options, &JsValue::from_str("fraction"))
                && let Some(val) = f.as_f64()
            {
                builder = builder.fraction(val);
            }
            if let Ok(iter) = Reflect::get(&options, &JsValue::from_str("iterations"))
                && let Some(val) = iter.as_f64()
            {
                builder = builder.iterations(val as usize);
            }
            if let Ok(par) = Reflect::get(&options, &JsValue::from_str("parallel"))
                && let Some(val) = par.as_bool()
            {
                builder = builder.parallel(val);
            }
        }

        let mut window_capacity = 100;
        let mut min_points = 2;
        let mut update_mode = UpdateMode::Full;

        if !online_opts.is_undefined() && !online_opts.is_null() {
            let oopts = Object::from(online_opts.clone());
            if let Ok(wc) = Reflect::get(&oopts, &JsValue::from_str("windowCapacity"))
                && let Some(val) = wc.as_f64()
            {
                window_capacity = val as usize;
            }
            if let Ok(mp) = Reflect::get(&oopts, &JsValue::from_str("minPoints"))
                && let Some(val) = mp.as_f64()
            {
                min_points = val as usize;
            }
            if let Ok(um) = Reflect::get(&oopts, &JsValue::from_str("updateMode"))
                && let Some(val) = um.as_string()
            {
                update_mode = parse_update_mode(&val)?;
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
