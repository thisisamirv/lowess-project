//! WebAssembly bindings for fastLowess.

use js_sys::Float64Array;
use serde::Deserialize;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

// ============================================================================
// TypeScript interface declarations injected into the generated .d.ts
// ============================================================================

#[wasm_bindgen(typescript_custom_section)]
const TS_TYPES: &'static str = r#"
/** Configuration options for LOWESS smoothing. */
export interface SmoothOptions {
    /** Smoothing fraction (0 < fraction <= 1). Default: 0.67. */
    fraction?: number;
    /** Number of robustness iterations. Default: 3. */
    iterations?: number;
    /** Delta for interpolation speedup. Default: auto. Set to 0 to disable. */
    delta?: number;
    /** Kernel function ("tricube", "epanechnikov", "gaussian", "uniform", "biweight", "triangle", "cosine"). Default: "tricube". */
    weight_function?: string;
    /** Robustness method ("bisquare", "huber", "talwar"). Default: "bisquare". */
    robustness_method?: string;
    /** Fallback when all weights are zero ("use_local_mean", "return_original", "return_none"). Default: "use_local_mean". */
    zero_weight_fallback?: string;
    /** Boundary handling ("extend", "reflect", "zero", "noboundary"). Default: "extend". */
    boundary_policy?: string;
    /** Scaling method ("mad", "mar", "mean"). Default: "mad". */
    scaling_method?: string;
    /** Auto-convergence tolerance. Disabled when absent. */
    auto_converge?: number;
    /** Include residuals in result. Default: false. */
    return_residuals?: boolean;
    /** Include robustness weights in result. Default: false. */
    return_robustness_weights?: boolean;
    /** Compute diagnostics (RMSE, MAE, R², etc.). Default: false. */
    return_diagnostics?: boolean;
    /** Include standard errors in result. Default: false. */
    return_se?: boolean;
    /** Confidence interval level (e.g. 0.95). Disabled when absent. */
    confidence_intervals?: number;
    /** Prediction interval level (e.g. 0.95). Disabled when absent. */
    prediction_intervals?: number;
    /** Enable parallel execution. Default: true. */
    parallel?: boolean;
    /** Fractions to test for cross-validation. CV disabled when absent. */
    cv_fractions?: number[];
    /** CV method ("kfold" or "loocv"). Default: "kfold". */
    cv_method?: string;
    /** Number of folds for k-fold CV. Default: 5. */
    cv_k?: number;
    /** Random seed for CV fold assignment. */
    cv_seed?: number;
}

/** Configuration options for streaming LOWESS. */
export interface StreamingOptions {
    /** Size of each processing chunk. Default: 5000. */
    chunk_size?: number;
    /** Overlap between adjacent chunks. Default: 500. */
    overlap?: number;
    /** Strategy for merging chunks (\"average\", \"weighted_average\", \"take_first\", \"take_last\"). Default: \"weighted_average\". */
    merge_strategy?: string;
}

/** Configuration options for online LOWESS. */
export interface OnlineOptions {
    /** Maximum number of points to retain in the sliding window. Default: 100. */
    window_capacity?: number;
    /** Minimum points required before smoothing starts. Default: 2. */
    min_points?: number;
    /** Update strategy ("full" or "incremental"). Default: "full". */
    update_mode?: string;
}

/** Batch LOWESS smoother. */
export class Lowess {
    free(): void;
    constructor(options?: SmoothOptions);
    /** Fit the model to data and return smoothed values. */
    fit(x: Float64Array, y: Float64Array, customWeights?: Float64Array): LowessResult;
}

/** Streaming LOWESS smoother for large datasets. */
export class StreamingLowess {
    free(): void;
    constructor(options?: SmoothOptions, streamingOpts?: StreamingOptions);
    /** Process a chunk of data. */
    process_chunk(x: Float64Array, y: Float64Array): LowessResult;
    /** Finalize the stream and return remaining data. */
    finalize(): LowessResult;
}

/** Online LOWESS smoother for real-time data. */
export class OnlineLowess {
    free(): void;
    constructor(options?: SmoothOptions, onlineOpts?: OnlineOptions);
    /** Add a single point and get the smoothed value (or undefined if not enough points yet). */
    add_point(x: number, y: number): number | undefined;
}
"#;

use ::fastLowess::internals::adapters::online::ParallelOnlineLowess;
use ::fastLowess::internals::adapters::streaming::ParallelStreamingLowess;
use ::fastLowess::internals::api::{Batch, LowessBuilder, Online, Streaming};
use ::fastLowess::internals::binding_support as shared_parse;
use ::fastLowess::internals::binding_support::{MergeStrategy, UpdateMode};
use ::fastLowess::prelude::LowessResult as InnerLowessResult;

#[derive(Deserialize)]
pub struct SmoothOptions {
    pub fraction: Option<f64>,
    pub iterations: Option<usize>,
    pub delta: Option<f64>,
    pub weight_function: Option<String>,
    pub robustness_method: Option<String>,
    pub zero_weight_fallback: Option<String>,
    pub boundary_policy: Option<String>,
    pub scaling_method: Option<String>,
    pub auto_converge: Option<f64>,
    pub return_residuals: Option<bool>,
    pub return_robustness_weights: Option<bool>,
    pub return_diagnostics: Option<bool>,
    pub return_se: Option<bool>,
    pub confidence_intervals: Option<f64>,
    pub prediction_intervals: Option<f64>,
    #[serde(rename = "parallel")]
    pub parallel: Option<bool>,
    pub cv_fractions: Option<Vec<f64>>,
    pub cv_method: Option<String>,
    pub cv_k: Option<u32>,
    pub cv_seed: Option<u32>,
}

#[derive(Deserialize)]
pub struct StreamingOptions {
    pub chunk_size: Option<usize>,
    pub overlap: Option<usize>,
    pub merge_strategy: Option<String>,
}

#[derive(Deserialize)]
pub struct OnlineOptions {
    pub window_capacity: Option<usize>,
    pub min_points: Option<usize>,
    pub update_mode: Option<String>,
}

#[wasm_bindgen]
pub struct Diagnostics {
    pub rmse: f64,
    pub mae: f64,
    #[wasm_bindgen(js_name = r_squared)]
    pub r_squared: f64,
    pub aic: Option<f64>,
    pub aicc: Option<f64>,
    #[wasm_bindgen(js_name = effective_df)]
    pub effective_df: Option<f64>,
    #[wasm_bindgen(js_name = residual_sd)]
    pub residual_sd: f64,
}

#[wasm_bindgen]
pub struct LowessResult {
    inner: InnerLowessResult<f64>,
}

#[wasm_bindgen]
impl LowessResult {
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

    #[wasm_bindgen(getter, js_name = standard_errors)]
    pub fn standard_errors(&self) -> Option<Float64Array> {
        self.inner
            .standard_errors
            .as_ref()
            .map(|v| unsafe { Float64Array::view(v) })
    }

    #[wasm_bindgen(getter, js_name = confidence_lower)]
    pub fn confidence_lower(&self) -> Option<Float64Array> {
        self.inner
            .confidence_lower
            .as_ref()
            .map(|v| unsafe { Float64Array::view(v) })
    }

    #[wasm_bindgen(getter, js_name = confidence_upper)]
    pub fn confidence_upper(&self) -> Option<Float64Array> {
        self.inner
            .confidence_upper
            .as_ref()
            .map(|v| unsafe { Float64Array::view(v) })
    }

    #[wasm_bindgen(getter, js_name = prediction_lower)]
    pub fn prediction_lower(&self) -> Option<Float64Array> {
        self.inner
            .prediction_lower
            .as_ref()
            .map(|v| unsafe { Float64Array::view(v) })
    }

    #[wasm_bindgen(getter, js_name = prediction_upper)]
    pub fn prediction_upper(&self) -> Option<Float64Array> {
        self.inner
            .prediction_upper
            .as_ref()
            .map(|v| unsafe { Float64Array::view(v) })
    }

    #[wasm_bindgen(getter, js_name = robustness_weights)]
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

    #[wasm_bindgen(getter, js_name = cv_scores)]
    pub fn cv_scores(&self) -> Option<Float64Array> {
        self.inner
            .cv_scores
            .as_ref()
            .map(|v| unsafe { Float64Array::view(v) })
    }

    #[wasm_bindgen(getter, js_name = fraction_used)]
    pub fn fraction_used(&self) -> f64 {
        self.inner.fraction_used
    }

    #[wasm_bindgen(getter, js_name = iterations_used)]
    pub fn iterations_used(&self) -> Option<u32> {
        self.inner.iterations_used.map(|i| i as u32)
    }
}

// LOWESS smoother.
#[wasm_bindgen(skip_typescript)]
pub struct Lowess {
    options: JsValue,
}

#[wasm_bindgen]
impl Lowess {
    /// Create a new `Lowess` model with the given options.
    #[wasm_bindgen(constructor, skip_typescript)]
    pub fn new(options: JsValue) -> Lowess {
        Lowess { options }
    }

    /// Fit the model to data and return smoothed values.
    #[wasm_bindgen(skip_typescript)]
    #[allow(non_snake_case)]
    pub fn fit(
        &self,
        x: &Float64Array,
        y: &Float64Array,
        customWeights: Option<Box<[f64]>>,
    ) -> Result<LowessResult, JsValue> {
        smooth(
            x,
            y,
            self.options.clone(),
            customWeights.map(|b| b.to_vec()),
        )
    }
}

fn smooth(
    x: &Float64Array,
    y: &Float64Array,
    options: JsValue,
    custom_weights: Option<Vec<f64>>,
) -> Result<LowessResult, JsValue> {
    let opts = if !options.is_undefined() && !options.is_null() {
        Some(serde_wasm_bindgen::from_value::<SmoothOptions>(options)?)
    } else {
        None
    };
    let o = opts.as_ref();
    let mut builder = shared_parse::apply_builder_options(
        LowessBuilder::new(),
        shared_parse::BuilderOptionSet {
            fraction: o.and_then(|x| x.fraction),
            iterations: o.and_then(|x| x.iterations),
            delta: o.and_then(|x| x.delta),
            weight_function: o.and_then(|x| x.weight_function.as_deref()),
            robustness_method: o.and_then(|x| x.robustness_method.as_deref()),
            zero_weight_fallback: o.and_then(|x| x.zero_weight_fallback.as_deref()),
            boundary_policy: o.and_then(|x| x.boundary_policy.as_deref()),
            scaling_method: o.and_then(|x| x.scaling_method.as_deref()),
            auto_converge: o.and_then(|x| x.auto_converge),
            return_residuals: o.is_some_and(|x| x.return_residuals.unwrap_or(false)),
            return_robustness_weights: o
                .is_some_and(|x| x.return_robustness_weights.unwrap_or(false)),
            return_diagnostics: o.is_some_and(|x| x.return_diagnostics.unwrap_or(false)),
            return_se: o.is_some_and(|x| x.return_se.unwrap_or(false)),
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
            cv_seed: o.and_then(|x| x.cv_seed).map(|s| s as u64),
        },
    )
    .map_err(|e| JsValue::from_str(&e))?;

    if let Some(cw) = custom_weights {
        builder = builder.custom_weights(cw);
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

    Ok(LowessResult { inner: result })
}

// Streaming LOWESS smoother.
#[wasm_bindgen(skip_typescript)]
pub struct StreamingLowess {
    inner: ParallelStreamingLowess<f64>,
}

#[wasm_bindgen]
impl StreamingLowess {
    // Create a new smoother.
    #[wasm_bindgen(constructor, skip_typescript)]
    #[allow(non_snake_case)]
    pub fn new(options: JsValue, streamingOpts: JsValue) -> Result<StreamingLowess, JsValue> {
        let opts = if !options.is_undefined() && !options.is_null() {
            Some(serde_wasm_bindgen::from_value::<SmoothOptions>(options)?)
        } else {
            None
        };
        let sopts = if !streamingOpts.is_undefined() && !streamingOpts.is_null() {
            Some(serde_wasm_bindgen::from_value::<StreamingOptions>(
                streamingOpts,
            )?)
        } else {
            None
        };
        let o = opts.as_ref();
        let so = sopts.as_ref();
        let builder = shared_parse::apply_builder_options(
            LowessBuilder::new(),
            shared_parse::BuilderOptionSet {
                fraction: o.and_then(|x| x.fraction),
                iterations: o.and_then(|x| x.iterations),
                delta: o.and_then(|x| x.delta),
                weight_function: o.and_then(|x| x.weight_function.as_deref()),
                robustness_method: o.and_then(|x| x.robustness_method.as_deref()),
                zero_weight_fallback: o.and_then(|x| x.zero_weight_fallback.as_deref()),
                boundary_policy: o.and_then(|x| x.boundary_policy.as_deref()),
                scaling_method: o.and_then(|x| x.scaling_method.as_deref()),
                auto_converge: o.and_then(|x| x.auto_converge),
                return_residuals: o.is_some_and(|x| x.return_residuals.unwrap_or(false)),
                return_robustness_weights: o
                    .is_some_and(|x| x.return_robustness_weights.unwrap_or(false)),
                return_diagnostics: o.is_some_and(|x| x.return_diagnostics.unwrap_or(false)),
                return_se: false,
                confidence_intervals: None,
                prediction_intervals: None,
                parallel: o.and_then(|x| x.parallel),
                chunk_size: None,
                overlap: None,
                merge_strategy: None,
                window_capacity: None,
                min_points: None,
                update_mode: None,
                cv_fractions: None,
                cv_method: None,
                cv_k: None,
                cv_seed: None,
            },
        )
        .map_err(|e| JsValue::from_str(&e))?;

        let mut chunk_size = 5000_usize;
        let mut overlap = 500_usize;
        let mut merge_strategy = MergeStrategy::WeightedAverage;

        if let Some(s) = so {
            if let Some(cs) = s.chunk_size {
                chunk_size = cs;
            }
            if let Some(ov) = s.overlap {
                overlap = ov;
            }
            if let Some(ms) = &s.merge_strategy {
                merge_strategy =
                    shared_parse::parse_merge_strategy(ms).map_err(|e| JsValue::from_str(&e))?;
            }
        }

        let model = builder
            .adapter(Streaming)
            .chunk_size(chunk_size)
            .overlap(overlap)
            .merge_strategy(merge_strategy)
            .build()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(StreamingLowess { inner: model })
    }

    #[wasm_bindgen(js_name = process_chunk, skip_typescript)]
    pub fn process_chunk(
        &mut self,
        x: &Float64Array,
        y: &Float64Array,
    ) -> Result<LowessResult, JsValue> {
        let x_vec = x.to_vec();
        let y_vec = y.to_vec();
        let result: ::fastLowess::prelude::LowessResult<f64> = self
            .inner
            .process_chunk(&x_vec, &y_vec)
            .map_err(|e: ::fastLowess::prelude::LowessError| JsValue::from_str(&e.to_string()))?;
        Ok(LowessResult { inner: result })
    }

    #[wasm_bindgen(skip_typescript)]
    pub fn finalize(&mut self) -> Result<LowessResult, JsValue> {
        let result: ::fastLowess::prelude::LowessResult<f64> = self
            .inner
            .finalize()
            .map_err(|e: ::fastLowess::prelude::LowessError| JsValue::from_str(&e.to_string()))?;
        Ok(LowessResult { inner: result })
    }
}

// Online LOWESS smoother.
#[wasm_bindgen(skip_typescript)]
pub struct OnlineLowess {
    inner: ParallelOnlineLowess<f64>,
}

#[wasm_bindgen]
impl OnlineLowess {
    // Create a new smoother.
    #[wasm_bindgen(constructor, skip_typescript)]
    #[allow(non_snake_case)]
    pub fn new(options: JsValue, onlineOpts: JsValue) -> Result<OnlineLowess, JsValue> {
        let opts = if !options.is_undefined() && !options.is_null() {
            Some(serde_wasm_bindgen::from_value::<SmoothOptions>(options)?)
        } else {
            None
        };
        let oopts = if !onlineOpts.is_undefined() && !onlineOpts.is_null() {
            Some(serde_wasm_bindgen::from_value::<OnlineOptions>(onlineOpts)?)
        } else {
            None
        };
        let o = opts.as_ref();
        let oo = oopts.as_ref();
        let builder = shared_parse::apply_builder_options(
            LowessBuilder::new(),
            shared_parse::BuilderOptionSet {
                fraction: o.and_then(|x| x.fraction),
                iterations: o.and_then(|x| x.iterations),
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
                window_capacity: None,
                min_points: None,
                update_mode: None,
                cv_fractions: None,
                cv_method: None,
                cv_k: None,
                cv_seed: None,
            },
        )
        .map_err(|e| JsValue::from_str(&e))?;

        let mut window_capacity = 100_usize;
        let mut min_points = 2_usize;
        let mut update_mode = UpdateMode::Full;

        if let Some(online_o) = oo {
            if let Some(wc) = online_o.window_capacity {
                window_capacity = wc;
            }
            if let Some(mp) = online_o.min_points {
                min_points = mp;
            }
            if let Some(um) = &online_o.update_mode {
                update_mode =
                    shared_parse::parse_update_mode(um).map_err(|e| JsValue::from_str(&e))?;
            }
        }

        let model = builder
            .adapter(Online)
            .window_capacity(window_capacity)
            .min_points(min_points)
            .update_mode(update_mode)
            .build()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(OnlineLowess { inner: model })
    }

    #[wasm_bindgen(skip_typescript)]
    pub fn add_point(&mut self, x: f64, y: f64) -> Result<Option<f64>, JsValue> {
        let result = self
            .inner
            .add_point(x, y)
            .map_err(|e: ::fastLowess::prelude::LowessError| JsValue::from_str(&e.to_string()))?;
        Ok(result.map(|o| o.smoothed))
    }
}
