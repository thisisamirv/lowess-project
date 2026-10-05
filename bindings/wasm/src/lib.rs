//! WebAssembly bindings for fastLowess.

use js_sys::{Float64Array, Object, Reflect};
use serde::Deserialize;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

/// Returns the version of this WebAssembly binding package.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_owned()
}

// ============================================================================
// TypeScript interface declarations injected into the generated .d.ts
// ============================================================================

#[wasm_bindgen(typescript_custom_section)]
const TS_TYPES: &'static str = r#"
/** Grouped confidence/prediction levels and optional residual-bootstrap refits. */
export interface IntervalsOptions {
    /** Confidence level for the mean response (e.g. 0.95). Disabled when absent. */
    confidence?: number;
    /** Prediction level for a new observation (e.g. 0.95). Disabled when absent. */
    prediction?: number;
    /** Residual-bootstrap refits (at least 2); 0 or absent keeps analytic intervals. */
    bootstrap?: number;
}

/** Grouped cross-validation configuration. Seed k-fold shuffling with the outer `seed`. */
export interface CVOptions {
    /** CV method ("kfold" or "loocv"). Default: "kfold". */
    method?: string;
    /** Number of folds for k-fold CV. Default: 5. */
    k?: number;
    /** Candidate smoothing fractions. */
    fractions: number[];
}

/** Configuration options for LOWESS smoothing. */
export interface SmoothOptions {
    /** Optional output components: se, diagnostics, residuals, weights, derivative, sorted. */
    outputs?: string[];
    /** Grouped confidence/prediction levels and optional residual-bootstrap refits. */
    intervals?: IntervalsOptions;
    /** Grouped cross-validation configuration. */
    cv?: CVOptions;
    /** Shared seed for k-fold CV and residual bootstrap; 0 is valid. Does not enable either by itself. */
    seed?: number;
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
    /** Enable parallel execution. Ignored by OnlineLowess (it processes one point at a time). Default: true. */
    parallel?: boolean;
    /** Policy for non-finite (NaN/Inf) values in input data ("error", "drop"). Default: "error". */
    missing?: string;
    /** Retain the fitted model's training data, enabling `LowessResult.predict()`. Batch (Lowess) only. Default: false. */
    retain_model?: boolean;
}

/** Options for `LowessResult.predict()`. */
export interface PredictOptions {
    /** Optional prediction components: se and/or derivative. */
    outputs?: string[];
    /** Grouped confidence/prediction levels and optional residual-bootstrap refits. */
    intervals?: IntervalsOptions;
    /** Prediction-time bootstrap seed, independent of the fit seed. */
    seed?: number;
    /** Behavior for query points outside the training range ("clamp", "linear", "error"). Default: "clamp". */
    extrapolation?: string;
    /** Under "linear" extrapolation, the maximum allowed distance beyond the training boundary before `predict()` errors instead of returning an unbounded value. */
    max_extrapolation_distance?: number;
    /** Maximum allowed distance to the farthest point in a query's local window before `predict()` errors, catching in-range-but-sparse query points. */
    max_neighbor_distance?: number;
}

/** Result of `LowessResult.predict()`. */
export interface PredictOutput {
    /** Release the WASM result allocation. */
    free(): void;
    /** Predicted y values, one per query point. */
    readonly y: Float64Array;
    /** Standard errors (if requested). */
    readonly standard_errors: Float64Array | undefined;
    /** Lower confidence interval bounds (if requested). */
    readonly confidence_lower: Float64Array | undefined;
    /** Upper confidence interval bounds (if requested). */
    readonly confidence_upper: Float64Array | undefined;
    /** Lower prediction interval bounds (if requested). */
    readonly prediction_lower: Float64Array | undefined;
    /** Upper prediction interval bounds (if requested). */
    readonly prediction_upper: Float64Array | undefined;
    /** Local fit's derivative (slope) at each query point (if requested). */
    readonly derivative: Float64Array | undefined;
}

export interface LowessResult {
    /** Evaluate the fitted model at out-of-sample query points. Requires `retain_model: true`. */
    predict(newX: Float64Array, options?: PredictOptions): PredictOutput;
}

/** Configuration options for streaming LOWESS smoothing. A subset of `SmoothOptions`: cross-validation and the `sorted` output have no equivalent here. */
export interface StreamingSmoothOptions {
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
    /** Optional output components: se, diagnostics, residuals, weights, derivative. */
    outputs?: string[];
    /** Confidence/prediction levels and per-chunk residual-bootstrap refits. */
    intervals?: IntervalsOptions;
    /** Bootstrap seed; each combined chunk restarts from it. */
    seed?: number;
    /** Enable parallel execution. Default: true. */
    parallel?: boolean;
    /** Policy for non-finite (NaN/Inf) values in each chunk ("error", "drop"). Default: "error". */
    missing?: string;
}

/** Configuration options for online LOWESS smoothing. A subset of `SmoothOptions`: diagnostics, residuals, parallel execution, cross-validation, and the `sorted` output have no equivalent here. The `se` output and `intervals` require `update_mode = "full"`. */
export interface OnlineSmoothOptions {
    /** Smoothing fraction (0 < fraction <= 1). Default: 0.67. */
    fraction?: number;
    /** Number of robustness iterations. Requires `update_mode = "full"`. Default: 0. */
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
    /** Optional output components: se, weights, derivative. */
    outputs?: string[];
    /** Confidence/prediction levels and per-window residual-bootstrap refits. Requires `update_mode = "full"`. */
    intervals?: IntervalsOptions;
    /** Bootstrap seed; each full-update window restarts from it. */
    seed?: number;
    /** Policy for non-finite (NaN/Inf) `x`/`y` values passed to `add_point` ("error", "drop"). Default: "error". */
    missing?: string;
}

/** Configuration options for streaming LOWESS. */
export interface StreamingOptions {
    /** Size of each processing chunk. Default: 5000. */
    chunk_size?: number;
    /** Overlap between adjacent chunks. Default: chunk_size / 10, min. 1. */
    overlap?: number;
    /** Strategy for merging chunks (\"average\", \"weighted_average\", \"take_first\", \"take_last\"). Default: \"weighted_average\". */
    merge_strategy?: string;
}

/** Configuration options for online LOWESS. */
export interface OnlineOptions {
    /** Maximum number of points to retain in the sliding window. Default: 1000. */
    window_capacity?: number;
    /** Minimum points required before smoothing starts. Default: 2. */
    min_points?: number;
    /** Update strategy ("full" or "incremental"). Default: "incremental". */
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
    constructor(options?: StreamingSmoothOptions, streamingOpts?: StreamingOptions);
    /** Process a chunk of data. */
    process_chunk(x: Float64Array, y: Float64Array): LowessResult;
    /** Finalize the stream and return remaining data. */
    finalize(): LowessResult;
}

/** Online LOWESS smoother for real-time data. */
export class OnlineLowess {
    free(): void;
    constructor(options?: OnlineSmoothOptions, onlineOpts?: OnlineOptions);
    /** Add a single point and get the smoothed output (or null if not enough points yet). */
    add_point(x: number, y: number): OnlineOutput | null;
}

/** Result from a single online update step. */
export class OnlineOutput {
    free(): void;
    get y(): number;
    get standard_error(): number | undefined;
    get residual(): number | undefined;
    get robustness_weight(): number | undefined;
    get iterations_used(): number | undefined;
    get derivative(): number | undefined;
    get confidence_lower(): number | undefined;
    get confidence_upper(): number | undefined;
    get prediction_lower(): number | undefined;
    get prediction_upper(): number | undefined;
}
"#;

use ::fastLowess::internals::LowessBuilder;
use ::fastLowess::internals::adapters::online::ParallelOnlineLowess;
use ::fastLowess::internals::adapters::streaming::ParallelStreamingLowess;
use ::fastLowess::internals::binding_support as shared_parse;
use ::fastLowess::prelude::IntervalsBuilder;
use ::fastLowess::prelude::LowessResult as InnerLowessResult;

fn to_js_error(err: shared_parse::BindingError) -> JsValue {
    JsValue::from_str(&err.message)
}

fn map_invalid_arg<T, E: ToString>(result: Result<T, E>) -> Result<T, JsValue> {
    shared_parse::map_invalid_arg(result).map_err(to_js_error)
}

fn map_runtime<T, E: ToString>(result: Result<T, E>) -> Result<T, JsValue> {
    shared_parse::map_runtime(result).map_err(to_js_error)
}

fn has_output(outputs: Option<&Vec<String>>, name: &str) -> bool {
    outputs.is_some_and(|values| values.iter().any(|value| value == name))
}

fn validate_outputs(outputs: Option<&Vec<String>>, allowed: &[&str]) -> Result<(), JsValue> {
    if let Some(output) = outputs
        .into_iter()
        .flatten()
        .find(|value| !allowed.contains(&value.as_str()))
    {
        return Err(JsValue::from_str(&format!(
            "unknown output '{output}'. Valid outputs: {}",
            allowed.join(", ")
        )));
    }
    Ok(())
}

fn to_float64_array(values: &[f64]) -> Float64Array {
    Float64Array::from(values)
}

fn validate_option_keys(value: &JsValue, name: &str, allowed: &[&str]) -> Result<(), JsValue> {
    if value.is_undefined() || value.is_null() || !value.is_object() {
        return Ok(());
    }

    let object: &Object = value.unchecked_ref();
    for key in Object::keys(object).iter() {
        if let Some(key) = key.as_string()
            && !allowed.contains(&key.as_str())
        {
            return Err(JsValue::from_str(&format!(
                "unknown {name} option '{key}'. Valid options: {}",
                allowed.join(", ")
            )));
        }
    }
    Ok(())
}

fn validate_nested_option_keys(
    value: &JsValue,
    parent: &str,
    key: &str,
    allowed: &[&str],
) -> Result<(), JsValue> {
    if value.is_undefined() || value.is_null() || !value.is_object() {
        return Ok(());
    }
    let nested = Reflect::get(value, &JsValue::from_str(key))?;
    validate_option_keys(&nested, parent, allowed)
}

fn bootstrap_count(intervals: Option<&IntervalsOptionsJs>) -> Option<usize> {
    intervals.and_then(|iv| iv.bootstrap).filter(|&n| n > 0)
}

fn apply_bootstrap_and_seed(
    mut builder: LowessBuilder<f64>,
    intervals: Option<&IntervalsOptionsJs>,
    seed: Option<u64>,
) -> LowessBuilder<f64> {
    if let Some(n_boot) = bootstrap_count(intervals) {
        builder = builder.intervals(IntervalsBuilder::new().bootstrap(n_boot));
    }
    if let Some(seed) = seed {
        builder = builder.seed(seed);
    }
    builder
}

#[derive(Deserialize, Default)]
pub struct IntervalsOptionsJs {
    pub confidence: Option<f64>,
    pub prediction: Option<f64>,
    pub bootstrap: Option<usize>,
}

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
    pub outputs: Option<Vec<String>>,
    pub intervals: Option<IntervalsOptionsJs>,
    pub cv: Option<CVOptionsJs>,
    pub seed: Option<u64>,
    #[serde(rename = "parallel")]
    pub parallel: Option<bool>,
    pub missing: Option<String>,
    pub retain_model: Option<bool>,
}

#[derive(Deserialize)]
pub struct CVOptionsJs {
    pub method: Option<String>,
    pub k: Option<u32>,
    pub fractions: Vec<f64>,
}

#[derive(Deserialize, Default)]
pub struct PredictOptionsJs {
    pub outputs: Option<Vec<String>>,
    pub intervals: Option<IntervalsOptionsJs>,
    pub seed: Option<u64>,
    pub extrapolation: Option<String>,
    pub max_extrapolation_distance: Option<f64>,
    pub max_neighbor_distance: Option<f64>,
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

#[derive(Deserialize)]
pub struct StreamingSmoothOptions {
    pub fraction: Option<f64>,
    pub iterations: Option<usize>,
    pub delta: Option<f64>,
    pub weight_function: Option<String>,
    pub robustness_method: Option<String>,
    pub zero_weight_fallback: Option<String>,
    pub boundary_policy: Option<String>,
    pub scaling_method: Option<String>,
    pub auto_converge: Option<f64>,
    pub outputs: Option<Vec<String>>,
    pub intervals: Option<IntervalsOptionsJs>,
    pub seed: Option<u64>,
    pub parallel: Option<bool>,
    pub missing: Option<String>,
}

#[derive(Deserialize)]
pub struct OnlineSmoothOptions {
    pub fraction: Option<f64>,
    pub iterations: Option<usize>,
    pub delta: Option<f64>,
    pub weight_function: Option<String>,
    pub robustness_method: Option<String>,
    pub zero_weight_fallback: Option<String>,
    pub boundary_policy: Option<String>,
    pub scaling_method: Option<String>,
    pub auto_converge: Option<f64>,
    pub outputs: Option<Vec<String>>,
    pub intervals: Option<IntervalsOptionsJs>,
    pub seed: Option<u64>,
    pub missing: Option<String>,
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
    /// Batch: robust residual scale estimate (1.4826 * MAD); Streaming: cumulative sample SD of emitted residuals.
    #[wasm_bindgen(js_name = residual_sd)]
    pub residual_sd: f64,
}

// Result of a single online update step.
#[wasm_bindgen]
pub struct OnlineOutput {
    y: f64,
    standard_error: Option<f64>,
    residual: Option<f64>,
    robustness_weight: Option<f64>,
    iterations_used: Option<usize>,
    derivative: Option<f64>,
    confidence_lower: Option<f64>,
    confidence_upper: Option<f64>,
    prediction_lower: Option<f64>,
    prediction_upper: Option<f64>,
}

#[wasm_bindgen]
impl OnlineOutput {
    #[wasm_bindgen(getter)]
    pub fn y(&self) -> f64 {
        self.y
    }

    #[wasm_bindgen(getter, js_name = "standard_error")]
    pub fn standard_error(&self) -> Option<f64> {
        self.standard_error
    }

    #[wasm_bindgen(getter)]
    pub fn residual(&self) -> Option<f64> {
        self.residual
    }

    #[wasm_bindgen(getter, js_name = "robustness_weight")]
    pub fn robustness_weight(&self) -> Option<f64> {
        self.robustness_weight
    }

    #[wasm_bindgen(getter, js_name = "iterations_used")]
    pub fn iterations_used(&self) -> Option<u32> {
        self.iterations_used.map(|i| i as u32)
    }

    #[wasm_bindgen(getter)]
    pub fn derivative(&self) -> Option<f64> {
        self.derivative
    }

    #[wasm_bindgen(getter, js_name = "confidence_lower")]
    pub fn confidence_lower(&self) -> Option<f64> {
        self.confidence_lower
    }

    #[wasm_bindgen(getter, js_name = "confidence_upper")]
    pub fn confidence_upper(&self) -> Option<f64> {
        self.confidence_upper
    }

    #[wasm_bindgen(getter, js_name = "prediction_lower")]
    pub fn prediction_lower(&self) -> Option<f64> {
        self.prediction_lower
    }

    #[wasm_bindgen(getter, js_name = "prediction_upper")]
    pub fn prediction_upper(&self) -> Option<f64> {
        self.prediction_upper
    }
}

#[wasm_bindgen]
pub struct LowessResult {
    inner: InnerLowessResult<f64>,
}

#[wasm_bindgen]
impl LowessResult {
    #[wasm_bindgen(getter)]
    pub fn x(&self) -> Float64Array {
        to_float64_array(&self.inner.x)
    }

    #[wasm_bindgen(getter)]
    pub fn y(&self) -> Float64Array {
        to_float64_array(&self.inner.y)
    }

    #[wasm_bindgen(getter)]
    pub fn residuals(&self) -> Option<Float64Array> {
        self.inner.residuals.as_ref().map(|v| to_float64_array(v))
    }

    #[wasm_bindgen(getter, js_name = standard_errors)]
    pub fn standard_errors(&self) -> Option<Float64Array> {
        self.inner
            .standard_errors
            .as_ref()
            .map(|v| to_float64_array(v))
    }

    #[wasm_bindgen(getter, js_name = confidence_lower)]
    pub fn confidence_lower(&self) -> Option<Float64Array> {
        self.inner
            .confidence_lower
            .as_ref()
            .map(|v| to_float64_array(v))
    }

    #[wasm_bindgen(getter, js_name = confidence_upper)]
    pub fn confidence_upper(&self) -> Option<Float64Array> {
        self.inner
            .confidence_upper
            .as_ref()
            .map(|v| to_float64_array(v))
    }

    #[wasm_bindgen(getter, js_name = prediction_lower)]
    pub fn prediction_lower(&self) -> Option<Float64Array> {
        self.inner
            .prediction_lower
            .as_ref()
            .map(|v| to_float64_array(v))
    }

    #[wasm_bindgen(getter, js_name = prediction_upper)]
    pub fn prediction_upper(&self) -> Option<Float64Array> {
        self.inner
            .prediction_upper
            .as_ref()
            .map(|v| to_float64_array(v))
    }

    #[wasm_bindgen(getter, js_name = robustness_weights)]
    pub fn robustness_weights(&self) -> Option<Float64Array> {
        self.inner
            .robustness_weights
            .as_ref()
            .map(|v| to_float64_array(v))
    }

    #[wasm_bindgen(getter)]
    pub fn derivative(&self) -> Option<Float64Array> {
        self.inner.derivative.as_ref().map(|v| to_float64_array(v))
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
        self.inner.cv_scores.as_ref().map(|v| to_float64_array(v))
    }

    #[wasm_bindgen(getter, js_name = fraction_used)]
    pub fn fraction_used(&self) -> f64 {
        self.inner.fraction_used
    }

    #[wasm_bindgen(getter, js_name = iterations_used)]
    pub fn iterations_used(&self) -> Option<u32> {
        self.inner.iterations_used.map(|i| i as u32)
    }

    /// Evaluate the fitted model at out-of-sample query points not in the training set.
    ///
    /// Requires `retain_model: true` to have been set on the builder before `fit()`.
    #[wasm_bindgen(skip_typescript)]
    pub fn predict(
        &self,
        new_x: &Float64Array,
        options: JsValue,
    ) -> Result<PredictOutput, JsValue> {
        validate_option_keys(
            &options,
            "prediction",
            &[
                "outputs",
                "intervals",
                "seed",
                "extrapolation",
                "max_extrapolation_distance",
                "max_neighbor_distance",
            ],
        )?;
        validate_nested_option_keys(
            &options,
            "intervals",
            "intervals",
            &["confidence", "prediction", "bootstrap"],
        )?;
        let opts: PredictOptionsJs = if options.is_undefined() || options.is_null() {
            PredictOptionsJs::default()
        } else {
            serde_wasm_bindgen::from_value(options)?
        };
        validate_outputs(opts.outputs.as_ref(), &["se", "derivative"])?;
        let new_x_vec = new_x.to_vec();
        let intervals = opts.intervals.as_ref();
        let query = shared_parse::build_predict_options_with_bootstrap(
            shared_parse::PredictOptionSet {
                return_se: has_output(opts.outputs.as_ref(), "se"),
                confidence_level: intervals.and_then(|iv| iv.confidence),
                prediction_level: intervals.and_then(|iv| iv.prediction),
                return_derivative: has_output(opts.outputs.as_ref(), "derivative"),
                extrapolation: opts.extrapolation.as_deref(),
                max_extrapolation_distance: opts.max_extrapolation_distance,
                max_neighbor_distance: opts.max_neighbor_distance,
            },
            bootstrap_count(intervals),
            opts.seed,
        )
        .map_err(to_js_error)?;
        let output = map_invalid_arg(query.call(&self.inner, &new_x_vec))?;
        Ok(PredictOutput { inner: output })
    }
}

/// Result of `LowessResult.predict()`.
#[wasm_bindgen(skip_typescript)]
pub struct PredictOutput {
    inner: shared_parse::PredictOutput<f64>,
}

#[wasm_bindgen]
impl PredictOutput {
    #[wasm_bindgen(getter)]
    pub fn y(&self) -> Float64Array {
        to_float64_array(&self.inner.y)
    }

    #[wasm_bindgen(getter, js_name = standard_errors)]
    pub fn standard_errors(&self) -> Option<Float64Array> {
        self.inner
            .standard_errors
            .as_ref()
            .map(|v| to_float64_array(v))
    }

    #[wasm_bindgen(getter, js_name = confidence_lower)]
    pub fn confidence_lower(&self) -> Option<Float64Array> {
        self.inner
            .confidence_lower
            .as_ref()
            .map(|v| to_float64_array(v))
    }

    #[wasm_bindgen(getter, js_name = confidence_upper)]
    pub fn confidence_upper(&self) -> Option<Float64Array> {
        self.inner
            .confidence_upper
            .as_ref()
            .map(|v| to_float64_array(v))
    }

    #[wasm_bindgen(getter, js_name = prediction_lower)]
    pub fn prediction_lower(&self) -> Option<Float64Array> {
        self.inner
            .prediction_lower
            .as_ref()
            .map(|v| to_float64_array(v))
    }

    #[wasm_bindgen(getter, js_name = prediction_upper)]
    pub fn prediction_upper(&self) -> Option<Float64Array> {
        self.inner
            .prediction_upper
            .as_ref()
            .map(|v| to_float64_array(v))
    }

    #[wasm_bindgen(getter)]
    pub fn derivative(&self) -> Option<Float64Array> {
        self.inner.derivative.as_ref().map(|v| to_float64_array(v))
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

/// Build a `LowessBuilder` from Batch options, applying every field.
fn batch_options_to_builder(opts: Option<SmoothOptions>) -> Result<LowessBuilder<f64>, JsValue> {
    let mut builder = LowessBuilder::<f64>::new();
    if let Some(opts) = opts {
        validate_outputs(
            opts.outputs.as_ref(),
            &[
                "diagnostics",
                "residuals",
                "weights",
                "derivative",
                "se",
                "sorted",
            ],
        )?;
        let cv = opts.cv.as_ref();
        let intervals = opts.intervals.as_ref();
        builder = map_invalid_arg(shared_parse::apply_builder_options(
            builder,
            shared_parse::BuilderOptionSet {
                fraction: opts.fraction,
                iterations: opts.iterations,
                delta: opts.delta,
                weight_function: opts.weight_function.as_deref(),
                robustness_method: opts.robustness_method.as_deref(),
                zero_weight_fallback: opts.zero_weight_fallback.as_deref(),
                boundary_policy: opts.boundary_policy.as_deref(),
                scaling_method: opts.scaling_method.as_deref(),
                auto_converge: opts.auto_converge,
                return_residuals: has_output(opts.outputs.as_ref(), "residuals"),
                return_robustness_weights: has_output(opts.outputs.as_ref(), "weights"),
                return_diagnostics: has_output(opts.outputs.as_ref(), "diagnostics"),
                return_se: has_output(opts.outputs.as_ref(), "se"),
                return_sorted: has_output(opts.outputs.as_ref(), "sorted"),
                confidence_intervals: intervals.and_then(|iv| iv.confidence),
                prediction_intervals: intervals.and_then(|iv| iv.prediction),
                parallel: opts.parallel,
                backend: None,
                missing: opts.missing.as_deref(),
                cv_fractions: cv.map(|value| value.fractions.as_slice()),
                cv_method: cv.and_then(|value| value.method.as_deref()),
                cv_k: cv.and_then(|value| value.k).map(|value| value as usize),
                retain_model: opts.retain_model,
                ..Default::default()
            },
        ))?;
        if has_output(opts.outputs.as_ref(), "derivative") {
            builder = builder.return_derivative();
        }
        builder = apply_bootstrap_and_seed(builder, intervals, opts.seed);
    }
    Ok(builder)
}

/// Build a `LowessBuilder` from Streaming options, applying every field.
fn streaming_options_to_builder(
    opts: Option<StreamingSmoothOptions>,
) -> Result<LowessBuilder<f64>, JsValue> {
    let mut builder = LowessBuilder::<f64>::new();
    if let Some(opts) = opts {
        validate_outputs(
            opts.outputs.as_ref(),
            &["diagnostics", "residuals", "weights", "derivative", "se"],
        )?;
        builder = map_invalid_arg(shared_parse::apply_builder_options(
            builder,
            shared_parse::BuilderOptionSet {
                fraction: opts.fraction,
                iterations: opts.iterations,
                delta: opts.delta,
                weight_function: opts.weight_function.as_deref(),
                robustness_method: opts.robustness_method.as_deref(),
                zero_weight_fallback: opts.zero_weight_fallback.as_deref(),
                boundary_policy: opts.boundary_policy.as_deref(),
                scaling_method: opts.scaling_method.as_deref(),
                auto_converge: opts.auto_converge,
                return_residuals: has_output(opts.outputs.as_ref(), "residuals"),
                return_robustness_weights: has_output(opts.outputs.as_ref(), "weights"),
                return_diagnostics: has_output(opts.outputs.as_ref(), "diagnostics"),
                return_se: has_output(opts.outputs.as_ref(), "se"),
                confidence_intervals: opts.intervals.as_ref().and_then(|iv| iv.confidence),
                prediction_intervals: opts.intervals.as_ref().and_then(|iv| iv.prediction),
                parallel: opts.parallel,
                missing: opts.missing.as_deref(),
                ..Default::default()
            },
        ))?;
        if has_output(opts.outputs.as_ref(), "derivative") {
            builder = builder.return_derivative();
        }
        builder = apply_bootstrap_and_seed(builder, opts.intervals.as_ref(), opts.seed);
    }
    Ok(builder)
}

/// Build a `LowessBuilder` from Online options, applying every field.
fn online_options_to_builder(
    opts: Option<OnlineSmoothOptions>,
) -> Result<LowessBuilder<f64>, JsValue> {
    let mut builder = LowessBuilder::<f64>::new();
    if let Some(opts) = opts {
        validate_outputs(opts.outputs.as_ref(), &["weights", "derivative", "se"])?;
        builder = map_invalid_arg(shared_parse::apply_builder_options(
            builder,
            shared_parse::BuilderOptionSet {
                fraction: opts.fraction,
                iterations: opts.iterations,
                delta: opts.delta,
                weight_function: opts.weight_function.as_deref(),
                robustness_method: opts.robustness_method.as_deref(),
                zero_weight_fallback: opts.zero_weight_fallback.as_deref(),
                boundary_policy: opts.boundary_policy.as_deref(),
                scaling_method: opts.scaling_method.as_deref(),
                auto_converge: opts.auto_converge,
                return_robustness_weights: has_output(opts.outputs.as_ref(), "weights"),
                return_se: has_output(opts.outputs.as_ref(), "se"),
                confidence_intervals: opts.intervals.as_ref().and_then(|iv| iv.confidence),
                prediction_intervals: opts.intervals.as_ref().and_then(|iv| iv.prediction),
                missing: opts.missing.as_deref(),
                ..Default::default()
            },
        ))?;
        if has_output(opts.outputs.as_ref(), "derivative") {
            builder = builder.return_derivative();
        }
        builder = apply_bootstrap_and_seed(builder, opts.intervals.as_ref(), opts.seed);
    }
    Ok(builder)
}

fn smooth(
    x: &Float64Array,
    y: &Float64Array,
    options: JsValue,
    custom_weights: Option<Vec<f64>>,
) -> Result<LowessResult, JsValue> {
    validate_option_keys(
        &options,
        "batch",
        &[
            "fraction",
            "iterations",
            "delta",
            "weight_function",
            "robustness_method",
            "zero_weight_fallback",
            "boundary_policy",
            "scaling_method",
            "auto_converge",
            "outputs",
            "intervals",
            "cv",
            "seed",
            "parallel",
            "missing",
            "retain_model",
        ],
    )?;
    validate_nested_option_keys(
        &options,
        "intervals",
        "intervals",
        &["confidence", "prediction", "bootstrap"],
    )?;
    validate_nested_option_keys(&options, "cv", "cv", &["method", "k", "fractions"])?;
    let opts = if !options.is_undefined() && !options.is_null() {
        Some(serde_wasm_bindgen::from_value::<SmoothOptions>(options)?)
    } else {
        None
    };
    let builder = batch_options_to_builder(opts)?;

    let x_vec = x.to_vec();
    let y_vec = y.to_vec();

    let model = map_runtime(shared_parse::build_batch(builder, custom_weights))?;
    let result = map_runtime(model.fit(&x_vec, &y_vec))?;

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
        validate_option_keys(
            &options,
            "streaming",
            &[
                "fraction",
                "iterations",
                "delta",
                "weight_function",
                "robustness_method",
                "zero_weight_fallback",
                "boundary_policy",
                "scaling_method",
                "auto_converge",
                "outputs",
                "intervals",
                "seed",
                "parallel",
                "missing",
            ],
        )?;
        validate_nested_option_keys(
            &options,
            "intervals",
            "intervals",
            &["confidence", "prediction", "bootstrap"],
        )?;
        validate_option_keys(
            &streamingOpts,
            "streamingOpts",
            &["chunk_size", "overlap", "merge_strategy"],
        )?;
        let opts = if !options.is_undefined() && !options.is_null() {
            Some(serde_wasm_bindgen::from_value::<StreamingSmoothOptions>(
                options,
            )?)
        } else {
            None
        };
        let builder = streaming_options_to_builder(opts)?;

        let (chunk_size, overlap, merge_strategy) =
            if !streamingOpts.is_undefined() && !streamingOpts.is_null() {
                let sopts: StreamingOptions = serde_wasm_bindgen::from_value(streamingOpts)?;
                (sopts.chunk_size, sopts.overlap, sopts.merge_strategy)
            } else {
                (None, None, None)
            };

        let model = map_runtime(shared_parse::build_streaming(
            builder,
            chunk_size,
            overlap,
            merge_strategy.as_deref(),
        ))?;

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
        let result: InnerLowessResult<f64> = map_runtime(self.inner.process_chunk(&x_vec, &y_vec))?;
        Ok(LowessResult { inner: result })
    }

    #[wasm_bindgen(skip_typescript)]
    pub fn finalize(&mut self) -> Result<LowessResult, JsValue> {
        let result: InnerLowessResult<f64> = map_runtime(self.inner.finalize())?;
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
        validate_option_keys(
            &options,
            "online",
            &[
                "fraction",
                "iterations",
                "delta",
                "weight_function",
                "robustness_method",
                "zero_weight_fallback",
                "boundary_policy",
                "scaling_method",
                "auto_converge",
                "outputs",
                "intervals",
                "seed",
                "missing",
            ],
        )?;
        validate_nested_option_keys(
            &options,
            "intervals",
            "intervals",
            &["confidence", "prediction", "bootstrap"],
        )?;
        validate_option_keys(
            &onlineOpts,
            "onlineOpts",
            &["window_capacity", "min_points", "update_mode"],
        )?;
        let opts = if !options.is_undefined() && !options.is_null() {
            Some(serde_wasm_bindgen::from_value::<OnlineSmoothOptions>(
                options,
            )?)
        } else {
            None
        };
        let builder = online_options_to_builder(opts)?;

        let (window_capacity, min_points, update_mode) =
            if !onlineOpts.is_undefined() && !onlineOpts.is_null() {
                let oopts: OnlineOptions = serde_wasm_bindgen::from_value(onlineOpts)?;
                (oopts.window_capacity, oopts.min_points, oopts.update_mode)
            } else {
                (None, None, None)
            };

        let model = map_runtime(shared_parse::build_online(
            builder,
            window_capacity,
            min_points,
            update_mode.as_deref(),
        ))?;

        Ok(OnlineLowess { inner: model })
    }

    #[wasm_bindgen(js_name = "add_point", skip_typescript)]
    pub fn add_point(&mut self, x: f64, y: f64) -> Result<JsValue, JsValue> {
        let output = map_invalid_arg(self.inner.add_point(x, y))?;
        Ok(match output {
            Some(o) => JsValue::from(OnlineOutput {
                y: o.y,
                standard_error: o.standard_error,
                residual: o.residual,
                robustness_weight: o.robustness_weight,
                iterations_used: o.iterations_used,
                derivative: o.derivative,
                confidence_lower: o.confidence_lower,
                confidence_upper: o.confidence_upper,
                prediction_lower: o.prediction_lower,
                prediction_upper: o.prediction_upper,
            }),
            None => JsValue::null(),
        })
    }
}
