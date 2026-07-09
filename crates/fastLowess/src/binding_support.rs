//! Shared helpers for language bindings.
//!
//! This module centralizes string option parsing used by C/C++, Julia,
//! Node.js, Python, R, and WASM bindings so option aliases and validation
//! behavior stay consistent across all binding frontends.

use crate::adapters::batch::ParallelBatchLowessBuilder;
use crate::adapters::online::ParallelOnlineLowessBuilder;
use crate::adapters::streaming::ParallelStreamingLowessBuilder;
use crate::api::{LowessBuilder, LowessError};
use crate::parse::IntoEnum;
use lowess::internals::evaluation::intervals::IntervalMethod;
pub use lowess::internals::primitives::backend::Backend;
use num_traits::Float;

pub use lowess::internals::adapters::online::UpdateMode;
pub use lowess::internals::adapters::streaming::MergeStrategy;
pub use lowess::internals::algorithms::regression::ZeroWeightFallback;
pub use lowess::internals::algorithms::robustness::RobustnessMethod;
pub use lowess::internals::math::boundary::BoundaryPolicy;
pub use lowess::internals::math::kernel::WeightFunction;
pub use lowess::internals::math::scaling::ScalingMethod;
use std::ffi::CString;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindingErrorCategory {
    InvalidArg,
    Runtime,
}

#[derive(Debug, Clone)]
pub struct BindingError {
    pub category: BindingErrorCategory,
    pub message: String,
}

impl BindingError {
    pub fn invalid_arg(msg: impl Into<String>) -> Self {
        Self {
            category: BindingErrorCategory::InvalidArg,
            message: msg.into(),
        }
    }

    pub fn runtime(msg: impl Into<String>) -> Self {
        Self {
            category: BindingErrorCategory::Runtime,
            message: msg.into(),
        }
    }
}

pub fn map_invalid_arg<T, E: ToString>(result: Result<T, E>) -> Result<T, BindingError> {
    result.map_err(|e| BindingError::invalid_arg(e.to_string()))
}

pub fn map_runtime<T, E: ToString>(result: Result<T, E>) -> Result<T, BindingError> {
    result.map_err(|e| BindingError::runtime(e.to_string()))
}

pub const PANIC_FALLBACK_MESSAGE: &str = "Panic in Rust library";
pub const CONFIG_POINTER_IS_NULL: &str = "Config pointer is null";
pub const MODEL_POINTER_IS_NULL: &str = "Model pointer is null";
pub const PROCESSOR_POINTER_IS_NULL: &str = "Processor pointer is null";
pub const INVALID_DATA_INPUTS: &str = "Invalid data inputs";
pub const XY_ARRAYS_MUST_NOT_BE_NULL: &str = "x and y arrays must not be null";
pub const ARRAY_LENGTH_MUST_BE_GREATER_THAN_ZERO: &str = "Array length must be greater than 0";
pub const CUSTOM_WEIGHTS_MUST_BE_NON_NEGATIVE: &str = "custom_weights must be non-negative";

pub fn sanitize_error_message(msg: &str) -> String {
    msg.replace('\0', " ")
}

pub fn to_cstring_lossy(msg: &str) -> CString {
    CString::new(sanitize_error_message(msg)).unwrap_or_default()
}

pub fn panic_fallback_message() -> &'static str {
    PANIC_FALLBACK_MESSAGE
}

pub fn xy_length_mismatch_message(x_len: usize, y_len: usize) -> String {
    format!("x length ({}) must equal y length ({})", x_len, y_len)
}

pub fn custom_weights_length_mismatch_message(weights_len: usize, y_len: usize) -> String {
    format!(
        "custom_weights length ({}) must match y length ({})",
        weights_len, y_len
    )
}

pub fn custom_weights_length_mismatch_message_for(
    label: &str,
    weights_len: usize,
    y_len: usize,
) -> String {
    format!(
        "{} length ({}) must match y length ({})",
        label, weights_len, y_len
    )
}

pub fn custom_weights_must_be_non_negative_message_for(label: &str) -> String {
    format!("{} must be non-negative", label)
}

pub fn required_option_message(option_name: &str) -> String {
    format!("{} must be provided", option_name)
}

pub fn mutex_poisoned_message(details: &str) -> String {
    format!("Mutex poisoned: {}", details)
}

// All string-keyed options accepted by language binding frontends.
pub struct BuilderOptionSet<'a> {
    pub fraction: Option<f64>,
    pub iterations: Option<usize>,
    pub delta: Option<f64>,
    pub weight_function: Option<&'a str>,
    pub robustness_method: Option<&'a str>,
    pub zero_weight_fallback: Option<&'a str>,
    pub boundary_policy: Option<&'a str>,
    pub scaling_method: Option<&'a str>,
    pub auto_converge: Option<f64>,
    pub return_residuals: bool,
    pub return_robustness_weights: bool,
    pub return_diagnostics: bool,
    pub return_se: bool,
    pub confidence_intervals: Option<f64>,
    pub prediction_intervals: Option<f64>,
    pub parallel: Option<bool>,
    // Streaming-only options
    pub chunk_size: Option<usize>,
    pub overlap: Option<usize>,
    pub merge_strategy: Option<&'a str>,
    // Online-only options
    pub window_capacity: Option<usize>,
    pub min_points: Option<usize>,
    pub update_mode: Option<&'a str>,
    // Cross-validation
    pub cv_fractions: Option<&'a [f64]>,
    pub cv_method: Option<&'a str>,
    pub cv_k: Option<usize>,
    pub cv_seed: Option<u64>,
}

// Pre-parsed typed form of BuilderOptionSet.
pub struct TypedBuilderOptionSet {
    pub fraction: Option<f64>,
    pub iterations: Option<usize>,
    pub delta: Option<f64>,
    pub weight_function: Option<WeightFunction>,
    pub robustness_method: Option<RobustnessMethod>,
    pub zero_weight_fallback: Option<ZeroWeightFallback>,
    pub boundary_policy: Option<BoundaryPolicy>,
    pub scaling_method: Option<ScalingMethod>,
    pub auto_converge: Option<f64>,
    pub return_residuals: bool,
    pub return_robustness_weights: bool,
    pub return_diagnostics: bool,
    pub return_se: bool,
    pub confidence_intervals: Option<f64>,
    pub prediction_intervals: Option<f64>,
    pub parallel: Option<bool>,
    // Streaming-only options
    pub chunk_size: Option<usize>,
    pub overlap: Option<usize>,
    pub merge_strategy: Option<MergeStrategy>,
    // Online-only options
    pub window_capacity: Option<usize>,
    pub min_points: Option<usize>,
    pub update_mode: Option<UpdateMode>,
    // Cross-validation
    pub cv_fractions: Option<Vec<f64>>,
    pub cv_method: Option<String>,
    pub cv_k: Option<usize>,
    pub cv_seed: Option<u64>,

    // Per-observation case weights. When provided, multiplies each local kernel weight:
    // `w_ij = custom_weights[j] * K(d_ij / h) * robustness_j`.
    pub custom_weights: Option<Vec<f64>>,
}

// ============================================================================
// Parse functions (string → typed enum)
// ============================================================================

pub fn parse_weight_function(name: &str) -> Result<WeightFunction, String> {
    match name.to_lowercase().as_str() {
        "tricube" => Ok(WeightFunction::Tricube),
        "epanechnikov" => Ok(WeightFunction::Epanechnikov),
        "gaussian" => Ok(WeightFunction::Gaussian),
        "uniform" | "boxcar" => Ok(WeightFunction::Uniform),
        "biweight" | "bisquare" => Ok(WeightFunction::Biweight),
        "triangle" | "triangular" => Ok(WeightFunction::Triangle),
        "cosine" => Ok(WeightFunction::Cosine),
        _ => Err(format!(
            "Unknown weight function: {}. Valid options: tricube, epanechnikov, gaussian, uniform, biweight, triangle, cosine",
            name
        )),
    }
}

pub fn parse_robustness_method(name: &str) -> Result<RobustnessMethod, String> {
    match name.to_lowercase().as_str() {
        "bisquare" | "biweight" => Ok(RobustnessMethod::Bisquare),
        "huber" => Ok(RobustnessMethod::Huber),
        "talwar" => Ok(RobustnessMethod::Talwar),
        _ => Err(format!(
            "Unknown robustness method: {}. Valid options: bisquare, huber, talwar",
            name
        )),
    }
}

pub fn parse_zero_weight_fallback(name: &str) -> Result<ZeroWeightFallback, String> {
    match name.to_lowercase().as_str() {
        "use_local_mean" | "local_mean" | "mean" => Ok(ZeroWeightFallback::UseLocalMean),
        "return_original" | "original" => Ok(ZeroWeightFallback::ReturnOriginal),
        "return_none" | "none" | "nan" => Ok(ZeroWeightFallback::ReturnNone),
        _ => Err(format!(
            "Unknown zero weight fallback: {}. Valid options: use_local_mean, return_original, return_none",
            name
        )),
    }
}

pub fn parse_boundary_policy(name: &str) -> Result<BoundaryPolicy, String> {
    match name.to_lowercase().as_str() {
        "extend" | "pad" => Ok(BoundaryPolicy::Extend),
        "reflect" | "mirror" => Ok(BoundaryPolicy::Reflect),
        "zero" | "none" => Ok(BoundaryPolicy::Zero),
        "noboundary" => Ok(BoundaryPolicy::NoBoundary),
        _ => Err(format!(
            "Unknown boundary policy: {}. Valid options: extend, reflect, zero, noboundary",
            name
        )),
    }
}

pub fn parse_scaling_method(name: &str) -> Result<ScalingMethod, String> {
    match name.to_lowercase().as_str() {
        "mad" => Ok(ScalingMethod::MAD),
        "mar" => Ok(ScalingMethod::MAR),
        "mean" => Ok(ScalingMethod::Mean),
        _ => Err(format!(
            "Unknown scaling method: {}. Valid options: mad, mar, mean",
            name
        )),
    }
}

pub fn parse_merge_strategy(name: &str) -> Result<MergeStrategy, String> {
    match name.to_lowercase().as_str() {
        "average" | "mean" => Ok(MergeStrategy::Average),
        "weighted" | "weighted_average" => Ok(MergeStrategy::WeightedAverage),
        "first" | "take_first" | "left" => Ok(MergeStrategy::TakeFirst),
        "last" | "take_last" | "right" => Ok(MergeStrategy::TakeLast),
        _ => Err(format!(
            "Unknown merge strategy: {}. Valid options: average, weighted_average, take_first, take_last",
            name
        )),
    }
}

pub fn parse_update_mode(name: &str) -> Result<UpdateMode, String> {
    match name.to_lowercase().as_str() {
        "full" | "resmooth" => Ok(UpdateMode::Full),
        "incremental" | "single" => Ok(UpdateMode::Incremental),
        _ => Err(format!(
            "Unknown update mode: {}. Valid options: full, incremental",
            name
        )),
    }
}

// ============================================================================
// Display functions (typed enum → canonical string)
// ============================================================================

pub fn weight_function_str(value: WeightFunction) -> &'static str {
    match value {
        WeightFunction::Tricube => "tricube",
        WeightFunction::Epanechnikov => "epanechnikov",
        WeightFunction::Gaussian => "gaussian",
        WeightFunction::Uniform => "uniform",
        WeightFunction::Biweight => "biweight",
        WeightFunction::Triangle => "triangle",
        WeightFunction::Cosine => "cosine",
    }
}

pub fn robustness_method_str(value: RobustnessMethod) -> &'static str {
    match value {
        RobustnessMethod::Bisquare => "bisquare",
        RobustnessMethod::Huber => "huber",
        RobustnessMethod::Talwar => "talwar",
    }
}

pub fn scaling_method_str(value: ScalingMethod) -> &'static str {
    match value {
        ScalingMethod::MAD => "mad",
        ScalingMethod::MAR => "mar",
        ScalingMethod::Mean => "mean",
    }
}

pub fn zero_weight_fallback_str(value: ZeroWeightFallback) -> &'static str {
    match value {
        ZeroWeightFallback::UseLocalMean => "use_local_mean",
        ZeroWeightFallback::ReturnOriginal => "return_original",
        ZeroWeightFallback::ReturnNone => "return_none",
    }
}

pub fn boundary_policy_str(value: BoundaryPolicy) -> &'static str {
    match value {
        BoundaryPolicy::Extend => "extend",
        BoundaryPolicy::Reflect => "reflect",
        BoundaryPolicy::Zero => "zero",
        BoundaryPolicy::NoBoundary => "noboundary",
    }
}

pub fn merge_strategy_str(value: MergeStrategy) -> &'static str {
    match value {
        MergeStrategy::Average => "average",
        MergeStrategy::WeightedAverage => "weighted_average",
        MergeStrategy::TakeFirst => "take_first",
        MergeStrategy::TakeLast => "take_last",
    }
}

pub fn update_mode_str(value: UpdateMode) -> &'static str {
    match value {
        UpdateMode::Full => "full",
        UpdateMode::Incremental => "incremental",
    }
}

// ============================================================================
// Cross-validation builder helper
// ============================================================================

pub fn apply_cross_validation(
    mut builder: LowessBuilder<f64>,
    fractions: Option<&[f64]>,
    method: Option<&str>,
    k: Option<usize>,
    seed: Option<u64>,
) -> Result<LowessBuilder<f64>, String> {
    let Some(fractions) = fractions else {
        return Ok(builder);
    };

    let method = method.unwrap_or("kfold");
    let k = k.unwrap_or(5);

    match method.to_lowercase().as_str() {
        "simple" | "loo" | "loocv" | "leave_one_out" => {
            builder = builder.cv_method("loocv");
            builder = builder.cv_fractions(fractions.to_vec());
            if let Some(s) = seed {
                builder = builder.cv_seed(s);
            }
            Ok(builder)
        }
        "kfold" | "k_fold" | "k-fold" => {
            builder = builder.cv_method("kfold");
            builder = builder.cv_k(k);
            builder = builder.cv_fractions(fractions.to_vec());
            if let Some(s) = seed {
                builder = builder.cv_seed(s);
            }
            Ok(builder)
        }
        _ => Err(format!(
            "Unknown CV method: {}. Valid options: loocv, kfold",
            method
        )),
    }
}

// ============================================================================
// Builder option application
// ============================================================================

// Parse all string-keyed options and apply them to a builder.
//
// Use this from binding frontends that receive all options as strings
// (C, Julia, Node.js, Python, R, WASM).
pub fn apply_builder_options(
    builder: LowessBuilder<f64>,
    options: BuilderOptionSet<'_>,
) -> Result<LowessBuilder<f64>, String> {
    let typed = TypedBuilderOptionSet {
        fraction: options.fraction,
        iterations: options.iterations,
        delta: options.delta,
        weight_function: options
            .weight_function
            .map(parse_weight_function)
            .transpose()?,
        robustness_method: options
            .robustness_method
            .map(parse_robustness_method)
            .transpose()?,
        zero_weight_fallback: options
            .zero_weight_fallback
            .map(parse_zero_weight_fallback)
            .transpose()?,
        boundary_policy: options
            .boundary_policy
            .map(parse_boundary_policy)
            .transpose()?,
        scaling_method: options
            .scaling_method
            .map(parse_scaling_method)
            .transpose()?,
        auto_converge: options.auto_converge,
        return_residuals: options.return_residuals,
        return_robustness_weights: options.return_robustness_weights,
        return_diagnostics: options.return_diagnostics,
        return_se: options.return_se,
        confidence_intervals: options.confidence_intervals,
        prediction_intervals: options.prediction_intervals,
        parallel: options.parallel,
        chunk_size: options.chunk_size,
        overlap: options.overlap,
        merge_strategy: options
            .merge_strategy
            .map(parse_merge_strategy)
            .transpose()?,
        window_capacity: options.window_capacity,
        min_points: options.min_points,
        update_mode: options.update_mode.map(parse_update_mode).transpose()?,
        cv_fractions: options.cv_fractions.map(|v| v.to_vec()),
        cv_method: options.cv_method.map(str::to_string),
        cv_k: options.cv_k,
        cv_seed: options.cv_seed,
        // custom_weights cannot be provided via string-based BuilderOptionSet
        custom_weights: None,
    };

    apply_typed_builder_options(builder, typed)
}

// Apply pre-parsed typed options to a builder.
//
// Use this from internal code that already holds typed enum values.
pub fn apply_typed_builder_options(
    mut builder: LowessBuilder<f64>,
    options: TypedBuilderOptionSet,
) -> Result<LowessBuilder<f64>, String> {
    if let Some(f) = options.fraction {
        builder = builder.fraction(f);
    }
    if let Some(iter) = options.iterations {
        builder = builder.iterations(iter);
    }
    if let Some(d) = options.delta {
        builder = builder.delta(d);
    }
    if let Some(wf) = options.weight_function {
        builder = builder.weight_function(wf);
    }
    if let Some(rm) = options.robustness_method {
        builder = builder.robustness_method(rm);
    }
    if let Some(zw) = options.zero_weight_fallback {
        builder = builder.zero_weight_fallback(zw);
    }
    if let Some(bp) = options.boundary_policy {
        builder = builder.boundary_policy(bp);
    }
    if let Some(sm) = options.scaling_method {
        builder = builder.scaling_method(sm);
    }
    if let Some(ac) = options.auto_converge {
        builder = builder.auto_converge(ac);
    }
    if options.return_residuals {
        builder = builder.return_residuals();
    }
    if options.return_robustness_weights {
        builder = builder.return_robustness_weights();
    }
    if options.return_diagnostics {
        builder = builder.return_diagnostics();
    }
    if options.return_se {
        builder = builder.return_se();
    }
    if let Some(cw) = options.custom_weights {
        builder = builder.custom_weights(cw);
    }
    if let Some(ci) = options.confidence_intervals {
        builder = builder.confidence_intervals(ci);
    }
    if let Some(pi) = options.prediction_intervals {
        builder = builder.prediction_intervals(pi);
    }
    if let Some(par) = options.parallel {
        builder = builder.parallel(par);
    }
    if let Some(size) = options.chunk_size {
        builder = builder.chunk_size(size);
    }
    if let Some(ov) = options.overlap {
        builder = builder.overlap(ov);
    }
    if let Some(ms) = options.merge_strategy {
        builder = builder.merge_strategy(ms);
    }
    if let Some(cap) = options.window_capacity {
        builder = builder.window_capacity(cap);
    }
    if let Some(mp) = options.min_points {
        builder = builder.min_points(mp);
    }
    if let Some(um) = options.update_mode {
        builder = builder.update_mode(um);
    }

    builder = apply_cross_validation(
        builder,
        options.cv_fractions.as_deref(),
        options.cv_method.as_deref(),
        options.cv_k,
        options.cv_seed,
    )?;

    Ok(builder)
}

// ============================================================================
// LowessError helpers
// ============================================================================

pub fn lowess_error_message(err: &LowessError) -> String {
    err.to_string()
}

// ─── Builder setter methods ───────────────────────────────────────────────────
//
// These impl blocks are in binding_support.rs (compiled only with the `dev`
// feature) so the adapter files (batch.rs, online.rs, streaming.rs) stay free
// of any `#[cfg]` attributes.

#[allow(private_bounds)]
impl<T: Float> ParallelBatchLowessBuilder<T> {
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.base.parallel = Some(parallel);
        self
    }

    pub fn backend(mut self, backend: Backend) -> Self {
        self.base.backend = Some(backend);
        self
    }

    pub fn fraction(mut self, fraction: T) -> Self {
        self.base.fraction = fraction;
        self
    }

    pub fn iterations(mut self, iterations: usize) -> Self {
        self.base.iterations = iterations;
        self
    }

    pub fn delta(mut self, delta: T) -> Self {
        self.base.delta = Some(delta);
        self
    }

    pub fn weight_function(mut self, wf: impl IntoEnum<WeightFunction>) -> Self {
        match wf.into_enum() {
            Ok(w) => self.base.weight_function = w,
            Err(e) => {
                self.parse_errors.push(e);
            }
        }
        self
    }

    pub fn robustness_method(mut self, method: impl IntoEnum<RobustnessMethod>) -> Self {
        match method.into_enum() {
            Ok(m) => self.base.robustness_method = m,
            Err(e) => {
                self.parse_errors.push(e);
            }
        }
        self
    }

    pub fn zero_weight_fallback(mut self, fallback: impl IntoEnum<ZeroWeightFallback>) -> Self {
        match fallback.into_enum() {
            Ok(f) => self.base.zero_weight_fallback = f,
            Err(e) => {
                self.parse_errors.push(e);
            }
        }
        self
    }

    pub fn boundary_policy(mut self, policy: impl IntoEnum<BoundaryPolicy>) -> Self {
        match policy.into_enum() {
            Ok(p) => self.base.boundary_policy = p,
            Err(e) => {
                self.parse_errors.push(e);
            }
        }
        self
    }

    pub fn auto_converge(mut self, tolerance: T) -> Self {
        self.base.auto_converge = Some(tolerance);
        self
    }

    pub fn compute_residuals(mut self, enabled: bool) -> Self {
        self.base.compute_residuals = enabled;
        self
    }

    pub fn return_robustness_weights(mut self, enabled: bool) -> Self {
        self.base.return_robustness_weights = enabled;
        self
    }

    pub fn return_diagnostics(mut self, enabled: bool) -> Self {
        self.base.return_diagnostics = enabled;
        self
    }

    pub fn confidence_intervals(mut self, level: T) -> Self {
        self.base.interval_type = Some(IntervalMethod::confidence(level));
        self
    }

    pub fn prediction_intervals(mut self, level: T) -> Self {
        self.base.interval_type = Some(IntervalMethod::prediction(level));
        self
    }

    pub fn cv_fractions(mut self, fractions: Vec<T>) -> Self {
        self.base.cv_fractions = Some(fractions);
        self
    }

    pub fn cv_method(mut self, method: &str) -> Self {
        self.cv_method_str = Some(method.to_string());
        self
    }

    pub fn cv_k(mut self, k: usize) -> Self {
        self.cv_k_val = k;
        self
    }

    pub fn cv_seed(mut self, seed: u64) -> Self {
        self.base.cv_seed = Some(seed);
        self
    }

    pub fn custom_weights(mut self, weights: Vec<T>) -> Self {
        self.base.custom_weights = Some(weights);
        self
    }
}

#[allow(private_bounds)]
impl<T: Float> ParallelOnlineLowessBuilder<T> {
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.base.parallel = Some(parallel);
        self
    }

    pub fn backend(mut self, backend: Backend) -> Self {
        self.base.backend = Some(backend);
        self
    }

    pub fn fraction(mut self, fraction: T) -> Self {
        self.base.fraction = fraction;
        self
    }

    pub fn iterations(mut self, iterations: usize) -> Self {
        self.base.iterations = iterations;
        self
    }

    pub fn delta(mut self, delta: T) -> Self {
        self.base.delta = delta;
        self
    }

    pub fn weight_function(mut self, wf: impl IntoEnum<WeightFunction>) -> Self {
        match wf.into_enum() {
            Ok(w) => self.base.weight_function = w,
            Err(e) => {
                self.parse_errors.push(e);
            }
        }
        self
    }

    pub fn robustness_method(mut self, method: impl IntoEnum<RobustnessMethod>) -> Self {
        match method.into_enum() {
            Ok(m) => self.base.robustness_method = m,
            Err(e) => {
                self.parse_errors.push(e);
            }
        }
        self
    }

    pub fn zero_weight_fallback(mut self, fallback: impl IntoEnum<ZeroWeightFallback>) -> Self {
        match fallback.into_enum() {
            Ok(f) => self.base.zero_weight_fallback = f,
            Err(e) => {
                self.parse_errors.push(e);
            }
        }
        self
    }

    pub fn boundary_policy(mut self, policy: impl IntoEnum<BoundaryPolicy>) -> Self {
        match policy.into_enum() {
            Ok(p) => self.base.boundary_policy = p,
            Err(e) => {
                self.parse_errors.push(e);
            }
        }
        self
    }

    pub fn auto_converge(mut self, tolerance: T) -> Self {
        self.base.auto_converge = Some(tolerance);
        self
    }

    pub fn compute_residuals(mut self, enabled: bool) -> Self {
        self.base.compute_residuals = enabled;
        self
    }

    pub fn return_robustness_weights(mut self, enabled: bool) -> Self {
        self.base.return_robustness_weights = enabled;
        self
    }

    pub fn window_capacity(mut self, capacity: usize) -> Self {
        self.base.window_capacity = capacity;
        self
    }

    pub fn min_points(mut self, min_points: usize) -> Self {
        self.base.min_points = min_points;
        self
    }

    pub fn update_mode(mut self, mode: impl IntoEnum<UpdateMode>) -> Self {
        match mode.into_enum() {
            Ok(m) => self.base.update_mode = m,
            Err(e) => {
                self.parse_errors.push(e);
            }
        }
        self
    }
}

#[allow(private_bounds)]
impl<T: Float> ParallelStreamingLowessBuilder<T> {
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.base.parallel = Some(parallel);
        self
    }

    pub fn backend(mut self, backend: Backend) -> Self {
        self.base.backend = Some(backend);
        self
    }

    pub fn fraction(mut self, fraction: T) -> Self {
        self.base.fraction = fraction;
        self
    }

    pub fn iterations(mut self, iterations: usize) -> Self {
        self.base.iterations = iterations;
        self
    }

    pub fn delta(mut self, delta: T) -> Self {
        self.base.delta = delta;
        self
    }

    pub fn weight_function(mut self, wf: impl IntoEnum<WeightFunction>) -> Self {
        match wf.into_enum() {
            Ok(w) => self.base.weight_function = w,
            Err(e) => {
                self.parse_errors.push(e);
            }
        }
        self
    }

    pub fn robustness_method(mut self, method: impl IntoEnum<RobustnessMethod>) -> Self {
        match method.into_enum() {
            Ok(m) => self.base.robustness_method = m,
            Err(e) => {
                self.parse_errors.push(e);
            }
        }
        self
    }

    pub fn zero_weight_fallback(mut self, fallback: impl IntoEnum<ZeroWeightFallback>) -> Self {
        match fallback.into_enum() {
            Ok(f) => self.base.zero_weight_fallback = f,
            Err(e) => {
                self.parse_errors.push(e);
            }
        }
        self
    }

    pub fn boundary_policy(mut self, policy: impl IntoEnum<BoundaryPolicy>) -> Self {
        match policy.into_enum() {
            Ok(p) => self.base.boundary_policy = p,
            Err(e) => {
                self.parse_errors.push(e);
            }
        }
        self
    }

    pub fn auto_converge(mut self, tolerance: T) -> Self {
        self.base.auto_converge = Some(tolerance);
        self
    }

    pub fn compute_residuals(mut self, enabled: bool) -> Self {
        self.base.compute_residuals = enabled;
        self
    }

    pub fn return_robustness_weights(mut self, enabled: bool) -> Self {
        self.base.return_robustness_weights = enabled;
        self
    }

    pub fn chunk_size(mut self, size: usize) -> Self {
        self.base.chunk_size = size;
        self
    }

    pub fn overlap(mut self, size: usize) -> Self {
        self.base.overlap = size;
        self
    }

    pub fn merge_strategy(mut self, strategy: impl IntoEnum<MergeStrategy>) -> Self {
        match strategy.into_enum() {
            Ok(s) => self.base.merge_strategy = s,
            Err(e) => {
                self.parse_errors.push(e);
            }
        }
        self
    }

    pub fn return_diagnostics(mut self, enabled: bool) -> Self {
        self.base.return_diagnostics = enabled;
        self
    }
}
