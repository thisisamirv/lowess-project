//! Shared helpers for language bindings.
//!
//! This module centralizes string option parsing used by C/C++, Julia,
//! Node.js, Python, R, and WASM bindings so option aliases and validation
//! behavior stay consistent across all binding frontends.

use crate::adapters::batch::{ParallelBatchLowess, ParallelBatchLowessBuilder};
use crate::adapters::online::{ParallelOnlineLowess, ParallelOnlineLowessBuilder};
use crate::adapters::streaming::{ParallelStreamingLowess, ParallelStreamingLowessBuilder};
use crate::api::{Batch, LowessBuilder, LowessError, LowessResult, Online, Streaming};
use crate::parse::IntoEnum;
use lowess::internals::adapters::online::OnlineOutput;
use lowess::internals::evaluation::intervals::IntervalMethod;
pub use lowess::internals::primitives::backend::Backend;
use num_traits::Float;

pub use lowess::internals::adapters::online::UpdateMode;
pub use lowess::internals::adapters::streaming::MergeStrategy;
pub use lowess::internals::algorithms::regression::ZeroWeightFallback;
pub use lowess::internals::algorithms::robustness::RobustnessMethod;
use lowess::internals::alias;
pub use lowess::internals::math::boundary::BoundaryPolicy;
pub use lowess::internals::math::kernel::WeightFunction;
pub use lowess::internals::math::scaling::ScalingMethod;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

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

impl std::fmt::Display for BindingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

pub fn map_invalid_arg<T, E: ToString>(result: Result<T, E>) -> Result<T, BindingError> {
    result.map_err(|e| BindingError::invalid_arg(e.to_string()))
}

pub fn map_runtime<T, E: ToString>(result: Result<T, E>) -> Result<T, BindingError> {
    result.map_err(|e| BindingError::runtime(e.to_string()))
}

/// Categorize a [`LowessError`]: only [`LowessError::RuntimeError`] maps to
/// [`BindingErrorCategory::Runtime`]; every other variant is an invalid-argument
/// error that originates from bad user input.
pub fn map_lowess_result<T>(result: Result<T, LowessError>) -> Result<T, BindingError> {
    result.map_err(|err| match &err {
        LowessError::RuntimeError(_) => BindingError::runtime(err.to_string()),
        _ => BindingError::invalid_arg(err.to_string()),
    })
}

pub const PANIC_FALLBACK_MESSAGE: &str = "Panic in Rust library";
pub const CONFIG_POINTER_IS_NULL: &str = "Config pointer is null";
pub const MODEL_POINTER_IS_NULL: &str = "Model pointer is null";
pub const MODEL_NOT_INITIALIZED: &str = "Model not initialized";
pub const PROCESSOR_POINTER_IS_NULL: &str = "Processor pointer is null";
pub const INVALID_DATA_INPUTS: &str = "Invalid data inputs";
pub const XY_ARRAYS_MUST_NOT_BE_NULL: &str = "x and y arrays must not be null";
pub const ARRAY_LENGTH_MUST_BE_GREATER_THAN_ZERO: &str = "Array length must be greater than 0";
pub const CUSTOM_WEIGHTS_MUST_BE_NON_NEGATIVE: &str = "custom_weights must be non-negative";

// Default string values for all parser-facing options. Re-exported from
// `lowess::defaults` so that all bindings share a single source of truth.
pub use lowess::internals::defaults::DEFAULT_BOUNDARY_POLICY;
pub use lowess::internals::defaults::DEFAULT_CV_METHOD;
pub use lowess::internals::defaults::DEFAULT_ONLINE_UPDATE_MODE;
pub use lowess::internals::defaults::DEFAULT_ROBUSTNESS_METHOD;
pub use lowess::internals::defaults::DEFAULT_SCALING_METHOD;
pub use lowess::internals::defaults::DEFAULT_STREAMING_MERGE_STRATEGY;
pub use lowess::internals::defaults::DEFAULT_WEIGHT_FUNCTION;
pub use lowess::internals::defaults::DEFAULT_ZERO_WEIGHT_FALLBACK;

pub fn sanitize_error_message(msg: &str) -> String {
    msg.replace('\0', " ")
}

pub fn to_cstring_lossy(msg: &str) -> CString {
    CString::new(sanitize_error_message(msg)).unwrap_or_default()
}

pub fn panic_fallback_message() -> &'static str {
    PANIC_FALLBACK_MESSAGE
}

/// Parse a C string safely, returning `default` when the pointer is null or
/// the bytes are not valid UTF-8.
///
/// # Safety
/// If `s` is non-null it must point to a valid null-terminated C string that
/// lives at least as long as the returned `&str`.
pub unsafe fn parse_c_str_or_default(s: *const c_char, default: &str) -> &str {
    if s.is_null() {
        return default;
    }
    // SAFETY: caller guarantees `s` is a valid null-terminated C string.
    unsafe { CStr::from_ptr(s) }.to_str().unwrap_or(default)
}

// Validate that a signed integer is > 0 and safely cast to usize.
// Used by C/Julia FFI bindings to convert c_int parameters with bounds checking.
pub fn require_positive_usize(name: &str, value: i32) -> Result<usize, String> {
    if value <= 0 {
        Err(format!("{name} must be greater than 0, got {value}"))
    } else {
        Ok(value as usize)
    }
}

// Validate that a signed integer is >= 0 and safely cast to usize.
pub fn require_non_negative_usize(name: &str, value: i32) -> Result<usize, String> {
    if value < 0 {
        Err(format!("{name} must be non-negative, got {value}"))
    } else {
        Ok(value as usize)
    }
}

pub fn dims_mismatch_message(x_len: usize, y_len: usize, dimensions: usize) -> String {
    format!(
        "x length ({}) must equal y length ({}) * dimensions ({})",
        x_len, y_len, dimensions
    )
}

/// Returns a non-null raw pointer as a slice, or `None` if the pointer is null or `len` is 0.
///
/// # Safety
/// `ptr` must point to at least `len` valid, initialized elements of type `T` that remain
/// live for at least as long as the returned slice is used.
pub unsafe fn option_slice_from_ptr<'a, T>(ptr: *const T, len: usize) -> Option<&'a [T]> {
    if !ptr.is_null() && len > 0 {
        Some(unsafe { std::slice::from_raw_parts(ptr, len) })
    } else {
        None
    }
}

/// Like [`option_slice_from_ptr`] but clones the slice into a `Vec`.
///
/// # Safety
/// Same preconditions as [`option_slice_from_ptr`].
pub unsafe fn option_vec_from_ptr<T: Clone>(ptr: *const T, len: usize) -> Option<Vec<T>> {
    unsafe { option_slice_from_ptr(ptr, len) }.map(<[T]>::to_vec)
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

// Message returned by setter stubs on streaming/online models (C++ binding).
pub fn setter_unsupported_eager_message(name: &str) -> String {
    format!(
        "{name} is not supported: streaming/online models are eagerly initialized at construction"
    )
}

// Message returned by setter stubs that require constructor-time configuration (Julia binding).
pub fn setter_unsupported_constructor_only_message(name: &str) -> String {
    format!("{name} is not supported: configure model options at construction time")
}

// Converts a Vec<f64> into a heap-allocated raw pointer.
// The caller is responsible for freeing the memory via Box::from_raw / Vec::from_raw_parts.
pub fn vec_to_raw_ptr(v: Vec<f64>) -> *mut f64 {
    let mut boxed = v.into_boxed_slice();
    let ptr = boxed.as_mut_ptr();
    std::mem::forget(boxed);
    ptr
}

// Same as vec_to_raw_ptr but for an Option, returning null for None.
pub fn opt_vec_to_raw_ptr(v: Option<Vec<f64>>) -> *mut f64 {
    match v {
        Some(vec) => vec_to_raw_ptr(vec),
        None => std::ptr::null_mut(),
    }
}

// Extracts flat scalar diagnostics from a LowessResult.
// Returns (rmse, mae, r_squared, aic, aicc, effective_df, residual_sd) with
// f64::NAN for any field that was not computed.
pub fn extract_diagnostics(result: &LowessResult<f64>) -> (f64, f64, f64, f64, f64, f64, f64) {
    if let Some(ref d) = result.diagnostics {
        (
            d.rmse,
            d.mae,
            d.r_squared,
            d.aic.unwrap_or(f64::NAN),
            d.aicc.unwrap_or(f64::NAN),
            d.effective_df.unwrap_or(f64::NAN),
            d.residual_sd,
        )
    } else {
        (
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
        )
    }
}

// Extracts the optional scalar fields from an online add_point output.
// Returns (std_error, residual, robustness_weight, iterations_used).
// Optional f64 fields default to f64::NAN; iterations_used defaults to -1.
pub fn extract_online_output(o: &OnlineOutput<f64>) -> (f64, f64, f64, i32) {
    (
        o.std_error.unwrap_or(f64::NAN),
        o.residual.unwrap_or(f64::NAN),
        o.robustness_weight.unwrap_or(f64::NAN),
        o.iterations_used.map(|i| i as i32).unwrap_or(-1),
    )
}

// Neutral FFI lowess result parts produced by extract_ffi_lowess_result.
// All raw-pointer fields are either Rust-allocated heap memory (freed via
// free_raw_f64_buffer) or null. Types use Rust-native widths; the binding casts
// to its platform ABI types (e.g. usize → c_ulong) and appends its own error
// field before exposing the struct to C.
pub struct FfiLowessResult {
    pub x: *mut f64,
    pub y: *mut f64,
    pub n: usize,
    pub standard_errors: *mut f64,
    pub confidence_lower: *mut f64,
    pub confidence_upper: *mut f64,
    pub prediction_lower: *mut f64,
    pub prediction_upper: *mut f64,
    pub residuals: *mut f64,
    pub robustness_weights: *mut f64,
    pub fraction_used: f64,
    pub iterations_used: i32,
    pub rmse: f64,
    pub mae: f64,
    pub r_squared: f64,
    pub aic: f64,
    pub aicc: f64,
    pub effective_df: f64,
    pub residual_sd: f64,
    pub cv_scores: *mut f64,
    pub cv_scores_len: usize,
}

// Extract all fields from a LowessResult into an FfiLowessResult. All optional
// vectors are moved to heap-allocated raw pointers; scalar optionals are
// NaN/−1-defaulted. The caller owns the returned pointers.
pub fn extract_ffi_lowess_result(result: LowessResult<f64>) -> FfiLowessResult {
    let n = result.y.len();
    let (rmse, mae, r_squared, aic, aicc, effective_df, residual_sd) = extract_diagnostics(&result);
    let cv_scores_len = result.cv_scores.as_ref().map(|v| v.len()).unwrap_or(0);
    FfiLowessResult {
        x: vec_to_raw_ptr(result.x),
        y: vec_to_raw_ptr(result.y),
        n,
        standard_errors: opt_vec_to_raw_ptr(result.standard_errors),
        confidence_lower: opt_vec_to_raw_ptr(result.confidence_lower),
        confidence_upper: opt_vec_to_raw_ptr(result.confidence_upper),
        prediction_lower: opt_vec_to_raw_ptr(result.prediction_lower),
        prediction_upper: opt_vec_to_raw_ptr(result.prediction_upper),
        residuals: opt_vec_to_raw_ptr(result.residuals),
        robustness_weights: opt_vec_to_raw_ptr(result.robustness_weights),
        fraction_used: result.fraction_used,
        iterations_used: result.iterations_used.map(|i| i as i32).unwrap_or(-1),
        rmse,
        mae,
        r_squared,
        aic,
        aicc,
        effective_df,
        residual_sd,
        cv_scores: opt_vec_to_raw_ptr(result.cv_scores),
        cv_scores_len,
    }
}

/// Free a heap-allocated `f64` buffer produced by `vec_to_raw_ptr` / `opt_vec_to_raw_ptr`.
/// No-op when `ptr` is null. Both functions allocate via `into_boxed_slice`, so the
/// correct counterpart is `Box::from_raw(slice_from_raw_parts_mut)`.
///
/// # Safety
/// `ptr` must either be null or have been produced by `vec_to_raw_ptr` /
/// `opt_vec_to_raw_ptr` with the same `len`.
pub unsafe fn free_raw_f64_buffer(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len));
        }
    }
}

/// Free a heap-allocated C string produced by `CString::into_raw`.
///
/// # Safety
/// `ptr` must either be null or have been produced by `CString::into_raw`.
pub unsafe fn free_raw_c_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe {
            let _ = CString::from_raw(ptr);
        }
    }
}

// Convert an error message to a heap-allocated C string and return its raw
// pointer. The caller must eventually free it via free_raw_c_string /
// CString::from_raw. The message is sanitized (null bytes replaced with spaces)
// before conversion.
pub fn into_raw_error_c_string(msg: &str) -> *mut c_char {
    to_cstring_lossy(msg).into_raw()
}

// All string-keyed options accepted by language binding frontends.
#[derive(Default)]
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
    alias::parse_weight_function(name).map_err(|e| e.to_string())
}

pub fn parse_robustness_method(name: &str) -> Result<RobustnessMethod, String> {
    alias::parse_robustness_method(name).map_err(|e| e.to_string())
}

pub fn parse_zero_weight_fallback(name: &str) -> Result<ZeroWeightFallback, String> {
    alias::parse_zero_weight_fallback(name).map_err(|e| e.to_string())
}

pub fn parse_boundary_policy(name: &str) -> Result<BoundaryPolicy, String> {
    alias::parse_boundary_policy(name).map_err(|e| e.to_string())
}

pub fn parse_scaling_method(name: &str) -> Result<ScalingMethod, String> {
    alias::parse_scaling_method(name).map_err(|e| e.to_string())
}

pub fn parse_merge_strategy(name: &str) -> Result<MergeStrategy, String> {
    alias::parse_merge_strategy(name).map_err(|e| e.to_string())
}

pub fn parse_update_mode(name: &str) -> Result<UpdateMode, String> {
    alias::parse_update_mode(name).map_err(|e| e.to_string())
}

// ============================================================================
// Display functions (typed enum → canonical string)
// ============================================================================

pub fn weight_function_str(value: WeightFunction) -> &'static str {
    alias::weight_function_str(value)
}

pub fn robustness_method_str(value: RobustnessMethod) -> &'static str {
    alias::robustness_method_str(value)
}

pub fn scaling_method_str(value: ScalingMethod) -> &'static str {
    alias::scaling_method_str(value)
}

pub fn zero_weight_fallback_str(value: ZeroWeightFallback) -> &'static str {
    alias::zero_weight_fallback_str(value)
}

pub fn boundary_policy_str(value: BoundaryPolicy) -> &'static str {
    alias::boundary_policy_str(value)
}

pub fn merge_strategy_str(value: MergeStrategy) -> &'static str {
    alias::merge_strategy_str(value)
}

pub fn update_mode_str(value: UpdateMode) -> &'static str {
    alias::update_mode_str(value)
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

    let method = method.unwrap_or(lowess::internals::defaults::DEFAULT_CV_METHOD);
    let k = k.unwrap_or(lowess::internals::defaults::DEFAULT_CV_K);

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

// ─── Adapter build helpers ────────────────────────────────────────────────────
//
// These functions centralize the "extract defaults → parse string enum →
// chain adapter setters → build" pattern that every binding repeats for its
// batch, streaming, and online constructors.

// Compute the default overlap size from a chunk size when the caller does not
// specify one. cpp, julia, python, and r all use this same formula instead of
// the flat 500-point default used by wasm/nodejs.
pub fn default_overlap(chunk_size: usize) -> usize {
    let default = chunk_size / 10;
    default.min(chunk_size.saturating_sub(10)).max(1)
}

// Build a parallel batch processor. Applies optional case weights before
// building.
pub fn build_batch(
    builder: LowessBuilder<f64>,
    custom_weights: Option<Vec<f64>>,
) -> Result<ParallelBatchLowess<f64>, BindingError> {
    let builder = if let Some(cw) = custom_weights {
        builder.custom_weights(cw)
    } else {
        builder
    };
    map_lowess_result(builder.adapter(Batch).build())
}

// Build a parallel streaming processor.
// Defaults: chunk_size = 5000, overlap = default_overlap(chunk_size), merge_strategy = WeightedAverage.
pub fn build_streaming(
    builder: LowessBuilder<f64>,
    chunk_size: Option<usize>,
    overlap: Option<usize>,
    merge_strategy: Option<&str>,
) -> Result<ParallelStreamingLowess<f64>, BindingError> {
    let cs = chunk_size.unwrap_or(lowess::internals::defaults::DEFAULT_STREAMING_CHUNK_SIZE);
    let ov = overlap.unwrap_or_else(|| default_overlap(cs));
    let ms = match merge_strategy {
        Some(s) => map_invalid_arg(parse_merge_strategy(s))?,
        None => map_invalid_arg(parse_merge_strategy(
            lowess::internals::defaults::DEFAULT_STREAMING_MERGE_STRATEGY,
        ))?,
    };
    map_lowess_result(
        builder
            .adapter(Streaming)
            .chunk_size(cs)
            .overlap(ov)
            .merge_strategy(ms)
            .build(),
    )
}

// Build a parallel online processor.
// Defaults: window_capacity = 1000, min_points = 3, update_mode = Full.
pub fn build_online(
    builder: LowessBuilder<f64>,
    window_capacity: Option<usize>,
    min_points: Option<usize>,
    update_mode: Option<&str>,
) -> Result<ParallelOnlineLowess<f64>, BindingError> {
    let wc = window_capacity.unwrap_or(lowess::internals::defaults::DEFAULT_ONLINE_WINDOW_CAPACITY);
    let mp = min_points.unwrap_or(lowess::internals::defaults::DEFAULT_ONLINE_MIN_POINTS);
    let um = match update_mode {
        Some(s) => map_invalid_arg(parse_update_mode(s))?,
        None => map_invalid_arg(parse_update_mode(
            lowess::internals::defaults::DEFAULT_ONLINE_UPDATE_MODE,
        ))?,
    };
    map_lowess_result(
        builder
            .adapter(Online)
            .window_capacity(wc)
            .min_points(mp)
            .update_mode(um)
            .build(),
    )
}

// ─── Builder setter methods ───────────────────────────────────────────────────
//
// These impl blocks are in binding_support.rs (compiled only with the `dev`
// feature) so the adapter files (batch.rs, online.rs, streaming.rs) stay free
// of any `#[cfg]` attributes.

impl<T: Float> ParallelBatchLowessBuilder<T> {
    // Set parallel execution mode.
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.base.parallel = Some(parallel);
        self
    }

    // Set the execution backend.
    pub fn backend(mut self, backend: Backend) -> Self {
        self.base.backend = Some(backend);
        self
    }

    // Set the smoothing fraction (span).
    pub fn fraction(mut self, fraction: T) -> Self {
        self.base.fraction = fraction;
        self
    }

    // Set the number of robustness iterations.
    pub fn iterations(mut self, iterations: usize) -> Self {
        self.base.iterations = iterations;
        self
    }

    // Set the interpolation threshold (skip fitting for nearby points).
    pub fn delta(mut self, delta: T) -> Self {
        self.base.delta = Some(delta);
        self
    }

    // Set the kernel weight function.
    #[allow(private_bounds)]
    pub fn weight_function(mut self, wf: impl IntoEnum<WeightFunction>) -> Self {
        match wf.into_enum() {
            Ok(w) => self.base.weight_function = w,
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the robustness method for outlier handling.
    #[allow(private_bounds)]
    pub fn robustness_method(mut self, method: impl IntoEnum<RobustnessMethod>) -> Self {
        match method.into_enum() {
            Ok(m) => self.base.robustness_method = m,
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the residual scaling method (MAR/MAD).
    #[allow(private_bounds)]
    pub fn scaling_method(mut self, method: impl IntoEnum<ScalingMethod>) -> Self {
        match method.into_enum() {
            Ok(m) => self.base.scaling_method = m,
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the zero-weight fallback policy.
    #[allow(private_bounds)]
    pub fn zero_weight_fallback(mut self, fallback: impl IntoEnum<ZeroWeightFallback>) -> Self {
        match fallback.into_enum() {
            Ok(f) => self.base.zero_weight_fallback = f,
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the boundary handling policy.
    #[allow(private_bounds)]
    pub fn boundary_policy(mut self, policy: impl IntoEnum<BoundaryPolicy>) -> Self {
        match policy.into_enum() {
            Ok(p) => self.base.boundary_policy = p,
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Enable auto-convergence for robustness iterations.
    pub fn auto_converge(mut self, tolerance: T) -> Self {
        self.base.auto_converge = Some(tolerance);
        self
    }

    // Enable returning residuals in the output.
    pub fn compute_residuals(mut self, enabled: bool) -> Self {
        self.base.compute_residuals = enabled;
        self
    }

    // Enable returning robustness weights in the result.
    pub fn return_robustness_weights(mut self, enabled: bool) -> Self {
        self.base.return_robustness_weights = enabled;
        self
    }

    // Enable returning diagnostics in the result.
    pub fn return_diagnostics(mut self, enabled: bool) -> Self {
        self.base.return_diagnostics = enabled;
        self
    }

    // Enable confidence intervals at the specified level.
    pub fn confidence_intervals(mut self, level: T) -> Self {
        self.base.interval_type = Some(IntervalMethod::confidence(level));
        self
    }

    // Enable prediction intervals at the specified level.
    pub fn prediction_intervals(mut self, level: T) -> Self {
        self.base.interval_type = Some(IntervalMethod::prediction(level));
        self
    }

    // Enable returning standard errors in the result.
    pub fn return_se(mut self, enabled: bool) -> Self {
        if enabled && self.base.interval_type.is_none() {
            self.base.interval_type = Some(IntervalMethod::se());
        }
        self
    }

    // Set the cross-validation method: `"kfold"` or `"loocv"`.
    pub fn cv_method(mut self, method: &str) -> Self {
        self.cv_method_str = Some(method.to_string());
        self
    }

    // Set the number of folds for K-fold cross-validation (default: 5).
    pub fn cv_k(mut self, k: usize) -> Self {
        self.cv_k_val = k;
        self
    }

    // Set the candidate fractions to evaluate during cross-validation.
    pub fn cv_fractions(mut self, fractions: Vec<T>) -> Self {
        self.base.cv_fractions = Some(fractions);
        self
    }

    // Set the random seed for reproducible cross-validation fold splitting.
    pub fn cv_seed(mut self, seed: u64) -> Self {
        self.base.cv_seed = Some(seed);
        self
    }

    // Set user-defined case weights (one per observation).
    pub fn custom_weights(mut self, weights: Vec<T>) -> Self {
        self.base.custom_weights = Some(weights);
        self
    }
}

impl<T: Float> ParallelOnlineLowessBuilder<T> {
    // Set parallel execution mode.
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.base.parallel = Some(parallel);
        self
    }

    // Set the execution backend.
    pub fn backend(mut self, backend: Backend) -> Self {
        self.base.backend = Some(backend);
        self
    }

    // Set the smoothing fraction (span).
    pub fn fraction(mut self, fraction: T) -> Self {
        self.base.fraction = fraction;
        self
    }

    // Set the number of robustness iterations.
    pub fn iterations(mut self, iterations: usize) -> Self {
        self.base.iterations = iterations;
        self
    }

    // Set the interpolation threshold.
    pub fn delta(mut self, delta: T) -> Self {
        self.base.delta = delta;
        self
    }

    // Set the kernel weight function.
    #[allow(private_bounds)]
    pub fn weight_function(mut self, wf: impl IntoEnum<WeightFunction>) -> Self {
        match wf.into_enum() {
            Ok(w) => self.base.weight_function = w,
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the robustness method for outlier handling.
    #[allow(private_bounds)]
    pub fn robustness_method(mut self, method: impl IntoEnum<RobustnessMethod>) -> Self {
        match method.into_enum() {
            Ok(m) => self.base.robustness_method = m,
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the zero-weight fallback policy.
    #[allow(private_bounds)]
    pub fn zero_weight_fallback(mut self, fallback: impl IntoEnum<ZeroWeightFallback>) -> Self {
        match fallback.into_enum() {
            Ok(f) => self.base.zero_weight_fallback = f,
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the boundary handling policy.
    #[allow(private_bounds)]
    pub fn boundary_policy(mut self, policy: impl IntoEnum<BoundaryPolicy>) -> Self {
        match policy.into_enum() {
            Ok(p) => self.base.boundary_policy = p,
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Enable auto-convergence for robustness iterations.
    pub fn auto_converge(mut self, tolerance: T) -> Self {
        self.base.auto_converge = Some(tolerance);
        self
    }

    // Set whether to compute residuals.
    pub fn compute_residuals(mut self, enabled: bool) -> Self {
        self.base.compute_residuals = enabled;
        self
    }

    // Set whether to return robustness weights.
    pub fn return_robustness_weights(mut self, enabled: bool) -> Self {
        self.base.return_robustness_weights = enabled;
        self
    }

    // Set the window capacity.
    pub fn window_capacity(mut self, capacity: usize) -> Self {
        self.base.window_capacity = capacity;
        self
    }

    // Set the minimum points required before smoothing.
    pub fn min_points(mut self, min_points: usize) -> Self {
        self.base.min_points = min_points;
        self
    }

    // Set the update mode (Incremental/Full).
    #[allow(private_bounds)]
    pub fn update_mode(mut self, mode: impl IntoEnum<UpdateMode>) -> Self {
        match mode.into_enum() {
            Ok(m) => self.base.update_mode = m,
            Err(e) => self.parse_errors.push(e),
        }
        self
    }
}

impl<T: Float> ParallelStreamingLowessBuilder<T> {
    // Set parallel execution mode.
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.base.parallel = Some(parallel);
        self
    }

    // Set the execution backend.
    pub fn backend(mut self, backend: Backend) -> Self {
        self.base.backend = Some(backend);
        self
    }

    // Set the smoothing fraction (span).
    pub fn fraction(mut self, fraction: T) -> Self {
        self.base.fraction = fraction;
        self
    }

    // Set the number of robustness iterations.
    pub fn iterations(mut self, iterations: usize) -> Self {
        self.base.iterations = iterations;
        self
    }

    // Set the interpolation threshold.
    pub fn delta(mut self, delta: T) -> Self {
        self.base.delta = delta;
        self
    }

    // Set the kernel weight function.
    #[allow(private_bounds)]
    pub fn weight_function(mut self, wf: impl IntoEnum<WeightFunction>) -> Self {
        match wf.into_enum() {
            Ok(w) => self.base.weight_function = w,
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the robustness method for outlier handling.
    #[allow(private_bounds)]
    pub fn robustness_method(mut self, method: impl IntoEnum<RobustnessMethod>) -> Self {
        match method.into_enum() {
            Ok(m) => self.base.robustness_method = m,
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the zero-weight fallback policy.
    #[allow(private_bounds)]
    pub fn zero_weight_fallback(mut self, fallback: impl IntoEnum<ZeroWeightFallback>) -> Self {
        match fallback.into_enum() {
            Ok(f) => self.base.zero_weight_fallback = f,
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the boundary handling policy.
    #[allow(private_bounds)]
    pub fn boundary_policy(mut self, policy: impl IntoEnum<BoundaryPolicy>) -> Self {
        match policy.into_enum() {
            Ok(p) => self.base.boundary_policy = p,
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Enable auto-convergence for robustness iterations.
    pub fn auto_converge(mut self, tolerance: T) -> Self {
        self.base.auto_converge = Some(tolerance);
        self
    }

    // Enable returning residuals in the output.
    pub fn compute_residuals(mut self, enabled: bool) -> Self {
        self.base.compute_residuals = enabled;
        self
    }

    // Enable returning robustness weights in the result.
    pub fn return_robustness_weights(mut self, enabled: bool) -> Self {
        self.base.return_robustness_weights = enabled;
        self
    }

    // Set chunk size for processing.
    pub fn chunk_size(mut self, size: usize) -> Self {
        self.base.chunk_size = size;
        self
    }

    // Set overlap between chunks.
    pub fn overlap(mut self, size: usize) -> Self {
        self.base.overlap = size;
        self
    }

    // Set the merge strategy for overlapping chunks.
    #[allow(private_bounds)]
    pub fn merge_strategy(mut self, strategy: impl IntoEnum<MergeStrategy>) -> Self {
        match strategy.into_enum() {
            Ok(s) => self.base.merge_strategy = s,
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Enable returning diagnostics in the result.
    pub fn return_diagnostics(mut self, enabled: bool) -> Self {
        self.base.return_diagnostics = enabled;
        self
    }
}
