//! C/C++ bindings for fastLowess.
//!
//! Provides C access to the fastLowess Rust library via C FFI.
//! A C++ wrapper header (fastlowess.hpp) provides idiomatic C++ usage.

#![allow(non_snake_case)]
#![allow(unsafe_op_in_unsafe_fn)]

use std::cell::RefCell;
use std::ffi::CString;
use std::os::raw::{c_char, c_double, c_int, c_ulong};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;

use fastLowess::internals::adapters::online::ParallelOnlineLowess;
use fastLowess::internals::adapters::streaming::ParallelStreamingLowess;
use fastLowess::internals::api::LowessBuilder;
use fastLowess::internals::binding_support as shared_parse;
use fastLowess::prelude::LowessResult;

thread_local! {
    #[allow(clippy::missing_const_for_thread_local)]
    static CPP_LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_last_error(msg: &str) {
    let cmsg = shared_parse::to_cstring_lossy(msg);
    CPP_LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = Some(cmsg);
    });
}

fn clear_last_error() {
    CPP_LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = None;
    });
}

fn null_with_error<T>(msg: &str) -> *mut T {
    set_last_error(msg);
    ptr::null_mut()
}

fn error_result_from(err: shared_parse::BindingError) -> CppLowessResult {
    error_result(&err.message)
}

#[allow(clippy::result_large_err)]
fn map_invalid_arg_result<T, E: ToString>(
    result: std::result::Result<T, E>,
) -> std::result::Result<T, CppLowessResult> {
    shared_parse::map_invalid_arg(result).map_err(error_result_from)
}

#[allow(clippy::result_large_err)]
fn map_runtime_result<T, E: ToString>(
    result: std::result::Result<T, E>,
) -> std::result::Result<T, CppLowessResult> {
    shared_parse::map_runtime(result).map_err(error_result_from)
}

fn with_panic_result<F>(f: F) -> CppLowessResult
where
    F: FnOnce() -> CppLowessResult,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(v) => v,
        Err(_) => error_result(shared_parse::panic_fallback_message()),
    }
}

fn with_panic_ptr<T, F>(f: F) -> *mut T
where
    F: FnOnce() -> *mut T,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(v) => v,
        Err(_) => null_with_error(shared_parse::panic_fallback_message()),
    }
}

fn with_panic_void<F>(f: F)
where
    F: FnOnce(),
{
    if catch_unwind(AssertUnwindSafe(f)).is_err() {
        set_last_error(shared_parse::panic_fallback_message());
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn cpp_last_error_message() -> *const c_char {
    CPP_LAST_ERROR.with(|slot| {
        if let Some(msg) = slot.borrow().as_ref() {
            msg.as_ptr()
        } else {
            ptr::null()
        }
    })
}

// Per-point result from an online update, passed across the FFI boundary.
// has_value = 1 means the window is ready and smoothed is valid; 0 means the
// window is still filling (caller should treat it as no output yet).
// Non-computed optional fields use f64::NAN (for floats) or -1 (for int).
// error = NULL if no error, otherwise points to a null-terminated error string.
#[repr(C)]
pub struct CppOnlineOutput {
    pub has_value: c_int,
    pub smoothed: c_double,
    pub std_error: c_double,
    pub residual: c_double,
    pub robustness_weight: c_double,
    pub iterations_used: c_int,
    pub error: *mut c_char, // NULL if no error
}

/// Result struct that can be passed across FFI boundary.
/// All arrays are allocated by Rust and must be freed by Rust.
#[repr(C)]
pub struct CppLowessResult {
    /// Sorted x values (length = n)
    pub x: *mut c_double,
    /// Smoothed y values (length = n)
    pub y: *mut c_double,
    /// Number of data points
    pub n: c_ulong,

    /// Standard errors (NULL if not computed)
    pub standard_errors: *mut c_double,
    /// Lower confidence bounds (NULL if not computed)
    pub confidence_lower: *mut c_double,
    /// Upper confidence bounds (NULL if not computed)
    pub confidence_upper: *mut c_double,
    /// Lower prediction bounds (NULL if not computed)
    pub prediction_lower: *mut c_double,
    /// Upper prediction bounds (NULL if not computed)
    pub prediction_upper: *mut c_double,
    /// Residuals (NULL if not computed)
    pub residuals: *mut c_double,
    /// Robustness weights (NULL if not computed)
    pub robustness_weights: *mut c_double,
    /// Cross-validation scores (NULL if not computed, length = cv_scores_len)
    pub cv_scores: *mut c_double,
    /// Number of cross-validation scores
    pub cv_scores_len: c_ulong,

    /// Fraction used for smoothing
    pub fraction_used: c_double,
    /// Number of iterations performed (-1 if not available)
    pub iterations_used: c_int,

    /// Diagnostics (NaN if not computed)
    pub rmse: c_double,
    pub mae: c_double,
    pub r_squared: c_double,
    pub aic: c_double,
    pub aicc: c_double,
    pub effective_df: c_double,
    pub residual_sd: c_double,

    /// Error message (NULL if no error)
    pub error: *mut c_char,
}

impl Default for CppLowessResult {
    fn default() -> Self {
        CppLowessResult {
            x: ptr::null_mut(),
            y: ptr::null_mut(),
            n: 0,
            standard_errors: ptr::null_mut(),
            confidence_lower: ptr::null_mut(),
            confidence_upper: ptr::null_mut(),
            prediction_lower: ptr::null_mut(),
            prediction_upper: ptr::null_mut(),
            residuals: ptr::null_mut(),
            robustness_weights: ptr::null_mut(),
            cv_scores: ptr::null_mut(),
            cv_scores_len: 0,
            fraction_used: 0.0,
            iterations_used: -1,
            rmse: f64::NAN,
            mae: f64::NAN,
            r_squared: f64::NAN,
            aic: f64::NAN,
            aicc: f64::NAN,
            effective_df: f64::NAN,
            residual_sd: f64::NAN,
            error: ptr::null_mut(),
        }
    }
}

// Create an error result with the given message.
fn error_result(msg: &str) -> CppLowessResult {
    CppLowessResult {
        error: shared_parse::into_raw_error_c_string(msg),
        ..Default::default()
    }
}

impl Default for CppOnlineOutput {
    fn default() -> Self {
        CppOnlineOutput {
            has_value: 0,
            smoothed: f64::NAN,
            std_error: f64::NAN,
            residual: f64::NAN,
            robustness_weight: f64::NAN,
            iterations_used: -1,
            error: ptr::null_mut(),
        }
    }
}

impl From<LowessResult<f64>> for CppLowessResult {
    fn from(result: LowessResult<f64>) -> Self {
        let p = shared_parse::extract_ffi_lowess_result(result);
        CppLowessResult {
            x: p.x,
            y: p.y,
            n: p.n as c_ulong,
            standard_errors: p.standard_errors,
            confidence_lower: p.confidence_lower,
            confidence_upper: p.confidence_upper,
            prediction_lower: p.prediction_lower,
            prediction_upper: p.prediction_upper,
            residuals: p.residuals,
            robustness_weights: p.robustness_weights,
            cv_scores: p.cv_scores,
            cv_scores_len: p.cv_scores_len as c_ulong,
            fraction_used: p.fraction_used,
            iterations_used: p.iterations_used,
            rmse: p.rmse,
            mae: p.mae,
            r_squared: p.r_squared,
            aic: p.aic,
            aicc: p.aicc,
            effective_df: p.effective_df,
            residual_sd: p.residual_sd,
            error: ptr::null_mut(),
        }
    }
}

// Opaque handle to a Lowess batch model.
pub struct CppLowess {
    builder: Option<LowessBuilder<f64>>,
    // Store CV options to apply lazily because of lifetime constraints
    cv_fractions: Option<Vec<f64>>,
    cv_method: Option<String>,
    cv_k: usize,
    cv_seed: Option<u64>,
}

// Opaque handle to a Lowess streaming model.
pub struct CppStreamingLowess {
    model: Option<ParallelStreamingLowess<f64>>,
}

// Opaque handle to a Lowess online model.
pub struct CppOnlineLowess {
    model: Option<ParallelOnlineLowess<f64>>,
}

#[allow(dead_code)]
fn setter_unsupported_eager_lifecycle(name: &str) {
    set_last_error(&shared_parse::setter_unsupported_eager_message(name));
}

/// C++ wrapper constructor.
///
/// # Safety
/// Pointers must be valid null-terminated strings or null. Arrays must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_lowess_new(
    fraction: c_double,
    iterations: c_int,
    delta: c_double,
    weight_function: *const c_char,
    robustness_method: *const c_char,
    scaling_method: *const c_char,
    boundary_policy: *const c_char,
    confidence_intervals: c_double,
    prediction_intervals: c_double,
    return_diagnostics: c_int,
    return_residuals: c_int,
    return_robustness_weights: c_int,
    zero_weight_fallback: *const c_char,
    auto_converge: c_double,
    cv_fractions: *const c_double,
    cv_fractions_len: c_ulong,
    cv_method: *const c_char,
    cv_k: c_int,
    parallel: c_int,
) -> *mut CppLowess {
    with_panic_ptr(|| {
        clear_last_error();
        let wf_str = shared_parse::parse_c_str_or_default(
            weight_function,
            shared_parse::DEFAULT_WEIGHT_FUNCTION,
        );
        let rm_str = shared_parse::parse_c_str_or_default(
            robustness_method,
            shared_parse::DEFAULT_ROBUSTNESS_METHOD,
        );
        let sm_str = shared_parse::parse_c_str_or_default(
            scaling_method,
            shared_parse::DEFAULT_SCALING_METHOD,
        );
        let bp_str = shared_parse::parse_c_str_or_default(
            boundary_policy,
            shared_parse::DEFAULT_BOUNDARY_POLICY,
        );
        let zwf_str = shared_parse::parse_c_str_or_default(
            zero_weight_fallback,
            shared_parse::DEFAULT_ZERO_WEIGHT_FALLBACK,
        );

        let iterations = match shared_parse::require_non_negative_usize("iterations", iterations) {
            Ok(v) => v,
            Err(e) => return null_with_error(&e),
        };

        let cv_fractions_vec =
            shared_parse::option_vec_from_ptr(cv_fractions, cv_fractions_len as usize);
        let cv_method_str =
            shared_parse::parse_c_str_or_default(cv_method, shared_parse::DEFAULT_CV_METHOD)
                .to_string();
        let cv_k_usize = cv_k.max(2) as usize;

        let builder = match shared_parse::apply_builder_options(
            LowessBuilder::<f64>::new(),
            shared_parse::BuilderOptionSet {
                fraction: Some(fraction),
                iterations: Some(iterations),
                delta: (!delta.is_nan()).then_some(delta),
                weight_function: Some(wf_str),
                robustness_method: Some(rm_str),
                zero_weight_fallback: Some(zwf_str),
                boundary_policy: Some(bp_str),
                scaling_method: Some(sm_str),
                auto_converge: (!auto_converge.is_nan()).then_some(auto_converge),
                return_residuals: return_residuals != 0,
                return_robustness_weights: return_robustness_weights != 0,
                return_diagnostics: return_diagnostics != 0,
                confidence_intervals: (!confidence_intervals.is_nan())
                    .then_some(confidence_intervals),
                prediction_intervals: (!prediction_intervals.is_nan())
                    .then_some(prediction_intervals),
                parallel: Some(parallel != 0),
                ..Default::default()
            },
        ) {
            Ok(v) => v,
            Err(e) => return null_with_error(&e),
        };

        Box::into_raw(Box::new(CppLowess {
            builder: Some(builder),
            cv_fractions: cv_fractions_vec,
            cv_method: Some(cv_method_str),
            cv_k: cv_k_usize,
            cv_seed: None,
        }))
    })
}

/// Set CV seed for reproducible K-fold splits.
///
/// # Safety
/// ptr must be valid.
#[unsafe(no_mangle)]
#[allow(clippy::useless_conversion)] // c_ulong is u32 on Windows, u64 on Linux/macOS
pub unsafe extern "C" fn cpp_lowess_set_cv_seed(ptr: *mut CppLowess, seed: c_ulong) {
    with_panic_void(|| {
        if !ptr.is_null() {
            unsafe { (*ptr).cv_seed = Some(u64::from(seed)) };
        }
    });
}

/// Fit the batch model.
///
/// # Safety
/// `ptr` must be a valid CppLowess pointer. `x_values` and `y_values` must be
/// valid arrays of length `n`. `custom_weights` is optional: pass null and 0 to omit.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_lowess_fit(
    ptr: *mut CppLowess,
    x_values: *const c_double,
    y_values: *const c_double,
    n: c_ulong,
    custom_weights: *const c_double,
    custom_weights_len: c_ulong,
) -> CppLowessResult {
    with_panic_result(|| {
        if ptr.is_null() {
            return error_result(shared_parse::MODEL_POINTER_IS_NULL);
        }
        if x_values.is_null() || y_values.is_null() || n == 0 {
            return error_result(shared_parse::INVALID_DATA_INPUTS);
        }

        let lowess = &mut *ptr;
        let x_slice = std::slice::from_raw_parts(x_values, n as usize);
        let y_slice = std::slice::from_raw_parts(y_values, n as usize);

        let cw = shared_parse::option_vec_from_ptr(custom_weights, custom_weights_len as usize);

        if let Some(mut builder) = lowess.builder.clone() {
            builder = match map_invalid_arg_result(shared_parse::apply_cross_validation(
                builder,
                lowess.cv_fractions.as_deref(),
                lowess.cv_method.as_deref(),
                Some(lowess.cv_k),
                lowess.cv_seed,
            )) {
                Ok(b) => b,
                Err(e) => return e,
            };

            let model = match shared_parse::build_batch(builder, cw) {
                Ok(m) => m,
                Err(e) => return error_result(&e.message),
            };
            match map_runtime_result(model.fit(x_slice, y_slice)) {
                Ok(r) => r.into(),
                Err(e) => e,
            }
        } else {
            error_result(shared_parse::MODEL_NOT_INITIALIZED)
        }
    })
}

/// Free batch model.
///
/// # Safety
/// `ptr` must be a valid pointer returned by `cpp_lowess_new` or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_lowess_free(ptr: *mut CppLowess) {
    with_panic_void(|| {
        if !ptr.is_null() {
            let _ = Box::from_raw(ptr);
        }
    });
}

/// Create a new Streaming Lowess model.
///
/// # Safety
/// Pointers must be valid null-terminated strings or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_streaming_new(
    fraction: c_double,
    iterations: c_int,
    delta: c_double,
    weight_function: *const c_char,
    robustness_method: *const c_char,
    scaling_method: *const c_char,
    boundary_policy: *const c_char,
    return_diagnostics: c_int,
    return_residuals: c_int,
    return_robustness_weights: c_int,
    zero_weight_fallback: *const c_char,
    auto_converge: c_double,
    parallel: c_int,
    // Streaming opts
    chunk_size: c_int,
    overlap: c_int,
    merge_strategy: *const c_char,
    // Interval options
    confidence_intervals: c_double,
    prediction_intervals: c_double,
) -> *mut CppStreamingLowess {
    with_panic_ptr(|| {
        clear_last_error();
        let wf_str = shared_parse::parse_c_str_or_default(
            weight_function,
            shared_parse::DEFAULT_WEIGHT_FUNCTION,
        );
        let rm_str = shared_parse::parse_c_str_or_default(
            robustness_method,
            shared_parse::DEFAULT_ROBUSTNESS_METHOD,
        );
        let sm_str = shared_parse::parse_c_str_or_default(
            scaling_method,
            shared_parse::DEFAULT_SCALING_METHOD,
        );
        let bp_str = shared_parse::parse_c_str_or_default(
            boundary_policy,
            shared_parse::DEFAULT_BOUNDARY_POLICY,
        );
        let zwf_str = shared_parse::parse_c_str_or_default(
            zero_weight_fallback,
            shared_parse::DEFAULT_ZERO_WEIGHT_FALLBACK,
        );
        let ms_str = shared_parse::parse_c_str_or_default(
            merge_strategy,
            shared_parse::DEFAULT_STREAMING_MERGE_STRATEGY,
        );

        let chunk_size = match shared_parse::require_positive_usize("chunk_size", chunk_size) {
            Ok(v) => v,
            Err(e) => return null_with_error(&e),
        };

        let builder = match shared_parse::apply_builder_options(
            LowessBuilder::<f64>::new(),
            shared_parse::BuilderOptionSet {
                fraction: Some(fraction),
                iterations: Some(iterations as usize),
                delta: (!delta.is_nan()).then_some(delta),
                weight_function: Some(wf_str),
                robustness_method: Some(rm_str),
                zero_weight_fallback: Some(zwf_str),
                boundary_policy: Some(bp_str),
                scaling_method: Some(sm_str),
                auto_converge: (!auto_converge.is_nan()).then_some(auto_converge),
                return_residuals: return_residuals != 0,
                return_robustness_weights: return_robustness_weights != 0,
                return_diagnostics: return_diagnostics != 0,
                confidence_intervals: (!confidence_intervals.is_nan())
                    .then_some(confidence_intervals),
                prediction_intervals: (!prediction_intervals.is_nan())
                    .then_some(prediction_intervals),
                parallel: Some(parallel != 0),
                ..Default::default()
            },
        ) {
            Ok(v) => v,
            Err(e) => return null_with_error(&e),
        };

        let model = match shared_parse::build_streaming(
            builder,
            Some(chunk_size),
            (overlap >= 0).then_some(overlap as usize),
            Some(ms_str),
        ) {
            Ok(m) => m,
            Err(e) => return null_with_error(&e.message),
        };

        Box::into_raw(Box::new(CppStreamingLowess { model: Some(model) }))
    })
}

/// Process a chunk of data.
///
/// # Safety
/// `ptr` must be valid. `x_values` and `y_values` must be valid arrays of
/// length `n`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_streaming_process(
    ptr: *mut CppStreamingLowess,
    x_values: *const c_double,
    y_values: *const c_double,
    n: c_ulong,
) -> CppLowessResult {
    with_panic_result(|| {
        if ptr.is_null() {
            return error_result(shared_parse::MODEL_POINTER_IS_NULL);
        }
        let lowess = &mut *ptr;
        if x_values.is_null() || y_values.is_null() || n == 0 {
            return error_result(shared_parse::INVALID_DATA_INPUTS);
        }
        let x_slice = std::slice::from_raw_parts(x_values, n as usize);
        let y_slice = std::slice::from_raw_parts(y_values, n as usize);

        if let Some(model) = &mut lowess.model {
            match map_runtime_result(model.process_chunk(x_slice, y_slice)) {
                Ok(r) => r.into(),
                Err(e) => e,
            }
        } else {
            error_result(shared_parse::MODEL_NOT_INITIALIZED)
        }
    })
}

/// Finalize the streaming process.
///
/// # Safety
/// `ptr` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_streaming_finalize(ptr: *mut CppStreamingLowess) -> CppLowessResult {
    with_panic_result(|| {
        if ptr.is_null() {
            return error_result(shared_parse::MODEL_POINTER_IS_NULL);
        }
        let lowess = &mut *ptr;
        if let Some(model) = &mut lowess.model {
            match map_runtime_result(model.finalize()) {
                Ok(r) => r.into(),
                Err(e) => e,
            }
        } else {
            error_result(shared_parse::MODEL_NOT_INITIALIZED)
        }
    })
}

/// Free streaming model.
///
/// # Safety
/// `ptr` must be valid or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_streaming_free(ptr: *mut CppStreamingLowess) {
    with_panic_void(|| {
        if !ptr.is_null() {
            let _ = Box::from_raw(ptr);
        }
    });
}

/// Create a new Online Lowess model.
///
/// # Safety
/// Pointers must be valid null-terminated strings or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_online_new(
    fraction: c_double,
    iterations: c_int,
    delta: c_double,
    weight_function: *const c_char,
    robustness_method: *const c_char,
    scaling_method: *const c_char,
    boundary_policy: *const c_char,
    return_robustness_weights: c_int,
    return_diagnostics: c_int,
    return_residuals: c_int,
    zero_weight_fallback: *const c_char,
    auto_converge: c_double,
    parallel: c_int,
    // Online opts
    window_capacity: c_int,
    min_points: c_int,
    update_mode: *const c_char,
    // Interval options
    confidence_intervals: c_double,
    prediction_intervals: c_double,
) -> *mut CppOnlineLowess {
    with_panic_ptr(|| {
        clear_last_error();
        let wf_str = shared_parse::parse_c_str_or_default(
            weight_function,
            shared_parse::DEFAULT_WEIGHT_FUNCTION,
        );
        let rm_str = shared_parse::parse_c_str_or_default(
            robustness_method,
            shared_parse::DEFAULT_ROBUSTNESS_METHOD,
        );
        let sm_str = shared_parse::parse_c_str_or_default(
            scaling_method,
            shared_parse::DEFAULT_SCALING_METHOD,
        );
        let bp_str = shared_parse::parse_c_str_or_default(
            boundary_policy,
            shared_parse::DEFAULT_BOUNDARY_POLICY,
        );
        let zwf_str = shared_parse::parse_c_str_or_default(
            zero_weight_fallback,
            shared_parse::DEFAULT_ZERO_WEIGHT_FALLBACK,
        );
        let um_str = shared_parse::parse_c_str_or_default(
            update_mode,
            shared_parse::DEFAULT_ONLINE_UPDATE_MODE,
        );

        let window_capacity =
            match shared_parse::require_positive_usize("window_capacity", window_capacity) {
                Ok(v) => v,
                Err(e) => return null_with_error(&e),
            };
        let min_points = match shared_parse::require_positive_usize("min_points", min_points) {
            Ok(v) => v,
            Err(e) => return null_with_error(&e),
        };

        let builder = match shared_parse::apply_builder_options(
            LowessBuilder::<f64>::new(),
            shared_parse::BuilderOptionSet {
                fraction: Some(fraction),
                iterations: Some(iterations as usize),
                delta: (!delta.is_nan()).then_some(delta),
                weight_function: Some(wf_str),
                robustness_method: Some(rm_str),
                zero_weight_fallback: Some(zwf_str),
                boundary_policy: Some(bp_str),
                scaling_method: Some(sm_str),
                auto_converge: (!auto_converge.is_nan()).then_some(auto_converge),
                return_residuals: return_residuals != 0,
                return_robustness_weights: return_robustness_weights != 0,
                return_diagnostics: return_diagnostics != 0,
                confidence_intervals: (!confidence_intervals.is_nan())
                    .then_some(confidence_intervals),
                prediction_intervals: (!prediction_intervals.is_nan())
                    .then_some(prediction_intervals),
                parallel: Some(parallel != 0),
                ..Default::default()
            },
        ) {
            Ok(v) => v,
            Err(e) => return null_with_error(&e),
        };

        let model = match shared_parse::build_online(
            builder,
            Some(window_capacity),
            Some(min_points),
            Some(um_str),
        ) {
            Ok(m) => m,
            Err(e) => return null_with_error(&e.message),
        };

        Box::into_raw(Box::new(CppOnlineLowess { model: Some(model) }))
    })
}

/// Add a single point to the model and return its smoothed value.
/// `has_value = 0` in the result means the window is still filling.
///
/// # Safety
/// `ptr` must be a valid `CppOnlineLowess` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_online_add_point(
    ptr: *mut CppOnlineLowess,
    x: c_double,
    y: c_double,
) -> CppOnlineOutput {
    let make_error = |msg: &str| -> CppOnlineOutput {
        CppOnlineOutput {
            error: shared_parse::to_cstring_lossy(msg).into_raw(),
            ..CppOnlineOutput::default()
        }
    };

    match catch_unwind(AssertUnwindSafe(|| {
        if ptr.is_null() {
            return make_error(shared_parse::MODEL_POINTER_IS_NULL);
        }
        let lowess = unsafe { &mut *ptr };

        if let Some(model) = &mut lowess.model {
            match model.add_point(x, y) {
                Err(e) => make_error(&e.to_string()),
                Ok(None) => CppOnlineOutput::default(),
                Ok(Some(o)) => {
                    let (std_error, residual, robustness_weight, iterations_used) =
                        shared_parse::extract_online_output(&o);
                    CppOnlineOutput {
                        has_value: 1,
                        smoothed: o.smoothed,
                        std_error,
                        residual,
                        robustness_weight,
                        iterations_used,
                        error: ptr::null_mut(),
                    }
                }
            }
        } else {
            make_error(shared_parse::MODEL_NOT_INITIALIZED)
        }
    })) {
        Ok(v) => v,
        Err(_) => make_error(shared_parse::panic_fallback_message()),
    }
}

/// Free the error string in a CppOnlineOutput (call only when error != NULL).
///
/// # Safety
/// `output` must be a valid pointer and `output->error` must have been allocated by Rust.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_online_free_output(output: *mut CppOnlineOutput) {
    with_panic_void(|| {
        if !output.is_null() {
            let out = unsafe { &mut *output };
            shared_parse::free_raw_c_string(out.error);
            out.error = ptr::null_mut();
        }
    });
}

/// Free online model.
///
/// # Safety
/// `ptr` must be valid or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_online_free(ptr: *mut CppOnlineLowess) {
    with_panic_void(|| {
        if !ptr.is_null() {
            let _ = Box::from_raw(ptr);
        }
    });
}

/// Free a CppLowessResult.
///
/// # Safety
/// `result` must be a valid pointer to a CppLowessResult struct.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_lowess_free_result(result: *mut CppLowessResult) {
    with_panic_void(|| {
        if result.is_null() {
            return;
        }

        let r = &mut *result;
        let n = r.n as usize;
        let cv_n = r.cv_scores_len as usize;

        shared_parse::free_raw_f64_buffer(r.x, n);
        shared_parse::free_raw_f64_buffer(r.y, n);
        shared_parse::free_raw_f64_buffer(r.standard_errors, n);
        shared_parse::free_raw_f64_buffer(r.confidence_lower, n);
        shared_parse::free_raw_f64_buffer(r.confidence_upper, n);
        shared_parse::free_raw_f64_buffer(r.prediction_lower, n);
        shared_parse::free_raw_f64_buffer(r.prediction_upper, n);
        shared_parse::free_raw_f64_buffer(r.residuals, n);
        shared_parse::free_raw_f64_buffer(r.robustness_weights, n);
        shared_parse::free_raw_f64_buffer(r.cv_scores, cv_n);
        shared_parse::free_raw_c_string(r.error);
    });
}
