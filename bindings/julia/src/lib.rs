//! Julia bindings for fastLowess.
//!
//! Provides Julia access to the fastLowess Rust library via C FFI.

#![allow(non_snake_case)]
#![allow(unsafe_op_in_unsafe_fn)]

use ptr::null_mut;
use std::cell::RefCell;
use std::ffi::CString;
use std::os::raw::{c_char, c_double, c_int, c_ulong};
use std::panic::catch_unwind;
use std::ptr;
use std::slice::from_raw_parts;

use fastLowess::internals::api::LowessBuilder;
use fastLowess::internals::binding_support as shared_parse;
use fastLowess::prelude::LowessResult;

thread_local! {
    static JL_LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_last_error_message(msg: &str) {
    let cmsg = shared_parse::to_cstring_lossy(msg);
    JL_LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = Some(cmsg);
    });
}

fn clear_last_error_message() {
    JL_LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = None;
    });
}

fn null_with_last_error<T>(msg: &str) -> *mut T {
    set_last_error_message(msg);
    null_mut()
}

fn setter_unsupported_constructor_only(name: &str) {
    set_last_error_message(&shared_parse::setter_unsupported_constructor_only_message(
        name,
    ));
}

/// Export the last error message set by a failed constructor.
///
/// Returns a null pointer when no error has been set since the last successful
/// constructor call.  The returned pointer is valid until the next call to any
/// `jl_lowess_*` function on the current thread.
#[unsafe(no_mangle)]
pub extern "C" fn jl_last_error_message() -> *const c_char {
    JL_LAST_ERROR.with(|slot| {
        if let Some(msg) = slot.borrow().as_ref() {
            msg.as_ptr()
        } else {
            ptr::null()
        }
    })
}

// Per-point result from an online update, passed across the FFI boundary.
// has_value = 1 means the window is ready and smoothed is valid; 0 means
// the window is still filling (all other fields are undefined in that case).
#[repr(C)]
pub struct JlOnlineOutput {
    pub has_value: c_int,
    pub smoothed: c_double,
    pub std_error: c_double,         // f64::NAN when not computed
    pub residual: c_double,          // f64::NAN when not computed
    pub robustness_weight: c_double, // f64::NAN when not computed
    pub iterations_used: c_int,      // -1 when not computed
    pub error: *mut c_char,          // NULL if no error
}

impl Default for JlOnlineOutput {
    fn default() -> Self {
        JlOnlineOutput {
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

/// Result struct that can be passed across FFI boundary.
/// All arrays are allocated by Rust and must be freed by Rust.
#[repr(C)]
pub struct JlLowessResult {
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

impl Default for JlLowessResult {
    fn default() -> Self {
        JlLowessResult {
            x: null_mut(),
            y: null_mut(),
            n: 0,
            standard_errors: null_mut(),
            confidence_lower: null_mut(),
            confidence_upper: null_mut(),
            prediction_lower: null_mut(),
            prediction_upper: null_mut(),
            residuals: null_mut(),
            robustness_weights: null_mut(),
            cv_scores: null_mut(),
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
            error: null_mut(),
        }
    }
}

fn error_result(msg: &str) -> JlLowessResult {
    JlLowessResult {
        error: shared_parse::into_raw_error_c_string(msg),
        ..Default::default()
    }
}

fn error_result_from(err: shared_parse::BindingError) -> JlLowessResult {
    error_result(&err.message)
}

fn map_runtime_result<T, E: ToString>(
    result: std::result::Result<T, E>,
) -> std::result::Result<T, Box<JlLowessResult>> {
    shared_parse::map_runtime(result).map_err(|e| Box::new(error_result_from(e)))
}

// Convert LowessResult to JlLowessResult.
fn lowess_result_to_jl(result: LowessResult<f64>) -> JlLowessResult {
    let p = shared_parse::extract_ffi_lowess_result(result);
    JlLowessResult {
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
        error: null_mut(),
    }
}

// ============================================================================
// Stateful Structs (Opaque to C)
// ============================================================================

use fastLowess::internals::adapters::online::ParallelOnlineLowess;
use fastLowess::internals::adapters::streaming::ParallelStreamingLowess;

pub struct JlLowessConfig {
    base_builder: LowessBuilder<f64>,
}

pub struct JlStreamingLowess {
    inner: ParallelStreamingLowess<f64>,
}

pub struct JlOnlineLowess {
    inner: ParallelOnlineLowess<f64>,
}

// ============================================================================
// Lowess (Batch) C API
// ============================================================================

/// Create a new Lowess configuration.
///
/// # Safety
/// The returned pointer must be freed with jl_lowess_free.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jl_lowess_new(
    fraction: c_double,
    iterations: c_int,
    delta: c_double, // NaN for auto
    weight_function: *const c_char,
    robustness_method: *const c_char,
    scaling_method: *const c_char,
    boundary_policy: *const c_char,
    confidence_intervals: c_double, // NaN for disable
    prediction_intervals: c_double, // NaN for disable
    return_diagnostics: c_int,
    return_residuals: c_int,
    return_robustness_weights: c_int,
    zero_weight_fallback: *const c_char,
    auto_converge: c_double, // NaN for disable
    cv_fractions: *const c_double,
    cv_fractions_len: c_ulong,
    cv_method: *const c_char,
    cv_k: c_int,
    parallel: c_int,
    cv_seed: c_ulong, // 0 for none
    return_se: c_int,
) -> *mut JlLowessConfig {
    clear_last_error_message();
    let result = catch_unwind(|| {
        let wf_str = unsafe {
            shared_parse::parse_c_str_or_default(
                weight_function,
                shared_parse::DEFAULT_WEIGHT_FUNCTION,
            )
        };
        let rm_str = unsafe {
            shared_parse::parse_c_str_or_default(
                robustness_method,
                shared_parse::DEFAULT_ROBUSTNESS_METHOD,
            )
        };
        let sm_str = unsafe {
            shared_parse::parse_c_str_or_default(
                scaling_method,
                shared_parse::DEFAULT_SCALING_METHOD,
            )
        };
        let bp_str = unsafe {
            shared_parse::parse_c_str_or_default(
                boundary_policy,
                shared_parse::DEFAULT_BOUNDARY_POLICY,
            )
        };
        let zwf_str = unsafe {
            shared_parse::parse_c_str_or_default(
                zero_weight_fallback,
                shared_parse::DEFAULT_ZERO_WEIGHT_FALLBACK,
            )
        };
        let cv_method_str = unsafe { shared_parse::parse_c_str_or_default(cv_method, "kfold") };

        let cv_fractions_slice =
            unsafe { shared_parse::option_slice_from_ptr(cv_fractions, cv_fractions_len as usize) };

        let base_builder = match shared_parse::map_invalid_arg(shared_parse::apply_builder_options(
            LowessBuilder::<f64>::new(),
            shared_parse::BuilderOptionSet {
                fraction: Some(fraction),
                iterations: Some(iterations as usize),
                delta: (!delta.is_nan()).then_some(delta),
                weight_function: Some(wf_str),
                robustness_method: Some(rm_str),
                scaling_method: Some(sm_str),
                boundary_policy: Some(bp_str),
                zero_weight_fallback: Some(zwf_str),
                auto_converge: (!auto_converge.is_nan()).then_some(auto_converge),
                return_residuals: return_residuals != 0,
                return_robustness_weights: return_robustness_weights != 0,
                return_diagnostics: return_diagnostics != 0,
                return_se: return_se != 0,
                confidence_intervals: (!confidence_intervals.is_nan())
                    .then_some(confidence_intervals),
                prediction_intervals: (!prediction_intervals.is_nan())
                    .then_some(prediction_intervals),
                parallel: Some(parallel != 0),
                cv_fractions: cv_fractions_slice,
                cv_method: Some(cv_method_str),
                cv_k: Some(cv_k as usize),
                cv_seed: (cv_seed != 0).then_some(cv_seed as u64),
                ..Default::default()
            },
        )) {
            Ok(b) => b,
            Err(e) => return null_with_last_error(&e.message),
        };

        Box::into_raw(Box::new(JlLowessConfig { base_builder }))
    });

    match result {
        Ok(ptr) => ptr,
        Err(_) => {
            set_last_error_message(shared_parse::panic_fallback_message());
            null_mut()
        }
    }
}

/// Fit the Lowess model to data.
///
/// # Safety
/// config_ptr must be a valid pointer returned by jl_lowess_new.
/// x and y must be valid arrays of length n.
/// custom_weights is optional: pass null and 0 to omit.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jl_lowess_fit(
    config_ptr: *const JlLowessConfig,
    x: *const c_double,
    y: *const c_double,
    n: c_ulong,
    custom_weights: *const c_double,
    custom_weights_len: c_ulong,
) -> JlLowessResult {
    let result = catch_unwind(|| {
        if config_ptr.is_null() {
            return error_result(shared_parse::CONFIG_POINTER_IS_NULL);
        }
        let config = unsafe { &*config_ptr };

        if x.is_null() || y.is_null() {
            return error_result(shared_parse::XY_ARRAYS_MUST_NOT_BE_NULL);
        }
        if n == 0 {
            return error_result(shared_parse::ARRAY_LENGTH_MUST_BE_GREATER_THAN_ZERO);
        }

        let x_slice = unsafe { from_raw_parts(x, n as usize) };
        let y_slice = unsafe { from_raw_parts(y, n as usize) };

        let cw = unsafe {
            shared_parse::option_vec_from_ptr(custom_weights, custom_weights_len as usize)
        };
        let builder = config.base_builder.clone();

        let model = match shared_parse::build_batch(builder, cw) {
            Ok(m) => m,
            Err(e) => return error_result(&e.message),
        };

        let result = match map_runtime_result(model.fit(x_slice, y_slice)) {
            Ok(r) => r,
            Err(e) => return *e,
        };

        lowess_result_to_jl(result)
    });

    match result {
        Ok(res) => res,
        Err(_) => error_result(shared_parse::panic_fallback_message()),
    }
}

/// Free the LowessResult.
///
/// # Safety
/// `result` must be a valid pointer to a `JlLowessResult` struct.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jl_lowess_free_result(result: *mut JlLowessResult) {
    if result.is_null() {
        return;
    }
    let res = &mut *result;
    let n = res.n as usize;
    let cv_n = res.cv_scores_len as usize;

    shared_parse::free_raw_f64_buffer(res.x, n);
    shared_parse::free_raw_f64_buffer(res.y, n);
    shared_parse::free_raw_f64_buffer(res.standard_errors, n);
    shared_parse::free_raw_f64_buffer(res.confidence_lower, n);
    shared_parse::free_raw_f64_buffer(res.confidence_upper, n);
    shared_parse::free_raw_f64_buffer(res.prediction_lower, n);
    shared_parse::free_raw_f64_buffer(res.prediction_upper, n);
    shared_parse::free_raw_f64_buffer(res.residuals, n);
    shared_parse::free_raw_f64_buffer(res.robustness_weights, n);
    shared_parse::free_raw_f64_buffer(res.cv_scores, cv_n);
    shared_parse::free_raw_c_string(res.error);
}

/// Free the Lowess configuration.
///
/// # Safety
/// `ptr` must be a valid pointer to a `JlLowessConfig` struct.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jl_lowess_free(ptr: *mut JlLowessConfig) {
    if !ptr.is_null() {
        let _ = Box::from_raw(ptr);
    }
}

/// Legacy setter retained for ABI compatibility.
///
/// Configure `cv_seed` in `jl_lowess_new` instead.
///
/// # Safety
/// config_ptr must be a valid mutable pointer returned by jl_lowess_new.
#[unsafe(no_mangle)]
#[allow(clippy::useless_conversion)] // c_ulong is u32 on Windows, u64 on Linux/macOS
pub unsafe extern "C" fn jl_lowess_set_cv_seed(config_ptr: *mut JlLowessConfig, seed: c_ulong) {
    let _ = seed;
    if config_ptr.is_null() {
        set_last_error_message(shared_parse::CONFIG_POINTER_IS_NULL);
        return;
    }
    setter_unsupported_constructor_only("jl_lowess_set_cv_seed");
}

// ============================================================================
// StreamingLowess C API
// ============================================================================

/// Create a new StreamingLowess processor.
///
/// # Safety
/// Pointers must be valid null-terminated strings or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jl_streaming_lowess_new(
    fraction: c_double,
    chunk_size: c_int,
    overlap: c_int, // -1 for auto
    iterations: c_int,
    delta: c_double,
    weight_function: *const c_char,
    robustness_method: *const c_char,
    scaling_method: *const c_char,
    boundary_policy: *const c_char,
    auto_converge: c_double,
    return_diagnostics: c_int,
    return_residuals: c_int,
    return_robustness_weights: c_int,
    zero_weight_fallback: *const c_char,
    merge_strategy: *const c_char,
    parallel: c_int,
) -> *mut JlStreamingLowess {
    clear_last_error_message();
    let result = catch_unwind(|| {
        let wf_str = unsafe {
            shared_parse::parse_c_str_or_default(
                weight_function,
                shared_parse::DEFAULT_WEIGHT_FUNCTION,
            )
        };
        let rm_str = unsafe {
            shared_parse::parse_c_str_or_default(
                robustness_method,
                shared_parse::DEFAULT_ROBUSTNESS_METHOD,
            )
        };
        let sm_str = unsafe {
            shared_parse::parse_c_str_or_default(
                scaling_method,
                shared_parse::DEFAULT_SCALING_METHOD,
            )
        };
        let bp_str = unsafe {
            shared_parse::parse_c_str_or_default(
                boundary_policy,
                shared_parse::DEFAULT_BOUNDARY_POLICY,
            )
        };
        let zwf_str = unsafe {
            shared_parse::parse_c_str_or_default(
                zero_weight_fallback,
                shared_parse::DEFAULT_ZERO_WEIGHT_FALLBACK,
            )
        };
        let ms_str = unsafe {
            shared_parse::parse_c_str_or_default(
                merge_strategy,
                shared_parse::DEFAULT_STREAMING_MERGE_STRATEGY,
            )
        };

        let chunk_size_usize = unwrap_or_return_null!(shared_parse::require_positive_usize(
            "chunk_size",
            chunk_size
        ));
        let iterations_usize = unwrap_or_return_null!(shared_parse::require_non_negative_usize(
            "iterations",
            iterations
        ));

        let builder = match shared_parse::map_invalid_arg(shared_parse::apply_builder_options(
            LowessBuilder::<f64>::new(),
            shared_parse::BuilderOptionSet {
                fraction: Some(fraction),
                iterations: Some(iterations_usize),
                delta: (!delta.is_nan()).then_some(delta),
                weight_function: Some(wf_str),
                robustness_method: Some(rm_str),
                scaling_method: Some(sm_str),
                boundary_policy: Some(bp_str),
                zero_weight_fallback: Some(zwf_str),
                auto_converge: (!auto_converge.is_nan()).then_some(auto_converge),
                return_residuals: return_residuals != 0,
                return_robustness_weights: return_robustness_weights != 0,
                return_diagnostics: return_diagnostics != 0,
                return_se: false,
                parallel: Some(parallel != 0),
                ..Default::default()
            },
        )) {
            Ok(b) => b,
            Err(e) => return null_with_last_error(&e.message),
        };

        let processor = match shared_parse::build_streaming(
            builder,
            Some(chunk_size_usize),
            (overlap >= 0).then_some(overlap as usize),
            Some(ms_str),
        ) {
            Ok(p) => p,
            Err(e) => return null_with_last_error(&e.message),
        };

        Box::into_raw(Box::new(JlStreamingLowess { inner: processor }))
    });

    match result {
        Ok(ptr) => ptr,
        Err(_) => {
            set_last_error_message(shared_parse::panic_fallback_message());
            null_mut()
        }
    }
}

/// Process a chunk of data.
///
/// # Safety
/// `ptr` must be a valid pointer. `x` and `y` must be valid arrays of length `n`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jl_streaming_lowess_process_chunk(
    ptr: *mut JlStreamingLowess,
    x: *const c_double,
    y: *const c_double,
    n: c_ulong,
) -> JlLowessResult {
    let result = catch_unwind(|| {
        if ptr.is_null() {
            return error_result(shared_parse::PROCESSOR_POINTER_IS_NULL);
        }
        let processor = unsafe { &mut *ptr };

        if x.is_null() || y.is_null() {
            return error_result(shared_parse::XY_ARRAYS_MUST_NOT_BE_NULL);
        }
        if n == 0 {
            return error_result(shared_parse::ARRAY_LENGTH_MUST_BE_GREATER_THAN_ZERO);
        }

        let x_slice = unsafe { from_raw_parts(x, n as usize) };
        let y_slice = unsafe { from_raw_parts(y, n as usize) };

        match map_runtime_result(processor.inner.process_chunk(x_slice, y_slice)) {
            Ok(r) => lowess_result_to_jl(r),
            Err(e) => *e,
        }
    });

    match result {
        Ok(res) => res,
        Err(_) => error_result(shared_parse::panic_fallback_message()),
    }
}

/// Finalize streaming and return remaining data.
///
/// # Safety
/// `ptr` must be a valid pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jl_streaming_lowess_finalize(
    ptr: *mut JlStreamingLowess,
) -> JlLowessResult {
    let result = catch_unwind(|| {
        if ptr.is_null() {
            return error_result(shared_parse::PROCESSOR_POINTER_IS_NULL);
        }
        let processor = unsafe { &mut *ptr };

        match map_runtime_result(processor.inner.finalize()) {
            Ok(r) => lowess_result_to_jl(r),
            Err(e) => *e,
        }
    });

    match result {
        Ok(res) => res,
        Err(_) => error_result(shared_parse::panic_fallback_message()),
    }
}

/// Free the StreamingLowess processor.
///
/// # Safety
/// `ptr` must be a valid pointer or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jl_streaming_lowess_free(ptr: *mut JlStreamingLowess) {
    if !ptr.is_null() {
        let _ = Box::from_raw(ptr);
    }
}

// ============================================================================
// OnlineLowess C API
// ============================================================================

/// Create a new OnlineLowess processor.
///
/// # Safety
/// Pointers must be valid null-terminated strings or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jl_online_lowess_new(
    fraction: c_double,
    window_capacity: c_int,
    min_points: c_int,
    iterations: c_int,
    delta: c_double,
    weight_function: *const c_char,
    robustness_method: *const c_char,
    scaling_method: *const c_char,
    boundary_policy: *const c_char,
    update_mode: *const c_char,
    auto_converge: c_double,
    return_robustness_weights: c_int,
    return_diagnostics: c_int,
    return_residuals: c_int,
    zero_weight_fallback: *const c_char,
    parallel: c_int,
) -> *mut JlOnlineLowess {
    clear_last_error_message();
    let result =
        catch_unwind(|| {
            let wf_str = unsafe {
                shared_parse::parse_c_str_or_default(
                    weight_function,
                    shared_parse::DEFAULT_WEIGHT_FUNCTION,
                )
            };
            let rm_str = unsafe {
                shared_parse::parse_c_str_or_default(
                    robustness_method,
                    shared_parse::DEFAULT_ROBUSTNESS_METHOD,
                )
            };
            let sm_str = unsafe {
                shared_parse::parse_c_str_or_default(
                    scaling_method,
                    shared_parse::DEFAULT_SCALING_METHOD,
                )
            };
            let bp_str = unsafe {
                shared_parse::parse_c_str_or_default(
                    boundary_policy,
                    shared_parse::DEFAULT_BOUNDARY_POLICY,
                )
            };
            let zwf_str = unsafe {
                shared_parse::parse_c_str_or_default(
                    zero_weight_fallback,
                    shared_parse::DEFAULT_ZERO_WEIGHT_FALLBACK,
                )
            };
            let um_str = unsafe {
                shared_parse::parse_c_str_or_default(
                    update_mode,
                    shared_parse::DEFAULT_ONLINE_UPDATE_MODE,
                )
            };

            let iterations_usize = unwrap_or_return_null!(
                shared_parse::require_non_negative_usize("iterations", iterations)
            );
            let window_capacity_usize = unwrap_or_return_null!(
                shared_parse::require_positive_usize("window_capacity", window_capacity)
            );
            let min_points_usize = unwrap_or_return_null!(shared_parse::require_positive_usize(
                "min_points",
                min_points
            ));

            let builder = match shared_parse::map_invalid_arg(shared_parse::apply_builder_options(
                LowessBuilder::<f64>::new(),
                shared_parse::BuilderOptionSet {
                    fraction: Some(fraction),
                    iterations: Some(iterations_usize),
                    delta: (!delta.is_nan()).then_some(delta),
                    weight_function: Some(wf_str),
                    robustness_method: Some(rm_str),
                    scaling_method: Some(sm_str),
                    boundary_policy: Some(bp_str),
                    zero_weight_fallback: Some(zwf_str),
                    auto_converge: (!auto_converge.is_nan()).then_some(auto_converge),
                    return_residuals: return_residuals != 0,
                    return_robustness_weights: return_robustness_weights != 0,
                    return_diagnostics: return_diagnostics != 0,
                    return_se: false,
                    parallel: Some(parallel != 0),
                    ..Default::default()
                },
            )) {
                Ok(b) => b,
                Err(e) => return null_with_last_error(&e.message),
            };

            let processor = match shared_parse::build_online(
                builder,
                Some(window_capacity_usize),
                Some(min_points_usize),
                Some(um_str),
            ) {
                Ok(p) => p,
                Err(e) => return null_with_last_error(&e.message),
            };

            Box::into_raw(Box::new(JlOnlineLowess { inner: processor }))
        });

    match result {
        Ok(ptr) => ptr,
        Err(_) => {
            set_last_error_message(shared_parse::panic_fallback_message());
            null_mut()
        }
    }
}

/// Add a single point to the online processor and return the smoothed value for that
/// point, or a result with `has_value = 0` if the window is still filling.
///
/// Returns NaN if not enough points are available or an error occurs.
///
/// # Safety
/// `ptr` must be a valid pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jl_online_lowess_add_point(
    ptr: *mut JlOnlineLowess,
    x: c_double,
    y: c_double,
) -> JlOnlineOutput {
    let result = catch_unwind(|| {
        if ptr.is_null() {
            return JlOnlineOutput {
                error: shared_parse::into_raw_error_c_string(
                    shared_parse::PROCESSOR_POINTER_IS_NULL,
                ),
                ..JlOnlineOutput::default()
            };
        }
        let processor = unsafe { &mut *ptr };

        match processor.inner.add_point(x, y) {
            Err(e) => JlOnlineOutput {
                error: shared_parse::into_raw_error_c_string(&e.to_string()),
                ..JlOnlineOutput::default()
            },
            Ok(None) => JlOnlineOutput::default(),
            Ok(Some(o)) => {
                let (std_error, residual, robustness_weight, iterations_used) =
                    shared_parse::extract_online_output(&o);
                JlOnlineOutput {
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
    });

    result.unwrap_or_else(|_| JlOnlineOutput {
        error: shared_parse::into_raw_error_c_string(shared_parse::panic_fallback_message()),
        ..JlOnlineOutput::default()
    })
}

/// Free the error field of a JlOnlineOutput returned by jl_online_lowess_add_point.
/// No-op when output is null or the error field is already null.
///
/// # Safety
/// `output` must be a valid pointer to a JlOnlineOutput or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jl_online_free_output(output: *mut JlOnlineOutput) {
    if !output.is_null() {
        shared_parse::free_raw_c_string((*output).error);
        (*output).error = ptr::null_mut();
    }
}

/// Free the OnlineLowess processor.
///
/// # Safety
/// `ptr` must be a valid pointer or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jl_online_lowess_free(ptr: *mut JlOnlineLowess) {
    if !ptr.is_null() {
        let _ = Box::from_raw(ptr);
    }
}

#[macro_export]
macro_rules! unwrap_or_return_null {
    ($e:expr) => {
        match $e {
            Ok(val) => val,
            Err(err) => {
                set_last_error_message(&err.to_string());
                return null_mut();
            }
        }
    };
}
