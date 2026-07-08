//! C/C++ bindings for fastLowess.
//!
//! Provides C access to the fastLowess Rust library via C FFI.
//! A C++ wrapper header (fastlowess.hpp) provides idiomatic C++ usage.

#![allow(non_snake_case)]
#![allow(unsafe_op_in_unsafe_fn)]

use std::ffi::CStr;
use std::os::raw::{c_char, c_double, c_int, c_ulong};
use std::ptr;

use fastLowess::binding_support;
use fastLowess::internals::adapters::online::ParallelOnlineLowess;
use fastLowess::internals::adapters::streaming::ParallelStreamingLowess;
use fastLowess::internals::api::{MergeStrategy, UpdateMode};
use fastLowess::prelude::{Batch, Lowess as LowessBuilder, LowessResult, Online, Streaming};

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

/// Convert a Vec<f64> to a raw pointer.
fn vec_to_ptr(v: Vec<f64>) -> *mut c_double {
    let mut boxed = v.into_boxed_slice();
    let ptr = boxed.as_mut_ptr();
    std::mem::forget(boxed);
    ptr
}

/// Convert an optional Vec<f64> to a raw pointer.
fn opt_vec_to_ptr(v: Option<Vec<f64>>) -> *mut c_double {
    match v {
        Some(vec) => vec_to_ptr(vec),
        None => ptr::null_mut(),
    }
}

/// Create an error result with the given message.
fn error_result(msg: &str) -> CppLowessResult {
    let mut result = CppLowessResult::default();
    let c_string = std::ffi::CString::new(msg).unwrap_or_default();
    result.error = c_string.into_raw();
    result
}

/// Parse a C string safely.
unsafe fn parse_c_str(s: *const c_char, default: &str) -> &str {
    if s.is_null() {
        default
    } else {
        CStr::from_ptr(s).to_str().unwrap_or(default)
    }
}

impl From<LowessResult<f64>> for CppLowessResult {
    fn from(result: LowessResult<f64>) -> Self {
        let n = result.y.len();

        let (rmse, mae, r_squared, aic, aicc, effective_df, residual_sd) =
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
            };

        CppLowessResult {
            x: vec_to_ptr(result.x),
            y: vec_to_ptr(result.y),
            n: n as c_ulong,
            standard_errors: opt_vec_to_ptr(result.standard_errors),
            confidence_lower: opt_vec_to_ptr(result.confidence_lower),
            confidence_upper: opt_vec_to_ptr(result.confidence_upper),
            prediction_lower: opt_vec_to_ptr(result.prediction_lower),
            prediction_upper: opt_vec_to_ptr(result.prediction_upper),
            residuals: opt_vec_to_ptr(result.residuals),
            robustness_weights: opt_vec_to_ptr(result.robustness_weights),
            cv_scores: opt_vec_to_ptr(result.cv_scores.clone()),
            cv_scores_len: result
                .cv_scores
                .as_ref()
                .map(|v| v.len() as c_ulong)
                .unwrap_or(0),
            fraction_used: result.fraction_used,
            iterations_used: result.iterations_used.map(|i| i as c_int).unwrap_or(-1),
            rmse,
            mae,
            r_squared,
            aic,
            aicc,
            effective_df,
            residual_sd,
            error: ptr::null_mut(),
        }
    }
}

/// Opaque handle to a Lowess batch model.
pub struct CppLowess {
    builder: Option<LowessBuilder<f64>>,
    // Store CV options to apply lazily because of lifetime constraints
    cv_fractions: Option<Vec<f64>>,
    cv_method: Option<String>,
    cv_k: usize,
}

/// Opaque handle to a Lowess streaming model.
pub struct CppStreamingLowess {
    builder: LowessBuilder<f64>,
    streaming_opts: Option<(usize, usize, MergeStrategy)>,
    model: Option<ParallelStreamingLowess<f64>>,
}

/// Opaque handle to a Lowess online model.
pub struct CppOnlineLowess {
    builder: LowessBuilder<f64>,
    online_opts: Option<(usize, usize, UpdateMode)>,
    model: Option<ParallelOnlineLowess<f64>>,
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
    let wf_str = parse_c_str(weight_function, "tricube");
    let rm_str = parse_c_str(robustness_method, "bisquare");
    let sm_str = parse_c_str(scaling_method, "mad");
    let bp_str = parse_c_str(boundary_policy, "extend");
    let zwf_str = parse_c_str(zero_weight_fallback, "use_local_mean");

    let wf = match binding_support::parse_weight_function(wf_str) {
        Ok(v) => v,
        Err(_) => return ptr::null_mut(),
    };
    let rm = match binding_support::parse_robustness_method(rm_str) {
        Ok(v) => v,
        Err(_) => return ptr::null_mut(),
    };
    let sm = match binding_support::parse_scaling_method(sm_str) {
        Ok(v) => v,
        Err(_) => return ptr::null_mut(),
    };
    let bp = match binding_support::parse_boundary_policy(bp_str) {
        Ok(v) => v,
        Err(_) => return ptr::null_mut(),
    };
    let zwf = match binding_support::parse_zero_weight_fallback(zwf_str) {
        Ok(v) => v,
        Err(_) => return ptr::null_mut(),
    };

    let mut builder = LowessBuilder::<f64>::new();
    builder = builder.fraction(fraction);
    builder = builder.iterations(iterations as usize);
    builder = builder.weight_function(wf);
    builder = builder.robustness_method(rm);
    builder = builder.scaling_method(sm);
    builder = builder.zero_weight_fallback(zwf);
    builder = builder.boundary_policy(bp);
    builder = builder.parallel(parallel != 0);

    if !delta.is_nan() {
        builder = builder.delta(delta);
    }
    if !confidence_intervals.is_nan() {
        builder = builder.confidence_intervals(confidence_intervals);
    }
    if !prediction_intervals.is_nan() {
        builder = builder.prediction_intervals(prediction_intervals);
    }
    if return_diagnostics != 0 {
        builder = builder.return_diagnostics();
    }
    if return_residuals != 0 {
        builder = builder.return_residuals();
    }
    if return_robustness_weights != 0 {
        builder = builder.return_robustness_weights();
    }
    if !auto_converge.is_nan() {
        builder = builder.auto_converge(auto_converge);
    }

    // Store CV options
    let cv_fractions_vec = if !cv_fractions.is_null() && cv_fractions_len > 0 {
        let fractions = std::slice::from_raw_parts(cv_fractions, cv_fractions_len as usize);
        Some(fractions.to_vec())
    } else {
        None
    };

    let cv_method_str = parse_c_str(cv_method, "kfold").to_string();

    Box::into_raw(Box::new(CppLowess {
        builder: Some(builder),
        cv_fractions: cv_fractions_vec,
        cv_method: Some(cv_method_str),
        cv_k: cv_k as usize,
    }))
}

/// Fit the batch model.
///
/// # Safety
/// `ptr` must be a valid CppLowess pointer. `x_values` and `y_values` must be
/// valid arrays of length `n`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_lowess_fit(
    ptr: *mut CppLowess,
    x_values: *const c_double,
    y_values: *const c_double,
    n: c_ulong,
) -> CppLowessResult {
    if ptr.is_null() {
        return error_result("Model pointer is null");
    }
    if x_values.is_null() || y_values.is_null() || n == 0 {
        return error_result("Invalid data inputs");
    }

    let lowess = &mut *ptr;
    let x_slice = std::slice::from_raw_parts(x_values, n as usize);
    let y_slice = std::slice::from_raw_parts(y_values, n as usize);

    if let Some(mut builder) = lowess.builder.clone() {
        // Apply CV options if present
        if let Some(fractions) = &lowess.cv_fractions
            && let Some(method) = &lowess.cv_method
        {
            builder = match binding_support::apply_cross_validation(
                builder,
                Some(fractions),
                Some(method),
                Some(lowess.cv_k),
                None,
            ) {
                Ok(b) => b,
                Err(e) => return error_result(&e),
            };
        }

        match builder.adapter(Batch).build() {
            Ok(m) => match m.fit(x_slice, y_slice) {
                Ok(r) => r.into(),
                Err(e) => error_result(&e.to_string()),
            },
            Err(e) => error_result(&e.to_string()),
        }
    } else {
        error_result("Model initialization failed")
    }
}

/// Free batch model.
///
/// # Safety
/// `ptr` must be a valid pointer returned by `cpp_lowess_new` or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_lowess_free(ptr: *mut CppLowess) {
    if !ptr.is_null() {
        let _ = Box::from_raw(ptr);
    }
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
) -> *mut CppStreamingLowess {
    let wf_str = parse_c_str(weight_function, "tricube");
    let rm_str = parse_c_str(robustness_method, "bisquare");
    let sm_str = parse_c_str(scaling_method, "mad");
    let bp_str = parse_c_str(boundary_policy, "extend");
    let zwf_str = parse_c_str(zero_weight_fallback, "use_local_mean");
    let ms_str = parse_c_str(merge_strategy, "weighted");

    let wf = match binding_support::parse_weight_function(wf_str) {
        Ok(v) => v,
        Err(_) => return ptr::null_mut(),
    };
    let rm = match binding_support::parse_robustness_method(rm_str) {
        Ok(v) => v,
        Err(_) => return ptr::null_mut(),
    };
    let sm = match binding_support::parse_scaling_method(sm_str) {
        Ok(v) => v,
        Err(_) => return ptr::null_mut(),
    };
    let bp = match binding_support::parse_boundary_policy(bp_str) {
        Ok(v) => v,
        Err(_) => return ptr::null_mut(),
    };
    let zwf = match binding_support::parse_zero_weight_fallback(zwf_str) {
        Ok(v) => v,
        Err(_) => return ptr::null_mut(),
    };
    let ms = match binding_support::parse_merge_strategy(ms_str) {
        Ok(v) => v,
        Err(_) => return ptr::null_mut(),
    };

    let mut builder = LowessBuilder::<f64>::new();
    builder = builder.fraction(fraction);
    builder = builder.iterations(iterations as usize);
    builder = builder.weight_function(wf);
    builder = builder.robustness_method(rm);
    builder = builder.scaling_method(sm);
    builder = builder.zero_weight_fallback(zwf);
    builder = builder.boundary_policy(bp);
    builder = builder.parallel(parallel != 0);

    if !delta.is_nan() {
        builder = builder.delta(delta);
    }
    if return_diagnostics != 0 {
        builder = builder.return_diagnostics();
    }
    if return_residuals != 0 {
        builder = builder.return_residuals();
    }
    if return_robustness_weights != 0 {
        builder = builder.return_robustness_weights();
    }
    if !auto_converge.is_nan() {
        builder = builder.auto_converge(auto_converge);
    }

    let chunk_size = chunk_size as usize;
    let overlap_size = if overlap < 0 {
        let default = chunk_size / 10;
        default.min(chunk_size.saturating_sub(10)).max(1)
    } else {
        overlap as usize
    };

    Box::into_raw(Box::new(CppStreamingLowess {
        builder,
        streaming_opts: Some((chunk_size, overlap_size, ms)),
        model: None,
    }))
}

#[unsafe(no_mangle)]
/// Process a chunk of data.
///
/// # Safety
/// `ptr` must be valid. `x_values` and `y_values` must be valid arrays of
/// length `n`.
pub unsafe extern "C" fn cpp_streaming_process(
    ptr: *mut CppStreamingLowess,
    x_values: *const c_double,
    y_values: *const c_double,
    n: c_ulong,
) -> CppLowessResult {
    if ptr.is_null() {
        return error_result("Model pointer is null");
    }
    let lowess = &mut *ptr;
    if x_values.is_null() || y_values.is_null() || n == 0 {
        return error_result("Invalid data inputs");
    }
    let x_slice = std::slice::from_raw_parts(x_values, n as usize);
    let y_slice = std::slice::from_raw_parts(y_values, n as usize);

    if lowess.model.is_none()
        && let Some((cs, ov, ms)) = lowess.streaming_opts
    {
        match lowess
            .builder
            .clone()
            .adapter(Streaming)
            .chunk_size(cs)
            .overlap(ov)
            .merge_strategy(ms)
            .build()
        {
            Ok(m) => lowess.model = Some(m),
            Err(e) => return error_result(&e.to_string()),
        }
    }

    if let Some(model) = &mut lowess.model {
        match model.process_chunk(x_slice, y_slice) {
            Ok(r) => r.into(),
            Err(e) => error_result(&e.to_string()),
        }
    } else {
        error_result("Streaming model initialization failed")
    }
}

#[unsafe(no_mangle)]
/// Finalize the streaming process.
///
/// # Safety
/// `ptr` must be valid.
pub unsafe extern "C" fn cpp_streaming_finalize(ptr: *mut CppStreamingLowess) -> CppLowessResult {
    if ptr.is_null() {
        return error_result("Model pointer is null");
    }
    let lowess = &mut *ptr;
    if let Some(model) = &mut lowess.model {
        match model.finalize() {
            Ok(r) => r.into(),
            Err(e) => error_result(&e.to_string()),
        }
    } else {
        error_result("Streaming model not initialized")
    }
}

#[unsafe(no_mangle)]
/// Free streaming model.
///
/// # Safety
/// `ptr` must be valid or null.
pub unsafe extern "C" fn cpp_streaming_free(ptr: *mut CppStreamingLowess) {
    if !ptr.is_null() {
        let _ = Box::from_raw(ptr);
    }
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
    zero_weight_fallback: *const c_char,
    auto_converge: c_double,
    parallel: c_int,
    // Online opts
    window_capacity: c_int,
    min_points: c_int,
    update_mode: *const c_char,
) -> *mut CppOnlineLowess {
    let wf_str = parse_c_str(weight_function, "tricube");
    let rm_str = parse_c_str(robustness_method, "bisquare");
    let sm_str = parse_c_str(scaling_method, "mad");
    let bp_str = parse_c_str(boundary_policy, "extend");
    let zwf_str = parse_c_str(zero_weight_fallback, "use_local_mean");
    let um_str = parse_c_str(update_mode, "full");

    let wf = match binding_support::parse_weight_function(wf_str) {
        Ok(v) => v,
        Err(_) => return ptr::null_mut(),
    };
    let rm = match binding_support::parse_robustness_method(rm_str) {
        Ok(v) => v,
        Err(_) => return ptr::null_mut(),
    };
    let sm = match binding_support::parse_scaling_method(sm_str) {
        Ok(v) => v,
        Err(_) => return ptr::null_mut(),
    };
    let bp = match binding_support::parse_boundary_policy(bp_str) {
        Ok(v) => v,
        Err(_) => return ptr::null_mut(),
    };
    let zwf = match binding_support::parse_zero_weight_fallback(zwf_str) {
        Ok(v) => v,
        Err(_) => return ptr::null_mut(),
    };
    let um = match binding_support::parse_update_mode(um_str) {
        Ok(v) => v,
        Err(_) => return ptr::null_mut(),
    };

    let mut builder = LowessBuilder::<f64>::new();
    builder = builder.fraction(fraction);
    builder = builder.iterations(iterations as usize);
    builder = builder.weight_function(wf);
    builder = builder.robustness_method(rm);
    builder = builder.scaling_method(sm);
    builder = builder.zero_weight_fallback(zwf);
    builder = builder.boundary_policy(bp);
    builder = builder.parallel(parallel != 0);

    if !delta.is_nan() {
        builder = builder.delta(delta);
    }
    if return_robustness_weights != 0 {
        builder = builder.return_robustness_weights();
    }
    if !auto_converge.is_nan() {
        builder = builder.auto_converge(auto_converge);
    }

    Box::into_raw(Box::new(CppOnlineLowess {
        builder,
        online_opts: Some((window_capacity as usize, min_points as usize, um)),
        model: None,
    }))
}

#[unsafe(no_mangle)]
/// Add a single point to the online model. Returns NaN if not enough points yet.
///
/// # Safety
/// `ptr` must be valid.
pub unsafe extern "C" fn cpp_online_add_point(
    ptr: *mut CppOnlineLowess,
    x: c_double,
    y: c_double,
) -> c_double {
    if ptr.is_null() {
        return f64::NAN;
    }
    let lowess = &mut *ptr;

    if lowess.model.is_none()
        && let Some((wc, mp, um)) = lowess.online_opts
    {
        match lowess
            .builder
            .clone()
            .adapter(Online)
            .window_capacity(wc)
            .min_points(mp)
            .update_mode(um)
            .build()
        {
            Ok(m) => lowess.model = Some(m),
            Err(_) => return f64::NAN,
        }
    }

    if let Some(model) = &mut lowess.model {
        match model.add_point(x, y) {
            Ok(Some(o)) => o.smoothed,
            Ok(None) => f64::NAN,
            Err(_) => f64::NAN,
        }
    } else {
        f64::NAN
    }
}

#[unsafe(no_mangle)]
/// Free online model.
///
/// # Safety
/// `ptr` must be valid or null.
pub unsafe extern "C" fn cpp_online_free(ptr: *mut CppOnlineLowess) {
    if !ptr.is_null() {
        let _ = Box::from_raw(ptr);
    }
}

/// Free a CppLowessResult.
///
/// # Safety
/// `result` must be a valid pointer to a CppLowessResult struct.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_lowess_free_result(result: *mut CppLowessResult) {
    if result.is_null() {
        return;
    }

    let r = &mut *result;
    let n = r.n as usize;

    // Free arrays
    if !r.x.is_null() {
        let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(r.x, n));
    }
    if !r.y.is_null() {
        let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(r.y, n));
    }
    if !r.standard_errors.is_null() {
        let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(r.standard_errors, n));
    }
    if !r.confidence_lower.is_null() {
        let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(r.confidence_lower, n));
    }
    if !r.confidence_upper.is_null() {
        let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(r.confidence_upper, n));
    }
    if !r.prediction_lower.is_null() {
        let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(r.prediction_lower, n));
    }
    if !r.prediction_upper.is_null() {
        let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(r.prediction_upper, n));
    }
    if !r.residuals.is_null() {
        let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(r.residuals, n));
    }
    if !r.robustness_weights.is_null() {
        let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(r.robustness_weights, n));
    }
    if !r.cv_scores.is_null() {
        let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(
            r.cv_scores,
            r.cv_scores_len as usize,
        ));
    }

    // Free error string
    if !r.error.is_null() {
        let _ = std::ffi::CString::from_raw(r.error);
    }
}
