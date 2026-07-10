//! Julia bindings for fastLowess.
//!
//! Provides Julia access to the fastLowess Rust library via C FFI.

#![allow(non_snake_case)]
#![allow(unsafe_op_in_unsafe_fn)]

use ptr::null_mut;
use std::os::raw::{c_char, c_double, c_int, c_ulong};
use std::panic::catch_unwind;
use std::ptr;
use std::slice::from_raw_parts;

use fastLowess::internals::api::{Batch, LowessBuilder, Online, Streaming};
use fastLowess::internals::binding_support::{
    self, BoundaryPolicy, RobustnessMethod, ScalingMethod, WeightFunction, ZeroWeightFallback,
};
use fastLowess::prelude::LowessResult;

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

/// Create an error result with the given message.
fn error_result(msg: &str) -> JlLowessResult {
    JlLowessResult {
        error: binding_support::into_raw_error_c_string(msg),
        ..Default::default()
    }
}

/// Convert LowessResult to JlLowessResult.
fn lowess_result_to_jl(result: LowessResult<f64>) -> JlLowessResult {
    let n = result.y.len();
    let (rmse, mae, r_squared, aic, aicc, effective_df, residual_sd) =
        binding_support::extract_diagnostics(&result);
    let cv_scores_len = result
        .cv_scores
        .as_ref()
        .map(|v| v.len() as c_ulong)
        .unwrap_or(0);

    JlLowessResult {
        x: binding_support::vec_to_raw_ptr(result.x),
        y: binding_support::vec_to_raw_ptr(result.y),
        n: n as c_ulong,
        standard_errors: binding_support::opt_vec_to_raw_ptr(result.standard_errors),
        confidence_lower: binding_support::opt_vec_to_raw_ptr(result.confidence_lower),
        confidence_upper: binding_support::opt_vec_to_raw_ptr(result.confidence_upper),
        prediction_lower: binding_support::opt_vec_to_raw_ptr(result.prediction_lower),
        prediction_upper: binding_support::opt_vec_to_raw_ptr(result.prediction_upper),
        residuals: binding_support::opt_vec_to_raw_ptr(result.residuals),
        robustness_weights: binding_support::opt_vec_to_raw_ptr(result.robustness_weights),
        cv_scores: binding_support::opt_vec_to_raw_ptr(result.cv_scores),
        cv_scores_len,
        fraction_used: result.fraction_used,
        iterations_used: result.iterations_used.map(|i| i as c_int).unwrap_or(-1),
        rmse,
        mae,
        r_squared,
        aic,
        aicc,
        effective_df,
        residual_sd,
        error: null_mut(),
    }
}

// ============================================================================
// Stateful Structs (Opaque to C)
// ============================================================================

use fastLowess::internals::adapters::online::ParallelOnlineLowess;
use fastLowess::internals::adapters::streaming::ParallelStreamingLowess;

pub struct JlLowessConfig {
    fraction: f64,
    iterations: usize,
    delta: Option<f64>,
    weight_function: WeightFunction,
    robustness_method: RobustnessMethod,
    scaling_method: ScalingMethod,
    zero_weight_fallback: ZeroWeightFallback,
    boundary_policy: BoundaryPolicy,
    auto_converge: Option<f64>,
    confidence_intervals: Option<f64>,
    prediction_intervals: Option<f64>,
    return_diagnostics: bool,
    return_residuals: bool,
    return_robustness_weights: bool,
    cv_fractions: Option<Vec<f64>>,
    cv_method: String,
    cv_k: usize,
    parallel: bool,
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
) -> *mut JlLowessConfig {
    let result = catch_unwind(|| {
        let wf_str = unsafe { binding_support::parse_c_str_or_default(weight_function, "tricube") };
        let rm_str =
            unsafe { binding_support::parse_c_str_or_default(robustness_method, "bisquare") };
        let sm_str = unsafe { binding_support::parse_c_str_or_default(scaling_method, "mad") };
        let bp_str = unsafe { binding_support::parse_c_str_or_default(boundary_policy, "extend") };
        let zwf_str = unsafe {
            binding_support::parse_c_str_or_default(zero_weight_fallback, "use_local_mean")
        };
        let cv_method_str = unsafe { binding_support::parse_c_str_or_default(cv_method, "kfold") };

        let wf = unwrap_or_return_null!(binding_support::parse_weight_function(wf_str));
        let rm = unwrap_or_return_null!(binding_support::parse_robustness_method(rm_str));
        let sm = unwrap_or_return_null!(binding_support::parse_scaling_method(sm_str));
        let bp = unwrap_or_return_null!(binding_support::parse_boundary_policy(bp_str));
        let zwf = unwrap_or_return_null!(binding_support::parse_zero_weight_fallback(zwf_str));

        let cv_fractions_vec = if !cv_fractions.is_null() && cv_fractions_len > 0 {
            let slice = unsafe { from_raw_parts(cv_fractions, cv_fractions_len as usize) };
            Some(slice.to_vec())
        } else {
            None
        };

        let config = JlLowessConfig {
            fraction,
            iterations: iterations as usize,
            delta: if delta.is_nan() { None } else { Some(delta) },
            weight_function: wf,
            robustness_method: rm,
            scaling_method: sm,
            zero_weight_fallback: zwf,
            boundary_policy: bp,
            auto_converge: if auto_converge.is_nan() {
                None
            } else {
                Some(auto_converge)
            },
            confidence_intervals: if confidence_intervals.is_nan() {
                None
            } else {
                Some(confidence_intervals)
            },
            prediction_intervals: if prediction_intervals.is_nan() {
                None
            } else {
                Some(prediction_intervals)
            },
            return_diagnostics: return_diagnostics != 0,
            return_residuals: return_residuals != 0,
            return_robustness_weights: return_robustness_weights != 0,
            cv_fractions: cv_fractions_vec,
            cv_method: cv_method_str.to_string(),
            cv_k: cv_k as usize,
            parallel: parallel != 0,
        };

        Box::into_raw(Box::new(config))
    });

    match result {
        Ok(ptr) => ptr,
        Err(_) => null_mut(),
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
            return error_result("Config pointer is null");
        }
        let config = unsafe { &*config_ptr };

        if x.is_null() || y.is_null() {
            return error_result("x and y arrays must not be null");
        }
        if n == 0 {
            return error_result("Array length must be greater than 0");
        }

        let x_slice = unsafe { from_raw_parts(x, n as usize) };
        let y_slice = unsafe { from_raw_parts(y, n as usize) };

        let mut builder = match binding_support::apply_typed_builder_options(
            LowessBuilder::<f64>::new(),
            binding_support::TypedBuilderOptionSet {
                fraction: Some(config.fraction),
                iterations: Some(config.iterations),
                delta: config.delta,
                weight_function: Some(config.weight_function),
                robustness_method: Some(config.robustness_method),
                scaling_method: Some(config.scaling_method),
                zero_weight_fallback: Some(config.zero_weight_fallback),
                boundary_policy: Some(config.boundary_policy),
                auto_converge: config.auto_converge,
                confidence_intervals: config.confidence_intervals,
                prediction_intervals: config.prediction_intervals,
                return_diagnostics: config.return_diagnostics,
                return_residuals: config.return_residuals,
                return_robustness_weights: config.return_robustness_weights,
                return_se: false,
                parallel: Some(config.parallel),
                chunk_size: None,
                overlap: None,
                merge_strategy: None,
                window_capacity: None,
                min_points: None,
                update_mode: None,
                cv_fractions: config.cv_fractions.clone(),
                cv_method: Some(config.cv_method.clone()),
                cv_k: Some(config.cv_k),
                cv_seed: None,
                custom_weights: None,
            },
        ) {
            Ok(b) => b,
            Err(e) => return error_result(&e),
        };

        // Apply per-observation case weights if provided
        if !custom_weights.is_null() && custom_weights_len > 0 {
            let cw =
                unsafe { from_raw_parts(custom_weights, custom_weights_len as usize) }.to_vec();
            builder = builder.custom_weights(cw);
        }

        let result = match builder.adapter(Batch).build() {
            Ok(m) => match m.fit(x_slice, y_slice) {
                Ok(r) => r,
                Err(e) => return error_result(&e.to_string()),
            },
            Err(e) => return error_result(&e.to_string()),
        };

        lowess_result_to_jl(result)
    });

    match result {
        Ok(res) => res,
        Err(_) => error_result("Panic in Rust library"),
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

    unsafe fn free_vec(ptr: *mut c_double, len: usize) {
        if !ptr.is_null() {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }

    free_vec(res.x, n);
    free_vec(res.y, n);
    free_vec(res.standard_errors, n);
    free_vec(res.confidence_lower, n);
    free_vec(res.confidence_upper, n);
    free_vec(res.prediction_lower, n);
    free_vec(res.prediction_upper, n);
    free_vec(res.residuals, n);
    free_vec(res.robustness_weights, n);
    free_vec(res.cv_scores, res.cv_scores_len as usize);

    if !res.error.is_null() {
        let _ = std::ffi::CString::from_raw(res.error);
    }
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
    parallel: c_int,
) -> *mut JlStreamingLowess {
    let result = catch_unwind(|| {
        let wf_str = unsafe { binding_support::parse_c_str_or_default(weight_function, "tricube") };
        let rm_str =
            unsafe { binding_support::parse_c_str_or_default(robustness_method, "bisquare") };
        let sm_str = unsafe { binding_support::parse_c_str_or_default(scaling_method, "mad") };
        let bp_str = unsafe { binding_support::parse_c_str_or_default(boundary_policy, "extend") };
        let zwf_str = unsafe {
            binding_support::parse_c_str_or_default(zero_weight_fallback, "use_local_mean")
        };

        let wf = unwrap_or_return_null!(binding_support::parse_weight_function(wf_str));
        let rm = unwrap_or_return_null!(binding_support::parse_robustness_method(rm_str));
        let sm = unwrap_or_return_null!(binding_support::parse_scaling_method(sm_str));
        let bp = unwrap_or_return_null!(binding_support::parse_boundary_policy(bp_str));
        let zwf = unwrap_or_return_null!(binding_support::parse_zero_weight_fallback(zwf_str));

        let mut builder = LowessBuilder::<f64>::new();
        builder = builder.fraction(fraction);
        builder = builder.iterations(iterations as usize);
        builder = builder.weight_function(wf);
        builder = builder.robustness_method(rm);
        builder = builder.scaling_method(sm);
        builder = builder.zero_weight_fallback(zwf);
        builder = builder.boundary_policy(bp);

        if return_diagnostics != 0 {
            builder = builder.return_diagnostics();
        }
        if return_residuals != 0 {
            builder = builder.return_residuals();
        }
        if return_robustness_weights != 0 {
            builder = builder.return_robustness_weights();
        }

        let chunk_size_usize = chunk_size as usize;
        let overlap_size = if overlap < 0 {
            binding_support::default_overlap(chunk_size_usize)
        } else {
            overlap as usize
        };

        let mut s_builder = builder.adapter(Streaming);
        s_builder = s_builder.chunk_size(chunk_size_usize);
        s_builder = s_builder.overlap(overlap_size);
        s_builder = s_builder.parallel(parallel != 0);

        if !delta.is_nan() {
            s_builder = s_builder.delta(delta);
        }
        if !auto_converge.is_nan() {
            s_builder = s_builder.auto_converge(auto_converge);
        }

        let processor = match s_builder.build() {
            Ok(p) => p,
            Err(_) => return null_mut(),
        };

        Box::into_raw(Box::new(JlStreamingLowess { inner: processor }))
    });

    match result {
        Ok(ptr) => ptr,
        Err(_) => null_mut(),
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
            return error_result("Processor pointer is null");
        }
        let processor = unsafe { &mut *ptr };

        if x.is_null() || y.is_null() {
            return error_result("x and y arrays must not be null");
        }
        if n == 0 {
            return error_result("Array length must be greater than 0");
        }

        let x_slice = unsafe { from_raw_parts(x, n as usize) };
        let y_slice = unsafe { from_raw_parts(y, n as usize) };

        match processor.inner.process_chunk(x_slice, y_slice) {
            Ok(r) => lowess_result_to_jl(r),
            Err(e) => error_result(&e.to_string()),
        }
    });

    match result {
        Ok(res) => res,
        Err(_) => error_result("Panic in Rust library"),
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
            return error_result("Processor pointer is null");
        }
        let processor = unsafe { &mut *ptr };

        match processor.inner.finalize() {
            Ok(r) => lowess_result_to_jl(r),
            Err(e) => error_result(&e.to_string()),
        }
    });

    match result {
        Ok(res) => res,
        Err(_) => error_result("Panic in Rust library"),
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
    zero_weight_fallback: *const c_char,
    parallel: c_int,
) -> *mut JlOnlineLowess {
    let result = catch_unwind(|| {
        let wf_str = unsafe { binding_support::parse_c_str_or_default(weight_function, "tricube") };
        let rm_str =
            unsafe { binding_support::parse_c_str_or_default(robustness_method, "bisquare") };
        let sm_str = unsafe { binding_support::parse_c_str_or_default(scaling_method, "mad") };
        let bp_str = unsafe { binding_support::parse_c_str_or_default(boundary_policy, "extend") };
        let zwf_str = unsafe {
            binding_support::parse_c_str_or_default(zero_weight_fallback, "use_local_mean")
        };
        let um_str = unsafe { binding_support::parse_c_str_or_default(update_mode, "full") };

        let wf = unwrap_or_return_null!(binding_support::parse_weight_function(wf_str));
        let rm = unwrap_or_return_null!(binding_support::parse_robustness_method(rm_str));
        let sm = unwrap_or_return_null!(binding_support::parse_scaling_method(sm_str));
        let bp = unwrap_or_return_null!(binding_support::parse_boundary_policy(bp_str));
        let zwf = unwrap_or_return_null!(binding_support::parse_zero_weight_fallback(zwf_str));
        let um = unwrap_or_return_null!(binding_support::parse_update_mode(um_str));

        let mut builder = LowessBuilder::<f64>::new();
        builder = builder.fraction(fraction);
        builder = builder.iterations(iterations as usize);
        builder = builder.weight_function(wf);
        builder = builder.robustness_method(rm);
        builder = builder.scaling_method(sm);
        builder = builder.zero_weight_fallback(zwf);
        builder = builder.boundary_policy(bp);

        let mut o_builder = builder.adapter(Online);
        o_builder = o_builder.window_capacity(window_capacity as usize);
        o_builder = o_builder.min_points(min_points as usize);
        o_builder = o_builder.update_mode(um);
        o_builder = o_builder.parallel(parallel != 0);

        if !delta.is_nan() {
            o_builder = o_builder.delta(delta);
        }
        if !auto_converge.is_nan() {
            o_builder = o_builder.auto_converge(auto_converge);
        }
        if return_robustness_weights != 0 {
            o_builder = o_builder.return_robustness_weights(true);
        }

        let processor = match o_builder.build() {
            Ok(p) => p,
            Err(_) => return null_mut(),
        };

        Box::into_raw(Box::new(JlOnlineLowess { inner: processor }))
    });

    match result {
        Ok(ptr) => ptr,
        Err(_) => null_mut(),
    }
}

/// Add a single point to the online processor and return the smoothed value.
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
) -> c_double {
    let result = catch_unwind(|| {
        if ptr.is_null() {
            return f64::NAN;
        }
        let processor = unsafe { &mut *ptr };

        match processor.inner.add_point(x, y) {
            Ok(Some(o)) => o.smoothed,
            Ok(None) => f64::NAN,
            Err(_) => f64::NAN,
        }
    });

    result.unwrap_or(f64::NAN)
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
            Err(_) => return null_mut(),
        }
    };
}
