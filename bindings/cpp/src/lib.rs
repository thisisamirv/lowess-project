//! C/C++ bindings for fastLowess.
//!
//! Provides C access to the fastLowess Rust library via C FFI.
//! A C++ wrapper header (fastlowess.hpp) provides idiomatic C++ usage.

#![allow(non_snake_case)]
#![allow(unsafe_op_in_unsafe_fn)]

use std::ffi::CStr;
use std::os::raw::{c_char, c_double, c_int, c_ulong};
use std::ptr;

use fastLowess::internals::api::{
    BoundaryPolicy, MergeStrategy, RobustnessMethod, ScalingMethod, UpdateMode, WeightFunction,
    ZeroWeightFallback,
};
use fastLowess::prelude::{
    Batch, KFold, LOOCV, Lowess as LowessBuilder, LowessResult, MAD, MAR, Online, Streaming,
};

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

/// Parse weight function from string.
fn parse_weight_function(name: &str) -> Result<WeightFunction, String> {
    match name.to_lowercase().as_str() {
        "tricube" => Ok(WeightFunction::Tricube),
        "epanechnikov" => Ok(WeightFunction::Epanechnikov),
        "gaussian" => Ok(WeightFunction::Gaussian),
        "uniform" | "boxcar" => Ok(WeightFunction::Uniform),
        "biweight" | "bisquare" => Ok(WeightFunction::Biweight),
        "triangle" | "triangular" => Ok(WeightFunction::Triangle),
        "cosine" => Ok(WeightFunction::Cosine),
        _ => Err(format!(
            "Unknown weight function: {}. Valid: tricube, epanechnikov, gaussian, uniform, biweight, triangle, cosine",
            name
        )),
    }
}

/// Parse robustness method from string.
fn parse_robustness_method(name: &str) -> Result<RobustnessMethod, String> {
    match name.to_lowercase().as_str() {
        "bisquare" | "biweight" => Ok(RobustnessMethod::Bisquare),
        "huber" => Ok(RobustnessMethod::Huber),
        "talwar" => Ok(RobustnessMethod::Talwar),
        _ => Err(format!(
            "Unknown robustness method: {}. Valid: bisquare, huber, talwar",
            name
        )),
    }
}

/// Parse zero weight fallback from string.
fn parse_zero_weight_fallback(name: &str) -> Result<ZeroWeightFallback, String> {
    match name.to_lowercase().as_str() {
        "use_local_mean" | "local_mean" | "mean" => Ok(ZeroWeightFallback::UseLocalMean),
        "return_original" | "original" => Ok(ZeroWeightFallback::ReturnOriginal),
        "return_none" | "none" | "nan" => Ok(ZeroWeightFallback::ReturnNone),
        _ => Err(format!(
            "Unknown zero weight fallback: {}. Valid: use_local_mean, return_original, return_none",
            name
        )),
    }
}

/// Parse boundary policy from string.
fn parse_boundary_policy(name: &str) -> Result<BoundaryPolicy, String> {
    match name.to_lowercase().as_str() {
        "extend" | "pad" => Ok(BoundaryPolicy::Extend),
        "reflect" | "mirror" => Ok(BoundaryPolicy::Reflect),
        "zero" | "none" => Ok(BoundaryPolicy::Zero),
        "noboundary" => Ok(BoundaryPolicy::NoBoundary),
        _ => Err(format!(
            "Unknown boundary policy: {}. Valid: extend, reflect, zero, noboundary",
            name
        )),
    }
}

/// Parse scaling method from string.
fn parse_scaling_method(name: &str) -> Result<ScalingMethod, String> {
    match name.to_lowercase().as_str() {
        "mad" => Ok(MAD),
        "mar" => Ok(MAR),
        _ => Err(format!("Unknown scaling method: {}. Valid: mad, mar", name)),
    }
}

/// Parse update mode from string.
fn parse_update_mode(name: &str) -> Result<UpdateMode, String> {
    match name.to_lowercase().as_str() {
        "full" | "resmooth" => Ok(UpdateMode::Full),
        "incremental" | "single" => Ok(UpdateMode::Incremental),
        _ => Err(format!(
            "Unknown update mode: {}. Valid: full, incremental",
            name
        )),
    }
}

/// Parse merge strategy from string.
fn parse_merge_strategy(name: &str) -> Result<MergeStrategy, String> {
    match name.to_lowercase().as_str() {
        "average" | "mean" => Ok(MergeStrategy::Average),
        "weighted" | "weighted_average" | "weightedaverage" => Ok(MergeStrategy::WeightedAverage),
        "first" | "take_first" | "takefirst" | "left" => Ok(MergeStrategy::TakeFirst),
        "last" | "take_last" | "takelast" | "right" => Ok(MergeStrategy::TakeLast),
        _ => Err(format!(
            "Unknown merge strategy: {}. Valid: average, weighted, first, last",
            name
        )),
    }
}

/// Convert LowessResult to CppLowessResult.
fn lowess_result_to_cpp(result: LowessResult<f64>) -> CppLowessResult {
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

/// LOWESS smoothing with batch adapter.
///
/// # Safety
/// All pointer arguments must be valid and point to arrays of the specified length.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_lowess_smooth(
    x: *const c_double,
    y: *const c_double,
    n: c_ulong,
    fraction: c_double,
    iterations: c_int,
    delta: c_double, // Use NaN to auto-calculate
    weight_function: *const c_char,
    robustness_method: *const c_char,
    scaling_method: *const c_char,
    boundary_policy: *const c_char,
    confidence_intervals: c_double, // Use NaN to disable
    prediction_intervals: c_double, // Use NaN to disable
    return_diagnostics: c_int,
    return_residuals: c_int,
    return_robustness_weights: c_int,
    zero_weight_fallback: *const c_char,
    auto_converge: c_double, // Use NaN to disable
    cv_fractions: *const c_double,
    cv_fractions_len: c_ulong,
    cv_method: *const c_char,
    cv_k: c_int,
    parallel: c_int,
) -> CppLowessResult {
    // Validate input pointers
    if x.is_null() || y.is_null() {
        return error_result("x and y arrays must not be null");
    }
    if n == 0 {
        return error_result("Array length must be greater than 0");
    }

    // Convert input arrays to slices
    let x_slice = std::slice::from_raw_parts(x, n as usize);
    let y_slice = std::slice::from_raw_parts(y, n as usize);

    // Parse string parameters
    let wf_str = parse_c_str(weight_function, "tricube");
    let rm_str = parse_c_str(robustness_method, "bisquare");
    let sm_str = parse_c_str(scaling_method, "mad");
    let bp_str = parse_c_str(boundary_policy, "extend");
    let zwf_str = parse_c_str(zero_weight_fallback, "use_local_mean");
    let cv_method_str = parse_c_str(cv_method, "kfold");

    let wf = match parse_weight_function(wf_str) {
        Ok(v) => v,
        Err(e) => return error_result(&e),
    };
    let rm = match parse_robustness_method(rm_str) {
        Ok(v) => v,
        Err(e) => return error_result(&e),
    };
    let sm = match parse_scaling_method(sm_str) {
        Ok(v) => v,
        Err(e) => return error_result(&e),
    };
    let bp = match parse_boundary_policy(bp_str) {
        Ok(v) => v,
        Err(e) => return error_result(&e),
    };
    let zwf = match parse_zero_weight_fallback(zwf_str) {
        Ok(v) => v,
        Err(e) => return error_result(&e),
    };

    // Build the LOWESS model
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

    // Cross-validation
    if !cv_fractions.is_null() && cv_fractions_len > 0 {
        let fractions = std::slice::from_raw_parts(cv_fractions, cv_fractions_len as usize);
        let fractions_vec: Vec<f64> = fractions.to_vec();

        match cv_method_str.to_lowercase().as_str() {
            "simple" | "loo" | "loocv" | "leave_one_out" => {
                builder = builder.cross_validate(LOOCV(&fractions_vec));
            }
            "kfold" | "k_fold" | "k-fold" => {
                builder = builder.cross_validate(KFold(cv_k as usize, &fractions_vec));
            }
            _ => {
                return error_result(&format!(
                    "Unknown CV method: {}. Valid: loocv, kfold",
                    cv_method_str
                ));
            }
        }
    }

    // Build and fit
    let model = match builder.adapter(Batch).build() {
        Ok(m) => m,
        Err(e) => return error_result(&e.to_string()),
    };

    let result = match model.fit(x_slice, y_slice) {
        Ok(r) => r,
        Err(e) => return error_result(&e.to_string()),
    };

    lowess_result_to_cpp(result)
}

/// Streaming LOWESS for large datasets.
///
/// # Safety
/// All pointer arguments must be valid and point to arrays of the specified length.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_lowess_streaming(
    x: *const c_double,
    y: *const c_double,
    n: c_ulong,
    fraction: c_double,
    chunk_size: c_int,
    overlap: c_int, // Use -1 for auto
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
) -> CppLowessResult {
    if x.is_null() || y.is_null() {
        return error_result("x and y arrays must not be null");
    }
    if n == 0 {
        return error_result("Array length must be greater than 0");
    }

    let x_slice = std::slice::from_raw_parts(x, n as usize);
    let y_slice = std::slice::from_raw_parts(y, n as usize);

    let chunk_size = chunk_size as usize;
    let overlap_size = if overlap < 0 {
        let default = chunk_size / 10;
        default.min(chunk_size.saturating_sub(10)).max(1)
    } else {
        overlap as usize
    };

    let wf_str = parse_c_str(weight_function, "tricube");
    let rm_str = parse_c_str(robustness_method, "bisquare");
    let sm_str = parse_c_str(scaling_method, "mad");
    let bp_str = parse_c_str(boundary_policy, "extend");
    let zwf_str = parse_c_str(zero_weight_fallback, "use_local_mean");
    let ms_str = parse_c_str(merge_strategy, "weighted");

    let wf = match parse_weight_function(wf_str) {
        Ok(v) => v,
        Err(e) => return error_result(&e),
    };
    let rm = match parse_robustness_method(rm_str) {
        Ok(v) => v,
        Err(e) => return error_result(&e),
    };
    let sm = match parse_scaling_method(sm_str) {
        Ok(v) => v,
        Err(e) => return error_result(&e),
    };
    let bp = match parse_boundary_policy(bp_str) {
        Ok(v) => v,
        Err(e) => return error_result(&e),
    };
    let zwf = match parse_zero_weight_fallback(zwf_str) {
        Ok(v) => v,
        Err(e) => return error_result(&e),
    };
    let ms = match parse_merge_strategy(ms_str) {
        Ok(v) => v,
        Err(e) => return error_result(&e),
    };

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

    let mut builder = builder.adapter(Streaming);
    builder = builder.chunk_size(chunk_size);
    builder = builder.overlap(overlap_size);
    builder = builder.merge_strategy(ms);
    builder = builder.parallel(parallel != 0);

    if !delta.is_nan() {
        builder = builder.delta(delta);
    }
    if !auto_converge.is_nan() {
        builder = builder.auto_converge(auto_converge);
    }

    let mut processor = match builder.build() {
        Ok(p) => p,
        Err(e) => return error_result(&e.to_string()),
    };

    let chunk_result = match processor.process_chunk(x_slice, y_slice) {
        Ok(r) => r,
        Err(e) => return error_result(&e.to_string()),
    };

    let final_result = match processor.finalize() {
        Ok(r) => r,
        Err(e) => return error_result(&e.to_string()),
    };

    // Combine results
    let mut combined_x = chunk_result.x;
    let mut combined_y = chunk_result.y;
    let mut combined_se = chunk_result.standard_errors;
    let mut combined_cl = chunk_result.confidence_lower;
    let mut combined_cu = chunk_result.confidence_upper;
    let mut combined_pl = chunk_result.prediction_lower;
    let mut combined_pu = chunk_result.prediction_upper;
    let mut combined_res = chunk_result.residuals;
    let mut combined_rw = chunk_result.robustness_weights;

    combined_x.extend(final_result.x);
    combined_y.extend(final_result.y);

    if let (Some(mut s), Some(f)) = (combined_se.take(), final_result.standard_errors) {
        s.extend(f);
        combined_se = Some(s);
    }
    if let (Some(mut s), Some(f)) = (combined_cl.take(), final_result.confidence_lower) {
        s.extend(f);
        combined_cl = Some(s);
    }
    if let (Some(mut s), Some(f)) = (combined_cu.take(), final_result.confidence_upper) {
        s.extend(f);
        combined_cu = Some(s);
    }
    if let (Some(mut s), Some(f)) = (combined_pl.take(), final_result.prediction_lower) {
        s.extend(f);
        combined_pl = Some(s);
    }
    if let (Some(mut s), Some(f)) = (combined_pu.take(), final_result.prediction_upper) {
        s.extend(f);
        combined_pu = Some(s);
    }
    if let (Some(mut s), Some(f)) = (combined_res.take(), final_result.residuals) {
        s.extend(f);
        combined_res = Some(s);
    }
    if let (Some(mut s), Some(f)) = (combined_rw.take(), final_result.robustness_weights) {
        s.extend(f);
        combined_rw = Some(s);
    }

    let result = LowessResult {
        x: combined_x,
        y: combined_y,
        standard_errors: combined_se,
        confidence_lower: combined_cl,
        confidence_upper: combined_cu,
        prediction_lower: combined_pl,
        prediction_upper: combined_pu,
        residuals: combined_res,
        robustness_weights: combined_rw,
        diagnostics: final_result.diagnostics,
        iterations_used: chunk_result.iterations_used,
        fraction_used: chunk_result.fraction_used,
        cv_scores: None,
    };

    lowess_result_to_cpp(result)
}

/// Online LOWESS with sliding window.
///
/// # Safety
/// All pointer arguments must be valid and point to arrays of the specified length.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_lowess_online(
    x: *const c_double,
    y: *const c_double,
    n: c_ulong,
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
) -> CppLowessResult {
    if x.is_null() || y.is_null() {
        return error_result("x and y arrays must not be null");
    }
    if n == 0 {
        return error_result("Array length must be greater than 0");
    }

    let x_slice = std::slice::from_raw_parts(x, n as usize);
    let y_slice = std::slice::from_raw_parts(y, n as usize);

    let wf_str = parse_c_str(weight_function, "tricube");
    let rm_str = parse_c_str(robustness_method, "bisquare");
    let sm_str = parse_c_str(scaling_method, "mad");
    let bp_str = parse_c_str(boundary_policy, "extend");
    let zwf_str = parse_c_str(zero_weight_fallback, "use_local_mean");
    let um_str = parse_c_str(update_mode, "full");

    let wf = match parse_weight_function(wf_str) {
        Ok(v) => v,
        Err(e) => return error_result(&e),
    };
    let rm = match parse_robustness_method(rm_str) {
        Ok(v) => v,
        Err(e) => return error_result(&e),
    };
    let sm = match parse_scaling_method(sm_str) {
        Ok(v) => v,
        Err(e) => return error_result(&e),
    };
    let bp = match parse_boundary_policy(bp_str) {
        Ok(v) => v,
        Err(e) => return error_result(&e),
    };
    let zwf = match parse_zero_weight_fallback(zwf_str) {
        Ok(v) => v,
        Err(e) => return error_result(&e),
    };
    let um = match parse_update_mode(um_str) {
        Ok(v) => v,
        Err(e) => return error_result(&e),
    };

    let mut builder = LowessBuilder::<f64>::new();
    builder = builder.fraction(fraction);
    builder = builder.iterations(iterations as usize);
    builder = builder.weight_function(wf);
    builder = builder.robustness_method(rm);
    builder = builder.scaling_method(sm);
    builder = builder.zero_weight_fallback(zwf);
    builder = builder.boundary_policy(bp);

    let mut builder = builder.adapter(Online);
    builder = builder.window_capacity(window_capacity as usize);
    builder = builder.min_points(min_points as usize);
    builder = builder.update_mode(um);
    builder = builder.parallel(parallel != 0);

    if !delta.is_nan() {
        builder = builder.delta(delta);
    }
    if !auto_converge.is_nan() {
        builder = builder.auto_converge(auto_converge);
    }
    if return_robustness_weights != 0 {
        builder = builder.return_robustness_weights(true);
    }

    let mut processor = match builder.build() {
        Ok(p) => p,
        Err(e) => return error_result(&e.to_string()),
    };

    let outputs = match processor.add_points(x_slice, y_slice) {
        Ok(o) => o,
        Err(e) => return error_result(&e.to_string()),
    };

    let smoothed: Vec<f64> = outputs
        .into_iter()
        .zip(y_slice.iter())
        .map(|(opt, &original_y)| opt.map_or(original_y, |o| o.smoothed))
        .collect();

    let result = LowessResult {
        x: x_slice.to_vec(),
        y: smoothed,
        standard_errors: None,
        confidence_lower: None,
        confidence_upper: None,
        prediction_lower: None,
        prediction_upper: None,
        residuals: None,
        robustness_weights: None,
        diagnostics: None,
        iterations_used: Some(iterations as usize),
        fraction_used: fraction,
        cv_scores: None,
    };

    lowess_result_to_cpp(result)
}

/// Free a CppLowessResult allocated by Rust.
///
/// # Safety
/// The result pointer must have been returned by one of the cpp_lowess_* functions.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_lowess_free_result(result: *mut CppLowessResult) {
    if result.is_null() {
        return;
    }

    let r = &mut *result;
    let n = r.n as usize;

    // Free arrays using ptr::slice_from_raw_parts_mut
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

    // Free error string
    if !r.error.is_null() {
        let _ = std::ffi::CString::from_raw(r.error);
    }
}
