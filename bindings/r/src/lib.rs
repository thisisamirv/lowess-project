//! R bindings for fastLowess.
//!
//! Provides R access to the fastLowess Rust library via extendr.

#![allow(non_snake_case)]

use extendr_api::prelude::*;

use fastLowess::internals::api::{
    BoundaryPolicy, RobustnessMethod,
    ScalingMethod::{self, MAD, MAR},
    UpdateMode, WeightFunction, ZeroWeightFallback,
};
use fastLowess::prelude::{
    Batch, KFold, LOOCV, Lowess as LowessBuilder, LowessResult, Online, Streaming,
};

// ============================================================================
// Helper Functions
// ============================================================================

/// Parse weight function from string
fn parse_weight_function(name: &str) -> Result<WeightFunction> {
    match name.to_lowercase().as_str() {
        "tricube" => Ok(WeightFunction::Tricube),
        "epanechnikov" => Ok(WeightFunction::Epanechnikov),
        "gaussian" => Ok(WeightFunction::Gaussian),
        "uniform" | "boxcar" => Ok(WeightFunction::Uniform),
        "biweight" | "bisquare" => Ok(WeightFunction::Biweight),
        "triangle" | "triangular" => Ok(WeightFunction::Triangle),
        "cosine" => Ok(WeightFunction::Cosine),
        _ => Err(Error::Other(format!(
            "Unknown weight function: {}. Valid options: tricube, epanechnikov, gaussian, uniform, biweight, triangle, cosine",
            name
        ))),
    }
}

/// Parse robustness method from string
fn parse_robustness_method(name: &str) -> Result<RobustnessMethod> {
    match name.to_lowercase().as_str() {
        "bisquare" | "biweight" => Ok(RobustnessMethod::Bisquare),
        "huber" => Ok(RobustnessMethod::Huber),
        "talwar" => Ok(RobustnessMethod::Talwar),
        _ => Err(Error::Other(format!(
            "Unknown robustness method: {}. Valid options: bisquare, huber, talwar",
            name
        ))),
    }
}

/// Parse zero weight fallback from string
fn parse_zero_weight_fallback(name: &str) -> Result<ZeroWeightFallback> {
    match name.to_lowercase().as_str() {
        "use_local_mean" | "local_mean" | "mean" => Ok(ZeroWeightFallback::UseLocalMean),
        "return_original" | "original" => Ok(ZeroWeightFallback::ReturnOriginal),
        "return_none" | "none" | "nan" => Ok(ZeroWeightFallback::ReturnNone),
        _ => Err(Error::Other(format!(
            "Unknown zero weight fallback: {}. Valid options: use_local_mean, return_original, return_none",
            name
        ))),
    }
}

/// Parse boundary policy from string
fn parse_boundary_policy(name: &str) -> Result<BoundaryPolicy> {
    match name.to_lowercase().as_str() {
        "extend" | "pad" => Ok(BoundaryPolicy::Extend),
        "reflect" | "mirror" => Ok(BoundaryPolicy::Reflect),
        "zero" | "none" => Ok(BoundaryPolicy::Zero),
        "noboundary" => Ok(BoundaryPolicy::NoBoundary),
        _ => Err(Error::Other(format!(
            "Unknown boundary policy: {}. Valid options: extend, reflect, zero, noboundary",
            name
        ))),
    }
}

/// Parse scaling method from string
fn parse_scaling_method(name: &str) -> Result<ScalingMethod> {
    match name.to_lowercase().as_str() {
        "mad" => Ok(MAD),
        "mar" => Ok(MAR),
        _ => Err(Error::Other(format!(
            "Unknown scaling method: {}. Valid options: mad, mar",
            name
        ))),
    }
}

/// Parse update mode from string
fn parse_update_mode(name: &str) -> Result<UpdateMode> {
    match name.to_lowercase().as_str() {
        "full" | "resmooth" => Ok(UpdateMode::Full),
        "incremental" | "single" => Ok(UpdateMode::Incremental),
        _ => Err(Error::Other(format!(
            "Unknown update mode: {}. Valid options: full, incremental",
            name
        ))),
    }
}

// ============================================================================
// R Functions
// ============================================================================

/// LOWESS smoothing with the batch adapter.
///
/// This is the primary interface for LOWESS smoothing. Processes the entire
/// dataset in memory with optional parallel execution.
///
/// @param x Numeric vector of independent variable values.
/// @param y Numeric vector of dependent variable values.
/// @param fraction Smoothing fraction (default: 0.67).
/// @param iterations Number of robustness iterations (default: 3).
/// @param delta Interpolation optimization threshold (NULL to auto-calculate).
/// @param weight_function Kernel function: "tricube", "epanechnikov", "gaussian", "uniform", "biweight", "triangle".
/// @param robustness_method Robustness method: "bisquare", "huber", "talwar".
/// @param scaling_method Scaling method for robustness: "mad", "mar" (default: "mad").
/// @param boundary_policy Handling of edge effects: "extend", "reflect", "zero", "noboundary" (default: "extend").
/// @param confidence_intervals Confidence level for confidence intervals (e.g., 0.95), NULL to disable.
/// @param prediction_intervals Confidence level for prediction intervals (e.g., 0.95), NULL to disable.
/// @param return_diagnostics Whether to compute RMSE, MAE, R², etc.
/// @param return_residuals Whether to include residuals in output.
/// @param return_robustness_weights Whether to include robustness weights in output.
/// @param zero_weight_fallback Fallback when all weights are zero: "use_local_mean", "return_original", "return_none".
/// @param auto_converge Tolerance for auto-convergence (NULL to disable).
/// @param cv_fractions Numeric vector of fractions to test for cross-validation (NULL to disable).
/// @param cv_method CV method: "loocv" (leave-one-out) or "kfold". Default: "kfold".
/// @param cv_k Number of folds for k-fold CV (default: 5).
/// @param parallel Enable parallel execution (default: TRUE).
/// @return A list with smoothed values and optional outputs.
/// @export
#[extendr]
#[allow(clippy::too_many_arguments)]
fn fastlowess(
    x: &[f64],
    y: &[f64],
    fraction: f64,
    iterations: i32,
    delta: Nullable<f64>,
    weight_function: &str,
    robustness_method: &str,
    scaling_method: &str,
    boundary_policy: &str,
    confidence_intervals: Nullable<f64>,
    prediction_intervals: Nullable<f64>,
    return_diagnostics: bool,
    return_residuals: bool,
    return_robustness_weights: bool,
    zero_weight_fallback: &str,
    auto_converge: Nullable<f64>,
    cv_fractions: Nullable<Vec<f64>>,
    cv_method: &str,
    cv_k: i32,
    parallel: bool,
) -> Result<List> {
    let wf = parse_weight_function(weight_function)?;
    let rm = parse_robustness_method(robustness_method)?;
    let sm = parse_scaling_method(scaling_method)?;
    let zwf = parse_zero_weight_fallback(zero_weight_fallback)?;
    let bp = parse_boundary_policy(boundary_policy)?;

    let mut builder = LowessBuilder::<f64>::new();
    builder = builder.fraction(fraction);
    builder = builder.iterations(iterations as usize);
    builder = builder.weight_function(wf);
    builder = builder.robustness_method(rm);
    builder = builder.scaling_method(sm);
    builder = builder.zero_weight_fallback(zwf);
    builder = builder.boundary_policy(bp);

    if let NotNull(d) = delta {
        builder = builder.delta(d);
    }

    if let NotNull(cl) = confidence_intervals {
        builder = builder.confidence_intervals(cl);
    }

    if let NotNull(pl) = prediction_intervals {
        builder = builder.prediction_intervals(pl);
    }

    if return_diagnostics {
        builder = builder.return_diagnostics();
    }

    if return_residuals {
        builder = builder.return_residuals();
    }

    if return_robustness_weights {
        builder = builder.return_robustness_weights();
    }

    if let NotNull(tol) = auto_converge {
        builder = builder.auto_converge(tol);
    }

    // Cross-validation if fractions are provided
    if let NotNull(fractions) = cv_fractions {
        match cv_method.to_lowercase().as_str() {
            "simple" | "loo" | "loocv" | "leave_one_out" => {
                builder = builder.cross_validate(LOOCV(&fractions));
            }
            "kfold" | "k_fold" | "k-fold" => {
                builder = builder.cross_validate(KFold(cv_k as usize, &fractions));
            }
            _ => {
                return Err(Error::Other(format!(
                    "Unknown CV method: {}. Valid options: loocv, kfold",
                    cv_method
                )));
            }
        }
    }

    let result = builder
        .adapter(Batch)
        .parallel(parallel)
        .build()
        .map_err(|e| Error::Other(e.to_string()))?
        .fit(x, y)
        .map_err(|e| Error::Other(e.to_string()))?;

    lowess_result_to_list(result)
}

/// Streaming LOWESS for large datasets.
///
/// Processes data in chunks to maintain constant memory usage.
///
/// @param x Numeric vector of independent variable values.
/// @param y Numeric vector of dependent variable values.
/// @param fraction Smoothing fraction (default: 0.3).
/// @param chunk_size Size of each processing chunk (default: 5000).
/// @param overlap Overlap between chunks (default: 10% of chunk_size).
/// @param iterations Number of robustness iterations (default: 3).
/// @param delta Interpolation optimization threshold (NULL to auto-calculate).
/// @param weight_function Kernel function: "tricube", "epanechnikov", "gaussian", "uniform", "biweight", "triangle".
/// @param robustness_method Robustness method: "bisquare", "huber", "talwar".
/// @param scaling_method Scaling method for robustness: "mad", "mar" (default: "mad").
/// @param boundary_policy Handling of edge effects: "extend", "reflect", "zero", "noboundary" (default: "extend").
/// @param auto_converge Tolerance for auto-convergence (NULL to disable).
/// @param return_diagnostics Whether to compute RMSE, MAE, R², etc.
/// @param return_robustness_weights Whether to include robustness weights in output.
/// @param parallel Enable parallel execution (default: TRUE).
/// @return A list with smoothed values.
/// @export
#[extendr]
#[allow(clippy::too_many_arguments)]
fn fastlowess_streaming(
    x: &[f64],
    y: &[f64],
    fraction: f64,
    chunk_size: i32,
    overlap: Nullable<i32>,
    iterations: i32,
    delta: Nullable<f64>,
    weight_function: &str,
    robustness_method: &str,
    scaling_method: &str,
    boundary_policy: &str,
    auto_converge: Nullable<f64>,
    return_diagnostics: bool,
    return_robustness_weights: bool,
    parallel: bool,
) -> Result<List> {
    let chunk_size = chunk_size as usize;

    // Default overlap to 10% of chunk_size, capped at chunk_size - 10
    let overlap_size = match overlap {
        NotNull(o) => o as usize,
        Null => {
            let default = chunk_size / 10;
            default.min(chunk_size.saturating_sub(10)).max(1)
        }
    };

    let wf = parse_weight_function(weight_function)?;
    let rm = parse_robustness_method(robustness_method)?;
    let sm = parse_scaling_method(scaling_method)?;
    let bp = parse_boundary_policy(boundary_policy)?;

    let mut builder = LowessBuilder::<f64>::new();
    builder = builder.fraction(fraction);
    builder = builder.iterations(iterations as usize);
    builder = builder.weight_function(wf);
    builder = builder.robustness_method(rm);
    builder = builder.scaling_method(sm);
    builder = builder.boundary_policy(bp);

    let mut builder = builder.adapter(Streaming);
    builder = builder.chunk_size(chunk_size);
    builder = builder.overlap(overlap_size);
    builder = builder.parallel(parallel);

    if let NotNull(d) = delta {
        builder = builder.delta(d);
    }
    if let NotNull(tol) = auto_converge {
        builder = builder.auto_converge(tol);
    }
    if return_diagnostics {
        builder = builder.return_diagnostics(true);
    }
    if return_robustness_weights {
        builder = builder.return_robustness_weights(true);
    }

    let mut processor = builder.build().map_err(|e| Error::Other(e.to_string()))?;

    // Process the data as a single chunk
    let chunk_result = processor
        .process_chunk(x, y)
        .map_err(|e| Error::Other(e.to_string()))?;

    // Finalize to get remaining buffered overlap data
    let final_result = processor
        .finalize()
        .map_err(|e| Error::Other(e.to_string()))?;

    // Combine results from process_chunk and finalize
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

    // Create combined result
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
        diagnostics: final_result.diagnostics, // diagnostics are cumulative in final
        iterations_used: chunk_result.iterations_used,
        fraction_used: chunk_result.fraction_used,
        cv_scores: None,
    };

    lowess_result_to_list(result)
}

/// Online LOWESS with sliding window.
///
/// Maintains a sliding window for incremental updates.
///
/// @param x Numeric vector of independent variable values.
/// @param y Numeric vector of dependent variable values.
/// @param fraction Smoothing fraction (default: 0.2).
/// @param window_capacity Maximum points to retain in window (default: 100).
/// @param min_points Minimum points before smoothing starts (default: 2).
/// @param iterations Number of robustness iterations (default: 3).
/// @param delta Interpolation optimization threshold (NULL to auto-calculate).
/// @param weight_function Kernel function: "tricube", "epanechnikov", "gaussian", "uniform", "biweight", "triangle".
/// @param robustness_method Robustness method: "bisquare", "huber", "talwar".
/// @param scaling_method Scaling method for robustness: "mad", "mar" (default: "mad").
/// @param boundary_policy Handling of edge effects: "extend", "reflect", "zero", "noboundary" (default: "extend").
/// @param update_mode Update strategy: "full" or "incremental" (default: "full").
/// @param auto_converge Tolerance for auto-convergence (NULL to disable).
/// @param return_robustness_weights Whether to include robustness weights in output.
/// @param parallel Enable parallel execution (default: FALSE for online).
/// @return A list with smoothed values.
/// @export
#[extendr]
#[allow(clippy::too_many_arguments)]
fn fastlowess_online(
    x: &[f64],
    y: &[f64],
    fraction: f64,
    window_capacity: i32,
    min_points: i32,
    iterations: i32,
    delta: Nullable<f64>,
    weight_function: &str,
    robustness_method: &str,
    scaling_method: &str,
    boundary_policy: &str,
    update_mode: &str,
    auto_converge: Nullable<f64>,
    return_robustness_weights: bool,
    parallel: bool,
) -> Result<List> {
    let wf = parse_weight_function(weight_function)?;
    let rm = parse_robustness_method(robustness_method)?;
    let sm = parse_scaling_method(scaling_method)?;
    let bp = parse_boundary_policy(boundary_policy)?;
    let um = parse_update_mode(update_mode)?;

    let mut builder = LowessBuilder::<f64>::new();
    builder = builder.fraction(fraction);
    builder = builder.iterations(iterations as usize);
    builder = builder.weight_function(wf);
    builder = builder.robustness_method(rm);
    builder = builder.scaling_method(sm);
    builder = builder.boundary_policy(bp);

    let mut builder = builder.adapter(Online);
    builder = builder.window_capacity(window_capacity as usize);
    builder = builder.min_points(min_points as usize);
    builder = builder.update_mode(um);
    builder = builder.parallel(parallel);

    if let NotNull(d) = delta {
        builder = builder.delta(d);
    }
    if let NotNull(tol) = auto_converge {
        builder = builder.auto_converge(tol);
    }
    if return_robustness_weights {
        builder = builder.return_robustness_weights(true);
    }

    let mut processor = builder.build().map_err(|e| Error::Other(e.to_string()))?;

    let outputs = processor
        .add_points(x, y)
        .map_err(|e| Error::Other(e.to_string()))?;

    // Extract smoothed values (use original y for points that haven't accumulated enough data)
    let smoothed: Vec<f64> = outputs
        .into_iter()
        .zip(y.iter())
        .map(|(opt, &original_y)| opt.map_or(original_y, |o| o.smoothed))
        .collect();

    // Create result
    let result = LowessResult {
        x: x.to_vec(),
        y: smoothed,
        standard_errors: None,
        confidence_lower: None,
        confidence_upper: None,
        prediction_lower: None,
        prediction_upper: None,
        residuals: None,
        robustness_weights: if return_robustness_weights {
            // Extract weights from the final state of the processor
            // In online mode, we might only want the latest weights or all historical ones
            // For now, let's keep it consistent with LowessResult if possible
            None
        } else {
            None
        },
        diagnostics: None,
        iterations_used: Some(iterations as usize),
        fraction_used: fraction,
        cv_scores: None,
    };

    lowess_result_to_list(result)
}

// ============================================================================
// Helper: Convert LowessResult to R List
// ============================================================================

fn lowess_result_to_list(result: LowessResult<f64>) -> Result<List> {
    let mut list_items: Vec<(&str, Robj)> = vec![
        ("x", result.x.into_robj()),
        ("y", result.y.into_robj()),
        ("fraction_used", result.fraction_used.into_robj()),
    ];

    if let Some(se) = result.standard_errors {
        list_items.push(("standard_errors", se.into_robj()));
    }

    if let Some(cl) = result.confidence_lower {
        list_items.push(("confidence_lower", cl.into_robj()));
    }

    if let Some(cu) = result.confidence_upper {
        list_items.push(("confidence_upper", cu.into_robj()));
    }

    if let Some(pl) = result.prediction_lower {
        list_items.push(("prediction_lower", pl.into_robj()));
    }

    if let Some(pu) = result.prediction_upper {
        list_items.push(("prediction_upper", pu.into_robj()));
    }

    if let Some(res) = result.residuals {
        list_items.push(("residuals", res.into_robj()));
    }

    if let Some(rw) = result.robustness_weights {
        list_items.push(("robustness_weights", rw.into_robj()));
    }

    if let Some(iters) = result.iterations_used {
        list_items.push(("iterations_used", (iters as i32).into_robj()));
    }

    if let Some(cv) = result.cv_scores {
        list_items.push(("cv_scores", cv.into_robj()));
    }

    if let Some(diag) = result.diagnostics {
        let diag_list = list!(
            rmse = diag.rmse,
            mae = diag.mae,
            r_squared = diag.r_squared,
            aic = diag.aic.unwrap_or(f64::NAN),
            aicc = diag.aicc.unwrap_or(f64::NAN),
            effective_df = diag.effective_df.unwrap_or(f64::NAN),
            residual_sd = diag.residual_sd
        );
        list_items.push(("diagnostics", diag_list.into_robj()));
    }

    // Build the list manually
    let names: Vec<&str> = list_items.iter().map(|(k, _)| *k).collect();
    let values: Vec<Robj> = list_items.into_iter().map(|(_, v)| v).collect();
    let result_list = List::from_names_and_values(names, values)?;

    Ok(result_list)
}

// ============================================================================
// Module Registration
// ============================================================================

extendr_module! {
    mod rfastlowess;
    fn fastlowess;
    fn fastlowess_streaming;
    fn fastlowess_online;
}
