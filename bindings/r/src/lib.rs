//! R bindings for fastLowess.
//!
//! Provides R access to the fastLowess Rust library via extendr.
//!
//! @srrstats G1.0 Documentation of core R-to-Rust interface.
//! @srrstats G1.1 Implementation of thin R wrapper for statistical algorithms.

#![allow(non_snake_case)]

use extendr_api::prelude::*;

use fastLowess::internals::api::{
    BoundaryPolicy, RobustnessMethod,
    ScalingMethod::{self, MAD, MAR, Mean},
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
        "mean" => Ok(Mean),
        _ => Err(Error::Other(format!(
            "Unknown scaling method: {}. Valid options: mad, mar, mean",
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
// Stateful API: Lowess
// ============================================================================

#[extendr]
pub struct RLowess {
    builder: LowessBuilder<f64>,
    parallel: bool,
}

#[extendr]
impl RLowess {
    /// Create a new Lowess model
    #[allow(clippy::too_many_arguments)]
    fn new(
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
    ) -> Result<Self> {
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

        Ok(Self { builder, parallel })
    }

    /// Fit the model to data
    fn fit(&self, x: &[f64], y: &[f64]) -> Result<List> {
        let result = self
            .builder
            .clone()
            .adapter(Batch)
            .parallel(self.parallel)
            .build()
            .map_err(|e| Error::Other(e.to_string()))?
            .fit(x, y)
            .map_err(|e| Error::Other(e.to_string()))?;

        lowess_result_to_list(result)
    }
}

// ============================================================================
// Stateful API: StreamingLowess
// ============================================================================

#[extendr]
pub struct RStreamingLowess {
    inner: fastLowess::internals::adapters::streaming::ParallelStreamingLowess<f64>,
    fraction: f64,
    iterations: usize,
}

#[extendr]
impl RStreamingLowess {
    #[allow(clippy::too_many_arguments)]
    fn new(
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
    ) -> Result<Self> {
        let chunk_size = chunk_size as usize;
        let overlap_size = match overlap {
            NotNull(o) => o as usize,
            Null => (chunk_size / 10).min(chunk_size.saturating_sub(10)).max(1),
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

        let model = builder.build().map_err(|e| Error::Other(e.to_string()))?;
        Ok(Self {
            inner: model,
            fraction,
            iterations: iterations as usize,
        })
    }

    fn process_chunk(&mut self, x: &[f64], y: &[f64]) -> Result<List> {
        let mut result = self
            .inner
            .process_chunk(x, y)
            .map_err(|e| Error::Other(e.to_string()))?;
        result.fraction_used = self.fraction;
        result.iterations_used = Some(self.iterations);
        lowess_result_to_list(result)
    }

    fn finalize(&mut self) -> Result<List> {
        let mut result = self
            .inner
            .finalize()
            .map_err(|e| Error::Other(e.to_string()))?;
        result.fraction_used = self.fraction;
        result.iterations_used = Some(self.iterations);
        lowess_result_to_list(result)
    }
}

// ============================================================================
// Stateful API: OnlineLowess
// ============================================================================

#[extendr]
pub struct ROnlineLowess {
    inner: fastLowess::internals::adapters::online::ParallelOnlineLowess<f64>,
    fraction: f64,
    iterations: usize,
}

#[extendr]
impl ROnlineLowess {
    #[allow(clippy::too_many_arguments)]
    fn new(
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
    ) -> Result<Self> {
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

        let model = builder.build().map_err(|e| Error::Other(e.to_string()))?;
        Ok(Self {
            inner: model,
            fraction,
            iterations: iterations as usize,
        })
    }

    fn add_points(&mut self, x: &[f64], y: &[f64]) -> Result<List> {
        let outputs = self
            .inner
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
            robustness_weights: None,
            diagnostics: None,
            iterations_used: Some(self.iterations),
            fraction_used: self.fraction,
            cv_scores: None,
        };

        lowess_result_to_list(result)
    }
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
    List::from_names_and_values(names, values)
}

// ============================================================================
// Module Registration
// ============================================================================

extendr_module! {
    mod rfastlowess;
    impl RLowess;
    impl RStreamingLowess;
    impl ROnlineLowess;
}
