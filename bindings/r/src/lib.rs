//! R bindings for fastLowess.
//!
//! Provides R access to the fastLowess Rust library via extendr.
//!
//! @srrstats {G1.0} Documentation of core R-to-Rust interface.
//! @srrstats {G1.1} Implementation of thin R wrapper for statistical algorithms.

#![allow(non_snake_case)]

use extendr_api::prelude::*;

// Provide the Result alias that was removed from extendr_api::prelude in 0.9.0
type Result<T> = std::result::Result<T, extendr_api::Error>;

use fastLowess::internals::binding_support;
use fastLowess::internals::api::{
    Batch, BoundaryPolicy, LowessBuilder, Online, RobustnessMethod, ScalingMethod, Streaming,
    UpdateMode, WeightFunction, ZeroWeightFallback,
};
use fastLowess::prelude::LowessResult;

// ============================================================================
// Helper Functions
// ============================================================================

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
        let wf =
            binding_support::parse_weight_function(weight_function).map_err(|e| Error::Other(e))?;
        let rm = binding_support::parse_robustness_method(robustness_method)
            .map_err(|e| Error::Other(e))?;
        let sm =
            binding_support::parse_scaling_method(scaling_method).map_err(|e| Error::Other(e))?;
        let zwf = binding_support::parse_zero_weight_fallback(zero_weight_fallback)
            .map_err(|e| Error::Other(e))?;
        let bp =
            binding_support::parse_boundary_policy(boundary_policy).map_err(|e| Error::Other(e))?;

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
            builder = binding_support::apply_cross_validation(
                builder,
                Some(&fractions),
                Some(cv_method),
                Some(cv_k as usize),
                None,
            )
            .map_err(|e| Error::Other(e))?;
        }

        Ok(Self { builder, parallel })
    }

    /// Fit the model to data
    fn fit(&self, x: &[f64], y: &[f64], custom_weights: Nullable<Vec<f64>>) -> Result<List> {
        let mut builder = self.builder.clone();
        if let NotNull(cw) = custom_weights {
            builder = builder.custom_weights(cw);
        }
        let result = builder
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

        let wf =
            binding_support::parse_weight_function(weight_function).map_err(|e| Error::Other(e))?;
        let rm = binding_support::parse_robustness_method(robustness_method)
            .map_err(|e| Error::Other(e))?;
        let sm =
            binding_support::parse_scaling_method(scaling_method).map_err(|e| Error::Other(e))?;
        let bp =
            binding_support::parse_boundary_policy(boundary_policy).map_err(|e| Error::Other(e))?;

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
        let wf =
            binding_support::parse_weight_function(weight_function).map_err(|e| Error::Other(e))?;
        let rm = binding_support::parse_robustness_method(robustness_method)
            .map_err(|e| Error::Other(e))?;
        let sm =
            binding_support::parse_scaling_method(scaling_method).map_err(|e| Error::Other(e))?;
        let bp =
            binding_support::parse_boundary_policy(boundary_policy).map_err(|e| Error::Other(e))?;
        let um = binding_support::parse_update_mode(update_mode).map_err(|e| Error::Other(e))?;

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

    fn add_point(&mut self, x: f64, y: f64) -> Result<Option<f64>> {
        let result = self
            .inner
            .add_point(x, y)
            .map_err(|e| Error::Other(e.to_string()))?;
        Ok(result.map(|o| o.smoothed))
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
    let mut list = List::from_names_and_values(names, values)?;
    list.set_class(&["LowessResult"])?;
    Ok(list)
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
