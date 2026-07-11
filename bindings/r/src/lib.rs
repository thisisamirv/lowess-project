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

use fastLowess::internals::api::{LowessBuilder, LowessResult};
use fastLowess::internals::binding_support as shared_parse;

// ============================================================================
// Helper Functions
// ============================================================================

fn to_r_error(err: shared_parse::BindingError) -> Error {
    let prefix = match err.category {
        shared_parse::BindingErrorCategory::InvalidArg => "[invalid-arg]",
        shared_parse::BindingErrorCategory::Runtime => "[runtime]",
    };
    Error::Other(format!("{} {}", prefix, err.message))
}

fn map_invalid_arg<T, E: ToString>(result: std::result::Result<T, E>) -> Result<T> {
    shared_parse::map_invalid_arg(result).map_err(to_r_error)
}

fn map_runtime<T, E: ToString>(result: std::result::Result<T, E>) -> Result<T> {
    shared_parse::map_runtime(result).map_err(to_r_error)
}

fn require_positive_usize(name: &str, value: i32) -> Result<usize> {
    shared_parse::require_positive_usize(name, value)
        .map_err(|e| to_r_error(shared_parse::BindingError::invalid_arg(e)))
}

fn require_non_negative_usize(name: &str, value: i32) -> Result<usize> {
    shared_parse::require_non_negative_usize(name, value)
        .map_err(|e| to_r_error(shared_parse::BindingError::invalid_arg(e)))
}

// ============================================================================
// Stateful API: Lowess
// ============================================================================

#[extendr]
pub struct RLowess {
    builder: LowessBuilder<f64>,
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
        cv_seed: Nullable<i32>,
        return_se: bool,
    ) -> Result<Self> {
        let fractions = match cv_fractions {
            NotNull(v) => Some(v),
            Null => None,
        };
        let seed = match cv_seed {
            NotNull(s) => Some(s as u64),
            Null => None,
        };
        let iterations = require_non_negative_usize("iterations", iterations)?;
        let cv_k = require_positive_usize("cv_k", cv_k)?;

        let builder = map_invalid_arg(shared_parse::apply_builder_options(
            LowessBuilder::<f64>::new(),
            shared_parse::BuilderOptionSet {
                fraction: Some(fraction),
                iterations: Some(iterations),
                delta: match delta {
                    NotNull(d) => Some(d),
                    Null => None,
                },
                weight_function: Some(weight_function),
                robustness_method: Some(robustness_method),
                zero_weight_fallback: Some(zero_weight_fallback),
                boundary_policy: Some(boundary_policy),
                scaling_method: Some(scaling_method),
                auto_converge: match auto_converge {
                    NotNull(v) => Some(v),
                    Null => None,
                },
                return_residuals,
                return_robustness_weights,
                return_diagnostics,
                return_se,
                confidence_intervals: match confidence_intervals {
                    NotNull(v) => Some(v),
                    Null => None,
                },
                prediction_intervals: match prediction_intervals {
                    NotNull(v) => Some(v),
                    Null => None,
                },
                parallel: Some(parallel),
                cv_fractions: fractions.as_deref(),
                cv_method: Some(cv_method),
                cv_k: Some(cv_k),
                cv_seed: seed,
                ..Default::default()
            },
        ))?;

        Ok(Self { builder })
    }

    /// Fit the model to data
    fn fit(&self, x: &[f64], y: &[f64], custom_weights: Nullable<Vec<f64>>) -> Result<List> {
        let cw = match custom_weights {
            NotNull(w) => Some(w),
            Null => None,
        };
        let model = map_runtime(shared_parse::build_batch(self.builder.clone(), cw))?;
        let result = map_runtime(model.fit(x, y))?;
        lowess_result_to_list(result)
    }
}

// ============================================================================
// Stateful API: StreamingLowess
// ============================================================================

#[extendr]
pub struct RStreamingLowess {
    inner: fastLowess::internals::adapters::streaming::ParallelStreamingLowess<f64>,
}

#[extendr]
impl RStreamingLowess {
    #[allow(clippy::too_many_arguments)]
    fn new(
        fraction: f64,
        chunk_size: i32,
        overlap: Nullable<i32>,
        iterations: i32,
        weight_function: &str,
        robustness_method: &str,
        scaling_method: &str,
        boundary_policy: &str,
        zero_weight_fallback: &str,
        auto_converge: Nullable<f64>,
        return_diagnostics: bool,
        return_residuals: bool,
        return_robustness_weights: bool,
        merge_strategy: &str,
        parallel: bool,
        delta: Nullable<f64>,
        confidence_intervals: Nullable<f64>,
        prediction_intervals: Nullable<f64>,
    ) -> Result<Self> {
        let chunk_size = require_positive_usize("chunk_size", chunk_size)?;
        let overlap_size = match overlap {
            NotNull(o) => Some(require_non_negative_usize("overlap", o)?),
            Null => None,
        };
        let iterations = require_non_negative_usize("iterations", iterations)?;

        let builder = map_invalid_arg(shared_parse::apply_builder_options(
            LowessBuilder::<f64>::new(),
            shared_parse::BuilderOptionSet {
                fraction: Some(fraction),
                iterations: Some(iterations),
                delta: match delta {
                    NotNull(d) => Some(d),
                    Null => None,
                },
                weight_function: Some(weight_function),
                robustness_method: Some(robustness_method),
                zero_weight_fallback: Some(zero_weight_fallback),
                boundary_policy: Some(boundary_policy),
                scaling_method: Some(scaling_method),
                auto_converge: match auto_converge {
                    NotNull(v) => Some(v),
                    Null => None,
                },
                return_residuals,
                return_robustness_weights,
                return_diagnostics,
                confidence_intervals: match confidence_intervals {
                    NotNull(v) => Some(v),
                    Null => None,
                },
                prediction_intervals: match prediction_intervals {
                    NotNull(v) => Some(v),
                    Null => None,
                },
                parallel: Some(parallel),
                ..Default::default()
            },
        ))?;

        let model = map_runtime(shared_parse::build_streaming(
            builder,
            Some(chunk_size),
            overlap_size,
            Some(merge_strategy),
        ))?;
        Ok(Self { inner: model })
    }

    fn process_chunk(&mut self, x: &[f64], y: &[f64]) -> Result<List> {
        let result = map_runtime(self.inner.process_chunk(x, y))?;
        lowess_result_to_list(result)
    }

    fn finalize(&mut self) -> Result<List> {
        let result = map_runtime(self.inner.finalize())?;
        lowess_result_to_list(result)
    }
}

// ============================================================================
// Stateful API: OnlineLowess
// ============================================================================

#[extendr]
pub struct ROnlineLowess {
    inner: fastLowess::internals::adapters::online::ParallelOnlineLowess<f64>,
}

#[extendr]
impl ROnlineLowess {
    #[allow(clippy::too_many_arguments)]
    fn new(
        fraction: f64,
        window_capacity: i32,
        min_points: i32,
        iterations: i32,
        weight_function: &str,
        robustness_method: &str,
        scaling_method: &str,
        boundary_policy: &str,
        zero_weight_fallback: &str,
        update_mode: &str,
        auto_converge: Nullable<f64>,
        return_robustness_weights: bool,
        return_diagnostics: bool,
        return_residuals: bool,
        parallel: bool,
        delta: Nullable<f64>,
        confidence_intervals: Nullable<f64>,
        prediction_intervals: Nullable<f64>,
    ) -> Result<Self> {
        let window_capacity = require_positive_usize("window_capacity", window_capacity)?;
        let min_points = require_positive_usize("min_points", min_points)?;
        let iterations = require_non_negative_usize("iterations", iterations)?;

        let builder = map_invalid_arg(shared_parse::apply_builder_options(
            LowessBuilder::<f64>::new(),
            shared_parse::BuilderOptionSet {
                fraction: Some(fraction),
                iterations: Some(iterations),
                delta: match delta {
                    NotNull(d) => Some(d),
                    Null => None,
                },
                weight_function: Some(weight_function),
                robustness_method: Some(robustness_method),
                zero_weight_fallback: Some(zero_weight_fallback),
                boundary_policy: Some(boundary_policy),
                scaling_method: Some(scaling_method),
                auto_converge: match auto_converge {
                    NotNull(v) => Some(v),
                    Null => None,
                },
                return_residuals,
                return_robustness_weights,
                return_diagnostics,
                confidence_intervals: match confidence_intervals {
                    NotNull(v) => Some(v),
                    Null => None,
                },
                prediction_intervals: match prediction_intervals {
                    NotNull(v) => Some(v),
                    Null => None,
                },
                parallel: Some(parallel),
                ..Default::default()
            },
        ))?;

        let model = map_runtime(shared_parse::build_online(
            builder,
            Some(window_capacity),
            Some(min_points),
            Some(update_mode),
        ))?;
        Ok(Self { inner: model })
    }

    fn add_point(&mut self, x: f64, y: f64) -> Result<Nullable<List>> {
        let output = map_runtime(self.inner.add_point(x, y))?;

        match output {
            None => Ok(Null),
            Some(o) => {
                let mut items: Vec<(&str, Robj)> = vec![("smoothed", o.smoothed.into_robj())];
                if let Some(se) = o.std_error {
                    items.push(("std_error", se.into_robj()));
                }
                if let Some(res) = o.residual {
                    items.push(("residual", res.into_robj()));
                }
                if let Some(rw) = o.robustness_weight {
                    items.push(("robustness_weight", rw.into_robj()));
                }
                if let Some(iters) = o.iterations_used {
                    items.push(("iterations_used", (iters as i32).into_robj()));
                }
                Ok(NotNull(List::from_pairs(items)))
            }
        }
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
