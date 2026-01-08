//! Python bindings for fastLowess.

#![allow(non_snake_case)]

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::fmt::Display;

use ::fastLowess::internals::api::{
    BoundaryPolicy, RobustnessMethod, ScalingMethod, UpdateMode, WeightFunction, ZeroWeightFallback,
};
use ::fastLowess::prelude::{
    Batch, KFold, LOOCV, Lowess as LowessBuilder, LowessResult, MAD, MAR, Online, Streaming,
};

// ============================================================================
// Helper Functions
// ============================================================================

/// Convert a LowessError to a PyErr
fn to_py_error(e: impl Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Parse weight function from string
fn parse_weight_function(name: &str) -> PyResult<WeightFunction> {
    match name.to_lowercase().as_str() {
        "tricube" => Ok(WeightFunction::Tricube),
        "epanechnikov" => Ok(WeightFunction::Epanechnikov),
        "gaussian" => Ok(WeightFunction::Gaussian),
        "uniform" | "boxcar" => Ok(WeightFunction::Uniform),
        "biweight" | "bisquare" => Ok(WeightFunction::Biweight),
        "triangle" | "triangular" => Ok(WeightFunction::Triangle),
        "cosine" => Ok(WeightFunction::Cosine),
        _ => Err(PyValueError::new_err(format!(
            "Unknown weight function: {}. Valid options: tricube, epanechnikov, gaussian, uniform, biweight, triangle, cosine",
            name
        ))),
    }
}

/// Parse robustness method from string
fn parse_robustness_method(name: &str) -> PyResult<RobustnessMethod> {
    match name.to_lowercase().as_str() {
        "bisquare" | "biweight" => Ok(RobustnessMethod::Bisquare),
        "huber" => Ok(RobustnessMethod::Huber),
        "talwar" => Ok(RobustnessMethod::Talwar),
        _ => Err(PyValueError::new_err(format!(
            "Unknown robustness method: {}. Valid options: bisquare, huber, talwar",
            name
        ))),
    }
}

/// Parse zero weight fallback from string
fn parse_zero_weight_fallback(name: &str) -> PyResult<ZeroWeightFallback> {
    match name.to_lowercase().as_str() {
        "use_local_mean" | "local_mean" | "mean" => Ok(ZeroWeightFallback::UseLocalMean),
        "return_original" | "original" => Ok(ZeroWeightFallback::ReturnOriginal),
        "return_none" | "none" | "nan" => Ok(ZeroWeightFallback::ReturnNone),
        _ => Err(PyValueError::new_err(format!(
            "Unknown zero weight fallback: {}. Valid options: use_local_mean, return_original, return_none",
            name
        ))),
    }
}

/// Parse boundary policy from string
fn parse_boundary_policy(name: &str) -> PyResult<BoundaryPolicy> {
    match name.to_lowercase().as_str() {
        "extend" | "pad" => Ok(BoundaryPolicy::Extend),
        "reflect" | "mirror" => Ok(BoundaryPolicy::Reflect),
        "zero" | "none" => Ok(BoundaryPolicy::Zero),
        "noboundary" => Ok(BoundaryPolicy::NoBoundary),
        _ => Err(PyValueError::new_err(format!(
            "Unknown boundary policy: {}. Valid options: extend, reflect, zero, noboundary",
            name
        ))),
    }
}

/// Parse scaling method from string
fn parse_scaling_method(name: &str) -> PyResult<ScalingMethod> {
    match name.to_lowercase().as_str() {
        "mad" => Ok(MAD),
        "mar" => Ok(MAR),
        _ => Err(PyValueError::new_err(format!(
            "Unknown scaling method: {}. Valid options: mad, mar",
            name
        ))),
    }
}

/// Parse update mode from string
fn parse_update_mode(name: &str) -> PyResult<UpdateMode> {
    match name.to_lowercase().as_str() {
        "full" | "resmooth" => Ok(UpdateMode::Full),
        "incremental" | "single" => Ok(UpdateMode::Incremental),
        _ => Err(PyValueError::new_err(format!(
            "Unknown update mode: {}. Valid options: full, incremental",
            name
        ))),
    }
}

// ============================================================================
// Python Classes
// ============================================================================

/// Diagnostic statistics for LOWESS fit quality.
#[pyclass(name = "Diagnostics")]
#[derive(Clone)]
pub struct PyDiagnostics {
    /// Root Mean Squared Error
    #[pyo3(get)]
    pub rmse: f64,

    /// Mean Absolute Error
    #[pyo3(get)]
    pub mae: f64,

    /// R-squared (coefficient of determination)
    #[pyo3(get)]
    pub r_squared: f64,

    /// Akaike Information Criterion
    #[pyo3(get)]
    pub aic: Option<f64>,

    /// Corrected AIC
    #[pyo3(get)]
    pub aicc: Option<f64>,

    /// Effective degrees of freedom
    #[pyo3(get)]
    pub effective_df: Option<f64>,

    /// Residual standard deviation
    #[pyo3(get)]
    pub residual_sd: f64,
}

#[pymethods]
impl PyDiagnostics {
    fn __repr__(&self) -> String {
        format!(
            "Diagnostics(rmse={:.6}, mae={:.6}, r_squared={:.6})",
            self.rmse, self.mae, self.r_squared
        )
    }
}

/// Result from LOWESS smoothing.
#[pyclass(name = "LowessResult")]
pub struct PyLowessResult {
    inner: LowessResult<f64>,
}

#[pymethods]
impl PyLowessResult {
    /// Sorted x values
    #[getter]
    fn x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.x.clone())
    }

    /// Smoothed y values
    #[getter]
    fn y<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.y.clone())
    }

    /// Standard errors (if computed)
    #[getter]
    fn standard_errors<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .standard_errors
            .as_ref()
            .map(|v| PyArray1::from_vec(py, v.clone()))
    }

    /// Lower confidence interval bounds
    #[getter]
    fn confidence_lower<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .confidence_lower
            .as_ref()
            .map(|v| PyArray1::from_vec(py, v.clone()))
    }

    /// Upper confidence interval bounds
    #[getter]
    fn confidence_upper<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .confidence_upper
            .as_ref()
            .map(|v| PyArray1::from_vec(py, v.clone()))
    }

    /// Lower prediction interval bounds
    #[getter]
    fn prediction_lower<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .prediction_lower
            .as_ref()
            .map(|v| PyArray1::from_vec(py, v.clone()))
    }

    /// Upper prediction interval bounds
    #[getter]
    fn prediction_upper<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .prediction_upper
            .as_ref()
            .map(|v| PyArray1::from_vec(py, v.clone()))
    }

    /// Residuals (original y - smoothed y)
    #[getter]
    fn residuals<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .residuals
            .as_ref()
            .map(|v| PyArray1::from_vec(py, v.clone()))
    }

    /// Robustness weights from final iteration
    #[getter]
    fn robustness_weights<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .robustness_weights
            .as_ref()
            .map(|v| PyArray1::from_vec(py, v.clone()))
    }

    /// Diagnostic metrics
    #[getter]
    fn diagnostics(&self) -> Option<PyDiagnostics> {
        self.inner.diagnostics.as_ref().map(|d| PyDiagnostics {
            rmse: d.rmse,
            mae: d.mae,
            r_squared: d.r_squared,
            aic: d.aic,
            aicc: d.aicc,
            effective_df: d.effective_df,
            residual_sd: d.residual_sd,
        })
    }

    /// Number of iterations performed
    #[getter]
    fn iterations_used(&self) -> Option<usize> {
        self.inner.iterations_used
    }

    /// Fraction used for smoothing
    #[getter]
    fn fraction_used(&self) -> f64 {
        self.inner.fraction_used
    }

    /// CV scores for tested fractions
    #[getter]
    fn cv_scores<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .cv_scores
            .as_ref()
            .map(|v| PyArray1::from_vec(py, v.clone()))
    }

    fn __repr__(&self) -> String {
        format!(
            "LowessResult(n={}, fraction_used={:.4})",
            self.inner.y.len(),
            self.inner.fraction_used
        )
    }
}

// ============================================================================
// Python Functions
// ============================================================================

/// LOWESS smoothing with the batch adapter.
///
/// Parameters
/// ----------
/// x : array_like
///     Independent variable values.
/// y : array_like
///     Dependent variable values.
/// fraction : float, optional
///     Smoothing fraction (default: 0.67).
/// iterations : int, optional
///     Number of robustness iterations (default: 3).
/// delta : float, optional
///     Interpolation optimization threshold.
/// weight_function : str, optional
///     Kernel function: "tricube", "epanechnikov", "gaussian", "uniform", "biweight", "triangle".
/// robustness_method : str, optional
///     Robustness method: "bisquare", "huber", "talwar".
/// scaling_method : str, optional
///     Scaling method for robustness: "mad", "mar" (default: "mad").
/// boundary_policy : str, optional
///     Handling of edge effects: "extend", "reflect", "zero", "noboundary" (default: "extend").
/// confidence_intervals : float, optional
///     Confidence level for confidence intervals (e.g., 0.95).
/// prediction_intervals : float, optional
///     Confidence level for prediction intervals (e.g., 0.95).
/// return_diagnostics : bool, optional
///     Whether to compute RMSE, MAE, R², etc.
/// return_residuals : bool, optional
///     Whether to include residuals in output.
/// return_robustness_weights : bool, optional
///     Whether to include robustness weights in output.
/// zero_weight_fallback : str, optional
///     Fallback when all weights are zero: "use_local_mean", "return_original", "return_none".
/// auto_converge : float, optional
///     Tolerance for auto-convergence (disabled by default).
/// cv_fractions : list of float, optional
///     Fractions to test for cross-validation (disabled by default).
///     When provided, enables cross-validation to select optimal fraction.
/// cv_method : str, optional
///     CV method: "loocv" (leave-one-out) or "kfold". Default: "kfold".
/// cv_k : int, optional
///     Number of folds for k-fold CV (default: 5).
#[pyfunction]
#[pyo3(signature = (
    x, y,
    fraction=0.67,
    iterations=3,
    delta=None,
    weight_function="tricube",
    robustness_method="bisquare",
    scaling_method="mad",
    boundary_policy="extend",
    confidence_intervals=None,
    prediction_intervals=None,
    return_diagnostics=false,
    return_residuals=false,
    return_robustness_weights=false,
    zero_weight_fallback="use_local_mean",
    auto_converge=None,
    cv_fractions=None,
    cv_method="kfold",
    cv_k=5,
    parallel=true
))]
#[allow(clippy::too_many_arguments)]
fn smooth<'py>(
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    fraction: f64,
    iterations: usize,
    delta: Option<f64>,
    weight_function: &str,
    robustness_method: &str,
    scaling_method: &str,
    boundary_policy: &str,
    confidence_intervals: Option<f64>,
    prediction_intervals: Option<f64>,
    return_diagnostics: bool,
    return_residuals: bool,
    return_robustness_weights: bool,
    zero_weight_fallback: &str,
    auto_converge: Option<f64>,
    cv_fractions: Option<Vec<f64>>,
    cv_method: &str,
    cv_k: usize,
    parallel: bool,
) -> PyResult<PyLowessResult> {
    let x_slice = x.as_slice().map_err(to_py_error)?;
    let y_slice = y.as_slice().map_err(to_py_error)?;

    let wf = parse_weight_function(weight_function)?;
    let rm = parse_robustness_method(robustness_method)?;
    let sm = parse_scaling_method(scaling_method)?;
    let zwf = parse_zero_weight_fallback(zero_weight_fallback)?;
    let bp = parse_boundary_policy(boundary_policy)?;

    let mut builder = LowessBuilder::<f64>::new();
    builder = builder.fraction(fraction);
    builder = builder.iterations(iterations);
    builder = builder.weight_function(wf);
    builder = builder.robustness_method(rm);
    builder = builder.scaling_method(sm);
    builder = builder.zero_weight_fallback(zwf);
    builder = builder.boundary_policy(bp);
    builder = builder.parallel(parallel);

    if let Some(d) = delta {
        builder = builder.delta(d);
    }

    if let Some(cl) = confidence_intervals {
        builder = builder.confidence_intervals(cl);
    }

    if let Some(pl) = prediction_intervals {
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

    if let Some(tol) = auto_converge {
        builder = builder.auto_converge(tol);
    }

    // Cross-validation if fractions are provided
    if let Some(fractions) = cv_fractions {
        match cv_method.to_lowercase().as_str() {
            "simple" | "loo" | "loocv" | "leave_one_out" => {
                builder = builder.cross_validate(LOOCV(&fractions));
            }
            "kfold" | "k_fold" | "k-fold" => {
                builder = builder.cross_validate(KFold(cv_k, &fractions));
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown CV method: {}. Valid options: loocv, kfold",
                    cv_method
                )));
            }
        };
    }

    let result = builder
        .adapter(Batch)
        .build()
        .map_err(to_py_error)?
        .fit(x_slice, y_slice)
        .map_err(to_py_error)?;

    Ok(PyLowessResult { inner: result })
}

/// Streaming LOWESS for large datasets.
///
/// Parameters
/// ----------
/// x : array_like
///     Independent variable values.
/// y : array_like
///     Dependent variable values.
/// fraction : float, optional
///     Smoothing fraction (default: 0.3).
/// chunk_size : int, optional
///     Size of each processing chunk (default: 5000).
/// overlap : int, optional
///     Overlap between chunks (default: 10% of chunk_size).
/// iterations : int, optional
///     Number of robustness iterations (default: 3).
/// delta : float, optional
///     Interpolation optimization threshold.
/// weight_function : str, optional
///     Kernel function: "tricube", "epanechnikov", "gaussian", "uniform", "biweight", "triangle".
/// robustness_method : str, optional
///     Robustness method: "bisquare", "huber", "talwar".
/// scaling_method : str, optional
///     Scaling method for robustness: "mad", "mar" (default: "mad").
/// boundary_policy : str, optional
///     Handling of edge effects: "extend", "reflect", "zero", "noboundary" (default: "extend").
/// auto_converge : float, optional
///     Tolerance for auto-convergence (disabled by default).
/// return_diagnostics : bool, optional
///     Whether to compute cumulative diagnostics across chunks.
/// return_robustness_weights : bool, optional
///     Whether to include robustness weights.
/// parallel : bool, optional
///     Enable parallel execution (default: True).
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    x, y,
    fraction=0.3,
    chunk_size=5000,
    overlap=None,
    iterations=3,
    delta=None,
    weight_function="tricube",
    robustness_method="bisquare",
    scaling_method="mad",
    boundary_policy="extend",
    auto_converge=None,
    return_diagnostics=false,
    return_residuals=false,
    return_robustness_weights=false,
    zero_weight_fallback="use_local_mean",
    parallel=true
))]
fn smooth_streaming<'py>(
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    fraction: f64,
    chunk_size: usize,
    overlap: Option<usize>,
    iterations: usize,
    delta: Option<f64>,
    weight_function: &str,
    robustness_method: &str,
    scaling_method: &str,
    boundary_policy: &str,
    auto_converge: Option<f64>,
    return_diagnostics: bool,
    return_residuals: bool,
    return_robustness_weights: bool,
    zero_weight_fallback: &str,
    parallel: bool,
) -> PyResult<PyLowessResult> {
    let x_slice = x.as_slice().map_err(to_py_error)?;
    let y_slice = y.as_slice().map_err(to_py_error)?;

    // Default overlap to 10% of chunk_size, capped at chunk_size - 10
    let overlap_size = overlap.unwrap_or_else(|| {
        let default = chunk_size / 10;
        default.min(chunk_size.saturating_sub(10)).max(1)
    });

    let wf = parse_weight_function(weight_function)?;
    let rm = parse_robustness_method(robustness_method)?;
    let sm = parse_scaling_method(scaling_method)?;
    let zwf = parse_zero_weight_fallback(zero_weight_fallback)?;
    let bp = parse_boundary_policy(boundary_policy)?;

    let mut builder = LowessBuilder::<f64>::new();
    builder = builder.fraction(fraction);
    builder = builder.iterations(iterations);
    builder = builder.weight_function(wf);
    builder = builder.robustness_method(rm);
    builder = builder.scaling_method(sm);
    builder = builder.zero_weight_fallback(zwf);
    builder = builder.boundary_policy(bp);

    if return_diagnostics {
        builder = builder.return_diagnostics();
    }
    if return_residuals {
        builder = builder.return_residuals();
    }
    if return_robustness_weights {
        builder = builder.return_robustness_weights();
    }

    let mut builder = builder.adapter(Streaming);
    builder = builder.chunk_size(chunk_size);
    builder = builder.overlap(overlap_size);
    builder = builder.parallel(parallel);

    if let Some(d) = delta {
        builder = builder.delta(d);
    }
    if let Some(tol) = auto_converge {
        builder = builder.auto_converge(tol);
    }

    let mut processor = builder.build().map_err(to_py_error)?;

    // Process the data as a single chunk
    let chunk_result = processor
        .process_chunk(x_slice, y_slice)
        .map_err(to_py_error)?;

    // Finalize to get remaining buffered overlap data
    let final_result = processor.finalize().map_err(to_py_error)?;

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

    Ok(PyLowessResult { inner: result })
}

/// Online LOWESS with sliding window.
///
/// Parameters
/// ----------
/// x : array_like
///     Independent variable values.
/// y : array_like
///     Dependent variable values.
/// fraction : float, optional
///     Smoothing fraction (default: 0.2).
/// window_capacity : int, optional
///     Maximum points to retain in window (default: 100).
/// min_points : int, optional
///     Minimum points before smoothing starts (default: 2).
/// iterations : int, optional
///     Number of robustness iterations (default: 3).
/// delta : float, optional
///     Interpolation optimization threshold.
/// weight_function : str, optional
///     Kernel function: "tricube", "epanechnikov", "gaussian", "uniform", "biweight", "triangle".
/// robustness_method : str, optional
///     Robustness method: "bisquare", "huber", "talwar".
/// boundary_policy : str, optional
///     Handling of edge effects: "extend", "reflect", "zero", "noboundary" (default: "extend").
/// update_mode : str, optional
///     Update strategy: "full" or "incremental" (default: "full").
/// auto_converge : float, optional
///     Tolerance for auto-convergence (disabled by default).
/// return_robustness_weights : bool, optional
///     Whether to include robustness weights.
/// parallel : bool, optional
///     Enable parallel execution (default: False for online).
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    x, y,
    fraction=0.2,
    window_capacity=100,
    min_points=2,
    iterations=3,
    delta=None,
    weight_function="tricube",
    robustness_method="bisquare",
    scaling_method="mad",
    boundary_policy="extend",
    update_mode="full",
    auto_converge=None,
    return_robustness_weights=false,
    zero_weight_fallback="use_local_mean",
    parallel=false
))]
fn smooth_online<'py>(
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    fraction: f64,
    window_capacity: usize,
    min_points: usize,
    iterations: usize,
    delta: Option<f64>,
    weight_function: &str,
    robustness_method: &str,
    scaling_method: &str,
    boundary_policy: &str,
    update_mode: &str,
    auto_converge: Option<f64>,
    return_robustness_weights: bool,
    zero_weight_fallback: &str,
    parallel: bool,
) -> PyResult<PyLowessResult> {
    let x_slice = x.as_slice().map_err(to_py_error)?;
    let y_slice = y.as_slice().map_err(to_py_error)?;

    let wf = parse_weight_function(weight_function)?;
    let rm = parse_robustness_method(robustness_method)?;
    let sm = parse_scaling_method(scaling_method)?;
    let bp = parse_boundary_policy(boundary_policy)?;
    let zwf = parse_zero_weight_fallback(zero_weight_fallback)?;
    let um = parse_update_mode(update_mode)?;

    let mut builder = LowessBuilder::<f64>::new();
    builder = builder.fraction(fraction);
    builder = builder.iterations(iterations);
    builder = builder.weight_function(wf);
    builder = builder.robustness_method(rm);
    builder = builder.scaling_method(sm);
    builder = builder.zero_weight_fallback(zwf);
    builder = builder.boundary_policy(bp);

    let mut builder = builder.adapter(Online);
    builder = builder.window_capacity(window_capacity);
    builder = builder.min_points(min_points);
    builder = builder.update_mode(um);
    builder = builder.parallel(parallel);

    if let Some(d) = delta {
        builder = builder.delta(d);
    }
    if let Some(tol) = auto_converge {
        builder = builder.auto_converge(tol);
    }
    if return_robustness_weights {
        builder = builder.return_robustness_weights(true);
    }

    let mut processor = builder.build().map_err(to_py_error)?;

    let outputs = processor
        .add_points(x_slice, y_slice)
        .map_err(to_py_error)?;

    // Extract smoothed values (use original y for points that haven't accumulated enough data)
    let smoothed: Vec<f64> = outputs
        .into_iter()
        .zip(y_slice.iter())
        .map(|(opt, &original_y)| opt.map_or(original_y, |o| o.smoothed))
        .collect();

    // Create result
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
        iterations_used: Some(iterations),
        fraction_used: fraction,
        cv_scores: None,
    };

    Ok(PyLowessResult { inner: result })
}

// ============================================================================
// Module Registration
// ============================================================================

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLowessResult>()?;
    m.add_class::<PyDiagnostics>()?;
    m.add_function(wrap_pyfunction!(smooth, m)?)?;
    m.add_function(wrap_pyfunction!(smooth_streaming, m)?)?;
    m.add_function(wrap_pyfunction!(smooth_online, m)?)?;
    Ok(())
}
