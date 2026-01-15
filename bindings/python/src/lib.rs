//! Python bindings for fastLowess.

#![allow(non_snake_case)]

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::fmt::Display;

use ::fastLowess::internals::adapters::online::ParallelOnlineLowess;
use ::fastLowess::internals::adapters::streaming::ParallelStreamingLowess;
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
// Python Classes - Stateful Adapters
// ============================================================================

/// Streaming LOWESS processor for incremental chunk-based smoothing.
#[pyclass(name = "StreamingLowess")]
pub struct PyStreamingLowess {
    inner: ParallelStreamingLowess<f64>,
}

#[pymethods]
impl PyStreamingLowess {
    #[new]
    #[pyo3(signature = (
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
    #[allow(clippy::too_many_arguments)]
    fn new(
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
    ) -> PyResult<Self> {
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

        let overlap_size = overlap.unwrap_or_else(|| {
            let default = chunk_size / 10;
            default.min(chunk_size.saturating_sub(10)).max(1)
        });

        let mut streaming_builder = builder.adapter(Streaming);
        streaming_builder = streaming_builder.chunk_size(chunk_size);
        streaming_builder = streaming_builder.overlap(overlap_size);
        streaming_builder = streaming_builder.parallel(parallel);

        if let Some(d) = delta {
            streaming_builder = streaming_builder.delta(d);
        }
        if let Some(tol) = auto_converge {
            streaming_builder = streaming_builder.auto_converge(tol);
        }

        let processor = streaming_builder.build().map_err(to_py_error)?;
        Ok(PyStreamingLowess { inner: processor })
    }

    /// Process a chunk of data.
    fn process_chunk<'py>(
        &mut self,
        x: PyReadonlyArray1<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyLowessResult> {
        let x_slice = x.as_slice().map_err(to_py_error)?;
        let y_slice = y.as_slice().map_err(to_py_error)?;

        let result = self
            .inner
            .process_chunk(x_slice, y_slice)
            .map_err(to_py_error)?;
        Ok(PyLowessResult { inner: result })
    }

    /// Finalize smoothing and return remaining buffered data.
    fn finalize(&mut self) -> PyResult<PyLowessResult> {
        let result = self.inner.finalize().map_err(to_py_error)?;
        Ok(PyLowessResult { inner: result })
    }
}

/// Online LOWESS processor for real-time data streams.
#[pyclass(name = "OnlineLowess")]
pub struct PyOnlineLowess {
    inner: ParallelOnlineLowess<f64>,
    fraction: f64,
    iterations: usize,
}

#[pymethods]
impl PyOnlineLowess {
    #[new]
    #[pyo3(signature = (
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
    #[allow(clippy::too_many_arguments)]
    fn new(
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
    ) -> PyResult<Self> {
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

        let mut online_builder = builder.adapter(Online);
        online_builder = online_builder.window_capacity(window_capacity);
        online_builder = online_builder.min_points(min_points);
        online_builder = online_builder.update_mode(um);
        online_builder = online_builder.parallel(parallel);

        if let Some(d) = delta {
            online_builder = online_builder.delta(d);
        }
        if let Some(tol) = auto_converge {
            online_builder = online_builder.auto_converge(tol);
        }
        if return_robustness_weights {
            online_builder = online_builder.return_robustness_weights(true);
        }

        let processor = online_builder.build().map_err(to_py_error)?;
        Ok(PyOnlineLowess {
            inner: processor,
            fraction,
            iterations,
        })
    }

    /// Add a single point and return smoothed value if enough points are available.
    fn update(&mut self, x: f64, y: f64) -> PyResult<Option<f64>> {
        let result = self.inner.add_point(x, y).map_err(to_py_error)?;
        Ok(result.map(|o| o.smoothed))
    }

    /// Add multiple points.
    fn add_points<'py>(
        &mut self,
        x: PyReadonlyArray1<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyLowessResult> {
        let x_slice = x.as_slice().map_err(to_py_error)?;
        let y_slice = y.as_slice().map_err(to_py_error)?;

        let outputs = self
            .inner
            .add_points(x_slice, y_slice)
            .map_err(to_py_error)?;

        // Extract smoothed values
        let smoothed: Vec<f64> = outputs
            .into_iter()
            .zip(y_slice.iter())
            .map(|(opt, &original_y)| opt.map_or(original_y, |o| o.smoothed))
            .collect();

        Ok(PyLowessResult {
            inner: LowessResult {
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
                iterations_used: Some(self.iterations),
                fraction_used: self.fraction,
                cv_scores: None,
            },
        })
    }
}

/// Batch LOWESS processor with configurable parameters.
///
/// This class allows you to configure LOWESS parameters once and then
/// call `fit()` multiple times with different datasets.
#[pyclass(name = "Lowess")]
pub struct PyLowess {
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

#[pymethods]
impl PyLowess {
    #[new]
    #[pyo3(signature = (
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
    fn new(
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
    ) -> PyResult<Self> {
        let wf = parse_weight_function(weight_function)?;
        let rm = parse_robustness_method(robustness_method)?;
        let sm = parse_scaling_method(scaling_method)?;
        let zwf = parse_zero_weight_fallback(zero_weight_fallback)?;
        let bp = parse_boundary_policy(boundary_policy)?;

        Ok(PyLowess {
            fraction,
            iterations,
            delta,
            weight_function: wf,
            robustness_method: rm,
            scaling_method: sm,
            zero_weight_fallback: zwf,
            boundary_policy: bp,
            auto_converge,
            confidence_intervals,
            prediction_intervals,
            return_diagnostics,
            return_residuals,
            return_robustness_weights,
            cv_fractions,
            cv_method: cv_method.to_string(),
            cv_k,
            parallel,
        })
    }

    /// Fit LOWESS model to data.
    ///
    /// Parameters
    /// ----------
    /// x : array_like
    ///     Independent variable values.
    /// y : array_like
    ///     Dependent variable values.
    ///
    /// Returns
    /// -------
    /// LowessResult
    ///     Smoothed values and optional diagnostics.
    fn fit<'py>(
        &self,
        x: PyReadonlyArray1<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyLowessResult> {
        let x_slice = x.as_slice().map_err(to_py_error)?;
        let y_slice = y.as_slice().map_err(to_py_error)?;

        let mut builder = LowessBuilder::<f64>::new();
        builder = builder.fraction(self.fraction);
        builder = builder.iterations(self.iterations);
        builder = builder.weight_function(self.weight_function);
        builder = builder.robustness_method(self.robustness_method);
        builder = builder.scaling_method(self.scaling_method);
        builder = builder.zero_weight_fallback(self.zero_weight_fallback);
        builder = builder.boundary_policy(self.boundary_policy);
        builder = builder.parallel(self.parallel);

        if let Some(d) = self.delta {
            builder = builder.delta(d);
        }

        if let Some(cl) = self.confidence_intervals {
            builder = builder.confidence_intervals(cl);
        }

        if let Some(pl) = self.prediction_intervals {
            builder = builder.prediction_intervals(pl);
        }

        if self.return_diagnostics {
            builder = builder.return_diagnostics();
        }

        if self.return_residuals {
            builder = builder.return_residuals();
        }

        if self.return_robustness_weights {
            builder = builder.return_robustness_weights();
        }

        if let Some(tol) = self.auto_converge {
            builder = builder.auto_converge(tol);
        }

        // Cross-validation if fractions are provided
        if let Some(ref fractions) = self.cv_fractions {
            match self.cv_method.to_lowercase().as_str() {
                "simple" | "loo" | "loocv" | "leave_one_out" => {
                    builder = builder.cross_validate(LOOCV(fractions));
                }
                "kfold" | "k_fold" | "k-fold" => {
                    builder = builder.cross_validate(KFold(self.cv_k, fractions));
                }
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "Unknown CV method: {}. Valid options: loocv, kfold",
                        self.cv_method
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

    fn __repr__(&self) -> String {
        format!(
            "Lowess(fraction={:.4}, iterations={}, parallel={})",
            self.fraction, self.iterations, self.parallel
        )
    }
}

// ============================================================================
// Module Registration
// ============================================================================

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLowessResult>()?;
    m.add_class::<PyDiagnostics>()?;
    m.add_class::<PyLowess>()?;
    m.add_class::<PyStreamingLowess>()?;
    m.add_class::<PyOnlineLowess>()?;
    Ok(())
}
