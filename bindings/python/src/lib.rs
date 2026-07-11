//! Python bindings for fastLowess.

#![allow(non_snake_case)]
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::fmt::Display;
use std::sync::Mutex;

use fastLowess::internals::adapters::online::ParallelOnlineLowess;
use fastLowess::internals::adapters::streaming::ParallelStreamingLowess;
use fastLowess::internals::api::{LowessBuilder, Online, Streaming};
use fastLowess::internals::binding_support;

use fastLowess::prelude::LowessResult;

// ============================================================================
// Helper Functions
// ============================================================================

fn to_py_error(err: binding_support::BindingError) -> PyErr {
    match err.category {
        binding_support::BindingErrorCategory::InvalidArg => PyValueError::new_err(err.message),
        binding_support::BindingErrorCategory::Runtime => PyRuntimeError::new_err(err.message),
    }
}

fn map_invalid_arg<T, E: Display>(result: Result<T, E>) -> PyResult<T> {
    binding_support::map_invalid_arg(result).map_err(to_py_error)
}

fn to_py_invalid_arg_error(e: impl Display) -> PyErr {
    to_py_error(binding_support::BindingError::invalid_arg(e.to_string()))
}

// ============================================================================
// Python Classes
// ============================================================================

/// Diagnostic statistics for LOWESS fit quality.
#[pyclass(name = "Diagnostics", from_py_object)]
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
    inner: Mutex<ParallelStreamingLowess<f64>>,
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
        parallel=true,
        merge_strategy="weighted_average"
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
        merge_strategy: &str,
    ) -> PyResult<Self> {
        let overlap_size = overlap.unwrap_or_else(|| binding_support::default_overlap(chunk_size));

        let builder = map_invalid_arg(binding_support::apply_builder_options(
            LowessBuilder::<f64>::new(),
            binding_support::BuilderOptionSet {
                fraction: Some(fraction),
                iterations: Some(iterations),
                delta,
                weight_function: Some(weight_function),
                robustness_method: Some(robustness_method),
                scaling_method: Some(scaling_method),
                boundary_policy: Some(boundary_policy),
                zero_weight_fallback: Some(zero_weight_fallback),
                auto_converge,
                return_residuals,
                return_robustness_weights,
                return_diagnostics,
                return_se: false,
                confidence_intervals: None,
                prediction_intervals: None,
                parallel: Some(parallel),
                chunk_size: Some(chunk_size),
                overlap: Some(overlap_size),
                merge_strategy: Some(merge_strategy),
                window_capacity: None,
                min_points: None,
                update_mode: None,
                cv_fractions: None,
                cv_method: None,
                cv_k: None,
                cv_seed: None,
            },
        ))?;

        let processor = binding_support::map_lowess_result(builder.adapter(Streaming).build())
            .map_err(to_py_error)?;
        Ok(PyStreamingLowess {
            inner: Mutex::new(processor),
        })
    }

    /// Process a chunk of data.
    fn process_chunk<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyLowessResult> {
        let x_vec = x.as_slice().map_err(to_py_invalid_arg_error)?.to_vec();
        let y_vec = y.as_slice().map_err(to_py_invalid_arg_error)?.to_vec();

        let result = py.detach(move || {
            self.inner
                .lock()
                .map_err(|e| {
                    to_py_error(binding_support::BindingError::runtime(
                        binding_support::mutex_poisoned_message(&e.to_string()),
                    ))
                })?
                .process_chunk(&x_vec, &y_vec)
                .map_err(|e| to_py_error(binding_support::BindingError::runtime(e.to_string())))
        })?;

        Ok(PyLowessResult { inner: result })
    }

    /// Finalize smoothing and return remaining buffered data.
    fn finalize(&self, py: Python<'_>) -> PyResult<PyLowessResult> {
        let result = py.detach(move || {
            self.inner
                .lock()
                .map_err(|e| {
                    to_py_error(binding_support::BindingError::runtime(
                        binding_support::mutex_poisoned_message(&e.to_string()),
                    ))
                })?
                .finalize()
                .map_err(|e| to_py_error(binding_support::BindingError::runtime(e.to_string())))
        })?;

        Ok(PyLowessResult { inner: result })
    }
}

/// Result from a single online update step.
#[pyclass(name = "OnlineOutput", from_py_object)]
#[derive(Clone)]
pub struct PyOnlineOutput {
    /// Smoothed value for the latest point
    #[pyo3(get)]
    pub smoothed: f64,
    /// Standard error (if computed)
    #[pyo3(get)]
    pub std_error: Option<f64>,
    /// Residual y − smoothed (if computed)
    #[pyo3(get)]
    pub residual: Option<f64>,
    /// Robustness weight for the latest point (if computed)
    #[pyo3(get)]
    pub robustness_weight: Option<f64>,
    /// Number of robustness iterations performed (if tracked)
    #[pyo3(get)]
    pub iterations_used: Option<usize>,
}

#[pymethods]
impl PyOnlineOutput {
    fn __repr__(&self) -> String {
        format!("OnlineOutput(smoothed={:.4})", self.smoothed)
    }
}

/// Online LOWESS processor for real-time data streams.
#[pyclass(name = "OnlineLowess")]
pub struct PyOnlineLowess {
    inner: Mutex<ParallelOnlineLowess<f64>>,
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
        return_diagnostics=false,
        return_residuals=false,
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
        return_diagnostics: bool,
        return_residuals: bool,
        zero_weight_fallback: &str,
        parallel: bool,
    ) -> PyResult<Self> {
        let builder = map_invalid_arg(binding_support::apply_builder_options(
            LowessBuilder::<f64>::new(),
            binding_support::BuilderOptionSet {
                fraction: Some(fraction),
                iterations: Some(iterations),
                delta,
                weight_function: Some(weight_function),
                robustness_method: Some(robustness_method),
                scaling_method: Some(scaling_method),
                boundary_policy: Some(boundary_policy),
                zero_weight_fallback: Some(zero_weight_fallback),
                auto_converge,
                return_residuals,
                return_robustness_weights,
                return_diagnostics,
                return_se: false,
                confidence_intervals: None,
                prediction_intervals: None,
                parallel: Some(parallel),
                chunk_size: None,
                overlap: None,
                merge_strategy: None,
                window_capacity: Some(window_capacity),
                min_points: Some(min_points),
                update_mode: Some(update_mode),
                cv_fractions: None,
                cv_method: None,
                cv_k: None,
                cv_seed: None,
            },
        ))?;

        let processor = binding_support::map_lowess_result(builder.adapter(Online).build())
            .map_err(to_py_error)?;
        Ok(PyOnlineLowess {
            inner: Mutex::new(processor),
        })
    }

    /// Add a single point and return its smoothed value, or None if the window
    /// is still filling up.
    fn add_point(&self, x: f64, y: f64) -> PyResult<Option<PyOnlineOutput>> {
        let mut inner = self.inner.lock().map_err(|e| {
            to_py_error(binding_support::BindingError::runtime(
                binding_support::mutex_poisoned_message(&e.to_string()),
            ))
        })?;
        let result = inner
            .add_point(x, y)
            .map_err(|e| to_py_error(binding_support::BindingError::invalid_arg(e.to_string())))?;
        Ok(result.map(|o| PyOnlineOutput {
            smoothed: o.smoothed,
            std_error: o.std_error,
            residual: o.residual,
            robustness_weight: o.robustness_weight,
            iterations_used: o.iterations_used,
        }))
    }
}

/// Batch LOWESS processor with configurable parameters.
///
/// This class allows you to configure LOWESS parameters once and then
/// call `fit()` multiple times with different datasets.
#[pyclass(name = "Lowess", from_py_object)]
#[derive(Clone)]
pub struct PyLowess {
    builder: LowessBuilder<f64>,
    // Kept only for __repr__
    fraction: f64,
    iterations: usize,
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
        parallel=true,
        cv_seed=None,
        return_se=false
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
        cv_seed: Option<u64>,
        return_se: bool,
    ) -> PyResult<Self> {
        let builder = map_invalid_arg(binding_support::apply_builder_options(
            LowessBuilder::<f64>::new(),
            binding_support::BuilderOptionSet {
                fraction: Some(fraction),
                iterations: Some(iterations),
                delta,
                weight_function: Some(weight_function),
                robustness_method: Some(robustness_method),
                scaling_method: Some(scaling_method),
                boundary_policy: Some(boundary_policy),
                zero_weight_fallback: Some(zero_weight_fallback),
                auto_converge,
                return_residuals,
                return_robustness_weights,
                return_diagnostics,
                return_se,
                confidence_intervals,
                prediction_intervals,
                parallel: Some(parallel),
                chunk_size: None,
                overlap: None,
                merge_strategy: None,
                window_capacity: None,
                min_points: None,
                update_mode: None,
                cv_fractions: cv_fractions.as_deref(),
                cv_method: Some(cv_method),
                cv_k: Some(cv_k),
                cv_seed,
            },
        ))?;

        Ok(PyLowess {
            builder,
            fraction,
            iterations,
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
    /// custom_weights : array_like, optional
    ///     Per-observation weights multiplied into the kernel weight before
    ///     each local regression. Use 0.0 to suppress known-bad points.
    ///
    /// Returns
    /// -------
    /// LowessResult
    ///     Smoothed values and optional diagnostics.
    #[pyo3(signature = (x, y, custom_weights=None))]
    fn fit<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
        custom_weights: Option<PyReadonlyArray1<'py, f64>>,
    ) -> PyResult<PyLowessResult> {
        // 1. Copy data (Must be done with GIL)
        let x_vec = x.as_slice().map_err(to_py_invalid_arg_error)?.to_vec();
        let y_vec = y.as_slice().map_err(to_py_invalid_arg_error)?.to_vec();
        let uw_vec: Option<Vec<f64>> = custom_weights
            .map(|uw| {
                uw.as_slice()
                    .map(|s| s.to_vec())
                    .map_err(to_py_invalid_arg_error)
            })
            .transpose()?;

        // Clone the pre-built builder for this fit call
        let builder = self.builder.clone();

        // 2. Release GIL
        let result = py.detach(move || {
            let model = binding_support::build_batch(builder, uw_vec)?;
            binding_support::map_lowess_result(model.fit(&x_vec, &y_vec))
        });

        // 3. Handle result (Back with GIL)
        match result {
            Ok(inner) => Ok(PyLowessResult { inner }),
            Err(e) => Err(to_py_error(e)),
        }
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
    m.add_class::<PyOnlineOutput>()?;
    m.add_class::<PyLowess>()?;
    m.add_class::<PyStreamingLowess>()?;
    m.add_class::<PyOnlineLowess>()?;
    Ok(())
}
