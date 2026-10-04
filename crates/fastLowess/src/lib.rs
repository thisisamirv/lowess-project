//! # Fast LOWESS (Locally Weighted Scatterplot Smoothing)
//!
//! The fastest, most robust, and most feature-complete language-agnostic
//! LOWESS (Locally Weighted Scatterplot Smoothing) implementation for **Rust**,
//! **Python**, and **R**.
//!
//! ## What is LOWESS?
//!
//! LOWESS (Locally Weighted Scatterplot Smoothing) is a nonparametric regression
//! method that fits smooth curves through scatter plots. At each point, it fits
//! a weighted polynomial (typically linear) using nearby data points, with weights
//! decreasing smoothly with distance. This creates flexible, data-adaptive curves
//! without assuming a global functional form.
//!
//! ## Documentation
//!
//! The [`doc`] module contains the full user guide, browsable on [docs.rs](https://docs.rs/fastLowess):
//!
//! - **Getting Started**
//!   - [Concepts](doc::introduction::concepts)
//!   - [Installation](doc::introduction::installation)
//!   - [Quick Start](doc::introduction::quickstart)
//! - **API Reference**
//!   - [Batch API](doc::api)
//!   - [Streaming API](doc::api::streaming)
//!   - [Online API](doc::api::online)
//! - **Adapters**
//!   - [Choosing an Adapter](doc::guide::adapter_choice)
#![cfg_attr(feature = "gpu", doc = "  - [GPU Backend](doc::advanced::gpu_backend)")]
//! - **Analysis**
//!   - [Intervals](doc::guide::intervals) (analytic or residual-bootstrap via `.intervals(...)`)
//!   - [Cross-Validation](doc::guide::cross_validation)
//!   - [Prediction](doc::guide::predict)
//! - **Customization**
//!   - [Kernels](doc::weighting::kernels)
//!   - [Robustness](doc::weighting::robustness)
//!   - [Scaling](doc::weighting::scaling)
//!   - [Custom Weights](doc::weighting::custom_weights)
//!   - [Boundary](doc::advanced::boundary)
//!   - [Merge Strategies](doc::advanced::merge)
//! - **Use Cases**
//!   - [Genomics](doc::use_case::genomics)
//!   - [Time Series](doc::use_case::time_series)
//!   - [Real-Time](doc::use_case::real_time)
//! - **News**
//!   - [Release Notes](doc::news)
//!
//! ## Quick Start (Batch)
//!
//! ### Typical Use
//!
//! ```rust
//! use fastLowess::prelude::*;
//! use ndarray::Array1;
//!
//! let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
//! let y = Array1::from_vec(vec![2.0, 4.1, 5.9, 8.2, 9.8]);
//!
//! // Build the model with parallel execution (default)
//! let model = Lowess::new()
//!     .fraction(0.5)      // Use 50% of data for each local fit
//!     .iterations(3)      // 3 robustness iterations
//!     .build()?;
//!
//! // Fit the model to the data
//! let result = model.fit(&x, &y)?;
//!
//! println!("LOWESS result:\n{}", result);
//! # Result::<(), LowessError>::Ok(())
//! ```
//!
//! ```text
//! Summary:
//!   Data points: 5
//!   Fraction: 0.5
//!
//! Smoothed Data:
//!        X     Y_smooth
//!   --------------------
//!     1.00     2.00000
//!     2.00     4.10000
//!     3.00     5.90000
//!     4.00     8.20000
//!     5.00     9.80000
//! ```
//!
//! ### Full Features
//!
//! ```rust
//! use fastLowess::prelude::*;
//! use ndarray::Array1;
//!
//! let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
//! let y = Array1::from_vec(vec![2.1, 3.8, 6.2, 7.9, 10.3, 11.8, 14.1, 15.7]);
//!
//! // Build model with all features enabled
//! let model = Lowess::new()
//!     .fraction(0.5)                                  // Use 50% of data for each local fit
//!     .iterations(3)                                  // 3 robustness iterations
//!     .weight_function("tricube")                     // Kernel function
//!     .robustness_method("bisquare")                  // Outlier handling
//!     .delta(0.01)                                    // Interpolation optimization
//!     .zero_weight_fallback("use_local_mean")         // Fallback policy
//!     .boundary_policy("extend")                      // Boundary handling policy
//!     .scaling_method("mad")                          // Robust scale estimation
//!     .auto_converge(1e-6)                            // Auto-convergence threshold
//!     .missing("error")                               // Reject non-finite (NaN/Inf) input
//!     .parallel(true)                                 // Rayon-parallel execution (default)
//!     .backend("cpu")                                 // Execution backend
//!     .outputs([
//!         "se",                                       // Standard errors
//!         "diagnostics",                              // Fit quality metrics
//!         "residuals",                                // Include residuals
//!         "weights",                                  // Include robustness weights
//!         "derivative",                               // Include per-point local slope
//!         "sorted"                                    // Sort output ascending by x
//!     ])
//!     .intervals(IntervalsBuilder::new()
//!         .confidence(0.95)                           // 95% confidence intervals
//!         .prediction(0.95)                           // 95% prediction intervals
//!         .bootstrap(1000)                            // Residual bootstrap instead of analytic intervals
//!     )
//!     .cv(CVBuilder::new()
//!         .method("kfold")                            // CV method: "kfold" or "loocv"
//!         .k(5)                                       // Number of folds for k-fold CV
//!         .fraction(vec![0.3, 0.7])                   // Candidate bandwidth fractions to evaluate
//!     )
//!     .seed(123)                                      // Shared CV and bootstrap seed
//!     .retain_model(true)                             // Retain state for out-of-sample predict()
//!     .custom_weights(vec![1.0; 8])                   // Per-observation case weights
//!     .build()?;
//!
//! let result = model.fit(&x, &y)?;
//! println!("LOWESS result:\n{}", result);
//! # Result::<(), LowessError>::Ok(())
//! ```
//!
//! ```text
//! Summary:
//!   Data points: 8
//!   Fraction: 0.5
//!   Robustness: Applied
//!
//! LOWESS Diagnostics:
//!   RMSE:         0.191925
//!   MAE:          0.181676
//!   R2:           0.998205
//!   Residual SD:  0.297750
//!   Effective DF: 8.00
//!   AIC:          -10.41
//!   AICc:         inf
//!
//! Smoothed Data:
//!        X     Y_smooth      Std_Err   Conf_Lower   Conf_Upper   Pred_Lower   Pred_Upper     Residual Rob_Weight
//!   ----------------------------------------------------------------------------------------------------------------
//!     1.00     2.01963     0.389365     1.256476     2.782788     1.058911     2.980353     0.080368     1.0000
//!     2.00     4.00251     0.345447     3.325438     4.679589     3.108641     4.896386    -0.202513     1.0000
//!     3.00     5.99959     0.423339     5.169846     6.829335     4.985168     7.014013     0.200410     1.0000
//!     4.00     8.09859     0.489473     7.139224     9.057960     6.975666     9.221518    -0.198592     1.0000
//!     5.00    10.03881     0.551687     8.957506    11.120118     8.810073    11.267551     0.261188     1.0000
//!     6.00    12.02872     0.539259    10.971775    13.085672    10.821364    13.236083    -0.228723     1.0000
//!     7.00    13.89828     0.371149    13.170829    14.625733    12.965670    14.830892     0.201719     1.0000
//!     8.00    15.77990     0.408300    14.979631    16.580167    14.789441    16.770356    -0.079899     1.0000
//! ```
//!
//! ### Result and Error Handling
//!
//! The `fit` method returns a `Result<LowessResult<T>, LowessError>`.
//!
//! - **`Ok(LowessResult<T>)`**: Contains the smoothed data and diagnostics.
//! - **`Err(LowessError)`**: Indicates a failure (e.g., mismatched input lengths, insufficient data).
//!
//! The `?` operator is idiomatic:
//!
//! ```rust
//! use fastLowess::prelude::*;
//! # let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! # let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];
//!
//! let model = Lowess::new().build()?;
//!
//! let result = model.fit(&x, &y)?;
//! // or to be more explicit:
//! // let result: LowessResult<f64> = model.fit(&x, &y)?;
//! # Result::<(), LowessError>::Ok(())
//! ```
//!
//! But you can also handle results explicitly:
//!
//! ```rust
//! use fastLowess::prelude::*;
//! # let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! # let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];
//!
//! let model = Lowess::new().build()?;
//!
//! match model.fit(&x, &y) {
//!     Ok(result) => {
//!         // result is LowessResult<f64>
//!         println!("Smoothed: {:?}", result.y);
//!     }
//!     Err(e) => {
//!         // e is LowessError
//!         eprintln!("Fitting failed: {}", e);
//!     }
//! }
//! # Result::<(), LowessError>::Ok(())
//! ```
//!
//! ### ndarray Integration
//!
//! `fastLowess` supports [ndarray](https://docs.rs/ndarray) natively, allowing for zero-copy
//! data passing and efficient numerical operations.
//!
//! ```rust
//! use fastLowess::prelude::*;
//! use ndarray::Array1;
//!
//! // Data as ndarray types
//! let x = Array1::from_vec((0..100).map(|i| i as f64 * 0.1).collect());
//! let y = Array1::from_elem(100, 1.0); // Replace with real data
//!
//! let model = Lowess::new().build()?;
//!
//! // fit() accepts &Array1<f64>, &[f64], or Vec<f64>
//! let result = model.fit(&x, &y)?;
//!
//! // result.y is an Array1<f64>
//! let smoothed_values = result.y;
//! # Result::<(), LowessError>::Ok(())
//! ```
//!
//! **Benefits:**
//! - **Zero-copy**: Pass data directly from your numerical pipeline.
//! - **Consistency**: If your project already uses `ndarray`, `fastLowess` fits right in.
//! - **Performance**: Optimized internal operations using `ndarray` primitives.
//!
//! ### Predict
//!
//! Retain the fitted Batch model to predict at new points. The prediction builder
//! controls optional outputs, intervals, and extrapolation limits:
//!
//! ```rust
//! use fastLowess::prelude::*;
//! use ndarray::Array1;
//!
//! let x = Array1::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
//! let y = Array1::from_vec(vec![2.1, 3.8, 6.2, 7.9, 10.3, 11.8, 14.1, 15.7]);
//! let model = Lowess::new()
//!     .fraction(0.5)
//!     .retain_model(true)                         // Required for out-of-sample prediction
//!     .build()?;
//! let fitted = model.fit(&x, &y)?;
//!
//! let predict = Predict::new()
//!     .outputs(["se", "derivative"])              // Standard errors and local slopes
//!     .intervals(IntervalsBuilder::new()
//!         .confidence(0.95)                       // Bounds for the mean response
//!         .prediction(0.95)                       // Bounds for a new observation
//!         .bootstrap(1000)                        // Residual-bootstrap refits
//!     )
//!     .seed(42)                                   // Reproducible bootstrap draws
//!     .extrapolation("linear")                    // Extend beyond the training range
//!     .max_extrapolation_distance(2.0)            // Limit distance outside that range
//!     .max_neighbor_distance(10.0)                // Reject sparse neighborhoods
//!     .build()?;
//! let prediction = predict.call(&fitted, &[2.5, 8.5])?;
//! println!("{prediction:#?}");
//! # assert_eq!(prediction.y.len(), 2);
//! # assert!(prediction.standard_errors.is_some());
//! # assert!(prediction.confidence_lower.is_some());
//! # assert!(prediction.prediction_upper.is_some());
//! # assert!(prediction.derivative.is_some());
//! # Result::<(), LowessError>::Ok(())
//! ```
//!
//! ## Quick Start (Streaming)
//!
//! ### Typical Use
//!
//! Process a dataset in chunks, then flush the points retained for overlap:
//!
//! ```rust
//! use fastLowess::prelude::*;
//!
//! let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
//! let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
//! let mut model = StreamingLowess::new()
//!     .fraction(0.5)
//!     .chunk_size(10)
//!     .overlap(2)
//!     .build()?;
//!
//! let mut emitted = 0;
//! for (xs, ys) in x.chunks(10).zip(y.chunks(10)) {
//!     emitted += model.process_chunk(xs, ys)?.y.len();
//! }
//! emitted += model.finalize()?.y.len();
//! println!("Smoothed {emitted} points across two chunks");
//! # assert_eq!(emitted, x.len());
//! # Result::<(), LowessError>::Ok(())
//! ```
//!
//! ```text
//! Smoothed 20 points across two chunks
//! ```
//!
//! ### Full Features
//!
//! Configure uncertainty estimates and optional outputs per chunk:
//! Common smoothing controls are shown explicitly; CV, case weights, retained prediction,
//! and sorted output remain Batch-only options.
//!
//! ```rust
//! use fastLowess::prelude::*;
//!
//! let x: Vec<f64> = (0..20).map(|i| i as f64 * 0.2).collect();
//! let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1 * (xi * 5.0).sin()).collect();
//! let mut model = StreamingLowess::new()
//!     .fraction(0.6)                          // Local smoothing span
//!     .iterations(1)                          // Robustness iterations
//!     .weight_function("tricube")             // Kernel function
//!     .robustness_method("bisquare")          // Outlier downweighting
//!     .delta(0.01)                            // Interpolation optimization
//!     .zero_weight_fallback("use_local_mean") // Zero-weight neighborhood fallback
//!     .boundary_policy("extend")              // Boundary handling
//!     .scaling_method("mad")                  // Robust residual scale
//!     .auto_converge(1e-6)                    // Robustness convergence tolerance
//!     .missing("error")                       // Reject non-finite observations
//!     .chunk_size(10)                         // Points per input chunk
//!     .overlap(2)                             // Points retained between chunks
//!     .merge_strategy("weighted_average")     // Blend estimates in the overlap
//!     .parallel(true)                         // Parallel CPU fits and bootstrap refits
//!     .outputs([
//!         "se",                               // Standard errors
//!         "diagnostics",                      // Cumulative fit diagnostics
//!         "residuals",                        // Observed minus fitted values
//!         "weights",                          // Final robustness weights
//!         "derivative"                        // Local slope at each point
//!     ])
//!     .intervals(IntervalsBuilder::new()
//!         .confidence(0.95)                   // 95% confidence intervals
//!         .prediction(0.95)                   // 95% prediction intervals
//!         .bootstrap(20)                      // Residual-bootstrap refits per chunk
//!     )
//!     .seed(7)                                // Reproducible bootstrap draws
//!     .build()?;
//!
//! let first = model.process_chunk(&x[..10], &y[..10])?;
//! # assert!(first.standard_errors.is_some());
//! # assert!(first.confidence_lower.is_some());
//! # assert!(first.prediction_upper.is_some());
//! # assert!(first.derivative.is_some());
//! # assert!(first.diagnostics.is_some());
//! # assert!(first.residuals.is_some());
//! # assert!(first.robustness_weights.is_some());
//! let second = model.process_chunk(&x[10..], &y[10..])?;
//! let final_chunk = model.finalize()?;
//! let emitted = first.y.len() + second.y.len() + final_chunk.y.len();
//! println!("Streaming intervals: {emitted} points");
//! # assert_eq!(emitted, x.len());
//! # Result::<(), LowessError>::Ok(())
//! ```
//!
//! ```text
//! Streaming intervals: 20 points
//! ```
//!
//! ### Result and Error Handling
//!
//! `process_chunk()` and `finalize()` return `Result<LowessResult<T>, LowessError>`.
//! A mismatched chunk is rejected without silently dropping points:
//!
//! ```rust
//! use fastLowess::prelude::*;
//!
//! let mut model = StreamingLowess::new().chunk_size(10).overlap(2).build()?;
//! let invalid = model.process_chunk(&[1.0, 2.0], &[3.0]);
//! assert!(invalid.is_err());
//! let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
//! let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi).collect();
//! let chunk = model.process_chunk(&x, &y)?;
//! let tail = model.finalize()?;
//! println!("Recovered {} points", chunk.y.len() + tail.y.len());
//! # Result::<(), LowessError>::Ok(())
//! ```
//!
//! ```text
//! Recovered 10 points
//! ```
//!
//! ### ndarray Integration
//!
//! Contiguous ndarray arrays can be passed chunk by chunk without copying:
//!
//! ```rust
//! use fastLowess::prelude::*;
//! use ndarray::Array1;
//!
//! let x = Array1::from_vec((0..20).map(|i| i as f64).collect());
//! let y = x.mapv(|xi| 2.0 * xi + 1.0);
//! let mut model = StreamingLowess::new().chunk_size(10).overlap(2).build()?;
//! let mut emitted = 0;
//! for (xs, ys) in x.as_slice().unwrap().chunks(10).zip(y.as_slice().unwrap().chunks(10)) {
//!     emitted += model.process_chunk(xs, ys)?.y.len();
//! }
//! emitted += model.finalize()?.y.len();
//! println!("Streaming ndarray points: {emitted}");
//! # assert_eq!(emitted, x.len());
//! # Result::<(), LowessError>::Ok(())
//! ```
//!
//! ```text
//! Streaming ndarray points: 20
//! ```
//!
//! ## Quick Start (Online)
//!
//! ### Typical Use
//!
//! Add points to a sliding window; updates begin once `min_points` is reached:
//!
//! ```rust
//! use fastLowess::prelude::*;
//!
//! let x = [1.0, 2.0, 3.0, 4.0, 5.0];
//! let y = [2.0, 4.0, 6.0, 8.0, 10.0];
//! let mut model = OnlineLowess::new()
//!     .fraction(0.5)
//!     .window_capacity(5)
//!     .min_points(3)
//!     .build()?;
//!
//! let mut updates = 0;
//! let mut latest = None;
//! for (&xi, &yi) in x.iter().zip(&y) {
//!     if let Some(output) = model.add_point(xi, yi)? {
//!         updates += 1;
//!         latest = Some(output.y);
//!     }
//! }
//! let latest = latest.expect("the window has enough points");
//! println!("Online updates: {updates}; latest estimate: {latest:.1}");
//! # assert_eq!(updates, 3);
//! # assert!((latest - 10.0).abs() < 1e-8);
//! # Result::<(), LowessError>::Ok(())
//! ```
//!
//! ```text
//! Online updates: 3; latest estimate: 10.0
//! ```
//!
//! ### Full Features
//!
//! Full updates support robust fitting and uncertainty estimates for each latest point:
//! Positive delta and convergence controls require full-update mode, and convergence
//! additionally requires robustness iterations. CV, case weights, and retained prediction
//! are Batch-only; Online reports the latest point rather than cumulative diagnostics.
//!
//! ```rust
//! use fastLowess::prelude::*;
//!
//! let mut model = OnlineLowess::new()
//!     .fraction(0.7)                          // Local smoothing span
//!     .iterations(1)                          // Robustness iterations
//!     .weight_function("tricube")             // Kernel function
//!     .robustness_method("bisquare")          // Outlier downweighting
//!     .delta(0.01)                            // Positive delta requires full updates
//!     .zero_weight_fallback("use_local_mean") // Zero-weight neighborhood fallback
//!     .boundary_policy("extend")              // Boundary handling
//!     .scaling_method("mad")                  // Robust residual scale
//!     .auto_converge(1e-6)                    // Full updates with robustness iterations
//!     .missing("error")                       // Reject non-finite observations
//!     .window_capacity(20)                    // Maximum sliding-window size
//!     .min_points(5)                          // Wait for five points before smoothing
//!     .update_mode("full")                    // Refit the whole window for intervals
//!     .outputs([
//!         "se",                               // Latest-point standard error
//!         "weights",                          // Latest-point robustness weight
//!         "derivative"                        // Latest-point local slope
//!     ])
//!     .intervals(IntervalsBuilder::new()
//!         .confidence(0.95)                   // 95% confidence interval
//!         .prediction(0.95)                   // 95% prediction interval
//!         .bootstrap(20)                      // Refit the current window 20 times
//!     )
//!     .seed(7)                                // Reproducible bootstrap draws
//!     .build()?;
//!
//! let mut updates = 0;
//! for i in 0..12 {
//!     let x = i as f64 * 0.2;
//!     if let Some(output) = model.add_point(x, x.sin() + 0.1 * (5.0 * x).sin())? {
//!         updates += 1;
//!         assert!(output.standard_error.is_some());
//!         assert!(output.confidence_lower.is_some());
//!         assert!(output.prediction_upper.is_some());
//!         assert!(output.derivative.is_some());
//!         assert!(output.robustness_weight.is_some());
//!     }
//! }
//! println!("Online full updates: {updates}");
//! # assert_eq!(updates, 8);
//! # Result::<(), LowessError>::Ok(())
//! ```
//!
//! ```text
//! Online full updates: 8
//! ```
//!
//! ### Result and Error Handling
//!
//! `build()` and `add_point()` return `Result`; `add_point()` returns `None` until
//! `min_points` is reached. Bulk updates also reject mismatched input lengths:
//!
//! ```rust
//! use fastLowess::prelude::*;
//!
//! let invalid = OnlineLowess::new()
//!     .intervals(IntervalsBuilder::new().confidence(0.95))
//!     .build();
//! assert!(matches!(invalid, Err(LowessError::StandardErrorRequiresFullUpdateMode)));
//!
//! let mut model = OnlineLowess::new().window_capacity(5).min_points(3).build()?;
//! let mismatched = model.add_points(&[1.0, 2.0][..], &[2.0][..]);
//! assert!(matches!(mismatched, Err(LowessError::InvalidInput(_))));
//! let outputs = model.add_points(&[1.0, 2.0, 3.0][..], &[2.0, 4.0, 6.0][..])?;
//! println!("Ready online outputs: {}", outputs.iter().flatten().count());
//! # Result::<(), LowessError>::Ok(())
//! ```
//!
//! ```text
//! Ready online outputs: 1
//! ```
//!
//! ### ndarray Integration
//!
//! `add_points()` accepts ndarray arrays and returns one optional update per point:
//!
//! ```rust
//! use fastLowess::prelude::*;
//! use ndarray::Array1;
//!
//! let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
//! let y = x.mapv(|xi| 2.0 * xi);
//! let mut model = OnlineLowess::new().window_capacity(5).min_points(3).build()?;
//! let outputs = model.add_points(&x, &y)?;
//! let ready = outputs.iter().flatten().count();
//! println!("Online ndarray updates: {ready}");
//! # assert_eq!(ready, 3);
//! # Result::<(), LowessError>::Ok(())
//! ```
//!
//! ```text
//! Online ndarray updates: 3
//! ```
//!
//!
//! ## References
//!
//! - Cleveland, W. S. (1979). "Robust Locally Weighted Regression and Smoothing Scatterplots"
//! - Cleveland, W. S. (1981). "LOWESS: A Program for Smoothing Scatterplots by Robust Locally Weighted Regression"
// ## srrstats Compliance for rOpenSci Statistical Software Review
//
// @srrstats {G1.0} Statistical literature references documented above (Cleveland 1979, 1981).
// @srrstats {G1.5} Parallel execution via Rayon for multi-threaded performance.
// @srrstats {G3.0} ndarray integration for zero-copy data passing and numerical operations.

//! ## License
//!
//! See the repository for license information and contribution guidelines.

#![allow(non_snake_case)]

/// User guide and worked examples (visible on docs.rs).
#[cfg(doc)]
pub mod doc;

/// GPU-accelerated execution engine.
#[cfg(feature = "gpu")]
pub mod gpu {
    pub use crate::engine::gpu::{cross_validate_gpu, fit_pass_gpu};
}

// Layer 4: Evaluation - post-processing and diagnostics.
mod evaluation;

// Layer 5: Engine - orchestration and execution control.
mod engine;

// Layer 6: Adapters - execution mode adapters.
mod adapters;

// High-level fluent API for LOWESS smoothing.
mod api;

// Input data handling.
mod input;

// Shared helpers for language bindings (only needed for dev/binding builds).
#[cfg(feature = "dev")]
mod binding_support;

// Standard fastLowess prelude.
pub mod prelude {
    pub use crate::api::{Lowess, OnlineLowess, StreamingLowess};
    pub use lowess::internals::adapters::predict::Predict;
    pub use lowess::internals::engine::executor::LowessResult;
    pub use lowess::internals::evaluation::cv::CVBuilder;
    pub use lowess::internals::evaluation::intervals::IntervalsBuilder;
    pub use lowess::internals::primitives::errors::LowessError;
}

// Internal modules for development and testing.
//
// This module re-exports internal modules for development and testing purposes.
// It is only available with the `dev` feature enabled.
#[cfg(feature = "dev")]
pub mod internals {
    pub type LowessBuilder<T, Mode = lowess::internals::api::BatchMode> =
        lowess::internals::api::LowessBuilder<T, Mode>;

    pub mod engine {
        pub use crate::engine::*;
    }
    pub mod evaluation {
        pub use crate::evaluation::*;
    }
    pub mod adapters {
        pub use crate::adapters::*;
    }
    pub mod api {
        pub use crate::api::*;
    }
    pub mod binding_support {
        pub use crate::binding_support::*;
    }
}
