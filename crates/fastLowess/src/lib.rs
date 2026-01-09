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
//! > 📚 **Full Documentation**: [lowess.readthedocs.io](https://lowess.readthedocs.io/)
//! >
//! > Comprehensive guides, API references, and tutorials for Rust, Python, and R.
//!
//! ## Quick Start
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
//!     .adapter(Batch)     // Parallel by default
//!     .build()?;
//!
//! // Fit the model to the data
//! let result = model.fit(&x, &y)?;
//!
//! println!("{}", result);
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
//!     .fraction(0.5)                                   // Use 50% of data for each local fit
//!     .iterations(3)                                   // 3 robustness iterations
//!     .weight_function(Tricube)                        // Kernel function
//!     .robustness_method(Bisquare)                     // Outlier handling
//!     .delta(0.01)                                     // Interpolation optimization
//!     .zero_weight_fallback(UseLocalMean)              // Fallback policy
//!     .boundary_policy(Extend)                         // Boundary handling policy
//!     .scaling_method(MAD)                             // Robust scale estimation
//!     .auto_converge(1e-6)                             // Auto-convergence threshold
//!     .confidence_intervals(0.95)                      // 95% confidence intervals
//!     .prediction_intervals(0.95)                      // 95% prediction intervals
//!     .return_diagnostics()                            // Fit quality metrics
//!     .return_residuals()                              // Include residuals
//!     .return_robustness_weights()                     // Include robustness weights
//!     .cross_validate(KFold(5, &[0.3, 0.7]).seed(123)) // K-fold CV with 5 folds and 2 fraction options
//!     .adapter(Batch)                                  // Batch adapter
//!     .parallel(true)                                  // Enable parallel execution
//!     .backend(CPU)                                    // Default to CPU backend, please read the docs for more information
//!     .build()?;
//!
//! let result = model.fit(&x, &y)?;
//! println!("{}", result);
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
//!   R²:           0.998205
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
//! let model = Lowess::new().adapter(Batch).build()?;
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
//! let model = Lowess::new().adapter(Batch).build()?;
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
//! let model = Lowess::new().adapter(Batch).build()?;
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
//!
//! ## References
//!
//! - Cleveland, W. S. (1979). "Robust Locally Weighted Regression and Smoothing Scatterplots"
//! - Cleveland, W. S. (1981). "LOWESS: A Program for Smoothing Scatterplots by Robust Locally Weighted Regression"
//!
//! ## License
//!
//! See the repository for license information and contribution guidelines.

#![allow(non_snake_case)]

/// GPU-accelerated execution engine.
#[cfg(feature = "gpu")]
pub mod gpu {
    pub use crate::engine::gpu::fit_pass_gpu;
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

// Standard fastLowess prelude.
pub mod prelude {
    pub use crate::api::{
        Adapter::{Batch, Online, Streaming},
        Backend::{CPU, GPU},
        BoundaryPolicy::{Extend, NoBoundary, Reflect, Zero},
        KFold, LOOCV, LowessBuilder as Lowess, LowessError, LowessResult,
        MergeStrategy::{Average, TakeFirst, WeightedAverage},
        RobustnessMethod::{Bisquare, Huber, Talwar},
        ScalingMethod::{MAD, MAR},
        UpdateMode::{Full, Incremental},
        WeightFunction::{Biweight, Cosine, Epanechnikov, Gaussian, Triangle, Tricube, Uniform},
        ZeroWeightFallback::{ReturnNone, ReturnOriginal, UseLocalMean},
    };
}

// Internal modules for development and testing.
//
// This module re-exports internal modules for development and testing purposes.
// It is only available with the `dev` feature enabled.
#[cfg(feature = "dev")]
pub mod internals {
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
}
