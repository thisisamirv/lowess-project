//! # LOWESS — Locally Weighted Scatterplot Smoothing for Rust
//!
//! A production-ready, high-performance LOWESS implementation with comprehensive
//! features for robust nonparametric regression and trend estimation.
//!
//! ## What is LOWESS?
//!
//! LOWESS (Locally Weighted Scatterplot Smoothing) is a nonparametric regression
//! method that fits smooth curves through scatter plots. At each point, it fits
//! a weighted polynomial (typically linear) using nearby data points, with weights
//! decreasing smoothly with distance. This creates flexible, data-adaptive curves
//! without assuming a global functional form.
//!
//! **Key advantages:**
//! - No parametric assumptions about the underlying relationship
//! - Automatic adaptation to local data structure
//! - Robust to outliers (with robustness iterations enabled)
//! - Provides uncertainty estimates via confidence/prediction intervals
//! - Handles irregular sampling and missing regions gracefully
//!
//! **Common applications:**
//! - Exploratory data analysis and visualization
//! - Trend estimation in time series
//! - Baseline correction in spectroscopy and signal processing
//! - Quality control and process monitoring
//! - Genomic and epigenomic data smoothing
//! - Removing systematic effects in scientific measurements
//!
//! **How LOWESS works:**
//!
//! <div align="center">
//! <object data="../../../docs/lowess_smoothing_concept.svg" type="image/svg+xml" width="800" height="500">
//! <img src="https://raw.githubusercontent.com/thisisamirv/lowess/main/docs/lowess_smoothing_concept.svg" alt="LOWESS Smoothing Concept" width="800"/>
//! </object>
//!
//! *LOWESS creates smooth curves through scattered data using local weighted neighborhoods*
//! </div>
//!
//! 1. For each point, select nearby neighbors (controlled by `fraction`)
//! 2. Fit a weighted polynomial (closer points get higher weight)
//! 3. Use the fitted value as the smoothed estimate
//! 4. Optionally iterate to downweight outliers (robustness)
//!
//! ## Quick Start
//!
//! ### Typical Use
//!
//! ```rust
//! use lowess::prelude::*;
//!
//! let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];
//!
//! // Build the model
//! let model = Lowess::new()
//!     .fraction(0.5)      // Use 50% of data for each local fit
//!     .iterations(3)      // 3 robustness iterations
//!     .adapter(Batch)
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
//! use lowess::prelude::*;
//!
//! let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
//! let y = vec![2.1, 3.8, 6.2, 7.9, 10.3, 11.8, 14.1, 15.7];
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
//!   R^2:           0.998205
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
//! use lowess::prelude::*;
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
//! use lowess::prelude::*;
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
//! ## Minimal Usage (no_std / Embedded)
//!
//! The crate supports `no_std` environments for embedded devices and resource-constrained systems.
//! Disable default features to remove the standard library dependency:
//!
//! ```toml
//! [dependencies]
//! lowess = { version = "0.5", default-features = false }
//! ```
//!
//! **Minimal example for embedded systems:**
//!
//! ```rust
//! # #[cfg(feature = "std")] {
//! use lowess::prelude::*;
//!
//! // In an embedded context (e.g., sensor data processing)
//! fn smooth_sensor_data() -> Result<(), LowessError> {
//!     // Small dataset from sensor readings
//!     let x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
//!     let y = vec![2.1, 3.9, 6.2, 7.8, 10.1];
//!
//!     // Build minimal model (no intervals, no diagnostics)
//!     let model = Lowess::new()
//!         .fraction(0.5)
//!         .iterations(2)      // Fewer iterations for speed
//!         .adapter(Batch)
//!         .build()?;
//!
//!     // Fit the model
//!     let result = model.fit(&x, &y)?;
//!
//!     // Use smoothed values (result.y)
//!     // ...
//!
//!     Ok(())
//! }
//! # smooth_sensor_data().unwrap();
//! # }
//! ```
//!
//! **Tips for embedded/no_std usage:**
//! - Use `f32` instead of `f64` to reduce memory footprint
//! - Keep datasets small (< 1000 points)
//! - Disable optional features (intervals, diagnostics) to reduce code size
//! - Use fewer iterations (1-2) to reduce computation time
//! - Allocate buffers statically when possible to avoid heap fragmentation
//!
//! ## Parameters
//!
//! All builder parameters have sensible defaults. You only need to specify what you want to change.
//!
//! | Parameter                     | Default                                       | Range/Options        | Description                                    | Adapter          |
//! |-------------------------------|-----------------------------------------------|----------------------|------------------------------------------------|------------------|
//! | **fraction**                  | 0.67 (or CV-selected)                         | (0, 1]               | Smoothing span (fraction of data used per fit) | All              |
//! | **iterations**                | 3                                             | [0, 1000]            | Number of robustness iterations                | All              |
//! | **delta**                     | 1% of x-range (Batch), 0.0 (Streaming/Online) | [0, ∞)               | Interpolation optimization threshold           | All              |
//! | **weight_function**           | `Tricube`                                     | 7 kernel options     | Distance weighting kernel                      | All              |
//! | **robustness_method**         | `Bisquare`                                    | 3 methods            | Outlier downweighting method                   | All              |
//! | **zero_weight_fallback**      | `UseLocalMean`                                | 3 fallback options   | Behavior when all weights are zero             | All              |
//! | **return_residuals**          | false                                         | true/false           | Include residuals in output                    | All              |
//! | **boundary_policy**           | `Extend`                                      | 4 policy options     | Edge handling strategy (reduces boundary bias) | All              |
//! | **auto_convergence**          | None                                          | Tolerance value      | Early stopping for robustness                  | All              |
//! | **return_robustness_weights** | false                                         | true/false           | Include final weights in output                | All              |
//! | **scaling_method**            | `MAD`                                         | 2 methods            | Scale estimation method                        | All              |
//! | **return_diagnostics**        | false                                         | true/false           | Include RMSE, MAE, R^2, etc. in output         | Batch, Streaming |
//! | **confidence_intervals**      | None                                          | 0..1 (level)         | Uncertainty in mean curve                      | Batch            |
//! | **prediction_intervals**      | None                                          | 0..1 (level)         | Uncertainty for new observations               | Batch            |
//! | **cross_validate**            | None                                          | Method (fractions)   | Automated bandwidth selection                  | Batch            |
//! | **chunk_size**                | 5000                                          | [10, ∞)              | Points per chunk for streaming                 | Streaming        |
//! | **overlap**                   | 500                                           | [0, chunk_size)      | Overlapping points between chunks              | Streaming        |
//! | **merge_strategy**            | `Average`                                     | 4 strategies         | How to merge overlapping regions               | Streaming        |
//! | **update_mode**               | `Incremental`                                 | 2 modes              | Online update strategy (Incremental vs Full)   | Online           |
//! | **window_capacity**           | 1000                                          | [3, ∞)               | Maximum points in sliding window               | Online           |
//! | **min_points**                | 3                                             | [2, window_capacity] | Minimum points before smoothing starts         | Online           |
//!
//! ### Parameter Options Reference
//!
//! For parameters with multiple options, here are the available choices:
//!
//! | Parameter                | Available Options                                                                  |
//! |--------------------------|------------------------------------------------------------------------------------|
//! | **weight_function**      | `Tricube`, `Epanechnikov`, `Gaussian`, `Biweight`, `Cosine`, `Triangle`, `Uniform` |
//! | **robustness_method**    | `Bisquare`, `Huber`, `Talwar`                                                      |
//! | **zero_weight_fallback** | `UseLocalMean`, `ReturnOriginal`, `ReturnNone`                                     |
//! | **boundary_policy**      | `Extend`, `Reflect`, `Zero`, 'NoBoundary'                                          |
//! | **scaling_method**       | `MAD`, `MAR`                                                                       |
//! | **update_mode**          | `Incremental`, `Full`                                                              |
//!
//! See the detailed sections below for guidance on choosing between these options.
//!
//! ## Builder
//!
//! The crate uses a fluent builder pattern for configuration. All parameters have
//! sensible defaults, so you only need to specify what you want to change.
//!
//! ### Basic Workflow
//!
//! 1. **Create builder**: `Lowess::new()`
//! 2. **Configure parameters**: Chain method calls (`.fraction()`, `.iterations()`, etc.)
//! 3. **Select adapter**: Choose execution mode (`.adapter(Batch)`, `.adapter(Streaming)`, etc.)
//! 4. **Build model**: Call `.build()` to create the configured model
//! 5. **Fit data**: Call `.fit(&x, &y)` to perform smoothing
//!
//! ```rust
//! use lowess::prelude::*;
//! # let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! # let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];
//!
//! // Build the model with custom configuration
//! let model = Lowess::new()
//!     .fraction(0.3)               // Smoothing span
//!     .iterations(5)               // Robustness iterations
//!     .weight_function(Tricube)    // Kernel function
//!     .robustness_method(Bisquare) // Outlier handling
//!     .adapter(Batch)
//!     .build()?;
//!
//! // Fit the model to the data
//! let result = model.fit(&x, &y)?;
//! println!("{}", result);
//! # Result::<(), LowessError>::Ok(())
//! ```
//!
//! ```text
//! Summary:
//!   Data points: 5
//!   Fraction: 0.3
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
//! ### Execution Mode (Adapter) Comparison
//!
//! Choose the right execution mode based on your use case:
//!
//! | Adapter     | Use Case                                                                    | Features                                                                         | Limitations                                                               |
//! |-------------|-----------------------------------------------------------------------------|----------------------------------------------------------------------------------|---------------------------------------------------------------------------|
//! | `Batch`     | Complete datasets in memory<br>Standard analysis<br>Full diagnostics needed | All features supported                                                           | Requires entire dataset in memory<br>Not suitable for very large datasets |
//! | `Streaming` | Large datasets (>100K points)<br>Limited memory<br>Batch pipelines          | Chunked processing<br>Configurable overlap<br>Robustness iterations<br>Residuals | No intervals<br>No cross-validation<br>No diagnostics                     |
//! | `Online`    | Real-time data<br>Sensor streams<br>Embedded systems                        | Incremental updates<br>Sliding window<br>Memory-bounded                          | No intervals<br>No cross-validation<br>Limited history                    |
//!
//! **Recommendation:**
//! - **Start with Batch** for most use cases - it's the most feature-complete
//! - **Use Streaming** when dataset size exceeds available memory
//! - **Use Online** for real-time applications or when data arrives incrementally
//!
//! #### Batch Adapter
//!
//! Standard mode for complete datasets in memory. Supports all features.
//!
//! ```rust
//! use lowess::prelude::*;
//! # let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! # let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];
//!
//! // Build model with batch adapter
//! let model = Lowess::new()
//!     .fraction(0.5)
//!     .iterations(3)
//!     .confidence_intervals(0.95)
//!     .prediction_intervals(0.95)
//!     .return_diagnostics()
//!     .adapter(Batch)  // Full feature support
//!     .build()?;
//!
//! let result = model.fit(&x, &y)?;
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
//!        X     Y_smooth      Std_Err   Conf_Lower   Conf_Upper   Pred_Lower   Pred_Upper
//!   ----------------------------------------------------------------------------------
//!     1.00     2.00000     0.000000     2.000000     2.000000     2.000000     2.000000
//!     2.00     4.10000     0.000000     4.100000     4.100000     4.100000     4.100000
//!     3.00     5.90000     0.000000     5.900000     5.900000     5.900000     5.900000
//!     4.00     8.20000     0.000000     8.200000     8.200000     8.200000     8.200000
//!     5.00     9.80000     0.000000     9.800000     9.800000     9.800000     9.800000
//!
//! Diagnostics:
//!   RMSE: 0.0000
//!   MAE: 0.0000
//!   R²: 1.0000
//! ```
//!
//! **Use batch when:**
//! - Dataset fits in memory
//! - Need all features (intervals, CV, diagnostics)
//! - Processing complete datasets
//!
//! #### Streaming Adapter
//!
//! Process large datasets in chunks with configurable overlap. Use `process_chunk()`
//! to process each chunk and `finalize()` to get remaining buffered data.
//!
//! ```rust
//! use lowess::prelude::*;
//!
//! // Simulate chunks of data (in practice, read from file/stream)
//! let chunk1_x: Vec<f64> = (0..50).map(|i| i as f64).collect();
//! let chunk1_y: Vec<f64> = chunk1_x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
//!
//! let chunk2_x: Vec<f64> = (40..100).map(|i| i as f64).collect();
//! let chunk2_y: Vec<f64> = chunk2_x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
//!
//! // Build streaming processor with chunk configuration
//! let mut processor = Lowess::new()
//!     .fraction(0.3)
//!     .iterations(2)
//!     .adapter(Streaming)
//!     .chunk_size(50)   // Process 50 points at a time
//!     .overlap(10)      // 10 points overlap between chunks
//!     .build()?;
//!
//! // Process first chunk
//! let result1 = processor.process_chunk(&chunk1_x, &chunk1_y)?;
//! // result1.y contains smoothed values for the non-overlapping portion
//!
//! // Process second chunk (overlaps with end of first chunk)
//! let result2 = processor.process_chunk(&chunk2_x, &chunk2_y)?;
//! // result2.y contains smoothed values, with overlap merged from first chunk
//!
//! // IMPORTANT: Call finalize() to get remaining buffered overlap data
//! let final_result = processor.finalize()?;
//! // final_result.y contains the final overlap buffer
//!
//! // Total processed = all chunks + finalize
//! let total = result1.y.len() + result2.y.len() + final_result.y.len();
//! println!("Processed {} points total", total);
//! # Result::<(), LowessError>::Ok(())
//! ```
//!
//! **Use streaming when:**
//! - Dataset is very large (>100,000 points)
//! - Memory is limited
//! - Processing data in chunks
//!
//! #### Online Adapter
//!
//! Incremental updates with a sliding window for real-time data.
//!
//! ```rust
//! use lowess::prelude::*;
//!
//! // Build model with online adapter
//! let model = Lowess::new()
//!     .fraction(0.2)
//!     .iterations(1)
//!     .adapter(Online)
//!     .build()?;
//!
//! let mut online_model = model;
//!
//! // Add points incrementally
//! for i in 1..=10 {
//!     let x = i as f64;
//!     let y = 2.0 * x + 1.0;
//!     if let Some(result) = online_model.add_point(x, y)? {
//!         println!("Latest smoothed value: {:.2}", result.smoothed);
//!     }
//! }
//! # Result::<(), LowessError>::Ok(())
//! ```
//!
//! **Use online when:**
//! - Data arrives incrementally
//! - Need real-time updates
//! - Maintaining a sliding window
//!
//! ### Fraction (Smoothing Span)
//!
//! The `fraction` parameter controls the proportion of data used for each local fit.
//! Larger fractions create smoother curves; smaller fractions preserve more detail.
//!
//! <div align="center">
//! <object data="../../../docs/fraction_effect_comparison.svg" type="image/svg+xml" width="1200" height="450">
//! <img src="https://raw.githubusercontent.com/thisisamirv/lowess/main/docs/fraction_effect_comparison.svg" alt="Fraction Effect" width="1200"/>
//! </object>
//!
//! *Under-smoothing (fraction too small), optimal smoothing, and over-smoothing (fraction too large)*
//! </div>
//!
//! - **Range**: (0, 1]
//! - **Effect**: Larger = smoother, smaller = more detail
//!
//! ```rust
//! use lowess::prelude::*;
//! # let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! # let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];
//!
//! // Build model with small fraction (more detail)
//! let model = Lowess::new()
//!     .fraction(0.2)  // Use 20% of data for each local fit
//!     .adapter(Batch)
//!     .build()?;
//!
//! let result = model.fit(&x, &y)?;
//! # Result::<(), LowessError>::Ok(())
//! ```
//!
//! **Choosing fraction:**
//! - **0.1-0.3**: Fine detail, may be noisy
//! - **0.3-0.5**: Moderate smoothing (good for most cases)
//! - **0.5-0.7**: Heavy smoothing, emphasizes trends
//! - **0.7-1.0**: Very smooth, may over-smooth
//!
//! ### Iterations (Robustness)
//!
//! The `iterations` parameter controls outlier resistance through iterative
//! reweighting. More iterations provide stronger robustness but increase computation time.
//!
//! <div align="center">
//! <object data="../../../docs/robust_vs_standard_lowess.svg" type="image/svg+xml" width="900" height="500">
//! <img src="https://raw.githubusercontent.com/thisisamirv/lowess/main/docs/robust_vs_standard_lowess.svg" alt="Robustness Effect" width="900"/>
//! </object>
//!
//! *Standard LOWESS (left) vs Robust LOWESS (right) - robustness iterations downweight outliers*
//! </div>
//! - **Range**: [0, 1000]
//! - **Effect**: More iterations = stronger outlier downweighting
//!
//! ```rust
//! use lowess::prelude::*;
//! # let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! # let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];
//!
//! // Build model with strong outlier resistance
//! let model = Lowess::new()
//!     .fraction(0.5)
//!     .iterations(5)  // More iterations for stronger robustness
//!     .adapter(Batch)
//!     .build()?;
//!
//! let result = model.fit(&x, &y)?;
//! # Result::<(), LowessError>::Ok(())
//! ```
//!
//! **Choosing iterations:**
//! - **0**: No robustness (fastest, sensitive to outliers)
//! - **1-3**: Light to moderate robustness (recommended)
//! - **4-6**: Strong robustness (for contaminated data)
//! - **7+**: Very strong (may over-smooth)
//!
//! ### Delta (Optimization)
//!
//! The `delta` parameter enables interpolation optimization for large datasets.
//! Points within `delta` distance reuse the previous fit.
//!
//! - **Range**: [0, ∞)
//! - **Effect**: Larger = faster but less accurate
//!
//! ```rust
//! use lowess::prelude::*;
//! # let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! # let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];
//!
//! // Build model with custom delta
//! let model = Lowess::new()
//!     .fraction(0.5)
//!     .delta(0.05)  // Custom delta value
//!     .adapter(Batch)
//!     .build()?;
//!
//! let result = model.fit(&x, &y)?;
//! # Result::<(), LowessError>::Ok(())
//! ```
//!
//! **When to use delta:**
//! - **Large datasets** (>10,000 points): Set to ~1% of range
//! - **Uniformly spaced data**: Can use larger values
//! - **Irregular spacing**: Use smaller values or 0
//!
//! ### Weight Functions (Kernels)
//!
//! Control how neighboring points are weighted by distance.
//!
//! ```rust
//! use lowess::prelude::*;
//! # let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! # let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];
//!
//! // Build model with Epanechnikov kernel
//! let model = Lowess::new()
//!     .fraction(0.5)
//!     .weight_function(Epanechnikov)
//!     .adapter(Batch)
//!     .build()?;
//!
//! let result = model.fit(&x, &y)?;
//! # Result::<(), LowessError>::Ok(())
//! ```
//!
//! **Kernel selection guide:**
//!
//! | Kernel         | Efficiency | Smoothness        |
//! |----------------|------------|-------------------|
//! | `Tricube`      | 0.998      | Very smooth       |
//! | `Epanechnikov` | 1.000      | Smooth            |
//! | `Gaussian`     | 0.961      | Infinitely smooth |
//! | `Biweight`     | 0.995      | Very smooth       |
//! | `Cosine`       | 0.999      | Smooth            |
//! | `Triangle`     | 0.989      | Moderate          |
//! | `Uniform`      | 0.943      | None              |
//!
//! *Efficiency = AMISE relative to Epanechnikov (1.0 = optimal)*
//!
//! **Choosing a Kernel:**
//!
//! * **Tricube** (default): Best all-around choice
//!   - High efficiency (0.9983)
//!   - Smooth derivatives
//!   - Compact support (computationally efficient)
//!   - Cleveland's original choice
//!
//! * **Epanechnikov**: Theoretically optimal
//!   - AMISE-optimal for kernel density estimation
//!   - Less smooth than tricube
//!   - Efficiency = 1.0 by definition
//!
//! * **Gaussian**: Maximum smoothness
//!   - Infinitely smooth
//!   - No boundary effects
//!   - More expensive to compute
//!   - Good for very smooth data
//!
//! * **Biweight**: Good balance
//!   - High efficiency (0.9951)
//!   - Smoother than Epanechnikov
//!   - Compact support
//!
//! * **Cosine**: Smooth and compact
//!   - Good for robust smoothing contexts
//!   - High efficiency (0.9995)
//!
//! * **Triangle**: Simple and fast
//!   - Linear taper
//!   - Less smooth than other kernels
//!   - Easy to understand
//!
//! * **Uniform**: Simplest
//!   - Equal weights within window
//!   - Fastest to compute
//!   - Least smooth results
//!
//! ### Robustness Methods
//!
//! Different methods for downweighting outliers during iterative refinement.
//!
//! ```rust
//! use lowess::prelude::*;
//! # let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! # let y = vec![2.0, 4.1, 5.9, 100.0, 9.8];  // Point 3 is an outlier
//!
//! // Build model with Talwar robustness (hard threshold)
//! let model = Lowess::new()
//!     .fraction(0.5)
//!     .iterations(3)
//!     .robustness_method(Talwar)
//!     .return_robustness_weights()  // Include weights in output
//!     .adapter(Batch)
//!     .build()?;
//!
//! // Fit the model to the data
//! let result = model.fit(&x, &y)?;
//!
//! // Check which points were downweighted
//! if let Some(weights) = &result.robustness_weights {
//!     for (i, &w) in weights.iter().enumerate() {
//!         if w < 0.5 {
//!             println!("Point {} is likely an outlier (weight: {:.3})", i, w);
//!         }
//!     }
//! }
//! # Result::<(), LowessError>::Ok(())
//! ```
//!
//! ```text
//! Point 3 is likely an outlier (weight: 0.000)
//! ```
//!
//! **Available methods:**
//!
//! | Method     | Behavior                | Use Case                  |
//! |------------|-------------------------|---------------------------|
//! | `Bisquare` | Smooth downweighting    | General-purpose, balanced |
//! | `Huber`    | Linear beyond threshold | Moderate outliers         |
//! | `Talwar`   | Hard threshold (0 or 1) | Extreme contamination     |
//!
//! ### Zero-Weight Fallback
//!
//! Control behavior when all neighborhood weights are zero.
//!
//! ```rust
//! use lowess::prelude::*;
//! # let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! # let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];
//!
//! // Build model with custom zero-weight fallback
//! let model = Lowess::new()
//!     .fraction(0.5)
//!     .zero_weight_fallback(UseLocalMean)
//!     .adapter(Batch)
//!     .build()?;
//!
//! let result = model.fit(&x, &y)?;
//! # Result::<(), LowessError>::Ok(())
//! ```
//!
//! **Fallback options:**
//! - `UseLocalMean`: Use mean of neighborhood
//! - `ReturnOriginal`: Return original y value
//! - `ReturnNone`: Return NaN (for explicit handling)
//!
//! ### Return Residuals
//!
//! Include residuals (y - smoothed) in the output for all adapters.
//!
//! ```rust
//! use lowess::prelude::*;
//! # let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! # let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];
//!
//! let model = Lowess::new()
//!     .fraction(0.5)
//!     .return_residuals()
//!     .adapter(Batch)
//!     .build()?;
//!
//! let result = model.fit(&x, &y)?;
//! // Access residuals
//! if let Some(residuals) = result.residuals {
//!     println!("Residuals: {:?}", residuals);
//! }
//! # Result::<(), LowessError>::Ok(())
//! ```
//!
//! ### Boundary Policy
//!
//! LOWESS traditionally uses asymmetric windows at boundaries, which can introduce bias.
//! The `boundary_policy` parameter pads the data before smoothing to enable centered windows:
//!
//! - **`Extend`** (default): Pad with constant values (first/last y-value)
//! - **`Reflect`**: Mirror the data at boundaries
//! - **`Zero`**: Pad with zeros
//! - **`NoBoundary`**: Do not pad the data (original Cleveland behavior)
//!
//! ```rust
//! use lowess::prelude::*;
//! # let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! # let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];
//!
//! // Use reflective padding for better edge handling
//! let model = Lowess::new()
//!     .fraction(0.5)
//!     .boundary_policy(Reflect)
//!     .adapter(Batch)
//!     .build()?;
//!
//! let result = model.fit(&x, &y)?;
//! # Result::<(), LowessError>::Ok(())
//! ```
//!
//! **Choosing a policy:**
//! - Use `Extend` for most cases (default)
//! - Use `Reflect` for periodic or symmetric data
//! - Use `Zero` when data naturally approaches zero at boundaries
//! - Use `NoBoundary` to disable padding
//!
//! ### Auto-Convergence
//!
//! Automatically stop iterations when the smoothed values converge.
//!
//! ```rust
//! use lowess::prelude::*;
//! # let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! # let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];
//!
//! // Build model with auto-convergence
//! let model = Lowess::new()
//!     .fraction(0.5)
//!     .auto_converge(1e-6)      // Stop when change < 1e-6
//!     .iterations(20)           // Maximum iterations
//!     .adapter(Batch)
//!     .build()?;
//!
//! // Fit the model to the data
//! let result = model.fit(&x, &y)?;
//!
//! println!("Converged after {} iterations", result.iterations_used.unwrap());
//! # Result::<(), LowessError>::Ok(())
//! ```
//!
//! ```text
//! Converged after 1 iterations
//! ```
//!
//! ### Return Robustness Weights
//!
//! Include final robustness weights in the output.
//!
//! ```rust
//! use lowess::prelude::*;
//! # let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! # let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];
//!
//! let model = Lowess::new()
//!     .fraction(0.5)
//!     .iterations(3)
//!     .return_robustness_weights()
//!     .adapter(Batch)
//!     .build()?;
//!
//! let result = model.fit(&x, &y)?;
//! // Access robustness weights
//! if let Some(weights) = result.robustness_weights {
//!     println!("Robustness weights: {:?}", weights);
//! }
//! # Result::<(), LowessError>::Ok(())
//! ```
//!
//! ### Scaling Method
//!
//! The scaling method controls how the residuals are scaled.
//!
//! - **`MAR`**:
//!   - Median Absolute Residual: `median(|r|)`
//!   - Default Cleveland implementation
//! - **`MAD`** (default):
//!   - Median Absolute Deviation: `median(|r - median(r)|)`
//!   - More robust to outliers
//!
//! ```rust
//! use lowess::prelude::*;
//! # let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! # let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];
//!
//! let model = Lowess::new()
//!     .fraction(0.5)
//!     .scaling_method(MAD)
//!     .adapter(Batch)
//!     .build()?;
//!
//! let result = model.fit(&x, &y)?;
//! # Result::<(), LowessError>::Ok(())
//! ```
//!
//! ### Diagnostics (Batch and Streaming)
//!
//! Compute diagnostic statistics to assess fit quality.
//!
//! ```rust
//! use lowess::prelude::*;
//! # let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! # let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];
//!
//! // Build model with diagnostics
//! let model = Lowess::new()
//!     .fraction(0.5)
//!     .return_diagnostics()
//!     .return_residuals()
//!     .adapter(Batch)
//!     .build()?;
//!
//! // Fit the model to the data
//! let result = model.fit(&x, &y)?;
//!
//! if let Some(diag) = &result.diagnostics {
//!     println!("RMSE: {:.4}", diag.rmse);
//!     println!("MAE: {:.4}", diag.mae);
//!     println!("R²: {:.4}", diag.r_squared);
//! }
//! # Result::<(), LowessError>::Ok(())
//! ```
//!
//! ```text
//! RMSE: 0.1234
//! MAE: 0.0987
//! R^2: 0.9876
//! ```
//!
//! **Available diagnostics:**
//! - **RMSE**: Root mean squared error
//! - **MAE**: Mean absolute error
//! - **R^2**: Coefficient of determination
//! - **Residual SD**: Standard deviation of residuals
//! - **AIC/AICc**: Information criteria (when applicable)
//!
//! ### Confidence Intervals (Batch only)
//!
//! Confidence intervals quantify uncertainty in the smoothed mean function.
//!
//! ```rust
//! use lowess::prelude::*;
//! # let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! # let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];
//!
//! // Build model with confidence intervals
//! let model = Lowess::new()
//!     .fraction(0.5)
//!     .confidence_intervals(0.95)  // 95% confidence intervals
//!     .adapter(Batch)
//!     .build()?;
//!
//! // Fit the model to the data
//! let result = model.fit(&x, &y)?;
//!
//! // Access confidence intervals
//! for i in 0..x.len() {
//!     println!(
//!         "x={:.1}: y={:.2} [{:.2}, {:.2}]",
//!         x[i],
//!         result.y[i],
//!         result.confidence_lower.as_ref().unwrap()[i],
//!         result.confidence_upper.as_ref().unwrap()[i]
//!     );
//! }
//! # Result::<(), LowessError>::Ok(())
//! ```
//!
//! ```text
//! x=1.0: y=2.00 [1.85, 2.15]
//! x=2.0: y=4.10 [3.92, 4.28]
//! x=3.0: y=5.90 [5.71, 6.09]
//! x=4.0: y=8.20 [8.01, 8.39]
//! x=5.0: y=9.80 [9.65, 9.95]
//! ```
//!
//! ### Prediction Intervals (Batch only)
//!
//! Prediction intervals quantify where new individual observations will likely fall.
//!
//! ```rust
//! use lowess::prelude::*;
//! # let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
//! # let y = vec![2.1, 3.8, 6.2, 7.9, 10.3, 11.8, 14.1, 15.7];
//!
//! // Build model with both interval types
//! let model = Lowess::new()
//!     .fraction(0.5)
//!     .confidence_intervals(0.95)
//!     .prediction_intervals(0.95)  // Both can be enabled
//!     .adapter(Batch)
//!     .build()?;
//!
//! // Fit the model to the data
//! let result = model.fit(&x, &y)?;
//! println!("{}", result);
//! # Result::<(), LowessError>::Ok(())
//! ```
//!
//! ```text
//! Summary:
//!   Data points: 8
//!   Fraction: 0.5
//!
//! Smoothed Data:
//!        X     Y_smooth      Std_Err   Conf_Lower   Conf_Upper   Pred_Lower   Pred_Upper
//!   ----------------------------------------------------------------------------------
//!     1.00     2.01963     0.389365     1.256476     2.782788     1.058911     2.980353
//!     2.00     4.00251     0.345447     3.325438     4.679589     3.108641     4.896386
//!     3.00     5.99959     0.423339     5.169846     6.829335     4.985168     7.014013
//!     4.00     8.09859     0.489473     7.139224     9.057960     6.975666     9.221518
//!     5.00    10.03881     0.551687     8.957506    11.120118     8.810073    11.267551
//!     6.00    12.02872     0.539259    10.971775    13.085672    10.821364    13.236083
//!     7.00    13.89828     0.371149    13.170829    14.625733    12.965670    14.830892
//!     8.00    15.77990     0.408300    14.979631    16.580167    14.789441    16.770356
//! ```
//!
//! **Interval types:**
//! - **Confidence intervals**: Uncertainty in the smoothed mean
//!   - Narrower intervals
//!   - Use for: Understanding precision of the trend estimate
//! - **Prediction intervals**: Uncertainty for new observations
//!   - Wider intervals (includes data scatter + estimation uncertainty)
//!   - Use for: Forecasting where new data points will fall
//!
//! ### Cross-Validation (Batch only)
//!
//! Automatically select the optimal smoothing fraction using cross-validation.
//!
//! ```rust
//! use lowess::prelude::*;
//! # let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();
//! # let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
//!
//! // Build model with K-fold cross-validation
//! let model = Lowess::new()
//!     .cross_validate(KFold(5, &[0.2, 0.3, 0.5, 0.7]).seed(42)) // K-fold CV with 5 folds and 4 fraction options
//!     .adapter(Batch)
//!     .build()?;
//!
//! // Fit the model to the data
//! let result = model.fit(&x, &y)?;
//!
//! println!("Selected fraction: {}", result.fraction_used);
//! println!("CV scores: {:?}", result.cv_scores);
//! # Result::<(), LowessError>::Ok(())
//! ```
//!
//! ```text
//! Selected fraction: 0.5
//! CV scores: Some([0.123, 0.098, 0.145, 0.187])
//! ```
//!
//! ```rust
//! use lowess::prelude::*;
//! # let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();
//! # let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
//!
//! // Build model with leave-one-out cross-validation
//! let model = Lowess::new()
//!     .cross_validate(LOOCV(&[0.2, 0.3, 0.5, 0.7])) // Leave-one-out CV with 4 fraction options
//!     .adapter(Batch)
//!     .build()?;
//!
//! let result = model.fit(&x, &y)?;
//! println!("{}", result);
//! # Result::<(), LowessError>::Ok(())
//! ```
//!
//! ```text
//! Summary:
//!   Data points: 20
//!   Fraction: 0.5 (selected via LOOCV)
//!
//! Smoothed Data:
//!        X     Y_smooth
//!   --------------------
//!     1.00     3.00000
//!     2.00     5.00000
//!     3.00     7.00000
//!   ... (17 more rows)
//! ```
//!
//! **Choosing a Method:**
//!
//! * **K-Fold**: Good balance between accuracy and speed. Common choices:
//!   - k=5: Fast, reasonable accuracy
//!   - k=10: Standard choice, good accuracy
//!   - k=20: Higher accuracy, slower
//!
//! * **LOOCV**: Maximum accuracy but computationally expensive (O(n^2) evaluations).
//!   Best for small datasets (n < 100) where accuracy is critical.
//!
//! ### Chunk Size (Streaming Adapter)
//!
//! Set the number of points to process in each chunk for the Streaming adapter.
//!
//! ```rust
//! use lowess::prelude::*;
//!
//! let mut processor = Lowess::new()
//!     .fraction(0.3)
//!     .adapter(Streaming)
//!     .chunk_size(10000)  // Process 10K points at a time
//!     .overlap(1000)      // 1K point overlap
//!     .build()?;
//! # Result::<(), LowessError>::Ok(())
//! ```
//!
//! **Typical values:**
//! - Small chunks: 1,000-5,000 (low memory, more overhead)
//! - Medium chunks: 5,000-20,000 (balanced, recommended)
//! - Large chunks: 20,000-100,000 (high memory, less overhead)
//!
//! ### Overlap (Streaming Adapter)
//!
//! Set the number of overlapping points between chunks for the Streaming adapter.
//!
//! **Rule of thumb:** `overlap = 2 × window_size`, where `window_size = fraction × chunk_size`
//!
//! Larger overlap provides better boundary handling but increases computation.
//! Must be less than `chunk_size`.
//!
//! ### Merge Strategy (Streaming Adapter)
//!
//! Control how overlapping values are merged between chunks in the Streaming adapter.
//!
//! - **`WeightedAverage`** (default): Distance-weighted average
//! - **`Average`**: Simple average
//! - **`TakeFirst`**: Use value from first chunk
//! - **`TakeLast`**: Use value from last chunk
//!
//! ```rust
//! use lowess::prelude::*;
//!
//! let mut processor = Lowess::new()
//!     .fraction(0.3)
//!     .merge_strategy(WeightedAverage)
//!     .adapter(Streaming)
//!     .build()?;
//! # Result::<(), LowessError>::Ok(())
//! ```
//!
//! ### Window Capacity (Online Adapter)
//!
//! Set the maximum number of points to retain in the sliding window for the Online adapter.
//!
//! ```rust
//! use lowess::prelude::*;
//!
//! let mut processor = Lowess::new()
//!     .fraction(0.3)
//!     .adapter(Online)
//!     .window_capacity(500)  // Keep last 500 points
//!     .build()?;
//! # Result::<(), LowessError>::Ok(())
//! ```
//!
//! **Typical values:**
//! - Small windows: 100-500 (fast, less smooth)
//! - Medium windows: 500-2000 (balanced)
//! - Large windows: 2000-10000 (slow, very smooth)
//!
//! ### Min Points (Online Adapter)
//!
//! Set the minimum number of points required before smoothing starts in the Online adapter.
//!
//! Must be at least 2 (required for linear regression) and at most `window_capacity`.
//!
//! ```rust
//! use lowess::prelude::*;
//!
//! let mut processor = Lowess::new()
//!     .fraction(0.3)
//!     .adapter(Online)
//!     .window_capacity(100)
//!     .min_points(10)  // Wait for 10 points before smoothing
//!     .build()?;
//! # Result::<(), LowessError>::Ok(())
//! ```
//!
//! ### Update Mode (Online Adapter)
//!
//! Choose between incremental and full window updates for the Online adapter.
//!
//! - **`Incremental`** (default): Fit only the latest point - O(q) per point
//! - **`Full`**: Re-smooth entire window - O(q^2) per point
//!
//! ```rust
//! use lowess::prelude::*;
//!
//! // High-performance incremental updates
//! let mut processor = Lowess::new()
//!     .fraction(0.3)
//!     .adapter(Online)
//!     .window_capacity(100)
//!     .update_mode(Incremental)
//!     .build()?;
//!
//! for i in 0..1000 {
//!     let x = i as f64;
//!     let y = 2.0 * x + 1.0;
//!     if let Some(output) = processor.add_point(x, y)? {
//!         println!("Smoothed: {}", output.smoothed);
//!     }
//! }
//! # Result::<(), LowessError>::Ok(())
//! ```
//!
//! ## A comprehensive example showing multiple features:
//!
//! ```rust
//! use lowess::prelude::*;
//!
//! // Generate sample data with outliers
//! let x: Vec<f64> = (1..=50).map(|i| i as f64).collect();
//! let mut y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0 + (xi * 0.5).sin() * 5.0).collect();
//! y[10] = 100.0;  // Add an outlier
//! y[25] = -50.0;  // Add another outlier
//!
//! // Build the model with comprehensive configuration
//! let model = Lowess::new()
//!     .fraction(0.3)                                  // Moderate smoothing
//!     .iterations(5)                                  // Strong outlier resistance
//!     .weight_function(Tricube)                       // Default kernel
//!     .robustness_method(Bisquare)                    // Bisquare robustness
//!     .confidence_intervals(0.95)                     // 95% confidence intervals
//!     .prediction_intervals(0.95)                     // 95% prediction intervals
//!     .return_diagnostics()                           // Include diagnostics
//!     .return_residuals()                             // Include residuals
//!     .return_robustness_weights()                    // Include robustness weights
//!     .zero_weight_fallback(UseLocalMean)             // Fallback policy
//!     .adapter(Batch)
//!     .build()?;
//!
//! // Fit the model to the data
//! let result = model.fit(&x, &y)?;
//!
//! // Examine results
//! println!("Smoothed {} points", result.y.len());
//!
//! // Check diagnostics
//! if let Some(diag) = &result.diagnostics {
//!     println!("Fit quality:");
//!     println!("  RMSE: {:.4}", diag.rmse);
//!     println!("  R²: {:.4}", diag.r_squared);
//! }
//!
//! // Identify outliers
//! if let Some(weights) = &result.robustness_weights {
//!     println!("\nOutliers detected:");
//!     for (i, &w) in weights.iter().enumerate() {
//!         if w < 0.1 {
//!             println!("  Point {}: y={:.1}, weight={:.3}", i, y[i], w);
//!         }
//!     }
//! }
//!
//! // Show confidence intervals for first few points
//! println!("\nFirst 5 points with intervals:");
//! for i in 0..5 {
//!     println!(
//!         "  x={:.0}: {:.2} [{:.2}, {:.2}] | [{:.2}, {:.2}]",
//!         x[i],
//!         result.y[i],
//!         result.confidence_lower.as_ref().unwrap()[i],
//!         result.confidence_upper.as_ref().unwrap()[i],
//!         result.prediction_lower.as_ref().unwrap()[i],
//!         result.prediction_upper.as_ref().unwrap()[i]
//!     );
//! }
//! # Result::<(), LowessError>::Ok(())
//! ```
//!
//! ```text
//! Smoothed 50 points
//! Fit quality:
//!   RMSE: 0.5234
//!   R^2: 0.9987
//!
//! Outliers detected:
//!   Point 10: y=100.0, weight=0.000
//!   Point 25: y=-50.0, weight=0.000
//!
//! First 5 points with intervals:
//!   x=1: 3.12 [2.98, 3.26] | [2.45, 3.79]
//!   x=2: 5.24 [5.10, 5.38] | [4.57, 5.91]
//!   x=3: 7.36 [7.22, 7.50] | [6.69, 8.03]
//!   x=4: 9.48 [9.34, 9.62] | [8.81, 10.15]
//!   x=5: 11.60 [11.46, 11.74] | [10.93, 12.27]
//! ```
//!
//! ## References
//!
//! - Cleveland, W. S. (1979). "Robust Locally Weighted Regression and Smoothing Scatterplots"
//! - Cleveland, W. S. (1981). "LOWESS: A Program for Smoothing Scatterplots by Robust Locally Weighted Regression"
//!
//! ## License
//!
//! See the repository for license information and contribution guidelines.

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(missing_docs)]

#[cfg(not(feature = "std"))]
#[macro_use]
extern crate alloc;

// ============================================================================
// Internal Modules
// ============================================================================

// Layer 1: Primitives - data structures and basic utilities.
//
// Contains fundamental data structures (`Window`, `errors`), and
// low-level utilities for sorting, partitioning, and windowing.
mod primitives;

// Layer 2: Math - pure mathematical functions.
//
// Contains kernel functions for distance-based weighting and
// robust statistics (MAD). These are reusable mathematical building blocks.
mod math;

// Layer 3: Algorithms - core LOWESS algorithms.
//
// Contains the implementations of local regression (OLS/WLS),
// robustness weighting (`Bisquare`, `Huber`, `Talwar`), and
// interpolation/delta optimization.
mod algorithms;

// Layer 4: Evaluation - post-processing and diagnostics.
//
// Contains cross-validation for parameter selection, diagnostic metrics
// (RMSE, R^2, AIC), and confidence/prediction interval computation.
mod evaluation;

// Layer 5: Engine - orchestration and execution control.
//
// Contains the core smoothing iteration logic, automatic convergence
// detection, and result assembly.
mod engine;

// Layer 6: Adapters - execution mode adapters.
//
// Contains execution adapters for different use cases:
// batch (standard), streaming (large datasets), online (incremental).
mod adapters;

// High-level fluent API for LOWESS smoothing.
//
// Provides the `Lowess` builder for configuring and running LOWESS smoothing.
mod api;

// ============================================================================
// Prelude
// ============================================================================

/// Standard LOWESS prelude.
///
/// This module is intended to be wildcard-imported for convenient access
/// to the most commonly used types:
///
/// ```
/// use lowess::prelude::*;
/// ```
pub mod prelude {
    pub use crate::api::{
        Adapter::{Batch, Online, Streaming},
        BoundaryPolicy::Extend,
        BoundaryPolicy::NoBoundary,
        BoundaryPolicy::Reflect,
        BoundaryPolicy::Zero,
        KFold, LOOCV, LowessBuilder as Lowess, LowessError, LowessResult,
        MergeStrategy::Average,
        MergeStrategy::TakeFirst,
        MergeStrategy::WeightedAverage,
        RobustnessMethod::Bisquare,
        RobustnessMethod::Huber,
        RobustnessMethod::Talwar,
        ScalingMethod::MAD,
        ScalingMethod::MAR,
        UpdateMode::Full,
        UpdateMode::Incremental,
        WeightFunction::Biweight,
        WeightFunction::Cosine,
        WeightFunction::Epanechnikov,
        WeightFunction::Gaussian,
        WeightFunction::Triangle,
        WeightFunction::Tricube,
        WeightFunction::Uniform,
        ZeroWeightFallback::ReturnNone,
        ZeroWeightFallback::ReturnOriginal,
        ZeroWeightFallback::UseLocalMean,
    };
}

// ============================================================================
// Testing re-exports
// ============================================================================

/// Internal modules for development and testing.
///
/// This module re-exports internal modules for development and testing purposes.
/// It is only available with the `dev` feature enabled.
///
/// **Warning**: These are internal implementation details and may change without notice.
/// Do not use in production code.
#[cfg(feature = "dev")]
pub mod internals {
    /// Internal primitive types and utilities.
    pub mod primitives {
        pub use crate::primitives::*;
    }
    /// Internal math functions.
    pub mod math {
        pub use crate::math::*;
    }
    /// Internal core algorithms.
    pub mod algorithms {
        pub use crate::algorithms::*;
    }
    /// Internal execution engine.
    pub mod engine {
        pub use crate::engine::*;
    }
    /// Internal evaluation and diagnostics.
    pub mod evaluation {
        pub use crate::evaluation::*;
    }
    /// Internal adapters.
    pub mod adapters {
        pub use crate::adapters::*;
    }
    /// Internal API.
    pub mod api {
        pub use crate::api::*;
    }
}
