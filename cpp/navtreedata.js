/*
 @licstart  The following is the entire license notice for the JavaScript code in this file.

 The MIT License (MIT)

 Copyright (C) 1997-2020 by Dimitri van Heesch

 Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 and associated documentation files (the "Software"), to deal in the Software without restriction,
 including without limitation the rights to use, copy, modify, merge, publish, distribute,
 sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all copies or
 substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

 @licend  The above is the entire license notice for the JavaScript code in this file
*/
var NAVTREE =
[
  [ "fastLowess", "index.html", [
    [ "Concepts", "index.html", "index" ],
    [ "Execution Modes", "md_docs_2adapter-choice.html", [
      [ "Overview", "md_docs_2adapter-choice.html#autotoc_md1", null ]
    ] ],
    [ "OnlineLowess API", "md_docs_2api-online.html", [
      [ "When to Use", "md_docs_2api-online.html#autotoc_md4", null ],
      [ "Class", "md_docs_2api-online.html#autotoc_md5", [
        [ "<tt>fastlowess::OnlineLowess</tt>", "md_docs_2api-online.html#autotoc_md6", null ]
      ] ],
      [ "Options Structure", "md_docs_2api-online.html#autotoc_md7", [
        [ "<tt>OnlineOptions</tt> (inherits <tt>LowessOptions</tt>)", "md_docs_2api-online.html#autotoc_md8", null ]
      ] ],
      [ "Result Structure", "md_docs_2api-online.html#autotoc_md9", [
        [ "<tt>fastlowess::OnlineOutput</tt>", "md_docs_2api-online.html#autotoc_md10", null ]
      ] ],
      [ "Options", "md_docs_2api-online.html#autotoc_md11", [
        [ "update_mode", "md_docs_2api-online.html#autotoc_md12", null ]
      ] ]
    ] ],
    [ "StreamingLowess API", "md_docs_2api-streaming.html", [
      [ "When to Use", "md_docs_2api-streaming.html#autotoc_md14", null ],
      [ "Class", "md_docs_2api-streaming.html#autotoc_md15", [
        [ "<tt>fastlowess::StreamingLowess</tt>", "md_docs_2api-streaming.html#autotoc_md16", null ]
      ] ],
      [ "Result Structure", "md_docs_2api-streaming.html#autotoc_md17", [
        [ "<tt>fastlowess::LowessResult</tt>", "md_docs_2api-streaming.html#autotoc_md18", null ]
      ] ],
      [ "Options Structure", "md_docs_2api-streaming.html#autotoc_md19", [
        [ "<tt>StreamingOptions</tt> (inherits <tt>LowessOptions</tt>)", "md_docs_2api-streaming.html#autotoc_md20", null ]
      ] ],
      [ "Options", "md_docs_2api-streaming.html#autotoc_md21", [
        [ "merge_strategy", "md_docs_2api-streaming.html#autotoc_md22", null ]
      ] ]
    ] ],
    [ "API", "md_docs_2api.html", [
      [ "When to Use", "md_docs_2api.html#autotoc_md25", null ],
      [ "Classes", "md_docs_2api.html#autotoc_md26", [
        [ "<tt>fastlowess::Lowess</tt>", "md_docs_2api.html#autotoc_md27", null ]
      ] ],
      [ "Options Structures", "md_docs_2api.html#autotoc_md28", [
        [ "<tt>LowessOptions</tt>", "md_docs_2api.html#autotoc_md29", null ]
      ] ],
      [ "GPU Acceleration", "md_docs_2api.html#autotoc_md30", [
        [ "Enabling GPU Support", "md_docs_2api.html#autotoc_md31", null ],
        [ "Usage", "md_docs_2api.html#autotoc_md32", null ],
        [ "Supported Features", "md_docs_2api.html#autotoc_md33", [
          [ "Feature Comparison", "md_docs_2api.html#autotoc_md34", null ]
        ] ],
        [ "Hardware Requirements", "md_docs_2api.html#autotoc_md35", null ],
        [ "Performance Considerations", "md_docs_2api.html#autotoc_md36", null ]
      ] ],
      [ "Result Structure", "md_docs_2api.html#autotoc_md37", [
        [ "<tt>fastlowess::LowessResult</tt>", "md_docs_2api.html#autotoc_md38", null ],
        [ "<tt>fastlowess::Diagnostics</tt>", "md_docs_2api.html#autotoc_md39", null ]
      ] ],
      [ "Options", "md_docs_2api.html#autotoc_md40", [
        [ "weight_function", "md_docs_2api.html#autotoc_md41", null ],
        [ "robustness_method", "md_docs_2api.html#autotoc_md42", null ],
        [ "boundary_policy", "md_docs_2api.html#autotoc_md43", null ],
        [ "scaling_method", "md_docs_2api.html#autotoc_md44", null ],
        [ "zero_weight_fallback", "md_docs_2api.html#autotoc_md45", null ],
        [ "merge_strategy", "md_docs_2api.html#autotoc_md46", null ],
        [ "update_mode", "md_docs_2api.html#autotoc_md47", null ]
      ] ],
      [ "Example", "md_docs_2api.html#autotoc_md48", null ]
    ] ],
    [ "Benchmarks", "md_docs_2benchmarks.html", [
      [ "CPU Benchmarks", "md_docs_2benchmarks.html#autotoc_md50", null ],
      [ "GPU Backend", "md_docs_2benchmarks.html#autotoc_md52", null ],
      [ "Reproducing Benchmarks", "md_docs_2benchmarks.html#autotoc_md54", null ]
    ] ],
    [ "Boundary Handling", "md_docs_2boundary.html", [
      [ "Overview", "md_docs_2boundary.html#autotoc_md56", null ],
      [ "Extend (Default)", "md_docs_2boundary.html#autotoc_md58", null ],
      [ "Reflect", "md_docs_2boundary.html#autotoc_md60", null ],
      [ "Zero", "md_docs_2boundary.html#autotoc_md62", null ],
      [ "No Boundary", "md_docs_2boundary.html#autotoc_md64", null ],
      [ "Choosing a Policy", "md_docs_2boundary.html#autotoc_md66", null ]
    ] ],
    [ "Cross-Validation", "md_docs_2cross-validation.html", [
      [ "Overview", "md_docs_2cross-validation.html#autotoc_md86", null ],
      [ "K-Fold Cross-Validation", "md_docs_2cross-validation.html#autotoc_md88", null ],
      [ "Leave-One-Out (LOOCV)", "md_docs_2cross-validation.html#autotoc_md90", null ],
      [ "Seeded Randomization", "md_docs_2cross-validation.html#autotoc_md92", null ],
      [ "Comparison", "md_docs_2cross-validation.html#autotoc_md94", null ],
      [ "CV Metrics", "md_docs_2cross-validation.html#autotoc_md96", null ],
      [ "Interpreting Results", "md_docs_2cross-validation.html#autotoc_md98", null ],
      [ "Availability", "md_docs_2cross-validation.html#autotoc_md100", null ],
      [ "Best Practices", "md_docs_2cross-validation.html#autotoc_md102", null ]
    ] ],
    [ "Custom Weights", "md_docs_2custom-weights.html", [
      [ "How Custom Weights Work", "md_docs_2custom-weights.html#autotoc_md104", null ],
      [ "When to Use Custom Weights", "md_docs_2custom-weights.html#autotoc_md106", [
        [ "Custom Weights vs. Robustness Iterations", "md_docs_2custom-weights.html#autotoc_md107", null ]
      ] ],
      [ "Basic Usage", "md_docs_2custom-weights.html#autotoc_md109", [
        [ "Suppress a Known Outlier", "md_docs_2custom-weights.html#autotoc_md110", null ],
        [ "Emphasize Important Points", "md_docs_2custom-weights.html#autotoc_md112", null ],
        [ "Propagate Measurement Uncertainty", "md_docs_2custom-weights.html#autotoc_md114", null ]
      ] ],
      [ "Combined with Robustness Iterations", "md_docs_2custom-weights.html#autotoc_md116", null ],
      [ "Validation Rules", "md_docs_2custom-weights.html#autotoc_md118", null ],
      [ "See Also", "md_docs_2custom-weights.html#autotoc_md120", null ]
    ] ],
    [ "GPU Backend", "md_docs_2gpu-backend.html", [
      [ "Overview", "md_docs_2gpu-backend.html#autotoc_md122", [
        [ "Supported Features", "md_docs_2gpu-backend.html#autotoc_md123", null ],
        [ "Feature Comparison", "md_docs_2gpu-backend.html#autotoc_md124", null ]
      ] ],
      [ "Checking Availability", "md_docs_2gpu-backend.html#autotoc_md126", null ],
      [ "Installing GPU Support", "md_docs_2gpu-backend.html#autotoc_md128", null ],
      [ "Usage", "md_docs_2gpu-backend.html#autotoc_md130", null ],
      [ "See also", "md_docs_2gpu-backend.html#autotoc_md131", null ]
    ] ],
    [ "Installation", "md_docs_2installation.html", [
      [ "Pre-built Binaries (Linux (x64))", "md_docs_2installation.html#autotoc_md133", null ],
      [ "Pre-built Binaries (macOS (x64))", "md_docs_2installation.html#autotoc_md134", null ],
      [ "Pre-built Binaries (Windows (x64))", "md_docs_2installation.html#autotoc_md135", null ],
      [ "From Source", "md_docs_2installation.html#autotoc_md136", null ],
      [ "From conda-forge", "md_docs_2installation.html#autotoc_md137", null ],
      [ "Verify Installation", "md_docs_2installation.html#autotoc_md139", null ]
    ] ],
    [ "Intervals", "md_docs_2intervals.html", [
      [ "Overview", "md_docs_2intervals.html#autotoc_md141", null ],
      [ "Confidence Intervals", "md_docs_2intervals.html#autotoc_md143", null ],
      [ "Prediction Intervals", "md_docs_2intervals.html#autotoc_md145", null ],
      [ "Both Intervals", "md_docs_2intervals.html#autotoc_md147", null ],
      [ "Confidence Levels", "md_docs_2intervals.html#autotoc_md149", null ],
      [ "Standard Errors", "md_docs_2intervals.html#autotoc_md151", null ],
      [ "Availability", "md_docs_2intervals.html#autotoc_md153", null ]
    ] ],
    [ "Weight Functions", "md_docs_2kernels.html", [
      [ "Overview", "md_docs_2kernels.html#autotoc_md155", null ],
      [ "Available Kernels", "md_docs_2kernels.html#autotoc_md157", null ],
      [ "Tricube (Default)", "md_docs_2kernels.html#autotoc_md159", null ],
      [ "Epanechnikov", "md_docs_2kernels.html#autotoc_md161", null ],
      [ "Gaussian", "md_docs_2kernels.html#autotoc_md163", null ],
      [ "Biweight", "md_docs_2kernels.html#autotoc_md165", null ],
      [ "Cosine", "md_docs_2kernels.html#autotoc_md167", null ],
      [ "Triangle", "md_docs_2kernels.html#autotoc_md169", null ],
      [ "Uniform", "md_docs_2kernels.html#autotoc_md171", null ],
      [ "Choosing a Kernel", "md_docs_2kernels.html#autotoc_md173", null ]
    ] ],
    [ "Merge Strategies", "md_docs_2merge.html", [
      [ "Overview", "md_docs_2merge.html#autotoc_md175", null ],
      [ "Average", "md_docs_2merge.html#autotoc_md177", null ],
      [ "Take First", "md_docs_2merge.html#autotoc_md179", null ],
      [ "Take Last", "md_docs_2merge.html#autotoc_md181", null ],
      [ "Weighted Average", "md_docs_2merge.html#autotoc_md183", null ],
      [ "Choosing a Strategy", "md_docs_2merge.html#autotoc_md185", null ]
    ] ],
    [ "Parameters", "md_docs_2parameters.html", [
      [ "Quick Reference", "md_docs_2parameters.html#autotoc_md187", null ],
      [ "Parameter Options Summary", "md_docs_2parameters.html#autotoc_md189", null ],
      [ "Core Parameters", "md_docs_2parameters.html#autotoc_md191", [
        [ "fraction", "md_docs_2parameters.html#autotoc_md192", null ],
        [ "iterations", "md_docs_2parameters.html#autotoc_md194", null ],
        [ "delta", "md_docs_2parameters.html#autotoc_md196", null ],
        [ "weight_function", "md_docs_2parameters.html#autotoc_md198", null ],
        [ "robustness_method", "md_docs_2parameters.html#autotoc_md200", null ],
        [ "boundary_policy", "md_docs_2parameters.html#autotoc_md202", null ],
        [ "scaling_method", "md_docs_2parameters.html#autotoc_md204", null ],
        [ "zero_weight_fallback", "md_docs_2parameters.html#autotoc_md206", null ],
        [ "auto_converge", "md_docs_2parameters.html#autotoc_md208", null ],
        [ "custom_weights", "md_docs_2parameters.html#autotoc_md210", null ]
      ] ],
      [ "Output Options", "md_docs_2parameters.html#autotoc_md212", [
        [ "return_residuals", "md_docs_2parameters.html#autotoc_md213", null ],
        [ "return_diagnostics", "md_docs_2parameters.html#autotoc_md215", null ],
        [ "return_robustness_weights", "md_docs_2parameters.html#autotoc_md217", null ],
        [ "return_se", "md_docs_2parameters.html#autotoc_md219", null ],
        [ "confidence_intervals / prediction_intervals", "md_docs_2parameters.html#autotoc_md221", null ]
      ] ],
      [ "CV Methods", "md_docs_2parameters.html#autotoc_md223", [
        [ "cv_method", "md_docs_2parameters.html#autotoc_md224", null ]
      ] ],
      [ "Adapter Parameters", "md_docs_2parameters.html#autotoc_md226", [
        [ "chunk_size", "md_docs_2parameters.html#autotoc_md227", null ],
        [ "overlap", "md_docs_2parameters.html#autotoc_md229", null ],
        [ "merge_strategy", "md_docs_2parameters.html#autotoc_md231", null ],
        [ "window_capacity", "md_docs_2parameters.html#autotoc_md233", null ],
        [ "min_points", "md_docs_2parameters.html#autotoc_md235", null ],
        [ "update_mode", "md_docs_2parameters.html#autotoc_md237", null ]
      ] ]
    ] ],
    [ "Quick Start", "md_docs_2quickstart.html", [
      [ "Basic Smoothing", "md_docs_2quickstart.html#autotoc_md239", null ],
      [ "With Confidence Intervals", "md_docs_2quickstart.html#autotoc_md241", null ],
      [ "Handling Outliers", "md_docs_2quickstart.html#autotoc_md243", null ],
      [ "Streaming Mode", "md_docs_2quickstart.html#autotoc_md245", null ],
      [ "Next Steps", "md_docs_2quickstart.html#autotoc_md247", null ]
    ] ],
    [ "Robustness", "md_docs_2robustness.html", [
      [ "How Robustness Works", "md_docs_2robustness.html#autotoc_md249", null ],
      [ "Robustness Methods", "md_docs_2robustness.html#autotoc_md251", [
        [ "Bisquare (Default)", "md_docs_2robustness.html#autotoc_md252", null ],
        [ "Huber", "md_docs_2robustness.html#autotoc_md254", null ],
        [ "Talwar", "md_docs_2robustness.html#autotoc_md256", null ]
      ] ],
      [ "Comparison", "md_docs_2robustness.html#autotoc_md258", null ],
      [ "Detecting Outliers", "md_docs_2robustness.html#autotoc_md260", null ],
      [ "Scale Estimation", "md_docs_2robustness.html#autotoc_md262", null ],
      [ "Auto-Convergence", "md_docs_2robustness.html#autotoc_md264", null ]
    ] ],
    [ "Scaling Methods", "md_docs_2scaling.html", [
      [ "Overview", "md_docs_2scaling.html#autotoc_md266", null ],
      [ "MAD — Median Absolute Deviation (Default)", "md_docs_2scaling.html#autotoc_md268", null ],
      [ "MAR — Median Absolute Residual", "md_docs_2scaling.html#autotoc_md270", null ],
      [ "Mean — Mean Absolute Residual", "md_docs_2scaling.html#autotoc_md272", null ],
      [ "Choosing a Scaling Method", "md_docs_2scaling.html#autotoc_md274", null ]
    ] ],
    [ "Genomic Data Smoothing", "md_docs_2use-case-genomics.html", [
      [ "Overview", "md_docs_2use-case-genomics.html#autotoc_md276", null ],
      [ "Methylation Profile Smoothing", "md_docs_2use-case-genomics.html#autotoc_md278", [
        [ "The Challenge", "md_docs_2use-case-genomics.html#autotoc_md279", null ],
        [ "Solution", "md_docs_2use-case-genomics.html#autotoc_md280", null ]
      ] ],
      [ "ChIP-seq Signal Smoothing", "md_docs_2use-case-genomics.html#autotoc_md282", [
        [ "Application", "md_docs_2use-case-genomics.html#autotoc_md283", null ]
      ] ],
      [ "Large Genome Coverage (Streaming)", "md_docs_2use-case-genomics.html#autotoc_md285", null ],
      [ "Best Practices for Genomic Data", "md_docs_2use-case-genomics.html#autotoc_md287", null ],
      [ "See Also", "md_docs_2use-case-genomics.html#autotoc_md289", null ]
    ] ],
    [ "Real-Time Processing", "md_docs_2use-case-real-time.html", [
      [ "Overview", "md_docs_2use-case-real-time.html#autotoc_md291", null ],
      [ "Online Mode: Point-by-Point", "md_docs_2use-case-real-time.html#autotoc_md293", [
        [ "Sensor Data Example", "md_docs_2use-case-real-time.html#autotoc_md294", null ]
      ] ],
      [ "Streaming Mode: Chunk Processing", "md_docs_2use-case-real-time.html#autotoc_md296", [
        [ "Log File Processing", "md_docs_2use-case-real-time.html#autotoc_md297", null ]
      ] ],
      [ "Real-Time Dashboard Example", "md_docs_2use-case-real-time.html#autotoc_md299", null ],
      [ "Choosing Parameters", "md_docs_2use-case-real-time.html#autotoc_md301", [
        [ "Online Mode", "md_docs_2use-case-real-time.html#autotoc_md302", null ],
        [ "Streaming Mode", "md_docs_2use-case-real-time.html#autotoc_md303", null ]
      ] ],
      [ "Performance Considerations", "md_docs_2use-case-real-time.html#autotoc_md305", null ],
      [ "See Also", "md_docs_2use-case-real-time.html#autotoc_md307", null ]
    ] ],
    [ "Time Series Analysis", "md_docs_2use-case-time-series.html", [
      [ "Overview", "md_docs_2use-case-time-series.html#autotoc_md309", null ],
      [ "Basic Trend Extraction", "md_docs_2use-case-time-series.html#autotoc_md311", null ],
      [ "Detrending", "md_docs_2use-case-time-series.html#autotoc_md313", null ],
      [ "Forecasting with Prediction Intervals", "md_docs_2use-case-time-series.html#autotoc_md315", null ],
      [ "Handling Missing Data", "md_docs_2use-case-time-series.html#autotoc_md317", null ],
      [ "Multi-Scale Analysis", "md_docs_2use-case-time-series.html#autotoc_md319", null ],
      [ "Gene Expression Time Course", "md_docs_2use-case-time-series.html#autotoc_md321", null ],
      [ "Choosing Fraction for Time Series", "md_docs_2use-case-time-series.html#autotoc_md323", null ],
      [ "See Also", "md_docs_2use-case-time-series.html#autotoc_md325", null ]
    ] ]
  ] ]
];

var NAVTREEINDEX =
[
"index.html"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';