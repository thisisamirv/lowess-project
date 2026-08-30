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
    [ "LOWESS Project", "index.html#autotoc_md0", [
      [ "Installation & Documentation", "index.html#autotoc_md2", [
        [ "GPU Backend", "index.html#autotoc_md3", null ]
      ] ],
      [ "LOESS vs. LOWESS", "index.html#autotoc_md5", null ],
      [ "</blockquote>", "index.html#autotoc_md6", null ],
      [ "Why this package?", "index.html#autotoc_md7", [
        [ "Speed", "index.html#autotoc_md8", null ],
        [ "Robustness", "index.html#autotoc_md9", null ],
        [ "Features", "index.html#autotoc_md10", null ]
      ] ],
      [ "Validation", "index.html#autotoc_md11", null ],
      [ "API Reference", "index.html#autotoc_md12", null ],
      [ "Contributing", "index.html#autotoc_md14", null ],
      [ "Changelog", "index.html#autotoc_md15", null ],
      [ "License", "index.html#autotoc_md16", null ],
      [ "Citation", "index.html#autotoc_md17", null ]
    ] ],
    [ "Execution Modes", "md_docs_2adapter-choice.html", [
      [ "Overview", "md_docs_2adapter-choice.html#autotoc_md19", null ]
    ] ],
    [ "OnlineLowess API", "md_docs_2api-online.html", [
      [ "When to Use", "md_docs_2api-online.html#autotoc_md22", null ],
      [ "Class", "md_docs_2api-online.html#autotoc_md23", [
        [ "fastlowess::OnlineLowess", "md_docs_2api-online.html#autotoc_md24", null ]
      ] ],
      [ "Options Structure", "md_docs_2api-online.html#autotoc_md25", [
        [ "OnlineOptions (inherits LowessOptions)", "md_docs_2api-online.html#autotoc_md26", null ]
      ] ],
      [ "Result Structure", "md_docs_2api-online.html#autotoc_md27", [
        [ "fastlowess::OnlineOutput", "md_docs_2api-online.html#autotoc_md28", null ]
      ] ],
      [ "Options", "md_docs_2api-online.html#autotoc_md29", [
        [ "update_mode", "md_docs_2api-online.html#autotoc_md30", null ]
      ] ]
    ] ],
    [ "StreamingLowess API", "md_docs_2api-streaming.html", [
      [ "When to Use", "md_docs_2api-streaming.html#autotoc_md32", null ],
      [ "Class", "md_docs_2api-streaming.html#autotoc_md33", [
        [ "fastlowess::StreamingLowess", "md_docs_2api-streaming.html#autotoc_md34", null ]
      ] ],
      [ "Result Structure", "md_docs_2api-streaming.html#autotoc_md35", [
        [ "fastlowess::LowessResult", "md_docs_2api-streaming.html#autotoc_md36", null ]
      ] ],
      [ "Options Structure", "md_docs_2api-streaming.html#autotoc_md37", [
        [ "StreamingOptions (inherits LowessOptions)", "md_docs_2api-streaming.html#autotoc_md38", null ]
      ] ],
      [ "Options", "md_docs_2api-streaming.html#autotoc_md39", [
        [ "merge_strategy", "md_docs_2api-streaming.html#autotoc_md40", null ]
      ] ]
    ] ],
    [ "API", "md_docs_2api.html", [
      [ "When to Use Batch Adapter", "md_docs_2api.html#autotoc_md43", null ],
      [ "Classes", "md_docs_2api.html#autotoc_md44", [
        [ "fastlowess::Lowess", "md_docs_2api.html#autotoc_md45", null ]
      ] ],
      [ "Options Structures", "md_docs_2api.html#autotoc_md46", [
        [ "LowessOptions", "md_docs_2api.html#autotoc_md47", null ]
      ] ],
      [ "GPU Acceleration", "md_docs_2api.html#autotoc_md48", [
        [ "Enabling GPU Support", "md_docs_2api.html#autotoc_md49", null ],
        [ "Usage", "md_docs_2api.html#autotoc_md50", null ],
        [ "Supported Features", "md_docs_2api.html#autotoc_md51", [
          [ "Feature Comparison", "md_docs_2api.html#autotoc_md52", null ]
        ] ],
        [ "Hardware Requirements", "md_docs_2api.html#autotoc_md53", null ],
        [ "Performance Considerations", "md_docs_2api.html#autotoc_md54", null ]
      ] ],
      [ "Result Structure", "md_docs_2api.html#autotoc_md55", [
        [ "fastlowess::LowessResult", "md_docs_2api.html#autotoc_md56", null ],
        [ "fastlowess::Diagnostics", "md_docs_2api.html#autotoc_md57", null ]
      ] ],
      [ "Options", "md_docs_2api.html#autotoc_md58", [
        [ "weight_function", "md_docs_2api.html#autotoc_md59", null ],
        [ "robustness_method", "md_docs_2api.html#autotoc_md60", null ],
        [ "boundary_policy", "md_docs_2api.html#autotoc_md61", null ],
        [ "scaling_method", "md_docs_2api.html#autotoc_md62", null ],
        [ "zero_weight_fallback", "md_docs_2api.html#autotoc_md63", null ],
        [ "merge_strategy", "md_docs_2api.html#autotoc_md64", null ],
        [ "update_mode", "md_docs_2api.html#autotoc_md65", null ]
      ] ],
      [ "Example", "md_docs_2api.html#autotoc_md66", null ]
    ] ],
    [ "Benchmarks", "md_docs_2benchmarks.html", [
      [ "CPU Benchmarks", "md_docs_2benchmarks.html#autotoc_md68", null ],
      [ "GPU Backend", "md_docs_2benchmarks.html#autotoc_md70", null ],
      [ "Reproducing Benchmarks", "md_docs_2benchmarks.html#autotoc_md72", null ]
    ] ],
    [ "Boundary Handling", "md_docs_2boundary.html", [
      [ "Overview", "md_docs_2boundary.html#autotoc_md74", null ],
      [ "Extend (Default)", "md_docs_2boundary.html#autotoc_md76", null ],
      [ "Reflect", "md_docs_2boundary.html#autotoc_md78", null ],
      [ "Zero", "md_docs_2boundary.html#autotoc_md80", null ],
      [ "No Boundary", "md_docs_2boundary.html#autotoc_md82", null ],
      [ "Choosing a Policy", "md_docs_2boundary.html#autotoc_md84", null ]
    ] ],
    [ "Concepts", "md_docs_2concepts.html", [
      [ "What is LOWESS?", "md_docs_2concepts.html#autotoc_md86", null ],
      [ "How It Works", "md_docs_2concepts.html#autotoc_md88", null ],
      [ "The Fraction Parameter", "md_docs_2concepts.html#autotoc_md90", null ],
      [ "</blockquote>", "md_docs_2concepts.html#autotoc_md91", null ],
      [ "Robustness Iterations", "md_docs_2concepts.html#autotoc_md92", null ],
      [ "Confidence vs Prediction Intervals", "md_docs_2concepts.html#autotoc_md94", null ],
      [ "Execution Modes", "md_docs_2concepts.html#autotoc_md96", null ],
      [ "Quick Decision Guide", "md_docs_2concepts.html#autotoc_md98", null ],
      [ "Key Advantages", "md_docs_2concepts.html#autotoc_md100", null ],
      [ "Next Steps", "md_docs_2concepts.html#autotoc_md102", null ]
    ] ],
    [ "Cross-Validation", "md_docs_2cross-validation.html", [
      [ "Overview", "md_docs_2cross-validation.html#autotoc_md104", null ],
      [ "K-Fold Cross-Validation", "md_docs_2cross-validation.html#autotoc_md106", null ],
      [ "Leave-One-Out (LOOCV)", "md_docs_2cross-validation.html#autotoc_md108", null ],
      [ "Seeded Randomization", "md_docs_2cross-validation.html#autotoc_md110", null ],
      [ "Comparison", "md_docs_2cross-validation.html#autotoc_md112", null ],
      [ "</blockquote>", "md_docs_2cross-validation.html#autotoc_md113", null ],
      [ "CV Metrics", "md_docs_2cross-validation.html#autotoc_md114", null ],
      [ "Interpreting Results", "md_docs_2cross-validation.html#autotoc_md116", null ],
      [ "Availability", "md_docs_2cross-validation.html#autotoc_md118", null ],
      [ "Best Practices", "md_docs_2cross-validation.html#autotoc_md120", null ]
    ] ],
    [ "Custom Weights", "md_docs_2custom-weights.html", [
      [ "How Custom Weights Work", "md_docs_2custom-weights.html#autotoc_md122", null ],
      [ "</blockquote>", "md_docs_2custom-weights.html#autotoc_md123", null ],
      [ "When to Use Custom Weights", "md_docs_2custom-weights.html#autotoc_md124", [
        [ "Custom Weights vs. Robustness Iterations", "md_docs_2custom-weights.html#autotoc_md125", null ]
      ] ],
      [ "Basic Usage", "md_docs_2custom-weights.html#autotoc_md127", [
        [ "Suppress a Known Outlier", "md_docs_2custom-weights.html#autotoc_md128", null ],
        [ "Emphasize Important Points", "md_docs_2custom-weights.html#autotoc_md130", null ],
        [ "Propagate Measurement Uncertainty", "md_docs_2custom-weights.html#autotoc_md132", null ]
      ] ],
      [ "Combined with Robustness Iterations", "md_docs_2custom-weights.html#autotoc_md134", null ],
      [ "Validation Rules", "md_docs_2custom-weights.html#autotoc_md136", null ],
      [ "</blockquote>", "md_docs_2custom-weights.html#autotoc_md137", null ],
      [ "See Also", "md_docs_2custom-weights.html#autotoc_md138", null ]
    ] ],
    [ "GPU Backend", "md_docs_2gpu-backend.html", [
      [ "Overview", "md_docs_2gpu-backend.html#autotoc_md140", [
        [ "Supported Features", "md_docs_2gpu-backend.html#autotoc_md141", null ],
        [ "Feature Comparison", "md_docs_2gpu-backend.html#autotoc_md142", null ]
      ] ],
      [ "Checking Availability", "md_docs_2gpu-backend.html#autotoc_md144", null ],
      [ "Installing GPU Support", "md_docs_2gpu-backend.html#autotoc_md146", null ],
      [ "Usage", "md_docs_2gpu-backend.html#autotoc_md148", null ],
      [ "See also", "md_docs_2gpu-backend.html#autotoc_md149", null ]
    ] ],
    [ "Installation", "md_docs_2installation.html", [
      [ "Pre-built Binaries (Linux (x64))", "md_docs_2installation.html#autotoc_md151", null ],
      [ "Pre-built Binaries (macOS (x64))", "md_docs_2installation.html#autotoc_md152", null ],
      [ "Pre-built Binaries (Windows (x64))", "md_docs_2installation.html#autotoc_md153", null ],
      [ "From Source", "md_docs_2installation.html#autotoc_md154", null ],
      [ "From conda-forge", "md_docs_2installation.html#autotoc_md155", null ],
      [ "Verify Installation", "md_docs_2installation.html#autotoc_md157", null ]
    ] ],
    [ "Intervals", "md_docs_2intervals.html", [
      [ "Overview", "md_docs_2intervals.html#autotoc_md159", null ],
      [ "Confidence Intervals", "md_docs_2intervals.html#autotoc_md161", null ],
      [ "Prediction Intervals", "md_docs_2intervals.html#autotoc_md163", null ],
      [ "Both Intervals", "md_docs_2intervals.html#autotoc_md165", null ],
      [ "Confidence Levels", "md_docs_2intervals.html#autotoc_md167", null ],
      [ "Standard Errors", "md_docs_2intervals.html#autotoc_md169", null ],
      [ "Availability", "md_docs_2intervals.html#autotoc_md171", null ]
    ] ],
    [ "Weight Functions", "md_docs_2kernels.html", [
      [ "Overview", "md_docs_2kernels.html#autotoc_md173", null ],
      [ "Available Kernels", "md_docs_2kernels.html#autotoc_md175", null ],
      [ "Tricube (Default)", "md_docs_2kernels.html#autotoc_md177", null ],
      [ "Epanechnikov", "md_docs_2kernels.html#autotoc_md179", null ],
      [ "Gaussian", "md_docs_2kernels.html#autotoc_md181", null ],
      [ "Biweight", "md_docs_2kernels.html#autotoc_md183", null ],
      [ "Cosine", "md_docs_2kernels.html#autotoc_md185", null ],
      [ "Triangle", "md_docs_2kernels.html#autotoc_md187", null ],
      [ "Uniform", "md_docs_2kernels.html#autotoc_md189", null ],
      [ "Choosing a Kernel", "md_docs_2kernels.html#autotoc_md191", null ]
    ] ],
    [ "Merge Strategies", "md_docs_2merge.html", [
      [ "Overview", "md_docs_2merge.html#autotoc_md193", null ],
      [ "Average", "md_docs_2merge.html#autotoc_md195", null ],
      [ "Take First", "md_docs_2merge.html#autotoc_md197", null ],
      [ "Take Last", "md_docs_2merge.html#autotoc_md199", null ],
      [ "Weighted Average", "md_docs_2merge.html#autotoc_md201", null ],
      [ "Choosing a Strategy", "md_docs_2merge.html#autotoc_md203", null ]
    ] ],
    [ "Parameters", "md_docs_2parameters.html", [
      [ "Quick Reference", "md_docs_2parameters.html#autotoc_md205", null ],
      [ "</blockquote>", "md_docs_2parameters.html#autotoc_md206", null ],
      [ "Parameter Options Summary", "md_docs_2parameters.html#autotoc_md207", null ],
      [ "Core Parameters", "md_docs_2parameters.html#autotoc_md209", [
        [ "fraction", "md_docs_2parameters.html#autotoc_md210", null ],
        [ "iterations", "md_docs_2parameters.html#autotoc_md212", null ],
        [ "delta", "md_docs_2parameters.html#autotoc_md214", null ],
        [ "weight_function", "md_docs_2parameters.html#autotoc_md216", null ],
        [ "robustness_method", "md_docs_2parameters.html#autotoc_md218", null ],
        [ "boundary_policy", "md_docs_2parameters.html#autotoc_md220", null ],
        [ "scaling_method", "md_docs_2parameters.html#autotoc_md222", null ],
        [ "zero_weight_fallback", "md_docs_2parameters.html#autotoc_md224", null ],
        [ "auto_converge", "md_docs_2parameters.html#autotoc_md226", null ],
        [ "custom_weights", "md_docs_2parameters.html#autotoc_md228", null ]
      ] ],
      [ "Output Options", "md_docs_2parameters.html#autotoc_md230", [
        [ "return_residuals", "md_docs_2parameters.html#autotoc_md231", null ],
        [ "return_diagnostics", "md_docs_2parameters.html#autotoc_md233", null ],
        [ "return_robustness_weights", "md_docs_2parameters.html#autotoc_md235", null ],
        [ "return_se", "md_docs_2parameters.html#autotoc_md237", null ],
        [ "confidence_intervals / prediction_intervals", "md_docs_2parameters.html#autotoc_md239", null ]
      ] ],
      [ "CV Methods", "md_docs_2parameters.html#autotoc_md241", [
        [ "cv_method", "md_docs_2parameters.html#autotoc_md242", null ]
      ] ],
      [ "Adapter Parameters", "md_docs_2parameters.html#autotoc_md244", [
        [ "chunk_size", "md_docs_2parameters.html#autotoc_md245", null ],
        [ "overlap", "md_docs_2parameters.html#autotoc_md247", null ],
        [ "merge_strategy", "md_docs_2parameters.html#autotoc_md249", null ],
        [ "window_capacity", "md_docs_2parameters.html#autotoc_md251", null ],
        [ "min_points", "md_docs_2parameters.html#autotoc_md253", null ],
        [ "update_mode", "md_docs_2parameters.html#autotoc_md255", null ]
      ] ]
    ] ],
    [ "Quick Start", "md_docs_2quickstart.html", [
      [ "Basic Smoothing", "md_docs_2quickstart.html#autotoc_md257", null ],
      [ "With Confidence Intervals", "md_docs_2quickstart.html#autotoc_md259", null ],
      [ "Handling Outliers", "md_docs_2quickstart.html#autotoc_md261", null ],
      [ "Streaming Mode", "md_docs_2quickstart.html#autotoc_md263", null ],
      [ "Next Steps", "md_docs_2quickstart.html#autotoc_md265", null ]
    ] ],
    [ "Robustness", "md_docs_2robustness.html", [
      [ "How Robustness Works", "md_docs_2robustness.html#autotoc_md267", null ],
      [ "Robustness Methods", "md_docs_2robustness.html#autotoc_md269", [
        [ "Bisquare (Default)", "md_docs_2robustness.html#autotoc_md270", null ],
        [ "Huber", "md_docs_2robustness.html#autotoc_md272", null ],
        [ "Talwar", "md_docs_2robustness.html#autotoc_md274", null ]
      ] ],
      [ "Comparison", "md_docs_2robustness.html#autotoc_md276", null ],
      [ "Detecting Outliers", "md_docs_2robustness.html#autotoc_md278", null ],
      [ "Scale Estimation", "md_docs_2robustness.html#autotoc_md280", null ],
      [ "Auto-Convergence", "md_docs_2robustness.html#autotoc_md282", null ]
    ] ],
    [ "Scaling Methods", "md_docs_2scaling.html", [
      [ "Overview", "md_docs_2scaling.html#autotoc_md284", null ],
      [ "MAD — Median Absolute Deviation (Default)", "md_docs_2scaling.html#autotoc_md286", null ],
      [ "MAR — Median Absolute Residual", "md_docs_2scaling.html#autotoc_md288", null ],
      [ "Mean — Mean Absolute Residual", "md_docs_2scaling.html#autotoc_md290", null ],
      [ "Choosing a Scaling Method", "md_docs_2scaling.html#autotoc_md292", null ]
    ] ],
    [ "Genomic Data Smoothing", "md_docs_2use-case-genomics.html", [
      [ "Overview", "md_docs_2use-case-genomics.html#autotoc_md294", null ],
      [ "Methylation Profile Smoothing", "md_docs_2use-case-genomics.html#autotoc_md296", [
        [ "The Challenge", "md_docs_2use-case-genomics.html#autotoc_md297", null ],
        [ "Solution", "md_docs_2use-case-genomics.html#autotoc_md298", null ]
      ] ],
      [ "ChIP-seq Signal Smoothing", "md_docs_2use-case-genomics.html#autotoc_md300", [
        [ "Application", "md_docs_2use-case-genomics.html#autotoc_md301", null ]
      ] ],
      [ "Large Genome Coverage (Streaming)", "md_docs_2use-case-genomics.html#autotoc_md303", null ],
      [ "Best Practices for Genomic Data", "md_docs_2use-case-genomics.html#autotoc_md305", null ],
      [ "See Also", "md_docs_2use-case-genomics.html#autotoc_md307", null ]
    ] ],
    [ "Real-Time Processing", "md_docs_2use-case-real-time.html", [
      [ "Overview", "md_docs_2use-case-real-time.html#autotoc_md309", null ],
      [ "Online Mode: Point-by-Point", "md_docs_2use-case-real-time.html#autotoc_md311", [
        [ "Sensor Data Example", "md_docs_2use-case-real-time.html#autotoc_md312", null ]
      ] ],
      [ "Streaming Mode: Chunk Processing", "md_docs_2use-case-real-time.html#autotoc_md314", [
        [ "Log File Processing", "md_docs_2use-case-real-time.html#autotoc_md315", null ]
      ] ],
      [ "Real-Time Dashboard Example", "md_docs_2use-case-real-time.html#autotoc_md317", null ],
      [ "Choosing Parameters", "md_docs_2use-case-real-time.html#autotoc_md319", [
        [ "Online Mode", "md_docs_2use-case-real-time.html#autotoc_md320", null ],
        [ "Streaming Mode", "md_docs_2use-case-real-time.html#autotoc_md321", null ]
      ] ],
      [ "Performance Considerations", "md_docs_2use-case-real-time.html#autotoc_md323", null ],
      [ "See Also", "md_docs_2use-case-real-time.html#autotoc_md325", null ]
    ] ],
    [ "Time Series Analysis", "md_docs_2use-case-time-series.html", [
      [ "Overview", "md_docs_2use-case-time-series.html#autotoc_md327", null ],
      [ "Basic Trend Extraction", "md_docs_2use-case-time-series.html#autotoc_md329", null ],
      [ "Detrending", "md_docs_2use-case-time-series.html#autotoc_md331", null ],
      [ "Forecasting with Prediction Intervals", "md_docs_2use-case-time-series.html#autotoc_md333", null ],
      [ "Handling Missing Data", "md_docs_2use-case-time-series.html#autotoc_md335", null ],
      [ "Multi-Scale Analysis", "md_docs_2use-case-time-series.html#autotoc_md337", null ],
      [ "Gene Expression Time Course", "md_docs_2use-case-time-series.html#autotoc_md339", null ],
      [ "Choosing Fraction for Time Series", "md_docs_2use-case-time-series.html#autotoc_md341", null ],
      [ "See Also", "md_docs_2use-case-time-series.html#autotoc_md343", null ]
    ] ]
  ] ]
];

var NAVTREEINDEX =
[
"index.html"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';