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
        [ "📚 <a href=\"https://thisisamirv.github.io/lowess-project/cpp/\" >View the full documentation</a>", "index.html#autotoc_md3", null ],
        [ "GPU Backend", "index.html#autotoc_md4", null ]
      ] ],
      [ "LOESS vs. LOWESS", "index.html#autotoc_md6", null ],
      [ "</blockquote>", "index.html#autotoc_md7", null ],
      [ "Why this package?", "index.html#autotoc_md8", [
        [ "Speed", "index.html#autotoc_md9", null ],
        [ "Robustness", "index.html#autotoc_md10", null ],
        [ "Features", "index.html#autotoc_md11", null ]
      ] ],
      [ "Validation", "index.html#autotoc_md12", null ],
      [ "API Reference", "index.html#autotoc_md13", null ],
      [ "Contributing", "index.html#autotoc_md15", null ],
      [ "Changelog", "index.html#autotoc_md16", null ],
      [ "License", "index.html#autotoc_md17", null ],
      [ "Citation", "index.html#autotoc_md18", null ]
    ] ],
    [ "Execution Modes", "md_docs_2adapter-choice.html", [
      [ "Overview", "md_docs_2adapter-choice.html#autotoc_md20", null ]
    ] ],
    [ "OnlineLowess API", "md_docs_2api-online.html", [
      [ "When to Use", "md_docs_2api-online.html#autotoc_md23", null ],
      [ "Class", "md_docs_2api-online.html#autotoc_md24", [
        [ "<tt>fastlowess::OnlineLowess</tt>", "md_docs_2api-online.html#autotoc_md25", null ]
      ] ],
      [ "Options Structure", "md_docs_2api-online.html#autotoc_md26", [
        [ "<tt>OnlineOptions</tt> (inherits <tt>LowessOptions</tt>)", "md_docs_2api-online.html#autotoc_md27", null ]
      ] ],
      [ "Result Structure", "md_docs_2api-online.html#autotoc_md28", [
        [ "<tt>fastlowess::OnlineOutput</tt>", "md_docs_2api-online.html#autotoc_md29", null ]
      ] ],
      [ "Options", "md_docs_2api-online.html#autotoc_md30", [
        [ "update_mode", "md_docs_2api-online.html#autotoc_md31", null ]
      ] ]
    ] ],
    [ "StreamingLowess API", "md_docs_2api-streaming.html", [
      [ "When to Use", "md_docs_2api-streaming.html#autotoc_md33", null ],
      [ "Class", "md_docs_2api-streaming.html#autotoc_md34", [
        [ "<tt>fastlowess::StreamingLowess</tt>", "md_docs_2api-streaming.html#autotoc_md35", null ]
      ] ],
      [ "Result Structure", "md_docs_2api-streaming.html#autotoc_md36", [
        [ "<tt>fastlowess::LowessResult</tt>", "md_docs_2api-streaming.html#autotoc_md37", null ]
      ] ],
      [ "Options Structure", "md_docs_2api-streaming.html#autotoc_md38", [
        [ "<tt>StreamingOptions</tt> (inherits <tt>LowessOptions</tt>)", "md_docs_2api-streaming.html#autotoc_md39", null ]
      ] ],
      [ "Options", "md_docs_2api-streaming.html#autotoc_md40", [
        [ "merge_strategy", "md_docs_2api-streaming.html#autotoc_md41", null ]
      ] ]
    ] ],
    [ "API", "md_docs_2api.html", [
      [ "When to Use", "md_docs_2api.html#autotoc_md44", null ],
      [ "Classes", "md_docs_2api.html#autotoc_md45", [
        [ "<tt>fastlowess::Lowess</tt>", "md_docs_2api.html#autotoc_md46", null ]
      ] ],
      [ "Options Structures", "md_docs_2api.html#autotoc_md47", [
        [ "<tt>LowessOptions</tt>", "md_docs_2api.html#autotoc_md48", null ]
      ] ],
      [ "GPU Acceleration", "md_docs_2api.html#autotoc_md49", [
        [ "Enabling GPU Support", "md_docs_2api.html#autotoc_md50", null ],
        [ "Usage", "md_docs_2api.html#autotoc_md51", null ],
        [ "Supported Features", "md_docs_2api.html#autotoc_md52", [
          [ "Feature Comparison", "md_docs_2api.html#autotoc_md53", null ]
        ] ],
        [ "Hardware Requirements", "md_docs_2api.html#autotoc_md54", null ],
        [ "Performance Considerations", "md_docs_2api.html#autotoc_md55", null ]
      ] ],
      [ "Result Structure", "md_docs_2api.html#autotoc_md56", [
        [ "<tt>fastlowess::LowessResult</tt>", "md_docs_2api.html#autotoc_md57", null ],
        [ "<tt>fastlowess::Diagnostics</tt>", "md_docs_2api.html#autotoc_md58", null ]
      ] ],
      [ "Options", "md_docs_2api.html#autotoc_md59", [
        [ "weight_function", "md_docs_2api.html#autotoc_md60", null ],
        [ "robustness_method", "md_docs_2api.html#autotoc_md61", null ],
        [ "boundary_policy", "md_docs_2api.html#autotoc_md62", null ],
        [ "scaling_method", "md_docs_2api.html#autotoc_md63", null ],
        [ "zero_weight_fallback", "md_docs_2api.html#autotoc_md64", null ],
        [ "merge_strategy", "md_docs_2api.html#autotoc_md65", null ],
        [ "update_mode", "md_docs_2api.html#autotoc_md66", null ]
      ] ],
      [ "Example", "md_docs_2api.html#autotoc_md67", null ]
    ] ],
    [ "Benchmarks", "md_docs_2benchmarks.html", [
      [ "CPU Benchmarks", "md_docs_2benchmarks.html#autotoc_md69", null ],
      [ "GPU Backend", "md_docs_2benchmarks.html#autotoc_md71", null ],
      [ "Reproducing Benchmarks", "md_docs_2benchmarks.html#autotoc_md73", null ]
    ] ],
    [ "Boundary Handling", "md_docs_2boundary.html", [
      [ "Overview", "md_docs_2boundary.html#autotoc_md75", null ],
      [ "Extend (Default)", "md_docs_2boundary.html#autotoc_md77", null ],
      [ "Reflect", "md_docs_2boundary.html#autotoc_md79", null ],
      [ "Zero", "md_docs_2boundary.html#autotoc_md81", null ],
      [ "No Boundary", "md_docs_2boundary.html#autotoc_md83", null ],
      [ "Choosing a Policy", "md_docs_2boundary.html#autotoc_md85", null ]
    ] ],
    [ "Concepts", "md_docs_2concepts.html", [
      [ "What is LOWESS?", "md_docs_2concepts.html#autotoc_md87", null ],
      [ "How It Works", "md_docs_2concepts.html#autotoc_md89", null ],
      [ "The Fraction Parameter", "md_docs_2concepts.html#autotoc_md91", null ],
      [ "Robustness Iterations", "md_docs_2concepts.html#autotoc_md93", null ],
      [ "Confidence vs Prediction Intervals", "md_docs_2concepts.html#autotoc_md95", null ],
      [ "Execution Modes", "md_docs_2concepts.html#autotoc_md97", null ],
      [ "Quick Decision Guide", "md_docs_2concepts.html#autotoc_md99", null ],
      [ "Key Advantages", "md_docs_2concepts.html#autotoc_md101", null ],
      [ "Next Steps", "md_docs_2concepts.html#autotoc_md103", null ]
    ] ],
    [ "Cross-Validation", "md_docs_2cross-validation.html", [
      [ "Overview", "md_docs_2cross-validation.html#autotoc_md105", null ],
      [ "K-Fold Cross-Validation", "md_docs_2cross-validation.html#autotoc_md107", null ],
      [ "Leave-One-Out (LOOCV)", "md_docs_2cross-validation.html#autotoc_md109", null ],
      [ "Seeded Randomization", "md_docs_2cross-validation.html#autotoc_md111", null ],
      [ "Comparison", "md_docs_2cross-validation.html#autotoc_md113", null ],
      [ "CV Metrics", "md_docs_2cross-validation.html#autotoc_md115", null ],
      [ "Interpreting Results", "md_docs_2cross-validation.html#autotoc_md117", null ],
      [ "Availability", "md_docs_2cross-validation.html#autotoc_md119", null ],
      [ "Best Practices", "md_docs_2cross-validation.html#autotoc_md121", null ]
    ] ],
    [ "Custom Weights", "md_docs_2custom-weights.html", [
      [ "How Custom Weights Work", "md_docs_2custom-weights.html#autotoc_md123", null ],
      [ "When to Use Custom Weights", "md_docs_2custom-weights.html#autotoc_md125", [
        [ "Custom Weights vs. Robustness Iterations", "md_docs_2custom-weights.html#autotoc_md126", null ]
      ] ],
      [ "Basic Usage", "md_docs_2custom-weights.html#autotoc_md128", [
        [ "Suppress a Known Outlier", "md_docs_2custom-weights.html#autotoc_md129", null ],
        [ "Emphasize Important Points", "md_docs_2custom-weights.html#autotoc_md131", null ],
        [ "Propagate Measurement Uncertainty", "md_docs_2custom-weights.html#autotoc_md133", null ]
      ] ],
      [ "Combined with Robustness Iterations", "md_docs_2custom-weights.html#autotoc_md135", null ],
      [ "Validation Rules", "md_docs_2custom-weights.html#autotoc_md137", null ],
      [ "See Also", "md_docs_2custom-weights.html#autotoc_md139", null ]
    ] ],
    [ "GPU Backend", "md_docs_2gpu-backend.html", [
      [ "Overview", "md_docs_2gpu-backend.html#autotoc_md141", [
        [ "Supported Features", "md_docs_2gpu-backend.html#autotoc_md142", null ],
        [ "Feature Comparison", "md_docs_2gpu-backend.html#autotoc_md143", null ]
      ] ],
      [ "Checking Availability", "md_docs_2gpu-backend.html#autotoc_md145", null ],
      [ "Installing GPU Support", "md_docs_2gpu-backend.html#autotoc_md147", null ],
      [ "Usage", "md_docs_2gpu-backend.html#autotoc_md149", null ],
      [ "See also", "md_docs_2gpu-backend.html#autotoc_md150", null ]
    ] ],
    [ "Installation", "md_docs_2installation.html", [
      [ "Pre-built Binaries (Linux (x64))", "md_docs_2installation.html#autotoc_md152", null ],
      [ "Pre-built Binaries (macOS (x64))", "md_docs_2installation.html#autotoc_md153", null ],
      [ "Pre-built Binaries (Windows (x64))", "md_docs_2installation.html#autotoc_md154", null ],
      [ "From Source", "md_docs_2installation.html#autotoc_md155", null ],
      [ "From conda-forge", "md_docs_2installation.html#autotoc_md156", null ],
      [ "Verify Installation", "md_docs_2installation.html#autotoc_md158", null ]
    ] ],
    [ "Intervals", "md_docs_2intervals.html", [
      [ "Overview", "md_docs_2intervals.html#autotoc_md160", null ],
      [ "Confidence Intervals", "md_docs_2intervals.html#autotoc_md162", null ],
      [ "Prediction Intervals", "md_docs_2intervals.html#autotoc_md164", null ],
      [ "Both Intervals", "md_docs_2intervals.html#autotoc_md166", null ],
      [ "Confidence Levels", "md_docs_2intervals.html#autotoc_md168", null ],
      [ "Standard Errors", "md_docs_2intervals.html#autotoc_md170", null ],
      [ "Availability", "md_docs_2intervals.html#autotoc_md172", null ]
    ] ],
    [ "Weight Functions", "md_docs_2kernels.html", [
      [ "Overview", "md_docs_2kernels.html#autotoc_md174", null ],
      [ "Available Kernels", "md_docs_2kernels.html#autotoc_md176", null ],
      [ "Tricube (Default)", "md_docs_2kernels.html#autotoc_md178", null ],
      [ "Epanechnikov", "md_docs_2kernels.html#autotoc_md180", null ],
      [ "Gaussian", "md_docs_2kernels.html#autotoc_md182", null ],
      [ "Biweight", "md_docs_2kernels.html#autotoc_md184", null ],
      [ "Cosine", "md_docs_2kernels.html#autotoc_md186", null ],
      [ "Triangle", "md_docs_2kernels.html#autotoc_md188", null ],
      [ "Uniform", "md_docs_2kernels.html#autotoc_md190", null ],
      [ "Choosing a Kernel", "md_docs_2kernels.html#autotoc_md192", null ]
    ] ],
    [ "Merge Strategies", "md_docs_2merge.html", [
      [ "Overview", "md_docs_2merge.html#autotoc_md194", null ],
      [ "Average", "md_docs_2merge.html#autotoc_md196", null ],
      [ "Take First", "md_docs_2merge.html#autotoc_md198", null ],
      [ "Take Last", "md_docs_2merge.html#autotoc_md200", null ],
      [ "Weighted Average", "md_docs_2merge.html#autotoc_md202", null ],
      [ "Choosing a Strategy", "md_docs_2merge.html#autotoc_md204", null ]
    ] ],
    [ "Parameters", "md_docs_2parameters.html", [
      [ "Quick Reference", "md_docs_2parameters.html#autotoc_md206", null ],
      [ "Parameter Options Summary", "md_docs_2parameters.html#autotoc_md208", null ],
      [ "Core Parameters", "md_docs_2parameters.html#autotoc_md210", [
        [ "fraction", "md_docs_2parameters.html#autotoc_md211", null ],
        [ "iterations", "md_docs_2parameters.html#autotoc_md213", null ],
        [ "delta", "md_docs_2parameters.html#autotoc_md215", null ],
        [ "weight_function", "md_docs_2parameters.html#autotoc_md217", null ],
        [ "robustness_method", "md_docs_2parameters.html#autotoc_md219", null ],
        [ "boundary_policy", "md_docs_2parameters.html#autotoc_md221", null ],
        [ "scaling_method", "md_docs_2parameters.html#autotoc_md223", null ],
        [ "zero_weight_fallback", "md_docs_2parameters.html#autotoc_md225", null ],
        [ "auto_converge", "md_docs_2parameters.html#autotoc_md227", null ],
        [ "custom_weights", "md_docs_2parameters.html#autotoc_md229", null ]
      ] ],
      [ "Output Options", "md_docs_2parameters.html#autotoc_md231", [
        [ "return_residuals", "md_docs_2parameters.html#autotoc_md232", null ],
        [ "return_diagnostics", "md_docs_2parameters.html#autotoc_md234", null ],
        [ "return_robustness_weights", "md_docs_2parameters.html#autotoc_md236", null ],
        [ "return_se", "md_docs_2parameters.html#autotoc_md238", null ],
        [ "confidence_intervals / prediction_intervals", "md_docs_2parameters.html#autotoc_md240", null ]
      ] ],
      [ "CV Methods", "md_docs_2parameters.html#autotoc_md242", [
        [ "cv_method", "md_docs_2parameters.html#autotoc_md243", null ]
      ] ],
      [ "Adapter Parameters", "md_docs_2parameters.html#autotoc_md245", [
        [ "chunk_size", "md_docs_2parameters.html#autotoc_md246", null ],
        [ "overlap", "md_docs_2parameters.html#autotoc_md248", null ],
        [ "merge_strategy", "md_docs_2parameters.html#autotoc_md250", null ],
        [ "window_capacity", "md_docs_2parameters.html#autotoc_md252", null ],
        [ "min_points", "md_docs_2parameters.html#autotoc_md254", null ],
        [ "update_mode", "md_docs_2parameters.html#autotoc_md256", null ]
      ] ]
    ] ],
    [ "Quick Start", "md_docs_2quickstart.html", [
      [ "Basic Smoothing", "md_docs_2quickstart.html#autotoc_md258", null ],
      [ "With Confidence Intervals", "md_docs_2quickstart.html#autotoc_md260", null ],
      [ "Handling Outliers", "md_docs_2quickstart.html#autotoc_md262", null ],
      [ "Streaming Mode", "md_docs_2quickstart.html#autotoc_md264", null ],
      [ "Next Steps", "md_docs_2quickstart.html#autotoc_md266", null ]
    ] ],
    [ "Robustness", "md_docs_2robustness.html", [
      [ "How Robustness Works", "md_docs_2robustness.html#autotoc_md268", null ],
      [ "Robustness Methods", "md_docs_2robustness.html#autotoc_md270", [
        [ "Bisquare (Default)", "md_docs_2robustness.html#autotoc_md271", null ],
        [ "Huber", "md_docs_2robustness.html#autotoc_md273", null ],
        [ "Talwar", "md_docs_2robustness.html#autotoc_md275", null ]
      ] ],
      [ "Comparison", "md_docs_2robustness.html#autotoc_md277", null ],
      [ "Detecting Outliers", "md_docs_2robustness.html#autotoc_md279", null ],
      [ "Scale Estimation", "md_docs_2robustness.html#autotoc_md281", null ],
      [ "Auto-Convergence", "md_docs_2robustness.html#autotoc_md283", null ]
    ] ],
    [ "Scaling Methods", "md_docs_2scaling.html", [
      [ "Overview", "md_docs_2scaling.html#autotoc_md285", null ],
      [ "MAD — Median Absolute Deviation (Default)", "md_docs_2scaling.html#autotoc_md287", null ],
      [ "MAR — Median Absolute Residual", "md_docs_2scaling.html#autotoc_md289", null ],
      [ "Mean — Mean Absolute Residual", "md_docs_2scaling.html#autotoc_md291", null ],
      [ "Choosing a Scaling Method", "md_docs_2scaling.html#autotoc_md293", null ]
    ] ],
    [ "Genomic Data Smoothing", "md_docs_2use-case-genomics.html", [
      [ "Overview", "md_docs_2use-case-genomics.html#autotoc_md295", null ],
      [ "Methylation Profile Smoothing", "md_docs_2use-case-genomics.html#autotoc_md297", [
        [ "The Challenge", "md_docs_2use-case-genomics.html#autotoc_md298", null ],
        [ "Solution", "md_docs_2use-case-genomics.html#autotoc_md299", null ]
      ] ],
      [ "ChIP-seq Signal Smoothing", "md_docs_2use-case-genomics.html#autotoc_md301", [
        [ "Application", "md_docs_2use-case-genomics.html#autotoc_md302", null ]
      ] ],
      [ "Large Genome Coverage (Streaming)", "md_docs_2use-case-genomics.html#autotoc_md304", null ],
      [ "Best Practices for Genomic Data", "md_docs_2use-case-genomics.html#autotoc_md306", null ],
      [ "See Also", "md_docs_2use-case-genomics.html#autotoc_md308", null ]
    ] ],
    [ "Real-Time Processing", "md_docs_2use-case-real-time.html", [
      [ "Overview", "md_docs_2use-case-real-time.html#autotoc_md310", null ],
      [ "Online Mode: Point-by-Point", "md_docs_2use-case-real-time.html#autotoc_md312", [
        [ "Sensor Data Example", "md_docs_2use-case-real-time.html#autotoc_md313", null ]
      ] ],
      [ "Streaming Mode: Chunk Processing", "md_docs_2use-case-real-time.html#autotoc_md315", [
        [ "Log File Processing", "md_docs_2use-case-real-time.html#autotoc_md316", null ]
      ] ],
      [ "Real-Time Dashboard Example", "md_docs_2use-case-real-time.html#autotoc_md318", null ],
      [ "Choosing Parameters", "md_docs_2use-case-real-time.html#autotoc_md320", [
        [ "Online Mode", "md_docs_2use-case-real-time.html#autotoc_md321", null ],
        [ "Streaming Mode", "md_docs_2use-case-real-time.html#autotoc_md322", null ]
      ] ],
      [ "Performance Considerations", "md_docs_2use-case-real-time.html#autotoc_md324", null ],
      [ "See Also", "md_docs_2use-case-real-time.html#autotoc_md326", null ]
    ] ],
    [ "Time Series Analysis", "md_docs_2use-case-time-series.html", [
      [ "Overview", "md_docs_2use-case-time-series.html#autotoc_md328", null ],
      [ "Basic Trend Extraction", "md_docs_2use-case-time-series.html#autotoc_md330", null ],
      [ "Detrending", "md_docs_2use-case-time-series.html#autotoc_md332", null ],
      [ "Forecasting with Prediction Intervals", "md_docs_2use-case-time-series.html#autotoc_md334", null ],
      [ "Handling Missing Data", "md_docs_2use-case-time-series.html#autotoc_md336", null ],
      [ "Multi-Scale Analysis", "md_docs_2use-case-time-series.html#autotoc_md338", null ],
      [ "Gene Expression Time Course", "md_docs_2use-case-time-series.html#autotoc_md340", null ],
      [ "Choosing Fraction for Time Series", "md_docs_2use-case-time-series.html#autotoc_md342", null ],
      [ "See Also", "md_docs_2use-case-time-series.html#autotoc_md344", null ]
    ] ]
  ] ]
];

var NAVTREEINDEX =
[
"index.html"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';