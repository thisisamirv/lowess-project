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
      [ "Why this package?", "index.html#autotoc_md6", [
        [ "Speed", "index.html#autotoc_md7", null ],
        [ "Robustness", "index.html#autotoc_md8", null ],
        [ "Features", "index.html#autotoc_md9", null ]
      ] ],
      [ "Validation", "index.html#autotoc_md10", null ],
      [ "Contributing", "index.html#autotoc_md12", null ],
      [ "License", "index.html#autotoc_md13", null ],
      [ "Citation", "index.html#autotoc_md14", null ]
    ] ],
    [ "Execution Modes", "md_docs_2adapter-choice.html", [
      [ "Overview", "md_docs_2adapter-choice.html#autotoc_md16", null ]
    ] ],
    [ "OnlineLowess API", "md_docs_2api-online.html", [
      [ "When to Use", "md_docs_2api-online.html#autotoc_md19", null ],
      [ "Class", "md_docs_2api-online.html#autotoc_md20", [
        [ "fastlowess::OnlineLowess", "md_docs_2api-online.html#autotoc_md21", null ]
      ] ],
      [ "Options Structure", "md_docs_2api-online.html#autotoc_md22", [
        [ "OnlineOptions (inherits LowessOptions)", "md_docs_2api-online.html#autotoc_md23", null ]
      ] ],
      [ "Result Structure", "md_docs_2api-online.html#autotoc_md24", [
        [ "fastlowess::OnlineOutput", "md_docs_2api-online.html#autotoc_md25", null ]
      ] ],
      [ "Options", "md_docs_2api-online.html#autotoc_md26", [
        [ "update_mode", "md_docs_2api-online.html#autotoc_md27", null ]
      ] ]
    ] ],
    [ "StreamingLowess API", "md_docs_2api-streaming.html", [
      [ "When to Use", "md_docs_2api-streaming.html#autotoc_md29", null ],
      [ "Class", "md_docs_2api-streaming.html#autotoc_md30", [
        [ "fastlowess::StreamingLowess", "md_docs_2api-streaming.html#autotoc_md31", null ]
      ] ],
      [ "Result Structure", "md_docs_2api-streaming.html#autotoc_md32", [
        [ "fastlowess::LowessResult", "md_docs_2api-streaming.html#autotoc_md33", null ]
      ] ],
      [ "Options Structure", "md_docs_2api-streaming.html#autotoc_md34", [
        [ "StreamingOptions (inherits LowessOptions)", "md_docs_2api-streaming.html#autotoc_md35", null ]
      ] ],
      [ "Options", "md_docs_2api-streaming.html#autotoc_md36", [
        [ "merge_strategy", "md_docs_2api-streaming.html#autotoc_md37", null ]
      ] ]
    ] ],
    [ "API", "md_docs_2api.html", [
      [ "When to Use Batch Adapter", "md_docs_2api.html#autotoc_md40", null ],
      [ "Classes", "md_docs_2api.html#autotoc_md41", [
        [ "fastlowess::Lowess", "md_docs_2api.html#autotoc_md42", null ]
      ] ],
      [ "Options Structures", "md_docs_2api.html#autotoc_md43", [
        [ "LowessOptions", "md_docs_2api.html#autotoc_md44", null ]
      ] ],
      [ "GPU Acceleration", "md_docs_2api.html#autotoc_md45", null ],
      [ "Result Structure", "md_docs_2api.html#autotoc_md46", [
        [ "fastlowess::LowessResult", "md_docs_2api.html#autotoc_md47", null ],
        [ "fastlowess::Diagnostics", "md_docs_2api.html#autotoc_md48", null ]
      ] ],
      [ "Options", "md_docs_2api.html#autotoc_md49", [
        [ "weight_function", "md_docs_2api.html#autotoc_md50", null ],
        [ "robustness_method", "md_docs_2api.html#autotoc_md51", null ],
        [ "boundary_policy", "md_docs_2api.html#autotoc_md52", null ],
        [ "scaling_method", "md_docs_2api.html#autotoc_md53", null ],
        [ "zero_weight_fallback", "md_docs_2api.html#autotoc_md54", null ]
      ] ],
      [ "Example", "md_docs_2api.html#autotoc_md55", null ]
    ] ],
    [ "Benchmarks", "md_docs_2benchmarks.html", [
      [ "CPU Benchmarks", "md_docs_2benchmarks.html#autotoc_md57", null ],
      [ "GPU Backend", "md_docs_2benchmarks.html#autotoc_md59", null ],
      [ "Reproducing Benchmarks", "md_docs_2benchmarks.html#autotoc_md61", null ]
    ] ],
    [ "Boundary Handling", "md_docs_2boundary.html", [
      [ "Overview", "md_docs_2boundary.html#autotoc_md63", null ],
      [ "Extend (Default)", "md_docs_2boundary.html#autotoc_md65", null ],
      [ "Reflect", "md_docs_2boundary.html#autotoc_md67", null ],
      [ "Zero", "md_docs_2boundary.html#autotoc_md69", null ],
      [ "No Boundary", "md_docs_2boundary.html#autotoc_md71", null ],
      [ "Choosing a Policy", "md_docs_2boundary.html#autotoc_md73", null ]
    ] ],
    [ "Concepts", "md_docs_2concepts.html", [
      [ "What is LOWESS?", "md_docs_2concepts.html#autotoc_md75", null ],
      [ "How It Works", "md_docs_2concepts.html#autotoc_md77", null ],
      [ "The Fraction Parameter", "md_docs_2concepts.html#autotoc_md79", null ],
      [ "Robustness Iterations", "md_docs_2concepts.html#autotoc_md80", null ],
      [ "Confidence vs Prediction Intervals", "md_docs_2concepts.html#autotoc_md82", null ],
      [ "Execution Modes", "md_docs_2concepts.html#autotoc_md84", null ],
      [ "Quick Decision Guide", "md_docs_2concepts.html#autotoc_md86", null ],
      [ "Key Advantages", "md_docs_2concepts.html#autotoc_md88", null ],
      [ "Next Steps", "md_docs_2concepts.html#autotoc_md90", null ]
    ] ],
    [ "Cross-Validation", "md_docs_2cross-validation.html", [
      [ "Overview", "md_docs_2cross-validation.html#autotoc_md92", null ],
      [ "K-Fold Cross-Validation", "md_docs_2cross-validation.html#autotoc_md94", null ],
      [ "Leave-One-Out (LOOCV)", "md_docs_2cross-validation.html#autotoc_md96", null ],
      [ "Seeded Randomization", "md_docs_2cross-validation.html#autotoc_md98", null ],
      [ "Comparison", "md_docs_2cross-validation.html#autotoc_md100", null ],
      [ "CV Metrics", "md_docs_2cross-validation.html#autotoc_md101", null ],
      [ "Interpreting Results", "md_docs_2cross-validation.html#autotoc_md103", null ],
      [ "Availability", "md_docs_2cross-validation.html#autotoc_md105", null ],
      [ "Best Practices", "md_docs_2cross-validation.html#autotoc_md107", null ]
    ] ],
    [ "Custom Weights", "md_docs_2custom-weights.html", [
      [ "How Custom Weights Work", "md_docs_2custom-weights.html#autotoc_md109", null ],
      [ "When to Use Custom Weights", "md_docs_2custom-weights.html#autotoc_md110", [
        [ "Custom Weights vs. Robustness Iterations", "md_docs_2custom-weights.html#autotoc_md111", null ]
      ] ],
      [ "Basic Usage", "md_docs_2custom-weights.html#autotoc_md113", [
        [ "Suppress a Known Outlier", "md_docs_2custom-weights.html#autotoc_md114", null ],
        [ "Emphasize Important Points", "md_docs_2custom-weights.html#autotoc_md116", null ],
        [ "Propagate Measurement Uncertainty", "md_docs_2custom-weights.html#autotoc_md118", null ]
      ] ],
      [ "Combined with Robustness Iterations", "md_docs_2custom-weights.html#autotoc_md120", null ],
      [ "Validation Rules", "md_docs_2custom-weights.html#autotoc_md122", null ],
      [ "See Also", "md_docs_2custom-weights.html#autotoc_md123", null ]
    ] ],
    [ "GPU Backend", "md_docs_2gpu-backend.html", [
      [ "Overview", "md_docs_2gpu-backend.html#autotoc_md125", [
        [ "Supported Features", "md_docs_2gpu-backend.html#autotoc_md126", null ],
        [ "Feature Comparison", "md_docs_2gpu-backend.html#autotoc_md127", null ]
      ] ],
      [ "Checking Availability", "md_docs_2gpu-backend.html#autotoc_md129", null ],
      [ "Installing GPU Support", "md_docs_2gpu-backend.html#autotoc_md131", null ],
      [ "Usage", "md_docs_2gpu-backend.html#autotoc_md133", null ],
      [ "Hardware Requirements", "md_docs_2gpu-backend.html#autotoc_md134", null ],
      [ "Performance Considerations", "md_docs_2gpu-backend.html#autotoc_md135", null ]
    ] ],
    [ "Installation", "md_docs_2installation.html", [
      [ "Pre-built Binaries (Linux (x64))", "md_docs_2installation.html#autotoc_md137", null ],
      [ "Pre-built Binaries (macOS (x64))", "md_docs_2installation.html#autotoc_md138", null ],
      [ "Pre-built Binaries (Windows (x64))", "md_docs_2installation.html#autotoc_md139", null ],
      [ "From Source", "md_docs_2installation.html#autotoc_md140", null ],
      [ "From conda-forge", "md_docs_2installation.html#autotoc_md141", null ],
      [ "Verify Installation", "md_docs_2installation.html#autotoc_md143", null ]
    ] ],
    [ "Intervals", "md_docs_2intervals.html", [
      [ "Overview", "md_docs_2intervals.html#autotoc_md145", null ],
      [ "Confidence Intervals", "md_docs_2intervals.html#autotoc_md147", null ],
      [ "Prediction Intervals", "md_docs_2intervals.html#autotoc_md149", null ],
      [ "Both Intervals", "md_docs_2intervals.html#autotoc_md151", null ],
      [ "Confidence Levels", "md_docs_2intervals.html#autotoc_md153", null ],
      [ "Standard Errors", "md_docs_2intervals.html#autotoc_md155", null ],
      [ "Availability", "md_docs_2intervals.html#autotoc_md157", null ]
    ] ],
    [ "Weight Functions", "md_docs_2kernels.html", [
      [ "Overview", "md_docs_2kernels.html#autotoc_md159", null ],
      [ "Available Kernels", "md_docs_2kernels.html#autotoc_md161", null ],
      [ "Tricube (Default)", "md_docs_2kernels.html#autotoc_md163", null ],
      [ "Epanechnikov", "md_docs_2kernels.html#autotoc_md165", null ],
      [ "Gaussian", "md_docs_2kernels.html#autotoc_md167", null ],
      [ "Biweight", "md_docs_2kernels.html#autotoc_md169", null ],
      [ "Cosine", "md_docs_2kernels.html#autotoc_md171", null ],
      [ "Triangle", "md_docs_2kernels.html#autotoc_md173", null ],
      [ "Uniform", "md_docs_2kernels.html#autotoc_md175", null ],
      [ "Choosing a Kernel", "md_docs_2kernels.html#autotoc_md177", null ]
    ] ],
    [ "Merge Strategies", "md_docs_2merge.html", [
      [ "Overview", "md_docs_2merge.html#autotoc_md179", null ],
      [ "Average", "md_docs_2merge.html#autotoc_md181", null ],
      [ "Take First", "md_docs_2merge.html#autotoc_md183", null ],
      [ "Take Last", "md_docs_2merge.html#autotoc_md185", null ],
      [ "Weighted Average", "md_docs_2merge.html#autotoc_md187", null ],
      [ "Choosing a Strategy", "md_docs_2merge.html#autotoc_md189", null ]
    ] ],
    [ "NEWS", "md_docs_2NEWS.html", [
      [ "fastlowess (C++) (development version)", "md_docs_2NEWS.html#autotoc_md190", [
        [ "Changed", "md_docs_2NEWS.html#autotoc_md191", null ],
        [ "Fixed", "md_docs_2NEWS.html#autotoc_md192", null ]
      ] ],
      [ "fastlowess (C++) 3.1.0", "md_docs_2NEWS.html#autotoc_md193", [
        [ "Added", "md_docs_2NEWS.html#autotoc_md194", null ],
        [ "Changed", "md_docs_2NEWS.html#autotoc_md195", null ],
        [ "Fixed", "md_docs_2NEWS.html#autotoc_md196", null ]
      ] ],
      [ "fastlowess (C++) 3.0.0", "md_docs_2NEWS.html#autotoc_md197", [
        [ "Added", "md_docs_2NEWS.html#autotoc_md198", null ],
        [ "Changed", "md_docs_2NEWS.html#autotoc_md199", null ]
      ] ],
      [ "fastlowess (C++) 2.0.0", "md_docs_2NEWS.html#autotoc_md200", [
        [ "Added", "md_docs_2NEWS.html#autotoc_md201", null ],
        [ "Changed", "md_docs_2NEWS.html#autotoc_md202", null ],
        [ "Fixed", "md_docs_2NEWS.html#autotoc_md203", null ]
      ] ],
      [ "fastlowess (C++) 1.3.0", "md_docs_2NEWS.html#autotoc_md204", [
        [ "Added", "md_docs_2NEWS.html#autotoc_md205", null ],
        [ "Changed", "md_docs_2NEWS.html#autotoc_md206", null ],
        [ "Fixed", "md_docs_2NEWS.html#autotoc_md207", null ]
      ] ],
      [ "fastlowess (C++) 1.2.0", "md_docs_2NEWS.html#autotoc_md208", [
        [ "Fixed", "md_docs_2NEWS.html#autotoc_md209", null ]
      ] ],
      [ "fastlowess (C++) 1.0.0", "md_docs_2NEWS.html#autotoc_md210", [
        [ "Added", "md_docs_2NEWS.html#autotoc_md211", null ],
        [ "Changed", "md_docs_2NEWS.html#autotoc_md212", null ]
      ] ],
      [ "fastlowess (C++) 0.99.9", "md_docs_2NEWS.html#autotoc_md213", [
        [ "Changed", "md_docs_2NEWS.html#autotoc_md214", null ]
      ] ],
      [ "fastlowess (C++) 0.99.8", "md_docs_2NEWS.html#autotoc_md215", [
        [ "Added", "md_docs_2NEWS.html#autotoc_md216", null ]
      ] ],
      [ "fastlowess (C++) 0.99.7", "md_docs_2NEWS.html#autotoc_md217", [
        [ "Fixed", "md_docs_2NEWS.html#autotoc_md218", null ]
      ] ],
      [ "fastlowess (C++) 0.99.6", "md_docs_2NEWS.html#autotoc_md219", [
        [ "Fixed", "md_docs_2NEWS.html#autotoc_md220", null ]
      ] ],
      [ "fastlowess (C++) 0.99.5", "md_docs_2NEWS.html#autotoc_md221", [
        [ "Changed", "md_docs_2NEWS.html#autotoc_md222", null ]
      ] ]
    ] ],
    [ "Quick Start", "md_docs_2quickstart.html", [
      [ "Basic Smoothing", "md_docs_2quickstart.html#autotoc_md224", null ],
      [ "With Confidence Intervals", "md_docs_2quickstart.html#autotoc_md226", null ],
      [ "Handling Outliers", "md_docs_2quickstart.html#autotoc_md228", null ],
      [ "Streaming Mode", "md_docs_2quickstart.html#autotoc_md230", null ],
      [ "Next Steps", "md_docs_2quickstart.html#autotoc_md232", null ]
    ] ],
    [ "Robustness", "md_docs_2robustness.html", [
      [ "How Robustness Works", "md_docs_2robustness.html#autotoc_md234", null ],
      [ "Robustness Methods", "md_docs_2robustness.html#autotoc_md236", [
        [ "Bisquare (Default)", "md_docs_2robustness.html#autotoc_md237", null ],
        [ "Huber", "md_docs_2robustness.html#autotoc_md239", null ],
        [ "Talwar", "md_docs_2robustness.html#autotoc_md241", null ]
      ] ],
      [ "Comparison", "md_docs_2robustness.html#autotoc_md243", null ],
      [ "Detecting Outliers", "md_docs_2robustness.html#autotoc_md245", null ],
      [ "Scale Estimation", "md_docs_2robustness.html#autotoc_md247", null ],
      [ "Auto-Convergence", "md_docs_2robustness.html#autotoc_md249", null ]
    ] ],
    [ "Scaling Methods", "md_docs_2scaling.html", [
      [ "Overview", "md_docs_2scaling.html#autotoc_md251", null ],
      [ "MAD — Median Absolute Deviation (Default)", "md_docs_2scaling.html#autotoc_md253", null ],
      [ "MAR — Median Absolute Residual", "md_docs_2scaling.html#autotoc_md255", null ],
      [ "Mean — Mean Absolute Residual", "md_docs_2scaling.html#autotoc_md257", null ],
      [ "Choosing a Scaling Method", "md_docs_2scaling.html#autotoc_md259", null ]
    ] ],
    [ "Genomic Data Smoothing", "md_docs_2use-case-genomics.html", [
      [ "Overview", "md_docs_2use-case-genomics.html#autotoc_md261", null ],
      [ "Methylation Profile Smoothing", "md_docs_2use-case-genomics.html#autotoc_md263", [
        [ "The Challenge", "md_docs_2use-case-genomics.html#autotoc_md264", null ],
        [ "Solution", "md_docs_2use-case-genomics.html#autotoc_md265", null ]
      ] ],
      [ "ChIP-seq Signal Smoothing", "md_docs_2use-case-genomics.html#autotoc_md267", [
        [ "Application", "md_docs_2use-case-genomics.html#autotoc_md268", null ]
      ] ],
      [ "Large Genome Coverage (Streaming)", "md_docs_2use-case-genomics.html#autotoc_md270", null ],
      [ "Best Practices for Genomic Data", "md_docs_2use-case-genomics.html#autotoc_md272", null ],
      [ "See Also", "md_docs_2use-case-genomics.html#autotoc_md274", null ]
    ] ],
    [ "Real-Time Processing", "md_docs_2use-case-real-time.html", [
      [ "Overview", "md_docs_2use-case-real-time.html#autotoc_md276", null ],
      [ "Online Mode: Point-by-Point", "md_docs_2use-case-real-time.html#autotoc_md278", [
        [ "Sensor Data Example", "md_docs_2use-case-real-time.html#autotoc_md279", null ]
      ] ],
      [ "Streaming Mode: Chunk Processing", "md_docs_2use-case-real-time.html#autotoc_md281", [
        [ "Log File Processing", "md_docs_2use-case-real-time.html#autotoc_md282", null ]
      ] ],
      [ "Real-Time Dashboard Example", "md_docs_2use-case-real-time.html#autotoc_md284", null ],
      [ "Choosing Parameters", "md_docs_2use-case-real-time.html#autotoc_md286", [
        [ "Online Mode", "md_docs_2use-case-real-time.html#autotoc_md287", null ],
        [ "Streaming Mode", "md_docs_2use-case-real-time.html#autotoc_md288", null ]
      ] ],
      [ "Performance Considerations", "md_docs_2use-case-real-time.html#autotoc_md290", null ],
      [ "See Also", "md_docs_2use-case-real-time.html#autotoc_md292", null ]
    ] ],
    [ "Time Series Analysis", "md_docs_2use-case-time-series.html", [
      [ "Overview", "md_docs_2use-case-time-series.html#autotoc_md294", null ],
      [ "Basic Trend Extraction", "md_docs_2use-case-time-series.html#autotoc_md296", null ],
      [ "Detrending", "md_docs_2use-case-time-series.html#autotoc_md298", null ],
      [ "Forecasting with Prediction Intervals", "md_docs_2use-case-time-series.html#autotoc_md300", null ],
      [ "Handling Missing Data", "md_docs_2use-case-time-series.html#autotoc_md302", null ],
      [ "Multi-Scale Analysis", "md_docs_2use-case-time-series.html#autotoc_md304", null ],
      [ "Gene Expression Time Course", "md_docs_2use-case-time-series.html#autotoc_md306", null ],
      [ "Choosing Fraction for Time Series", "md_docs_2use-case-time-series.html#autotoc_md308", null ],
      [ "See Also", "md_docs_2use-case-time-series.html#autotoc_md310", null ]
    ] ]
  ] ]
];

var NAVTREEINDEX =
[
"index.html"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';