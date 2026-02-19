# nolint
#' srr_stats
#'
#' Standards compliance for rfastlowess package (rOpenSci regression category).
#' See SOURCE_MANIFEST.toml for Rust source locations.
#'
#' @srrstatsVerbose TRUE
#' @noRd
NULL

#' NA_standards
#'
#' Standards not applicable to this pure numerical smoothing package.
#'
#' @srrstatsNA {G2.4a, G2.4b, G2.4c, G2.4d, G2.4e} No tabular/factor input.
#' @srrstatsNA {G2.5} No factor input expected.
#' @srrstatsNA {G2.7, G2.8, G2.9, G2.10, G2.11, G2.12} No tabular data input.
#' @srrstatsNA {G3.1, G3.1a} No covariance - uses distance weights.
#' @srrstatsNA {G4.0} Returns simple list, not file output.
#' @srrstatsNA {G5.11, G5.11a, G5.12} No extended test data downloads needed.
#' @srrstatsNA {RE1.0, RE1.1} No formula interface - uses x,y vectors directly.
#' @srrstatsNA {RE1.3, RE1.3a} Row names not relevant for numeric vector I/O.
#' @srrstatsNA {RE2.3} No centering/offsetting - LOWESS is local regression.
#' @srrstatsNA {RE2.4, RE2.4a, RE2.4b} No collinearity - single predictor only.
#' @srrstatsNA {RE4.2, RE4.3, RE4.4, RE4.5, RE4.6} Non-parametric, no coeffs.
#' @srrstatsNA {RE4.12, RE4.13} No data transformations applied.
#' @srrstatsNA {RE4.14, RE4.15} No forecasting - smoothing only.
#' @srrstatsNA {RE4.16} No categorical groups.
#' @srrstatsNA {RE6.1, RE6.3} Default plot method is standard S3.
#' @srrstatsNA {RE7.0, RE7.0a, RE7.1, RE7.1a} Exact input not special-cased.
#' @srrstatsNA {RE7.2, RE7.3, RE7.4} No row names or accessor tests needed.
#' @noRd
NULL

#' Addressed in Rust (crates/lowess, crates/fastLowess)
#'
#' @srrstats {G1.6} Benchmarks vs stats::lowess in examples.
#' @srrstats {G2.0, G2.0a} Length assertions in validator.rs.
#' @srrstats {G2.1, G2.1a} Type assertions in validator.rs.
#' @srrstats {G2.3, G2.3a, G2.3b} match.arg and tolower in R wrappers.
#' @srrstats {G2.4} Explicit type coercion via as.double/as.integer in utils.R.
#' @srrstats {G2.13, G2.14, G2.14a, G2.14b, G2.14c} NA handling in Rust.
#' @srrstats {G2.15} NA checks before passing to algorithms.
#' @srrstats {G2.16} Inf/NaN handling in validator.rs.
#' @srrstats {G5.0} Tests use standard data patterns.
#' @srrstats {G5.1} Test datasets exported in examples.
#' @srrstats {G5.2, G5.2a, G5.2b} Error/warning tests in tests/.
#' @srrstats {G5.3} No NA/NaN in outputs tested.
#' @srrstats {G5.4, G5.4a, G5.4b, G5.4c} Correctness tests vs stats::lowess.
#' @srrstats {G5.5} Fixed random seeds in tests.
#' @srrstats {G5.6, G5.6a, G5.6b} Parameter recovery within tolerance.
#' @srrstats {G5.7} Algorithm performance scales with data size.
#' @srrstats {G5.8, G5.8a, G5.8b, G5.8c, G5.8d} Edge condition tests.
#' @srrstats {G5.9, G5.9a, G5.9b} Noise susceptibility tests.
#' @srrstats {G5.10} Extended tests via environment variable.
#' @srrstats {RE3.2, RE3.3} Threshold defaults documented, settable.
#' @noRd
NULL
